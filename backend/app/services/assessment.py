"""
The orchestrator: evidence -> agents -> score -> stored assessment.

`run_pipeline` is the pure part (no database): given a location and its
raw evidence it runs Text Analysis, Image Analysis, Scoring and RAG
Verification and returns an AssessmentReport. `assess_location` wraps it
with evidence collection, caching and storage.

Principles
  * A failing agent never fails the assessment. It becomes a warning, the
    other agents' evidence is still used, and the report is marked
    incomplete (and therefore never served from cache, so a retry re-runs
    the missing parts).
  * Teammate agents (image analysis, RAG) are imported lazily, so a missing
    heavy dependency in their code cannot stop the app from starting.
  * Explanations are only kept when they carry guideline citations; an
    ungrounded explanation is worse than none.
"""

import math
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.agents.contracts import (
    Claim,
    ImageAnalysisInput,
    ImageAnalysisOutput,
    LocationResolutionOutput,
    Observation,
    OSMTag,
    RAGVerificationInput,
    RAGVerificationOutput,
    ScoringInput,
    TextAnalysisInput,
)
from app.agents.scoring import score_location
from app.agents.text_analysis import analyze_text, observations_from_osm
from app.features import DETECTED, FEATURES, NOT_DETECTED, NOT_VERIFIED
from app.models import Assessment, Evidence, Location
from app.schemas import AssessmentReport, AssessmentResponse, EvidenceSummary
from app.services import evidence as evidence_service
from app.services.errors import ServiceError
from app.services.location_store import get_location, to_output

# Stored assessments and evidence older than this are redone.
MAX_AGE = timedelta(days=30)

DEFAULT_IMAGE_CONFIDENCE = 0.5   # detector gave a verdict but no confidence


# ── agent adapters (lazy imports; tests replace these) ───────────────────

def _analyze_images(agent_input: ImageAnalysisInput) -> ImageAnalysisOutput:
    from app.agents.image_analysis import analyze_images
    return analyze_images(agent_input)


def _verify_against_guidelines(agent_input: RAGVerificationInput) -> RAGVerificationOutput:
    from app.agents.rag_verification import verify_against_guidelines
    return verify_against_guidelines(agent_input)


# ── helpers ──────────────────────────────────────────────────────────────

def image_observations(output: ImageAnalysisOutput) -> list[Observation]:
    """Image findings -> Observations. `detected=None` means "not verified"
    and produces nothing: no evidence is never turned into a finding."""
    result = []
    for key, finding in output.features.items():
        if key not in FEATURES or finding.detected is None:
            continue
        confidence = finding.confidence
        if confidence is None or not math.isfinite(confidence):
            confidence = DEFAULT_IMAGE_CONFIDENCE
        method = finding.detection_method or "image model"
        refs = ", ".join(finding.source_images[:2]) or "no image reference"
        result.append(Observation(
            feature=key,
            status=DETECTED if finding.detected else NOT_DETECTED,
            confidence=min(1.0, max(0.0, confidence)),
            source="image",
            evidence=f"{method}: {refs}",
        ))
    return result


def claims_from_scoring(scoring) -> list[Claim]:
    return [
        Claim(
            feature=f.feature,
            status=f.status if f.status in (DETECTED, NOT_DETECTED) else NOT_VERIFIED,
            confidence=f.probability,
        )
        for f in scoring.features
    ]


# ── the pipeline ─────────────────────────────────────────────────────────

def run_pipeline(
    location: LocationResolutionOutput,
    osm_tags: list[OSMTag],
    image_urls: list[str],
    text_snippets: list[str],
    collected_at: datetime | None = None,
    source_status: dict[str, str] | None = None,
) -> AssessmentReport:
    source_status = source_status or {}
    # A source that FAILED is different from one that found nothing: it makes
    # the report incomplete (so it is not cached and a retry re-collects it).
    warnings: list[str] = [
        f"evidence source '{source}' {status}"
        for source, status in source_status.items() if status.startswith("unavailable")
    ]

    # 1. Text + OSM tags
    try:
        text_result = analyze_text(TextAnalysisInput(
            location_id=location.location_id, location_name=location.name,
            category=location.category, osm_tags=osm_tags, text_snippets=text_snippets,
        ))
        observations = list(text_result.observations)
        warnings.extend(text_result.warnings)
    except Exception as e:
        warnings.append(f"text analysis failed ({e}); only OpenStreetMap tags were used")
        observations = observations_from_osm(osm_tags)

    # 2. Images (only when there are any)
    if image_urls:
        try:
            observations.extend(image_observations(_analyze_images(ImageAnalysisInput(
                location_id=location.location_id, location_name=location.name, image_urls=image_urls,
            ))))
        except Exception as e:
            warnings.append(f"image analysis skipped: {e}")

    # 3. Score
    scoring = score_location(ScoringInput(
        location_id=location.location_id, category=location.category, observations=observations,
    ))
    warnings.extend(scoring.warnings)

    # 4. Guideline-grounded explanations
    explanations = []
    try:
        rag = _verify_against_guidelines(RAGVerificationInput(
            location_id=location.location_id, category=location.category,
            claims=claims_from_scoring(scoring),
        ))
        explanations = [e for e in rag.explanations if e.feature in FEATURES and e.citations]
    except Exception as e:
        warnings.append(f"guideline verification skipped: {e}")

    return AssessmentReport(
        location=location,
        scoring=scoring,
        observations=observations,
        explanations=explanations,
        evidence=EvidenceSummary(
            osm_tag_count=len(osm_tags), image_count=len(image_urls),
            snippet_count=len(text_snippets), collected_at=collected_at,
            sources=source_status,
        ),
        warnings=warnings,
        complete=not warnings,
        generated_at=datetime.utcnow(),
    )


# ── database side: caching and storage ───────────────────────────────────

def _is_fresh(created_at: datetime | None) -> bool:
    return created_at is not None and datetime.utcnow() - created_at < MAX_AGE


def _respond(row: Assessment, cached: bool) -> AssessmentResponse:
    report = AssessmentReport.model_validate(row.report)
    return AssessmentResponse(**report.model_dump(), assessment_id=row.id, cached=cached)


def latest_assessment(db: Session, location_id: str) -> Assessment | None:
    return (
        db.query(Assessment)
        .filter(Assessment.location_id == location_id)
        .order_by(Assessment.created_at.desc(), Assessment.id.desc())
        .first()
    )


def _usable_cached_assessment(db: Session, location_id: str) -> Assessment | None:
    """Latest COMPLETE assessment, if still fresh and not made stale by
    evidence collected after it."""
    row = (
        db.query(Assessment)
        .filter(Assessment.location_id == location_id, Assessment.complete.is_(True))
        .order_by(Assessment.created_at.desc(), Assessment.id.desc())
        .first()
    )
    if row is None or not _is_fresh(row.created_at):
        return None
    newer_evidence = (
        db.query(Evidence.id)
        .filter(Evidence.location_id == location_id, Evidence.created_at > row.created_at)
        .first()
    )
    return None if newer_evidence else row


def assess_location(db: Session, location: Location, refresh: bool = False) -> AssessmentResponse:
    if not refresh:
        cached = _usable_cached_assessment(db, location.id)
        if cached is not None:
            return _respond(cached, cached=True)

    # Reuse stored evidence unless asked to refresh or it has gone stale, so
    # retrying after a transient LLM failure does not re-scrape the web.
    evidence = evidence_service.latest_evidence(db, location.id)
    if (refresh or evidence is None or not _is_fresh(evidence.created_at)
            or not evidence_service.is_reusable(evidence)):
        evidence = evidence_service.collect_evidence(db, location)

    report = run_pipeline(
        to_output(location),
        osm_tags=[OSMTag.model_validate(t) for t in (evidence.osm_tags or [])],
        image_urls=evidence.image_urls or [],
        text_snippets=evidence.text_snippets or [],
        collected_at=evidence.created_at,
        source_status=evidence.source_status or {},
    )

    row = Assessment(
        location_id=location.id,
        overall_score=report.scoring.overall_score,
        confidence_score=report.scoring.confidence_score,
        complete=report.complete,
        report=report.model_dump(mode="json"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _respond(row, cached=False)


def stored_assessment(db: Session, location_id: str) -> AssessmentResponse:
    get_location(db, location_id)
    row = latest_assessment(db, location_id)
    if row is None:
        raise ServiceError(404, f"No assessment yet for {location_id!r}; POST /locations/{location_id}/assess first")
    return _respond(row, cached=True)
