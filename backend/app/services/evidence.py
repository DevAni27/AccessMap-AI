"""Collecting and reading stored web evidence."""

from sqlalchemy.orm import Session

from app.agents.contracts import OSMTag, WebEvidenceInput, WebEvidenceOutput
from app.agents.web_evidence import fetch_web_evidence
from app.models import Evidence, Location
from app.services.errors import ServiceError


def latest_evidence(db: Session, location_id: str) -> Evidence | None:
    return (
        db.query(Evidence)
        .filter(Evidence.location_id == location_id)
        .order_by(Evidence.created_at.desc(), Evidence.id.desc())
        .first()
    )


def is_reusable(row: Evidence) -> bool:
    """Stored evidence is worth reusing only if every source worked. Rows from
    before source statuses existed (NULL) are of unknown quality: redo them."""
    status = row.source_status
    return status is not None and not any(v.startswith("unavailable") for v in status.values())


def collect_evidence(db: Session, location: Location) -> Evidence:
    """Run the Web Evidence agent for a location and store the result as a
    new Evidence row (earlier runs are kept)."""
    agent_input = WebEvidenceInput(
        location_id=location.id,
        location_name=location.name,
        latitude=location.latitude,
        longitude=location.longitude,
        osm_ref=location.osm_ref,
    )
    try:
        result = fetch_web_evidence(agent_input)
    except Exception as e:
        raise ServiceError(502, f"Web evidence retrieval failed: {e}")

    row = Evidence(
        location_id=location.id,
        osm_tags=[tag.model_dump() for tag in result.osm_tags],
        image_urls=result.image_urls,
        text_snippets=result.text_snippets,
        source_status=result.source_status,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def to_web_evidence(row: Evidence) -> WebEvidenceOutput:
    return WebEvidenceOutput(
        location_id=row.location_id,
        osm_tags=[OSMTag.model_validate(t) for t in (row.osm_tags or [])],
        image_urls=row.image_urls or [],
        text_snippets=row.text_snippets or [],
        source_status=row.source_status or {},
    )
