"""
Try the whole assessment pipeline (evidence -> text analysis -> scoring).

Run from inside backend/, with your venv active:

    python scripts/try_assessment.py --demo
        Self-checking, needs no database: runs the pipeline on real snippets
        captured earlier (one genuinely informative, one vague) plus OSM
        tags. Uses the real Gemini API (1 call per case). Prints PASS/FAIL.

    python scripts/try_assessment.py "Phoenix Marketcity Pune"
    python scripts/try_assessment.py "Phoenix Marketcity Pune" --refresh
        The real thing, end to end: resolves the place (Nominatim), stores
        it, collects web evidence, analyses, scores and stores the
        assessment. Needs the database + internet. A second run without
        --refresh is served from the stored assessment (cached: True).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.contracts import LocationResolutionOutput, OSMTag
from app.features import FEATURES
from app.schemas import AssessmentReport
from app.services.assessment import run_pipeline

BAR = "#"


def show(report: AssessmentReport, header: str = "") -> None:
    s = report.scoring
    loc = report.location
    print(f"\n{loc.name} ({loc.category})  {header}".rstrip())
    e = report.evidence
    print(f"evidence: {e.osm_tag_count} OSM tags, {e.image_count} images, {e.snippet_count} text snippets")
    print(f"\nOVERALL {s.overall_score:5.1f}/100   confidence {s.confidence_score:5.1f}/100 ({s.evidence_level})")

    print(f"\n{'CATEGORY':36} {'SCORE':>6} {'CONF':>5}  ")
    for c in s.category_scores:
        if c.assessed:
            print(f"{c.label:36} {c.score:6.1f} {c.confidence:5.2f}  {BAR * round(c.score / 10)}")
        else:
            print(f"{c.label:36} {'  --':>6} {'':5}  not verified")

    print(f"\n{'FEATURE':30} {'STATUS':13} {'P':>5} {'CONF':>5}  EVIDENCE")
    for f in s.features:
        p = "  -- " if f.probability is None else f"{f.probability:5.2f}"
        print(f"{f.feature:30} {f.status:13} {p} {f.confidence:5.2f}  {'; '.join(f.evidence)[:80]}")

    print("\nPresent:        ", ", ".join(s.verified_features) or "none")
    print("Barriers:       ", ", ".join(s.barrier_features) or "none")
    print("Confirmed absent:", ", ".join(s.absent_features) or "none")
    print("Not verified:   ", ", ".join(s.unverified_features) or "none")
    print(f"Explanations with guideline citations: {len(report.explanations)}")
    for w in report.warnings:
        print("WARNING:", w)
    print("complete:", report.complete)


# ── --demo ───────────────────────────────────────────────────────────────

_PHOENIX = ("January 14, 2026 - Spa and salon services are available ... parental needs.[60] "
            "Accessibility features include wheelchair availability, ramps for entry and navigation, "
            "reserved parking for expecting mothers and differently abled visitors, accessible ...")
_VAGUE = ("Nearby landmarks Phoenix Marketcity Pune Pune Airport Viman Nagar Kharadi Koregaon Park "
          "Planning a furniture shopping trip near Viman Nagar or Kharadi? IKEA Pune offers easy "
          "accessibility and a convenient in-mall shopping experience.")
_NEW_YORK = ("3 weeks ago - Between 1986 and 1991, the number of disabled people using buses in New York "
             "City increased from 11,000 rides a year to 120,000. In 1991, ninety percent of buses were "
             "equipped with wheelchair lifts and ten of the 54 key stations were made wheelchair-accessible; "
             "at the time, 20 of 469 subway stations had ramps ...")


def _loc(name, category):
    return LocationResolutionOutput(location_id="loc_demo", name=name, category=category,
                                    latitude=18.56, longitude=73.91, osm_ref=None)


def run_demo() -> int:
    informative = run_pipeline(
        _loc("Phoenix Marketcity", "mall"),
        osm_tags=[OSMTag(key="wheelchair", value="yes")],
        image_urls=[], text_snippets=[_PHOENIX, _VAGUE])
    show(informative, "[demo: OSM wheelchair=yes + informative text]")

    empty = run_pipeline(
        _loc("Some Unmapped Cafe", "restaurant"),
        osm_tags=[], image_urls=[], text_snippets=[_NEW_YORK])
    show(empty, "[demo: nothing usable, only text about another place]")

    s, e = informative.scoring, empty.scoring
    checks = [
        ("complete run, no warnings (Gemini reachable)", informative.complete and empty.complete),
        ("informative case: ramp and parking detected, entrance detected from OSM",
         {"ramp", "accessible_parking_sign", "accessible_entrance"} <= set(s.verified_features)),
        ("informative case scores above neutral", s.overall_score > 55),
        ("informative case has more confidence than the empty case", s.confidence_score > e.confidence_score),
        ("unrelated text produced NO findings: score 50, confidence 0, level 'insufficient'",
         e.overall_score == 50.0 and e.confidence_score == 0.0 and e.evidence_level == "insufficient"),
        ("empty case: all features 'not verified' (none marked absent)",
         set(e.unverified_features) == set(FEATURES) and not e.absent_features and not e.barrier_features),
    ]
    print("\n" + "-" * 78)
    failed = 0
    for label, ok in checks:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
    print("\nALL DEMO CHECKS PASSED" if not failed else f"\n{failed} DEMO CHECK(S) FAILED")
    return 1 if failed else 0


# ── real, end to end ─────────────────────────────────────────────────────

def run_real(query: str, refresh: bool) -> int:
    from app.database import SessionLocal
    from app.services.assessment import assess_location
    from app.services.errors import ServiceError
    from app.services.location_store import get_location, resolve_and_store

    with SessionLocal() as db:
        try:
            resolved, created = resolve_and_store(db, query)
            print(f"{'Stored new' if created else 'Found existing'} location: "
                  f"{resolved.name} ({resolved.location_id})")
            result = assess_location(db, get_location(db, resolved.location_id), refresh=refresh)
        except ServiceError as e:
            print(f"FAILED ({e.status_code}): {e.detail}")
            return 1
    show(result, f"[assessment #{result.assessment_id}, cached: {result.cached}]")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--refresh"]
    if len(args) != 1:
        sys.exit(__doc__)
    sys.exit(run_demo() if args[0] == "--demo" else run_real(args[0], "--refresh" in sys.argv))
