"""
Try the Text Analysis agent. Uses the real Gemini API (1 call per location).

Run from inside backend/, with your venv active:

    python scripts/try_text_analysis.py --demo
        Self-checking: runs two real snippet sets captured earlier in this
        project (one with genuine accessibility facts, one full of
        off-topic/generic text) and prints PASS/FAIL. Needs no database.

    python scripts/try_text_analysis.py phoenix
    python scripts/try_text_analysis.py loc_ab12cd34ef56
        Runs on the evidence stored in the database for that location
        (match by id or part of the name). Collect evidence first with
        POST /locations/{id}/evidence in /docs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.contracts import OSMTag, TextAnalysisInput, TextAnalysisOutput
from app.agents.text_analysis import analyze_text
from app.features import FEATURES


def show(agent_input: TextAnalysisInput) -> TextAnalysisOutput:
    print(f"{agent_input.location_name} ({agent_input.category}) - "
          f"{len(agent_input.osm_tags)} OSM tags, {len(agent_input.text_snippets)} snippets")
    for i, snippet in enumerate(agent_input.text_snippets, 1):
        print(f"  [{i}] {snippet[:110]}{'...' if len(snippet) > 110 else ''}")

    result = analyze_text(agent_input)

    print(f"\n{'FEATURE':30} {'STATUS':13} {'CONF':5} {'SOURCE':6} EVIDENCE")
    for o in result.observations:
        print(f"{o.feature:30} {o.status:13} {o.confidence:<5.2f} {o.source:6} {o.evidence[:70]}")
    if not result.observations:
        print("(no observations)")

    seen = {o.feature for o in result.observations}
    print("\nNot verified (no evidence):", ", ".join(k for k in FEATURES if k not in seen) or "none")
    for warning in result.warnings:
        print("WARNING:", warning)
    return result


# ── --demo: real snippets captured from earlier runs of the Web Evidence agent ──

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
_COURTS = ("January 10, 2024 - More than half of the District Court complexes in the country do not have "
           "ramps, only 25.2% have availability of wheelchairs.")
_METRA = ("Metra currently has 186 fully ... stations will be able to access train platforms from the "
          "street. However, ramps, ticket windows, buildings and shelters may not fully conform to ADA "
          "guidelines....")


def _has(result, feature, status, source):
    return any((o.feature, o.status, o.source) == (feature, status, source) for o in result.observations)


DEMO_CASES = [
    (
        TextAnalysisInput(location_id="demo1", location_name="Phoenix Marketcity", category="mall",
                          text_snippets=[_PHOENIX, _VAGUE]),
        [
            ("ramp is reported as detected, from the text",
             lambda r: _has(r, "ramp", "detected", "text")),
            ("accessible parking is reported as detected, from the text",
             lambda r: _has(r, "accessible_parking_sign", "detected", "text")),
            ("the vague 'easy accessibility' sentence produced nothing",
             lambda r: not any("easy accessibility" in o.evidence.lower() for o in r.observations)),
        ],
    ),
    (
        TextAnalysisInput(location_id="demo2", location_name="District Court Metro Station", category="unknown",
                          text_snippets=[_NEW_YORK, _COURTS, _METRA]),
        [
            ("generic / other-place text (New York, nationwide courts, Chicago) produced NO observations",
             lambda r: not any(o.source == "text" for o in r.observations)),
        ],
    ),
]


def run_demo() -> int:
    failed = 0
    for agent_input, expectations in DEMO_CASES:
        result = show(agent_input)
        print()
        for label, check in expectations:
            ok = check(result)
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {label}")
        print("\n" + "-" * 78 + "\n")
    print("ALL DEMO CHECKS PASSED" if not failed else f"{failed} DEMO CHECK(S) FAILED")
    return 1 if failed else 0


# ── database mode ──────────────────────────────────────────────────────────

def run_stored(query: str) -> int:
    from app.database import SessionLocal
    from app.models import Evidence, Location

    with SessionLocal() as db:
        location = db.get(Location, query) or (
            db.query(Location).filter(Location.name.ilike(f"%{query}%")).first()
        )
        if location is None:
            print(f"No stored location matches {query!r}. Known: "
                  f"{[l.name for l in db.query(Location).all()]}")
            return 1
        evidence = (
            db.query(Evidence).filter(Evidence.location_id == location.id)
            .order_by(Evidence.created_at.desc(), Evidence.id.desc()).first()
        )
        if evidence is None:
            print(f"{location.name} has no evidence yet. "
                  f"Run POST /locations/{location.id}/evidence first.")
            return 1

        agent_input = TextAnalysisInput(
            location_id=location.id,
            location_name=location.name,
            category=location.category,
            osm_tags=[OSMTag(**t) for t in (evidence.osm_tags or [])],
            text_snippets=evidence.text_snippets or [],
        )
    show(agent_input)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(run_demo() if sys.argv[1] == "--demo" else run_stored(sys.argv[1]))
