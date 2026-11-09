"""
The single source of truth for WHAT we assess.

Every part of the pipeline speaks in these feature keys: OSM-tag rules and
the text analyser (Part 1), the image analyser, the scorer, the RAG
verifier and the report. Nothing ever re-maps names between agents; if a
feature is added or renamed, it happens here and nowhere else.

Two kinds of things live here:
  * FEATURES  — observable facts about a place ("has a ramp", "stairs at
                the entrance"). Each belongs to one scoring category, or
                to none (shown in the report but not scored).
  * SCORING_CATEGORIES — the weighted categories from the project plan,
                which together make up the 0-100 accessibility score.

The first 8 feature keys are exactly `ACCESSIBILITY_FEATURE_CLASSES` from
contracts.py (the classes the image analyser reports), so image findings
need no translation. The last 2 can only be learned from text or map data.
"""

from dataclasses import dataclass
from typing import Literal

from app.agents.contracts import ACCESSIBILITY_FEATURE_CLASSES

# Status vocabulary shared with contracts.Claim / contracts.Observation.
DETECTED = "detected"          # the evidence says the feature is present
NOT_DETECTED = "not_detected"  # the evidence explicitly says it is absent
NOT_VERIFIED = "not_verified"  # no evidence either way (never means "absent")

Polarity = Literal["benefit", "barrier"]


@dataclass(frozen=True)
class Category:
    label: str
    weight: float   # share of the overall score; all weights sum to 1.0


# Weights from the project plan (Evidence Fusion section); adjustable after
# reviewing the Harmonised Guidelines — change them here only.
SCORING_CATEGORIES: dict[str, Category] = {
    "entrance":          Category("Step-free entrance", 0.25),
    "ramp_or_lift":      Category("Ramp or lift availability", 0.20),
    "internal_movement": Category("Accessible internal movement", 0.15),
    "restroom":          Category("Accessible restroom", 0.15),
    "parking":           Category("Accessible parking or drop-off", 0.10),
    "pathway":           Category("Obstruction-free pathway", 0.10),
    "seating":           Category("Accessible seating or service area", 0.05),
}


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    description: str          # wording shown to the LLM, so keep it unambiguous
    polarity: Polarity        # "benefit": presence helps; "barrier": presence hurts
    category: str | None      # key of SCORING_CATEGORIES, or None = not scored


_FEATURE_LIST = [
    Feature(
        "accessible_entrance", "Step-free entrance",
        "A step-free way into the building for wheelchair users: a level entrance, "
        "a ramped entrance, or an entrance served by a lift.",
        "benefit", "entrance"),
    Feature(
        "stairs", "Stairs on the route",
        "Steps or stairs a wheelchair user would have to climb to enter the place or "
        "reach its main areas. Use 'detected' only when steps are stated to exist.",
        "barrier", "entrance"),
    Feature(
        "ramp", "Ramp",
        "A ramp usable by wheelchair users, at an entrance or inside the building.",
        "benefit", "ramp_or_lift"),
    Feature(
        "lift_or_elevator_sign", "Lift / elevator",
        "A passenger lift or elevator that visitors can use (or its sign).",
        "benefit", "ramp_or_lift"),
    Feature(
        "handrail", "Handrails",
        "Handrails along ramps, stairs or corridors.",
        "benefit", None),
    Feature(
        "accessible_toilet_sign", "Accessible restroom",
        "A wheelchair-accessible toilet or restroom (or its sign).",
        "benefit", "restroom"),
    Feature(
        "accessible_parking_sign", "Accessible parking / drop-off",
        "Reserved accessible or disabled parking, or a drop-off point for disabled "
        "visitors (or its sign).",
        "benefit", "parking"),
    Feature(
        "pathway_obstruction", "Pathway obstruction",
        "Something that blocks or narrows walking or rolling routes: blocked or very "
        "narrow paths, broken or uneven surfaces, clutter, vendors on the pavement.",
        "barrier", "pathway"),
    Feature(
        "accessible_internal_movement", "Moving around inside",
        "Getting around inside once in: level floors, wide or automatic doors, wide "
        "corridors, all main areas reachable without stairs. (Lifts are reported "
        "under lift_or_elevator_sign instead.)",
        "benefit", "internal_movement"),
    Feature(
        "accessible_seating", "Accessible seating / service area",
        "Seating, tables or service counters usable by wheelchair users.",
        "benefit", "seating"),
]

FEATURES: dict[str, Feature] = {f.key: f for f in _FEATURE_LIST}

# Features the image analyser can report: exactly the classes in contracts.py.
IMAGE_FEATURES: tuple[str, ...] = tuple(ACCESSIBILITY_FEATURE_CLASSES)


def _validate() -> None:
    """Fail loudly at import time if the taxonomy is inconsistent, instead of
    letting a typo surface later as a silently wrong score."""
    if len(FEATURES) != len(_FEATURE_LIST):
        raise ValueError("duplicate feature key in features.py")
    missing = set(IMAGE_FEATURES) - set(FEATURES)
    if missing:
        raise ValueError(f"image classes missing from FEATURES: {sorted(missing)}")
    total = sum(c.weight for c in SCORING_CATEGORIES.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"category weights must sum to 1.0, got {total}")
    for f in FEATURES.values():
        if f.category is not None and f.category not in SCORING_CATEGORIES:
            raise ValueError(f"{f.key}: unknown category {f.category!r}")
    unused = set(SCORING_CATEGORIES) - {f.category for f in FEATURES.values()}
    if unused:
        raise ValueError(f"categories with no features: {sorted(unused)}")


_validate()


def feature_prompt_list() -> str:
    """The feature list as bullet lines for LLM prompts."""
    return "\n".join(f"- {f.key}: {f.description}" for f in FEATURES.values())
