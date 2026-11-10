"""
Evidence Fusion & Scoring Agent.

Takes every Observation about a place (OSM tags, text, images) and turns
them into one 0-100 accessibility score, a 0-100 confidence score and a
per-feature / per-category breakdown. Pure function: no database, no
network, no LLM — the same input always gives the same output, so every
number in a report can be traced and tested.

The model, in four steps
------------------------
1. Per feature — a weighted vote.
   Each observation votes "present" (1) or "absent" (0) with weight
       w = source reliability x the source's own confidence
   (an image that does NOT show a ramp counts for much less than a text
   that says "no ramp": a photo not showing something proves little).
   With a small prior of 0.5 pulling towards "don't know":
       P(present) = (PRIOR + sum(w*vote)) / (PRIOR + sum(w))
   so one weak mention can never produce certainty, two agreeing sources
   are stronger than one, and contradicting sources land near 50%.
   Repeats of the same claim from the same source are counted once.

2. Per category — best provision, penalised by barriers.
   A place needs ONE way in / ONE ramp-or-lift, so the category value is
   the best benefit feature's P(present), multiplied down when a barrier
   (stairs, obstructions) is probably present. If only barrier evidence
   exists, the value is P(barrier absent).

3. Unknown stays unknown.
   A category with no evidence scores a neutral 50 with confidence 0 and
   is flagged `assessed=False` — absence of evidence is never scored as
   inaccessible.

4. Overall — the weighted sum of category scores (weights in features.py);
   confidence is the same weighted sum of per-category evidence strength,
   so a place with one OSM tag can never look "high confidence".
"""

from app.agents.contracts import (
    CategoryScore,
    FeatureAssessment,
    Observation,
    ScoringInput,
    ScoringOutput,
)
from app.features import (
    DETECTED,
    FEATURES,
    NOT_DETECTED,
    NOT_VERIFIED,
    SCORING_CATEGORIES,
    Feature,
)

# How much each kind of source is trusted. OSM tags are explicit, structured
# statements by mappers; text is scraped web prose read by an LLM; image
# detectors are the noisiest.
SOURCE_RELIABILITY: dict[str, float] = {"osm": 0.85, "text": 0.6, "image": 0.5}

# "Not detected" from these sources is weak evidence of absence: a photo
# that does not show a ramp does not mean there is no ramp.
NOT_DETECTED_DISCOUNT: dict[str, float] = {"image": 0.4}

PRIOR_WEIGHT = 0.5          # strength of the "don't know" pull towards 50%
BARRIER_PENALTY = 0.3       # how much a surely-present barrier cuts a category
NEUTRAL_SCORE = 50.0        # score of a category nobody has any evidence about

DETECTED_ABOVE = 0.6        # P(present) >= this  -> "detected"
NOT_DETECTED_BELOW = 0.4    # P(present) <= this  -> "not_detected"; between -> "uncertain"

# confidence_score (0-100) -> label shown to the user
EVIDENCE_LEVELS = ((10.0, "insufficient"), (35.0, "low"), (60.0, "moderate"))


def _weight(obs: Observation) -> float:
    weight = SOURCE_RELIABILITY.get(obs.source, 0.0) * obs.confidence
    if obs.status == NOT_DETECTED:
        weight *= NOT_DETECTED_DISCOUNT.get(obs.source, 1.0)
    return weight


def _strongest_per_claim(observations: list[Observation]) -> list[Observation]:
    """Keep the most confident observation for each (source, status): a
    source repeating itself must not count as several independent sources."""
    best: dict[tuple[str, str], Observation] = {}
    for obs in observations:
        key = (obs.source, obs.status)
        if key not in best or obs.confidence > best[key].confidence:
            best[key] = obs
    return list(best.values())


def assess_feature(feature: Feature, observations: list[Observation]) -> FeatureAssessment:
    """Combine all observations of one feature into a single assessment."""
    usable = [(obs, _weight(obs)) for obs in _strongest_per_claim(observations)]
    usable = [(obs, w) for obs, w in usable if w > 0]
    base = dict(feature=feature.key, label=feature.label, polarity=feature.polarity)
    if not usable:
        return FeatureAssessment(**base, status=NOT_VERIFIED)

    total = sum(w for _, w in usable)
    present = sum(w for obs, w in usable if obs.status == DETECTED)
    probability = (PRIOR_WEIGHT * 0.5 + present) / (PRIOR_WEIGHT + total)
    strength = total / (total + PRIOR_WEIGHT)

    if probability >= DETECTED_ABOVE:
        status = DETECTED
    elif probability <= NOT_DETECTED_BELOW:
        status = NOT_DETECTED
    else:
        status = "uncertain"

    usable.sort(key=lambda pair: (-pair[1], pair[0].source, pair[0].status))
    return FeatureAssessment(
        **base,
        status=status,
        probability=round(probability, 4),
        confidence=round(strength, 4),
        sources=sorted({obs.source for obs, _ in usable}),
        evidence=[f"{obs.source}: {obs.evidence}" for obs, _ in usable],
    )


def _score_category(key: str, assessed: dict[str, FeatureAssessment]) -> CategoryScore:
    category = SCORING_CATEGORIES[key]
    members = [
        assessed[f.key] for f in FEATURES.values()
        if f.category == key and assessed[f.key].status != NOT_VERIFIED
    ]
    base = dict(category=key, label=category.label, weight=category.weight)
    if not members:
        return CategoryScore(**base, score=NEUTRAL_SCORE, confidence=0.0, assessed=False)

    benefits = [a.probability for a in members if a.polarity == "benefit"]
    barriers = [a.probability for a in members if a.polarity == "barrier"]

    if benefits and barriers:
        provision = max(benefits) * (1 - BARRIER_PENALTY * max(barriers))
        value = max(provision, 1 - max(barriers))   # clear evidence of no barrier is a floor
    elif benefits:
        value = max(benefits)
    else:
        value = 1 - max(barriers)

    unseen = 1.0
    for a in members:
        unseen *= 1 - a.confidence
    return CategoryScore(**base, score=round(100 * value, 1), confidence=round(1 - unseen, 4), assessed=True)


def _evidence_level(confidence_score: float) -> str:
    for limit, label in EVIDENCE_LEVELS:
        if confidence_score < limit:
            return label
    return "high"


def score_location(input: ScoringInput) -> ScoringOutput:
    warnings: list[str] = []
    by_feature: dict[str, list[Observation]] = {key: [] for key in FEATURES}
    for obs in input.observations:
        if obs.feature in by_feature:
            by_feature[obs.feature].append(obs)
        else:
            warnings.append(f"ignored observation of unknown feature {obs.feature!r}")

    assessed = {key: assess_feature(FEATURES[key], obs) for key, obs in by_feature.items()}
    categories = [_score_category(key, assessed) for key in SCORING_CATEGORIES]

    overall = sum(c.weight * c.score for c in categories)
    confidence = 100 * sum(c.weight * c.confidence for c in categories)

    def keys(polarity: str, status: str) -> list[str]:
        return [k for k, a in assessed.items() if a.polarity == polarity and a.status == status]

    return ScoringOutput(
        location_id=input.location_id,
        overall_score=round(overall, 1),
        confidence_score=round(confidence, 1),
        evidence_level=_evidence_level(confidence),
        category_scores=categories,
        features=list(assessed.values()),
        verified_features=keys("benefit", DETECTED),
        barrier_features=keys("barrier", DETECTED),
        absent_features=keys("benefit", NOT_DETECTED),
        unverified_features=[k for k, a in assessed.items() if a.status in (NOT_VERIFIED, "uncertain")],
        warnings=warnings,
    )
