"""Scoring agent tests — pure functions, no database, no network.

The numbers asserted here are the scoring model's *behaviour* (what should
happen when evidence agrees, conflicts or is missing), not its constants,
so re-tuning the reliability weights does not break them unless it changes
a conclusion."""

import random

import pytest

from app.agents import scoring
from app.agents.contracts import Observation, ScoringInput
from app.agents.scoring import score_location
from app.features import FEATURES, SCORING_CATEGORIES


def obs(feature, status="detected", confidence=0.9, source="osm", evidence="e"):
    return Observation(feature=feature, status=status, confidence=confidence,
                       source=source, evidence=evidence)


def score(*observations):
    return score_location(ScoringInput(location_id="loc_t", category="mall",
                                       observations=list(observations)))


def category(result, key):
    return next(c for c in result.category_scores if c.category == key)


def feature(result, key):
    return next(f for f in result.features if f.feature == key)


# ── unknown stays unknown ────────────────────────────────────────────────

def test_no_evidence_is_neutral_with_zero_confidence():
    r = score()
    assert r.overall_score == 50.0 and r.confidence_score == 0.0
    assert r.evidence_level == "insufficient"
    assert all(not c.assessed and c.score == 50.0 and c.confidence == 0 for c in r.category_scores)
    assert set(r.unverified_features) == set(FEATURES)
    assert r.verified_features == r.barrier_features == r.absent_features == []


def test_unobserved_categories_do_not_drag_the_score_down():
    # One strong positive fact must never push the overall score below neutral.
    assert score(obs("accessible_entrance")).overall_score > 50


def test_output_lists_every_feature_and_category_once():
    r = score(obs("ramp"))
    assert [f.feature for f in r.features] == list(FEATURES)
    assert [c.category for c in r.category_scores] == list(SCORING_CATEGORIES)
    assert sum(c.weight for c in r.category_scores) == pytest.approx(1.0)


# ── per-feature combination ──────────────────────────────────────────────

def test_single_observation_is_never_certain():
    f = feature(score(obs("ramp", confidence=1.0)), "ramp")
    assert f.status == "detected" and 0.7 < f.probability < 0.9 and f.confidence < 0.7


def test_agreeing_sources_are_stronger_than_one():
    one = feature(score(obs("ramp", source="text", confidence=0.8)), "ramp")
    two = feature(score(obs("ramp", source="text", confidence=0.8),
                        obs("ramp", source="osm", confidence=0.9)), "ramp")
    assert two.probability > one.probability and two.confidence > one.confidence
    assert two.sources == ["osm", "text"]


def test_contradicting_sources_end_up_uncertain_and_are_flagged():
    r = score(obs("ramp", "detected", 0.9, "osm"), obs("ramp", "not_detected", 0.8, "text"))
    f = feature(r, "ramp")
    assert f.status == "uncertain" and 0.4 < f.probability < 0.7
    assert "ramp" in r.unverified_features
    assert "ramp" not in r.verified_features and "ramp" not in r.absent_features
    assert len(f.evidence) == 2              # both sides stay visible


def test_stronger_source_wins_a_contradiction():
    f = feature(score(obs("ramp", "detected", 0.9, "osm"), obs("ramp", "not_detected", 0.3, "text")), "ramp")
    assert f.status == "detected"


def test_image_not_detected_is_much_weaker_than_text_not_detected():
    image = feature(score(obs("ramp", "not_detected", 0.9, "image")), "ramp")
    text = feature(score(obs("ramp", "not_detected", 0.9, "text")), "ramp")
    assert image.probability > text.probability          # closer to "don't know"
    assert image.confidence < 0.3 < text.confidence      # one photo is thin evidence of absence
    # ...while a photo that DOES show a ramp is taken at face value
    shown = feature(score(obs("ramp", "detected", 0.9, "image")), "ramp")
    assert shown.probability > 0.65


def test_repeating_the_same_claim_does_not_add_evidence():
    once = feature(score(obs("ramp", source="text", confidence=0.8)), "ramp")
    thrice = feature(score(*[obs("ramp", source="text", confidence=c, evidence=f"quote {c}")
                             for c in (0.8, 0.5, 0.7)]), "ramp")
    assert thrice.probability == once.probability and thrice.confidence == once.confidence


def test_zero_confidence_observation_is_the_same_as_none():
    f = feature(score(obs("ramp", confidence=0.0)), "ramp")
    assert f.status == "not_verified" and f.probability is None and f.sources == []


def test_unknown_feature_is_ignored_with_a_warning():
    r = score(obs("jetpack_landing_pad"), obs("ramp"))
    assert r.warnings and "jetpack_landing_pad" in r.warnings[0]
    assert feature(r, "ramp").status == "detected"


# ── categories ───────────────────────────────────────────────────────────

def test_ramp_or_lift_uses_the_best_available_provision():
    ramp_only = category(score(obs("ramp")), "ramp_or_lift").score
    both = category(score(obs("ramp"), obs("lift_or_elevator_sign", "not_detected")), "ramp_or_lift").score
    assert both == ramp_only            # a missing lift does not cancel a working ramp


def test_stairs_alone_score_low_and_are_listed_as_barrier():
    r = score(obs("stairs", source="text", confidence=0.8))
    assert category(r, "entrance").score < 35
    assert r.barrier_features == ["stairs"]


def test_explicitly_no_stairs_is_good_news():
    r = score(obs("stairs", "not_detected", 0.8, "text"))
    assert category(r, "entrance").score > 65 and r.barrier_features == []


def test_barrier_lowers_but_does_not_erase_a_confirmed_accessible_entrance():
    clean = category(score(obs("accessible_entrance")), "entrance").score
    with_stairs = category(score(obs("accessible_entrance"), obs("stairs")), "entrance").score
    assert 40 < with_stairs < clean


def test_pathway_obstruction_is_a_barrier_feature():
    r = score(obs("pathway_obstruction", source="text", confidence=0.8))
    assert category(r, "pathway").score < 35 and r.barrier_features == ["pathway_obstruction"]


def test_unscored_feature_is_reported_but_does_not_change_scores():
    base = score(obs("accessible_entrance"))
    with_rail = score(obs("accessible_entrance"), obs("handrail"))
    assert with_rail.overall_score == base.overall_score
    assert "handrail" in with_rail.verified_features


def test_absent_benefit_is_listed_as_absent_not_as_barrier():
    r = score(obs("accessible_toilet_sign", "not_detected", 0.9, "text"))
    assert r.absent_features == ["accessible_toilet_sign"]
    assert r.barrier_features == [] and category(r, "restroom").score < 35


# ── overall score, confidence, level ─────────────────────────────────────

def test_better_evidence_gives_better_score_than_worse_evidence():
    good = score(obs("accessible_entrance"), obs("ramp"), obs("accessible_toilet_sign"),
                 obs("accessible_parking_sign"))
    bad = score(obs("stairs"), obs("pathway_obstruction"), obs("accessible_toilet_sign", "not_detected"),
                obs("ramp", "not_detected"))
    assert good.overall_score > 60 > 45 > bad.overall_score


def test_confidence_grows_with_coverage_and_agreement():
    levels = ["insufficient", "low", "moderate", "high"]
    thin = score(obs("accessible_entrance"))
    wider = score(obs("accessible_entrance"), obs("ramp"), obs("accessible_toilet_sign"),
                  obs("accessible_parking_sign"))
    covered = ("accessible_entrance", "ramp", "accessible_toilet_sign", "accessible_parking_sign",
               "accessible_internal_movement", "accessible_seating")
    rich = score(*[obs(k, source=s) for k in covered for s in ("osm", "text", "image")],
                 obs("pathway_obstruction", "not_detected", source="text"))
    assert thin.confidence_score < wider.confidence_score < rich.confidence_score
    assert levels.index(thin.evidence_level) <= levels.index(wider.evidence_level) <= levels.index(rich.evidence_level)
    assert thin.evidence_level in ("insufficient", "low") and rich.evidence_level == "high"


def test_confidence_is_weighted_by_category_importance():
    entrance = score(obs("accessible_entrance")).confidence_score      # 25% of the score
    seating = score(obs("accessible_seating")).confidence_score        # 5% of the score
    assert entrance == pytest.approx(5 * seating, rel=0.01)


def test_evidence_level_boundaries():
    f = scoring._evidence_level
    assert [f(0), f(9.9), f(10), f(34.9), f(35), f(59.9), f(60), f(100)] == \
        ["insufficient", "insufficient", "low", "low", "moderate", "moderate", "high", "high"]


# ── properties over random inputs ────────────────────────────────────────

def random_observations(rng, n):
    return [
        obs(rng.choice(list(FEATURES)), rng.choice(["detected", "not_detected"]),
            round(rng.uniform(0, 1), 2), rng.choice(["osm", "text", "image"]), f"ev{i}")
        for i in range(n)
    ]


@pytest.mark.parametrize("seed", range(40))
def test_scores_stay_in_range_and_are_order_independent(seed):
    rng = random.Random(seed)
    observations = random_observations(rng, rng.randint(0, 25))
    a = score(*observations)
    shuffled = observations[:]
    rng.shuffle(shuffled)
    b = score(*shuffled)
    assert a.model_dump() == b.model_dump()                      # deterministic, order-free
    assert 0 <= a.overall_score <= 100 and 0 <= a.confidence_score <= 100
    assert all(0 <= c.score <= 100 and 0 <= c.confidence <= 1 for c in a.category_scores)
    assert all(f.probability is None or 0 <= f.probability <= 1 for f in a.features)
    # every feature lands in exactly one of the report buckets
    buckets = a.verified_features + a.barrier_features + a.absent_features + a.unverified_features
    not_detected_barriers = [f.feature for f in a.features if f.polarity == "barrier" and f.status == "not_detected"]
    assert sorted(buckets + not_detected_barriers) == sorted(FEATURES)


@pytest.mark.parametrize("seed", range(40))
def test_adding_positive_evidence_never_lowers_the_score(seed):
    rng = random.Random(1000 + seed)
    # Only "no-barrier" worlds: random benefit/barrier evidence, then one extra
    # confirmed benefit. The score may only go up or stay equal.
    base = [o for o in random_observations(rng, rng.randint(0, 12))
            if not (FEATURES[o.feature].polarity == "benefit" and o.status == "not_detected")]
    extra_feature = rng.choice([k for k, f in FEATURES.items() if f.polarity == "benefit" and f.category])
    before = score(*base)
    after = score(*base, obs(extra_feature, "detected", 0.9, "osm", "extra"))
    assert after.overall_score >= before.overall_score - 1e-9
    assert after.confidence_score >= before.confidence_score - 1e-9


@pytest.mark.parametrize("seed", range(40))
def test_adding_a_confirmed_barrier_never_raises_the_score(seed):
    rng = random.Random(2000 + seed)
    base = random_observations(rng, rng.randint(0, 12))
    barrier = rng.choice([k for k, f in FEATURES.items() if f.polarity == "barrier"])
    before = score(*base)
    after = score(*base, obs(barrier, "detected", 0.95, "osm", "extra"))
    # (restricted to the strong-evidence case: a barrier is "confirmed" when P>0.5 by a strong source)
    assert after.overall_score <= before.overall_score + 1e-9
