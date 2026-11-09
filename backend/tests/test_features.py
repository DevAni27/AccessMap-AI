"""The taxonomy must stay internally consistent — scoring depends on it."""

from app.agents.contracts import ACCESSIBILITY_FEATURE_CLASSES
from app.features import FEATURES, IMAGE_FEATURES, SCORING_CATEGORIES, feature_prompt_list


def test_category_weights_sum_to_one():
    assert abs(sum(c.weight for c in SCORING_CATEGORIES.values()) - 1.0) < 1e-9


def test_image_classes_need_no_translation():
    assert set(ACCESSIBILITY_FEATURE_CLASSES) <= set(FEATURES)
    assert IMAGE_FEATURES == tuple(ACCESSIBILITY_FEATURE_CLASSES)


def test_every_scored_feature_points_at_a_real_category():
    for feature in FEATURES.values():
        assert feature.category is None or feature.category in SCORING_CATEGORIES


def test_every_category_has_at_least_one_feature():
    used = {f.category for f in FEATURES.values()}
    assert set(SCORING_CATEGORIES) <= used


def test_barriers_are_exactly_stairs_and_obstructions():
    barriers = {k for k, f in FEATURES.items() if f.polarity == "barrier"}
    assert barriers == {"stairs", "pathway_obstruction"}


def test_prompt_list_mentions_every_feature():
    text = feature_prompt_list()
    assert all(key in text for key in FEATURES)
