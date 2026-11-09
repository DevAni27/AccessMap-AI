"""Offline tests for the Location Resolution agent and shared HTTP helpers."""

import time

import pytest

from app.agents import location_resolution as lr
from app.agents.contracts import LocationResolutionInput
from app.http_utils import Throttle


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def nominatim_result(**overrides):
    result = {
        "osm_type": "way",
        "osm_id": 123456,
        "lat": "18.5620429",
        "lon": "73.916663",
        "category": "shop",
        "type": "mall",
        "display_name": "Phoenix Marketcity, Viman Nagar, Pune, Maharashtra, India",
    }
    result.update(overrides)
    return result


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(lr.nominatim_throttle, "wait", lambda: None)


@pytest.mark.parametrize("osm_type,osm_id,expected", [
    ("node", 42, "N42"),
    ("way", 123456, "W123456"),
    ("relation", 7, "R7"),
    ("way", None, None),
    ("weird", 5, None),
    (None, 5, None),
])
def test_build_osm_ref(osm_type, osm_id, expected):
    assert lr._build_osm_ref({"osm_type": osm_type, "osm_id": osm_id}) == expected


def test_resolve_location_returns_osm_ref_and_mapped_category(monkeypatch):
    monkeypatch.setattr(lr.requests, "get", lambda *a, **k: FakeResponse([nominatim_result()]))
    out = lr.resolve_location(LocationResolutionInput(query="Phoenix Marketcity Pune"))
    assert out.osm_ref == "W123456"
    assert out.category == "mall"          # from the OSM tag table, no LLM involved
    assert out.name == "Phoenix Marketcity"
    assert out.location_id.startswith("loc_")
    assert (out.latitude, out.longitude) == (18.5620429, 73.916663)


def test_resolve_location_falls_back_to_llm_for_unmapped_tags(monkeypatch):
    monkeypatch.setattr(lr.requests, "get", lambda *a, **k: FakeResponse(
        [nominatim_result(category="building", type="yes")]))
    monkeypatch.setattr(lr, "generate_text", lambda prompt: " Hospital \n")
    out = lr.resolve_location(LocationResolutionInput(query="Some Clinic Pune"))
    assert out.category == "hospital"


def test_resolve_location_survives_llm_failure(monkeypatch):
    monkeypatch.setattr(lr.requests, "get", lambda *a, **k: FakeResponse(
        [nominatim_result(category="building", type="yes")]))

    def boom(prompt):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(lr, "generate_text", boom)
    out = lr.resolve_location(LocationResolutionInput(query="Mystery Place Pune"))
    assert out.category == "unknown"       # degrade, never crash


def test_resolve_location_no_match_raises_value_error(monkeypatch):
    monkeypatch.setattr(lr.requests, "get", lambda *a, **k: FakeResponse([]))
    with pytest.raises(ValueError):
        lr.resolve_location(LocationResolutionInput(query="Nowhere"))


def test_requests_five_candidates_but_uses_top_one(monkeypatch):
    seen = {}

    def fake_get(url, params, **kwargs):
        seen.update(params)
        return FakeResponse([nominatim_result(osm_id=1), nominatim_result(osm_id=2)])

    monkeypatch.setattr(lr.requests, "get", fake_get)
    out = lr.resolve_location(LocationResolutionInput(query="Phoenix"))
    assert seen["limit"] == 5 and seen["bounded"] == 1
    assert out.osm_ref == "W1"


def test_throttle_spaces_calls_apart():
    throttle = Throttle(0.2)
    start = time.monotonic()
    throttle.wait()
    throttle.wait()
    throttle.wait()
    assert time.monotonic() - start >= 0.4 - 0.02  # first call is free, then 2 gaps
    