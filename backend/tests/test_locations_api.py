"""API tests for /locations — real PostGIS, rolled-back transaction, faked agents."""

import itertools

import pytest
from sqlalchemy import text

from app.agents.contracts import LocationResolutionOutput, OSMTag, WebEvidenceOutput
from app.models import Location
from app.routers import locations as router_module

_ids = itertools.count(1)


def make_resolved(osm_ref="W100", name="Phoenix Marketcity", lat=18.5620429, lon=73.916663, **kw):
    return LocationResolutionOutput(
        location_id=f"loc_test{next(_ids):06d}",   # fresh id per resolve, like the real agent
        name=name, category=kw.get("category", "mall"),
        latitude=lat, longitude=lon, address="Viman Nagar, Pune", osm_ref=osm_ref,
    )


@pytest.fixture
def fake_resolver(monkeypatch):
    """Resolver whose result is controlled by the test; returns a fresh
    location_id on every call, exactly like the real agent."""
    state = {"kwargs": {}}

    def resolve(_input):
        return make_resolved(**state["kwargs"])

    monkeypatch.setattr(router_module, "resolve_location", resolve)
    return state


def post_location(client, query="phoenix"):
    return client.post("/locations", json={"query": query})


def test_create_returns_201_and_fills_geom_from_lat_lon(client, db, fake_resolver):
    r = post_location(client)
    assert r.status_code == 201
    body = r.json()
    assert body["osm_ref"] == "W100" and body["category"] == "mall"

    lon, lat = db.execute(
        text("SELECT ST_X(geom::geometry), ST_Y(geom::geometry) FROM locations WHERE id = :i"),
        {"i": body["location_id"]},
    ).one()
    assert (lat, lon) == pytest.approx((18.5620429, 73.916663))   # PostGIS order is lon, lat


def test_same_place_twice_returns_existing_record(client, db, fake_resolver):
    first = post_location(client, "Phoenix Marketcity Pune")
    second = post_location(client, "phoenix mall viman nagar")      # different phrasing
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["location_id"] == first.json()["location_id"]
    assert db.query(Location).count() == 1


def test_different_places_get_different_rows(client, db, fake_resolver):
    fake_resolver["kwargs"] = {"osm_ref": "W100"}
    a = post_location(client)
    fake_resolver["kwargs"] = {"osm_ref": "N200", "name": "Ruby Hall Clinic", "category": "hospital"}
    b = post_location(client)
    assert (a.status_code, b.status_code) == (201, 201)
    assert a.json()["location_id"] != b.json()["location_id"]
    assert db.query(Location).count() == 2


def test_places_without_osm_ref_are_never_merged(client, db, fake_resolver):
    fake_resolver["kwargs"] = {"osm_ref": None}
    post_location(client)
    post_location(client)
    assert db.query(Location).count() == 2     # no identity -> can't prove they're the same


def test_race_between_check_and_insert_returns_existing_record(client, db, fake_resolver, monkeypatch):
    first = post_location(client)
    real_find = router_module._find_by_osm_ref
    calls = {"n": 0}

    def stale_then_real(db_, osm_ref):
        calls["n"] += 1
        return None if calls["n"] == 1 else real_find(db_, osm_ref)   # 1st check misses, as in a race

    monkeypatch.setattr(router_module, "_find_by_osm_ref", stale_then_real)
    second = post_location(client)
    assert second.status_code == 200
    assert second.json()["location_id"] == first.json()["location_id"]
    assert db.query(Location).count() == 1


def test_resolution_errors_map_to_http_codes(client, monkeypatch):
    def not_found(_):
        raise ValueError("no match")

    def broken(_):
        raise RuntimeError("nominatim down")

    monkeypatch.setattr(router_module, "resolve_location", not_found)
    assert post_location(client).status_code == 404
    monkeypatch.setattr(router_module, "resolve_location", broken)
    assert post_location(client).status_code == 502


def test_list_and_get_work_when_geom_is_set(client, fake_resolver):
    created = post_location(client).json()
    listing = client.get("/locations")
    assert listing.status_code == 200
    assert listing.json() == [created]
    assert "geom" not in listing.json()[0]
    assert client.get(f"/locations/{created['location_id']}").json() == created
    assert client.get("/locations/loc_missing").status_code == 404


def test_evidence_roundtrip(client, fake_resolver, monkeypatch):
    loc = post_location(client).json()

    def fake_fetch(agent_input):
        assert agent_input.location_name == "Phoenix Marketcity"
        return WebEvidenceOutput(
            location_id=agent_input.location_id,
            osm_tags=[OSMTag(key="wheelchair", value="yes")],
            image_urls=["https://upload.wikimedia.org/a.jpg"],
            text_snippets=[],
        )

    monkeypatch.setattr(router_module, "fetch_web_evidence", fake_fetch)
    created = client.post(f"/locations/{loc['location_id']}/evidence")
    assert created.status_code == 201
    assert created.json()["osm_tags"] == [{"key": "wheelchair", "value": "yes"}]

    stored = client.get(f"/locations/{loc['location_id']}/evidence").json()
    assert len(stored) == 1
    assert stored[0]["osm_tags"] == [{"key": "wheelchair", "value": "yes"}]
    assert stored[0]["text_snippets"] == []
    assert stored[0]["image_urls"] == ["https://upload.wikimedia.org/a.jpg"]


def test_evidence_endpoints_404_for_unknown_location(client):
    assert client.post("/locations/loc_missing/evidence").status_code == 404
    assert client.get("/locations/loc_missing/evidence").status_code == 404


def test_evidence_failure_maps_to_502(client, fake_resolver, monkeypatch):
    loc = post_location(client).json()

    def broken(_):
        raise RuntimeError("reverse geocode down")

    monkeypatch.setattr(router_module, "fetch_web_evidence", broken)
    assert client.post(f"/locations/{loc['location_id']}/evidence").status_code == 502