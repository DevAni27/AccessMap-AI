"""Orchestrator tests: the pure pipeline (no DB) and the /assess API (real
PostGIS in a rolled-back transaction). Every agent that would touch the
network (Gemini, Nominatim, ddgs) is replaced with a fake."""

import itertools
import math
from datetime import datetime, timedelta

import pytest

from app.agents.contracts import (
    FeatureFinding,
    GroundedExplanation,
    GuidelineCitation,
    ImageAnalysisOutput,
    LocationResolutionOutput,
    Observation,
    OSMTag,
    RAGVerificationOutput,
    TextAnalysisOutput,
    WebEvidenceOutput,
)
from app.models import Assessment, Evidence, Location
from app.services import assessment as svc
from app.services import evidence as evidence_service
from app.services import location_store

LOC = LocationResolutionOutput(
    location_id="loc_pipeline", name="Test Mall", category="mall",
    latitude=18.5, longitude=73.9, osm_ref="TEST-W1",
)


def obs(feature, status="detected", confidence=0.8, source="text", evidence="quote"):
    return Observation(feature=feature, status=status, confidence=confidence, source=source, evidence=evidence)


def citation():
    return GuidelineCitation(clause="4.2", text_excerpt="ramps shall ...", source="Harmonised Guidelines 2021")


class Fakes:
    """Replaces the three agents the orchestrator calls, recording calls."""

    def __init__(self, monkeypatch):
        self.text_calls = self.image_calls = self.rag_calls = 0
        self.text = lambda inp: TextAnalysisOutput(
            location_id=inp.location_id, observations=[obs("ramp"), obs("accessible_entrance", source="osm")])
        self.image = lambda inp: ImageAnalysisOutput(location_id=inp.location_id, features={}, processed_image_count=0)
        self.rag = lambda inp: RAGVerificationOutput(location_id=inp.location_id, explanations=[])
        self.rag_inputs = []
        monkeypatch.setattr(svc, "analyze_text", self._text)
        monkeypatch.setattr(svc, "_analyze_images", self._image)
        monkeypatch.setattr(svc, "_verify_against_guidelines", self._rag)

    def _text(self, inp):
        self.text_calls += 1
        return self.text(inp)

    def _image(self, inp):
        self.image_calls += 1
        return self.image(inp)

    def _rag(self, inp):
        self.rag_calls += 1
        self.rag_inputs.append(inp)
        return self.rag(inp)


@pytest.fixture
def fakes(monkeypatch):
    return Fakes(monkeypatch)


def run(osm=(), images=(), snippets=()):
    return svc.run_pipeline(LOC, list(osm), list(images), list(snippets))


# ── pipeline (no database) ───────────────────────────────────────────────

def test_happy_path_scores_and_is_complete(fakes):
    r = run(osm=[OSMTag(key="wheelchair", value="yes")], snippets=["a", "b"])
    assert r.complete and r.warnings == []
    assert {o.feature for o in r.observations} == {"ramp", "accessible_entrance"}
    assert r.scoring.overall_score > 50 and "ramp" in r.scoring.verified_features
    assert (r.evidence.osm_tag_count, r.evidence.image_count, r.evidence.snippet_count) == (1, 0, 2)


def test_images_are_only_analysed_when_there_are_any(fakes):
    run()
    assert fakes.image_calls == 0
    run(images=["https://x/a.jpg"])
    assert fakes.image_calls == 1


def test_text_warnings_make_the_report_incomplete(fakes):
    fakes.text = lambda inp: TextAnalysisOutput(
        location_id=inp.location_id, observations=[obs("ramp")], warnings=["Web-text analysis skipped: Gemini down"])
    r = run()
    assert not r.complete and "Gemini down" in r.warnings[0]
    assert "ramp" in r.scoring.verified_features        # what we do have is still used


def test_text_agent_crash_falls_back_to_osm_tags(fakes):
    def boom(_):
        raise RuntimeError("bug in agent")

    fakes.text = boom
    r = run(osm=[OSMTag(key="wheelchair", value="yes")])
    assert not r.complete and "bug in agent" in r.warnings[0]
    assert any(o.source == "osm" and o.feature == "accessible_entrance" for o in r.observations)


def test_image_findings_become_observations(fakes):
    fakes.image = lambda inp: ImageAnalysisOutput(location_id=inp.location_id, processed_image_count=3, features={
        "ramp": FeatureFinding(detected=True, confidence=0.7, source_images=["u1", "u2", "u3"], detection_method="yolo"),
        "stairs": FeatureFinding(detected=False, confidence=0.6, source_images=["u1"]),
        "handrail": FeatureFinding(detected=None),                       # not verified -> nothing
        "lift_or_elevator_sign": FeatureFinding(detected=True),          # no confidence -> default
        "made_up_class": FeatureFinding(detected=True, confidence=0.9),  # unknown -> dropped
        "accessible_entrance": FeatureFinding(detected=True, confidence=math.nan),
    })
    fakes.text = lambda inp: TextAnalysisOutput(location_id=inp.location_id, observations=[])
    r = run(images=["u1", "u2", "u3"])
    by_feature = {o.feature: o for o in r.observations}
    assert set(by_feature) == {"ramp", "stairs", "lift_or_elevator_sign", "accessible_entrance"}
    assert by_feature["ramp"].source == "image" and by_feature["ramp"].evidence == "yolo: u1, u2"
    assert by_feature["stairs"].status == "not_detected"
    assert by_feature["lift_or_elevator_sign"].confidence == svc.DEFAULT_IMAGE_CONFIDENCE
    assert by_feature["accessible_entrance"].confidence == svc.DEFAULT_IMAGE_CONFIDENCE   # NaN never leaks in
    assert r.complete


def test_image_agent_failure_keeps_other_evidence(fakes):
    def boom(_):
        raise RuntimeError("model not found")

    fakes.image = boom
    r = run(images=["u1"])
    assert not r.complete and "image analysis skipped" in r.warnings[0]
    assert "ramp" in r.scoring.verified_features


def test_only_cited_explanations_are_kept(fakes):
    fakes.rag = lambda inp: RAGVerificationOutput(location_id=inp.location_id, explanations=[
        GroundedExplanation(feature="ramp", explanation="grounded", citations=[citation()]),
        GroundedExplanation(feature="stairs", explanation="no citations", citations=[]),
        GroundedExplanation(feature="warp_drive", explanation="unknown feature", citations=[citation()]),
    ])
    r = run()
    assert [e.feature for e in r.explanations] == ["ramp"]
    assert r.complete                                        # dropping ungrounded text is not a failure


def test_rag_failure_is_a_warning_not_an_error(fakes):
    def boom(_):
        raise RuntimeError("vector store down")

    fakes.rag = boom
    r = run()
    assert not r.complete and "guideline verification skipped" in r.warnings[0]
    assert r.explanations == [] and r.scoring.overall_score > 50


def test_rag_receives_claims_for_every_feature_with_honest_statuses(fakes):
    fakes.text = lambda inp: TextAnalysisOutput(location_id=inp.location_id, observations=[
        obs("ramp"), obs("stairs", "not_detected"),
        obs("lift_or_elevator_sign", "detected", 0.9, "osm"), obs("lift_or_elevator_sign", "not_detected", 0.9, "text"),
    ])
    run()
    claims = {c.feature: c for c in fakes.rag_inputs[0].claims}
    assert len(claims) == 10
    assert claims["ramp"].status == "detected" and claims["stairs"].status == "not_detected"
    assert claims["handrail"].status == "not_verified" and claims["handrail"].confidence is None
    assert claims["lift_or_elevator_sign"].status == "not_verified"      # contradictory -> not claimed either way


def test_a_failed_evidence_source_makes_the_report_incomplete(fakes):
    status = {"osm": "ok (looked up W1)", "web_text": "unavailable: no results from 3 search attempt(s)",
              "images": "ok: none found on Wikimedia Commons"}
    r = svc.run_pipeline(LOC, [], [], [], source_status=status)
    assert not r.complete
    assert len(r.warnings) == 1 and "web_text" in r.warnings[0] and "unavailable" in r.warnings[0]
    assert r.evidence.sources == status                      # shown to the user


def test_searching_and_finding_nothing_is_a_complete_result(fakes):
    status = {"osm": "ok (looked up W1)", "web_text": "ok: nothing relevant in 15 results from 3 search(es)"}
    r = svc.run_pipeline(LOC, [], [], [], source_status=status)
    assert r.complete and r.warnings == []


def test_real_teammate_stubs_do_not_break_the_pipeline():
    """The shipped starter stubs: image analysis finds nothing, RAG has no citations."""
    r = svc.run_pipeline(LOC, [OSMTag(key="wheelchair", value="yes")], ["https://x/a.jpg"], [])
    assert r.complete and r.explanations == []
    assert "accessible_entrance" in r.scoring.verified_features


# ── API ──────────────────────────────────────────────────────────────────

_ids = itertools.count(1)


@pytest.fixture
def world(client, db, monkeypatch, fakes):
    """A fake resolver + fake web evidence, with call counters."""
    state = {"fetches": 0, "fetch_error": None, "ref": "TEST-ASSESS-1", "status": {"osm": "ok", "web_text": "ok"}}

    def resolve(_input):
        return LocationResolutionOutput(
            location_id=f"loc_asstest{next(_ids):05d}", name="Assess Mall", category="mall",
            latitude=18.56, longitude=73.91, address="Pune", osm_ref=state["ref"])

    def fetch(agent_input):
        state["fetches"] += 1
        if state["fetch_error"]:
            raise RuntimeError(state["fetch_error"])
        return WebEvidenceOutput(location_id=agent_input.location_id,
                                 osm_tags=[OSMTag(key="wheelchair", value="yes")],
                                 image_urls=[], text_snippets=["Ramps available for entry."],
                                 source_status=dict(state["status"]))

    monkeypatch.setattr(location_store, "resolve_location", resolve)
    monkeypatch.setattr(evidence_service, "fetch_web_evidence", fetch)
    state["fakes"] = fakes
    return state


def assess(client, refresh=False, query="assess mall pune"):
    return client.post("/assess", json={"query": query, "refresh": refresh})


def test_assess_by_name_runs_everything_and_stores_it(client, db, world):
    r = assess(client)
    assert r.status_code == 200
    body = r.json()
    assert body["cached"] is False and body["complete"] is True and body["warnings"] == []
    assert body["location"]["name"] == "Assess Mall"
    assert 0 <= body["scoring"]["overall_score"] <= 100 and body["scoring"]["features"]
    assert body["evidence"] == {**body["evidence"], "osm_tag_count": 1, "snippet_count": 1}
    assert "geom" not in str(body)
    row = db.get(Assessment, body["assessment_id"])
    assert row.complete and row.overall_score == body["scoring"]["overall_score"]
    assert db.query(Location).filter(Location.osm_ref == "TEST-ASSESS-1").count() == 1
    assert db.query(Evidence).filter(Evidence.location_id == body["location"]["location_id"]).count() == 1


def test_second_request_is_served_from_cache(client, world):
    first = assess(client).json()
    second = assess(client).json()                         # same place, different request
    assert second["cached"] is True and second["assessment_id"] == first["assessment_id"]
    assert second["scoring"] == first["scoring"]
    assert world["fetches"] == 1 and world["fakes"].text_calls == 1


def test_refresh_collects_new_evidence_and_reassesses(client, db, world):
    first = assess(client).json()
    second = assess(client, refresh=True).json()
    assert second["cached"] is False and second["assessment_id"] != first["assessment_id"]
    assert world["fetches"] == 2
    assert db.query(Evidence).filter(Evidence.location_id == first["location"]["location_id"]).count() == 2


def test_incomplete_assessment_is_returned_but_never_cached(client, db, world):
    fakes = world["fakes"]
    fakes.text = lambda inp: TextAnalysisOutput(
        location_id=inp.location_id, observations=[obs("ramp")], warnings=["Web-text analysis skipped: Gemini busy"])
    first = assess(client).json()
    assert first["complete"] is False and first["warnings"]
    assert db.get(Assessment, first["assessment_id"]).complete is False

    # Gemini recovers: the retry re-runs the analysis but reuses the stored evidence
    fakes.text = lambda inp: TextAnalysisOutput(location_id=inp.location_id, observations=[obs("ramp")])
    second = assess(client).json()
    assert second["cached"] is False and second["complete"] is True
    assert second["assessment_id"] != first["assessment_id"]
    assert world["fetches"] == 1
    third = assess(client).json()                          # now complete -> cached
    assert third["cached"] is True and third["assessment_id"] == second["assessment_id"]


def test_evidence_with_a_failed_source_is_reported_and_collected_again_next_time(client, db, world):
    world["status"] = {"osm": "ok", "web_text": "unavailable: no results from 3 search attempt(s)"}
    first = assess(client).json()
    assert first["complete"] is False
    assert "web_text" in first["warnings"][0] and first["evidence"]["sources"]["web_text"].startswith("unavailable")

    world["status"] = {"osm": "ok", "web_text": "ok"}              # DuckDuckGo recovers
    second = assess(client).json()
    assert world["fetches"] == 2                                   # NOT reused: it was collected incompletely
    assert second["complete"] is True and second["cached"] is False
    third = assess(client).json()
    assert third["cached"] is True and world["fetches"] == 2


def test_evidence_from_before_source_statuses_existed_is_collected_again(client, db, world):
    loc_id = assess(client).json()["location"]["location_id"]
    db.query(Evidence).filter(Evidence.location_id == loc_id).update({"source_status": None})
    db.flush()
    fresh = client.post(f"/locations/{loc_id}/assess", params={"refresh": "false"})
    assert fresh.status_code == 200 and world["fetches"] == 1       # still served from the cached assessment
    again = client.post(f"/locations/{loc_id}/assess", params={"refresh": "true"})
    assert world["fetches"] == 2 and again.json()["cached"] is False
    db.query(Assessment).filter(Assessment.location_id == loc_id).delete()
    db.query(Evidence).filter(Evidence.location_id == loc_id).update({"source_status": None})
    db.flush()
    client.post(f"/locations/{loc_id}/assess")
    assert world["fetches"] == 3                                   # NULL status -> unknown quality -> redo


def test_new_evidence_invalidates_the_cached_assessment(client, world):
    first = assess(client).json()
    loc_id = first["location"]["location_id"]
    assert client.post(f"/locations/{loc_id}/evidence").status_code == 201
    again = client.post(f"/locations/{loc_id}/assess").json()
    assert again["cached"] is False and again["assessment_id"] != first["assessment_id"]
    assert world["fetches"] == 2                           # the manual one; assess reused it


def test_old_assessment_and_old_evidence_are_redone(client, db, world):
    first = assess(client).json()
    loc_id = first["location"]["location_id"]
    old = datetime.utcnow() - svc.MAX_AGE - timedelta(days=1)
    db.query(Assessment).filter(Assessment.location_id == loc_id).update({"created_at": old})
    db.query(Evidence).filter(Evidence.location_id == loc_id).update({"created_at": old})
    db.flush()
    second = client.post(f"/locations/{loc_id}/assess").json()
    assert second["cached"] is False and world["fetches"] == 2


def test_evidence_failure_is_a_502_and_stores_no_assessment(client, db, world):
    world["fetch_error"] = "ddgs rate limited"
    r = assess(client)
    assert r.status_code == 502 and "ddgs rate limited" in r.json()["detail"]
    loc = db.query(Location).filter(Location.osm_ref == "TEST-ASSESS-1").one()
    assert db.query(Assessment).filter(Assessment.location_id == loc.id).count() == 0


def test_assess_by_id_get_latest_and_errors(client, world):
    assert client.post("/locations/loc_missing/assess").status_code == 404
    assert client.get("/locations/loc_missing/assessment").status_code == 404
    assert assess(client, query="zz").status_code == 200
    assert client.post("/assess", json={"query": "x"}).status_code == 422          # too short
    loc_id = assess(client).json()["location"]["location_id"]
    latest = client.get(f"/locations/{loc_id}/assessment")
    assert latest.status_code == 200 and latest.json()["location"]["location_id"] == loc_id


def test_get_latest_before_any_assessment_is_404(client, world):
    loc = client.post("/locations", json={"query": "assess mall"}).json()
    r = client.get(f"/locations/{loc['location_id']}/assessment")
    assert r.status_code == 404 and "assess" in r.json()["detail"]


def test_unknown_place_maps_to_404(client, monkeypatch):
    def not_found(_):
        raise ValueError("no match for 'nowhere'")

    monkeypatch.setattr(location_store, "resolve_location", not_found)
    r = assess(client, query="nowhere at all")
    assert r.status_code == 404 and "nowhere" in r.json()["detail"]
