"""Offline tests for the Web Evidence agent: relevance filtering, the
multi-attempt search, per-source status, and lookup by OSM id.

Every network call (Nominatim, ddgs, Wikimedia) is replaced by a fake; the
fake DuckDuckGo answers are modelled on what it really returned for
Phoenix Avenue of Stars and Ruby Hall Clinic."""

import pytest

from app.agents import web_evidence as we
from app.agents.contracts import WebEvidenceInput


class NoResults(Exception):
    """Stands in for ddgs's DDGSException: No results found."""


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr(we.ddgs_throttle, "wait", lambda: None)
    monkeypatch.setattr(we.nominatim_throttle, "wait", lambda: None)


def result(title="", body="", href=""):
    return {"title": title, "body": body, "href": href}


GOOD = result("Ruby Hall Clinic Pune - Facilities", "Wheelchair ramps and lifts are available at the Pune campus.",
              "https://example.org/pune/ruby-hall")


# ── names ────────────────────────────────────────────────────────────────

def test_clean_name_drops_the_city_word():
    assert we._clean_name("Ruby Hall Clinic Pune") == "Ruby Hall Clinic"
    assert we._clean_name("Pune Central Mall") == "Central Mall"
    assert we._clean_name("Punekar Sweets") == "Punekar Sweets"        # only the whole word


def test_alternative_names_come_from_osm_and_skip_the_primary():
    details = {"name": "Phoenix Avenue of Stars", "alt_name": "Phoenix Marketcity",
               "alt_name:mr": "marathi", "old_name": "Phoenix Marketcity; Phoenix Mall", "short_name": "PAS"}
    assert we._alternative_names(details, "Phoenix Avenue of Stars") == ["Phoenix Marketcity", "Phoenix Mall"]
    assert we._alternative_names({"alt_name": "ruby hall clinic pune"}, "Ruby Hall Clinic") == []
    assert we._alternative_names({}, "Anything") == []


# ── relevance ────────────────────────────────────────────────────────────

def test_relevant_snippet_is_kept():
    assert we._is_relevant(GOOD, ["Ruby Hall Clinic"])


def test_name_in_the_title_only_counts():
    """The real Ruby Hall result 'Accessibility Options: Lifts ... Disabled People' had the name
    only in its title / URL, never in the snippet body."""
    r = result("Ruby Hall Clinic, Pune - Hospital facilities",
               "Accessibility Options Lifts / Escalators Accessible to Disabled People", "https://x.org/pune/1")
    assert we._is_relevant(r, ["Ruby Hall Clinic"])


def test_chain_in_another_city_is_rejected():
    kurla = result("Phoenix Marketcity Kurla", "Visitors needing a wheelchair can ask at the help desk.",
                   "https://www.phoenixmarketcity.com/mumbai")
    assert not we._is_relevant(kurla, ["Phoenix Marketcity"])
    pune = result("Phoenix Marketcity Pune", "Visitors needing a wheelchair can ask at the help desk.",
                  "https://www.phoenixmarketcity.com/pune")
    assert we._is_relevant(pune, ["Phoenix Marketcity"])


def test_unrelated_accessibility_text_is_rejected():
    r = result("", "Wheelchair accessible seating is offered at all price points on a first come basis in Pune.")
    assert not we._is_relevant(r, ["Phoenix Avenue of Stars"])


def test_vague_accessibility_without_a_specific_keyword_is_rejected():
    r = result("Phoenix Avenue of Stars Pune", "Easy accessibility and a convenient shopping experience in Pune.")
    assert not we._is_relevant(r, ["Phoenix Avenue of Stars"])


def test_keywords_need_word_boundaries():
    r = result("Phoenix Avenue of Stars Pune", "Pune startup RampMyCity launches a rampage of offers at Phoenix Avenue of Stars.")
    assert not we._is_relevant(r, ["Phoenix Avenue of Stars"])
    tramp = result("Phoenix Avenue of Stars Pune", "A tramp walked past Phoenix Avenue of Stars in Pune.")
    assert not we._is_relevant(tramp, ["Phoenix Avenue of Stars"])


def test_a_generic_name_needs_the_full_phrase():
    assert we._is_about_place("city hospital pune", ["City Hospital"])
    assert not we._is_about_place("the hospital in pune", ["City Hospital"])


def test_one_distinctive_word_is_not_enough_for_a_two_word_name():
    assert not we._is_about_place("ruby tuesday restaurant pune", ["Ruby Hall Clinic"])


def test_snippet_text_carries_the_title_unless_already_in_the_body():
    assert we._snippet_text(result("Title", "Body text")) == "Title — Body text"
    assert we._snippet_text(result("Title", "About Title here")) == "About Title here"
    assert we._snippet_text(result("", "Body text")) == "Body text"


# ── multi-attempt search ─────────────────────────────────────────────────

def fake_ddgs(monkeypatch, behaviour):
    calls = []

    def fake(query, max_results):
        calls.append(query)
        return behaviour(len(calls), query)

    monkeypatch.setattr(we, "_ddgs_text", fake)
    return calls


def test_search_retries_after_random_failures(monkeypatch):
    def behaviour(n, query):
        if n < 3:
            raise NoResults("No results found.")
        return [GOOD]

    calls = fake_ddgs(monkeypatch, behaviour)
    snippets, status = we.search_text_snippets(["Ruby Hall Clinic"])
    assert status == "ok" and len(snippets) == 1 and len(calls) == 3
    assert all("Pune" in q and '"' not in q and " OR " not in q for q in calls)   # plain wordings


def test_search_uses_alternative_names_too(monkeypatch):
    calls = fake_ddgs(monkeypatch, lambda n, q: (_ for _ in ()).throw(NoResults("none")))
    we.search_text_snippets(["Phoenix Avenue of Stars", "Phoenix Marketcity"])
    assert any("Phoenix Marketcity" in q for q in calls) and any("Avenue of Stars" in q for q in calls)
    assert len(calls) == we.MAX_SEARCH_ATTEMPTS


def test_all_attempts_failing_is_unavailable_not_empty(monkeypatch):
    fake_ddgs(monkeypatch, lambda n, q: (_ for _ in ()).throw(NoResults("No results found.")))
    snippets, status = we.search_text_snippets(["Ruby Hall Clinic"])
    assert snippets == [] and status.startswith("unavailable") and "3 search attempt(s)" in status


def test_results_without_accessibility_text_is_a_finding_not_a_failure(monkeypatch):
    marketing = result("Ruby Hall Clinic Pune", "A premier multi-specialty hospital in Pune.", "https://x/pune")
    fake_ddgs(monkeypatch, lambda n, q: [marketing] * 5)
    snippets, status = we.search_text_snippets(["Ruby Hall Clinic"])
    assert snippets == [] and status.startswith("ok: nothing relevant") and not status.startswith("unavailable")


def test_search_stops_once_enough_snippets_are_found(monkeypatch):
    def behaviour(n, q):
        return [result("Ruby Hall Clinic Pune", f"Wheelchair ramp number {n}{i} at the entrance.", "https://x/pune")
                for i in range(3)]

    calls = fake_ddgs(monkeypatch, behaviour)
    snippets, _ = we.search_text_snippets(["Ruby Hall Clinic"])
    assert len(snippets) == we.MAX_SNIPPETS and len(calls) == 1


def test_duplicate_snippets_are_stored_once(monkeypatch):
    fake_ddgs(monkeypatch, lambda n, q: [GOOD, GOOD])
    snippets, _ = we.search_text_snippets(["Ruby Hall Clinic"])
    assert len(snippets) == 1


def test_a_failed_attempt_between_good_ones_does_not_lose_results(monkeypatch):
    def behaviour(n, q):
        if n == 2:
            raise TimeoutError("mojeek timed out")
        return [result("Ruby Hall Clinic Pune", f"Wheelchair ramp {n} at the main entrance.", "https://x/pune")]

    fake_ddgs(monkeypatch, behaviour)
    snippets, status = we.search_text_snippets(["Ruby Hall Clinic"])
    assert status == "ok" and len(snippets) == 2


# ── fetch_web_evidence: sources and statuses ─────────────────────────────

def make_input(osm_ref="W53098668", name="Ruby Hall Clinic"):
    return WebEvidenceInput(location_id="loc_x", location_name=name, latitude=18.53, longitude=73.87, osm_ref=osm_ref)


@pytest.fixture
def world(monkeypatch):
    state = {
        "object": {"extratags": {"wheelchair": "yes", "website": "https://x", "opening_hours": "24/7"},
                   "namedetails": {"name": "Ruby Hall Clinic", "alt_name": "Ruby Clinic Sassoon"}},
        "images": ["https://upload.wikimedia.org/a.jpg"],
        "lookups": [], "reverse_calls": 0, "image_queries": [],
    }

    def lookup(ref):
        state["lookups"].append(ref)
        if isinstance(state["object"], Exception):
            raise state["object"]
        return state["object"]

    def reverse(lat, lon):
        state["reverse_calls"] += 1
        return {"wheelchair": "no"}

    def images(name, max_results=5):
        state["image_queries"].append(name)
        if isinstance(state["images"], Exception):
            raise state["images"]
        return state["images"]

    monkeypatch.setattr(we, "_lookup_object", lookup)
    monkeypatch.setattr(we, "_reverse_geocode_tags", reverse)
    monkeypatch.setattr(we, "_search_commons_images", images)
    monkeypatch.setattr(we, "_ddgs_text", lambda q, n: [GOOD])
    return state


def test_tags_are_looked_up_by_osm_id_never_by_coordinates(world):
    out = we.fetch_web_evidence(make_input())
    assert world["lookups"] == ["W53098668"] and world["reverse_calls"] == 0
    assert [(t.key, t.value) for t in out.osm_tags] == [("wheelchair", "yes")]      # only accessibility tags kept
    assert out.source_status["osm"] == "ok (looked up W53098668)"


def test_without_an_osm_id_coordinates_are_the_fallback(world):
    out = we.fetch_web_evidence(make_input(osm_ref=None))
    assert world["lookups"] == [] and world["reverse_calls"] == 1
    assert "coordinates" in out.source_status["osm"] and out.osm_tags[0].value == "no"


def test_unknown_osm_object_is_reported_and_does_not_fall_back_to_coordinates(world):
    world["object"] = None
    out = we.fetch_web_evidence(make_input())
    assert out.osm_tags == [] and out.source_status["osm"].startswith("unavailable")
    assert world["reverse_calls"] == 0               # coordinates could return a neighbouring object


def test_nominatim_failure_is_unavailable_and_other_sources_still_run(world):
    world["object"] = ConnectionError("nominatim down")
    out = we.fetch_web_evidence(make_input())
    assert out.source_status["osm"].startswith("unavailable") and "nominatim down" in out.source_status["osm"]
    assert out.source_status["web_text"] == "ok" and out.text_snippets and out.image_urls


def test_image_failure_is_unavailable_and_text_still_works(world):
    world["images"] = TimeoutError("commons timeout")
    out = we.fetch_web_evidence(make_input())
    assert out.source_status["images"].startswith("unavailable") and out.image_urls == []
    assert out.source_status["web_text"] == "ok"


def test_no_images_found_is_not_a_failure(world):
    world["images"] = []
    out = we.fetch_web_evidence(make_input())
    assert out.source_status["images"].startswith("ok") and not out.source_status["images"].startswith("unavailable")


def test_alternative_names_are_searched_and_used_for_images(world):
    world["images"] = []
    we.fetch_web_evidence(make_input())
    assert world["image_queries"] == ["Ruby Hall Clinic", "Ruby Clinic Sassoon"]


def test_input_name_with_city_word_is_cleaned(world, monkeypatch):
    queries = []
    monkeypatch.setattr(we, "_ddgs_text", lambda q, n: queries.append(q) or [])
    we.fetch_web_evidence(make_input(name="Ruby Hall Clinic Pune"))
    assert all("Pune Pune" not in q for q in queries)
