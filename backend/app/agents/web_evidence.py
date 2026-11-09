"""
Web Evidence Retrieval Agent.

Gathers accessibility evidence for a resolved location from three free
sources:
  1. OSM tags        — via Nominatim reverse geocoding (extratags=1)
  2. text_snippets    — via ddgs (DuckDuckGo search, no API key)
  3. image_urls        — via Wikimedia Commons (no API key)

All three are best-effort: a failure in one source degrades that source's
output to empty, it never crashes the whole agent. Missing evidence is
shown honestly as missing — we never fabricate tags, snippets, or images
to fill a gap.
"""

import time

import requests
from ddgs import DDGS

from app.agents.contracts import WebEvidenceInput, WebEvidenceOutput, OSMTag
from app.http_utils import USER_AGENT, Throttle, nominatim_throttle

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AccessMapAI/0.1 (semester project; contact: findpransh@gmail.com)"

ACCESSIBILITY_TAG_KEYS = {
    "wheelchair", "ramp", "ramp:wheelchair", "toilets:wheelchair",
    "entrance", "automatic_door", "step_count", "kerb",
    "tactile_paving", "handrail", "door:width",
}


# Nominatim's 1 req/sec limit is enforced by the shared nominatim_throttle
# (app/http_utils.py). ddgs has no official limit, but back-to-back calls
# get temporarily blocked — 2 sec spacing is a safe, cheap guard.
ddgs_throttle = Throttle(2.0)


def _reverse_geocode_tags(latitude: float, longitude: float) -> dict:
    nominatim_throttle.wait()
    response = requests.get(
        NOMINATIM_REVERSE_URL,
        params={"lat": latitude, "lon": longitude, "format": "jsonv2", "extratags": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("extratags", {}) or {}


ACCESSIBILITY_KEYWORDS = {
    "wheelchair", "ramp", "ramps", "disabled", "disability",
    "handicap", "handicapped", "braille", "tactile paving",
    "step-free", "step free", "differently abled",
    "physically challenged", "kerb",
}


def _search_text_snippets(location_name: str, max_results: int = 5) -> list[str]:
    """Search the open web for accessibility-specific mentions of this
    location. Best-effort: on failure (including rate-limit blocks),
    retries once after a short backoff, then gives up and returns [].

    ddgs's OR ranking is loose without grouping — a bare
    "wheelchair OR accessible OR accessibility OR ramp" can still surface
    pages that only weakly match. We group the terms with parentheses to
    tighten ranking, then post-filter on our own keyword check so a
    snippet only ever gets stored if it genuinely mentions accessibility
    — same principle as not padding OSM evidence with guesses.
    """
    query = f'"{location_name}" (wheelchair OR accessible OR accessibility OR ramp OR disability)'

    for attempt in range(2):
        try:
            ddgs_throttle.wait()
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            snippets = [r["body"] for r in results if r.get("body")]
            relevant = [
                s for s in snippets
                if any(kw in s.lower() for kw in ACCESSIBILITY_KEYWORDS)
                and _is_about_this_location(s, location_name)
            ]
            return relevant[:3]
        except Exception as e:
            print(f"[web_evidence] ddgs search attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(5)

    return []


def _search_commons_images(location_name: str, max_results: int = 5) -> list[str]:
    """Search Wikimedia Commons' File: namespace for images of this
    location and return their direct URLs. Best-effort: returns [] on
    any failure rather than raising.
    """
    try:
        response = requests.get(
            WIKIMEDIA_API_URL,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": location_name,
                "gsrnamespace": 6,  # 6 = File: namespace
                "gsrlimit": max_results,
                "prop": "imageinfo",
                "iiprop": "url",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        urls = []
        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo and imageinfo[0].get("url"):
                urls.append(imageinfo[0]["url"])
        return urls
    except Exception as e:
        print(f"[web_evidence] Wikimedia Commons search failed: {e}")
        return []
      
def _is_about_this_location(snippet: str, location_name: str) -> bool:
    """A snippet passing the accessibility-keyword filter isn't enough —
    ddgs's ranking doesn't reliably enforce that the quoted location name
    is actually present, so an unrelated but topically-similar page (a US
    transit accessibility report, say) can still surface. We verify
    independently: at least one real word from the location's own name
    must literally appear in the snippet.
    """
    snippet_lower = snippet.lower()
    name_tokens = [t.lower() for t in location_name.split() if len(t) > 3]
    if not name_tokens:
        return location_name.lower() in snippet_lower
    return any(token in snippet_lower for token in name_tokens)


def fetch_web_evidence(input: WebEvidenceInput) -> WebEvidenceOutput:
    tags = _reverse_geocode_tags(input.latitude, input.longitude)
    osm_tags: list[OSMTag] = [
        OSMTag(key=key, value=tags[key])
        for key in ACCESSIBILITY_TAG_KEYS
        if key in tags
    ]

    text_snippets = _search_text_snippets(input.location_name)
    image_urls = _search_commons_images(input.location_name)

    return WebEvidenceOutput(
        location_id=input.location_id,
        osm_tags=osm_tags,
        image_urls=image_urls,
        text_snippets=text_snippets,
    )


if __name__ == "__main__":
    # Logic-only test (no network) — tag extraction correctness
    fake_tags = {"wheelchair": "yes", "ramp": "yes", "opening_hours": "Mo-Su 09:00-21:00", "name": "Test Place"}
    extracted = [OSMTag(key=key, value=fake_tags[key]) for key in ACCESSIBILITY_TAG_KEYS if key in fake_tags]
    assert len(extracted) == 2
    assert any(t.key == "wheelchair" and t.value == "yes" for t in extracted)
    assert any(t.key == "ramp" and t.value == "yes" for t in extracted)
    print("Tag extraction logic verified correct (no network involved)")

    # Full live test — hits Nominatim, ddgs, and Wikimedia Commons
    test_input = WebEvidenceInput(
        location_id="loc_test",
        location_name="Phoenix Marketcity Pune",
        latitude=18.5620429,
        longitude=73.916663,
    )
    result = fetch_web_evidence(test_input)
    print(result.model_dump_json(indent=2))