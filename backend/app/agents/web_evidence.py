"""
Web Evidence Retrieval Agent.

Gathers accessibility evidence for a resolved location from three free
sources:
  1. OSM tags        — Nominatim lookup of the place's own OSM object
  2. text_snippets   — ddgs (DuckDuckGo search, no API key)
  3. image_urls      — Wikimedia Commons (no API key)

All three are best-effort: a failure in one source degrades that source's
output to empty, it never crashes the whole agent. Missing evidence is
shown honestly as missing — we never fabricate tags, snippets, or images
to fill a gap.

What changed after the Phoenix / Ruby Hall investigation (Nov 2026):
  * Every source reports a status. "ok" (even when nothing relevant was
    found: that is a finding) is kept apart from "unavailable: <why>" (the
    source failed). The pipeline warns on the latter and never caches it.
  * Tags are looked up by the object's OSM id, not by coordinates: the
    object at a place's coordinates can be a different one (a metro
    station beside a hospital).
  * The web search is intermittent (about half of the queries tried in
    testing returned "No results found" at random), so it makes several
    attempts with different phrasings, and under the place's alternative
    names from OSM (e.g. "Phoenix Avenue of Stars" / "Phoenix Marketcity").
  * Relevance is judged on title + snippet + URL, and a result must
    mention Pune: chains such as Phoenix Marketcity exist in many cities.
"""

import re

import requests
from ddgs import DDGS

from app.agents.contracts import OSMTag, WebEvidenceInput, WebEvidenceOutput
from app.http_utils import USER_AGENT, Throttle, nominatim_throttle

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_LOOKUP_URL = "https://nominatim.openstreetmap.org/lookup"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"

ACCESSIBILITY_TAG_KEYS = {
    "wheelchair", "ramp", "ramp:wheelchair", "toilets:wheelchair",
    "entrance", "automatic_door", "step_count", "kerb",
    "tactile_paving", "handrail", "door:width",
}

# Nominatim's 1 req/sec limit is enforced by the shared nominatim_throttle
# (app/http_utils.py). ddgs has no official limit, but back-to-back calls
# get temporarily blocked — 2 sec spacing is a safe, cheap guard.
ddgs_throttle = Throttle(2.0)

PROJECT_CITY = "pune"
MAX_SNIPPETS = 3            # snippets kept per location
MAX_SEARCH_ATTEMPTS = 6     # ddgs calls per location, across names and phrasings
MAX_ALT_NAMES = 2

# Plain wordings, no quotes or OR groups. The city is always included
# because many places are chains (Phoenix Marketcity: Pune, Mumbai, Chennai...).
QUERY_TEMPLATES = (
    "{name} Pune wheelchair accessible",
    "{name} Pune accessibility ramp lift disabled facilities",
    "{name} Pune wheelchair accessibility",
)

# "accessible"/"accessibility" alone are deliberately NOT keywords: far too
# ambiguous ("easy accessibility for shoppers"). These are specific.
ACCESSIBILITY_KEYWORDS = {
    "wheelchair", "ramp", "ramps", "disabled", "disability",
    "handicap", "handicapped", "braille", "tactile paving",
    "step-free", "step free", "differently abled",
    "physically challenged", "kerb",
}
_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(ACCESSIBILITY_KEYWORDS, key=len, reverse=True)) + r")\b"
)

# Words that don't identify a place on their own ("City Hospital", "Pune").
GENERIC_NAME_WORDS = {
    "pune", "hospital", "clinic", "mall", "centre", "center", "road", "city", "restaurant",
    "college", "university", "cafe", "hotel", "museum", "park", "fort", "temple",
}


# ── OSM tags ───────────────────────────────────────────────────────────────

def _reverse_geocode_tags(latitude: float, longitude: float) -> dict:
    """Tags of whatever object is at these coordinates. Only a fallback for
    places with no stored OSM id: it can return a neighbouring object."""
    nominatim_throttle.wait()
    response = requests.get(
        NOMINATIM_REVERSE_URL,
        params={"lat": latitude, "lon": longitude, "format": "jsonv2", "extratags": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("extratags", {}) or {}


def _lookup_object(osm_ref: str) -> dict | None:
    """Nominatim's record of one specific OSM object (tags + all its names),
    or None if Nominatim does not know it."""
    nominatim_throttle.wait()
    response = requests.get(
        NOMINATIM_LOOKUP_URL,
        params={"osm_ids": osm_ref, "format": "jsonv2", "extratags": 1, "namedetails": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else None


def _clean_name(name: str) -> str:
    """Drop the city word: our queries add it themselves."""
    return " ".join(re.sub(r"\bpune\b", " ", name, flags=re.IGNORECASE).split())


def _alternative_names(namedetails: dict, primary: str) -> list[str]:
    """Other names OSM knows the place by (alt_name, old_name, ...), e.g. the
    former name "Phoenix Marketcity" of "Phoenix Avenue of Stars"."""
    seen = {_clean_name(primary).casefold()}
    names: list[str] = []
    for key in ("alt_name", "old_name", "short_name", "official_name", "name:en"):
        for raw in str(namedetails.get(key) or "").split(";"):
            name = _clean_name(raw.strip())
            if len(name) >= 4 and name.casefold() not in seen:
                seen.add(name.casefold())
                names.append(name)
    return names[:MAX_ALT_NAMES]


def _osm_tags(tags: dict) -> list[OSMTag]:
    return [OSMTag(key=key, value=tags[key]) for key in sorted(ACCESSIBILITY_TAG_KEYS) if key in tags]


# ── web text ───────────────────────────────────────────────────────────────

def _ddgs_text(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _is_about_place(text: str, names: list[str]) -> bool:
    """`text` (already lowercase: title + snippet + URL) must mention the
    city AND the place: at least two distinctive words of one of its names
    (one if the name has only one). Chains in other cities fail the city test."""
    if PROJECT_CITY not in text:
        return False
    for name in names:
        words = re.findall(r"[a-z0-9]+", name.lower())
        distinctive = [w for w in words if len(w) > 3 and w not in GENERIC_NAME_WORDS]
        if distinctive:
            if sum(w in text for w in distinctive) >= min(2, len(distinctive)):
                return True
        elif words and " ".join(words) in text:
            return True
    return False


def _is_relevant(result: dict, names: list[str]) -> bool:
    title, body, href = (result.get(k) or "" for k in ("title", "body", "href"))
    if not body:
        return False
    content = f"{title} {body}".lower()
    return bool(_KEYWORD_RE.search(content)) and _is_about_place(f"{content} {href.lower()}", names)


def _is_about_this_location(snippet: str, location_name: str) -> bool:
    """Kept for the debug script: the place test applied to one snippet."""
    return _is_about_place(snippet.lower(), [location_name])


def _snippet_text(result: dict) -> str:
    title, body = (result.get("title") or "").strip(), result["body"].strip()
    return body if not title or title.lower() in body.lower() else f"{title} — {body}"


def search_text_snippets(names: list[str], max_results: int = 5,
                         max_attempts: int = MAX_SEARCH_ATTEMPTS) -> tuple[list[str], str]:
    """Search the web for accessibility mentions of the place, under each of
    its names and several wordings, until MAX_SNIPPETS relevant ones are
    found or the attempts run out. Returns (snippets, status):

      "ok"                          found relevant snippets
      "ok: nothing relevant ..."    the search worked but no result mentioned
                                    accessibility at this place (a finding)
      "unavailable: ..."            no attempt returned any result, so the
                                    search itself probably failed
    """
    attempts = [(name, template) for template in QUERY_TEMPLATES for name in names][:max_attempts]
    kept: list[str] = []
    raw_total = made = 0
    last_error = "no results"

    for name, template in attempts:
        if len(kept) >= MAX_SNIPPETS:
            break
        made += 1
        try:
            ddgs_throttle.wait()
            results = _ddgs_text(template.format(name=name), max_results)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"[web_evidence] search attempt {made} failed: {last_error}")
            continue
        raw_total += len(results)
        for result in results:
            if _is_relevant(result, names):
                snippet = _snippet_text(result)
                if snippet not in kept:
                    kept.append(snippet)

    kept = kept[:MAX_SNIPPETS]
    if kept:
        return kept, "ok"
    if raw_total:
        return [], f"ok: nothing relevant in {raw_total} results from {made} search(es)"
    return [], f"unavailable: no results from {made} search attempt(s) (last: {last_error})"


# ── images ─────────────────────────────────────────────────────────────────

def _search_commons_images(location_name: str, max_results: int = 5) -> list[str]:
    """Search Wikimedia Commons' File: namespace for images of this
    location and return their direct URLs. Raises on failure."""
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
    pages = response.json().get("query", {}).get("pages", {})
    urls = []
    for page in pages.values():
        imageinfo = page.get("imageinfo", [])
        if imageinfo and imageinfo[0].get("url"):
            urls.append(imageinfo[0]["url"])
    return urls


# ── the agent ──────────────────────────────────────────────────────────────

def fetch_web_evidence(input: WebEvidenceInput) -> WebEvidenceOutput:
    status: dict[str, str] = {}
    tags: dict = {}
    alt_names: list[str] = []

    # 1. OSM tags
    try:
        if input.osm_ref:
            obj = _lookup_object(input.osm_ref)
            if obj is None:
                status["osm"] = f"unavailable: Nominatim has no object {input.osm_ref}"
            else:
                tags = obj.get("extratags") or {}
                alt_names = _alternative_names(obj.get("namedetails") or {}, input.location_name)
                status["osm"] = f"ok (looked up {input.osm_ref})"
        else:
            tags = _reverse_geocode_tags(input.latitude, input.longitude)
            status["osm"] = "ok (by coordinates: no OSM id stored)"
    except Exception as e:
        status["osm"] = f"unavailable: {type(e).__name__}: {e}"

    names = [_clean_name(input.location_name) or input.location_name] + alt_names

    # 2. Web text
    text_snippets, status["web_text"] = search_text_snippets(names)

    # 3. Images (first name that finds any)
    image_urls: list[str] = []
    try:
        for name in names:
            image_urls = _search_commons_images(name)
            if image_urls:
                break
        status["images"] = "ok" if image_urls else "ok: none found on Wikimedia Commons"
    except Exception as e:
        status["images"] = f"unavailable: {type(e).__name__}: {e}"

    return WebEvidenceOutput(
        location_id=input.location_id,
        osm_tags=_osm_tags(tags),
        image_urls=image_urls,
        text_snippets=text_snippets,
        source_status=status,
    )


if __name__ == "__main__":
    # Logic-only test (no network) — tag extraction correctness
    fake_tags = {"wheelchair": "yes", "ramp": "yes", "opening_hours": "Mo-Su 09:00-21:00", "name": "Test Place"}
    extracted = _osm_tags(fake_tags)
    assert len(extracted) == 2
    assert any(t.key == "wheelchair" and t.value == "yes" for t in extracted)
    assert any(t.key == "ramp" and t.value == "yes" for t in extracted)
    print("Tag extraction logic verified correct (no network involved)")

    # Full live test — hits Nominatim, ddgs, and Wikimedia Commons
    result = fetch_web_evidence(WebEvidenceInput(
        location_id="loc_test",
        location_name="Phoenix Avenue of Stars",
        latitude=18.56204,
        longitude=73.91666,
        osm_ref="W169256025",
    ))
    print(result.model_dump_json(indent=2))
