"""
Location Resolution Agent.

Given a place name (e.g. "Phoenix Marketcity Pune"), resolves it to a
specific geographic location: coordinates, address, and a best-guess
category.

Uses Nominatim (OpenStreetMap's free geocoding API) — no API key needed,
but their usage policy requires max 1 request/second and a real
User-Agent identifying the app (their default-library-UA is explicitly
disallowed — that's why we set one below rather than using requests'
default).
"""

# FIND
import time
import uuid

import requests

from app.agents.contracts import LocationResolutionInput, LocationResolutionOutput
from app.llm_client import generate_text

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "AccessMapAI/0.1 (semester project; contact: nike.dhingra27@gmail.com)"

# REPLACE WITH
import uuid

import requests

from app.agents.contracts import LocationResolutionInput, LocationResolutionOutput
from app.http_utils import USER_AGENT, nominatim_throttle
from app.llm_client import generate_text

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim reports an object's kind as "node"/"way"/"relation". osm_ref is
# that kind's one-letter prefix + the OSM id (way 123 -> "W123"); it is the
# key we use to avoid storing the same real-world place twice.
_OSM_TYPE_PREFIX = {"node": "N", "way": "W", "relation": "R"}

# Project scope is Pune only. Constraining Nominatim's search to Pune's
# bounding box eliminates the most common ambiguity: a generic name (e.g.
# "City Hospital") matching a place in a completely different city or
# country. viewbox + bounded=1 tells Nominatim to only consider results
# inside this box, not just prefer them.
# Format is left,top,right,bottom = min_lon,max_lat,max_lon,min_lat
PUNE_VIEWBOX = "73.70,18.70,74.00,18.40"

# Nominatim returns (class, type) tags. We map the ones we expect to see
# for our 5 categories — anything unmapped falls back to "unknown" rather
# than guessing wrong. Deliberately minimal for now: when we build the
# seed script for the curated Pune dataset, we already know each
# location's category and will pass it in directly instead of relying on
# this inference.
# OSM tags follow a documented, finite vocabulary (see wiki.openstreetmap.org/wiki/Map_features)
# — unlike free-text search queries, this is a bounded mapping problem.
# Still won't be perfect: OSM data quality varies place to place, especially
# for how Indian malls get tagged. That's a real data limitation, not a
# gap in this table — an LLM-based classifier (next up, post-review) is
# the actual fix for that, since it can reason about a name/address even
# when the OSM tags are missing or wrong.

VALID_CATEGORIES = {"hospital", "mall", "restaurant", "college", "tourist_attraction"}

CATEGORY_MAP = {
    ("amenity", "hospital"): "hospital",
    ("amenity", "clinic"): "hospital",
    ("amenity", "doctors"): "hospital",
    ("healthcare", "hospital"): "hospital",
    ("healthcare", "clinic"): "hospital",

    ("shop", "mall"): "mall",
    ("shop", "department_store"): "mall",

    ("amenity", "restaurant"): "restaurant",
    ("amenity", "cafe"): "restaurant",
    ("amenity", "fast_food"): "restaurant",
    ("amenity", "food_court"): "restaurant",
    ("amenity", "bar"): "restaurant",
    ("amenity", "pub"): "restaurant",

    ("amenity", "university"): "college",
    ("amenity", "college"): "college",

    ("tourism", "attraction"): "tourist_attraction",
    ("tourism", "museum"): "tourist_attraction",
    ("tourism", "viewpoint"): "tourist_attraction",
    ("tourism", "gallery"): "tourist_attraction",
    ("tourism", "zoo"): "tourist_attraction",
    ("tourism", "theme_park"): "tourist_attraction",
    ("historic", "fort"): "tourist_attraction",
    ("historic", "monument"): "tourist_attraction",
    ("historic", "castle"): "tourist_attraction",
    ("historic", "ruins"): "tourist_attraction",
    ("leisure", "park"): "tourist_attraction",
}



def _classify_category_with_llm(name: str, address: str | None, osm_class: str, osm_type: str) -> str:
    """
    Fallback for when OSM's own tags don't map cleanly. The LLM reasons
    about the actual name/address — this is what makes classification
    general (works on any place) instead of limited to a fixed table.
    """
    prompt = f"""You are classifying a real-world place in Pune, India into exactly one category.

Place name: {name}
Address: {address or "unknown"}
OpenStreetMap tag: {osm_class}={osm_type or "none"}

Respond with exactly one word from this list, and nothing else:
hospital, mall, restaurant, college, tourist_attraction, unknown

Use "unknown" only if you genuinely cannot tell from the name and address."""

    

    try:
        raw = generate_text(prompt).strip().lower()
    except Exception:
        # LLM call failed (network, rate limit, etc.) — degrade gracefully.
        # A classification failure should never break location resolution.
        return "unknown"

    return raw if raw in VALID_CATEGORIES else "unknown"


def _build_osm_ref(result: dict) -> str | None:
    """Stable OpenStreetMap identity for a Nominatim result, e.g. "W123456".
    Returns None if Nominatim gave no usable id."""
    prefix = _OSM_TYPE_PREFIX.get(result.get("osm_type"))
    osm_id = result.get("osm_id")
    if prefix is None or osm_id is None:
        return None
    return f"{prefix}{osm_id}"


def resolve_location(input: LocationResolutionInput) -> LocationResolutionOutput:
    nominatim_throttle.wait()

    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": input.query,
            "format": "jsonv2",
            "limit": 5,
            "addressdetails": 1,
            "viewbox": PUNE_VIEWBOX,
            "bounded": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(f"Nominatim found no match for: {input.query!r} within Pune")
    
    if len(results) > 1:
        print(f"[disambiguation] {len(results)} matches within Pune for {input.query!r} "
              f"— using top-ranked: {results[0].get('display_name')}")

    result = results[0]
    osm_class = result.get("category", "")
    osm_type = result.get("type", "")
    category = CATEGORY_MAP.get((osm_class, osm_type), "unknown")
    if category == "unknown":
        category = _classify_category_with_llm(
            name=result.get("display_name", "").split(",")[0].strip() or input.query,
            address=result.get("display_name"),
            osm_class=osm_class,
            osm_type=osm_type,
        )
    
    # Our own stable id — used everywhere else in the pipeline instead of
    # OSM's id, so we're not coupled to a third party's identifiers.
    location_id = f"loc_{uuid.uuid4().hex[:12]}"

    return LocationResolutionOutput(
        location_id=location_id,
        name=result.get("display_name", "").split(",")[0].strip() or input.query,
        category=category,
        latitude=float(result["lat"]),
        longitude=float(result["lon"]),
        address=result.get("display_name"),
        osm_ref=_build_osm_ref(result),
    )


if __name__ == "__main__":
    # Run this directly to test: python -m app.agents.location_resolution
    # (from backend/, venv active)
    test_input = LocationResolutionInput(query="Phoenix Marketcity Pune")
    result = resolve_location(test_input)
    print(result.model_dump_json(indent=2))
