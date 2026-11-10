"""
Find out WHY web evidence came back empty for a place. No database, no Gemini.

    python scripts/debug_web_evidence.py                       (the Phoenix mall)
    python scripts/debug_web_evidence.py "Ruby Hall Clinic Pune"
    python scripts/debug_web_evidence.py "Shaniwar Wada Pune"

With a place name it first finds the place on OpenStreetMap (same search the
app uses), then tests the web search under both OpenStreetMap's name and the
text you typed.

Part 1: runs several wordings of the web search under the place's current and
        former names and shows, for each raw result, whether our filters would
        keep it (accessibility keyword? mentions the place?).
Part 2: compares the OpenStreetMap tags found by coordinates (what the agent
        does today) with a lookup by the object's own OSM id.

"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from ddgs import DDGS

from app.agents.web_evidence import _is_relevant, _reverse_geocode_tags, ddgs_throttle
from app.http_utils import USER_AGENT, nominatim_throttle

NAMES = ["Phoenix Avenue of Stars", "Phoenix Marketcity"]
OSM_REF, LAT, LON = "W169256025", 18.56204, 73.91666


def find_place(query: str) -> None:
    """Point NAMES / OSM_REF / LAT / LON at the place Nominatim finds for `query`."""
    global NAMES, OSM_REF, LAT, LON
    from app.agents.location_resolution import NOMINATIM_URL, PUNE_VIEWBOX
    nominatim_throttle.wait()
    r = requests.get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": 5,
                                            "viewbox": PUNE_VIEWBOX, "bounded": 1},
                     headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    results = r.json()
    if not results:
        sys.exit(f"Nominatim found nothing for {query!r} inside Pune.")
    top = results[0]
    official = top["display_name"].split(",")[0].strip()
    NAMES = [official] + ([query] if query.lower() != official.lower() else [])
    OSM_REF = f"{top['osm_type'][0].upper()}{top['osm_id']}"
    LAT, LON = float(top["lat"]), float(top["lon"])
    print(f"Place: {top['display_name'][:90]}\n  OSM id {OSM_REF} at {LAT}, {LON}; "
          f"{len(results)} candidate(s); testing names {NAMES}")

WORDINGS = [
    ("agent's current query", '"{n}" (wheelchair OR accessible OR accessibility OR ramp OR disability)'),
    ("no quotes, same OR list", '{n} (wheelchair OR accessible OR accessibility OR ramp OR disability)'),
    ("plain: name + wheelchair accessibility", '{n} wheelchair accessibility'),
    ("plain: name + Pune + wheelchair", '{n} Pune wheelchair accessible ramp'),
]


def web_search_report() -> None:
    print("=" * 78 + "\nPART 1 - web search\n" + "=" * 78)
    for name in NAMES:
        for label, template in WORDINGS:
            query = template.format(n=name)
            print(f"\n[{name}] {label}\n  query: {query}")
            try:
                ddgs_throttle.wait()
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
            except Exception as e:
                print(f"  ddgs ERROR: {type(e).__name__}: {e}")
                continue
            print(f"  ddgs returned {len(results)} result(s)")
            kept = 0
            for r in results:
                body = r.get("body") or ""
                ok = _is_relevant(r, NAMES)
                kept += ok
                print(f"    {'KEEP' if ok else 'drop':34} {body[:70]!r}")
            print(f"  -> {kept} snippet(s) would be stored")


def tags_report() -> None:
    print("\n" + "=" * 78 + "\nPART 2 - OpenStreetMap tags\n" + "=" * 78)
    try:
        by_coords = _reverse_geocode_tags(LAT, LON)
        print(f"\nby coordinates ({LAT}, {LON}) - what the agent does today: {len(by_coords)} tag(s)")
        for k, v in sorted(by_coords.items()):
            print(f"    {k} = {v}")
    except Exception as e:
        print("reverse geocode failed:", e)

    try:
        nominatim_throttle.wait()
        r = requests.get("https://nominatim.openstreetmap.org/lookup",
                         params={"osm_ids": OSM_REF, "format": "jsonv2", "extratags": 1, "namedetails": 1},
                         headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        rows = r.json()
        print(f"\nby OSM id {OSM_REF}: {len(rows)} object(s)")
        for row in rows:
            print("  names:", row.get("namedetails"))
            for k, v in sorted((row.get("extratags") or {}).items()):
                print(f"    {k} = {v}")
    except Exception as e:
        print("lookup failed:", e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_place(" ".join(sys.argv[1:]))
    web_search_report()
    tags_report()
