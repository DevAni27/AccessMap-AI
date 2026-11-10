"""
Show what Nominatim returns for a place name, WITHOUT touching the database.

    python scripts/debug_resolution.py
    python scripts/debug_resolution.py "Phoenix Marketcity Pune" "Seasons Mall"

For every query it prints the raw candidates (what the Location Resolution
agent chooses its answer from; it always takes #1), first with the Pune
bounding box the agent uses, then without it for comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from app.agents.location_resolution import NOMINATIM_URL, PUNE_VIEWBOX
from app.http_utils import USER_AGENT, nominatim_throttle

DEFAULT_QUERIES = [
    "Phoenix Marketcity Pune",
    "Phoenix Marketcity",
    "Phoenix Marketcity Viman Nagar Pune",
    "Phoenix Mall Pune",
]


def search(query: str, bounded: bool) -> list[dict]:
    nominatim_throttle.wait()
    params = {"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1}
    if bounded:
        params.update(viewbox=PUNE_VIEWBOX, bounded=1)
    r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    return r.json()


def show(query: str, bounded: bool) -> None:
    label = "inside Pune box (what the agent does)" if bounded else "no box (comparison only)"
    results = search(query, bounded)
    print(f"\n{query!r} - {label}: {len(results)} result(s)")
    for i, res in enumerate(results, 1):
        ref = f"{res['osm_type'][0].upper()}{res['osm_id']}"
        print(f"  #{i} {ref:<12} {res.get('category')}={res.get('type'):<12} "
              f"{res['lat'][:8]},{res['lon'][:8]}  {res['display_name'][:85]}")


if __name__ == "__main__":
    for q in (sys.argv[1:] or DEFAULT_QUERIES):
        for bounded in (True, False):
            show(q, bounded)
