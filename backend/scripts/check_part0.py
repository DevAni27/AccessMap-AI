"""
Live check for Part 0 against a RUNNING server (uvicorn app.main:app --reload).

Talks to the real Nominatim, so it takes ~5 seconds. Prints PASS/FAIL per
check and exits non-zero if anything failed.

Run from inside backend/, with your venv active:
    python scripts/check_part0.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from sqlalchemy import text

from app.database import engine

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
QUERY = "Phoenix Marketcity Pune"
ALIAS = "Phoenix Mall Viman Nagar Pune"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label} {detail}".rstrip())
    if not ok:
        failures.append(label)


def search(query: str) -> requests.Response:
    return requests.post(f"{BASE}/locations", json={"query": query}, timeout=60)


first = search(QUERY)
check("first search succeeds (201 if new, 200 if stored earlier)",
      first.status_code in (200, 201), f"-> {first.status_code} {first.text[:200] if first.status_code >= 400 else ''}")
if first.status_code >= 400:
    sys.exit(1)
loc = first.json()
print(f"      stored as {loc['location_id']}  osm_ref={loc.get('osm_ref')}  category={loc['category']}")

second = search(QUERY)
check("repeating the search returns 200 (not a new row)", second.status_code == 200, f"-> {second.status_code}")
check("repeat returns the same location_id", second.json().get("location_id") == loc["location_id"])
check("osm_ref is filled in", bool(loc.get("osm_ref")))

listing = requests.get(f"{BASE}/locations", timeout=30)
check("GET /locations works now that geom is filled", listing.status_code == 200, f"-> {listing.status_code}")
if listing.status_code == 200:
    copies = [row for row in listing.json() if row.get("osm_ref") == loc["osm_ref"]]
    check("exactly one stored row for this place", len(copies) == 1, f"(found {len(copies)})")

with engine.connect() as conn:
    point = conn.execute(
        text("SELECT ST_Y(geom::geometry), ST_X(geom::geometry) FROM locations WHERE id = :i"),
        {"i": loc["location_id"]},
    ).one_or_none()
check("geom is derived from latitude/longitude",
      point is not None and point[0] is not None
      and abs(point[0] - loc["latitude"]) < 1e-6 and abs(point[1] - loc["longitude"]) < 1e-6)

# Informational only: whether a differently-worded search lands on the same OSM object depends on Nominatim.
alias = search(ALIAS)
if alias.status_code < 400:
    same = alias.json().get("location_id") == loc["location_id"]
    print(f"INFO  {ALIAS!r} -> {'same place (deduped)' if same else 'a different OSM object: ' + alias.json()['name']}")

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}")
sys.exit(1 if failures else 0)