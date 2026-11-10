"""
Try the Web Evidence agent on any place. Needs internet, no database, no Gemini.

    python scripts/try_evidence.py "Ruby Hall Clinic Pune"
    python scripts/try_evidence.py "Phoenix Marketcity Pune" "Shaniwar Wada Pune"

For each name it shows: which Nominatim candidate the Location Resolution
agent now picks (and why), what each evidence source did (ok / unavailable),
the names searched, and the evidence found. Takes ~10-40 s per place.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from app.agents.contracts import WebEvidenceInput
from app.agents.location_resolution import NOMINATIM_URL, PUNE_VIEWBOX, _build_osm_ref, choose_candidate
from app.agents.web_evidence import fetch_web_evidence
from app.http_utils import USER_AGENT, nominatim_throttle


def run(query: str) -> bool:
    print("=" * 78 + f"\n{query}\n" + "=" * 78)
    nominatim_throttle.wait()
    response = requests.get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": 5,
                                                   "viewbox": PUNE_VIEWBOX, "bounded": 1},
                            headers={"User-Agent": USER_AGENT}, timeout=10)
    response.raise_for_status()
    results = response.json()
    if not results:
        print("Nominatim found nothing inside Pune.")
        return False

    picked = choose_candidate(results)
    for i, r in enumerate(results, 1):
        mark = "  <== picked" if r is picked else ""
        print(f"  #{i} {_build_osm_ref(r):<13} {r.get('category')}={r.get('type'):<12} {r['display_name'][:60]}{mark}")

    name = picked["display_name"].split(",")[0].strip()
    out = fetch_web_evidence(WebEvidenceInput(
        location_id="loc_try", location_name=name, latitude=float(picked["lat"]),
        longitude=float(picked["lon"]), osm_ref=_build_osm_ref(picked)))

    print("\nSource status:")
    for source, status in out.source_status.items():
        print(f"  {source:9} {status}")
    print(f"\nOSM accessibility tags ({len(out.osm_tags)}):", ", ".join(f"{t.key}={t.value}" for t in out.osm_tags) or "none")
    print(f"Text snippets ({len(out.text_snippets)}):")
    for s in out.text_snippets:
        print("  -", s[:150] + ("..." if len(s) > 150 else ""))
    print(f"Images ({len(out.image_urls)}):")
    for u in out.image_urls[:5]:
        print("  -", u)
    print()
    return not any(v.startswith("unavailable") for v in out.source_status.values())


if __name__ == "__main__":
    queries = sys.argv[1:] or ["Ruby Hall Clinic Pune"]
    ok = [run(q) for q in queries]
    print("ALL SOURCES WORKED" if all(ok) else "SOME SOURCE WAS UNAVAILABLE (see above)")
    sys.exit(0 if all(ok) else 1)
