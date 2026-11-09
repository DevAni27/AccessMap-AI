"""
DANGER: deletes ALL data in our tables, then recreates them empty from
app/models.py.

Use only while the schema is still changing and the data is disposable
test data (true up to the curated-dataset seed run; after that, switch to
Alembic migrations instead of resetting).

Run from inside backend/, with your venv active:
    python scripts/reset_db.py          # asks for confirmation
    python scripts/reset_db.py --yes    # no prompt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect, text

from app import models  # noqa: F401
from app.database import Base, engine
from scripts.create_tables import create_tables


def _row_counts() -> dict[str, int]:
    existing = set(inspect(engine).get_table_names())
    counts = {}
    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name in existing:
                counts[table.name] = conn.execute(text(f'SELECT count(*) FROM "{table.name}"')).scalar()
    return counts


if __name__ == "__main__":
    counts = _row_counts()
    print("Rows that will be deleted:", counts or "(tables do not exist yet)")
    host = engine.url.host or "local socket"
    if "--yes" not in sys.argv:
        answer = input(f"Drop and recreate {[t.name for t in Base.metadata.sorted_tables]} on {host}? Type 'yes': ")
        if answer.strip().lower() != "yes":
            print("Aborted. Nothing changed.")
            sys.exit(1)

    Base.metadata.drop_all(bind=engine)   # dependency-ordered: evidence before locations
    create_tables()
    print("Done. Tables recreated empty.")