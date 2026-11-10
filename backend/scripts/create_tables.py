"""
Run this once to create all tables defined in app/models.py.

It is additive only: it creates missing tables but never alters or drops
existing ones (use scripts/reset_db.py when the schema changes during
early development; later we'll switch to Alembic migrations).

Run from inside backend/, with your venv active:
    python scripts/create_tables.py
"""

import os
import sys

# Make `app` importable when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from app import models  # noqa: F401 — import so Base.metadata knows about our tables
from app.database import Base, engine


def create_tables() -> None:
    with engine.begin() as conn:
        # Needed for the Geography column; harmless if already enabled.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(bind=engine)
    # create_all never alters existing tables; columns added after a table
    # was first created are added here (idempotent, keeps existing data).
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE evidence ADD COLUMN IF NOT EXISTS source_status JSON"))


if __name__ == "__main__":
    create_tables()
    print("Tables created successfully.")

