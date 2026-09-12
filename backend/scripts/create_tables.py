"""
Run this once to create all tables defined in app/models.py.

Later, once the schema stabilizes, we'll likely switch to Alembic for real
version-controlled migrations. For now, at this early stage where the
schema is still changing weekly, this is simpler and perfectly fine —
introducing a migrations tool before you need one is extra ceremony with
no payoff yet.

Run from inside backend/, with your venv active:
    python scripts/create_tables.py
"""

import sys
import os

# Make `app` importable when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base, engine
from app import models  # noqa: F401 — import so Base.metadata knows about Location

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")
