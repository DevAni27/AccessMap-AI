"""
Shared test fixtures.

API tests run against the real database (PostGIS behaviour is part of what
we are testing) but inside a transaction that is rolled back afterwards,
so they never leave rows behind. External services (Nominatim, ddgs,
Wikimedia, Gemini) are always replaced with fakes — tests need no network.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.database import engine, get_db
from app.main import app


@pytest.fixture(scope="session")
def _schema_is_current():
    columns = {c["name"] for c in inspect(engine).get_columns("locations")} \
        if inspect(engine).has_table("locations") else set()
    if "osm_ref" not in columns:
        pytest.exit(
            "Database schema is out of date. Run `python scripts/reset_db.py` first.",
            returncode=2,
        )


@pytest.fixture
def db(_schema_is_current):
    connection = engine.connect()
    outer = connection.begin()
    # Commits inside the app only release a SAVEPOINT, so the outer
    # transaction can still be rolled back at the end of the test.
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        connection.close()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()