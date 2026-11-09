"""
SQLAlchemy models — the actual database tables.
"""

from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    Column, Computed, DateTime, Float, ForeignKey, Integer, JSON, String,
)

from app.database import Base


class Location(Base):
    __tablename__ = "locations"

    id = Column(String, primary_key=True)  # our own id, e.g. "loc_ab12cd34ef56"

    # OpenStreetMap identity of the place, e.g. "W123456" (way), "N42" (node),
    # "R7" (relation). UNIQUE, so the same real-world place can only be
    # stored once no matter how the user phrased the search. NULL (rare:
    # Nominatim gave no OSM id) is exempt from uniqueness in Postgres.
    osm_ref = Column(String, unique=True, nullable=True)

    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # Derived by the database from latitude/longitude, never written by
    # application code, so it cannot drift out of sync with them. Powers
    # the "nearby alternatives" search (PostGIS ST_DWithin / ST_Distance).
    # Note PostGIS point order is (longitude, latitude).
    geom = Column(
        Geography(geometry_type="POINT", srid=4326),
        Computed(
            "ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography",
            persisted=True,
        ),
    )

    address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Evidence(Base):
    """
    One row per web-evidence-retrieval run for a location. Re-running the
    agent later adds a new row rather than overwriting — Evidence Fusion
    (later) decides how to combine them; for now we keep every run.
    """
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String, ForeignKey("locations.id"), nullable=False, index=True)
    osm_tags = Column(JSON, nullable=True)       # list of {"key": ..., "value": ...}
    image_urls = Column(JSON, nullable=True)     # list of strings
    text_snippets = Column(JSON, nullable=True)  # list of strings
    created_at = Column(DateTime, default=datetime.utcnow)