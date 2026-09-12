"""
SQLAlchemy models — the actual database tables.

Location is the first and central table: every agent's output eventually
attaches to a location_id here. More tables (Evidence, Scores, guideline
chunks for RAG) get added as we build the agents that produce them —
we're not building the whole schema up front, just what Day 2 needs.
"""

import uuid
from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Location(Base):
    __tablename__ = "locations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # hospital | mall | restaurant | college | tourist_attraction
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # A PostGIS geography point, kept alongside the plain lat/lon columns
    # above. lat/lon are simpler to read and display; `geom` is what
    # spatial queries (ST_DWithin, for the Recommendation Agent's "find
    # nearby locations" later) actually query against.
    geom = Column(Geography(geometry_type="POINT", srid=4326), nullable=True)

    address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
