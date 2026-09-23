"""
SQLAlchemy models — the actual database tables.
"""

from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID  # no longer used here, safe to leave or remove the import

from app.database import Base


class Location(Base):
    __tablename__ = "locations"

    # String, not UUID — matches the location_id the Location Resolution
    # Agent generates (e.g. "loc_14db6842144f"), so the DB's primary key
    # IS the same id every other agent's contract expects. No translating
    # between two id schemes later.
    id = Column(String, primary_key=True)

    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geom = Column(Geography(geometry_type="POINT", srid=4326), nullable=True)
    address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)