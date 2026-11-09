"""
API response schemas.

Why these exist: returning raw SQLAlchemy objects from an endpoint makes
FastAPI try to JSON-encode every column, including `geom`, which is a
PostGIS type with no JSON form (it raised a 500 as soon as a row had a
geom). Response schemas state exactly what leaves the API.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from app.agents.contracts import OSMTag


class EvidenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    location_id: str
    osm_tags: list[OSMTag]
    image_urls: list[str]
    text_snippets: list[str]
    created_at: datetime

    # The JSON columns are nullable in the database; the API always
    # presents "no data" as an empty list, never null.
    @field_validator("osm_tags", "image_urls", "text_snippets", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value):
        return [] if value is None else value