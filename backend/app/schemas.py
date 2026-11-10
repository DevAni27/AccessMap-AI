"""
API response schemas.

Why these exist: returning raw SQLAlchemy objects from an endpoint makes
FastAPI try to JSON-encode every column, including `geom`, which is a
PostGIS type with no JSON form (it raised a 500 as soon as a row had a
geom). Response schemas state exactly what leaves the API.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.contracts import (
    GroundedExplanation,
    LocationResolutionOutput,
    Observation,
    OSMTag,
    ScoringOutput,
)


class EvidenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    location_id: str
    osm_tags: list[OSMTag]
    image_urls: list[str]
    text_snippets: list[str]
    source_status: dict[str, str]
    created_at: datetime

    # The JSON columns are nullable in the database; the API always
    # presents "no data" as an empty list, never null.
    @field_validator("osm_tags", "image_urls", "text_snippets", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value):
        return [] if value is None else value

    @field_validator("source_status", mode="before")
    @classmethod
    def _none_to_empty_dict(cls, value):
        return {} if value is None else value


class EvidenceSummary(BaseModel):
    """How much raw evidence the assessment was based on."""
    osm_tag_count: int
    image_count: int
    snippet_count: int
    collected_at: datetime | None = None
    # What each evidence source did ("ok..." or "unavailable: why")
    sources: dict[str, str] = {}


class AssessmentReport(BaseModel):
    """Everything about one assessment run. Stored as JSON in the database
    and returned as-is by the API."""
    location: LocationResolutionOutput
    scoring: ScoringOutput
    observations: list[Observation]          # the evidence behind every number
    explanations: list[GroundedExplanation]  # only ones backed by guideline citations
    evidence: EvidenceSummary
    # Anything an agent could not do (LLM down, image analysis crashed, ...).
    # An assessment with warnings is shown but never cached.
    warnings: list[str] = []
    complete: bool
    generated_at: datetime


class AssessmentResponse(AssessmentReport):
    assessment_id: int
    cached: bool            # True when an earlier stored assessment was reused


class AssessRequest(BaseModel):
    query: str = Field(min_length=2, description='A place name, e.g. "Phoenix Marketcity Pune"')
    refresh: bool = Field(False, description="Collect fresh evidence and re-assess even if a stored assessment exists")
