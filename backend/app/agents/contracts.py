"""
Shared data contracts for every agent in the pipeline.

THIS FILE IS THE SOURCE OF TRUTH. Every agent — whoever writes it — takes one
of these *Input models in and returns one of these *Output models out. No
agent talks to the database directly; persistence happens in routers/scripts
that call these functions, not inside the agents themselves.

If you need to change a shape here, that's a conversation between both
people on the team first, since both sides depend on it staying stable.

IMPORTANT — embedding dimension lock-in:
  We're using sentence-transformers/all-MiniLM-L6-v2 for ALL embeddings
  (guideline chunks and any text evidence). That model outputs 384-dim
  vectors. The Neon DB's pgvector columns will be declared `vector(384)`.
  If you use a different embedding model, the numbers literally won't fit
  in the database column. Don't change this without updating both sides.
"""

from typing import Optional
from pydantic import BaseModel


# ── Location Resolution Agent (Pranshu, Day 4-5) ──────────────────────────

class LocationResolutionInput(BaseModel):
    query: str                      # what the user typed, e.g. "Phoenix Mall Pune"


class LocationResolutionOutput(BaseModel):
    location_id: str                # our own generated id, e.g. "loc_0001"
    name: str
    category: str                   # "hospital" | "mall" | "restaurant" | "college" | "tourist_attraction"
    latitude: float
    longitude: float
    address: Optional[str] = None


# ── Web Evidence Retrieval Agent (Pranshu, Day 6-7) ────────────────────────

class WebEvidenceInput(BaseModel):
    location_id: str
    location_name: str
    latitude: float
    longitude: float


class OSMTag(BaseModel):
    key: str                        # e.g. "wheelchair"
    value: str                      # e.g. "yes" / "limited" / "no"


class WebEvidenceOutput(BaseModel):
    location_id: str
    osm_tags: list[OSMTag]
    image_urls: list[str]           # collected from Wikimedia / official sites
    text_snippets: list[str]        # scraped/searched accessibility-related text


# ── Image Analysis Agent (Teammate, builds this) ───────────────────────────

ACCESSIBILITY_FEATURE_CLASSES = [
    "ramp",
    "stairs",
    "handrail",
    "accessible_entrance",
    "lift_or_elevator_sign",
    "accessible_toilet_sign",
    "accessible_parking_sign",
    "pathway_obstruction",
]


class ImageAnalysisInput(BaseModel):
    location_id: str
    location_name: str
    image_urls: list[str]


class FeatureFinding(BaseModel):
    # detected: True/False, or None meaning "not verified" (no evidence either way —
    # NEVER treat missing evidence as "not available", per the project's design)
    detected: Optional[bool] = None
    confidence: Optional[float] = None   # 0.0-1.0, None if not verified
    source_images: list[str] = []
    detection_method: Optional[str] = None  # "yolo" | "vlm" — which model found it


class ImageAnalysisOutput(BaseModel):
    location_id: str
    features: dict[str, FeatureFinding]   # keys are ACCESSIBILITY_FEATURE_CLASSES
    processed_image_count: int


# ── RAG Verification Agent (Teammate, builds this) ─────────────────────────

class Claim(BaseModel):
    feature: str                    # e.g. "ramp" — should match ACCESSIBILITY_FEATURE_CLASSES
    status: str                     # "detected" | "not_detected" | "not_verified"
    confidence: Optional[float] = None


class RAGVerificationInput(BaseModel):
    location_id: str
    category: str                   # location category, some guideline clauses are category-specific
    claims: list[Claim]


class GuidelineCitation(BaseModel):
    clause: str                     # e.g. "4.2"
    text_excerpt: str
    source: str                     # e.g. "Harmonised Guidelines 2021, p.14"


class GroundedExplanation(BaseModel):
    feature: str
    explanation: str                # LLM-generated, grounded in the citations below
    citations: list[GuidelineCitation]


class RAGVerificationOutput(BaseModel):
    location_id: str
    explanations: list[GroundedExplanation]


# ── Evidence Fusion & Scoring Agent (later, whoever picks it up) ──────────

class CategoryScore(BaseModel):
    category: str                   # e.g. "step_free_entrance"
    score: float                    # 0-100
    weight: float                   # from the weighted table in the project doc


class ScoringOutput(BaseModel):
    location_id: str
    overall_score: float            # 0-100
    confidence_score: float         # 0-100
    category_scores: list[CategoryScore]
    verified_features: list[str]
    barrier_features: list[str]
    unverified_features: list[str]
