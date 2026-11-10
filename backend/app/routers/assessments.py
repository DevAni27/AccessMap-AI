"""Assessment endpoints — the main thing the product does."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import AssessmentResponse, AssessRequest
from app.services import assessment as assessment_service
from app.services.location_store import get_location, resolve_and_store

router = APIRouter(tags=["assessments"])


@router.post("/assess", response_model=AssessmentResponse)
def assess_by_name(body: AssessRequest, db: Session = Depends(get_db)):
    """One call from a place name to a full accessibility report: resolves
    and stores the place if new, collects evidence, analyses, scores.
    A complete earlier assessment is reused unless `refresh` is true."""
    resolved, _ = resolve_and_store(db, body.query)
    location = get_location(db, resolved.location_id)
    return assessment_service.assess_location(db, location, refresh=body.refresh)


@router.post("/locations/{location_id}/assess", response_model=AssessmentResponse)
def assess_location(location_id: str, refresh: bool = False, db: Session = Depends(get_db)):
    return assessment_service.assess_location(db, get_location(db, location_id), refresh=refresh)


@router.get("/locations/{location_id}/assessment", response_model=AssessmentResponse)
def get_latest_assessment(location_id: str, db: Session = Depends(get_db)):
    """The most recent stored assessment (complete or not), without running anything."""
    return assessment_service.stored_assessment(db, location_id)
