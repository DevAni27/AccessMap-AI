"""
Locations router — the API surface for location resolution and web
evidence retrieval. Thin: the work lives in app/services.
"""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.agents.contracts import LocationResolutionInput, LocationResolutionOutput, WebEvidenceOutput
from app.database import get_db
from app.models import Evidence, Location
from app.schemas import EvidenceRead
from app.services import evidence as evidence_service
from app.services.location_store import get_location, resolve_and_store, to_output

router = APIRouter(prefix="/locations", tags=["locations"])


@router.post(
    "",
    response_model=LocationResolutionOutput,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "This place was already stored; the existing record is returned."}},
)
def create_location(
    input: LocationResolutionInput,
    response: Response,
    db: Session = Depends(get_db),
):
    """Resolve a place name and store it. Idempotent: searching for a place
    that is already stored (however it was phrased) returns the existing
    record with 200 instead of creating a duplicate."""
    location, created = resolve_and_store(db, input.query)
    if not created:
        response.status_code = status.HTTP_200_OK
    return location


@router.get("", response_model=list[LocationResolutionOutput])
def list_locations(db: Session = Depends(get_db)):
    locations = db.query(Location).order_by(Location.created_at).all()
    return [to_output(loc) for loc in locations]


@router.get("/{location_id}", response_model=LocationResolutionOutput)
def get_location_by_id(location_id: str, db: Session = Depends(get_db)):
    return to_output(get_location(db, location_id))


@router.post(
    "/{location_id}/evidence",
    response_model=WebEvidenceOutput,
    status_code=status.HTTP_201_CREATED,
)
def create_evidence(location_id: str, db: Session = Depends(get_db)):
    location = get_location(db, location_id)
    return evidence_service.to_web_evidence(evidence_service.collect_evidence(db, location))


@router.get("/{location_id}/evidence", response_model=list[EvidenceRead])
def list_evidence(location_id: str, db: Session = Depends(get_db)):
    get_location(db, location_id)
    return (
        db.query(Evidence)
        .filter(Evidence.location_id == location_id)
        .order_by(Evidence.created_at.desc(), Evidence.id.desc())
        .all()
    )
