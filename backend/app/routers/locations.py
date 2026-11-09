"""
Locations router — the API surface for location resolution and web
evidence retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.contracts import (
    LocationResolutionInput,
    LocationResolutionOutput,
    WebEvidenceInput,
    WebEvidenceOutput,
)
from app.agents.location_resolution import resolve_location
from app.agents.web_evidence import fetch_web_evidence
from app.database import get_db
from app.models import Evidence, Location
from app.schemas import EvidenceRead

router = APIRouter(prefix="/locations", tags=["locations"])


def _to_output(location: Location) -> LocationResolutionOutput:
    """DB row -> the same shape the Location Resolution agent returns."""
    return LocationResolutionOutput(
        location_id=location.id,
        name=location.name,
        category=location.category,
        latitude=location.latitude,
        longitude=location.longitude,
        address=location.address,
        osm_ref=location.osm_ref,
    )


def _find_by_osm_ref(db: Session, osm_ref: str | None) -> Location | None:
    if osm_ref is None:
        return None
    return db.query(Location).filter(Location.osm_ref == osm_ref).first()


def _get_location_or_404(db: Session, location_id: str) -> Location:
    location = db.get(Location, location_id)
    if location is None:
        raise HTTPException(status_code=404, detail=f"No location with id {location_id!r}")
    return location


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
    try:
        resolved = resolve_location(input)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Location resolution failed: {e}")

    existing = _find_by_osm_ref(db, resolved.osm_ref)
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return _to_output(existing)

    db.add(Location(
        id=resolved.location_id,
        osm_ref=resolved.osm_ref,
        name=resolved.name,
        category=resolved.category,
        latitude=resolved.latitude,
        longitude=resolved.longitude,
        address=resolved.address,
    ))
    try:
        db.commit()
    except IntegrityError:
        # Two requests for the same new place raced past the check above;
        # the UNIQUE constraint on osm_ref let exactly one of them in.
        db.rollback()
        existing = _find_by_osm_ref(db, resolved.osm_ref)
        if existing is None:
            raise  # a different integrity problem — don't hide it
        response.status_code = status.HTTP_200_OK
        return _to_output(existing)

    return resolved


@router.get("", response_model=list[LocationResolutionOutput])
def list_locations(db: Session = Depends(get_db)):
    locations = db.query(Location).order_by(Location.created_at).all()
    return [_to_output(loc) for loc in locations]


@router.get("/{location_id}", response_model=LocationResolutionOutput)
def get_location(location_id: str, db: Session = Depends(get_db)):
    return _to_output(_get_location_or_404(db, location_id))


@router.post(
    "/{location_id}/evidence",
    response_model=WebEvidenceOutput,
    status_code=status.HTTP_201_CREATED,
)
def create_evidence(location_id: str, db: Session = Depends(get_db)):
    location = _get_location_or_404(db, location_id)

    agent_input = WebEvidenceInput(
        location_id=location.id,
        location_name=location.name,
        latitude=location.latitude,
        longitude=location.longitude,
    )

    try:
        result = fetch_web_evidence(agent_input)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Web evidence retrieval failed: {e}")

    db.add(Evidence(
        location_id=result.location_id,
        osm_tags=[tag.model_dump() for tag in result.osm_tags],
        image_urls=result.image_urls,
        text_snippets=result.text_snippets,
    ))
    db.commit()

    return result


@router.get("/{location_id}/evidence", response_model=list[EvidenceRead])
def list_evidence(location_id: str, db: Session = Depends(get_db)):
    _get_location_or_404(db, location_id)
    return (
        db.query(Evidence)
        .filter(Evidence.location_id == location_id)
        .order_by(Evidence.created_at.desc(), Evidence.id.desc())
        .all()
    )