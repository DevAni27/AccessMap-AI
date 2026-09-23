"""
Locations router — the actual API surface for location resolution.

This is what turns the standalone agent function into a real feature:
POST a place name, the agent resolves it, we save the result to the
database, and return it. GET lists everything saved so far.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Location
from app.agents.contracts import LocationResolutionInput
from app.agents.location_resolution import resolve_location

router = APIRouter(prefix="/locations", tags=["locations"])


@router.post("")
def create_location(input: LocationResolutionInput, db: Session = Depends(get_db)):
    try:
        result = resolve_location(input)
    except ValueError as e:
        # Nominatim found nothing for this query
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # network error, Nominatim down, etc.
        raise HTTPException(status_code=502, detail=f"Location resolution failed: {e}")

    db_location = Location(
        id=result.location_id,
        name=result.name,
        category=result.category,
        latitude=result.latitude,
        longitude=result.longitude,
        address=result.address,
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)

    return result


@router.get("")
def list_locations(db: Session = Depends(get_db)):
    return db.query(Location).all()