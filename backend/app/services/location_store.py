"""Finding, resolving and storing locations (shared by several routers)."""

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.contracts import LocationResolutionInput, LocationResolutionOutput
from app.agents.location_resolution import resolve_location
from app.models import Location
from app.services.errors import ServiceError


def to_output(location: Location) -> LocationResolutionOutput:
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


def find_by_osm_ref(db: Session, osm_ref: str | None) -> Location | None:
    if osm_ref is None:
        return None
    return db.query(Location).filter(Location.osm_ref == osm_ref).first()


# The same real place can exist in OSM as several objects (a hospital drawn
# as a building AND as a point, 100 m apart), each with its own OSM id.
DUPLICATE_RADIUS_METERS = 150


def find_nearby_duplicate(db: Session, resolved: LocationResolutionOutput) -> Location | None:
    """A stored place with the same name within DUPLICATE_RADIUS_METERS: the
    same real-world place reached through a different OSM object."""
    point = func.ST_GeogFromText(f"SRID=4326;POINT({resolved.longitude} {resolved.latitude})")
    return (
        db.query(Location)
        .filter(
            func.lower(Location.name) == resolved.name.lower(),
            func.ST_DWithin(Location.geom, point, DUPLICATE_RADIUS_METERS),
        )
        .first()
    )


def get_location(db: Session, location_id: str) -> Location:
    location = db.get(Location, location_id)
    if location is None:
        raise ServiceError(404, f"No location with id {location_id!r}")
    return location


def resolve_and_store(db: Session, query: str) -> tuple[LocationResolutionOutput, bool]:
    """Resolve a place name and store it. Returns (location, created).

    Idempotent: a place that is already stored (however the user phrased
    the search) comes back as the existing record with created=False."""
    try:
        resolved = resolve_location(LocationResolutionInput(query=query))
    except ValueError as e:
        raise ServiceError(404, str(e))
    except Exception as e:
        raise ServiceError(502, f"Location resolution failed: {e}")

    existing = find_by_osm_ref(db, resolved.osm_ref) or find_nearby_duplicate(db, resolved)
    if existing is not None:
        return to_output(existing), False

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
        existing = find_by_osm_ref(db, resolved.osm_ref)
        if existing is None:
            raise  # a different integrity problem — don't hide it
        return to_output(existing), False

    return resolved, True
