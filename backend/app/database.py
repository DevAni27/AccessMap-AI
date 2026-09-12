"""
Database connection setup.

`engine` is the single connection pool to Postgres, created once at import
time. `SessionLocal` is a factory for individual DB sessions — one per
request, per FastAPI convention. `get_db()` is a FastAPI "dependency":
routes that need DB access take `db: Session = Depends(get_db)` as an
argument, and FastAPI handles opening/closing the session around the
request automatically.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
