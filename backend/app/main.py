"""
AccessMap AI — FastAPI entrypoint.

For now this just proves the server runs. As we build, every new
feature gets wired in here via an `app.include_router(...)` call —
see routers/locations.py once that exists (Day 8-9).
"""

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db

app = FastAPI(
    title="AccessMap AI",
    description="Agentic RAG-based accessibility assessment and inclusive place recommendation system",
    version="0.1.0",
)


@app.get("/")
def read_root():
    """Basic liveness check — hit this in a browser to confirm the server is up."""
    return {"status": "ok", "message": "AccessMap AI backend is running"}


@app.get("/health")
def health_check():
    """Separate from '/' on purpose: real deployments (Render, etc.) usually
    ping a dedicated /health endpoint to check the service is alive, rather
    than the root route, which might later do more work."""
    return {"status": "healthy"}


@app.get("/db-check")
def db_check(db: Session = Depends(get_db)):
    """Day 2 checkpoint: proves FastAPI can actually reach the Neon database
    through SQLAlchemy, not just that the server itself is running."""
    result = db.execute(text("SELECT 1")).scalar()
    return {"db_connected": result == 1}
