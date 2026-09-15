"""
FastAPI application for the Real-Time Wildfire Risk Tracking and Prediction
System.

Run it with:
    uv run uvicorn backend.app:app --reload

Interactive docs are then at http://127.0.0.1:8000/docs
"""

import os
import sys
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.database import engine
from Scripts.prediction_engine import load_model_bundle
from backend.routers import prediction as prediction_router
from backend.schemas.prediction import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the model once at startup rather than on the first request, so the
    first user doesn't pay the joblib deserialisation cost. A failure here is
    logged, not fatal -- /health will report it and the endpoints will return
    503, which is more useful than the whole service refusing to boot.
    """
    try:
        load_model_bundle()
        print("Model bundle loaded.")
    except Exception as exc:  # noqa: BLE001 - startup should not hard-fail
        print(f"WARNING: could not load model bundle at startup: {exc}")
    yield


app = FastAPI(
    title="Wildfire Risk Prediction API",
    description=(
        "Live fire-risk predictions for Australia. Combines the McArthur Mark 5 "
        "FFDI physics baseline with a trained classifier to predict next-day fire "
        "danger for each 0.5-degree grid cell."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# The Streamlit dashboard runs on a different port, so it needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(prediction_router.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Reports whether the model and database are both usable."""
    model_loaded = True
    db_ok = True
    problems = []

    try:
        load_model_bundle()
    except Exception as exc:  # noqa: BLE001
        model_loaded = False
        problems.append(f"model: {exc}")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        db_ok = False
        problems.append(f"database: {exc}")

    healthy = model_loaded and db_ok
    return HealthResponse(
        status="ok" if healthy else "degraded",
        model_loaded=model_loaded,
        database_reachable=db_ok,
        detail="; ".join(problems) if problems else None,
    )


@app.get("/", tags=["system"])
def root():
    return {
        "service": "Wildfire Risk Prediction API",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/predictions",
            "/predictions/location",
            "/predictions/refresh",
            "/predictions/stored",
        ],
    }
