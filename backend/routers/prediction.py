"""
FastAPI routes for live fire-risk prediction.

This layer is deliberately thin: it validates input, calls into
backend.api.prediction (DB access) and lets backend.prediction_engine do the
science, then shapes the result into JSON. No fire logic lives here.
"""

from datetime import date as date_type

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.api.fire_prediction import get_latest_predictions, get_stored_predictions
from backend.api.prediction import (
    get_live_fire_risk_by_location,
    get_live_fire_risk_predictions,
)
from Scripts.prediction_engine import OUTPUT_COLS
from backend.schemas.prediction import FireRiskResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Australia bounding box -- same bounds the fetch grid is built over.
AUS_LAT_MIN, AUS_LAT_MAX = -44.0, -10.0
AUS_LON_MIN, AUS_LON_MAX = 112.0, 154.0

MAX_HOURS_BACK = 720  # 30 days
VALID_RISK_LEVELS = {"Low", "Moderate", "High", "Extreme"}


def _to_response(df: pd.DataFrame, hours_back: int) -> FireRiskResponse:
    """
    Turns the pipeline's DataFrame into the JSON response envelope.

    The pipeline now carries more columns than the API returns (the extras
    exist so a saved prediction can be reproduced). They are dropped here
    explicitly rather than left for Pydantic to ignore, so the response
    payload is a deliberate choice and not a side effect of model config.
    """
    if df.empty:
        return FireRiskResponse(count=0, hours_back=hours_back, spinup_count=0, predictions=[])

    out = df[[c for c in OUTPUT_COLS if c in df.columns]].copy()
    # JSON has no NaN, and pandas/numpy scalar types don't serialise cleanly.
    out = out.replace({np.nan: None})
    out["acq_date"] = pd.to_datetime(out["acq_date"]).dt.date
    out["risk_level"] = out["risk_level"].astype(str)
    out["kbdi_spinup_flag"] = out["kbdi_spinup_flag"].astype(bool)
    out["fire_predicted"] = out["fire_predicted"].astype(int)

    records = out.to_dict(orient="records")
    return FireRiskResponse(
        count=len(records),
        hours_back=hours_back,
        spinup_count=int(out["kbdi_spinup_flag"].sum()),
        predictions=records,
    )


@router.get("", response_model=FireRiskResponse, summary="Fire risk for all grid cells")
def read_live_fire_risk(
    hours_back: int = Query(
        168,
        ge=1,
        le=MAX_HOURS_BACK,
        description="How many hours of weather data to pull. Defaults to 168 (7 days), "
                    "matching the forecast window fetched from Open-Meteo.",
    ),
    risk_level: str | None = Query(
        None,
        description="Optionally return only cells at this risk level "
                    "(Low, Moderate, High, Extreme).",
    ),
    limit: int | None = Query(
        None, ge=1, description="Optionally cap how many predictions are returned "
                                "(they are sorted by fire probability, highest first)."
    ),
) -> FireRiskResponse:
    """
    Next-day fire-risk predictions for every Australian grid cell that has
    live weather data in the lookback window, highest risk first.
    """
    try:
        df = get_live_fire_risk_predictions(hours_back=hours_back)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file not available: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if risk_level is not None and not df.empty:
        if risk_level not in VALID_RISK_LEVELS:
            raise HTTPException(
                status_code=422,
                detail=f"risk_level must be one of {sorted(VALID_RISK_LEVELS)}.",
            )
        df = df[df["risk_level"].astype(str) == risk_level]

    if limit is not None:
        df = df.head(limit)

    return _to_response(df, hours_back)


@router.post(
    "/refresh",
    response_model=FireRiskResponse,
    summary="Recompute fire risk and save it to the database",
)
def refresh_fire_risk(
    hours_back: int = Query(168, ge=1, le=MAX_HOURS_BACK),
    model_version: str | None = Query(
        None, description="Optional label for the model run, stored for audit."
    ),
) -> FireRiskResponse:
    """
    Runs the prediction pipeline over all grid cells and writes the results to
    the fire_prediction table, superseding any existing prediction for the
    same cell and day.

    This is the expensive call -- run it on a schedule after each weather
    fetch, then serve the dashboard from GET /predictions/stored.
    """
    try:
        df = get_live_fire_risk_predictions(
            hours_back=hours_back, persist=True, model_version=model_version
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file not available: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_response(df, hours_back)


@router.get(
    "/stored",
    response_model=FireRiskResponse,
    summary="Read saved predictions (fast, works offline)",
)
def read_stored_predictions(
    latest_only: bool = Query(
        True,
        description="Return only the most recent run. Turn off to query history.",
    ),
    start_date: date_type | None = Query(None, description="Only predictions from this date on."),
    end_date: date_type | None = Query(None, description="Only predictions up to this date."),
    risk_level: str | None = Query(None, description="Low, Moderate, High or Extreme."),
    limit: int | None = Query(None, ge=1, description="Cap how many rows are returned."),
) -> FireRiskResponse:
    """
    Reads predictions already saved in the database. No model inference and no
    Open-Meteo calls, so this is fast and still works if either is unavailable
    -- this is what the dashboard should call.
    """
    if risk_level is not None and risk_level not in VALID_RISK_LEVELS:
        raise HTTPException(
            status_code=422, detail=f"risk_level must be one of {sorted(VALID_RISK_LEVELS)}."
        )

    if latest_only:
        df = get_latest_predictions(limit=limit)
        if risk_level is not None and not df.empty:
            df = df[df["risk_level"] == risk_level]
    else:
        df = get_stored_predictions(
            start_date=start_date,
            end_date=end_date,
            risk_level=risk_level,
            limit=limit,
        )

    # No trimming needed here -- _to_response() selects the response columns.
    return _to_response(df, hours_back=0)

@router.get(
    "/location",
    response_model=FireRiskResponse,
    summary="Fire risk for a single location",
)
def read_live_fire_risk_by_location(
    latitude: float = Query(
        ...,
        ge=AUS_LAT_MIN,
        le=AUS_LAT_MAX,
        description="Latitude (WGS84). Snapped to the nearest 0.5-degree grid cell.",
    ),
    longitude: float = Query(
        ...,
        ge=AUS_LON_MIN,
        le=AUS_LON_MAX,
        description="Longitude (WGS84). Snapped to the nearest 0.5-degree grid cell.",
    ),
    hours_back: int = Query(168, ge=1, le=MAX_HOURS_BACK),
) -> FireRiskResponse:
    """
    Next-day fire-risk predictions for the single grid cell covering the
    given coordinates. Returns an empty list if that cell has no live
    weather data yet -- run the Open-Meteo live fetch first.
    """
    try:
        df = get_live_fire_risk_by_location(
            latitude=latitude, longitude=longitude, hours_back=hours_back
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file not available: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_response(df, hours_back)
