"""
FastAPI routes for fire-risk predictions.

This layer is deliberately thin: it validates input, reads predictions that
the pipeline has already stored (backend.api.fire_prediction), and shapes the
result into JSON. No fire logic and no model inference happen per request --
predictions are made once per forecast run by Scripts/fetch_weather_live.py
(or POST /predictions/refresh).
"""

from datetime import date as date_type, datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.api.features import build_run_features
from backend.api.fire_prediction import (
    VALID_RISK_LEVELS, get_current_predictions, predict_run,
)
from backend.api.weather import latest_forecast_run
from Scripts.prediction_engine import OUTPUT_COLS
from backend.schemas.prediction import FireRiskResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Australia bounding box -- same bounds the fetch grid is built over.
AUS_LAT_MIN, AUS_LAT_MAX = -44.0, -10.0
AUS_LON_MIN, AUS_LON_MAX = 112.0, 154.0

MAX_HOURS_BACK = 720  # 30 days

# Kept so existing clients can still send it; predictions now come from the
# latest forecast run, so it no longer changes the result.
HOURS_BACK_QUERY = Query(
    168, ge=1, le=MAX_HOURS_BACK,
    description="Ignored (kept for older clients). Predictions come from the latest forecast run.",
)


def _today_utc() -> date_type:
    return datetime.now(timezone.utc).date()


def _check_risk_level(risk_level: str | None):
    if risk_level is not None and risk_level not in VALID_RISK_LEVELS:
        raise HTTPException(
            status_code=422, detail=f"risk_level must be one of {sorted(VALID_RISK_LEVELS)}."
        )


def _to_response(df: pd.DataFrame, hours_back: int) -> FireRiskResponse:
    """Turns stored predictions into the JSON response envelope."""
    if df.empty:
        return FireRiskResponse(count=0, hours_back=hours_back, spinup_count=0, predictions=[])

    out = df[[c for c in OUTPUT_COLS if c in df.columns]].copy()
    # JSON has no NaN, and pandas/numpy scalar types don't serialise cleanly.
    out = out.replace({np.nan: None})
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
    hours_back: int = HOURS_BACK_QUERY,
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
    Current fire-risk predictions from today (UTC) onward for every grid cell,
    highest risk first.
    """
    _check_risk_level(risk_level)
    df = get_current_predictions(start_date=_today_utc(), risk_level=risk_level, limit=limit)
    return _to_response(df, hours_back)


@router.post(
    "/refresh",
    response_model=FireRiskResponse,
    summary="Recompute predictions for the latest forecast run",
)
def refresh_fire_risk(hours_back: int = HOURS_BACK_QUERY) -> FireRiskResponse:
    """
    Builds any missing features for the latest forecast run and predicts them
    with the current model. Use it after retraining the model; new weather
    comes from Scripts/fetch_weather_live.py, which also predicts.
    """
    run_id = latest_forecast_run()
    if run_id is None:
        raise HTTPException(
            status_code=409,
            detail="No forecast in the database yet -- run Scripts/fetch_weather_live.py first.",
        )
    try:
        build_run_features(run_id)
        predict_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file not available: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_response(get_current_predictions(start_date=_today_utc()), hours_back)


@router.get(
    "/stored",
    response_model=FireRiskResponse,
    summary="Read saved predictions",
)
def read_stored_predictions(
    latest_only: bool = Query(
        True,
        description="Only today (UTC) onward. Turn off to query past days with start_date/end_date.",
    ),
    start_date: date_type | None = Query(None, description="Only predictions from this date on."),
    end_date: date_type | None = Query(None, description="Only predictions up to this date."),
    risk_level: str | None = Query(None, description="Low, Moderate, High or Extreme."),
    limit: int | None = Query(None, ge=1, description="Cap how many rows are returned."),
) -> FireRiskResponse:
    """
    The current prediction for each (cell, day). Past days keep the last
    prediction made for them.
    """
    _check_risk_level(risk_level)
    if latest_only:
        start_date, end_date = _today_utc(), None
    df = get_current_predictions(
        start_date=start_date, end_date=end_date, risk_level=risk_level, limit=limit
    )
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
    hours_back: int = HOURS_BACK_QUERY,
) -> FireRiskResponse:
    """
    Current predictions from today (UTC) onward for the grid cell covering the
    given coordinates. Empty if that cell has no forecast yet.
    """
    df = get_current_predictions(
        start_date=_today_utc(), latitude=latitude, longitude=longitude
    )
    return _to_response(df, hours_back)
