"""
API for generating live fire-risk predictions from stored weather data,
returned as pandas DataFrames.

This module only handles DB access (pulling weather rows for the requested
grid cell(s)/window); the actual feature engineering and model inference
live in backend.prediction_engine, so the formulas have one home instead of
being duplicated here.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import func, select, tuple_

from backend.database import SessionLocal
from backend.api.fire_prediction import save_predictions
from backend.db_model.openmeteo import OpenMeteo
from Scripts.prediction_engine import predict_live, round_to_grid, PERSIST_COLS

WEATHER_LIVE_COLS = [
    "lat_round", "lon_round", "datetime_utc", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "precipitation", "wind_gusts_10m", "soil_moisture_0_to_7cm", "cape",
    "vapour_pressure_deficit", "et0_fao_evapotranspiration",
]


def _fetch_weather_live(
    hours_back: int,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> pd.DataFrame:
    """
    Pulls recent WeatherLive rows from the DB as a plain DataFrame, using
    SQLAlchemy Core select() + pd.read_sql() rather than looping over ORM
    objects -- matching this project's established DB query pattern.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    # Open-Meteo's land-surface model reports soil moisture only over land, so a
    # sea cell reads exactly 0 for every hour of the window. The fetch grid is a
    # plain lat/lon rectangle over Australia's bounding box (roughly half of it
    # ocean), and without this exclusion those cells reach the model as ordinary
    # weather and come back as extreme fire risk over open water.
    ocean_cells = (
        select(OpenMeteo.lat_round, OpenMeteo.lon_round)
        .where(OpenMeteo.datetime_utc >= since)
        .group_by(OpenMeteo.lat_round, OpenMeteo.lon_round)
        .having(func.max(OpenMeteo.soil_moisture_0_to_7cm) <= 0)
    )

    stmt = select(*(getattr(OpenMeteo, c) for c in WEATHER_LIVE_COLS)).where(
        OpenMeteo.datetime_utc >= since,
        tuple_(OpenMeteo.lat_round, OpenMeteo.lon_round).not_in(ocean_cells),
    )
    if latitude is not None and longitude is not None:
        stmt = stmt.where(
            OpenMeteo.lat_round == round_to_grid(latitude),
            OpenMeteo.lon_round == round_to_grid(longitude),
        )

    db = SessionLocal()
    try:
        df = pd.read_sql(stmt, db.bind)
    finally:
        db.close()
    return df if not df.empty else pd.DataFrame(columns=WEATHER_LIVE_COLS)


def get_live_fire_risk_predictions(
    hours_back: int = 168, persist: bool = False, model_version: str | None = None
) -> pd.DataFrame:
    """
    Generates next-day fire-risk predictions for every grid cell that has
    live weather data within the lookback window.

    Args:
        hours_back (int): How far back to pull hourly weather rows from.
                           Defaults to 168 (7 days), matching the forecast
                           window fetched by Scripts/fetch_openmeteo_live.py.
        persist (bool): If True, saves the results to the fire_prediction
                        table. Off by default so a plain read never writes.
        model_version (str, optional): Stored alongside persisted rows for audit.

    Returns:
        pd.DataFrame: one row per (grid cell, day), columns:
                      [lat_round, lon_round, acq_date, temperature_2m,
                       relative_humidity_2m, wind_speed_10m, ffdi, kbdi,
                       drought_factor, kbdi_spinup_flag, fire_probability,
                       fire_predicted, risk_level]
                      sorted by fire_probability descending. Empty (with the
                      same columns) if no live weather data is available.
    """
    raw_df = _fetch_weather_live(hours_back)
    if raw_df.empty:
        return pd.DataFrame(columns=PERSIST_COLS)

    result = predict_live(raw_df)
    if persist and not result.empty:
        save_predictions(result, model_version=model_version)
    return result


def get_live_fire_risk_by_location(
    latitude: float,
    longitude: float,
    hours_back: int = 168,
    persist: bool = False,
    model_version: str | None = None,
) -> pd.DataFrame:
    """
    Generates next-day fire-risk predictions for the single grid cell
    covering the given coordinates.

    Args:
        latitude (float): Raw latitude (WGS84) -- snapped to the nearest
                           0.5-degree grid cell before querying.
        longitude (float): Raw longitude (WGS84) -- snapped the same way.
        hours_back (int): How far back to pull hourly weather rows from.
                           Defaults to 168 (7 days).
        persist (bool): If True, saves the results to the fire_prediction table.
        model_version (str, optional): Stored alongside persisted rows for audit.

    Returns:
        pd.DataFrame: same columns as get_live_fire_risk_predictions(),
                      restricted to the requested grid cell. Empty (with
                      the same columns) if that cell has no live weather
                      data within the window.
    """
    raw_df = _fetch_weather_live(hours_back, latitude=latitude, longitude=longitude)
    if raw_df.empty:
        return pd.DataFrame(columns=PERSIST_COLS)

    result = predict_live(raw_df)
    if persist and not result.empty:
        save_predictions(result, model_version=model_version)
    return result
