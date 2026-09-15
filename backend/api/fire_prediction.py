"""
API for storing and querying generated fire-risk predictions in the
`fire_prediction` table.

Predictions are written here after each run so the dashboard can read them
back instantly, so there is an offline fallback when Open-Meteo or the model
is unavailable, and so past predictions can later be compared against what
FIRMS actually detected.
"""

import threading
from datetime import date as date_type, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import delete, func, select

from backend.database import SessionLocal, engine
from backend.db_model.firePrediction import FirePrediction

# SQLite allows concurrent readers but not concurrent writers.
_write_lock = threading.Lock()

PREDICTION_INSERT_COLS = [
    "lat_round", "lon_round", "acq_date",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_gusts_10m", "wind_direction_10m", "precipitation",
    "soil_moisture_0_to_7cm", "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "ffdi", "kbdi", "drought_factor", "kbdi_spinup_flag",
    "days_since_rain", "days_since_cell_start",
    "fire_probability", "fire_predicted", "risk_level",
    "predicted_at", "model_version",
]


def save_predictions(
    df: pd.DataFrame,
    model_version: Optional[str] = None,
    replace_existing: bool = True,
) -> int:
    """
    Saves prediction rows produced by backend.prediction_engine.predict_live().

    A re-run supersedes the previous prediction for the same (cell, day)
    rather than adding a second row -- otherwise the unique constraint would
    reject the write, and the dashboard would have no way to tell which of
    two rows is current.

    Args:
        df (pd.DataFrame): Predictions, using the engine's OUTPUT_COLS.
        model_version (str, optional): Identifier for the model that produced
                                       these rows, stored for audit.
        replace_existing (bool): If True (default), removes any existing
                                 prediction for each (cell, day) in `df`
                                 before inserting.

    Returns:
        int: Number of rows inserted.
    """
    if df.empty:
        return 0

    required = ["lat_round", "lon_round", "acq_date", "fire_probability", "risk_level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot save predictions, missing columns: {missing}")

    out = df.copy()
    out["acq_date"] = pd.to_datetime(out["acq_date"]).dt.date
    out["risk_level"] = out["risk_level"].astype(str)
    out["kbdi_spinup_flag"] = out["kbdi_spinup_flag"].astype(bool)
    out["fire_predicted"] = out["fire_predicted"].astype(int)
    out["predicted_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    out["model_version"] = model_version

    cols = [c for c in PREDICTION_INSERT_COLS if c in out.columns]
    out = out[cols]

    with _write_lock:
        if replace_existing:
            keys = out[["lat_round", "lon_round", "acq_date"]].drop_duplicates()
            with engine.begin() as conn:
                for row in keys.itertuples(index=False):
                    conn.execute(
                        delete(FirePrediction).where(
                            FirePrediction.lat_round == float(row.lat_round),
                            FirePrediction.lon_round == float(row.lon_round),
                            FirePrediction.acq_date == row.acq_date,
                        )
                    )
        out.to_sql(FirePrediction.__tablename__, engine, if_exists="append", index=False)

    return len(out)


def get_stored_predictions(
    start_date: Optional[date_type] = None,
    end_date: Optional[date_type] = None,
    risk_level: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reads stored predictions back, highest fire probability first.

    All filters are optional; with none supplied this returns every stored
    prediction. This is the read path the dashboard should use -- it does no
    model inference and no API calls, so it stays fast and works offline.

    Args:
        start_date (date, optional): Only predictions for this date onward.
        end_date (date, optional): Only predictions up to this date.
        risk_level (str, optional): One of Low, Moderate, High, Extreme.
        latitude (float, optional): Grid cell latitude (already grid-rounded).
        longitude (float, optional): Grid cell longitude (already grid-rounded).
        limit (int, optional): Cap the number of rows returned.

    Returns:
        pd.DataFrame: Stored predictions, or an empty DataFrame if none match.
    """
    stmt = select(FirePrediction)

    if start_date is not None:
        stmt = stmt.where(FirePrediction.acq_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(FirePrediction.acq_date <= end_date)
    if risk_level is not None:
        stmt = stmt.where(FirePrediction.risk_level == risk_level)
    if latitude is not None and longitude is not None:
        stmt = stmt.where(
            FirePrediction.lat_round == latitude,
            FirePrediction.lon_round == longitude,
        )

    stmt = stmt.order_by(FirePrediction.fire_probability.desc())
    if limit is not None:
        stmt = stmt.limit(limit)

    db = SessionLocal()
    try:
        return pd.read_sql(stmt, db.bind)
    finally:
        db.close()


def get_latest_predictions(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Returns predictions from the most recent run only.

    Filtering on the latest `predicted_at` avoids mixing a fresh run with
    leftovers from an older one, which would otherwise show contradictory
    risk levels for the same area on the map.

    Args:
        limit (int, optional): Cap the number of rows returned.

    Returns:
        pd.DataFrame: Predictions from the latest run, highest probability first.
    """
    db = SessionLocal()
    try:
        latest_run = db.query(func.max(FirePrediction.predicted_at)).scalar()
        if latest_run is None:
            return pd.DataFrame()

        # Predictions written by one run share a timestamp only to the
        # microsecond, so allow a small window rather than exact equality.
        window_start = latest_run - timedelta(minutes=5)
        stmt = (
            select(FirePrediction)
            .where(FirePrediction.predicted_at >= window_start)
            .order_by(FirePrediction.fire_probability.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return pd.read_sql(stmt, db.bind)
    finally:
        db.close()


def delete_stale_predictions(older_than_days: int = 30) -> int:
    """
    Removes predictions older than the cutoff, based on when they were made.

    Args:
        older_than_days (int): Age cutoff in days. Defaults to 30.

    Returns:
        int: Number of rows deleted.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).replace(tzinfo=None)
    with _write_lock:
        with engine.begin() as conn:
            result = conn.execute(
                delete(FirePrediction).where(FirePrediction.predicted_at < cutoff)
            )
    return result.rowcount or 0


def count_predictions() -> int:
    """Total number of rows currently in the fire_prediction table."""
    db = SessionLocal()
    try:
        return db.query(func.count(FirePrediction.id)).scalar() or 0
    finally:
        db.close()
