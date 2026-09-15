"""
API for writing and querying live (forecast) Open-Meteo weather data in the
`weather_live` table.

The fetch script (Scripts/fetch_openmeteo_live.py) is responsible for talking
to Open-Meteo; this module is responsible for getting those rows into the
database correctly. Keeping the two apart means the fetch script doesn't carry
its own DB engine or schema knowledge, and the prediction API reads from
exactly the same database the fetcher writes to.
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import delete, func, select

from backend.database import SessionLocal, engine
from backend.db_model.openmeteo import OpenMeteo

# SQLite tolerates concurrent readers but not concurrent writers. The fetch
# script runs a thread pool, so every write goes through one lock.
_write_lock = threading.Lock()

WEATHER_LIVE_INSERT_COLS = [
    "lat_round", "lon_round", "datetime_utc", "fetched_at",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "wind_gusts_10m",
    "soil_moisture_0_to_7cm", "cape", "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
]


def _normalise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts timestamp columns to timezone-naive UTC before insert.

    The `weather_live` columns are plain TIMESTAMP (no timezone), and SQLite
    has no native timestamp type at all -- it stores whatever string it is
    handed. Writing timezone-aware values risks storing an offset suffix that
    would not compare correctly against the timezone-naive values SQLAlchemy
    generates when the prediction API filters by date. Stripping to naive UTC
    here keeps both sides in one format.
    """
    df = df.copy()
    for col in ("datetime_utc", "fetched_at"):
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col], utc=True)
        df[col] = series.dt.tz_localize(None)
    return df


def save_weather_live(df: pd.DataFrame, replace_existing: bool = True) -> int:
    """
    Saves live forecast rows for one or more grid cells.

    Forecasts are re-fetched regularly and each fetch supersedes the last for
    that cell, so the default is delete-then-insert per cell rather than a
    blind append -- an append would both duplicate rows and violate the
    (lat_round, lon_round, datetime_utc) unique constraint on the second run.

    Args:
        df (pd.DataFrame): Forecast rows. Must contain lat_round, lon_round,
                           datetime_utc and the Open-Meteo weather variables.
        replace_existing (bool): If True (default), removes any existing rows
                                 for each grid cell present in `df` before
                                 inserting.

    Returns:
        int: Number of rows inserted.
    """
    if df.empty:
        return 0

    missing = [c for c in ("lat_round", "lon_round", "datetime_utc") if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot save weather_live rows, missing columns: {missing}")

    out = _normalise_timestamps(df)
    if "fetched_at" not in out.columns:
        out["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    # Keep only real table columns, in table order -- an unexpected extra
    # column would otherwise make the insert fail.
    cols = [c for c in WEATHER_LIVE_INSERT_COLS if c in out.columns]
    out = out[cols]

    cells = out[["lat_round", "lon_round"]].drop_duplicates().itertuples(index=False)

    with _write_lock:
        if replace_existing:
            with engine.begin() as conn:
                for lat, lon in cells:
                    conn.execute(
                        delete(OpenMeteo).where(
                            OpenMeteo.lat_round == float(lat),
                            OpenMeteo.lon_round == float(lon),
                        )
                    )
        out.to_sql(OpenMeteo.__tablename__, engine, if_exists="append", index=False)

    return len(out)


def delete_stale_weather_live(older_than_hours: int = 168) -> int:
    """
    Removes forecast rows whose `fetched_at` is older than the cutoff.

    Without this the table grows every fetch cycle, and stale forecasts would
    still be picked up by the prediction API's lookback window.

    Args:
        older_than_hours (int): Age cutoff in hours. Defaults to 168 (7 days).

    Returns:
        int: Number of rows deleted.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).replace(tzinfo=None)
    with _write_lock:
        with engine.begin() as conn:
            result = conn.execute(
                delete(OpenMeteo).where(OpenMeteo.fetched_at < cutoff)
            )
    return result.rowcount or 0


def get_weather_live_coverage() -> pd.DataFrame:
    """
    Summarises what live weather data is currently stored, one row per grid
    cell. Useful for checking a fetch actually landed, and for spotting cells
    the prediction API will silently skip because they have no data.

    Returns:
        pd.DataFrame: columns [lat_round, lon_round, row_count,
                      first_datetime_utc, last_datetime_utc, last_fetched_at]
    """
    stmt = (
        select(
            OpenMeteo.lat_round,
            OpenMeteo.lon_round,
            func.count().label("row_count"),
            func.min(OpenMeteo.datetime_utc).label("first_datetime_utc"),
            func.max(OpenMeteo.datetime_utc).label("last_datetime_utc"),
            func.max(OpenMeteo.fetched_at).label("last_fetched_at"),
        )
        .group_by(OpenMeteo.lat_round, OpenMeteo.lon_round)
        .order_by(OpenMeteo.lat_round, OpenMeteo.lon_round)
    )
    db = SessionLocal()
    try:
        return pd.read_sql(stmt, db.bind)
    finally:
        db.close()


def count_weather_live_rows() -> int:
    """Total number of rows currently in the weather_live table."""
    db = SessionLocal()
    try:
        return db.query(func.count(OpenMeteo.id)).scalar() or 0
    finally:
        db.close()
