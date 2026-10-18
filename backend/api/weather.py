"""
Writing and reading grid cells, fetch runs and daily weather.

The fetch scripts talk to Open-Meteo; this module gets their rows into the
database. Weather rows are append-only: a save never updates or replaces an
existing row, so features and predictions built from it stay reproducible.
"""

from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, insert, select, update

from backend.database import engine, write_lock
from backend.db_model.feature_engineered import FeatureEngineered
from backend.db_model.fetch_run import FetchRun
from backend.db_model.grid_cell import GridCell
from backend.db_model.prediction import Prediction
from backend.db_model.weather_daily import WeatherDaily
from Scripts.prediction_engine import DAILY_PEAK_COLS

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "wind_gusts_10m",
    "soil_moisture_0_to_7cm", "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
]
WEATHER_INSERT_COLS = (["run_id", "cell_id", "date", "source", "filled_from"]
                       + WEATHER_VARS + DAILY_PEAK_COLS)

# fetch_weather_history.py marks each day with where its values came from.
_FILLED_FROM = {"archive": None, "forecast": "forecast", "interpolated": "interpolated"}


def utc_now() -> datetime:
    """Timezone-naive UTC, the format every TIMESTAMP column uses."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cell_key(lat, lon) -> tuple:
    return round(float(lat), 3), round(float(lon), 3)


# ==============================================================================
# Grid cells
# ==============================================================================
def ensure_cells(cells: Iterable[tuple], is_land: Optional[bool] = None) -> dict:
    """
    Makes sure every (lat_round, lon_round) in `cells` has a grid_cell row and
    returns {(lat_round, lon_round): cell_id} for the whole grid. With
    `is_land`, also records the cells as land / ocean.
    """
    rows = list({_cell_key(a, o) for a, o in cells})
    with write_lock, engine.begin() as conn:
        if rows:
            # Raw SQL on purpose: SQLAlchemy's executemany() measured 1.3x slower.
            land = None if is_land is None else int(is_land)
            conn.exec_driver_sql(
                "INSERT OR IGNORE INTO grid_cell (lat_round, lon_round, is_land) VALUES (?, ?, ?)",
                [(a, o, land) for a, o in rows],
            )
            if is_land is not None:
                conn.exec_driver_sql(
                    "UPDATE grid_cell SET is_land = ? WHERE lat_round = ? AND lon_round = ? "
                    "AND (is_land IS NULL OR is_land != ?)",
                    [(land, a, o, land) for a, o in rows],
                )
        ids = conn.execute(select(GridCell.lat_round, GridCell.lon_round, GridCell.cell_id)).all()
    return {_cell_key(a, o): cid for a, o, cid in ids}


def seed_grid_cells(mask: dict) -> int:
    """Loads a land mask {(lat, lon): is_land} into grid_cell. Returns the cell count."""
    for flag in (True, False):
        ensure_cells([c for c, land in mask.items() if land is flag], is_land=flag)
    return len(mask)


# ==============================================================================
# Fetch runs
# ==============================================================================
def start_run(source: str, window_start=None, window_end=None, note: Optional[str] = None) -> int:
    """Records the start of a fetch and returns its run_id."""
    with write_lock, engine.begin() as conn:
        result = conn.execute(insert(FetchRun).values(
            source=source, issued_at=utc_now(), status="running",
            window_start=pd.Timestamp(window_start).date() if window_start else None,
            window_end=pd.Timestamp(window_end).date() if window_end else None,
            rows_written=0, note=note,
        ))
        return result.inserted_primary_key[0]


def finish_run(run_id: int, status: str) -> int:
    """Marks a run finished and stores how many weather rows it wrote."""
    with write_lock, engine.begin() as conn:
        n = conn.execute(
            select(func.count()).select_from(WeatherDaily).where(WeatherDaily.run_id == run_id)
        ).scalar_one()
        conn.execute(update(FetchRun).where(FetchRun.run_id == run_id).values(
            status=status, finished_at=utc_now(), rows_written=n,
        ))
    return n


def latest_forecast_run() -> Optional[int]:
    """The newest forecast run that wrote any weather, or None."""
    with engine.connect() as conn:
        return conn.execute(
            select(func.max(FetchRun.run_id))
            .where(FetchRun.source == "forecast", FetchRun.status.in_(("ok", "partial")),
                   FetchRun.rows_written > 0)
        ).scalar()


def list_runs(limit: int = 10) -> pd.DataFrame:
    query = (select(FetchRun.run_id, FetchRun.source, FetchRun.issued_at, FetchRun.finished_at,
                    FetchRun.status, FetchRun.window_start, FetchRun.window_end,
                    FetchRun.rows_written, FetchRun.note)
             .order_by(FetchRun.run_id.desc())
             .limit(limit))
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# ==============================================================================
# Weather
# ==============================================================================
def save_weather_daily(df: pd.DataFrame, run_id: int, source: str) -> int:
    """
    Appends daily weather rows for one fetch run.

    Args:
        df: lat_round, lon_round, a date column (`date` or `acq_date`) and the
            nine WEATHER_VARS. An optional `data_source` column (from
            fetch_weather_history.py) is stored as `filled_from`.
        run_id: the fetch run these rows belong to.
        source: 'archive' or 'forecast' (must match the run).

    DAILY_PEAK_COLS are stored when present (NULL otherwise).

    Archive rows for a cell-day that is already stored are not inserted again,
    so re-loading the same history is harmless. The one exception: if the
    stored row has no peak columns and the new one does, the peak columns are
    filled in (nothing already stored is changed).

    Returns:
        Number of rows inserted or filled in.
    """
    if df.empty:
        return 0
    date_col = "date" if "date" in df.columns else "acq_date"
    missing = [c for c in ["lat_round", "lon_round", date_col] + WEATHER_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot save weather rows, missing columns: {missing}")
    if df[WEATHER_VARS].isna().any().any():
        raise ValueError("Cannot save weather rows with missing values -- fill or drop them first.")

    ids = ensure_cells(zip(df["lat_round"], df["lon_round"]), is_land=True)
    dates = pd.to_datetime(df[date_col])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    out = pd.DataFrame({
        "run_id": run_id,
        "cell_id": [ids[_cell_key(a, o)] for a, o in zip(df["lat_round"], df["lon_round"])],
        "date": dates.dt.strftime("%Y-%m-%d").to_numpy(),
        "source": source,
        "filled_from": df["data_source"].map(_FILLED_FROM).to_numpy()
        if "data_source" in df.columns else None,
    })
    for v in WEATHER_VARS + DAILY_PEAK_COLS:
        out[v] = df[v].to_numpy(float) if v in df.columns else np.nan
    # object dtype turns numpy scalars into plain Python values sqlite3 can bind.
    out = out.astype(object).where(out.notna(), None)

    # Raw SQL on purpose: SQLAlchemy's executemany() measured 2x slower here
    # (millions of rows on a history load).
    sql = (f"INSERT INTO weather_daily ({', '.join(WEATHER_INSERT_COLS)}) "
           f"VALUES ({', '.join('?' * len(WEATHER_INSERT_COLS))})")
    if source == "archive":
        fill = ", ".join(f"{c} = excluded.{c}" for c in DAILY_PEAK_COLS)
        sql += (" ON CONFLICT (cell_id, date) WHERE source = 'archive' DO UPDATE SET "
                f"{fill} WHERE weather_daily.temperature_2m_max IS NULL "
                "AND excluded.temperature_2m_max IS NOT NULL")
    else:
        sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
    with write_lock, engine.begin() as conn:
        result = conn.exec_driver_sql(sql, list(out[WEATHER_INSERT_COLS].itertuples(index=False, name=None)))
    return max(result.rowcount, 0)


def archive_last_dates() -> dict:
    """
    {(lat_round, lon_round): last archive day stored in weather_daily} for
    every cell with any archive weather. Days are pd.Timestamp.
    """
    last = (select(WeatherDaily.cell_id, func.max(WeatherDaily.date).label("last_date"))
            .where(WeatherDaily.source == "archive")
            .group_by(WeatherDaily.cell_id)
            .subquery())
    query = (select(GridCell.lat_round, GridCell.lon_round, last.c.last_date)
             .join(last, last.c.cell_id == GridCell.cell_id))
    with engine.connect() as conn:
        rows = conn.execute(query).all()
    return {_cell_key(a, o): pd.Timestamp(d) for a, o, d in rows}


def archive_coverage(cell_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """
    Per cell: the last archive day that already has features. The forecast
    continues the KBDI recursion from that day.

    Returns:
        DataFrame [cell_id, lat_round, lon_round, last_date] (last_date as date).
    """
    last = (select(FeatureEngineered.cell_id,
                   func.max(FeatureEngineered.date).label("last_date"))
            .where(FeatureEngineered.source == "archive")
            .group_by(FeatureEngineered.cell_id)
            .subquery())
    query = (select(GridCell.cell_id, GridCell.lat_round, GridCell.lon_round, last.c.last_date)
             .join(last, last.c.cell_id == GridCell.cell_id))
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if cell_ids is not None:
        df = df[df["cell_id"].isin(list(cell_ids))]
    df["last_date"] = pd.to_datetime(df["last_date"]).dt.date
    return df.reset_index(drop=True)


def table_counts() -> dict:
    models = [GridCell, FetchRun, WeatherDaily, FeatureEngineered, Prediction]
    with engine.connect() as conn:
        return {m.__tablename__: conn.execute(select(func.count()).select_from(m)).scalar_one()
                for m in models}
