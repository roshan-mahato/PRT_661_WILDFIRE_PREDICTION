"""
Builds feature_engineered rows from weather_daily rows.

Two kinds of series, handled differently because KBDI is a recursion:

  archive   One continuous series per cell. New days continue from the
            cell's latest archive feature row. If days are added BEFORE
            that row (history extended backwards), the cell's archive
            features are rebuilt from its first day.

  forecast  Each run starts from the cell's latest archive feature row and
            runs through that run's own days only. A new forecast therefore
            never changes an older one, and nothing earlier is touched.

The formulas live in Scripts/prediction_engine.py (shared with training).
"""

from typing import Callable, Iterable

import pandas as pd
from sqlalchemy import text

from backend.api.weather import utc_now
from backend.database import engine, write_lock
from Scripts.prediction_engine import KBDI_RESULT_COLS, compute_ffdi_max, engineer_features

FEATURE_COLS = KBDI_RESULT_COLS + ["emc", "ffdi", "rate_of_spread", "ffdi_max",
                                   "month_sin", "month_cos"]
FEATURE_INSERT_COLS = ["weather_id", "cell_id", "date", "source"] + FEATURE_COLS + ["computed_at"]
CELL_CHUNK = 250   # archive cells per batch (~250k rows for three years)

_WEATHER_SQL = """
SELECT w.weather_id, w.cell_id, w.date, w.temperature_2m, w.relative_humidity_2m,
       w.wind_speed_10m, w.precipitation, w.peak_fire_temperature_2m,
       w.peak_fire_relative_humidity_2m, w.peak_fire_wind_speed_10m
FROM weather_daily w
WHERE {where}
  AND NOT EXISTS (SELECT 1 FROM feature_engineered f WHERE f.weather_id = w.weather_id)
"""


def _in_list(ids: Iterable[int]) -> str:
    return ",".join(str(int(i)) for i in ids)


def latest_archive_state(conn, cell_ids: Iterable[int]) -> pd.DataFrame:
    """Each cell's latest archive feature row -- the state to continue from."""
    ids = _in_list(cell_ids)
    if not ids:
        return pd.DataFrame()
    return pd.read_sql(text(f"""
        SELECT f.cell_id, f.date, f.kbdi, f.days_since_rain, f.days_since_cell_start,
               f.last_rain_amt, f.in_rain_event, f.event_cumulative_rain
        FROM feature_engineered f
        JOIN (SELECT cell_id, MAX(date) AS date FROM feature_engineered
              WHERE source = 'archive' AND cell_id IN ({ids}) GROUP BY cell_id) m
          ON m.cell_id = f.cell_id AND m.date = f.date
        WHERE f.source = 'archive'
    """), conn)


def _save_features(feats: pd.DataFrame, source: str) -> int:
    if feats.empty:
        return 0
    out = feats.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["source"] = source
    out["computed_at"] = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")
    for c in ("kbdi_spinup_flag", "in_rain_event"):
        out[c] = out[c].astype(int)
    out = out[FEATURE_INSERT_COLS].astype(object)
    out = out.where(out.notna(), None)
    sql = (f"INSERT INTO feature_engineered ({', '.join(FEATURE_INSERT_COLS)}) "
           f"VALUES ({', '.join('?' * len(FEATURE_INSERT_COLS))})")
    with write_lock, engine.begin() as conn:
        conn.exec_driver_sql(sql, list(out.itertuples(index=False, name=None)))
    return len(out)


def build_archive_features(log: Callable[[str], None] = print) -> int:
    """
    Computes features for every archive weather row that has none yet.
    Returns the number of feature rows written.
    """
    with engine.connect() as conn:
        todo = pd.read_sql(text("""
            SELECT w.cell_id, MIN(w.date) AS first_new, COUNT(*) AS n_new
            FROM weather_daily w
            WHERE w.source = 'archive'
              AND NOT EXISTS (SELECT 1 FROM feature_engineered f WHERE f.weather_id = w.weather_id)
            GROUP BY w.cell_id
        """), conn)
        done = pd.read_sql(text("""
            SELECT cell_id, MAX(date) AS last_done FROM feature_engineered
            WHERE source = 'archive' GROUP BY cell_id
        """), conn)
    if todo.empty:
        log("Archive features: up to date.")
        fill_missing_ffdi_max(log)
        return 0

    todo = todo.merge(done, on="cell_id", how="left")
    backfill = todo[todo["last_done"].notna() & (todo["first_new"] <= todo["last_done"])]
    if not backfill.empty:
        # Days were added before the cell's latest feature row, so every later
        # row is out of date. Archive features are never referenced by a
        # prediction, so they can be rebuilt.
        log(f"Archive features: {len(backfill):,} cell(s) got earlier days -- rebuilding them.")
        with write_lock, engine.begin() as conn:
            conn.exec_driver_sql(
                f"DELETE FROM feature_engineered WHERE source = 'archive' "
                f"AND cell_id IN ({_in_list(backfill['cell_id'])})"
            )

    cont = todo[todo["last_done"].notna() & ~todo["cell_id"].isin(backfill["cell_id"])]
    gaps = cont[pd.to_datetime(cont["first_new"]) != pd.to_datetime(cont["last_done"]) + pd.Timedelta(days=1)]
    if not gaps.empty:
        log(f"Archive features: WARNING {len(gaps):,} cell(s) have missing days before their "
            f"new rows; the recursion treats the gap as rain-free days.")

    cells = todo["cell_id"].tolist()
    total = 0
    for i in range(0, len(cells), CELL_CHUNK):
        chunk = cells[i:i + CELL_CHUNK]
        with engine.connect() as conn:
            weather = pd.read_sql(text(_WEATHER_SQL.format(
                where=f"w.source = 'archive' AND w.cell_id IN ({_in_list(chunk)})")), conn)
            state = latest_archive_state(conn, chunk)
        feats = engineer_features(weather, state)
        total += _save_features(feats, "archive")
        log(f"  archive features: {min(i + CELL_CHUNK, len(cells)):,}/{len(cells):,} cells, "
            f"{total:,} rows")
    fill_missing_ffdi_max(log)
    return total


def fill_missing_ffdi_max(log: Callable[[str], None] = print) -> int:
    """
    Computes ffdi_max for feature rows whose weather row got its peak columns
    after the features were built (history re-fetched with the peaks). Only
    fills a NULL; the Drought Factor is already stored on the row.
    """
    with engine.connect() as conn:
        rows = pd.read_sql(text("""
            SELECT f.weather_id, f.drought_factor, w.peak_fire_temperature_2m,
                   w.peak_fire_relative_humidity_2m, w.peak_fire_wind_speed_10m
            FROM feature_engineered f JOIN weather_daily w ON w.weather_id = f.weather_id
            WHERE f.ffdi_max IS NULL AND w.peak_fire_temperature_2m IS NOT NULL
        """), conn)
    if rows.empty:
        return 0
    rows["ffdi_max"] = compute_ffdi_max(rows)
    with write_lock, engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE feature_engineered SET ffdi_max = ? WHERE weather_id = ?",
            list(zip(rows["ffdi_max"].astype(float).tolist(), rows["weather_id"].tolist())),
        )
    log(f"ffdi_max filled in for {len(rows):,} existing feature row(s).")
    return len(rows)


def build_run_features(run_id: int, log: Callable[[str], None] = print) -> int:
    """
    Computes features for one forecast run, continuing each cell from its
    latest archive feature row. Days the archive already covers are skipped:
    observed weather is better than a forecast of the same day.

    A cell with no archive history cold-starts; its first KBDI_SPINUP_DAYS
    rows carry kbdi_spinup_flag = True.

    Returns the number of feature rows written.
    """
    with engine.connect() as conn:
        weather = pd.read_sql(text(_WEATHER_SQL.format(where="w.run_id = :run")), conn,
                              params={"run": run_id})
        if weather.empty:
            return 0
        state = latest_archive_state(conn, weather["cell_id"].unique())

    last = weather["cell_id"].map(state.set_index("cell_id")["date"]) if not state.empty \
        else pd.Series(pd.NA, index=weather.index)
    cold = weather.loc[last.isna(), "cell_id"].nunique()
    weather = weather[last.isna() | (pd.to_datetime(weather["date"]) > pd.to_datetime(last))]
    if cold:
        log(f"Forecast features: {cold:,} cell(s) have no archive history -- cold start "
            f"(load history with Scripts/pipeline.py to fix).")
    return _save_features(engineer_features(weather, state), "forecast")
