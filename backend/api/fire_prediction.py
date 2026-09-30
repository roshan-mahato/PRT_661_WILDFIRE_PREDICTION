"""
Generating, storing and reading fire-risk predictions (`prediction` table).

Predictions are computed once per forecast run (Scripts/fetch_weather_live.py
or POST /predictions/refresh) and the dashboard reads them back with a single
indexed query -- no feature engineering or model inference per request.
"""

from datetime import date as date_type
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text

from backend.api.weather import WEATHER_VARS, utc_now
from backend.database import engine, write_lock
from Scripts.prediction_engine import (
    DAILY_PEAK_COLS, load_model_bundle, model_version, predict_from_features, round_to_grid,
)

PREDICTION_INSERT_COLS = [
    "weather_id", "run_id", "cell_id", "date", "lead_days", "fire_probability",
    "fire_predicted", "risk_level", "threshold", "model_version", "predicted_at",
    "is_current",
]
VALID_RISK_LEVELS = ("Low", "Moderate", "High", "Extreme")


def predict_run(run_id: int, log: Callable[[str], None] = print) -> int:
    """
    Predicts every feature row of a forecast run with the current model, then
    makes those predictions the current ones for their cell-days.

    Safe to call again: rows this model already predicted are skipped, and a
    retrained model (new model_version) adds its own rows next to the old
    ones instead of overwriting them.

    Returns:
        Number of prediction rows written.
    """
    version = model_version()
    bundle = load_model_bundle()
    weather_cols = ", ".join(f"w.{v}" for v in WEATHER_VARS + DAILY_PEAK_COLS)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT f.*, {weather_cols}, g.lat_round, g.lon_round, w.run_id, r.issued_at
            FROM feature_engineered f
            JOIN weather_daily w ON w.weather_id = f.weather_id
            JOIN grid_cell g ON g.cell_id = f.cell_id
            JOIN fetch_run r ON r.run_id = w.run_id
            WHERE w.run_id = :run
              AND NOT EXISTS (SELECT 1 FROM prediction p
                              WHERE p.weather_id = f.weather_id AND p.model_version = :mv)
        """), conn, params={"run": run_id, "mv": version})

    n = 0
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        pred = predict_from_features(df, bundle)
        pred["lead_days"] = (pred["date"] - pd.to_datetime(pred["issued_at"]).dt.normalize()).dt.days
        pred["date"] = pred["date"].dt.strftime("%Y-%m-%d")
        pred["risk_level"] = pred["risk_level"].astype(str)
        pred["threshold"] = float(bundle.get("threshold_f1_optimal", 0.5))
        pred["model_version"] = version
        pred["predicted_at"] = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")
        pred["is_current"] = 0
        rows = list(pred[PREDICTION_INSERT_COLS].astype(object).itertuples(index=False, name=None))
        n = len(rows)
    else:
        rows = []

    sql = (f"INSERT INTO prediction ({', '.join(PREDICTION_INSERT_COLS)}) "
           f"VALUES ({', '.join('?' * len(PREDICTION_INSERT_COLS))})")
    with write_lock, engine.begin() as conn:
        if rows:
            conn.exec_driver_sql(sql, rows)
        # Hand "current" over to this run for the cell-days it covers -- but
        # never take it from a NEWER run (e.g. when re-predicting an old run).
        conn.execute(text("""
            UPDATE prediction SET is_current = 0
            WHERE is_current = 1 AND run_id <= :run
              AND EXISTS (SELECT 1 FROM prediction n
                          WHERE n.run_id = :run AND n.model_version = :mv
                            AND n.cell_id = prediction.cell_id AND n.date = prediction.date)
        """), {"run": run_id, "mv": version})
        conn.execute(text("""
            UPDATE prediction SET is_current = 1
            WHERE run_id = :run AND model_version = :mv
              AND NOT EXISTS (SELECT 1 FROM prediction c
                              WHERE c.is_current = 1 AND c.cell_id = prediction.cell_id
                                AND c.date = prediction.date)
        """), {"run": run_id, "mv": version})
    log(f"Predictions: {n:,} written for run {run_id} (model {version}).")
    return n


def get_current_predictions(
    start_date: Optional[date_type] = None,
    end_date: Optional[date_type] = None,
    risk_level: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    The current prediction for each (cell, day), highest probability first,
    with the headline weather and fire-danger values joined back in.

    All filters are optional. latitude/longitude are raw coordinates and are
    snapped to the grid.
    """
    where, params = ["p.is_current = 1"], {}
    if start_date is not None:
        where.append("p.date >= :start")
        params["start"] = str(start_date)
    if end_date is not None:
        where.append("p.date <= :end")
        params["end"] = str(end_date)
    if risk_level is not None:
        where.append("p.risk_level = :risk")
        params["risk"] = risk_level
    if latitude is not None and longitude is not None:
        where.append("g.lat_round = :lat AND g.lon_round = :lon")
        params["lat"], params["lon"] = round_to_grid(latitude), round_to_grid(longitude)
    sql = f"""
        SELECT g.lat_round, g.lon_round, p.date AS acq_date,
               w.temperature_2m, w.relative_humidity_2m, w.wind_speed_10m,
               f.ffdi, f.kbdi, f.drought_factor, f.kbdi_spinup_flag,
               p.fire_probability, p.fire_predicted, p.risk_level,
               p.lead_days, p.run_id, p.model_version, p.predicted_at
        FROM prediction p
        JOIN feature_engineered f ON f.weather_id = p.weather_id
        JOIN weather_daily w ON w.weather_id = p.weather_id
        JOIN grid_cell g ON g.cell_id = p.cell_id
        WHERE {' AND '.join(where)}
        ORDER BY p.fire_probability DESC
    """
    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    df["kbdi_spinup_flag"] = df["kbdi_spinup_flag"].astype(bool)
    return df


def prune_forecasts(keep_days: int = 90, log: Callable[[str], None] = print) -> dict:
    """
    Deletes forecast runs issued more than `keep_days` ago, except the rows
    that still back a current prediction. Archive rows are never touched.

    Superseded predictions are what lead-time accuracy is measured on, so
    only prune after you have evaluated them.
    """
    # Same format as the stored issued_at (with microseconds), so the string
    # comparison is exact.
    cutoff = (pd.Timestamp(utc_now()) - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d %H:%M:%S.%f")
    old = "SELECT run_id FROM fetch_run WHERE source = 'forecast' AND issued_at < :cutoff"
    with write_lock, engine.begin() as conn:
        n_pred = conn.execute(text(
            f"DELETE FROM prediction WHERE is_current = 0 AND run_id IN ({old})"
        ), {"cutoff": cutoff}).rowcount
        n_feat = conn.execute(text(f"""
            DELETE FROM feature_engineered WHERE source = 'forecast'
              AND weather_id IN (SELECT weather_id FROM weather_daily WHERE run_id IN ({old}))
              AND NOT EXISTS (SELECT 1 FROM prediction p WHERE p.weather_id = feature_engineered.weather_id)
        """), {"cutoff": cutoff}).rowcount
        n_weather = conn.execute(text(f"""
            DELETE FROM weather_daily WHERE source = 'forecast' AND run_id IN ({old})
              AND NOT EXISTS (SELECT 1 FROM feature_engineered f WHERE f.weather_id = weather_daily.weather_id)
        """), {"cutoff": cutoff}).rowcount
        n_runs = conn.execute(text(f"""
            DELETE FROM fetch_run WHERE run_id IN ({old})
              AND NOT EXISTS (SELECT 1 FROM weather_daily w WHERE w.run_id = fetch_run.run_id)
              AND NOT EXISTS (SELECT 1 FROM prediction p WHERE p.run_id = fetch_run.run_id)
        """), {"cutoff": cutoff}).rowcount
    counts = {"prediction": n_pred, "feature_engineered": n_feat,
              "weather_daily": n_weather, "fetch_run": n_runs}
    log(f"Pruned forecasts issued before {cutoff}: {counts}")
    return counts
