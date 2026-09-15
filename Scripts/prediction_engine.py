"""
prediction_engine.py
================================================================================
Pure feature-engineering and model-inference functions for live fire-risk
prediction. No DB access here -- this module only knows how to turn raw
hourly weather rows into a fire-risk prediction. Fetching the weather rows
is the job of backend/api/prediction.py.

This is the SAME logic as Scripts/predict_live_fire_risk.py (kept there as
a standalone CLI entry point for ad-hoc CSV runs); the formulas must stay
identical to what the model was trained on, so both call into this module
rather than each keeping their own copy.
================================================================================
"""
import json
import os
import threading
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "best_fire_model.joblib")
STATE_PATH = os.path.join(REPO_ROOT, "Scripts", "fire_state.json")

MEAN_ANNUAL_RAINFALL_MM = 700.0  # same constant as the training pipeline
FIXED_FUEL_LOAD_T_HA = 8.0
KBDI_SPINUP_DAYS = 30
GRID_SIZE_DEGREES = 0.5
DAILY_CACHE_MAX_DAYS = 60  # per-cell replay cache; live window is only 7 days

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit", "et0_fao_evapotranspiration",
]

CRITICAL_WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]

# Physical bounds from the training pipeline (wildfire_data_pipeline_full.ipynb,
# handle_outliers tier 1). Readings outside these are data errors, not extreme
# weather, and are dropped. Deliberately wide -- a genuine 48C day is real fire
# signal, not noise.
#
# The training data had these rows removed before the model ever saw them, so
# inference has to remove them too: a corrupt reading that reached the KBDI
# recursion would corrupt that cell's stored drought state for every later day,
# not just its own row.
PHYSICAL_BOUNDS = {
    "temperature_2m": (-10.0, 55.0),
    "relative_humidity_2m": (0.0, 100.0),
    "wind_speed_10m": (0.0, 200.0),
    "wind_gusts_10m": (0.0, 250.0),
    "precipitation": (0.0, 500.0),
}


# ==============================================================================
# Grid convention (matches the training pipeline's double-round pattern)
# ==============================================================================
def round_to_grid(value: float, grid_size: float = GRID_SIZE_DEGREES) -> float:
    """Snap a raw lat/lon to the 0.5-degree grid used across the pipeline."""
    return round(round(value / grid_size) * grid_size, 3)


# ==============================================================================
# Fire-danger formulas -- identical to wildfire_data_pipeline_full.ipynb
# ==============================================================================
def compute_emc(rh, temp_c):
    """Equilibrium Moisture Content (Simard, 1968)."""
    rh = np.asarray(rh, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    emc = np.empty_like(rh, dtype=float)
    low = rh < 10
    mid = (rh >= 10) & (rh < 50)
    high = rh >= 50
    emc[low] = 0.03229 + 0.281073 * rh[low] - 0.000578 * rh[low] * temp_c[low]
    emc[mid] = 2.22749 + 0.160107 * rh[mid] - 0.014784 * temp_c[mid]
    emc[high] = (21.0606 + 0.005565 * rh[high] ** 2
                 - 0.00035 * rh[high] * temp_c[high] - 0.483199 * rh[high])
    return emc


def compute_ffdi(temp_c, rh, wind_kmh, drought_factor):
    """McArthur Mark 5 FFDI (Noble et al., 1980)."""
    df_safe = np.maximum(drought_factor, 1e-3)
    return 2 * np.exp(-0.45 + 0.987 * np.log(df_safe)
                       - 0.0345 * rh + 0.0338 * temp_c + 0.0234 * wind_kmh)


def compute_ros(ffdi):
    """Simplified Rate of Spread (fixed fuel load)."""
    return 0.0012 * ffdi * FIXED_FUEL_LOAD_T_HA


def kbdi_df_step(state, date, temp_c, rain_mm):
    """
    Advances ONE grid cell's KBDI / Drought Factor recursion by one day,
    given its saved `state` (or a fresh cold-start state for a new cell).
    Mirrors compute_kbdi_and_df() from the training pipeline exactly, but
    operating one day at a time against persisted state.

    IDEMPOTENCY: the recursion must advance once per cell-day and no more.
    A day already computed is served from `daily_cache` instead of being
    advanced again -- without this, calling the prediction endpoint twice
    on the same data would inflate KBDI (and therefore FFDI) each time,
    silently making every cell look drier than it is.
    """
    if state is None:
        state = {
            "q_prev": 0.0, "n_dry_days": 0.0, "last_rain_amt": 0.0,
            "in_rain_event": False, "event_cumulative_rain": 0.0,
            "cell_start_date": date.strftime("%Y-%m-%d"), "last_date": None,
            "daily_cache": {},
        }
    state.setdefault("daily_cache", {})

    date_str = date.strftime("%Y-%m-%d")
    cached = state["daily_cache"].get(date_str)
    if cached is not None:
        return dict(cached), state

    last_date = pd.Timestamp(state["last_date"]) if state["last_date"] else None
    dt = 1.0 if last_date is None else max((date - last_date).days, 1)

    if rain_mm > 0:
        if not state["in_rain_event"]:
            net_rain = max(rain_mm - 5.0, 0.0)
            state["in_rain_event"] = True
            state["event_cumulative_rain"] = rain_mm
        else:
            net_rain = rain_mm
            state["event_cumulative_rain"] += rain_mm
    else:
        net_rain = 0.0
        if state["in_rain_event"]:
            state["last_rain_amt"] = state["event_cumulative_rain"]
            state["in_rain_event"] = False
            state["event_cumulative_rain"] = 0.0

    q_after_rain = max(state["q_prev"] - net_rain * 10, 0.0)
    dq = ((203.2 - q_after_rain) *
          (0.968 * np.exp(0.0875 * temp_c + 1.5552) - 8.30) * dt /
          (1 + 10.88 * np.exp(-0.001736 * MEAN_ANNUAL_RAINFALL_MM))) * 1e-3
    dq = max(dq, 0.0)
    q_new = min(q_after_rain + dq, 203.2)

    if rain_mm >= 2.0:
        state["n_dry_days"] = 0.0
    else:
        state["n_dry_days"] += dt

    N = max(state["n_dry_days"], 0.0)
    P = max(state["last_rain_amt"], 1.0)
    df_val = (0.191 * (q_new + 104) * (N + 1) ** 1.5) / (3.52 * (N + 1) ** 1.5 + P - 1)
    drought_factor = min(max(df_val, 0.0), 10.0)

    days_since_cell_start = (date - pd.Timestamp(state["cell_start_date"])).days

    state["q_prev"] = q_new
    state["last_date"] = date.strftime("%Y-%m-%d")

    result = {
        "kbdi": q_new,
        "drought_factor": drought_factor,
        "days_since_rain": state["n_dry_days"],
        "days_since_cell_start": days_since_cell_start,
        "kbdi_spinup_flag": days_since_cell_start < KBDI_SPINUP_DAYS,
    }

    # Cache this day so a repeat call replays it instead of advancing again.
    # Pruned to the most recent days so the state file stays small -- older
    # days are already baked into q_prev and are never re-requested in normal
    # operation (the live weather window is only 7 days).
    state["daily_cache"][date_str] = dict(result)
    if len(state["daily_cache"]) > DAILY_CACHE_MAX_DAYS:
        keep = sorted(state["daily_cache"])[-DAILY_CACHE_MAX_DAYS:]
        state["daily_cache"] = {d: state["daily_cache"][d] for d in keep}

    return result, state


# ==============================================================================
# State persistence (per grid-cell KBDI recursion memory across API calls)
# ==============================================================================
# fire_state.json is shared mutable state. FastAPI runs sync endpoints in a
# threadpool, so two requests can land here at once -- without this lock, one
# request's KBDI state could overwrite the other's.
_state_lock = threading.Lock()


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(all_state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(all_state, f, indent=2)


def cell_key(lat, lon) -> str:
    return f"{lat}_{lon}"


def drop_implausible_readings(raw_df: pd.DataFrame) -> tuple:
    """
    Drops physically impossible weather readings, matching tier 1 of
    handle_outliers() in the training notebook.

    NaN is deliberately NOT treated as out-of-bounds: pandas `.between()`
    returns False for NaN, so `~between()` would be True and would silently
    drop missing data here, before aggregate_to_daily()'s own missing-value
    handling (which fills precipitation with 0 rather than dropping it) ever
    runs. Requiring notna() lets missing values pass through untouched, as
    the notebook does.

    Returns:
        (filtered DataFrame, number of rows dropped)
    """
    if raw_df.empty:
        return raw_df, 0

    out_of_bounds = pd.Series(False, index=raw_df.index)
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col not in raw_df.columns:
            continue
        out_of_bounds |= raw_df[col].notna() & ~raw_df[col].between(low, high)

    n_dropped = int(out_of_bounds.sum())
    if n_dropped:
        return raw_df[~out_of_bounds].reset_index(drop=True), n_dropped
    return raw_df, 0


# ==============================================================================
# Step 1: raw hourly rows -> one row per (grid cell, day)
# ==============================================================================
def aggregate_to_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw hourly weather rows to one row per (lat_round, lon_round,
    acq_date). Precipitation is SUMMED (accumulation); everything else is
    averaged (snapshot variables) -- identical to the training pipeline's
    aggregate_to_daily() / build_daily_weather_series().

    Expects `raw_df` to already have `lat_round`, `lon_round`, and
    `datetime_utc` columns.
    """
    df = raw_df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["acq_date"] = df["datetime_utc"].dt.tz_localize(None).dt.floor("D")

    cols_present = [c for c in WEATHER_COLS if c in df.columns]
    agg_dict = {c: ("sum" if c == "precipitation" else "mean") for c in cols_present}
    daily = (
        df.groupby(["lat_round", "lon_round", "acq_date"], as_index=False)
        .agg(agg_dict)
        .sort_values(["lat_round", "lon_round", "acq_date"])
        .reset_index(drop=True)
    )

    critical = [c for c in CRITICAL_WEATHER_COLS if c in daily.columns]
    daily = daily[~daily[critical].isna().any(axis=1)].reset_index(drop=True)
    daily["precipitation"] = daily.get("precipitation", 0.0)
    daily["precipitation"] = daily["precipitation"].fillna(0.0)
    return daily


# ==============================================================================
# Step 2: daily weather -> fire-danger features (stateful KBDI recursion)
# ==============================================================================
def engineer_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Adds KBDI, Drought Factor, EMC, FFDI, Rate of Spread, and cyclical month
    features to a (grid cell, day) weather DataFrame. Loads and saves the
    persisted per-cell KBDI state (fire_state.json) so the recursion
    continues correctly across separate calls/days.
    """
    daily = daily.copy()
    # The whole read-recurse-write cycle is held under one lock: loading and
    # saving separately would let a concurrent request read stale state.
    with _state_lock:
        all_state = load_state()
        kbdi_rows = []
        for _, row in daily.iterrows():
            key = cell_key(row["lat_round"], row["lon_round"])
            cell_state = all_state.get(key)
            result, cell_state = kbdi_df_step(
                cell_state, row["acq_date"], row["temperature_2m"], row["precipitation"]
            )
            all_state[key] = cell_state
            kbdi_rows.append(result)
        save_state(all_state)

    kbdi_df = pd.DataFrame(kbdi_rows)
    for c in kbdi_df.columns:
        daily[c] = kbdi_df[c].values

    daily["emc"] = compute_emc(daily["relative_humidity_2m"], daily["temperature_2m"])
    daily["ffdi"] = compute_ffdi(daily["temperature_2m"], daily["relative_humidity_2m"],
                                  daily["wind_speed_10m"], daily["drought_factor"])
    daily["rate_of_spread"] = compute_ros(daily["ffdi"])

    month = pd.to_datetime(daily["acq_date"]).dt.month
    daily["month_sin"] = np.sin(2 * np.pi * month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * month / 12)
    return daily


# ==============================================================================
# Step 3: engineered features -> model prediction
# ==============================================================================
@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    """
    Loads the trained model bundle. Cached, because under FastAPI this would
    otherwise re-read (and re-deserialise) the joblib file on every single
    request -- the model doesn't change between requests, so load it once.
    """
    return joblib.load(MODEL_PATH)


def predict_from_features(daily: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Applies the trained model bundle to an already-feature-engineered daily DataFrame."""
    daily = daily.copy()
    bundle = bundle or load_model_bundle()
    model, features = bundle["model"], bundle["features"]
    threshold = bundle["threshold_f1_optimal"]

    missing_feats = [f for f in features if f not in daily.columns]
    if missing_feats:
        raise ValueError(f"Missing required model features after engineering: {missing_feats}")

    X = daily[features]
    if bundle.get("scaler") is not None:
        X = pd.DataFrame(bundle["scaler"].transform(X), columns=features, index=X.index)

    proba = model.predict_proba(X)[:, 1]
    daily["fire_probability"] = proba
    daily["fire_predicted"] = (proba >= threshold).astype(int)
    daily["risk_level"] = pd.cut(
        proba, bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["Low", "Moderate", "High", "Extreme"]
    )
    return daily


OUTPUT_COLS = [
    "lat_round", "lon_round", "acq_date", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "ffdi", "kbdi",
    "drought_factor", "kbdi_spinup_flag", "fire_probability",
    "fire_predicted", "risk_level",
]

# Extra columns kept for persistence but not returned in the API response.
# Together with OUTPUT_COLS these cover every feature the model consumes, so a
# saved row can be replayed through the model to reproduce its probability.
# They are excluded from the API response to keep the dashboard payload small
# -- the dashboard shows the headline conditions, not the full feature vector.
PERSIST_EXTRA_COLS = [
    "wind_gusts_10m", "wind_direction_10m", "precipitation",
    "soil_moisture_0_to_7cm", "vapour_pressure_deficit",
    "et0_fao_evapotranspiration", "days_since_rain", "days_since_cell_start",
]

PERSIST_COLS = OUTPUT_COLS + PERSIST_EXTRA_COLS


def predict_live(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end: raw hourly weather rows -> physical-bounds filter -> daily
    aggregation -> feature engineering -> prediction. This is the single entry
    point both the CLI script and the backend API should call.

    The step order mirrors the training notebook exactly (handle_outliers ->
    aggregate_to_daily -> engineer_features), so the same raw input produces
    the same features the model was trained on.
    """
    if raw_df.empty:
        return pd.DataFrame(columns=PERSIST_COLS)

    raw_df, n_dropped = drop_implausible_readings(raw_df)
    if n_dropped:
        print(f"Dropped {n_dropped:,} physically implausible weather reading(s).")
    if raw_df.empty:
        return pd.DataFrame(columns=PERSIST_COLS)

    daily = aggregate_to_daily(raw_df)
    if daily.empty:
        return pd.DataFrame(columns=PERSIST_COLS)

    daily = engineer_daily_features(daily)
    daily = predict_from_features(daily)
    cols = [c for c in PERSIST_COLS if c in daily.columns]
    return daily[cols].sort_values("fire_probability", ascending=False).reset_index(drop=True)
