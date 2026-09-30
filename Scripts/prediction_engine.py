"""
prediction_engine.py
================================================================================
Pure feature-engineering and model-inference functions for fire-risk
prediction. No DB access here -- this module only knows how to turn daily
weather rows into features and features into a prediction. Reading and
writing the weather_daily / feature_engineered / prediction tables is the job
of backend/api/features.py and backend/api/fire_prediction.py.

The formulas must stay identical to what the model was trained on, so the
training pipeline (build_labeled_dataset.py) imports them from here too.
================================================================================
"""
import hashlib
import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "best_fire_model.joblib")
MODEL_METADATA_PATH = os.path.join(REPO_ROOT, "models", "best_fire_model_metadata.json")

MEAN_ANNUAL_RAINFALL_MM = 700.0  # same constant as the training pipeline
FIXED_FUEL_LOAD_T_HA = 8.0
KBDI_SPINUP_DAYS = 30
GRID_SIZE_DEGREES = 0.5
DAILY_CACHE_MAX_DAYS = 60  # kbdi_df_step replay cache (training pipeline only)

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit", "et0_fao_evapotranspiration",
]

CRITICAL_WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]

# Daily extremes of the hourly series. Fire weather peaks in the afternoon and
# a daily mean hides it (a 38C / 8% RH afternoon can average 28C / 30%). They
# are stored NEXT TO the means, which the current model was trained on.
DAILY_EXTREME_AGG = {
    "temperature_2m_max": ("temperature_2m", "max"),
    "relative_humidity_2m_min": ("relative_humidity_2m", "min"),
    "wind_speed_10m_max": ("wind_speed_10m", "max"),
    "wind_gusts_10m_max": ("wind_gusts_10m", "max"),
    "vapour_pressure_deficit_max": ("vapour_pressure_deficit", "max"),
    "et0_fao_evapotranspiration_sum": ("et0_fao_evapotranspiration", "sum"),   # mm/day
}
# Conditions at ONE hour: the hour with the highest FFDI. The separate maxima
# above can come from different hours (hottest at 3pm, windiest at 8am);
# FFDI needs temperature, humidity and wind that happened together.
PEAK_HOUR_COLS = [
    "peak_fire_hour_utc", "peak_fire_temperature_2m",
    "peak_fire_relative_humidity_2m", "peak_fire_wind_speed_10m",
    # A plain mean of angles is wrong (350 and 10 average to 180), so the
    # direction is taken at the windiest hour instead.
    "wind_direction_at_max_wind",
]
DAILY_PEAK_COLS = list(DAILY_EXTREME_AGG) + PEAK_HOUR_COLS

# Model inputs built from the columns above (used once a model is retrained
# on them; peak_fire_hour_utc is left out because it mostly encodes longitude).
PEAK_FEATURES = list(DAILY_EXTREME_AGG) + ["wind_direction_at_max_wind", "ffdi_max"]

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


def ffdi_weather_term(temp_c, rh, wind_kmh):
    """
    The hourly part of FFDI. The Drought Factor is one value per day, so the
    hour with the largest term is the hour with the largest FFDI.
    """
    return -0.0345 * rh + 0.0338 * temp_c + 0.0234 * wind_kmh


def daily_peaks(hourly: pd.DataFrame, keys: list) -> pd.DataFrame:
    """
    Hourly rows -> DAILY_PEAK_COLS per (keys..., UTC day).

    Args:
        hourly: a `datetime_utc` column, the `keys` columns and the hourly
                weather variables.
        keys:   columns identifying a cell (may be empty for one cell).

    Returns:
        keys + acq_date (timezone-naive midnight) + DAILY_PEAK_COLS. A day
        with a missing hour gets values from the hours it has; callers that
        need complete days drop those days themselves.
    """
    h = hourly.reset_index(drop=True).copy()
    t = pd.to_datetime(h["datetime_utc"], utc=True)
    h["acq_date"] = t.dt.tz_localize(None).dt.floor("D")
    h["_hour"] = t.dt.hour
    by = keys + ["acq_date"]
    g = h.groupby(by)

    out = pd.DataFrame({
        name: (g[col].sum(min_count=1) if fn == "sum" else g[col].agg(fn))
        for name, (col, fn) in DAILY_EXTREME_AGG.items()
    })

    # Row of the peak FFDI hour and of the windiest hour, per day. NaN terms
    # are pushed to -inf so an all-NaN day still returns a row (with NaNs).
    term = ffdi_weather_term(h["temperature_2m"], h["relative_humidity_2m"], h["wind_speed_10m"])
    at_peak = h.loc[term.fillna(-np.inf).groupby([h[k] for k in by]).idxmax().to_numpy()]
    at_wind = h.loc[h["wind_speed_10m"].fillna(-np.inf).groupby([h[k] for k in by]).idxmax().to_numpy()]
    out["peak_fire_hour_utc"] = at_peak["_hour"].to_numpy()
    out["peak_fire_temperature_2m"] = at_peak["temperature_2m"].to_numpy()
    out["peak_fire_relative_humidity_2m"] = at_peak["relative_humidity_2m"].to_numpy()
    out["peak_fire_wind_speed_10m"] = at_peak["wind_speed_10m"].to_numpy()
    out["wind_direction_at_max_wind"] = at_wind["wind_direction_10m"].to_numpy()
    return out.reset_index()


def compute_ffdi_max(daily: pd.DataFrame) -> np.ndarray:
    """
    FFDI at the day's peak fire-weather hour. Always >= the FFDI of the daily
    means (the term is linear, so its maximum is at least its mean). NaN
    where the peak-hour columns are missing (rows stored before they existed).
    """
    cols = ["peak_fire_temperature_2m", "peak_fire_relative_humidity_2m", "peak_fire_wind_speed_10m"]
    if not all(c in daily.columns for c in cols):
        return np.full(len(daily), np.nan)
    return compute_ffdi(daily[cols[0]].to_numpy(float), daily[cols[1]].to_numpy(float),
                        daily[cols[2]].to_numpy(float), daily["drought_factor"].to_numpy(float))


def kbdi_df_step(state, date, temp_c, rain_mm):
    """
    Advances ONE grid cell's KBDI / Drought Factor recursion by one day,
    given its saved `state` (or a fresh cold-start state for a new cell).
    Mirrors compute_kbdi_and_df() from the training pipeline exactly, but
    operating one day at a time against persisted state.

    This is the reference implementation, used by the training pipeline.
    run_kbdi() below is the same recursion advanced for every cell at once;
    tests/test_prediction_engine.py checks the two agree.

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
# The same recursion for every cell at once
# ==============================================================================
# A feature row carries everything needed to continue the recursion the next
# day, so the recursion state is simply "the cell's latest feature row":
#   q_prev -> kbdi, n_dry_days -> days_since_rain, last_date -> date,
#   cell_start_date -> date - days_since_cell_start, plus the three columns
# below that only exist to carry the rain-event state forward.
KBDI_EVENT_STATE_COLS = ["last_rain_amt", "in_rain_event", "event_cumulative_rain"]
KBDI_RESULT_COLS = [
    "kbdi", "drought_factor", "days_since_rain", "days_since_cell_start",
    "kbdi_spinup_flag",
] + KBDI_EVENT_STATE_COLS


def _day_number(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).to_numpy("datetime64[D]").astype(np.int64)


def run_kbdi(daily: pd.DataFrame, start_state: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Runs the KBDI / Drought Factor recursion of kbdi_df_step() over many
    cells at once: the loop is over days, and each day advances every cell
    that has a row that day as one numpy operation.

    Args:
        daily: one row per (cell_id, date) with temperature_2m and
               precipitation. Any order; a cell may have a gap (the gap is
               treated as rain-free days, as kbdi_df_step does).
        start_state: optional, one row per cell_id -- that cell's latest
               feature row (date, kbdi, days_since_rain, days_since_cell_start
               and KBDI_EVENT_STATE_COLS). Cells without one cold-start.
               Every row of `daily` must be after its cell's state date.

    Returns:
        `daily` (same index and order) with KBDI_RESULT_COLS added.
    """
    out = daily.copy()
    if out.empty:
        for c in KBDI_RESULT_COLS:
            out[c] = pd.Series(dtype=float)
        return out

    cells = pd.Index(out["cell_id"].unique())
    n = len(cells)
    q = np.zeros(n)
    n_dry = np.zeros(n)
    last_rain = np.zeros(n)
    in_event = np.zeros(n, dtype=bool)
    event_cum = np.zeros(n)
    start_day = np.full(n, -1, dtype=np.int64)   # -1 = no state yet (cold start)
    last_day = np.full(n, -1, dtype=np.int64)

    if start_state is not None and not start_state.empty:
        s = start_state[start_state["cell_id"].isin(cells)]
        p = cells.get_indexer(s["cell_id"])
        s_day = _day_number(s["date"])
        q[p] = s["kbdi"].to_numpy(float)
        n_dry[p] = s["days_since_rain"].to_numpy(float)
        last_rain[p] = s["last_rain_amt"].to_numpy(float)
        in_event[p] = s["in_rain_event"].to_numpy(bool)
        event_cum[p] = s["event_cumulative_rain"].to_numpy(float)
        last_day[p] = s_day
        start_day[p] = s_day - s["days_since_cell_start"].to_numpy(np.int64)

    day = _day_number(out["date"])
    pos = cells.get_indexer(out["cell_id"])
    temp = out["temperature_2m"].to_numpy(float)
    rain = np.nan_to_num(out["precipitation"].to_numpy(float), nan=0.0)

    res = {c: np.empty(len(out)) for c in KBDI_RESULT_COLS}
    denom = 1 + 10.88 * np.exp(-0.001736 * MEAN_ANNUAL_RAINFALL_MM)

    order = np.argsort(day, kind="stable")
    days, first = np.unique(day[order], return_index=True)
    bounds = list(first) + [len(order)]
    for i, d in enumerate(days):
        rows = order[bounds[i]:bounds[i + 1]]
        p = pos[rows]
        if len(np.unique(p)) != len(p):
            raise ValueError(f"run_kbdi: a cell has two rows on day {np.datetime64(int(d), 'D')}")
        known = last_day[p] >= 0
        if np.any(known & (last_day[p] >= d)):
            raise ValueError("run_kbdi: a row is not after its cell's start_state date")

        r, t = rain[rows], temp[rows]
        cold = start_day[p] < 0
        start_day[p[cold]] = d
        dt = np.where(known, np.maximum(d - last_day[p], 1), 1).astype(float)

        # Rain-event bookkeeping, exactly as kbdi_df_step.
        wet = r > 0
        was_in = in_event[p]
        new_event = wet & ~was_in
        end_event = ~wet & was_in
        net_rain = np.where(new_event, np.maximum(r - 5.0, 0.0), np.where(wet, r, 0.0))
        last_rain[p] = np.where(end_event, event_cum[p], last_rain[p])
        event_cum[p] = np.where(new_event, r,
                                np.where(wet, event_cum[p] + r,
                                         np.where(end_event, 0.0, event_cum[p])))
        in_event[p] = wet | (was_in & ~end_event)

        q_after = np.maximum(q[p] - net_rain * 10, 0.0)
        dq = ((203.2 - q_after) * (0.968 * np.exp(0.0875 * t + 1.5552) - 8.30) * dt
              / denom) * 1e-3
        q_new = np.minimum(q_after + np.maximum(dq, 0.0), 203.2)

        n_dry[p] = np.where(r >= 2.0, 0.0, n_dry[p] + dt)
        N = np.maximum(n_dry[p], 0.0)
        P = np.maximum(last_rain[p], 1.0)
        df_val = (0.191 * (q_new + 104) * (N + 1) ** 1.5) / (3.52 * (N + 1) ** 1.5 + P - 1)

        q[p] = q_new
        last_day[p] = d
        since_start = d - start_day[p]

        res["kbdi"][rows] = q_new
        res["drought_factor"][rows] = np.clip(df_val, 0.0, 10.0)
        res["days_since_rain"][rows] = n_dry[p]
        res["days_since_cell_start"][rows] = since_start
        res["kbdi_spinup_flag"][rows] = since_start < KBDI_SPINUP_DAYS
        res["last_rain_amt"][rows] = last_rain[p]
        res["in_rain_event"][rows] = in_event[p]
        res["event_cumulative_rain"][rows] = event_cum[p]

    for c, v in res.items():
        out[c] = v
    for c in ("kbdi_spinup_flag", "in_rain_event"):
        out[c] = out[c].astype(bool)
    out["days_since_cell_start"] = out["days_since_cell_start"].astype(np.int64)
    return out


def add_fire_danger_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Adds the same-day features (EMC, FFDI, Rate of Spread, cyclical month)."""
    daily = daily.copy()
    rh = daily["relative_humidity_2m"].to_numpy(float)
    temp = daily["temperature_2m"].to_numpy(float)
    daily["emc"] = compute_emc(rh, temp)
    daily["ffdi"] = compute_ffdi(temp, rh, daily["wind_speed_10m"].to_numpy(float),
                                 daily["drought_factor"].to_numpy(float))
    daily["rate_of_spread"] = compute_ros(daily["ffdi"])
    daily["ffdi_max"] = compute_ffdi_max(daily)
    month = pd.to_datetime(daily["date"]).dt.month
    daily["month_sin"] = np.sin(2 * np.pi * month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * month / 12)
    return daily


def engineer_features(daily: pd.DataFrame, start_state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily weather -> every model feature. See run_kbdi() for the arguments."""
    return add_fire_danger_features(run_kbdi(daily, start_state))


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
    aggregate_to_daily() / build_daily_weather_series(). With all hourly
    variables present, DAILY_PEAK_COLS are added too (see daily_peaks()).

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

    if all(c in df.columns for c in WEATHER_COLS):
        daily = daily.merge(daily_peaks(df, ["lat_round", "lon_round"]),
                            on=["lat_round", "lon_round", "acq_date"], how="left")

    critical = [c for c in CRITICAL_WEATHER_COLS if c in daily.columns]
    daily = daily[~daily[critical].isna().any(axis=1)].reset_index(drop=True)
    daily["precipitation"] = daily.get("precipitation", 0.0)
    daily["precipitation"] = daily["precipitation"].fillna(0.0)
    return daily


# ==============================================================================
# Step 2: engineered features -> model prediction
# ==============================================================================
@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    """
    Loads the trained model bundle. Cached, because under FastAPI this would
    otherwise re-read (and re-deserialise) the joblib file on every single
    request -- the model doesn't change between requests, so load it once.
    """
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def model_version() -> str:
    """
    Identifies the model file: its training time plus a short hash of the
    file, e.g. "2026-09-16T11:15:18-3f2a9c1d". Stored on every prediction, so
    a retrained model never overwrites the old model's predictions.
    """
    with open(MODEL_PATH, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()[:8]
    trained_at = "unknown"
    if os.path.exists(MODEL_METADATA_PATH):
        with open(MODEL_METADATA_PATH) as f:
            trained_at = json.load(f).get("trained_at", trained_at)
    return f"{trained_at}-{digest}"


def predict_from_features(
    daily: pd.DataFrame, bundle: dict | None = None
) -> pd.DataFrame:
    """Applies the trained model bundle to an already-feature-engineered daily DataFrame."""
    daily = daily.copy()
    bundle = bundle or load_model_bundle()
    model, features = bundle["model"], bundle["features"]
    threshold = bundle.get("threshold_f1_optimal", 0.5)

    missing_feats = [f for f in features if f not in daily.columns]
    if missing_feats:
        raise ValueError(
            f"Missing required model features after engineering: {missing_feats}"
        )

    # A column that is NULL for every row (e.g. peak columns on rows stored
    # before they existed) comes back from SQL as object dtype, which the
    # model rejects; as float it is NaN, which LightGBM handles.
    X = daily[features].apply(pd.to_numeric, errors="coerce").astype(float)
    if bundle.get("scaler") is not None:
        X = pd.DataFrame(
            bundle["scaler"].transform(X), columns=features, index=X.index
        )

    proba = model.predict_proba(X)[:, 1]
    daily["fire_probability"] = proba
    daily["fire_predicted"] = (proba >= threshold).astype(int)

    # Dynamic risk binning scaled around the optimal decision threshold
    # so anything the model flags as a fire (proba >= threshold) is High or Extreme.
    risk_bins = [
        -np.inf,
        threshold * 0.5,  # Low -> Moderate boundary
        threshold,        # Moderate -> High boundary (Decision Threshold)
        threshold + (1.0 - threshold) * 0.5,  # High -> Extreme boundary
        np.inf,
    ]

    daily["risk_level"] = pd.cut(
        proba,
        bins=risk_bins,
        labels=["Low", "Moderate", "High", "Extreme"],
        right=False,
    )

    return daily


# Columns the prediction endpoints return, one row per (grid cell, day).
OUTPUT_COLS = [
    "lat_round", "lon_round", "acq_date", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "ffdi", "kbdi",
    "drought_factor", "kbdi_spinup_flag", "fire_probability",
    "fire_predicted", "risk_level",
]
