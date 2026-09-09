"""
================================================================================
predict_live_fire_risk.py
================================================================================
Takes RAW, LIVE hourly weather rows (the same columns your fetch pipeline
produces: temperature_2m, relative_humidity_2m, wind_speed_10m,
wind_direction_10m, datetime_utc, precipitation, wind_gusts_10m,
soil_moisture_0_to_7cm, cape, vapour_pressure_deficit,
et0_fao_evapotranspiration, lat_round, lon_round, fetched_at) and produces a
fire-risk prediction for EVERY (grid cell, day) present in the data, using
the exact same feature-engineering logic as wildfire_data_pipeline_full.ipynb
(EMC / KBDI / Drought Factor / FFDI / Rate of Spread) and the trained
best_fire_model.joblib (LightGBM).

WHY DAILY, NOT HOURLY:
The model was trained on ONE ROW PER (grid cell, day) -- hourly weather is
aggregated to daily first (mean for snapshot variables, SUM for
precipitation) before any fire-danger index is computed. So "a prediction
for each row of live data" means one prediction per cell-day, built from
all the hourly rows that day -- not one prediction per hourly reading.
Feeding raw hourly rows straight into the model would silently mismatch
the distribution it was trained on.

WHY STATE PERSISTS ACROSS RUNS:
KBDI and Drought Factor are RECURSIVE -- each day's value depends on the
previous day's value for that same grid cell. A brand-new cell with no
history has to start its KBDI clock at 0 (exactly like a new monitoring
station would), so the FIRST prediction for a never-before-seen cell is
necessarily a cold-start estimate flagged via kbdi_spinup_flag-equivalent
logic. Every subsequent day, this script loads yesterday's saved state
(fire_state.json) and continues the recursion properly, so predictions
get more accurate over the cell's first ~30 days of monitoring, matching
KBDI_SPINUP_DAYS in the training pipeline.

USAGE
-----
    python predict_live_fire_risk.py new_weather.csv

    # or import and call directly:
    from predict_live_fire_risk import predict_live
    result_df = predict_live(new_weather_df)
================================================================================
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "./models/best_fire_model.joblib"
STATE_PATH = os.path.join(os.path.dirname(__file__), "fire_state.json")

MEAN_ANNUAL_RAINFALL_MM = 700.0   # same constant as the training pipeline
FIXED_FUEL_LOAD_T_HA = 8.0
KBDI_SPINUP_DAYS = 30

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit", "et0_fao_evapotranspiration",
]


# ==============================================================================
# Exact same fire-danger formulas as wildfire_data_pipeline_full.ipynb
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
    operating one day at a time against persisted state instead of a full
    historical DataFrame.
    """
    if state is None:
        state = {
            "q_prev": 0.0, "n_dry_days": 0.0, "last_rain_amt": 0.0,
            "in_rain_event": False, "event_cumulative_rain": 0.0,
            "cell_start_date": date.strftime("%Y-%m-%d"), "last_date": None,
        }

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

    return {
        "kbdi": q_new,
        "drought_factor": drought_factor,
        "days_since_rain": state["n_dry_days"],
        "days_since_cell_start": days_since_cell_start,
        "kbdi_spinup_flag": days_since_cell_start < KBDI_SPINUP_DAYS,
    }, state


# ==============================================================================
# State persistence (per grid-cell KBDI recursion memory across script runs)
# ==============================================================================
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(all_state):
    with open(STATE_PATH, "w") as f:
        json.dump(all_state, f, indent=2)


def cell_key(lat, lon):
    return f"{lat}_{lon}"


# ==============================================================================
# Main pipeline: raw hourly rows -> daily aggregation -> features -> predict
# ==============================================================================
def predict_live(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["acq_date"] = df["datetime_utc"].dt.tz_localize(None).dt.floor("D")

    # ---- 1. Aggregate hourly -> one row per (cell, day) ----
    # precipitation SUMMED (accumulation), everything else averaged (snapshots)
    # -- identical logic to aggregate_to_daily() / build_daily_weather_series()
    cols_present = [c for c in WEATHER_COLS if c in df.columns]
    agg_dict = {c: ("sum" if c == "precipitation" else "mean") for c in cols_present}
    daily = (
        df.groupby(["lat_round", "lon_round", "acq_date"], as_index=False)
        .agg(agg_dict)
        .sort_values(["lat_round", "lon_round", "acq_date"])
        .reset_index(drop=True)
    )
    print(f"Aggregated {len(df)} hourly rows -> {len(daily)} cell-day rows.")

    # ---- 2. Missing-value handling (same rules as engineer_features()) ----
    critical = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    n_missing = daily[critical].isna().any(axis=1).sum()
    if n_missing:
        print(f"WARNING: dropping {n_missing} cell-days missing temp/humidity/wind.")
        daily = daily[~daily[critical].isna().any(axis=1)].reset_index(drop=True)
    daily["precipitation"] = daily.get("precipitation", 0.0)
    daily["precipitation"] = daily["precipitation"].fillna(0.0)

    # ---- 3. KBDI / Drought Factor recursion (stateful, per grid cell) ----
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

    n_spinup = int(daily["kbdi_spinup_flag"].sum())
    if n_spinup:
        print(f"NOTE: {n_spinup} cell-day(s) are within the first {KBDI_SPINUP_DAYS} "
              f"days of that grid cell's monitoring history (kbdi_spinup_flag=True) "
              f"-- KBDI/Drought Factor/FFDI for these are cold-start estimates and "
              f"should be treated as lower-confidence until the cell has more history.")

    # ---- 4. Instantaneous fire-danger indices (EMC, FFDI, ROS) ----
    daily["emc"] = compute_emc(daily["relative_humidity_2m"], daily["temperature_2m"])
    daily["ffdi"] = compute_ffdi(daily["temperature_2m"], daily["relative_humidity_2m"],
                                  daily["wind_speed_10m"], daily["drought_factor"])
    daily["rate_of_spread"] = compute_ros(daily["ffdi"])

    # ---- 5. Cyclical month features (as used at training time) ----
    month = pd.to_datetime(daily["acq_date"]).dt.month
    daily["month_sin"] = np.sin(2 * np.pi * month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ---- 6. Load model & predict ----
    bundle = joblib.load(MODEL_PATH)
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

    out_cols = ["lat_round", "lon_round", "acq_date", "temperature_2m",
                "relative_humidity_2m", "wind_speed_10m", "ffdi", "kbdi",
                "drought_factor", "kbdi_spinup_flag", "fire_probability",
                "fire_predicted", "risk_level"]
    return daily[out_cols].sort_values("fire_probability", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python predict_live_fire_risk.py <live_weather.csv>")
    #     sys.exit(1)

    input_path = "./data/live_weather_grid.csv"
    raw_df = pd.read_csv(input_path)
    result = predict_live(raw_df)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 80)
    print("FIRE RISK PREDICTIONS")
    print("=" * 80)
    print(result.to_string(index=False))

    out_path = os.path.splitext(input_path)[0] + "_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
