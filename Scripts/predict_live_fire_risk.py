"""
================================================================================
predict_live_fire_risk.py  —  command-line entry point
================================================================================
Takes RAW, LIVE hourly weather rows (the same columns the fetch pipeline
produces: temperature_2m, relative_humidity_2m, wind_speed_10m,
wind_direction_10m, datetime_utc, precipitation, wind_gusts_10m,
soil_moisture_0_to_7cm, cape, vapour_pressure_deficit,
et0_fao_evapotranspiration, lat_round, lon_round, fetched_at) from a CSV and
produces a fire-risk prediction for EVERY (grid cell, day) in the file.

ALL the feature-engineering and model logic lives in
backend/prediction_engine.py. This file is only a CSV-in / CSV-out wrapper
around it.

That matters: this script and the FastAPI backend share the same
fire_state.json (the per-cell KBDI recursion memory). When the two kept
separate copies of the formulas, they drifted apart — the API gained a fix
this script did not have, while both kept writing the same state file.
Importing the one implementation makes that class of bug impossible.

WHY DAILY, NOT HOURLY:
The model was trained on ONE ROW PER (grid cell, day) — hourly weather is
aggregated to daily first (mean for snapshot variables, SUM for
precipitation) before any fire-danger index is computed. So "a prediction
for each row of live data" means one prediction per cell-day, built from
all the hourly rows that day.

WHY STATE PERSISTS ACROSS RUNS:
KBDI and Drought Factor are RECURSIVE — each day's value depends on the
previous day's value for that same grid cell. A brand-new cell starts its
KBDI clock at 0, so its first predictions are cold-start estimates (flagged
by kbdi_spinup_flag). Each later run continues the recursion from the saved
state, and re-running the same day replays the stored result instead of
advancing the recursion twice.

USAGE
-----
    python Scripts/predict_live_fire_risk.py data/live_weather_grid.csv
    python Scripts/predict_live_fire_risk.py data/live_weather_grid.csv --save-db

    # or import and call directly:
    from backend.prediction_engine import predict_live
    result_df = predict_live(new_weather_df)
================================================================================
"""
import argparse
import os
import sys

import pandas as pd

# Script lives in Scripts/, so the repo root must be on the path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.prediction_engine import KBDI_SPINUP_DAYS, predict_live

DEFAULT_INPUT = "data/live_weather_grid.csv"


def main():
    parser = argparse.ArgumentParser(description="Predict fire risk from a live weather CSV.")
    parser.add_argument(
        "input_path", nargs="?", default=DEFAULT_INPUT,
        help=f"CSV of raw hourly weather rows (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--save-db", action="store_true",
        help="Also save the predictions to the fire_prediction table.",
    )
    parser.add_argument(
        "--model-version", default=None,
        help="Optional label stored with saved predictions, for audit.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input file not found: {args.input_path}")
        print("Run Scripts/fetch_openmeteo_live.py first, or pass a path explicitly.")
        sys.exit(1)

    raw_df = pd.read_csv(args.input_path)
    print(f"Loaded {len(raw_df)} hourly rows from {args.input_path}.")

    result = predict_live(raw_df)
    if result.empty:
        print("No predictions produced — check the input file has the expected columns.")
        sys.exit(1)

    n_spinup = int(result["kbdi_spinup_flag"].sum())
    if n_spinup:
        print(
            f"NOTE: {n_spinup} cell-day(s) are within the first {KBDI_SPINUP_DAYS} days of "
            f"that grid cell's monitoring history (kbdi_spinup_flag=True) — their "
            f"KBDI/Drought Factor/FFDI are cold-start estimates and should be treated "
            f"as lower-confidence."
        )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 80)
    print("FIRE RISK PREDICTIONS")
    print("=" * 80)
    print(result.to_string(index=False))

    out_path = os.path.splitext(args.input_path)[0] + "_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    if args.save_db:
        from backend.api.fire_prediction import save_predictions

        n = save_predictions(result, model_version=args.model_version)
        print(f"Saved {n} prediction(s) to the fire_prediction table.")


if __name__ == "__main__":
    main()
