"""
================================================================================
build_labeled_dataset.py  —  one labelled row per (grid cell, day)
================================================================================
Builds the dataset used to train the fire-occurrence model. Replaces
`build_labeled_dataset()` from wildfire_data_pipeline_full.ipynb (Section 3).

This script emits EVERY cell-day in the observed window with its label and
features. It does no negative sampling and no buffering: those are training
decisions, applied to the TRAIN split only by retrain_fire_model.py, so that
the validation and test splits keep the natural class balance. Calibration
and the decision threshold are only meaningful against that natural rate.

WHY THE LABELLING WAS REBUILT (kept from the previous version)
--------------------------------------------------------------
The original version built the two classes in different ways:

  * POSITIVES  merged each FIRMS detection to weather on
                (cell, date, HOUR), so a positive cell-day was the weather
                of ~1 satellite overpass hour.
  * NEGATIVES  pulled ALL 24 hourly rows for the sampled day.

Both were then averaged by aggregate_to_daily(). So positives were the mean
of about one hour and negatives the mean of twenty-four. That difference has
nothing to do with fire, and a model can separate the classes on it alone --
most visibly through et0_fao_evapotranspiration, which is ~0 at night and
peaks at midday, and which was the most-used feature in the trained model.

Measured, with fire dates generated RANDOMLY and independently of weather
(so the honest answer is ROC-AUC 0.50):

    original pipeline   ROC-AUC 1.0000   <- perfect skill from pure noise
    this version        ROC-AUC 0.4931   <- chance, as it should be

The fix: build ONE daily weather series first, then label it by lookup.
`acq_time` is discarded entirely -- it is detection metadata, not weather.

WHY THE KBDI SERIES COMES FROM A SEPARATE FILE
----------------------------------------------
The hourly weather CSV holds 14-day windows around fire detections, not a
continuous record (median 33% of days present per cell, median longest gap
~200 days). Every feature but one is a same-day snapshot, so that is fine.
KBDI / Drought Factor are a recursion over consecutive days, and the
recursion treats a gap as that many rain-free days, so on the gappy series
the drought values the model learned from were badly inflated. The
recursion is therefore run over a continuous daily temperature/rain series
(Scripts/fetch_kbdi_history.py) and joined back by (cell, day).

Rows are restricted to the FIRMS archive's date range. Outside it a day
with no detection is unobserved, not fire-free, and would be a false
negative.

USAGE
-----
    uv run python Scripts/fetch_kbdi_history.py            # once; resumable
    uv run python Scripts/build_labeled_dataset.py \
        --firms data/fire_archive_SV-C2_792465.csv \
        --weather data/all_weather_data.csv \
        --kbdi-weather data/kbdi_weather_continuous.csv \
        --output data/final.csv

Then train on the result:

    uv run python Scripts/retrain_fire_model.py --input data/final.csv
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_engine import (
    KBDI_SPINUP_DAYS, WEATHER_COLS,
    compute_emc, compute_ffdi, compute_ros, kbdi_df_step,
)

GRID_SIZE = 0.5
CONFIDENCE_TOKEEP = ["n", "h"]
TYPE_TOKEEP = [0]
MAX_POSITIVE_RATE_PER_CELL = 0.80
NO_FIRE_SENTINEL = 9999   # days_to_nearest_fire for a cell with no detections


def round_to_grid(v, grid=GRID_SIZE):
    return (v / grid).round() * grid


# ==============================================================================
# Step 1: daily weather series from the hourly CSV (snapshot features)
# ==============================================================================
def build_daily_weather_series(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses hourly weather to one row per (cell, day).

    precipitation is SUMMED (it accumulates); every other variable is a
    snapshot reading and is averaged. Both classes come from this one table,
    so they are aggregated identically -- that is the whole point.
    """
    w = weather_df.copy()
    w["acq_date"] = pd.to_datetime(w["datetime_utc"]).dt.tz_localize(None).dt.floor("D")

    cols = [c for c in WEATHER_COLS if c in w.columns]
    agg = {c: ("sum" if c == "precipitation" else "mean") for c in cols}
    daily = (
        w.groupby(["lat_round", "lon_round", "acq_date"], as_index=False)
        .agg(agg)
        .sort_values(["lat_round", "lon_round", "acq_date"])
        .reset_index(drop=True)
    )
    print(f"Daily weather series: {len(daily):,} cell-days across "
          f"{daily.groupby(['lat_round','lon_round']).ngroups:,} cells")
    return daily


# ==============================================================================
# Step 2: fire detections -> (cell, date) keys and the observed window
# ==============================================================================
def build_fire_keys(firms_df: pd.DataFrame) -> tuple:
    """
    Reduces FIRMS detections to the set of (cell, date) pairs that burned,
    plus the first and last detection dates (the observed window).

    `acq_time` is deliberately NOT used. Keeping it was what tied positives
    to a single overpass hour. A cell-day either had a detection or it did
    not; what time the satellite passed over is not a property of the fire.
    """
    f = firms_df.copy()
    n_start = len(f)

    f = f.drop_duplicates()
    if "lat_round" not in f.columns:
        f["lat_round"] = round_to_grid(f["latitude"])
    if "lon_round" not in f.columns:
        f["lon_round"] = round_to_grid(f["longitude"])

    if "confidence" in f.columns:
        f = f[f["confidence"].isin(CONFIDENCE_TOKEEP)]
    if "type" in f.columns:
        f = f[f["type"].isin(TYPE_TOKEEP)]

    f["acq_date"] = pd.to_datetime(f["acq_date"]).dt.floor("D")
    keys = set(zip(f["lat_round"].round(3), f["lon_round"].round(3), f["acq_date"]))
    start, end = f["acq_date"].min(), f["acq_date"].max()
    print(f"FIRMS: {n_start:,} detections -> {len(keys):,} unique (cell, date) fire pairs, "
          f"{start.date()} to {end.date()}")
    return keys, start, end


# ==============================================================================
# Step 3: label every cell-day
# ==============================================================================
def label_all(daily: pd.DataFrame, fire_keys: set,
              max_pos_rate: float = MAX_POSITIVE_RATE_PER_CELL) -> pd.DataFrame:
    """
    Labels each cell-day by lookup and adds `days_to_nearest_fire`.

    `days_to_nearest_fire` is derived from the label and exists only so the
    training script can drop near-fire negatives from the TRAIN split. It
    must never be used as a model feature.

    Cells whose positive rate exceeds max_pos_rate are dropped: almost every
    day burned there, so a model can score well by memorising lat/lon
    instead of reading the weather.
    """
    d = daily.copy()
    d["acq_date"] = pd.to_datetime(d["acq_date"])

    keys = list(zip(d["lat_round"].round(3), d["lon_round"].round(3), d["acq_date"]))
    d["label"] = np.fromiter((k in fire_keys for k in keys), dtype=int, count=len(d))

    rate = d.groupby(["lat_round", "lon_round"])["label"].mean()
    dominated = rate[rate > max_pos_rate].index
    if len(dominated):
        before = len(d)
        idx = pd.MultiIndex.from_arrays([d["lat_round"], d["lon_round"]])
        d = d[~idx.isin(dominated)].reset_index(drop=True)
        print(f"Dropped {len(dominated):,} cell(s) with >{max_pos_rate:.0%} positive rate "
              f"({before - len(d):,} rows) -- they invite location memorisation.")

    nearest = np.full(len(d), NO_FIRE_SENTINEL, dtype=int)
    for _, idx in d.groupby(["lat_round", "lon_round"]).indices.items():
        days = d["acq_date"].values[idx].astype("datetime64[D]").astype(int)
        fires = np.sort(days[d["label"].values[idx] == 1])
        if len(fires) == 0:
            continue
        pos = np.searchsorted(fires, days)
        left = np.abs(days - fires[np.clip(pos - 1, 0, len(fires) - 1)])
        right = np.abs(fires[np.clip(pos, 0, len(fires) - 1)] - days)
        nearest[idx] = np.minimum(left, right)
    d["days_to_nearest_fire"] = nearest

    print(f"Labelled: {len(d):,} cell-days | fire rate {d['label'].mean():.1%} "
          f"({int(d['label'].sum()):,} fire, {int((d['label'] == 0).sum()):,} no-fire)")
    return d


# ==============================================================================
# Step 4: drought indices over the CONTINUOUS series, then join
# ==============================================================================
def load_kbdi_weather(path: str) -> pd.DataFrame:
    k = pd.read_csv(path)
    k["acq_date"] = pd.to_datetime(k["acq_date"])
    k = k.sort_values(["lat_round", "lon_round", "acq_date"]).reset_index(drop=True)
    gaps = k.groupby(["lat_round", "lon_round"])["acq_date"].diff().dt.days.dropna()
    contiguous = (gaps == 1).mean() * 100 if len(gaps) else 100.0
    print(f"KBDI weather series: {len(k):,} cell-days across "
          f"{k.groupby(['lat_round','lon_round']).ngroups:,} cells "
          f"({contiguous:.1f}% consecutive)")
    if contiguous < 99.9:
        raise SystemExit("The KBDI weather series has gaps -- re-run fetch_kbdi_history.py.")
    return k


def compute_drought_indices(kbdi_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the KBDI/Drought Factor recursion per cell over the continuous
    series. Running it on a sparse series inflates KBDI badly, because each
    step would span many real days.
    """
    out = []
    for (lat, lon), g in kbdi_daily.groupby(["lat_round", "lon_round"], sort=False):
        g = g.sort_values("acq_date").reset_index(drop=True)
        state = None
        recs = []
        for row in g.itertuples(index=False):
            res, state = kbdi_df_step(state, row.acq_date, row.temperature_2m,
                                      row.precipitation if pd.notna(row.precipitation) else 0.0)
            recs.append(res)
        r = pd.DataFrame(recs)
        r["lat_round"], r["lon_round"], r["acq_date"] = lat, lon, g["acq_date"].values
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    print(f"Drought indices computed for {len(res):,} cell-days.")
    return res[["lat_round", "lon_round", "acq_date", "kbdi", "drought_factor",
                "days_since_rain", "days_since_cell_start", "kbdi_spinup_flag"]]


def engineer_features(labeled: pd.DataFrame, drought: pd.DataFrame) -> pd.DataFrame:
    """Joins the drought indices, then adds the instantaneous features."""
    df = labeled.merge(drought, on=["lat_round", "lon_round", "acq_date"], how="left")
    unmatched = int(df["kbdi"].isna().sum())
    if unmatched:
        print(f"  {unmatched:,} rows had no drought index (cell missing from the "
              f"continuous series) -- dropping.")
        df = df[df["kbdi"].notna()].reset_index(drop=True)

    # The recursion's first KBDI_SPINUP_DAYS are a cold-start estimate; the
    # live pipeline never serves such rows now that its state is warm.
    spin = df["kbdi_spinup_flag"].astype(bool)
    if spin.any():
        print(f"  {int(spin.sum()):,} rows in KBDI spin-up (first {KBDI_SPINUP_DAYS} days "
              f"of the series) -- dropping.")
        df = df[~spin].reset_index(drop=True)

    df["emc"] = compute_emc(df["relative_humidity_2m"].to_numpy(float),
                            df["temperature_2m"].to_numpy(float))
    df["ffdi"] = compute_ffdi(df["temperature_2m"].to_numpy(float),
                              df["relative_humidity_2m"].to_numpy(float),
                              df["wind_speed_10m"].to_numpy(float),
                              df["drought_factor"].to_numpy(float))
    df["rate_of_spread"] = compute_ros(df["ffdi"].to_numpy(float))
    month = pd.to_datetime(df["acq_date"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    print(f"Features built: {len(df):,} rows.")
    return df


# ==============================================================================
# Verification
# ==============================================================================
def check_class_symmetry(df: pd.DataFrame) -> bool:
    """
    Compares the spread of each weather variable between classes.

    This is the check that would have caught the original bug. If one class
    was aggregated over a different number of hours, its standard deviation
    differs sharply -- in the original pipeline et0 had std 0.0279 for
    positives against 0.0027 for negatives, a 10x gap.
    """
    print("\n" + "=" * 74)
    print("CLASS SYMMETRY CHECK -- std ratio should be near 1 for every variable")
    print("=" * 74)
    worst, worst_col = 1.0, None
    for c in [c for c in WEATHER_COLS if c in df.columns]:
        s0 = df.loc[df.label == 0, c].std()
        s1 = df.loc[df.label == 1, c].std()
        ratio = max(s0, s1) / max(min(s0, s1), 1e-12)
        flag = "  <-- CHECK" if ratio > 2.0 else ""
        print(f"  {c:30s} std0={s0:9.4f} std1={s1:9.4f} ratio={ratio:5.2f}{flag}")
        if ratio > worst:
            worst, worst_col = ratio, c
    ok = worst <= 2.0
    print(f"\n  worst ratio: {worst:.2f} ({worst_col}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  A large gap means the classes were built differently. Investigate "
              "before training -- the model will learn the difference, not the fire.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    parser.add_argument("--firms", required=True, help="FIRMS detections CSV")
    parser.add_argument("--weather", required=True, help="Hourly weather CSV (snapshot features)")
    parser.add_argument("--kbdi-weather", default="data/kbdi_weather_continuous.csv",
                        help="Continuous daily temp/rain series from fetch_kbdi_history.py")
    parser.add_argument("--output", default="data/final.csv", help="Output CSV")
    args = parser.parse_args()

    for p in (args.firms, args.weather, args.kbdi_weather):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")

    print("=" * 74)
    print("BUILD LABELLED DATASET")
    print("=" * 74)
    weather = pd.read_csv(args.weather)
    firms = pd.read_csv(args.firms)
    print(f"Loaded {len(weather):,} weather rows, {len(firms):,} FIRMS rows")

    daily = build_daily_weather_series(weather)
    fire_keys, firms_start, firms_end = build_fire_keys(firms)

    before = len(daily)
    daily = daily[(daily["acq_date"] >= firms_start) & (daily["acq_date"] <= firms_end)]
    daily = daily.reset_index(drop=True)
    if len(daily) < before:
        print(f"Dropped {before - len(daily):,} cell-days outside the FIRMS window "
              f"(unobserved, not fire-free).")

    labeled = label_all(daily, fire_keys)
    drought = compute_drought_indices(load_kbdi_weather(args.kbdi_weather))
    final = engineer_features(labeled, drought)

    passed = check_class_symmetry(final)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}  ({len(final):,} rows, {len(final.columns)} columns)")
    if not passed:
        print("Symmetry check FAILED -- review before training on this file.")


if __name__ == "__main__":
    main()
