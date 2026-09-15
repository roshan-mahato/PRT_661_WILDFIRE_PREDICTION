"""
================================================================================
build_labeled_dataset.py  —  fixed labelling + negative sampling
================================================================================
Builds the labelled one-row-per-(grid cell, day) dataset used to train the
fire-occurrence model. Replaces `build_labeled_dataset()` from
wildfire_data_pipeline_full.ipynb (Section 3).

WHY THIS REPLACEMENT EXISTS
---------------------------
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

    et0 by class, original:  label 0  std 0.0027 |  label 1  std 0.0279
    et0 by class, fixed:     label 0  std 0.0027 |  label 1  std 0.0027

The ten-fold variance gap is the fingerprint: averaging 24 hours collapses
variance, averaging 1 hour does not.

THE FIX
-------
Build ONE continuous daily weather series first, then label it by lookup.
`acq_time` is discarded entirely -- it is detection metadata, not weather.
Both classes then come from identical 24-hour aggregation and the artefact
cannot exist.

USAGE
-----
    uv run python Scripts/build_labeled_dataset.py \
        --firms data/fire_archive_SV-C2_792465.csv \
        --weather data/all_weather_data.csv \
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

from backend.prediction_engine import (
    KBDI_SPINUP_DAYS, MEAN_ANNUAL_RAINFALL_MM, WEATHER_COLS,
    compute_emc, compute_ffdi, compute_ros, kbdi_df_step,
)

GRID_SIZE = 0.5
CONFIDENCE_TOKEEP = ["n", "h"]
TYPE_TOKEEP = [0]
BUFFER_DAYS = 3          # exclude +/- N days around each fire date
NEG_PER_POS = 1
NEG_SAMPLE_SEED = 42
MAX_POSITIVE_RATE_PER_CELL = 0.80


def round_to_grid(v, grid=GRID_SIZE):
    return (v / grid).round() * grid


# ==============================================================================
# Step 1: continuous daily weather series (the ONLY source of feature rows)
# ==============================================================================
def build_daily_weather_series(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses hourly weather to one row per (cell, day) for EVERY day.

    precipitation is SUMMED (it accumulates); every other variable is a
    snapshot reading and is averaged. This is the single source of feature
    rows for both classes -- that is the whole point of the fix.
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

    gaps = daily.groupby(["lat_round", "lon_round"])["acq_date"].diff().dt.days.dropna()
    contiguous = (gaps == 1).mean() * 100 if len(gaps) else 100.0
    print(f"Daily weather series: {len(daily):,} cell-days across "
          f"{daily.groupby(['lat_round','lon_round']).ngroups:,} cells "
          f"({contiguous:.1f}% consecutive)")
    if contiguous < 95:
        print("  WARNING: notable gaps -- KBDI/Drought Factor accuracy is reduced there.")
    return daily


# ==============================================================================
# Step 2: fire detections -> a set of (cell, date) keys
# ==============================================================================
def build_fire_keys(firms_df: pd.DataFrame) -> set:
    """
    Reduces FIRMS detections to the set of (cell, date) pairs that burned.

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
    print(f"FIRMS: {n_start:,} detections -> {len(keys):,} unique (cell, date) fire pairs")
    return keys


# ==============================================================================
# Step 3: label the series, then sample negatives
# ==============================================================================
def label_and_sample(daily: pd.DataFrame, fire_keys: set,
                     neg_per_pos: float = NEG_PER_POS,
                     buffer_days: int = BUFFER_DAYS,
                     max_pos_rate: float = MAX_POSITIVE_RATE_PER_CELL,
                     seed: int = NEG_SAMPLE_SEED) -> pd.DataFrame:
    """
    Labels each cell-day by lookup, then samples negatives from the same
    table. Both classes are rows of `daily`, so they are structurally
    identical and differ only in their label.

    Two guards:
      * days within +/- buffer_days of a fire are excluded from the negative
        pool -- conditions then are near-identical to the fire day itself,
        so those rows would teach the model contradictions.
      * cells whose positive rate exceeds max_pos_rate are dropped. In such
        cells almost every day burned, so the model can score well by
        memorising lat/lon instead of reading the weather.
    """
    rng = np.random.default_rng(seed)
    d = daily.copy()
    d["acq_date"] = pd.to_datetime(d["acq_date"])

    keys = list(zip(d["lat_round"].round(3), d["lon_round"].round(3), d["acq_date"]))
    d["label"] = np.fromiter((k in fire_keys for k in keys), dtype=int, count=len(d))
    print(f"Labelled: {len(d):,} cell-days | fire rate {d['label'].mean():.1%}")

    rate = d.groupby(["lat_round", "lon_round"])["label"].mean()
    dominated = rate[rate > max_pos_rate].index
    if len(dominated):
        before = len(d)
        idx = pd.MultiIndex.from_arrays([d["lat_round"], d["lon_round"]])
        d = d[~idx.isin(dominated)].reset_index(drop=True)
        print(f"Dropped {len(dominated):,} cell(s) with >{max_pos_rate:.0%} positive rate "
              f"({before - len(d):,} rows) -- they invite location memorisation.")

    pos = d[d["label"] == 1]
    if pos.empty:
        raise SystemExit("No positive rows after labelling -- check the grid rounding and dates.")

    excluded = set()
    for a, b, c in zip(pos["lat_round"].round(3), pos["lon_round"].round(3), pos["acq_date"]):
        for off in range(-buffer_days, buffer_days + 1):
            excluded.add((a, b, c + pd.Timedelta(days=off)))

    neg_all = d[d["label"] == 0]
    nkeys = list(zip(neg_all["lat_round"].round(3), neg_all["lon_round"].round(3),
                     neg_all["acq_date"]))
    ok = np.fromiter((k not in excluded for k in nkeys), dtype=bool, count=len(nkeys))
    pool = neg_all[ok]
    print(f"Negative pool: {len(pool):,} (excluded {len(neg_all) - len(pool):,} within "
          f"+/-{buffer_days} days of a fire)")

    n_target = int(neg_per_pos * len(pos))
    if n_target > len(pool):
        print(f"  Only {len(pool):,} negatives available for a target of {n_target:,}.")
        n_target = len(pool)
    neg = pool.iloc[rng.choice(len(pool), size=n_target, replace=False)]

    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sort_values(["lat_round", "lon_round", "acq_date"]).reset_index(drop=True)
    print(f"Final: {len(out):,} rows ({int(out['label'].sum()):,} fire, "
          f"{int((out['label'] == 0).sum()):,} no-fire | fire rate {out['label'].mean():.1%})")
    return out


# ==============================================================================
# Step 4: drought indices over the CONTINUOUS series, then join
# ==============================================================================
def compute_drought_indices(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the KBDI/Drought Factor recursion per cell over the full continuous
    series, not over the sampled rows. Running it on a sparse sample inflates
    KBDI badly, because each step would span many real days.
    """
    out = []
    for (lat, lon), g in daily.groupby(["lat_round", "lon_round"], sort=False):
        g = g.sort_values("acq_date").reset_index(drop=True)
        state = None
        recs = []
        for row in g.itertuples(index=False):
            res, state = kbdi_df_step(state, row.acq_date, row.temperature_2m,
                                      getattr(row, "precipitation", 0.0) or 0.0)
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
        print(f"  {unmatched:,} rows had no drought index -- dropping.")
        df = df[df["kbdi"].notna()].reset_index(drop=True)

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

    n_spin = int(df["kbdi_spinup_flag"].sum())
    print(f"Features built. {n_spin:,} rows ({100*n_spin/max(len(df),1):.1f}%) in KBDI spin-up "
          f"(first {KBDI_SPINUP_DAYS} days of a cell).")
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
    parser.add_argument("--weather", required=True, help="Hourly weather CSV")
    parser.add_argument("--output", default="data/final.csv", help="Output CSV")
    parser.add_argument("--neg-per-pos", type=float, default=NEG_PER_POS)
    args = parser.parse_args()

    for p in (args.firms, args.weather):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")

    print("=" * 74)
    print("BUILD LABELLED DATASET")
    print("=" * 74)
    weather = pd.read_csv(args.weather)
    firms = pd.read_csv(args.firms)
    print(f"Loaded {len(weather):,} weather rows, {len(firms):,} FIRMS rows")

    daily = build_daily_weather_series(weather)
    fire_keys = build_fire_keys(firms)
    labeled = label_and_sample(daily, fire_keys, neg_per_pos=args.neg_per_pos)
    drought = compute_drought_indices(daily)
    final = engineer_features(labeled, drought)

    passed = check_class_symmetry(final)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}  ({len(final):,} rows, {len(final.columns)} columns)")
    if not passed:
        print("Symmetry check FAILED -- review before training on this file.")


if __name__ == "__main__":
    main()
