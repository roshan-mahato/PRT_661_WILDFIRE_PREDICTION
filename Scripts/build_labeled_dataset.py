"""
================================================================================
build_labeled_dataset.py  —  one labelled row per (land grid cell, day)
================================================================================
Builds the dataset used to train the fire-occurrence model from

    --weather  data/full_grid_daily_weather.csv   (fetch_weather_history.py)
               every land cell, every day, daily means/sums + daily peaks
    --firms    data/firms_combined.csv            (combine_firms.py)
               VIIRS S-NPP + NOAA-20 detections, archive + NRT

and writes data/final_dataset.csv: every land cell-day in the FIRMS window,
with its weather, fire-danger features and label. There is no negative
sampling and no buffering here: those are training decisions, applied to the
TRAIN split only by retrain_fire_model.py, so validation and test keep the
natural class balance.

LABEL
-----
label = 1 when EITHER satellite detected a fire in the cell that UTC day,
counting detections with
    confidence  n (nominal) or h (high)          -- low is mostly false alarms
    type        0 (vegetation fire) or empty     -- NRT rows are not classified;
                                                    1/2/3 = volcano, static
                                                    industrial source, offshore
`acq_time` is not used: a cell-day either had a detection or it did not.

WHICH DAYS ARE OBSERVED
-----------------------
A satellite observed a day when it has at least one DAYTIME detection
anywhere in Australia; a day without one is an outage (whole-day or
night-only), not a fire-free day. Days no satellite observed are dropped --
labelling them 0 would teach the model false negatives. `satellites_observed`
(1 or 2) is kept for analysis; it is not a weather feature.

Rows are limited to the FIRMS window: outside it a day with no detection is
unobserved, not fire-free.

FEATURES
--------
KBDI / Drought Factor are a recursion over consecutive days, run over the
whole continuous weather series (which starts KBDI_SPINUP_DAYS before the
FIRMS window, so no training row is a cold start). The same code as the live
pipeline (prediction_engine.engineer_features) computes every feature,
including ffdi_max from the peak fire-weather hour.

WHY THE LABELLING WAS REBUILT (history)
---------------------------------------
The first version merged each detection to the weather of its overpass HOUR
but averaged 24 hours for non-fire days, so the classes differed in how they
were aggregated -- a model scored ROC-AUC 1.00 on randomly generated fire
dates. Both classes now come from one daily table and are labelled by lookup.
check_class_symmetry() guards against that bug coming back.

USAGE
-----
    uv run python Scripts/build_labeled_dataset.py
    uv run python Scripts/build_labeled_dataset.py --weather data/full_grid_daily_weather.csv \\
        --firms data/firms_combined.csv --output data/final_dataset.csv

Then train on the result:

    uv run python Scripts/retrain_fire_model.py --input data/final_dataset.csv
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_engine import (  # noqa: E402
    DAILY_PEAK_COLS, KBDI_EVENT_STATE_COLS, KBDI_SPINUP_DAYS, WEATHER_COLS, engineer_features,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEATHER = os.path.join(REPO_ROOT, "data", "full_grid_daily_weather.csv")
DEFAULT_FIRMS = os.path.join(REPO_ROOT, "data", "firms_combined.csv")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "data", "final_dataset.csv")

CONFIDENCE_TOKEEP = ["n", "h"]
TYPE_TOKEEP = [0]                  # plus empty (NRT, unclassified)
MAX_POSITIVE_RATE_PER_CELL = 0.80
NO_FIRE_SENTINEL = 9999            # days_to_nearest_fire for a cell with no detections

KEYS = ["lat_round", "lon_round", "acq_date"]


def cell_key(lat, lon) -> list:
    return list(zip(np.round(np.asarray(lat, float), 3), np.round(np.asarray(lon, float), 3)))


# ==============================================================================
# Step 1: weather
# ==============================================================================
def load_weather(path: str) -> pd.DataFrame:
    """The continuous daily series; stops if a cell has a gap or a value is missing."""
    w = pd.read_csv(path)
    w["acq_date"] = pd.to_datetime(w["acq_date"])
    missing = [c for c in WEATHER_COLS + DAILY_PEAK_COLS if c not in w.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing} -- re-run fetch_weather_history.py.")
    w = w.drop(columns=[c for c in ("datetime_utc",) if c in w.columns])
    w = w.sort_values(["lat_round", "lon_round", "acq_date"]).reset_index(drop=True)

    if w.duplicated(KEYS).any():
        raise SystemExit(f"{path} has duplicate cell-days.")
    step = w.groupby(["lat_round", "lon_round"])["acq_date"].diff().dt.days.dropna()
    if (step != 1).any():
        raise SystemExit(f"{path} has {int((step != 1).sum()):,} gap(s) in a cell's series -- "
                         "KBDI would treat them as rain-free days. Re-run fetch_weather_history.py.")
    n_nan = int(w[WEATHER_COLS + DAILY_PEAK_COLS].isna().any(axis=1).sum())
    if n_nan:
        raise SystemExit(f"{path} has {n_nan:,} rows with missing values.")

    n_cells = w.groupby(["lat_round", "lon_round"]).ngroups
    print(f"Weather: {len(w):,} cell-days, {n_cells:,} cells, "
          f"{w['acq_date'].min().date()} -> {w['acq_date'].max().date()}")
    return w


# ==============================================================================
# Step 2: fire detections -> fire cell-days and observed days
# ==============================================================================
def load_firms(path: str) -> pd.DataFrame:
    f = pd.read_csv(path, dtype={"version": str, "confidence": str, "satellite": str})
    f["acq_date"] = pd.to_datetime(f["acq_date"])
    need = ["lat_round", "lon_round", "acq_date", "satellite", "confidence", "type", "daynight"]
    missing = [c for c in need if c not in f.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing} -- build it with combine_firms.py.")
    print(f"FIRMS: {len(f):,} detections, {f['acq_date'].min().date()} -> "
          f"{f['acq_date'].max().date()}, satellites {sorted(f['satellite'].unique())}")
    return f


def observed_days(f: pd.DataFrame) -> pd.Series:
    """Per day, how many satellites observed it (had a daytime detection)."""
    start, end = f["acq_date"].min(), f["acq_date"].max()
    days = pd.date_range(start, end)
    obs = (f[f["daynight"] == "D"].groupby("acq_date")["satellite"].nunique()
           .reindex(days, fill_value=0).rename("satellites_observed"))
    per_sat = f[f["daynight"] == "D"].groupby("satellite")["acq_date"].nunique()
    print(f"Observed days: {len(days):,} in the window | per satellite "
          f"{ {s: f'{n:,} ({len(days) - n} missing)' for s, n in per_sat.items()} } | "
          f"by 1 satellite {int((obs == 1).sum())}, by none {int((obs == 0).sum())}")
    return obs


def fire_cell_days(f: pd.DataFrame, land: set) -> set:
    """(lat_round, lon_round, date) with a counted detection."""
    keep = f["confidence"].isin(CONFIDENCE_TOKEEP) & (f["type"].isna() | f["type"].isin(TYPE_TOKEEP))
    g = f[keep]
    keys = set(zip(*zip(*cell_key(g["lat_round"], g["lon_round"])), g["acq_date"]))
    on_land = {k for k in keys if (k[0], k[1]) in land}
    print(f"Fire detections counted: {int(keep.sum()):,} of {len(f):,} "
          f"(dropped: confidence low {int((f['confidence'] == 'l').sum()):,}, "
          f"type 1/2/3 {int(f['type'].isin([1, 2, 3]).sum()):,})")
    print(f"Fire cell-days: {len(keys):,} | on weather cells {len(on_land):,} | "
          f"in cells without weather (coast marked ocean / outside grid) {len(keys) - len(on_land):,}")
    return on_land


# ==============================================================================
# Step 3: label every observed land cell-day
# ==============================================================================
def label_all(daily: pd.DataFrame, fire_keys: set, obs: pd.Series,
              max_pos_rate: float = MAX_POSITIVE_RATE_PER_CELL) -> pd.DataFrame:
    """
    Keeps the observed days of the FIRMS window, labels each cell-day by
    lookup, and adds `days_to_nearest_fire`.

    `days_to_nearest_fire` is derived from the label and exists only so the
    training script can drop near-fire negatives from the TRAIN split. It
    must never be used as a model feature.

    Cells that burned on more than max_pos_rate of the observed days are
    dropped: a model can score well there by memorising lat/lon instead of
    reading the weather.
    """
    d = daily[daily["acq_date"].isin(obs.index)].copy()
    n_window = len(d)
    d["satellites_observed"] = d["acq_date"].map(obs).astype(int)
    d = d[d["satellites_observed"] > 0].reset_index(drop=True)
    if len(d) < n_window:
        print(f"Dropped {n_window - len(d):,} cell-days on days no satellite observed.")

    keys = zip(*zip(*cell_key(d["lat_round"], d["lon_round"])), d["acq_date"])
    d["label"] = np.fromiter((k in fire_keys for k in keys), dtype=np.int8, count=len(d))

    rate = d.groupby(["lat_round", "lon_round"])["label"].transform("mean")
    hit = (rate > max_pos_rate).to_numpy()
    if hit.any():
        n_cells = d.loc[hit, ["lat_round", "lon_round"]].drop_duplicates().shape[0]
        d = d[~hit].reset_index(drop=True)
        print(f"Dropped {n_cells:,} cell(s) that burned on >{max_pos_rate:.0%} of observed days "
              f"({int(hit.sum()):,} rows) -- they invite location memorisation.")

    nearest = np.full(len(d), NO_FIRE_SENTINEL, dtype=np.int32)
    day_num = d["acq_date"].to_numpy("datetime64[D]").astype(np.int64)
    label = d["label"].to_numpy()
    for idx in d.groupby(["lat_round", "lon_round"]).indices.values():
        fires = np.sort(day_num[idx][label[idx] == 1])
        if len(fires) == 0:
            continue
        pos = np.searchsorted(fires, day_num[idx])
        left = np.abs(day_num[idx] - fires[np.clip(pos - 1, 0, len(fires) - 1)])
        right = np.abs(fires[np.clip(pos, 0, len(fires) - 1)] - day_num[idx])
        nearest[idx] = np.minimum(left, right)
    d["days_to_nearest_fire"] = nearest

    print(f"Labelled: {len(d):,} cell-days | fire rate {d['label'].mean():.2%} "
          f"({int(d['label'].sum()):,} fire, {int((d['label'] == 0).sum()):,} no-fire)")
    return d


# ==============================================================================
# Step 4: features over the continuous series
# ==============================================================================
def build_features(w: pd.DataFrame) -> pd.DataFrame:
    """Every model feature, computed over each cell's full continuous series."""
    x = w.rename(columns={"acq_date": "date"})
    x["cell_id"] = x.groupby(["lat_round", "lon_round"], sort=False).ngroup()
    feats = engineer_features(x).rename(columns={"date": "acq_date"})
    return feats.drop(columns=["cell_id"] + KBDI_EVENT_STATE_COLS)


# ==============================================================================
# Verification
# ==============================================================================
def check_class_symmetry(df: pd.DataFrame) -> bool:
    """
    Compares the spread of each weather variable between classes.

    This is the check that would have caught the original aggregation bug:
    there, et0 had std 0.0279 for positives against 0.0027 for negatives.
    With both classes from one daily table a large ratio now points at a data
    problem (or at fires being confined to one climate), so read it before
    training.
    """
    print("\n" + "=" * 74)
    print("CLASS SYMMETRY CHECK -- std ratio should be near 1 for every variable")
    print("=" * 74)
    # Precipitation is a daily SUM with a hard floor at zero, and it genuinely
    # rains less on fire days, so its spread differs between classes for
    # physical reasons.
    worst, worst_col = 1.0, None
    for c in [c for c in WEATHER_COLS if c in df.columns and c != "precipitation"]:
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
        print("  A large gap means the classes differ in more than the weather. Investigate "
              "before training -- the model will learn the difference, not the fire.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weather", default=DEFAULT_WEATHER,
                        help="Continuous daily weather with peaks (fetch_weather_history.py)")
    parser.add_argument("--firms", default=DEFAULT_FIRMS,
                        help="Combined FIRMS detections (combine_firms.py)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV")
    args = parser.parse_args()

    for p in (args.weather, args.firms):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")

    print("=" * 74)
    print("BUILD LABELLED DATASET")
    print("=" * 74)
    weather = load_weather(args.weather)
    firms = load_firms(args.firms)

    first_fire_day = firms["acq_date"].min()
    lead = (first_fire_day - weather["acq_date"].min()).days
    if lead < KBDI_SPINUP_DAYS:
        print(f"WARNING: weather starts only {lead} day(s) before the FIRMS window; the first "
              f"rows are KBDI cold starts and will be dropped.")

    land = set(cell_key(weather["lat_round"], weather["lon_round"]))
    obs = observed_days(firms)
    fire_keys = fire_cell_days(firms, land)

    feats = build_features(weather)
    final = label_all(feats, fire_keys, obs)

    spin = final["kbdi_spinup_flag"].astype(bool)
    if spin.any():
        print(f"Dropped {int(spin.sum()):,} rows in KBDI spin-up (first {KBDI_SPINUP_DAYS} days).")
        final = final[~spin].reset_index(drop=True)
    final = final.sort_values(KEYS).reset_index(drop=True)

    passed = check_class_symmetry(final)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = final.copy()
    out["acq_date"] = out["acq_date"].dt.strftime("%Y-%m-%d")
    tmp = args.output + ".tmp"
    out.to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, args.output)
    print(f"\nSaved: {args.output}  ({len(final):,} rows, {len(final.columns)} columns, "
          f"{final['acq_date'].min().date()} -> {final['acq_date'].max().date()})")
    if not passed:
        print("Symmetry check FAILED -- review before training on this file.")


if __name__ == "__main__":
    main()
