"""
================================================================================
retrain_fire_model.py  —  retrain the fire-occurrence model with two fixes
================================================================================
Reads the labelled daily dataset produced by the preprocessing pipeline
(one row per grid cell per day, with a `label` column) and retrains the
classifier, applying two fixes to problems found in the existing model.

FIX 1 — drop `days_since_cell_start`
    It means "how many days this grid cell has existed in our dataset", not
    anything about fire. In the current model it is the 4th most-used
    feature, and holding weather constant on one cell it swings the
    prediction from 0.958 (value 0) to 0.008 (value 847). Live cells always
    start near 0, which is the maximum-risk end, so every newly monitored
    cell reads as high risk for its first weeks regardless of conditions.
    It cannot transfer to live use, so it is removed from the contract.

    `days_since_rain` is kept but capped (see DAYS_SINCE_RAIN_CAP): it is a
    real physical quantity, but in the current model its splits reach 740
    days, which is the length of the dataset rather than a genuine dry
    spell. Capping stops it encoding series length.

FIX 2 — temporal train/validation/test split
    The existing model splits randomly with stratification. Adjacent days of
    the same grid cell then land in both train and test, and KBDI/drought
    factor barely change from one day to the next, so the test set is not
    independent and the reported score is optimistic. Splitting by date
    instead — train on the earliest period, test on the most recent — is
    both leak-free and the way the system is actually used: predicting
    forward from history.

FIX 3 — natural class balance in validation and test, then calibration
    The dataset now contains every cell-day (build_labeled_dataset.py no
    longer subsamples negatives). Fires cluster in the dry season, so when
    negatives were subsampled globally a temporal cut produced a validation
    split that was 86% fire and a test split that was 18% fire; a threshold
    tuned on the first flagged nearly everything on the second, and the
    probabilities themselves answered "fire or not, given a 50/50 world".
    Negative sampling and the near-fire buffer are now applied to the TRAIN
    split only. Validation and test keep the real rate, the model is
    isotonic-calibrated on validation, and the reliability table reports
    predicted against observed fire rate by bin -- the number that says
    whether a "60%" means 60%. scale_pos_weight is gone for the same reason:
    it re-imposed the 50/50 prior the calibration exists to remove.

FIX 4 — near-fire buffer reduced to 1 day, train only
    The previous +/-3-day exclusion removed ~55% of dry-season days from the
    negative pool in the north, so the model never saw what "September but
    no fire" looks like there and fell back on location and month. One day
    still guards against detection-timing noise (a fire burning the day
    before the satellite pass).

EXPECT THE HEADLINE METRIC TO FALL. That is the point: the drop is the
leakage the old setup was supplying. A lower honest number is worth more
than a higher one you cannot defend.

USAGE
-----
    uv run python Scripts/retrain_fire_model.py --input data/final.csv
    uv run python Scripts/retrain_fire_model.py --input data/final.csv --buffer-days 0
    uv run python Scripts/retrain_fire_model.py --input data/final.csv --neg-per-pos 3
    uv run python Scripts/retrain_fire_model.py --input data/final.csv --no-save

The saved bundle keeps the same shape the backend expects
(model / model_name / features / scaler / threshold_default /
threshold_f1_optimal), so backend/prediction_engine.py needs NO change --
it reads the feature list from the bundle. Dropping a feature here is
picked up automatically at inference.
================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RANDOM_STATE = 42

# The existing 20-feature contract, for reference and for --keep-old-features.
FEATURES_ORIGINAL = [
    "wind_gusts_10m", "soil_moisture_0_to_7cm", "temperature_2m",
    "et0_fao_evapotranspiration", "days_since_cell_start", "wind_speed_10m",
    "wind_direction_10m", "emc", "relative_humidity_2m",
    "vapour_pressure_deficit", "rate_of_spread", "ffdi", "precipitation",
    "month_cos", "days_since_rain", "kbdi", "drought_factor", "month_sin",
    "lat_round", "lon_round",
]

# FIX 1: dataset-artefact feature removed.
DROPPED_FEATURES = ["days_since_cell_start"]
FEATURES = [f for f in FEATURES_ORIGINAL if f not in DROPPED_FEATURES]

# Longest plausible dry spell to encode. Beyond this the value stops being
# weather and starts being "how long our series is".
DAYS_SINCE_RAIN_CAP = 180.0

# FIX 2: fractions of the DATE RANGE (not of the rows) used for each split.
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20

# FIX 4: training negatives within this many days of a fire are dropped.
BUFFER_DAYS = 1

# Fields that exist only because a fire was detected, or are derived from
# the label. Using any of them as a predictor would leak the label.
FIRMS_ONLY_FIELDS = {
    "brightness", "bright_ti4", "bright_ti5", "frp", "confidence", "type",
    "satellite", "instrument", "scan", "track", "acq_time",
    "label", "days_to_nearest_fire",
}

# Reliability table bins: predicted probability -> observed fire rate.
CALIBRATION_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ==============================================================================
# Data preparation
# ==============================================================================
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(
            f"Input file not found: {path}\n"
            "Pass the labelled daily dataset with --input (the CSV produced by "
            "the preprocessing pipeline, one row per grid cell per day)."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    if "label" not in df.columns:
        raise SystemExit("Dataset has no 'label' column.")
    if "acq_date" not in df.columns:
        raise SystemExit("Dataset has no 'acq_date' column (needed for the temporal split).")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df[df["acq_date"].notna()].copy()

    dupes = int(df.duplicated(["lat_round", "lon_round", "acq_date"]).sum())
    if dupes:
        raise SystemExit(
            f"{dupes:,} duplicate grid-cell/day rows. Fix this in preprocessing "
            "before training -- duplicates put the same cell-day in more than one split."
        )

    print(f"Date range: {df['acq_date'].min().date()} to {df['acq_date'].max().date()}")
    print(f"Class balance: {df['label'].value_counts().to_dict()} "
          f"(fire rate {df['label'].mean():.1%})")
    return df


def build_features(df: pd.DataFrame, features: list) -> tuple:
    """Adds the cyclical month features and applies the days_since_rain cap."""
    df = df.copy()
    df["month"] = df["acq_date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "days_since_rain" in features and "days_since_rain" in df.columns:
        over = int((df["days_since_rain"] > DAYS_SINCE_RAIN_CAP).sum())
        if over:
            print(f"Capping days_since_rain at {DAYS_SINCE_RAIN_CAP:.0f} "
                  f"({over:,} rows above it).")
        df["days_since_rain"] = df["days_since_rain"].clip(upper=DAYS_SINCE_RAIN_CAP)

    leaked = set(features) & FIRMS_ONLY_FIELDS
    if leaked:
        raise SystemExit(f"FIRMS detection-only field(s) in the predictor list: {sorted(leaked)}")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required features: {missing}")

    return df, df[features].copy(), df["label"].copy()


def temporal_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    FIX 2: splits by date, not at random.

    Cut points are chosen on the date range so train is the earliest period
    and test the most recent. Every cell-day in test happens strictly after
    every cell-day in train, so no same-cell adjacent days straddle the
    boundary and the model is scored on genuinely unseen time.
    """
    dates = df["acq_date"].sort_values().unique()
    train_end = pd.Timestamp(dates[int(len(dates) * TRAIN_FRAC)])
    val_end = pd.Timestamp(dates[int(len(dates) * (TRAIN_FRAC + VAL_FRAC))])

    train_mask = df["acq_date"] <= train_end
    val_mask = (df["acq_date"] > train_end) & (df["acq_date"] <= val_end)
    test_mask = df["acq_date"] > val_end

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        m = mask.values
        splits[name] = (df.loc[m], X[m], y[m])

    print(f"\nTemporal split (train <= {train_end.date()} < val <= {val_end.date()} < test):")
    for name in ("train", "val", "test"):
        ds, _, ys = splits[name]
        if len(ys) == 0:
            raise SystemExit(f"The '{name}' split is empty -- the date range is too short.")
        print(f"  {name:5s}: {len(ys):>7,} rows | fire rate {ys.mean():.2%} "
              f"| {ds['acq_date'].min().date()} to {ds['acq_date'].max().date()}")

    for a, b in (("train", "val"), ("val", "test")):
        if splits[a][0]["acq_date"].max() >= splits[b][0]["acq_date"].min():
            raise SystemExit(f"Temporal split overlap between {a} and {b}.")
    print("  No date overlap between splits.")
    return splits


def prepare_train_split(df_train, X_train, y_train, buffer_days: int, neg_per_pos):
    """
    FIX 3/4: the only place negatives are dropped or sampled. Validation and
    test are never touched, so they keep the real class balance.
    """
    keep = np.ones(len(y_train), dtype=bool)

    if buffer_days > 0:
        if "days_to_nearest_fire" not in df_train.columns:
            raise SystemExit("Dataset has no 'days_to_nearest_fire' column -- rebuild it with "
                             "the current build_labeled_dataset.py.")
        near = df_train["days_to_nearest_fire"].values
        buffered = (y_train.values == 0) & (near > 0) & (near <= buffer_days)
        keep &= ~buffered
        print(f"\nTrain buffer: dropped {int(buffered.sum()):,} negatives within "
              f"{buffer_days} day(s) of a fire.")

    if neg_per_pos is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        pos_idx = np.flatnonzero(keep & (y_train.values == 1))
        neg_idx = np.flatnonzero(keep & (y_train.values == 0))
        n_target = min(int(neg_per_pos * len(pos_idx)), len(neg_idx))
        chosen = rng.choice(neg_idx, size=n_target, replace=False)
        keep = np.zeros(len(y_train), dtype=bool)
        keep[pos_idx] = True
        keep[chosen] = True
        print(f"Train sampling: {len(pos_idx):,} fire + {n_target:,} no-fire "
              f"({neg_per_pos}:1).")

    X_out, y_out = X_train[keep], y_train[keep]
    print(f"Train after preparation: {len(y_out):,} rows | fire rate {y_out.mean():.2%}")
    return X_out, y_out


# ==============================================================================
# Training and evaluation
# ==============================================================================
def train_model(X_train, y_train):
    """
    LightGBM with the existing hyperparameters. No scale_pos_weight: the
    train split is already close to the real balance, and reweighting would
    push the scores back towards the 50/50 prior calibration then has to undo.
    """
    import lightgbm as lgb

    print(f"\nTraining LightGBM on {len(y_train):,} rows...")
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def calibrate(model, X_val, y_val):
    """
    FIX 3: maps the model's scores onto observed fire frequency using the
    validation split, which has the real class balance. Isotonic regression
    is monotonic, so the ranking (ROC-AUC) is unchanged; only the meaning of
    the number changes -- afterwards 0.6 means "cell-days scored like this
    burned about 60% of the time".
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    print("Calibrating on validation (isotonic)...")
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)
    return calibrated


def evaluate(model, X, y, threshold: float, label: str) -> dict:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "split": label, "threshold": round(float(threshold), 4),
        "base_rate": round(float(y.mean()), 4),
        "mean_predicted": round(float(proba.mean()), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "precision_fire": round(precision_score(y, pred, zero_division=0), 4),
        "recall_fire": round(recall_score(y, pred, zero_division=0), 4),
        "f1_fire": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def reliability_table(model, X, y, label: str) -> pd.DataFrame:
    """
    Predicted probability against observed fire rate, by bin. For a
    calibrated model the two columns track each other; the old model put
    almost every row in the top bins regardless of what actually burned.
    """
    proba = model.predict_proba(X)[:, 1]
    bins = pd.cut(proba, CALIBRATION_BINS, include_lowest=True)
    table = (pd.DataFrame({"bin": bins, "predicted": proba, "observed": y.values})
             .groupby("bin", observed=True)
             .agg(rows=("observed", "size"), predicted=("predicted", "mean"),
                  observed=("observed", "mean"))
             .reset_index())
    table["bin"] = table["bin"].astype(str)
    print(f"\nReliability -- {label} (base rate {y.mean():.1%}):")
    print(table.to_string(index=False, formatters={
        "predicted": "{:.3f}".format, "observed": "{:.3f}".format, "rows": "{:,}".format}))
    return table


def tune_threshold(model, X_val, y_val) -> float:
    """Picks the F1-optimal threshold on VALIDATION only, never on test."""
    from sklearn.metrics import precision_recall_curve

    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = float(thresholds[int(np.nanargmax(f1[:-1]))])
    print(f"F1-optimal threshold on validation: {best:.3f}")
    return best


# ==============================================================================
# Sanity check -- the test that would have caught the original problem
# ==============================================================================
def sanity_check(model, features: list, threshold: float) -> bool:
    """
    Pushes five physically coherent profiles through the model, from very
    high fire danger down to soaking wet, and checks the probability falls.

    Values are kept inside realistic observed ranges on purpose. Feeding a
    tree model values it never saw in training (for example
    et0_fao_evapotranspiration above ~0.7, which is mm/hour, not mm/day)
    sends every input to an extreme leaf and makes the model look broken
    when it is not.
    """
    base = dict(
        wind_gusts_10m=25.0, soil_moisture_0_to_7cm=0.20, temperature_2m=20.0,
        et0_fao_evapotranspiration=0.15, days_since_cell_start=500.0,
        wind_speed_10m=14.0, wind_direction_10m=180.0, emc=12.0,
        relative_humidity_2m=55.0, vapour_pressure_deficit=1.2,
        rate_of_spread=0.10, ffdi=10.0, precipitation=0.0, month_cos=-0.9,
        days_since_rain=8.0, kbdi=60.0, drought_factor=5.0, month_sin=0.4,
        lat_round=-33.5, lon_round=150.5,   # on the 0.5-degree grid (Blue Mountains, NSW)
    )

    def mk(**kw):
        d = dict(base)
        d.update(kw)
        return d

    profiles = [
        ("Very high (35C, 15% RH, 30km/h, KBDI 190)", mk(
            temperature_2m=35, relative_humidity_2m=15, wind_speed_10m=30, wind_gusts_10m=55,
            emc=5.0, vapour_pressure_deficit=5.2, kbdi=190, drought_factor=9.8, ffdi=50,
            rate_of_spread=0.36, days_since_rain=120, soil_moisture_0_to_7cm=0.03,
            et0_fao_evapotranspiration=0.62)),
        ("High      (31C, 25% RH, 24km/h, KBDI 140)", mk(
            temperature_2m=31, relative_humidity_2m=25, wind_speed_10m=24, wind_gusts_10m=42,
            emc=7.5, vapour_pressure_deficit=3.5, kbdi=140, drought_factor=8.5, ffdi=28,
            rate_of_spread=0.25, days_since_rain=45, soil_moisture_0_to_7cm=0.07,
            et0_fao_evapotranspiration=0.45)),
        ("Moderate  (20C, 55% RH, 14km/h, KBDI 60)", mk()),
        ("Low       (14C, 80% RH, 8km/h, KBDI 25)", mk(
            temperature_2m=14, relative_humidity_2m=80, wind_speed_10m=8, wind_gusts_10m=14,
            emc=18, vapour_pressure_deficit=0.35, kbdi=25, drought_factor=2.5, ffdi=3,
            rate_of_spread=0.03, days_since_rain=2, soil_moisture_0_to_7cm=0.33,
            et0_fao_evapotranspiration=0.08)),
        ("Wet       (10C, 93% RH, 5km/h, 20mm rain)", mk(
            temperature_2m=10, relative_humidity_2m=93, wind_speed_10m=5, wind_gusts_10m=9,
            emc=23, vapour_pressure_deficit=0.08, kbdi=5, drought_factor=1.2, ffdi=0.9,
            rate_of_spread=0.01, days_since_rain=0, precipitation=20,
            soil_moisture_0_to_7cm=0.45, et0_fao_evapotranspiration=0.02)),
    ]

    print("\n" + "=" * 78)
    print("SANITY CHECK -- probability must fall from top to bottom")
    print("=" * 78)
    probs = []
    for name, d in profiles:
        p = float(model.predict_proba(pd.DataFrame([d])[features])[0, 1])
        probs.append(p)
        print(f"  {name:44s} prob={p:.4f}  -> {'FIRE' if p >= threshold else 'no fire'}")

    ok = all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1))
    spread = max(probs) - min(probs)
    print(f"\n  monotonic decreasing: {'YES' if ok else 'NO'}")
    print(f"  spread (max - min)  : {spread:.4f}")
    if not ok:
        print("  WARNING: ordering is not monotonic -- inspect before deploying.")
    if spread < 0.25:
        print("  WARNING: the model barely separates extreme from benign conditions.")
    return ok and spread >= 0.25


def report_importance(model, features: list) -> pd.DataFrame:
    imp = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    print("\nFeature importance (top 10):")
    print(imp.head(10).to_string(index=False))
    zero = imp[imp.importance == 0]["feature"].tolist()
    if zero:
        print(f"Never split on: {zero}")
        print("  (ffdi and rate_of_spread are exactly proportional -- "
              "rate_of_spread = 0.0096 x ffdi -- so the model using only one is expected.)")
    return imp


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    parser.add_argument("--input", default="data/final.csv",
                        help="Labelled daily dataset CSV (default: data/final.csv)")
    parser.add_argument("--output", default="models/best_fire_model.joblib",
                        help="Where to write the model bundle")
    parser.add_argument("--keep-old-features", action="store_true",
                        help="Keep days_since_cell_start (for an A/B comparison only)")
    parser.add_argument("--buffer-days", type=int, default=BUFFER_DAYS,
                        help=f"Drop TRAIN negatives within N days of a fire (default {BUFFER_DAYS}; 0 = off)")
    parser.add_argument("--neg-per-pos", type=float, default=None,
                        help="Subsample TRAIN negatives to this ratio (default: keep all)")
    parser.add_argument("--no-save", action="store_true",
                        help="Train and report, but do not write the bundle")
    args = parser.parse_args()

    features = FEATURES_ORIGINAL if args.keep_old_features else FEATURES
    print("=" * 78)
    print("RETRAIN FIRE MODEL")
    print("=" * 78)
    print(f"Features: {len(features)}"
          + ("" if args.keep_old_features else f" (dropped: {DROPPED_FEATURES})"))

    df = load_dataset(args.input)
    df, X, y = build_features(df, features)
    splits = temporal_split(df, X, y)
    df_train, X_train, y_train = splits["train"]
    _, X_val, y_val = splits["val"]
    _, X_test, y_test = splits["test"]

    X_train, y_train = prepare_train_split(df_train, X_train, y_train,
                                           args.buffer_days, args.neg_per_pos)

    base_model = train_model(X_train, y_train)
    model = calibrate(base_model, X_val, y_val)
    threshold = tune_threshold(model, X_val, y_val)

    rows = [
        evaluate(model, X_val, y_val, 0.50, "validation @0.50"),
        evaluate(model, X_val, y_val, threshold, "validation @tuned"),
        evaluate(model, X_test, y_test, 0.50, "TEST @0.50"),
        evaluate(model, X_test, y_test, threshold, "TEST @tuned"),
    ]
    print("\n" + "=" * 78)
    print("RESULTS (calibrated; test = most recent period, never seen in training)")
    print("=" * 78)
    print(pd.DataFrame(rows).to_string(index=False))

    reliability = {
        "validation": reliability_table(model, X_val, y_val, "validation (calibration set)"),
        "test": reliability_table(model, X_test, y_test, "TEST"),
    }

    report_importance(base_model, features)
    passed = sanity_check(model, features, threshold)

    if args.no_save:
        print("\n--no-save set; bundle not written.")
        return

    bundle = {
        "model": model,
        "model_name": "LightGBM (isotonic-calibrated)",
        "features": features,
        "scaler": None,
        "threshold_default": 0.50,
        "threshold_f1_optimal": float(threshold),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(bundle, args.output)

    # Reload and confirm the saved file reproduces the in-memory model.
    reloaded = joblib.load(args.output)
    assert reloaded["features"] == features
    np.testing.assert_allclose(
        model.predict_proba(X_test.iloc[:20])[:, 1],
        reloaded["model"].predict_proba(X_test.iloc[:20])[:, 1],
    )
    print(f"\nSaved and verified: {args.output}")

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "split": "temporal", "features": features,
        "dropped_features": [] if args.keep_old_features else DROPPED_FEATURES,
        "days_since_rain_cap": DAYS_SINCE_RAIN_CAP,
        "calibration": "isotonic on validation",
        "train_buffer_days": args.buffer_days,
        "train_neg_per_pos": args.neg_per_pos,
        "threshold_f1_optimal": float(threshold),
        "rows": {"train_after_preparation": int(len(y_train)),
                 **{k: int(len(splits[k][2])) for k in splits}},
        "metrics": rows,
        "reliability": {k: v.to_dict(orient="records") for k, v in reliability.items()},
        "sanity_check_passed": bool(passed),
    }
    meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Metadata:            {meta_path}")

    if not passed:
        print("\nSanity check did NOT pass -- review before using this model live.")


if __name__ == "__main__":
    main()
