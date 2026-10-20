"""
================================================================================
model_comparison.py  --  evidence for the A3 feedback (model choice, sanity
check, regional performance, risk bands)
================================================================================
Trains several models on EXACTLY the split the deployed model uses
(retrain_fire_model.py: temporal 60/20/20, 1-day near-fire buffer on train,
isotonic calibration on validation, F1-optimal threshold on validation) and
scores them all on the same held-out test period. Nothing is saved over the
deployed model.

Models
    ffdi_only       logistic regression on peak-hour FFDI alone -- the physics
                    baseline: does machine learning add anything to the index?
    logistic        logistic regression on all features (scaled)
    random_forest   random forest on all features
    lgbm            LightGBM, same settings as the deployed model
    lgbm_no_coords  LightGBM without latitude/longitude
    lgbm_monotone   LightGBM where risk may only rise with FFDI, KBDI, heat,
                    wind, VPD, dryness and only fall with humidity, EMC, soil
                    moisture and rain (the fix proposed for the sanity check)

Outputs (reports/model_comparison/)
    season_coverage.csv   rows and fire rate per split and season
    model_comparison.csv  test metrics for every model, incl. top-5% precision
    per_state_test.csv    test ROC-AUC / PR-AUC / F1 per state and model
    risk_bands_test.csv   cell-days and observed fire rate per risk band
    sanity_profiles.csv   the 5-profile sanity check at 5 places x 3 months
    summary.md            the main tables in Markdown, ready for the report

USAGE
-----
    uv run python Scripts/model_comparison.py                       # everything (~20-40 min)
    uv run python Scripts/model_comparison.py --models ffdi_only lgbm lgbm_monotone
    uv run python Scripts/model_comparison.py --train-frac 0.1      # quick trial run
================================================================================
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "frontend"))

import retrain_fire_model as rt  # noqa: E402  (same split, calibration, threshold, metrics)
from prediction_engine import PEAK_FEATURES  # noqa: E402
from core.geography import STATE_PRIORITY, assign_states  # noqa: E402  (dashboard's state rules)

OUT_DIR = os.path.join(REPO_ROOT, "reports", "model_comparison")
COORDS = ["lat_round", "lon_round"]

# Direction each weather input may push fire risk (monotone_constraints).
# Everything not listed (wind direction, month, coordinates) is unconstrained.
RISK_UP = [
    "temperature_2m", "temperature_2m_max", "wind_speed_10m", "wind_speed_10m_max",
    "wind_gusts_10m", "wind_gusts_10m_max", "vapour_pressure_deficit",
    "vapour_pressure_deficit_max", "et0_fao_evapotranspiration",
    "et0_fao_evapotranspiration_sum", "kbdi", "drought_factor", "days_since_rain",
    "ffdi", "ffdi_max", "rate_of_spread",
]
RISK_DOWN = ["relative_humidity_2m", "relative_humidity_2m_min", "emc",
             "soil_moisture_0_to_7cm", "precipitation"]

# Sanity-check places (all on the 0.5-degree land grid) and months.
SANITY_PLACES = [
    ("Blue Mountains, NSW", -33.5, 150.5),
    ("Central Victoria", -37.0, 144.5),
    ("South-west WA", -32.0, 116.0),
    ("SE Queensland", -27.5, 152.5),
    ("Top End, NT", -13.0, 131.5),
]
SANITY_MONTHS = {"Jan": 1, "May": 5, "Sep": 9}

SEASONS = {12: "Summer (Dec-Feb)", 1: "Summer (Dec-Feb)", 2: "Summer (Dec-Feb)",
           3: "Autumn (Mar-May)", 4: "Autumn (Mar-May)", 5: "Autumn (Mar-May)",
           6: "Winter (Jun-Aug)", 7: "Winter (Jun-Aug)", 8: "Winter (Jun-Aug)",
           9: "Spring (Sep-Nov)", 10: "Spring (Sep-Nov)", 11: "Spring (Sep-Nov)"}

TOP_SHARE = 0.05  # "precision among the riskiest 5% of cell-days"


# ==============================================================================
# Models
# ==============================================================================
def lgbm(**extra):
    import lightgbm as lgb
    # Same hyperparameters as rt.train_model, so only the tested change differs.
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              min_child_samples=30, random_state=rt.RANDOM_STATE,
                              n_jobs=-1, verbosity=-1, **extra)


def scaled_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    # max_samples keeps each tree to a quarter of the rows so it fits in memory
    # and finishes in minutes on 1.6 M rows.
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=50, max_features="sqrt",
                                  max_samples=0.25, n_jobs=-1, random_state=rt.RANDOM_STATE)


def model_specs(features: list) -> dict:
    """name -> (feature list, factory, description)."""
    mono = [1 if f in RISK_UP else -1 if f in RISK_DOWN else 0 for f in features]
    no_coords = [f for f in features if f not in COORDS]
    return {
        "ffdi_only": (["ffdi_max"], scaled_logistic, "Logistic regression on peak-hour FFDI only"),
        "logistic": (features, scaled_logistic, "Logistic regression, all features"),
        "random_forest": (features, random_forest, "Random forest, all features"),
        "lgbm": (features, lgbm, "LightGBM (deployed configuration)"),
        "lgbm_no_coords": (no_coords, lgbm, "LightGBM without latitude/longitude"),
        "lgbm_monotone": (features, lambda: lgbm(monotone_constraints=mono),
                          "LightGBM with monotone fire-weather constraints"),
    }


# ==============================================================================
# Sanity profiles at several places and months
# ==============================================================================
def sanity_profiles(lat: float, lon: float, month: int) -> list:
    """
    The five profiles of rt.sanity_check (very high -> wet), placed at
    (lat, lon) in `month`. Values copied from retrain_fire_model.sanity_check;
    keep the two in step if those change.
    """
    base = dict(
        wind_gusts_10m=25.0, soil_moisture_0_to_7cm=0.20, temperature_2m=20.0,
        et0_fao_evapotranspiration=0.15, wind_speed_10m=14.0, wind_direction_10m=180.0,
        emc=12.0, relative_humidity_2m=55.0, vapour_pressure_deficit=1.2,
        rate_of_spread=0.10, ffdi=10.0, precipitation=0.0, days_since_rain=8.0,
        kbdi=60.0, drought_factor=5.0, lat_round=lat, lon_round=lon,
        month_sin=np.sin(2 * np.pi * month / 12), month_cos=np.cos(2 * np.pi * month / 12),
    )

    def mk(**kw):
        d = dict(base)
        d.update(kw)
        d.setdefault("temperature_2m_max", d["temperature_2m"] + 7)
        d.setdefault("relative_humidity_2m_min", max(d["relative_humidity_2m"] - 20, 5))
        d.setdefault("wind_speed_10m_max", d["wind_speed_10m"] * 1.6)
        d.setdefault("wind_gusts_10m_max", d["wind_gusts_10m"] * 1.5)
        d.setdefault("vapour_pressure_deficit_max", d["vapour_pressure_deficit"] * 1.8)
        d.setdefault("et0_fao_evapotranspiration_sum", d["et0_fao_evapotranspiration"] * 24)
        d.setdefault("wind_direction_at_max_wind", d["wind_direction_10m"])
        d.setdefault("ffdi_max", d["ffdi"] * 2.0)
        return d

    return [
        ("Very high", mk(temperature_2m=35, relative_humidity_2m=15, wind_speed_10m=30,
                         wind_gusts_10m=55, emc=5.0, vapour_pressure_deficit=5.2, kbdi=190,
                         drought_factor=9.8, ffdi=50, rate_of_spread=0.36, days_since_rain=120,
                         soil_moisture_0_to_7cm=0.03, et0_fao_evapotranspiration=0.62)),
        ("High", mk(temperature_2m=31, relative_humidity_2m=25, wind_speed_10m=24,
                    wind_gusts_10m=42, emc=7.5, vapour_pressure_deficit=3.5, kbdi=140,
                    drought_factor=8.5, ffdi=28, rate_of_spread=0.25, days_since_rain=45,
                    soil_moisture_0_to_7cm=0.07, et0_fao_evapotranspiration=0.45)),
        ("Moderate", mk()),
        ("Low", mk(temperature_2m=14, relative_humidity_2m=80, wind_speed_10m=8,
                   wind_gusts_10m=14, emc=18, vapour_pressure_deficit=0.35, kbdi=25,
                   drought_factor=2.5, ffdi=3, rate_of_spread=0.03, days_since_rain=2,
                   soil_moisture_0_to_7cm=0.33, et0_fao_evapotranspiration=0.08)),
        ("Wet", mk(temperature_2m=10, relative_humidity_2m=93, wind_speed_10m=5,
                   wind_gusts_10m=9, emc=23, vapour_pressure_deficit=0.08, kbdi=5,
                   drought_factor=1.2, ffdi=0.9, rate_of_spread=0.01, days_since_rain=0,
                   precipitation=20, soil_moisture_0_to_7cm=0.45,
                   et0_fao_evapotranspiration=0.02)),
    ]


def run_sanity(name: str, model, feats: list) -> pd.DataFrame:
    rows = []
    for place, lat, lon in SANITY_PLACES:
        for mname, month in SANITY_MONTHS.items():
            profiles = sanity_profiles(lat, lon, month)
            X = pd.DataFrame([d for _, d in profiles])[feats]
            p = model.predict_proba(X)[:, 1]
            rows.append({
                "model": name, "place": place, "month": mname,
                **{f"p_{label.lower().replace(' ', '_')}": round(float(v), 4)
                   for (label, _), v in zip(profiles, p)},
                "monotonic": bool(all(p[i] >= p[i + 1] - 1e-9 for i in range(len(p) - 1))),
                "spread": round(float(p.max() - p.min()), 4),
            })
    return pd.DataFrame(rows)


# ==============================================================================
# Test-period breakdowns
# ==============================================================================
def top_share_precision(y: np.ndarray, proba: np.ndarray, share: float = TOP_SHARE) -> float:
    k = max(1, int(len(proba) * share))
    top = np.argsort(-proba, kind="stable")[:k]
    return float(y[top].mean())


def per_state(name: str, states: pd.Series, y: np.ndarray, proba: np.ndarray,
              threshold: float) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    rows = []
    for state in STATE_PRIORITY + ["Offshore / unassigned"]:
        m = (states == state).values if state != "Offshore / unassigned" else states.isna().values
        n, fires = int(m.sum()), int(y[m].sum()) if m.any() else 0
        ok = n > 0 and 20 <= fires < n  # too few fires -> scores are noise
        rows.append({
            "model": name, "state": state, "rows": n, "fires": fires,
            "fire_rate": round(fires / n, 4) if n else np.nan,
            "roc_auc": round(roc_auc_score(y[m], proba[m]), 4) if ok else np.nan,
            "pr_auc": round(average_precision_score(y[m], proba[m]), 4) if ok else np.nan,
            "f1": round(f1_score(y[m], (proba[m] >= threshold).astype(int)), 4) if ok else np.nan,
        })
    return pd.DataFrame(rows)


def risk_bands(name: str, y: np.ndarray, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Same band rule as prediction_engine.predict_from_features."""
    edges = [-np.inf, threshold * 0.5, threshold, threshold + (1.0 - threshold) * 0.5, np.inf]
    band = pd.cut(proba, edges, labels=["Low", "Moderate", "High", "Extreme"], right=False)
    t = (pd.DataFrame({"band": band, "y": y})
         .groupby("band", observed=False)
         .agg(cell_days=("y", "size"), fires=("y", "sum"))
         .reset_index())
    t.insert(0, "model", name)
    t["from_prob"] = [round(max(e, 0.0), 3) for e in edges[:-1]]
    t["share_of_cell_days"] = (t["cell_days"] / t["cell_days"].sum()).round(4)
    t["observed_fire_rate"] = (t["fires"] / t["cell_days"].where(t["cell_days"] > 0)).round(4)
    t["share_of_all_fires"] = (t["fires"] / t["fires"].sum()).round(4)
    return t


def season_coverage(splits: dict) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        df, _, y = splits[split]
        season = df["acq_date"].dt.month.map(SEASONS)
        for s in ["Summer (Dec-Feb)", "Autumn (Mar-May)", "Winter (Jun-Aug)", "Spring (Sep-Nov)"]:
            m = (season == s).values
            rows.append({"split": split, "season": s, "days": int(df.loc[m, "acq_date"].nunique()),
                         "rows": int(m.sum()), "fires": int(y[m].sum()),
                         "fire_rate": round(float(y[m].mean()), 4) if m.any() else np.nan})
    return pd.DataFrame(rows)


# ==============================================================================
# Main
# ==============================================================================
def to_md(df: pd.DataFrame) -> str:
    """Markdown table without needing the tabulate package."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def main():
    specs_all = model_specs([])  # names only, for the help text
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "data", "final_dataset.csv"))
    parser.add_argument("--models", nargs="+", default=list(specs_all), choices=list(specs_all))
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="Train on this random share of the training rows (quick runs). "
                             "Validation and test are always used in full.")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = rt.load_dataset(args.input)
    features = list(rt.FEATURES)
    if all(c in df.columns for c in PEAK_FEATURES):
        features += PEAK_FEATURES
    df, X, y = rt.build_features(df, features)
    splits = rt.temporal_split(df, X, y)
    df_train, X_train, y_train = splits["train"]
    df_val, X_val, y_val = splits["val"]
    df_test, X_test, y_test = splits["test"]
    X_train, y_train = rt.prepare_train_split(df_train, X_train, y_train, rt.BUFFER_DAYS, None)
    if args.train_frac < 1.0:
        idx = np.random.default_rng(rt.RANDOM_STATE).random(len(y_train)) < args.train_frac
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"--train-frac {args.train_frac}: training on {len(y_train):,} rows "
              "(trial run -- do not report these numbers)")

    seasons = season_coverage(splits)
    test_states = assign_states(df_test["lat_round"], df_test["lon_round"]).reset_index(drop=True)
    yt = y_test.to_numpy()

    specs = model_specs(features)
    results, states_out, bands_out, sanity_out = [], [], [], []
    for name in args.models:
        feats, factory, desc = specs[name]
        print("\n" + "=" * 78 + f"\n{name}: {desc} ({len(feats)} features)\n" + "=" * 78)
        t0 = time.time()
        base = factory()
        base.fit(X_train[feats], y_train)
        model = rt.calibrate(base, X_val[feats], y_val)
        threshold = rt.tune_threshold(model, X_val[feats], y_val)
        proba = model.predict_proba(X_test[feats])[:, 1]

        row = rt.evaluate(model, X_test[feats], y_test, threshold, "test @tuned")
        sanity = run_sanity(name, model, feats)
        results.append({
            "model": name, "description": desc, "features": len(feats),
            "roc_auc": row["roc_auc"], "pr_auc": row["pr_auc"], "brier": row["brier"],
            "threshold": row["threshold"], "precision": row["precision_fire"],
            "recall": row["recall_fire"], "f1": row["f1_fire"],
            f"precision_top_{int(TOP_SHARE * 100)}pct": round(top_share_precision(yt, proba), 4),
            "sanity_pass": f"{int(sanity['monotonic'].sum())}/{len(sanity)}",
            "sanity_median_spread": round(float(sanity["spread"].median()), 4),
            "train_minutes": round((time.time() - t0) / 60, 1),
        })
        states_out.append(per_state(name, test_states, yt, proba, threshold))
        bands_out.append(risk_bands(name, yt, proba, threshold))
        sanity_out.append(sanity)
        print(pd.DataFrame([results[-1]]).T.to_string(header=False))

    comparison = pd.DataFrame(results)
    states = pd.concat(states_out, ignore_index=True)
    bands = pd.concat(bands_out, ignore_index=True)
    sanity = pd.concat(sanity_out, ignore_index=True)

    for fname, table in [("season_coverage.csv", seasons), ("model_comparison.csv", comparison),
                         ("per_state_test.csv", states), ("risk_bands_test.csv", bands),
                         ("sanity_profiles.csv", sanity)]:
        table.to_csv(os.path.join(args.out, fname), index=False)

    # One wide table per metric is easier to read in a report than the long CSV.
    state_auc = states.pivot(index="state", columns="model", values="roc_auc")
    state_info = states.drop_duplicates("state").set_index("state")[["rows", "fires", "fire_rate"]]
    state_wide = state_info.join(state_auc).reindex(STATE_PRIORITY + ["Offshore / unassigned"])
    sanity_wide = (sanity.assign(result=sanity["monotonic"].map({True: "pass", False: "FAIL"}))
                   .pivot_table(index=["place", "month"], columns="model", values="result",
                                aggfunc="first"))

    md = [
        "# Model comparison -- held-out test period",
        "",
        f"Trained on {len(y_train):,} rows" + (f" (--train-frac {args.train_frac}: TRIAL RUN)"
                                              if args.train_frac < 1 else "")
        + f"; test {len(yt):,} cell-days, base rate {yt.mean():.2%}.",
        "",
        "## Season coverage of each split", "", to_md(seasons), "",
        "## Test metrics (threshold tuned on validation)", "",
        to_md(comparison.drop(columns=["description"])), "",
        "## Test ROC-AUC by state", "", to_md(state_wide.reset_index()), "",
        "## Risk bands on the test period", "", to_md(bands), "",
        "## Sanity check (probability must fall from very high to wet)", "",
        to_md(sanity_wide.reset_index()), "",
    ]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md))

    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(comparison.drop(columns=["description"]).to_string(index=False))
    print(f"\nAll tables written to {args.out}/ (summary.md has them in Markdown).")


if __name__ == "__main__":
    main()