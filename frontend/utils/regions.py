"""
Turns grid-cell predictions into the region-level numbers the dashboard shows.

The backend predicts per 0.5-degree grid cell; the UI selects by state. This
module bridges the two with approximate state bounding boxes, and derives the
summary figures the insight cards and threat panel display.

The bounding boxes are rectangles, not real state borders, so cells near a
border may be counted in the neighbouring state. That is acceptable for a
risk overview -- the map still shows every cell in its true position -- but
it is not suitable for official reporting.
"""

from typing import Any, Dict, Optional

import pandas as pd

# (lat_min, lat_max, lon_min, lon_max) -- approximate, rectangular.
#
# Real state borders are irregular (the NSW/Victoria border follows the Murray
# River diagonally), so rectangles inevitably overlap near borders. Rather than
# letting a border cell be counted in two states -- which would make per-state
# counts sum to more than the national total -- every cell is assigned to
# exactly ONE state via STATE_PRIORITY below.
STATE_BOUNDS: Dict[str, tuple] = {
    "Australian Capital Territory": (-35.92, -35.12, 148.76, 149.40),
    "Tasmania": (-43.70, -39.20, 143.80, 148.60),
    "Victoria": (-39.20, -35.80, 140.90, 150.10),
    "New South Wales": (-37.60, -28.10, 140.90, 153.70),
    "Queensland": (-29.20, -10.00, 137.90, 153.70),
    "South Australia": (-38.10, -25.95, 129.00, 141.05),
    "Northern Territory": (-26.05, -10.90, 129.00, 138.05),
    "Western Australia": (-35.20, -13.50, 112.90, 129.05),
}

# Checked in this order; the first box that contains a cell wins. Smaller and
# more enclosed regions come first, so the ACT is not swallowed by NSW and
# southern border cells resolve to Victoria rather than NSW.
STATE_PRIORITY = [
    "Australian Capital Territory",
    "Tasmania",
    "Victoria",
    "New South Wales",
    "Queensland",
    "South Australia",
    "Northern Territory",
    "Western Australia",
]

# Map centre and zoom per region, used by the map component.
REGION_VIEWS: Dict[str, tuple] = {
    "Australia (National)": ([-25.2744, 133.7751], 4),
    "New South Wales": ([-32.9, 147.0], 6),
    "Victoria": ([-36.9, 144.5], 6),
    "Queensland": ([-22.0, 145.0], 5),
    "Western Australia": ([-25.5, 122.5], 5),
    "South Australia": ([-30.0, 135.5], 5),
    "Tasmania": ([-42.0, 146.8], 7),
    "Northern Territory": ([-19.5, 133.5], 5),
    "Australian Capital Territory": ([-35.5, 149.0], 9),
}

RISK_ORDER = ["Low", "Moderate", "High", "Extreme"]

NATIONAL = "Australia (National)"


def is_national(region: str) -> bool:
    """
    True if the region means 'the whole country'.

    Written tolerantly because the label has appeared with and without a
    space ("Australia(national)" vs "Australia (National)") -- matching on
    the exact string meant a typo silently fell through to a default view.
    """
    if not region:
        return True
    normalised = region.lower().replace(" ", "").replace("(", "").replace(")", "")
    return normalised in {"australianational", "australia", "national"}


def assign_state(lat: float, lon: float) -> Optional[str]:
    """
    Returns the single state a grid cell belongs to, or None if it falls
    outside every box (offshore cells, mostly).

    Checking STATE_PRIORITY in order guarantees one cell maps to one state,
    so per-state counts add up to the national total.
    """
    for state in STATE_PRIORITY:
        lat_min, lat_max, lon_min, lon_max = STATE_BOUNDS[state]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return state
    return None


def filter_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Restricts predictions to the grid cells assigned to a state.

    Uses assign_state() rather than a plain bounding-box test, so a cell near
    a state border appears under one state only.
    """
    if df.empty or is_national(region):
        return df
    if region not in STATE_BOUNDS:
        return df
    assigned = df.apply(
        lambda row: assign_state(row["lat_round"], row["lon_round"]), axis=1
    )
    return df[assigned == region].reset_index(drop=True)


def get_region_view(region: str) -> tuple:
    """Returns ([lat, lon], zoom) for the map."""
    if is_national(region):
        return REGION_VIEWS[NATIONAL]
    return REGION_VIEWS.get(region, REGION_VIEWS[NATIONAL])


def latest_day_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the most recent forecast day.

    Summary cards should describe one day, not an average across the whole
    7-day window -- mixing days would understate a single extreme day.
    """
    if df.empty or "acq_date" not in df.columns:
        return df
    latest = pd.to_datetime(df["acq_date"]).max()
    return df[pd.to_datetime(df["acq_date"]) == latest].reset_index(drop=True)


def day_options(df: pd.DataFrame) -> list:
    """
    One entry per forecast day, oldest first, for the day selector strip.

    Each entry carries the headline figures a tile shows, so the strip can be
    drawn without re-grouping the frame once per day.
    """
    if df.empty or "acq_date" not in df.columns:
        return []

    out = []
    for day, group in df.groupby(pd.to_datetime(df["acq_date"]).dt.date, sort=True):
        counts = {level: int((group["risk_level"] == level).sum()) for level in RISK_ORDER}
        peak = next((level for level in reversed(RISK_ORDER) if counts[level] > 0), "Low")
        out.append(
            {
                "date": day,
                "peak_level": peak,
                "peak_probability": float(group["fire_probability"].max()),
                "at_risk_count": int(group["fire_predicted"].sum()),
                "cell_count": int(len(group)),
                "max_ffdi": float(group["ffdi"].max()),
            }
        )
    return out


def summarise_region(df: pd.DataFrame, region: str) -> Dict[str, Any]:
    """
    Derives the headline figures for a region from its prediction rows.

    Returns a dict with the same shape whether or not data exists, so the UI
    never has to guard against missing keys.
    """
    empty = {
        "has_data": False,
        "region": region,
        "cell_count": 0,
        "at_risk_count": 0,
        "max_risk_level": "Low",
        "max_probability": 0.0,
        "mean_probability": 0.0,
        "max_ffdi": 0.0,
        "mean_ffdi": 0.0,
        "mean_temp": 0.0,
        "mean_humidity": 0.0,
        "mean_wind": 0.0,
        "max_kbdi": 0.0,
        "hottest_cell": None,
        "risk_counts": {level: 0 for level in RISK_ORDER},
        "spinup_count": 0,
        "forecast_date": None,
    }
    if df.empty:
        return empty

    day = latest_day_only(df)
    if day.empty:
        return empty

    risk_counts = {level: int((day["risk_level"] == level).sum()) for level in RISK_ORDER}
    highest = next(
        (level for level in reversed(RISK_ORDER) if risk_counts[level] > 0), "Low"
    )
    top = day.loc[day["fire_probability"].idxmax()]

    return {
        "has_data": True,
        "region": region,
        "cell_count": int(len(day)),
        "at_risk_count": int(day["fire_predicted"].sum()),
        "max_risk_level": highest,
        "max_probability": float(day["fire_probability"].max()),
        "mean_probability": float(day["fire_probability"].mean()),
        "max_ffdi": float(day["ffdi"].max()),
        "mean_ffdi": float(day["ffdi"].mean()),
        "mean_temp": float(day["temperature_2m"].mean()),
        "mean_humidity": float(day["relative_humidity_2m"].mean()),
        "mean_wind": float(day["wind_speed_10m"].mean()),
        "max_kbdi": float(day["kbdi"].max()),
        "hottest_cell": (float(top["lat_round"]), float(top["lon_round"])),
        "risk_counts": risk_counts,
        "spinup_count": int(day["kbdi_spinup_flag"].sum()) if "kbdi_spinup_flag" in day else 0,
        "forecast_date": pd.to_datetime(day["acq_date"]).max().date(),
    }


def ffdi_band(ffdi: float) -> str:
    """
    McArthur FFDI danger bands.

    Note these are the classic FFDI ratings, which are a different scale from
    the model's probability-based risk_level -- a cell can sit in a high FFDI
    band while the classifier gives it a low fire probability, and vice versa.
    """
    if ffdi < 12:
        return "Low-Moderate"
    if ffdi < 25:
        return "High"
    if ffdi < 50:
        return "Very High"
    if ffdi < 75:
        return "Severe"
    if ffdi < 100:
        return "Extreme"
    return "Catastrophic"


def build_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates predictions into a per-day trend for the forecast chart.

    Returns columns [acq_date, mean_probability, max_probability, mean_ffdi,
    max_ffdi, at_risk_cells].
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "acq_date", "mean_probability", "max_probability",
                "mean_ffdi", "max_ffdi", "at_risk_cells",
            ]
        )
    grouped = (
        df.groupby("acq_date")
        .agg(
            mean_probability=("fire_probability", "mean"),
            max_probability=("fire_probability", "max"),
            mean_ffdi=("ffdi", "mean"),
            max_ffdi=("ffdi", "max"),
            at_risk_cells=("fire_predicted", "sum"),
        )
        .reset_index()
        .sort_values("acq_date")
    )
    return grouped
