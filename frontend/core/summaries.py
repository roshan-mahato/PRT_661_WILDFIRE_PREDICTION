"""
Region-level figures derived from per-cell predictions: the selected day's
summary, one entry per forecast day, and the riskiest / calmest day.
"""

from typing import List, Optional

import pandas as pd

from core.danger import RISK_ORDER
from core.types import DayOption, RegionSummary, WeekHighlights


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


def day_options(df: pd.DataFrame) -> List[DayOption]:
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
                "mean_probability": float(group["fire_probability"].mean()),
                "at_risk_count": int(group["fire_predicted"].sum()),
                "cell_count": int(len(group)),
                "max_ffdi": float(group["ffdi"].max()),
            }
        )
    return out


def summarise_region(df: pd.DataFrame, region: str) -> RegionSummary:
    """
    Derives the headline figures for a region from its prediction rows.

    Returns a dict with the same shape whether or not data exists, so the UI
    never has to guard against missing keys.
    """
    empty: RegionSummary = {
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


def week_highlights(options: List[DayOption]) -> Optional[WeekHighlights]:
    """
    Picks the riskiest and calmest forecast day from day_options().

    Days are ranked by the share of cells at risk -- how much of the region
    is in danger -- with mean probability breaking ties. The single peak cell
    is a poor ranking key: nationally some cell is Extreme almost every day.

    Returns None with fewer than two days. `flat` is True when the riskiest
    and calmest days are within 15% of each other (or 2 percentage points,
    whichever is larger), so the UI can say the outlook is steady instead of
    naming an arbitrary "worst" day. The test is relative because the share
    at risk is often small: nationally ~13%, where 11% vs 15% is a real
    difference of about a hundred cells.
    """
    if len(options) < 2:
        return None

    def score(option):
        share = option["at_risk_count"] / option["cell_count"] if option["cell_count"] else 0.0
        return (share, option.get("mean_probability", option["peak_probability"]))

    ranked = sorted(options, key=score)
    worst, best = ranked[-1], ranked[0]
    worst_share, best_share = score(worst)[0], score(best)[0]
    flat = (worst_share - best_share) < max(0.02, 0.15 * worst_share)
    return {"worst": worst, "best": best, "flat": flat}
