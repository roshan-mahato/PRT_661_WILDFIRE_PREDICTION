"""
Places that stay at High or Extreme risk on the most forecast days.
"""

from typing import Optional

import pandas as pd

from config import HOTSPOT_ROWS


def persistent_hotspots(df: pd.DataFrame, top: Optional[int] = HOTSPOT_ROWS) -> pd.DataFrame:
    """
    The grid cells that stay at High or Extreme risk on the most forecast days.

    A single bad day is shown by the map and day strip already; this answers
    the planning question instead -- where does the danger *last*? Cells are
    ranked by the number of High/Extreme days, then by how many of those are
    Extreme, then by peak probability. `top=None` returns every such cell.

    Returns columns [lat_round, lon_round, days_elevated, days_extreme,
    days_high, days_total, worst_level, peak_probability, elevated_dates];
    only cells with at least one elevated day. Empty if there are none.
    """
    cols = ["lat_round", "lon_round", "days_elevated", "days_extreme", "days_high",
            "days_total", "worst_level", "peak_probability", "elevated_dates"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    data = df.assign(
        acq_date=pd.to_datetime(df["acq_date"]).dt.normalize(),
        level=df["risk_level"].astype(str),
    )
    elevated = data[data["level"].isin(["High", "Extreme"])]
    if elevated.empty:
        return pd.DataFrame(columns=cols)

    per_cell = (
        elevated.assign(is_extreme=elevated["level"] == "Extreme")
        .groupby(["lat_round", "lon_round"])
        .agg(
            days_elevated=("acq_date", "nunique"),
            days_extreme=("is_extreme", "sum"),
            peak_probability=("fire_probability", "max"),
            elevated_dates=("acq_date", lambda s: sorted(set(s))),
        )
        .reset_index()
        .sort_values(["days_elevated", "days_extreme", "peak_probability"], ascending=False)
        .reset_index(drop=True)
    )
    if top is not None:
        per_cell = per_cell.head(top)
    per_cell["days_extreme"] = per_cell["days_extreme"].astype(int)
    per_cell["days_high"] = per_cell["days_elevated"] - per_cell["days_extreme"]
    per_cell["days_total"] = data["acq_date"].nunique()
    per_cell["worst_level"] = per_cell["days_extreme"].map(lambda n: "Extreme" if n else "High")
    return per_cell[cols]
