"""
The data behind the Insights charts, prepared without any plotting.

Each function takes the region's prediction rows and returns plain tables or
lists that a chart component draws. Keeping this apart from the Plotly code
means the numbers can be tested directly.
"""

import pandas as pd

from core.danger import RISK_ORDER
from core.places import describe_location, format_coords
from core.plain import cell_area_km2, weather_words


def day_dates(predictions: pd.DataFrame) -> pd.Series:
    """Each row's forecast day, as a midnight timestamp."""
    return pd.to_datetime(predictions["acq_date"]).dt.normalize()


def capitalise(text: str) -> str:
    """'near Kingaroy, QLD' -> 'Near Kingaroy, QLD'."""
    return text[:1].upper() + text[1:]


# --- Chart 1: fire weather ------------------------------------------------------
def daily_fire_weather(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Per forecast day: the typical (regional mean) FFDI and weather, and the
    worst FFDI with the place and weather where it occurs.
    """
    rows = []
    for day, group in predictions.groupby(day_dates(predictions), sort=True):
        worst = group.loc[group["ffdi"].idxmax()]
        rows.append({
            "date": day,
            "typical_ffdi": float(group["ffdi"].mean()),
            "worst_ffdi": float(worst["ffdi"]),
            "worst_lat": float(worst["lat_round"]),
            "worst_lon": float(worst["lon_round"]),
            "worst_weather": weather_words(worst),
            "typical_weather": weather_words(group[
                ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]].mean()),
        })
    return pd.DataFrame(rows)


def daily_town_fire_weather(town_rows: pd.DataFrame) -> pd.DataFrame:
    """Per forecast day, for one town's forecast area: its FFDI and weather."""
    rows = [
        {"date": day, "town_ffdi": float(group["ffdi"].iloc[0]),
         "town_weather": weather_words(group.iloc[0])}
        for day, group in town_rows.groupby(day_dates(town_rows), sort=True)
    ]
    return pd.DataFrame(rows, columns=["date", "town_ffdi", "town_weather"])


# --- Chart 2: how widespread ----------------------------------------------------
def daily_risk_shares(predictions: pd.DataFrame) -> tuple:
    """
    Share of the region's land area at each risk level, per day (rows: days,
    columns: RISK_ORDER, values 0-1), plus the matching areas in km^2 and
    the number of areas (grid cells). Area-weighted, because a cell near
    Darwin covers ~30% more ground than one in Tasmania.
    """
    data = predictions.assign(
        day=day_dates(predictions),
        area=predictions["lat_round"].map(cell_area_km2),
        level=predictions["risk_level"].astype(str),
    )
    area = data.pivot_table(index="day", columns="level", values="area", aggfunc="sum",
                            fill_value=0).reindex(columns=RISK_ORDER, fill_value=0).sort_index()
    counts = pd.crosstab(data["day"], data["level"]).reindex(
        columns=RISK_ORDER, fill_value=0).sort_index()
    shares = area.div(area.sum(axis=1), axis=0)
    return shares, area, counts


# --- Chart 3: places affected ---------------------------------------------------
def names_preview(names: list, limit: int = 6) -> str:
    """'Kingaroy, Gympie, … and 3 more' -- town names without their state."""
    if not names:
        return "none"
    shown = ", ".join(n.split(",")[0] for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


# --- Chart 4: where the danger lasts --------------------------------------------
def hotspot_labels(hotspots: pd.DataFrame) -> list:
    """Place names for the timeline rows. Two cells can share a description
    ("near Kingaroy, QLD"), and Plotly merges identical category labels into
    one row, so duplicates get their coordinates appended."""
    names = [
        capitalise(describe_location(r.lat_round, r.lon_round))
        for r in hotspots.itertuples(index=False)
    ]
    counts = pd.Series(names).value_counts()
    return [
        f"{n} ({format_coords(r.lat_round, r.lon_round)})" if counts[n] > 1 else n
        for n, r in zip(names, hotspots.itertuples(index=False))
    ]
