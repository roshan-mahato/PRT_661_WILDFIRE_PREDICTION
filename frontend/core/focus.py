"""
"Town focus": the dashboard centred on one searched town and the area
around it, instead of a whole region.

A town takes the forecast of the 0.5-degree grid area it sits in. Some towns
-- mostly on the coast -- sit in an area that has no forecast (ocean areas are
not forecast), so the nearest forecast area is used instead and the UI says
so. The "surroundings" are every forecast area whose centre is within the
chosen radius of the town, always including the town's own area.

Everything here works on the national predictions, by distance, ignoring the
rectangular state boxes used by the region filter: a border town such as
Albury must find its forecast and its neighbours on both sides of the border.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import FALLBACK_MAX_KM
from core.hotspots import persistent_hotspots
from core.places import _grid, direction, distances_km, town_coordinates, towns_at_risk
from core.summaries import day_options
from core.types import DayOption


@dataclass(frozen=True)
class Focus:
    town: str                         # "Kingaroy, QLD"
    lat: float                        # the town's own coordinates
    lon: float
    radius_km: int
    cell: Optional[Tuple[float, float]]  # forecast area used for the town, None if none nearby
    offset_km: float = 0.0            # distance from the town to that area's centre, if not its own
    offset_direction: str = ""        # direction from the town to that area, if not its own

    @property
    def name(self) -> str:
        """'Kingaroy' -- the town without its state."""
        return self.town.split(",")[0]

    @property
    def state(self) -> str:
        """'QLD'."""
        return self.town.split(",")[1].strip()

    @property
    def own_area(self) -> bool:
        """True when the town's own grid area has a forecast."""
        return self.cell is not None and self.offset_km == 0.0


def _cell_keys(predictions: pd.DataFrame) -> pd.DataFrame:
    """Unique forecast areas in `predictions` as lat_round / lon_round columns."""
    return predictions[["lat_round", "lon_round"]].drop_duplicates().reset_index(drop=True)


def make_focus(predictions: pd.DataFrame, town: str, radius_km: int) -> Focus:
    """
    Builds the focus for a town: which forecast area it uses. Its own grid
    area if that has a forecast, otherwise the nearest one within
    FALLBACK_MAX_KM, otherwise none.
    """
    lat, lon = town_coordinates(town)
    if predictions is None or predictions.empty:
        return Focus(town, lat, lon, radius_km, None)

    cells = _cell_keys(predictions)
    own = (_grid(lat), _grid(lon))
    keys = set(zip(cells["lat_round"].round(3), cells["lon_round"].round(3)))
    if own in keys:
        return Focus(town, lat, lon, radius_km, own)

    dist = distances_km(lat, lon, cells["lat_round"], cells["lon_round"])
    i = int(np.argmin(dist))
    if dist[i] > FALLBACK_MAX_KM:
        return Focus(town, lat, lon, radius_km, None)
    cell = (float(cells.at[i, "lat_round"]), float(cells.at[i, "lon_round"]))
    return Focus(town, lat, lon, radius_km, cell, float(dist[i]), direction(lat, lon, *cell))


def is_focus_cell(predictions: pd.DataFrame, focus: Focus) -> pd.Series:
    """True for the rows of the town's forecast area."""
    return ((predictions["lat_round"].round(3) == round(focus.cell[0], 3))
            & (predictions["lon_round"].round(3) == round(focus.cell[1], 3)))


def cells_within(predictions: pd.DataFrame, focus: Focus, radius_km: Optional[float] = None) -> pd.DataFrame:
    """
    Prediction rows for every forecast area whose centre is within the radius
    of the town (default: the focus radius), always including the town's own
    forecast area -- with a 25 km radius the area centre itself can be
    further than that from the town.
    """
    if predictions.empty or focus.cell is None:
        return predictions.iloc[0:0]
    radius = focus.radius_km if radius_km is None else radius_km
    dist = distances_km(focus.lat, focus.lon, predictions["lat_round"], predictions["lon_round"])
    keep = (dist <= radius) | is_focus_cell(predictions, focus).to_numpy()
    return predictions[keep].reset_index(drop=True)


def town_rows(predictions: pd.DataFrame, focus: Focus) -> pd.DataFrame:
    """The town's own forecast rows, one per day."""
    if predictions.empty or focus.cell is None:
        return predictions.iloc[0:0]
    return predictions[is_focus_cell(predictions, focus)].reset_index(drop=True)


def town_day_options(scope: pd.DataFrame, focus: Focus) -> List[DayOption]:
    """
    Day-strip entries for town focus: each day's level, chance of fire and
    FFDI are the TOWN's own; the "x of y at risk" counts describe the
    surrounding area, so the riskiest-day line compares the whole area.
    """
    options = day_options(scope)
    town = town_rows(scope, focus)
    by_day = {pd.Timestamp(d).date(): row for d, row in zip(town["acq_date"], town.itertuples(index=False))}
    for option in options:
        row = by_day.get(option["date"])
        if row is not None:
            option["peak_level"] = str(row.risk_level)
            option["peak_probability"] = float(row.fire_probability)
            option["max_ffdi"] = float(row.ffdi)
    return options


def nearby_towns(day_predictions: pd.DataFrame, focus: Focus, radius_km: float) -> pd.DataFrame:
    """
    Other listed towns within the radius of the searched town, with their
    forecast for the day: columns [town, risk_level, fire_probability,
    distance_km, direction], nearest first.

    Towns are chosen by their OWN distance from the searched town (town to
    town), then given their area's forecast from `day_predictions` (the
    national data). Selecting forecast areas first would wrongly drop a town
    that is in range while its area's centre is not -- e.g. Gladstone is
    ~90 km from Rockhampton but its area's centre is ~120 km away.
    """
    cols = ["town", "risk_level", "fire_probability", "distance_km", "direction"]
    if day_predictions is None or day_predictions.empty:
        return pd.DataFrame(columns=cols)
    towns = towns_at_risk(day_predictions)
    towns = towns[towns["town"] != focus.town]
    if towns.empty:
        return pd.DataFrame(columns=cols)
    coords = [town_coordinates(t) for t in towns["town"]]
    lats, lons = zip(*coords)
    towns = towns.assign(
        distance_km=distances_km(focus.lat, focus.lon, lats, lons),
        direction=[direction(focus.lat, focus.lon, la, lo) for la, lo in coords],
    )
    towns = towns[towns["distance_km"] <= radius_km]
    return towns.sort_values("distance_km", kind="mergesort").reset_index(drop=True)[cols]


def town_timeline(scope: pd.DataFrame, focus: Focus, rows: int) -> pd.DataFrame:
    """
    Rows for the where-and-when timeline in town focus: the town's own area
    first (even if it is never High or Extreme), then the most persistent
    places in the surroundings. Same columns as persistent_hotspots().
    """
    town = town_rows(scope, focus)
    own = persistent_hotspots(town, top=None)
    if own.empty:
        days = pd.to_datetime(scope["acq_date"]).dt.normalize().nunique()
        level = str(town.loc[town["fire_probability"].idxmax(), "risk_level"]) if not town.empty else "Low"
        own = pd.DataFrame([{
            "lat_round": focus.cell[0], "lon_round": focus.cell[1],
            "days_elevated": 0, "days_extreme": 0, "days_high": 0, "days_total": days,
            "worst_level": level,
            "peak_probability": float(town["fire_probability"].max()) if not town.empty else 0.0,
            "elevated_dates": [],
        }])
    others = persistent_hotspots(scope[~is_focus_cell(scope, focus)], top=max(rows - 1, 0))
    return pd.concat([own, others], ignore_index=True)
