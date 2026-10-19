"""
Turns grid-cell coordinates into place descriptions people recognise:
-33.5, 151.0  ->  "near Gosford, NSW"  or  "40 km NW of Katherine, NT".

The town list is frontend/assets/towns.json: 405 approximate town centres
(to about 0.05 degrees), compiled for this dashboard. That is far finer than
the 0.5-degree (~55 km) forecast grid, so it is accurate enough to say which
town a cell is near -- it is not a gazetteer. It deliberately includes small
remote towns, so outback cells get a nearby reference point instead of a
capital city 800 km away.

It lives in frontend/assets rather than data/ on purpose: every data/*.csv is
stored in Git LFS (see .gitattributes), and without git-lfs installed a file
there would arrive as a 3-line pointer and break this lookup.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import GRID_SIZE, NEAR_KM

EARTH_RADIUS_KM = 6371.0
TOWNS_FILE = Path(__file__).resolve().parents[1] / "assets" / "towns.json"

# (name, state, latitude, longitude)
TOWNS = [
    (t["name"], t["state"], t["lat"], t["lon"])
    for t in json.loads(TOWNS_FILE.read_text(encoding="utf-8"))
]

_NAMES = [f"{name}, {state}" for name, state, _, _ in TOWNS]
_LAT = np.radians([t[2] for t in TOWNS])
_LON = np.radians([t[3] for t in TOWNS])
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def format_coords(lat: float, lon: float) -> str:
    """-33.5, 151.0 -> '33.5°S, 151.0°E'."""
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return f"{abs(lat):.1f}°{ns}, {abs(lon):.1f}°{ew}"


def _distances_km(lat: float, lon: float) -> np.ndarray:
    """Great-circle distance from (lat, lon) to every town (haversine)."""
    la = np.radians(lat)
    a = (np.sin((_LAT - la) / 2) ** 2
         + np.cos(la) * np.cos(_LAT) * np.sin((_LON - np.radians(lon)) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _bearing(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> str:
    """8-point compass direction from the town to the cell."""
    la1, la2 = np.radians(from_lat), np.radians(to_lat)
    dlon = np.radians(to_lon - from_lon)
    y = np.sin(dlon) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon)
    degrees = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return _COMPASS[int((degrees + 22.5) // 45) % 8]


@lru_cache(maxsize=8192)
def nearest_town(lat: float, lon: float) -> tuple:
    """Returns ("Katherine, NT", distance_km, compass direction from the town)."""
    dist = _distances_km(lat, lon)
    i = int(np.argmin(dist))
    return _NAMES[i], float(dist[i]), _bearing(TOWNS[i][2], TOWNS[i][3], lat, lon)


def describe_location(lat: float, lon: float) -> str:
    """
    "near Katherine, NT" when the cell is close to a town, otherwise
    "40 km NW of Katherine, NT". Distances are rounded to 5 km -- the cell
    is ~55 km across, so more precision would be false precision.
    """
    name, km, direction = nearest_town(float(lat), float(lon))
    if km < NEAR_KM:
        return f"near {name}"
    return f"{int(round(km / 5) * 5)} km {direction} of {name}"


def _grid(value: float, size: float = GRID_SIZE) -> float:
    """Same double-round convention as the prediction pipeline's grid."""
    return round(round(value / size) * size, 3)


def towns_at_risk(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Every listed town whose grid area is in `predictions`, with that area's
    forecast: one row per town, columns [town, lat_round, lon_round,
    risk_level, fire_probability], riskiest first.

    `predictions` should be one forecast day for one region. Towns whose area
    has no forecast (offshore-filtered, or in another region) are left out,
    so the result only covers places the current view actually forecasts.
    """
    cols = ["town", "lat_round", "lon_round", "risk_level", "fire_probability"]
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=cols)

    forecast = {
        (round(float(r.lat_round), 3), round(float(r.lon_round), 3)): (str(r.risk_level), float(r.fire_probability))
        for r in predictions.itertuples(index=False)
    }
    rows = []
    for name, (_, _, lat, lon) in zip(_NAMES, TOWNS):
        key = (_grid(lat), _grid(lon))
        if key in forecast:
            level, prob = forecast[key]
            rows.append((name, key[0], key[1], level, prob))
    out = pd.DataFrame(rows, columns=cols)
    # Stable sort: towns with equal probability keep the list's order.
    return out.sort_values("fire_probability", ascending=False, kind="mergesort").reset_index(drop=True)
