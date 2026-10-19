"""
Everyday-language versions of the dashboard's numbers.

The model works in 0.5-degree grid cells; people think in shares ("1 in 8"),
land area ("larger than Victoria") and plain weather words. These helpers
translate. (Named fire-danger bands live in core.danger.)
"""

import math

from config import GRID_SIZE

KM_PER_DEGREE = 111.32

# Land area of each state and territory, km^2 (Geoscience Australia, rounded).
STATE_AREAS_KM2 = [
    ("Western Australia", 2_527_013),
    ("Queensland", 1_730_648),
    ("the Northern Territory", 1_347_791),
    ("South Australia", 984_321),
    ("New South Wales", 800_642),
    ("Victoria", 227_444),
    ("Tasmania", 68_401),
    ("the ACT", 2_358),
]


def one_in(share: float) -> str:
    """0.13 -> '1 in 8', 0.6 -> '6 in 10', 0 -> 'none'."""
    if share <= 0:
        return "none"
    if share >= 0.5:
        return f"{round(share * 10)} in 10"
    return f"1 in {max(2, round(1 / share))}"


def cell_area_km2(lat: float) -> float:
    """Area of one 0.5-degree grid cell. Cells shrink away from the equator:
    about 3,000 km^2 near Darwin and 2,300 km^2 in Tasmania."""
    side = GRID_SIZE * KM_PER_DEGREE
    return side * side * math.cos(math.radians(lat))


def format_area(km2: float) -> str:
    """Rounded to two significant figures -- the cells are coarse, so more
    digits would suggest precision the forecast doesn't have."""
    if km2 <= 0:
        return "0 km²"
    if km2 >= 1_000_000:
        return f"{km2 / 1_000_000:.1f} million km²"
    digits = int(math.floor(math.log10(km2)))
    return f"{round(km2, -(digits - 1)):,.0f} km²"


def area_comparison(km2: float) -> str:
    """
    A familiar size to compare an area with: 'about the size of Victoria'
    (within 10%), 'larger than New South Wales', or '' when it is smaller
    than the ACT.
    """
    for name, size in STATE_AREAS_KM2:
        if abs(km2 - size) <= 0.1 * size:
            return f"about the size of {name}"
    for name, size in STATE_AREAS_KM2:
        if km2 > size:
            return f"larger than {name}"
    return ""


def weather_words(row) -> str:
    """'hot (31°C), very dry air (14%), fresh wind (25 km/h)' for a prediction row."""
    t, rh, w = row["temperature_2m"], row["relative_humidity_2m"], row["wind_speed_10m"]
    heat = "very hot" if t >= 30 else "hot" if t >= 25 else "warm" if t >= 21 else "mild" if t >= 15 else "cool"
    air = ("very dry air" if rh <= 20 else "dry air" if rh <= 30 else "fairly dry air" if rh <= 40
           else "humid air" if rh >= 70 else "moderate humidity")
    wind = "strong wind" if w >= 35 else "fresh wind" if w >= 22 else "breezy" if w >= 15 else "light wind"
    return f"{heat} ({t:.0f}°C), {air} ({rh:.0f}%), {wind} ({w:.0f} km/h)"
