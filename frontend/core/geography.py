"""
Regions: which state each grid cell belongs to, and how the map frames it.

The backend predicts per 0.5-degree grid cell; the UI selects by state. This
module bridges the two with approximate state bounding boxes.

The bounding boxes are rectangles, not real state borders, so cells near a
border may be counted in the neighbouring state. That is acceptable for a
risk overview -- the map still shows every cell in its true position -- but
it is not suitable for official reporting.
"""

from typing import Dict, Optional

import pandas as pd

NATIONAL = "Australia (National)"

# Options of the region selector, in display order.
REGIONS = [
    NATIONAL,
    "New South Wales",
    "Victoria",
    "Queensland",
    "Western Australia",
    "South Australia",
    "Tasmania",
    "Northern Territory",
    "Australian Capital Territory",
]

# State codes used in the town list (assets/towns.json).
STATE_CODES = {
    "New South Wales": "NSW",
    "Victoria": "VIC",
    "Queensland": "QLD",
    "Western Australia": "WA",
    "South Australia": "SA",
    "Tasmania": "TAS",
    "Northern Territory": "NT",
    "Australian Capital Territory": "ACT",
}
STATE_NAMES = {code: name for name, code in STATE_CODES.items()}

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


def assign_states(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    assign_state() for whole columns at once: the state of every cell, or
    None. Same rules and priority order, but vectorised -- checking ~20,000
    rows one by one on every page interaction was the slowest step of the
    region filter.
    """
    states = pd.Series([None] * len(lat), index=lat.index, dtype=object)
    unassigned = pd.Series(True, index=lat.index)
    for state in STATE_PRIORITY:
        lat_min, lat_max, lon_min, lon_max = STATE_BOUNDS[state]
        inside = unassigned & lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
        states[inside] = state
        unassigned &= ~inside
    return states


def filter_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Restricts predictions to the grid cells assigned to a state.

    Uses the state assignment rather than a plain bounding-box test, so a
    cell near a state border appears under one state only.
    """
    if df.empty or is_national(region):
        return df
    if region not in STATE_BOUNDS:
        return df
    assigned = assign_states(df["lat_round"], df["lon_round"])
    return df[assigned == region].reset_index(drop=True)


def get_region_view(region: str) -> tuple:
    """Returns ([lat, lon], zoom) for the map."""
    if is_national(region):
        return REGION_VIEWS[NATIONAL]
    return REGION_VIEWS.get(region, REGION_VIEWS[NATIONAL])
