"""
Shapes of the dictionaries the dashboard passes from core/ to components/.

These are TypedDicts: at runtime they are ordinary dicts (so nothing about
how they are read changes), but editors and type checkers can now catch a
misspelt key, and the fields are documented in one place.
"""

from datetime import date
from typing import Dict, Optional, Tuple, TypedDict


class RegionSummary(TypedDict):
    """Headline figures for one region on one day (core.summaries.summarise_region)."""

    has_data: bool
    region: str
    cell_count: int
    at_risk_count: int              # cells at or above the model's alert threshold
    max_risk_level: str             # Low / Moderate / High / Extreme
    max_probability: float
    mean_probability: float
    max_ffdi: float
    mean_ffdi: float
    mean_temp: float
    mean_humidity: float
    mean_wind: float
    max_kbdi: float
    hottest_cell: Optional[Tuple[float, float]]   # (lat, lon) of the highest probability
    risk_counts: Dict[str, int]                   # cells per risk level
    spinup_count: int                             # cells with under 30 days of drought history
    forecast_date: Optional[date]


class DayOption(TypedDict):
    """One forecast day's headline figures (core.summaries.day_options)."""

    date: date
    peak_level: str
    peak_probability: float
    mean_probability: float
    at_risk_count: int
    cell_count: int
    max_ffdi: float


class WeekHighlights(TypedDict):
    """Riskiest and calmest day of the forecast (core.summaries.week_highlights)."""

    worst: DayOption
    best: DayOption
    flat: bool      # True when every day is about the same
