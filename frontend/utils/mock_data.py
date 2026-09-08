"""
Mock data for the wildfire prediction platform.

This is the file you'd replace to go live: swap `STATE_DATA` for a function
that calls a real hotspot feed (e.g. NASA FIRMS) and a weather feed (e.g.
the Bureau of Meteorology), keeping the same field names so no other file
needs to change.
"""

from typing import Any, Dict

STATE_CENTERS: Dict[str, Dict[str, Any]] = {
    "New South Wales": {"lat": -32.93, "lon": 151.78, "zoom": 7},
    "Victoria": {"lat": -37.87, "lon": 147.63, "zoom": 7},
    "Queensland": {"lat": -28.06, "lon": 152.99, "zoom": 7},
    "Western Australia": {"lat": -25.5, "lon": 122.5, "zoom": 5},
    "South Australia": {"lat": -30.0, "lon": 135.5, "zoom": 5},
    "Tasmania": {"lat": -42.0, "lon": 146.8, "zoom": 7},
    "Northern Territory": {"lat": -19.5, "lon": 133.5, "zoom": 5},
    "Australian Capital Territory": {"lat": -35.5, "lon": 149.0, "zoom": 9},
}

STATE_DATA: Dict[str, Dict[str, Any]] = {
    "New South Wales": {
        "primary_cluster": "Hunter Valley Sector",
        "hotspot_count": 80,
        "risk_level": "Extreme",
        "risk_reason": "Extreme spread probability due to low humidity.",
        "peak_intensity_kw_m": 9500,
        "wind_dir": "South-West",
        "wind_speed_kmh": 45,
        "spread_dir_deg": 135,
        "spread_dir_label": "South-East",
        "growth_rate_km2_hr": 2.4,
        "containment_pct": 15,
        "est_hours_to_containment": 36,
        "confidence_pct": 78,
        "ffdi": 62,
        "ffdi_label": "Extreme",
        "temp_c": 34,
        "humidity_pct": 14,
        "structures_at_risk": 340,
        "population_at_risk": 1250,
        "nearest_town": "Newcastle",
        "distance_to_town_km": 18,
        "last_year_comparison": "+38% vs. same week last year",
        "center": (-32.93, 151.78),
        "heat_points": [
            (-32.93, 151.78, 1.0),
            (-32.95, 151.80, 0.8),
            (-32.90, 151.75, 0.7),
            (-33.00, 151.70, 0.5),
            (-32.85, 151.85, 0.4),
        ],
        "risk_trend_pred": [58, 63, 68, 74, 80, 86, 91],
        "risk_trend_obs": [55, 60, 65, 70, 76, 82, 88],
        "authority": "NSW Rural Fire Service",
        "authority_url": "https://www.rfs.nsw.gov.au",
    },
    "Victoria": {
        "primary_cluster": "Gippsland Ranges",
        "hotspot_count": 34,
        "risk_level": "High",
        "risk_reason": "High spread probability, moderate humidity.",
        "peak_intensity_kw_m": 6200,
        "wind_dir": "North",
        "wind_speed_kmh": 30,
        "spread_dir_deg": 160,
        "spread_dir_label": "South",
        "growth_rate_km2_hr": 1.1,
        "containment_pct": 40,
        "est_hours_to_containment": 20,
        "confidence_pct": 82,
        "ffdi": 41,
        "ffdi_label": "High",
        "temp_c": 29,
        "humidity_pct": 22,
        "structures_at_risk": 90,
        "population_at_risk": 410,
        "nearest_town": "Bairnsdale",
        "distance_to_town_km": 26,
        "last_year_comparison": "-6% vs. same week last year",
        "center": (-37.87, 147.63),
        "heat_points": [
            (-37.87, 147.63, 0.9),
            (-37.90, 147.60, 0.6),
            (-37.83, 147.68, 0.5),
        ],
        "risk_trend_pred": [38, 44, 50, 56, 62, 68, 74],
        "risk_trend_obs": [35, 41, 47, 52, 58, 64, 70],
        "authority": "Country Fire Authority (CFA)",
        "authority_url": "https://www.cfa.vic.gov.au",
    },
    "Queensland": {
        "primary_cluster": "Scenic Rim Corridor",
        "hotspot_count": 12,
        "risk_level": "Moderate",
        "risk_reason": "Moderate risk, recent rainfall reduced fuel dryness.",
        "peak_intensity_kw_m": 3100,
        "wind_dir": "East",
        "wind_speed_kmh": 18,
        "spread_dir_deg": 260,
        "spread_dir_label": "West",
        "growth_rate_km2_hr": 0.4,
        "containment_pct": 65,
        "est_hours_to_containment": 10,
        "confidence_pct": 74,
        "ffdi": 22,
        "ffdi_label": "Moderate",
        "temp_c": 26,
        "humidity_pct": 38,
        "structures_at_risk": 12,
        "population_at_risk": 60,
        "nearest_town": "Beaudesert",
        "distance_to_town_km": 14,
        "last_year_comparison": "-20% vs. same week last year",
        "center": (-28.06, 152.99),
        "heat_points": [(-28.06, 152.99, 0.5), (-28.10, 153.02, 0.3)],
        "risk_trend_pred": [15, 18, 22, 26, 30, 34, 38],
        "risk_trend_obs": [13, 16, 20, 24, 27, 31, 35],
        "authority": "Queensland Fire Department",
        "authority_url": "https://www.qfes.qld.gov.au",
    },
}

DEFAULT_STATE: Dict[str, Any] = {
    "primary_cluster": "No active cluster",
    "hotspot_count": 0,
    "risk_level": "Low",
    "risk_reason": "No significant fire activity detected.",
    "peak_intensity_kw_m": 0,
    "wind_dir": "—",
    "wind_speed_kmh": 0,
    "spread_dir_deg": 0,
    "spread_dir_label": "—",
    "growth_rate_km2_hr": 0,
    "containment_pct": 100,
    "est_hours_to_containment": 0,
    "confidence_pct": 0,
    "ffdi": 5,
    "ffdi_label": "Low",
    "temp_c": 22,
    "humidity_pct": 55,
    "structures_at_risk": 0,
    "population_at_risk": 0,
    "nearest_town": "—",
    "distance_to_town_km": 0,
    "last_year_comparison": "No comparable data",
    "center": None,
    "heat_points": [],
    "risk_trend_pred": [5, 5, 6, 5, 6, 5, 5],
    "risk_trend_obs": [4, 5, 5, 4, 5, 4, 5],
    "authority": "State Emergency Service",
    "authority_url": "https://www.ses.gov.au",
}

# Feature importance is a property of the trained model, not the region, so
# it's kept as a single constant rather than duplicated per state.
FACTOR_WEIGHTS = [
    ("Fuel dryness", 32, "flame"),
    ("Wind speed", 24, "amber"),
    ("Temperature", 19, "gold"),
    ("Humidity", 14, "blue"),
    ("Ignition history", 11, "teal"),
]

# Generic seasonal reference pattern (not region-specific — see note in UI)
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
HISTORICAL_INCIDENTS = [8, 12, 20, 32, 48, 68, 78, 80, 66, 34, 20, 12]


def get_state_data(state_name: str) -> Dict[str, Any]:
    """Return the mock data dict for a state, falling back to a calm
    'no activity' placeholder for states we haven't authored detailed
    mock data for yet."""
    return STATE_DATA.get(state_name, DEFAULT_STATE)
