"""
Shared fixtures for the dashboard tests.

The sample predictions mimic what the backend's /predictions/stored endpoint
returns (after utils.api_client has parsed it): one row per grid cell per
forecast day, with cells placed in known states so region filtering can be
checked exactly.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

# (lat, lon) -> the state utils.regions.assign_state() should give it.
CELLS = {
    (-33.0, 147.0): "New South Wales",
    (-37.5, 145.0): "Victoria",
    # Inside both the Victoria and NSW boxes -- Victoria wins on priority.
    (-36.0, 147.0): "Victoria",
    (-35.5, 149.0): "Australian Capital Territory",
    (-30.0, 120.0): "Western Australia",
    # Offshore: outside every state box.
    (-20.0, 160.0): None,
}

TODAY = date.today()
TOMORROW = TODAY + timedelta(days=1)


def _row(lat, lon, day, probability, risk_level, ffdi=10.0, spinup=False):
    return {
        "lat_round": lat,
        "lon_round": lon,
        "acq_date": pd.Timestamp(day),
        "temperature_2m": 25.0,
        "relative_humidity_2m": 30.0,
        "wind_speed_10m": 20.0,
        "ffdi": ffdi,
        "kbdi": 80.0,
        "drought_factor": 6.0,
        "kbdi_spinup_flag": spinup,
        "fire_probability": probability,
        "fire_predicted": int(probability >= 0.5),
        "risk_level": risk_level,
    }


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Two forecast days; tomorrow is clearly worse than today."""
    today = [
        _row(-33.0, 147.0, TODAY, 0.30, "Moderate", ffdi=15.0),
        _row(-37.5, 145.0, TODAY, 0.10, "Low", ffdi=5.0),
        _row(-36.0, 147.0, TODAY, 0.60, "High", ffdi=30.0, spinup=True),
        _row(-35.5, 149.0, TODAY, 0.05, "Low", ffdi=4.0),
        _row(-30.0, 120.0, TODAY, 0.20, "Low", ffdi=8.0),
        _row(-20.0, 160.0, TODAY, 0.10, "Low", ffdi=3.0),
    ]
    tomorrow = [
        _row(-33.0, 147.0, TOMORROW, 0.90, "Extreme", ffdi=60.0),
        _row(-37.5, 145.0, TOMORROW, 0.70, "High", ffdi=35.0),
        _row(-36.0, 147.0, TOMORROW, 0.80, "Extreme", ffdi=55.0),
        _row(-35.5, 149.0, TOMORROW, 0.40, "Moderate", ffdi=20.0),
        _row(-30.0, 120.0, TOMORROW, 0.30, "Moderate", ffdi=14.0),
        _row(-20.0, 160.0, TOMORROW, 0.10, "Low", ffdi=3.0),
    ]
    return pd.DataFrame(today + tomorrow)


@pytest.fixture
def empty_predictions() -> pd.DataFrame:
    from utils.api_client import PREDICTION_COLS

    return pd.DataFrame(columns=PREDICTION_COLS)


@pytest.fixture
def feature_importance() -> pd.DataFrame:
    """Shape returned by api_client.fetch_feature_importance()."""
    return pd.DataFrame(
        [
            {"feature": "lat_round", "importance": 20.0},
            {"feature": "lon_round", "importance": 15.0},
            {"feature": "soil_moisture_0_to_7cm", "importance": 25.0},
            {"feature": "kbdi", "importance": 10.0},
            {"feature": "temperature_2m", "importance": 12.0},
            {"feature": "wind_speed_10m", "importance": 8.0},
            {"feature": "month_sin", "importance": 6.0},
            {"feature": "ffdi", "importance": 4.0},
        ]
    )
