"""
fetch_live_weather_grid.py

Purpose:
Fetch a 7-day (168-hour) weather forecast from Open-Meteo for a FIXED
GRID covering all of Australia — independent of FIRMS hotspot locations.
Use this if you want general weather coverage for the dashboard (e.g. a
country-wide risk map) even when there are no active fires, rather than
only fetching weather where fires currently exist (fetch_live_weather.py).

Rate limit note (important):
Unlike fetch_live_weather.py (which only fetches for a handful of active
hotspot locations), this generates MANY grid points across the whole
country. At GRID_SIZE_DEGREES = 1.0 (~111km cells), Australia's bounding
box produces roughly 1,400 points — each needing 1 API call.

If you run this every 15 minutes (like the hotspot-based live pipeline),
that's ~1,400 calls every 15 min = ~5,600/hour, close to or over the
5,000/hour limit. Two ways to stay safe:
  1. Use a COARSER grid (e.g. 2.0 degrees, ~350 points) if running every
     15 minutes, OR
  2. Keep this grid fine (1.0 or 0.5 degrees) but run it LESS often
     (e.g. every 1-3 hours) — forecasts don't change drastically in
     15-minute windows anyway, so this is usually the better trade-off.

This script does NOT include daily-budget/resume logic like the historical
fetch script, since a single run at a reasonable grid size comfortably
fits under the daily limit in one go.

Before running:
    pip install openmeteo-requests requests-cache retry-requests pandas sqlalchemy
"""

import os

import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sqlalchemy import create_engine, delete

# from backend.db_model.base import Base
# from backend.db_model.weather_live import WeatherLive

# ---------- CONFIG ----------
OUTPUT_CSV_PATH = "data/live_weather_grid.csv"
GRID_POINTS_CSV_PATH = "data/australia_grid_points.csv"   # cached grid, built once
DB_URL = "sqlite:///data/wildfire_db.db"
FORECAST_DAYS = 7

# Australia bounding box (mainland + Tasmania)
AUS_LAT_MIN, AUS_LAT_MAX = -44, -10
AUS_LON_MIN, AUS_LON_MAX = 112, 154

GRID_SIZE_DEGREES = 0.5   # see rate-limit note above before changing this

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "wind_gusts_10m",
    "soil_moisture_0_to_7cm",
    "cape",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

cache_session = requests_cache.CachedSession(".cache", expire_after=900)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def build_australia_grid(grid_size):
    """Generate a fixed grid of (lat, lon) points covering Australia's
    bounding box, spaced grid_size degrees apart."""
    lats = np.arange(AUS_LAT_MIN, AUS_LAT_MAX + grid_size, grid_size)
    lons = np.arange(AUS_LON_MIN, AUS_LON_MAX + grid_size, grid_size)
    grid = [(round(lat, 3), round(lon, 3)) for lat in lats for lon in lons]
    return grid


def get_grid_points(grid_size):
    """Load the cached grid from disk if it exists, otherwise build it
    once and save it — so the grid only needs to be computed one time,
    not regenerated on every run."""
    if os.path.exists(GRID_POINTS_CSV_PATH):
        grid_df = pd.read_csv(GRID_POINTS_CSV_PATH)
        print(f"Loaded cached grid from {GRID_POINTS_CSV_PATH} ({len(grid_df)} points).")
        return list(zip(grid_df["lat"], grid_df["lon"]))

    print("No cached grid found — building it now...")
    grid = build_australia_grid(grid_size)
    grid_df = pd.DataFrame(grid, columns=["lat", "lon"])
    grid_df.to_csv(GRID_POINTS_CSV_PATH, index=False)
    print(f"Saved grid to {GRID_POINTS_CSV_PATH} ({len(grid)} points) for future runs.")
    return grid


def fetch_forecast(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }
    responses = openmeteo.weather_api(FORECAST_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "datetime_utc": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, var_name in enumerate(HOURLY_VARS):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(hourly_data)
    df["lat_round"] = lat
    df["lon_round"] = lon
    df["fetched_at"] = pd.Timestamp.now(tz="UTC")
    return df


def save_to_db(forecast_df, grid_points):
    """Replace old forecast rows for these grid points with fresh ones."""
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        for lat, lon in grid_points:
            conn.execute(
                delete(WeatherLive).where(
                    WeatherLive.lat_round == lat,
                    WeatherLive.lon_round == lon,
                )
            )

    forecast_df.to_sql(WeatherLive.__tablename__, engine, if_exists="append", index=False)
    print(f"Refreshed forecast for {len(grid_points)} grid points "
          f"({len(forecast_df)} total forecast rows) in '{WeatherLive.__tablename__}'.")


def main():
    grid_points = get_grid_points(GRID_SIZE_DEGREES)
    print(f"Using {len(grid_points)} grid points over Australia "
          f"(grid size: {GRID_SIZE_DEGREES} degrees).")

    all_forecasts = []
    for i, (lat, lon) in enumerate(grid_points):
        try:
            forecast_df = fetch_forecast(lat, lon)
            all_forecasts.append(forecast_df)
        except Exception as e:
            print(f"FAILED for ({lat}, {lon}): {e}")

        if (i + 1) % 50 == 0:
            print(f"Fetched {i + 1}/{len(grid_points)} grid points")

    combined_df = pd.concat(all_forecasts, ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(combined_df)} forecast rows to {OUTPUT_CSV_PATH}")

    # save_to_db(combined_df, grid_points)


if __name__ == "__main__":
    main()