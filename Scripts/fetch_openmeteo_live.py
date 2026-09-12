"""
fetch_live_weather_grid.py (v2 - concurrent)

Purpose:
Fetch a 7-day (168-hour) weather forecast from Open-Meteo for a FIXED
GRID covering Australia — independent of FIRMS hotspot locations.

What changed from the sequential version:
The 600/minute rate limit actually allows ~10 requests PER SECOND, not
1 every 1-2 seconds. But that's not the only ceiling — Open-Meteo also
caps at 5,000/hour and 10,000/day. Fetching too fast just means you hit
the HOURLY or DAILY wall sooner, not that you avoid it. This version:
  - Fires several requests CONCURRENTLY (thread pool) for real speed.
  - Uses a rate limiter that tracks second, hour, AND day windows at
    once, automatically pausing (sleeping) whenever any window fills
    up, then resuming automatically once there's room again.
This means a grid under ~4,800 points finishes in minutes at near-max
speed. A larger grid (e.g. the full 5,865-point country-wide grid)
will burst fast, briefly pause once the hourly window fills, then
continue automatically — no manual reruns needed, no failures, just a
built-in wait.

RESUMABLE: same progress-log pattern as before — if interrupted, just
run it again; it skips everything already done.

Concurrency safety:
- A rate limiter (token-bucket style) ensures the TOTAL request rate
  across all worker threads stays under MAX_REQUESTS_PER_SECOND.
- Writes to the output CSV, the progress log, and the SQLite database
  are all protected by locks, since SQLite in particular doesn't handle
  truly concurrent writes well.

Before running:
    pip install openmeteo-requests requests-cache retry-requests pandas sqlalchemy
"""

import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

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
GRID_POINTS_CSV_PATH = "data/australia_grid_points.csv"
PROGRESS_LOG_PATH = "data/live_weather_grid_progress.log"
DB_URL = "sqlite:///data/wildfire_db.db"
FORECAST_DAYS = 7

AUS_LAT_MIN, AUS_LAT_MAX = -44, -10
AUS_LON_MIN, AUS_LON_MAX = 112, 154
GRID_SIZE_DEGREES = 0.5

CALL_LIMIT_PER_RUN = 9000
MAX_WORKERS = 6                 # reduced slightly to leave headroom for other concurrent scripts
MAX_REQUESTS_PER_SECOND = 6     # leaves room for fetch_live_firms/fetch_live_weather running concurrently
MAX_REQUESTS_PER_HOUR = 4000    # more headroom below the real 5,000/hour limit
MAX_REQUESTS_PER_DAY = 9000     # more headroom below the real 10,000/day limit

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

# engine = create_engine(DB_URL)
# Base.metadata.create_all(engine)

csv_lock = threading.Lock()
log_lock = threading.Lock()
db_lock = threading.Lock()


class RateLimiter:
    """Ensures the TOTAL request rate across all threads stays under
    limits at THREE different time windows simultaneously (second, hour,
    day) — matching Open-Meteo's actual tiered limits. This lets the
    fetch run as fast as genuinely allowed: it bursts quickly for the
    first few thousand calls, then automatically SLOWS ITSELF DOWN (by
    sleeping, not failing) once the hourly or daily window fills up,
    resuming full speed once that window has room again. No manual
    reruns needed even for grids larger than 5,000 points."""

    def __init__(self, per_second, per_hour, per_day):
        self.limits = [
            (per_second, 1),
            (per_hour, 3600),
            (per_day, 86400),
        ]
        self.lock = threading.Lock()
        self.timestamps = {window: deque() for _, window in self.limits}
        self.external_pause_until = 0   # monotonic time; set when the SERVER itself says "limit exceeded"

    def trigger_external_pause(self, seconds):
        """Call this when the API itself returns a rate-limit error — this
        means the REAL server-side quota is exhausted, possibly because
        another script (e.g. the 15-minute live scheduler) is also using
        the same account/IP at the same time. Pausing ALL threads here
        (not just skipping one point) avoids hammering the API with more
        requests that would just fail the same way."""
        with self.lock:
            new_until = time.monotonic() + seconds
            if new_until > self.external_pause_until:
                self.external_pause_until = new_until
                print(f"\n[Rate limiter] Server reported limit exceeded. "
                      f"Pausing ALL requests for {seconds // 60} minutes, then resuming automatically.\n")

    def acquire(self):
        while True:
            with self.lock:
                pause_remaining = self.external_pause_until - time.monotonic()
            if pause_remaining > 0:
                time.sleep(min(pause_remaining, 5))
                continue

            with self.lock:
                now = time.monotonic()
                wait_time = 0

                for max_count, window in self.limits:
                    ts = self.timestamps[window]
                    while ts and now - ts[0] > window:
                        ts.popleft()
                    if len(ts) >= max_count:
                        wait_time = max(wait_time, window - (now - ts[0]))

                if wait_time <= 0:
                    for _, window in self.limits:
                        self.timestamps[window].append(now)
                    return

            # sleep OUTSIDE the lock so other threads aren't blocked
            # from checking/acquiring while this one waits
            time.sleep(min(wait_time, 5))


rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND, MAX_REQUESTS_PER_HOUR, MAX_REQUESTS_PER_DAY)


def build_australia_grid(grid_size):
    lats = np.arange(AUS_LAT_MIN, AUS_LAT_MAX + grid_size, grid_size)
    lons = np.arange(AUS_LON_MIN, AUS_LON_MAX + grid_size, grid_size)
    return [(round(lat, 3), round(lon, 3)) for lat in lats for lon in lons]


def get_grid_points(grid_size):
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


def point_key(lat, lon):
    return f"{lat},{lon}"


def load_completed_points():
    if not os.path.exists(PROGRESS_LOG_PATH):
        return set()
    with open(PROGRESS_LOG_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_point_complete(key):
    with log_lock:
        with open(PROGRESS_LOG_PATH, "a") as f:
            f.write(key + "\n")


def fetch_forecast(lat, lon):
    """Rate-limited fetch — every call goes through the shared limiter
    first, regardless of which thread is calling it."""
    rate_limiter.acquire()

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


def append_to_output_csv(df):
    with csv_lock:
        file_exists = os.path.exists(OUTPUT_CSV_PATH)
        df.to_csv(OUTPUT_CSV_PATH, mode="a", header=not file_exists, index=False)


def save_point_to_db(forecast_df, lat, lon):
    with db_lock:
        with engine.begin() as conn:
            conn.execute(
                delete(WeatherLive).where(
                    WeatherLive.lat_round == lat,
                    WeatherLive.lon_round == lon,
                )
            )
        forecast_df.to_sql(WeatherLive.__tablename__, engine, if_exists="append", index=False)


def fetch_and_save_point(lat, lon):
    """The full unit of work for one grid point — run inside a worker
    thread. Returns (lat, lon, success, error_message_or_None)."""
    try:
        forecast_df = fetch_forecast(lat, lon)
        append_to_output_csv(forecast_df)
        # save_point_to_db(forecast_df, lat, lon)
        mark_point_complete(point_key(lat, lon))
        return (lat, lon, True, None)
    except Exception as e:
        error_text = str(e)
        # Detect the server telling us the REAL quota is exhausted (as
        # opposed to a random network error) — this can happen even when
        # our own tracking thinks there's room, e.g. if another script
        # (the 15-min live scheduler) is using the same account/IP too.
        if "limit exceeded" in error_text.lower() or "rate limit" in error_text.lower():
            rate_limiter.trigger_external_pause(3600)  # pause ~1 hour, matches Open-Meteo's hourly window
        return (lat, lon, False, error_text)


def main():
    grid_points = get_grid_points(GRID_SIZE_DEGREES)
    print(f"Total grid points: {len(grid_points)} (grid size: {GRID_SIZE_DEGREES} degrees).")

    completed = load_completed_points()
    print(f"Already completed (from previous runs): {len(completed)}")

    remaining_points = [
        (lat, lon) for lat, lon in grid_points if point_key(lat, lon) not in completed
    ]
    print(f"Remaining points to fetch: {len(remaining_points)}")

    if not remaining_points:
        print("Nothing left to fetch — this cycle is complete.")
        print(f"To force a full refresh, delete {PROGRESS_LOG_PATH} and run again.")
        return

    batch = remaining_points[:CALL_LIMIT_PER_RUN]
    print(f"Fetching {len(batch)} points this run, using {MAX_WORKERS} concurrent workers "
          f"(rate-limited to {MAX_REQUESTS_PER_SECOND}/sec, {MAX_REQUESTS_PER_HOUR}/hour, "
          f"{MAX_REQUESTS_PER_DAY}/day)...")

    pending = list(batch)
    max_retry_rounds = 5   # in case of repeated hourly-limit hits, keep retrying automatically
    round_num = 0

    while pending and round_num < max_retry_rounds:
        round_num += 1
        succeeded = 0
        failed_points = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_and_save_point, lat, lon): (lat, lon) for lat, lon in pending}

            for i, future in enumerate(as_completed(futures), start=1):
                lat, lon, success, error = future.result()
                if success:
                    succeeded += 1
                else:
                    failed_points.append((lat, lon))
                    print(f"FAILED for ({lat}, {lon}): {error}")

                if i % 50 == 0 or i == len(pending):
                    done_total = len(completed) + succeeded
                    print(f"Progress: {i}/{len(pending)} this round "
                          f"({succeeded} succeeded, {len(failed_points)} failed)")

        completed = load_completed_points()  # refresh — other threads/workers updated the log
        print(f"\nRound {round_num} complete. {succeeded} succeeded, {len(failed_points)} failed.")

        pending = failed_points
        if pending:
            print(f"{len(pending)} points still failing — retrying automatically "
                  f"(round {round_num + 1}/{max_retry_rounds})...")

    done_total = len(load_completed_points())
    print(f"\nOverall progress: {done_total}/{len(grid_points)} points done.")

    if done_total >= len(grid_points):
        print(f"All grid points fetched! Data saved to {OUTPUT_CSV_PATH} and the "
              f"'{WeatherLive.__tablename__}' table.")
    elif pending:
        print(f"{len(pending)} points still failing after {max_retry_rounds} rounds — "
              f"run this script again later to keep retrying.")
    else:
        print("Run this script again to continue fetching the rest.")


if __name__ == "__main__":
    main()