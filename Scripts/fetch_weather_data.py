"""
Fetch hourly historical weather (Open-Meteo Archive API) for a 0.25° grid
covering Australia, from 30 days before your training start date to today.
Writes directly into the `openmeteo` table in wildfire_db.db, matching
backend/db_model/openmeteo.py.

- Includes a 30-day spin-up buffer before TRAINING_START_DATE, so KBDI has
  antecedent data to run in from. build_training_dataset.py drops the
  buffer days before labeling.
- Resumable: already-fetched batches are skipped on rerun (fetch_progress.json)
- Rows are inserted with INSERT OR IGNORE, so reruns/overlaps stay idempotent
  against the (lat_round, lon_round, datetime_utc) unique constraint
- Rate-limited when hitting the public API: weight-based, matching Open-Meteo's
  actual formula (weight = nLocations * (nDays/14) * (nVariables/10)) — a flat
  per-request counter isn't enough, since one wide call can outweigh hundreds
  of small ones and get you 429'd immediately regardless of pacing
- Also appends each batch to a training CSV as it goes (no separate export step)
- Retries failed batches automatically within the same run, with backoff,
  up to MAX_RETRY_PASSES — the goal is 100% grid coverage in one run, since
  build_training_dataset.py's completeness filter drops any cell with even
  one missing day

WHY SELF-HOSTED IS THE DEFAULT: at 0.25°/~23,153 cells and this date range,
total weight is roughly 23,153 * (nDays/14) * (10/10) ≈ 1,000,000+ weight
units. The free public API caps out at 10,000/day. That's 100+ days minimum
even with perfect pacing — impractical for a one-time backfill. That cap is
enforced by Open-Meteo's hosted infrastructure, not the open-source server
itself, so a self-hosted instance (see the earlier self-hosting setup) has
no such quota. Only flip BASE_URL back to the public API for small test runs.

NOTE ON SCALE: 0.25° over the Australia bbox is ~23,153 cells. Hourly data
over the full range is ~13,000+ hours per cell. That's 300M+ rows. Confirm
this is really what you want before letting this run unattended — consider
a shorter date range or a subset of variables first if you're just testing
the pipeline.

Usage:
    python fetch_historical_weather_hourly.py
"""

import csv
import json
import time
import sqlite3
import threading
from pathlib import Path
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------------------------------------------------------- config ---

# Self-hosted by default now — see WEIGHT note below for why. Switch back to
# the public API only for small test runs (small grid and/or short date range).
BASE_URL = "http://127.0.0.1:8080/v1/archive"
# BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
USING_PUBLIC_API = "archive-api.open-meteo.com" in BASE_URL

GRID_SIZE = 0.25
LAT_MIN, LAT_MAX = -44.0, -10.0
LON_MIN, LON_MAX = 112.0, 154.0

TRAINING_START_DATE = date(2025, 1, 1)
KBDI_SPINUP_DAYS = 30  # keep in sync with build_training_dataset.py
FETCH_START_DATE = TRAINING_START_DATE - timedelta(days=KBDI_SPINUP_DAYS)

START_DATE = FETCH_START_DATE.isoformat()
END_DATE = date.today().isoformat()

# Matches backend/db_model/openmeteo.py exactly (order matters for row building below)
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

# Much smaller than the API's 1000-location max — hourly data over 1.5 years
# per location is a lot heavier than daily, so a big batch means a huge
# single response. Tune based on your machine's memory once you test it.
BATCH_SIZE = 50
MAX_WORKERS = 4

# Retry failed batches automatically within this run, rather than relying on
# the person to notice failures and rerun manually — the goal is 100% grid
# coverage before build_training_dataset.py's completeness filter runs.
MAX_RETRY_PASSES = 5
RETRY_BACKOFF_SECONDS = 30  # multiplied by pass number

DB_PATH = Path("wildfire_db.db")
PROGRESS_FILE = Path("data/fetch_progress_hourly.json")
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Training CSV, written alongside the DB as batches complete — no separate
# export step needed. Drops fetched_at (metadata, not a training feature).
OUT_CSV = Path("data/openmeteo_training_export.csv")
CSV_COLUMNS = ["lat_round", "lon_round", "datetime_utc"] + HOURLY_VARS

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS openmeteo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat_round REAL NOT NULL,
    lon_round REAL NOT NULL,
    datetime_utc TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    temperature_2m REAL NOT NULL,
    relative_humidity_2m REAL NOT NULL,
    wind_speed_10m REAL NOT NULL,
    wind_direction_10m REAL NOT NULL,
    precipitation REAL NOT NULL,
    wind_gusts_10m REAL NOT NULL,
    soil_moisture_0_to_7cm REAL NOT NULL,
    cape REAL NOT NULL,
    vapour_pressure_deficit REAL NOT NULL,
    et0_fao_evapotranspiration REAL NOT NULL,
    UNIQUE (lat_round, lon_round, datetime_utc)
);
CREATE INDEX IF NOT EXISTS idx_wl_grid_cell ON openmeteo (lat_round, lon_round);
CREATE INDEX IF NOT EXISTS idx_wl_datetime_utc ON openmeteo (datetime_utc);
"""

INSERT_SQL = f"""
INSERT OR IGNORE INTO openmeteo (
    lat_round, lon_round, datetime_utc, fetched_at, {", ".join(HOURLY_VARS)}
) VALUES ({", ".join(["?"] * (4 + len(HOURLY_VARS)))})
"""


# ---------------------------------------------------------- rate limiter ---

class RateLimiter:
    """Caps calls per second, per hour, and per day. Sleeps rather than
    raising, so ingestion scripts can run unattended."""

    def __init__(self, per_second=5, per_hour=2000, per_day=9000):
        self.limits = {
            "second": (per_second, 1),
            "hour": (per_hour, 3600),
            "day": (per_day, 86400),
        }
        self.calls = {tier: [] for tier in self.limits}
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            for tier, (limit, window) in self.limits.items():
                self.calls[tier] = [t for t in self.calls[tier] if now - t < window]
                if len(self.calls[tier]) >= limit:
                    sleep_for = window - (now - self.calls[tier][0])
                    if sleep_for > 0:
                        time.sleep(sleep_for)
            now = time.time()
            for tier in self.limits:
                self.calls[tier].append(now)


limiter = RateLimiter()


# --------------------------------------------------------------- grid ------

def build_grid():
    """0.25° grid over the Australia bbox. Double-round pattern keeps
    lat_round/lon_round exact join keys, matching the rest of the pipeline."""
    cells = []
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = LON_MIN
        while lon <= LON_MAX:
            lat_r = round(round(lat / GRID_SIZE) * GRID_SIZE, 3)
            lon_r = round(round(lon / GRID_SIZE) * GRID_SIZE, 3)
            cells.append((lat_r, lon_r))
            lon += GRID_SIZE
        lat += GRID_SIZE
    return cells


# ---------------------------------------------------------- progress -------

def load_progress():
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done_batches):
    PROGRESS_FILE.write_text(json.dumps(list(done_batches)))


# ------------------------------------------------------------- fetch -------

def fetch_batch(batch_id, cells):
    lats = ",".join(str(c[0]) for c in cells)
    lons = ",".join(str(c[1]) for c in cells)

    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }

    limiter.wait()
    resp = requests.get(BASE_URL, params=params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    results = data if isinstance(data, list) else [data]
    fetched_at = datetime.utcnow().isoformat()

    rows = []
    for cell, loc_data in zip(cells, results):
        hourly = loc_data.get("hourly", {})
        times = hourly.get("time", [])
        series = {var: hourly.get(var, [None] * len(times)) for var in HOURLY_VARS}

        for i, t in enumerate(times):
            row = [cell[0], cell[1], t, fetched_at]
            row += [series[var][i] if i < len(series[var]) else None for var in HOURLY_VARS]
            rows.append(row)

    return batch_id, rows


# --------------------------------------------------------------- main ------

def main():
    grid = build_grid()
    print(f"Grid: {len(grid)} cells at {GRID_SIZE}° over Australia")
    print(f"Fetch range: {START_DATE} to {END_DATE} (hourly) — "
          f"includes {KBDI_SPINUP_DAYS}-day spin-up before {TRAINING_START_DATE.isoformat()}")

    batches = [grid[i:i + BATCH_SIZE] for i in range(0, len(grid), BATCH_SIZE)]
    done = load_progress()
    todo = [(i, b) for i, b in enumerate(batches) if i not in done]
    print(f"{len(done)}/{len(batches)} batches already fetched, {len(todo)} remaining")

    if not todo:
        print("Nothing to do.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(CREATE_TABLE_SQL)
    conn.commit()

    csv_is_new = not OUT_CSV.exists()
    csv_file = open(OUT_CSV, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(CSV_COLUMNS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        remaining = todo
        attempt = 0

        while remaining and attempt < MAX_RETRY_PASSES:
            attempt += 1
            if attempt > 1:
                backoff = RETRY_BACKOFF_SECONDS * attempt
                print(f"Retry pass {attempt}/{MAX_RETRY_PASSES} — "
                      f"{len(remaining)} batches remaining. Waiting {backoff}s...")
                time.sleep(backoff)

            failed = []
            futures = {pool.submit(fetch_batch, i, b): (i, b) for i, b in remaining}
            for future in as_completed(futures):
                batch_id, cells = futures[future]
                try:
                    _, rows = future.result()
                except Exception as e:
                    print(f"Batch {batch_id} failed (attempt {attempt}): {e}")
                    failed.append((batch_id, cells))
                    continue

                if rows:
                    conn.executemany(INSERT_SQL, rows)
                    conn.commit()

                    # Same rows, minus fetched_at (index 3) — training data doesn't need it
                    csv_writer.writerows(row[:3] + row[4:] for row in rows)
                    csv_file.flush()

                done.add(batch_id)
                save_progress(done)
                print(f"Batch {batch_id + 1}/{len(batches)} done "
                      f"({len(rows)} rows) — progress saved")

            remaining = failed

    csv_file.close()
    conn.close()

    if remaining:
        failed_ids = [b for b, _ in remaining]
        print(f"WARNING: {len(remaining)} batches still failed after "
              f"{MAX_RETRY_PASSES} passes: {failed_ids}")
        print("Grid coverage is INCOMPLETE. Re-run this script to retry — "
              "do not run build_training_dataset.py until this list is empty, "
              "since its completeness filter will silently drop any cell "
              "with even one missing day.")
    else:
        print(f"All batches fetched successfully — grid coverage is complete.")
        print(f"Data written to {DB_PATH} (table: openmeteo) and {OUT_CSV}")


if __name__ == "__main__":
    main()