"""
fetch_live_firms.py

Purpose:
Fetch NEAR-REAL-TIME fire hotspot data for Australia from NASA FIRMS,
using the Area API (different from the bulk Archive Download used for
your 2-year historical dataset). This is meant to be run repeatedly
(e.g. every 15 minutes, matching your proposal's architecture) to feed
live data into the dashboard/prediction pipeline — NOT for building
training data.

Endpoint pattern:
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]

- No DATE given -> returns most recent data (today back DAY_RANGE-1 days)
- AREA_COORDINATES must be: west,south,east,north (a bounding box)
- DAY_RANGE: 1-10. Use 1 for "just the latest available data".

Rate limit: 5,000 transactions per 10-minute window. Requesting more than
1 day counts as multiple transactions, so keep DAY_RANGE small for
frequent polling (e.g. every 15 minutes).

Before running:
    pip install pandas requests python-dotenv
    Put your FIRMS MAP_KEY in a .env file as: FIRMS_MAP_KEY=your_key_here
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

# from backend.db_model.base import Base
# from backend.db_model.firms_live import FirmsLive

load_dotenv()

# ---------- CONFIG ----------
MAP_KEY = os.getenv("FIRMS_MAP_KEY")
SOURCE = (
    "VIIRS_SNPP_NRT"  # matches the VIIRS SNPP sensor used in your historical dataset
)
DAY_RANGE = 1  # 1 = most recent available data only

# Bounding box for mainland Australia + Tasmania: west,south,east,north
AUSTRALIA_BBOX = "112,-44,154,-10"

OUTPUT_CSV_PATH = "data/live_hotspots.csv"
DB_URL = "sqlite:///data/wildfire_db.db"  # <-- change if using Postgres/MySQL etc
GRID_SIZE_DEGREES = (
    0.5  # matches your historical grid, for consistent joins with live weather
)


def round_to_grid(value, grid_size):
    return round(round(value / grid_size) * grid_size, 3)


def fetch_live_hotspots():
    if not MAP_KEY:
        raise ValueError("FIRMS_MAP_KEY not found. Check your .env file.")

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{SOURCE}/{AUSTRALIA_BBOX}/{DAY_RANGE}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # The API returns raw CSV text directly in the response body
    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    if df.empty:
        print("No active hotspots returned for this window.")
        return df

    df["lat_round"] = df["latitude"].apply(
        lambda v: round_to_grid(v, GRID_SIZE_DEGREES)
    )
    df["lon_round"] = df["longitude"].apply(
        lambda v: round_to_grid(v, GRID_SIZE_DEGREES)
    )
    df["fetched_at"] = pd.Timestamp.now(tz="UTC")

    return df


def save_to_db(df):
    """Insert new hotspot rows into firms_live, skipping any that already
    exist (matched on latitude, longitude, acq_date, acq_time, satellite —
    same key as the table's unique constraint). Safe to call every run."""
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)  # creates firms_live table if it doesn't exist yet

    key_cols = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]

    existing = pd.read_sql(
        f"SELECT {', '.join(key_cols)} FROM {FirmsLive.__tablename__}", engine
    )

    if not existing.empty:
        merged = df.merge(existing, on=key_cols, how="left", indicator=True)
        new_rows = df[merged["_merge"] == "left_only"].copy()
    else:
        new_rows = df.copy()

    if new_rows.empty:
        print("No new hotspots to insert (all already in the database).")
        return

    insert_df = new_rows.rename(
        columns={
            "brightness": "bright_ti4",
            "bright_t31": "bright_ti5",
        }
    )
    insert_df["acq_time"] = insert_df["acq_time"].astype(int)

    model_columns = [
        "latitude",
        "longitude",
        "lat_round",
        "lon_round",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "frp",
        "daynight",
        "fetched_at",
    ]
    # Only keep columns that actually exist in this response (some FIRMS
    # sources omit certain fields)
    model_columns = [c for c in model_columns if c in insert_df.columns]

    insert_df[model_columns].to_sql(
        FirmsLive.__tablename__, engine, if_exists="append", index=False
    )
    print(
        f"Inserted {len(insert_df)} new hotspot rows into '{FirmsLive.__tablename__}'."
    )


def main():
    print(f"Fetching live FIRMS data for Australia (last {DAY_RANGE} day(s))...")
    df = fetch_live_hotspots()

    if df.empty:
        return

    print(f"Retrieved {len(df)} active hotspot records.")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved to {OUTPUT_CSV_PATH}")

    # save_to_db(df)

    # Quick sanity check: show confidence breakdown
    if "confidence" in df.columns:
        print("\nConfidence breakdown:")
        print(df["confidence"].value_counts())


if __name__ == "__main__":
    main()
COUNTRY_CODE = "AUS"
QUERY_METHOD = country
