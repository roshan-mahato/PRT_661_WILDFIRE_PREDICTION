"""
================================================================================
fetch_weather_live.py  --  forecast + predictions for every LAND grid cell
================================================================================
Fetches the Open-Meteo hourly forecast for the 0.5-degree Australia grid,
turns it into daily rows and stores them as ONE forecast run:

    fetch_run (source='forecast')  ->  weather_daily  ->  feature_engineered
                                                      ->  prediction

then builds the features and predictions for that run and makes them the
current predictions the API serves.

Nothing is overwritten: every run adds its own rows, so the forecast issued
yesterday for Friday is still there next to today's forecast for Friday.

WHAT IS FETCHED
---------------
- Land cells only: ocean cells come from data/full_grid_land_mask.csv (built
  by fetch_weather_history.py; probed here on first use if missing). The
  full grid is 5,865 cells; land is ~2,861.
- From the day after the archive history ends (weather_daily, source
  'archive') to today + FORECAST_DAYS - 1, hourly. KBDI continues from the
  last archive feature row, so the days in between have to be fetched too.
  Normally the archive ends yesterday and this is just the 7-day forecast.
  Cells with no history at all cold-start COLD_START_DAYS back.
- The 9 model variables (up to 10 variables weigh the same).

NO EMPTY VALUES REACH THE DATABASE
----------------------------------
Short hour gaps are interpolated (max 6 h), physically impossible readings
are dropped, and any day without all 24 hours left is dropped before
aggregating. A cell left with under MIN_HOURS_PER_CELL hours is not saved;
its previous predictions stay current.

API COST
--------
Up to 14 days and 10 variables cost 1 call per cell, so a refresh is
~2,861 calls however cells are batched. Batching (BATCH_SIZE cells per
request) only cuts HTTP overhead. The default is the self-hosted server
(docker-compose.yml: `docker compose up -d`), which has no quota. On the
public API (600/min, 5,000/hour, 10,000/day) requests are paced: a refresh
takes ~5 min and fits 3 times a day.

USAGE
-----
    uv run python Scripts/fetch_weather_live.py                  # fetch + predict
    uv run python Scripts/fetch_weather_live.py --bbox -34 -33 150 152 --no-db   # quick test
    uv run python Scripts/fetch_weather_live.py --no-predict     # weather only
    OPENMETEO_FORECAST_URL=https://api.open-meteo.com/v1/forecast \\
        uv run python Scripts/fetch_weather_live.py --workers 1  # public API
================================================================================
"""

import argparse
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "Scripts"))

# Grid, land mask, pacing and the fetch itself are shared with the history
# script so both use exactly the same cells and rate-limit handling.
import fetch_weather_history as hist  # noqa: E402
from Scripts.prediction_engine import aggregate_to_daily, drop_implausible_readings  # noqa: E402

# ---------- CONFIG ----------
# FORECAST_URL = os.environ.get("OPENMETEO_FORECAST_URL", "http://127.0.0.1:8080/v1/forecast")
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'


ARCHIVE_URL = os.environ.get("OPENMETEO_ARCHIVE_URL", hist.DEFAULT_ARCHIVE_URL)  # land-mask probe only
FORECAST_DAYS = 7
COLD_START_DAYS = 60     # cells with no archive history start this many days back

LIVE_VARS = hist.HOURLY_VARS

BATCH_SIZE = 50          # cells per request; must stay under 600 on the public API
MIN_HOURS_PER_CELL = 24 * 3   # fewer usable hours than this -> keep the old predictions
MAX_RETRY_PASSES = 3
RETRY_BACKOFF_SECONDS = 30    # multiplied by the pass number


# ==============================================================================
# Window
# ==============================================================================
def forecast_window(days: int, cells: list, use_db: bool) -> tuple:
    """
    (start, end) as YYYY-MM-DD. End is today + days - 1. Start is the day
    after the earliest cell's last archive feature row, so every cell's KBDI
    can continue without a gap; limited to what the forecast API keeps.
    """
    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days - 1)
    start = today
    if use_db:
        from backend.api.weather import archive_coverage
        cov = archive_coverage()
        last = {(round(a, 3), round(o, 3)): d
                for a, o, d in zip(cov["lat_round"], cov["lon_round"], cov["last_date"])}
        for c in cells:
            d = last.get(c)
            start = min(start, today - timedelta(days=COLD_START_DAYS) if d is None
                        else d + timedelta(days=1))
    start = max(start, today - timedelta(days=hist.FORECAST_MAX_PAST_DAYS))
    return str(start), str(end)


# ==============================================================================
# Fetch + clean one batch
# ==============================================================================
def clean_cell(raw: pd.DataFrame) -> tuple:
    """
    Fills short hour gaps in one cell's forecast. Returns (complete hours
    indexed by datetime_utc, number of hours dropped).
    """
    h = raw.drop_duplicates("datetime_utc").set_index("datetime_utc")[LIVE_VARS].copy()
    if h.isna().any().any():
        h, _ = hist.fill_short_hour_gaps(h)
    complete = h.notna().all(axis=1)
    return h[complete], int((~complete).sum())


def to_daily(h: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Hourly -> daily, keeping only days with all 24 plausible hours."""
    df = h.reset_index()
    df["lat_round"], df["lon_round"] = lat, lon
    df, _ = drop_implausible_readings(df)
    day = df["datetime_utc"].dt.floor("D")
    df = df[day.map(day.value_counts()) == 24]
    return aggregate_to_daily(df) if not df.empty else pd.DataFrame()


def fetch_batch(client, limiter, args, cells: list) -> dict:
    frames = hist.fetch_hourly(client, limiter, args.forecast_url, cells, args.start, args.end,
                               variables=LIVE_VARS)
    good, short, dropped = [], [], 0
    for (lat, lon), raw in zip(cells, frames):
        h, n_dropped = clean_cell(raw)
        dropped += n_dropped
        daily = to_daily(h, lat, lon) if len(h) >= MIN_HOURS_PER_CELL else pd.DataFrame()
        if daily.empty:
            short.append((lat, lon))
            continue
        good.append(daily)
    out = pd.concat(good, ignore_index=True) if good else pd.DataFrame()
    if args.save_db and not out.empty:
        from backend.api.weather import save_weather_daily
        save_weather_daily(out, args.run_id, "forecast")
    return {"df": out, "short": short, "dropped_hours": dropped}


def run_pass(client, limiter, args, batches: list, results: list, tot: dict, t0: float) -> list:
    """
    One pass over `batches`; returns the batches that raised (to retry). On
    the daily limit, sets tot["stop"] and returns every batch not yet handled.
    """
    failed, handled = [], set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_batch, client, limiter, args, b): b for b in batches}
        try:
            for fut in as_completed(futures):
                batch = futures[fut]
                handled.add(fut)
                try:
                    r = fut.result()
                except hist.DailyLimitReached:
                    print("\nDaily API limit reached -- stopping. Cells not refreshed keep "
                          "their previous predictions.", flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    tot["stop"] = True
                    return failed + [batch] + [b for f, b in futures.items() if f not in handled]
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED batch starting {batch[0]}: {exc}", flush=True)
                    failed.append(batch)
                    continue
                results.append(r["df"])
                tot["cells"] += len(batch)
                tot["saved"] += len(batch) - len(r["short"])
                tot["short"] += r["short"]
                tot["dropped"] += r["dropped_hours"]
                elapsed = time.time() - t0
                print(f"  {tot['cells']:,}/{tot['todo']:,} cells | saved {tot['saved']:,} | "
                      f"{elapsed:,.0f}s", flush=True)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return failed


# ==============================================================================
# Main
# ==============================================================================
def check_server_reachable(url: str):
    """Fail fast if the self-hosted server is down, instead of retrying every batch."""
    if hist.is_public(url):
        return
    try:
        requests.get(url, timeout=5)   # any response, even a 400, means it is up
    except requests.exceptions.ConnectionError:
        sys.exit(f"Cannot reach the self-hosted Open-Meteo server at {url}.\n"
                 "From the repo root run:  docker compose up -d\n"
                 "or set OPENMETEO_FORECAST_URL=https://api.open-meteo.com/v1/forecast")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forecast-url", default=FORECAST_URL)
    parser.add_argument("--archive-url", default=ARCHIVE_URL,
                        help="Only used to build the land mask if it does not exist yet")
    parser.add_argument("--days", type=int, default=FORECAST_DAYS, help="Forecast days incl. today")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Cells per request")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel requests (default 4 self-hosted, 2 public)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Only this part of the grid (test run)")
    parser.add_argument("--limit", type=int, default=None, help="Only this many land cells (test run)")
    parser.add_argument("--no-db", dest="save_db", action="store_false",
                        help="Fetch and report only; write nothing to wildfire_db.db")
    parser.add_argument("--no-predict", dest="predict", action="store_false",
                        help="Store the weather but do not build features or predict")
    args = parser.parse_args()
    public = hist.is_public(args.forecast_url)
    args.workers = args.workers or (2 if public else 4)

    if args.save_db:
        from backend.db_init import create_tables
        from backend.api.weather import finish_run, seed_grid_cells, start_run
        create_tables()

    check_server_reachable(args.forecast_url)
    client = hist.make_client()
    limiter = hist.WeightLimiter()

    grid = hist.build_grid(*args.bbox) if args.bbox else hist.build_grid()
    probe_args = types.SimpleNamespace(start=hist.DEFAULT_START, archive_url=args.archive_url)
    try:
        mask = hist.build_land_mask(client, limiter, probe_args, grid)
    except hist.DailyLimitReached:
        sys.exit("Daily API limit reached while probing the land mask. Run again later.")
    land = [c for c in grid if mask.get(c) is not False]   # unknown cells are fetched
    todo = land[: args.limit] if args.limit else land
    if args.save_db:
        seed_grid_cells({c: mask[c] for c in grid if c in mask})

    args.start, args.end = forecast_window(args.days, todo, args.save_db)
    weight = hist.request_weight(len(todo), args.start, args.end, len(LIVE_VARS))
    print("=" * 74)
    print("FETCH LIVE FORECAST (LAND CELLS)")
    print("=" * 74)
    print(f"Grid: {len(grid):,} cells | ocean skipped: {len(grid) - len(land):,} | to fetch: {len(todo):,}")
    print(f"Window: {args.start} -> {args.end} (hourly, UTC; starts after the archive history)")
    print(f"Endpoint: {args.forecast_url} ({'public API, paced' if public else 'self-hosted, no quota'})")
    print(f"Cost: ~{weight:,.0f} API calls in {-(-len(todo) // args.batch_size):,} requests | "
          f"DB: {'yes' if args.save_db else 'no (--no-db)'}\n")

    if args.save_db:
        scope = f"bbox={args.bbox}" if args.bbox else "full grid"
        args.run_id = start_run("forecast", args.start, args.end,
                                note=scope + (f", limit={args.limit}" if args.limit else ""))
        print(f"Forecast run: {args.run_id}")

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    results = []
    tot = {"todo": len(todo), "cells": 0, "saved": 0, "short": [], "dropped": 0, "stop": False}
    t0 = time.time()
    try:
        for attempt in range(1, MAX_RETRY_PASSES + 1):
            if attempt > 1:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"\nRetry pass {attempt}/{MAX_RETRY_PASSES}: {len(batches)} failed "
                      f"batch(es), waiting {wait}s...", flush=True)
                time.sleep(wait)
            batches = run_pass(client, limiter, args, batches, results, tot, t0)
            if not batches or tot["stop"]:
                break
    except KeyboardInterrupt:
        if args.save_db:
            finish_run(args.run_id, "failed")
        print("\nStopped. The run is marked failed; the current predictions are unchanged.")
        sys.exit(1)

    failed_cells = {c for b in batches for c in b} | set(tot["short"])
    n_days = sum(len(r) for r in results)
    print(f"\nFetched {tot['saved']:,}/{len(todo):,} land cells in {time.time() - t0:,.0f}s "
          f"({n_days:,} daily rows).")
    if tot["dropped"]:
        print(f"Dropped {tot['dropped']:,} incomplete hour(s) (after filling short gaps).")
    if failed_cells:
        print(f"{len(failed_cells):,} cell(s) NOT refreshed -- their previous predictions stay "
              f"current. First few: {sorted(failed_cells)[:5]}")

    if args.save_db:
        status = "ok" if not failed_cells else ("partial" if tot["saved"] else "failed")
        n_rows = finish_run(args.run_id, status)
        print(f"DB: run {args.run_id} {status}, {n_rows:,} weather rows.")
        if args.predict and n_rows:
            from backend.api.features import build_run_features
            from backend.api.fire_prediction import predict_run
            n_feat = build_run_features(args.run_id)
            print(f"Features: {n_feat:,} rows.")
            predict_run(args.run_id)
    sys.exit(1 if failed_cells else 0)


if __name__ == "__main__":
    main()
