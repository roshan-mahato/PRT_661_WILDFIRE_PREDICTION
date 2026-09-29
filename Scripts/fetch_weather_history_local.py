"""
================================================================================
fetch_full_grid_history.py  --  CONTINUOUS daily weather for EVERY grid cell
================================================================================
Downloads Open-Meteo weather for the full 0.5-degree Australia grid
(lat -44..-10, lon 112..154 = 5,865 cells) from START_DATE to END_DATE and
saves ONE ROW PER (land cell, day) -- every day, no gaps, no NaN.

HOW CONTINUITY IS GUARANTEED
----------------------------
1. Archive API for the whole range (ERA5 reanalysis).
2. The archive runs ~5 days behind today. Any missing days at the END are
   filled from the Forecast API (which keeps the last ~3 months of past data).
3. Short holes INSIDE the series (a few missing hours) are filled by time
   interpolation of the hourly values (max 6 hours in a row). Missing rain
   hours in such a hole count as 0 mm.
4. If a whole day is still missing after that, it is filled by interpolating
   the neighbouring days (max 3 days in a row) with 0 mm rain.
5. If a cell STILL has a hole after all that, it is NOT marked done and is
   listed in data/full_grid_incomplete.log -- the next run retries it.
Every row carries a `data_source` column (archive / forecast / interpolated)
so you can always see which values were filled.

The final combine step checks every land cell has every single day from
START to END with no NaN and no duplicates, and says PASS or FAIL.

Aggregation (same as the training and live pipelines):
    precipitation -> SUM of the day,  everything else -> MEAN of the day.
Days are UTC days, same as the rest of the project.

The combined file can be used as BOTH inputs of build_labeled_dataset.py:
    --weather       data/full_grid_daily_weather.csv
    --kbdi-weather  data/full_grid_daily_weather.csv

Ocean cells (soil moisture 0 for the whole series) are logged and skipped.

RESUMABLE: each batch is saved as its own part file, and finished cells are
logged. Stop any time (Ctrl+C) and run again -- it carries on.

USAGE
-----
    uv run python Scripts/fetch_full_grid_history.py --limit 20     # quick test
    uv run python Scripts/fetch_full_grid_history.py                # full run + combine
    uv run python Scripts/fetch_full_grid_history.py --combine-only # rebuild + verify output
    uv run python Scripts/fetch_full_grid_history.py --end 2026-09-28
================================================================================
"""

import argparse
import glob
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
from retry_requests import retry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
LAT_MIN, LAT_MAX = -44.0, -10.0
LON_MIN, LON_MAX = 112.0, 154.0
GRID_SIZE = 0.5

DEFAULT_START = "2023-12-01"
# Default end = yesterday (UTC): today is not finished yet, so its "daily"
# values would be part observation, part forecast.
DEFAULT_END = str(datetime.now(timezone.utc).date() - timedelta(days=1))

DEFAULT_ARCHIVE_URL = "http://127.0.0.1:8080/v1/archive"      # self-hosted (docker-compose.yml)
DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_MAX_PAST_DAYS = 90   # the forecast API keeps roughly 3 months of past data

DATA_DIR = os.path.join(REPO_ROOT, "data")
PARTS_DIR = os.path.join(DATA_DIR, "full_grid_parts")
OUTPUT_CSV = os.path.join(DATA_DIR, "full_grid_daily_weather.csv")
PROGRESS_LOG = os.path.join(DATA_DIR, "full_grid_progress.log")
OCEAN_LOG = os.path.join(DATA_DIR, "full_grid_ocean_cells.log")
INCOMPLETE_LOG = os.path.join(DATA_DIR, "full_grid_incomplete.log")

# Same 10 variables as the other fetch scripts (10 = 1 call weight).
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
SNAPSHOT_VARS = [v for v in HOURLY_VARS if v != "precipitation"]

MAX_HOURLY_FILL = 6      # longest run of missing hours filled by interpolation
MAX_DAILY_FILL = 3       # longest run of missing days filled by interpolation

REQUEST_TIMEOUT_SECONDS = 300
RATE_LIMIT_WAITS = {"minutely": 60, "hourly": 61 * 60}

_write_lock = threading.Lock()


class DailyLimitReached(Exception):
    pass


# ==============================================================================
# Grid and logs
# ==============================================================================
def round_to_grid(v: float, g: float = GRID_SIZE) -> float:
    """Double-round pattern used everywhere in the project."""
    return round(round(v / g) * g, 3)


def build_grid() -> list:
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_SIZE / 2, GRID_SIZE)
    lons = np.arange(LON_MIN, LON_MAX + GRID_SIZE / 2, GRID_SIZE)
    return [(round_to_grid(a), round_to_grid(o)) for a in lats for o in lons]


def read_cell_log(path: str) -> set:
    if not os.path.exists(path):
        return set()
    cells = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                cells.add((round_to_grid(float(parts[0])), round_to_grid(float(parts[1]))))
    return cells


def append_log(path: str, lines: list) -> None:
    if lines:
        with open(path, "a") as f:
            f.write("".join(f"{line}\n" for line in lines))


# ==============================================================================
# Fetch
# ==============================================================================
def make_client() -> openmeteo_requests.Client:
    # Plain session (no cache): multi-year responses are large and each is
    # needed once, so caching them only fills the disk.
    session = retry(requests.Session(), retries=5, backoff_factor=0.5)
    return openmeteo_requests.Client(session=session)


def fetch_hourly(client, url: str, cells: list, start: str, end: str) -> list:
    """
    One request for several cells (comma-separated coordinates). Returns one
    hourly DataFrame per cell, in the same order. Minute/hour limits are
    waited out; the daily limit raises DailyLimitReached.
    """
    params = {
        "latitude": [c[0] for c in cells],
        "longitude": [c[1] for c in cells],
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    while True:
        try:
            responses = client.weather_api(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            break
        except Exception as exc:  # noqa: BLE001 - client wraps the HTTP error
            text = str(exc).lower()
            if "daily" in text:
                raise DailyLimitReached(text) from exc
            wait = next((s for k, s in RATE_LIMIT_WAITS.items() if k in text), None)
            if wait is None:
                raise
            print(f"  rate limit hit; waiting {wait // 60} min...", flush=True)
            time.sleep(wait)

    frames = []
    for (lat, lon), resp in zip(cells, responses):
        hourly = resp.Hourly()
        data = {
            "datetime_utc": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }
        for i, var in enumerate(HOURLY_VARS):
            data[var] = hourly.Variables(i).ValuesAsNumpy()
        df = pd.DataFrame(data)
        df["lat_round"], df["lon_round"] = lat, lon
        frames.append(df)
    return frames


# ==============================================================================
# Make one cell's hourly series complete
# ==============================================================================
def full_hour_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz="UTC"),
                         pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                         freq="h", inclusive="left")


def on_full_index(hourly: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Puts the series on the complete hourly index (missing hours become NaN)."""
    h = hourly.drop_duplicates("datetime_utc").set_index("datetime_utc")[HOURLY_VARS]
    return h.reindex(idx)


def missing_days(h: pd.DataFrame) -> list:
    """Days where ANY variable has ANY missing hour."""
    bad = h.isna().any(axis=1)
    return sorted(set(h.index[bad].normalize().tz_localize(None).date))


def merge_forecast(h: pd.DataFrame, fc: pd.DataFrame) -> tuple:
    """Fills NaN hours in `h` from the forecast series. Returns (h, filled_hours)."""
    fc = fc.drop_duplicates("datetime_utc").set_index("datetime_utc")[HOURLY_VARS]
    fc = fc.reindex(h.index)
    was_nan = h.isna()
    h = h.fillna(fc)
    filled = was_nan & h.notna()
    return h, filled.any(axis=1)


def fill_short_hour_gaps(h: pd.DataFrame) -> tuple:
    """
    Interpolates runs of at most MAX_HOURLY_FILL missing hours (snapshot
    variables); missing rain inside such a run counts as 0. Longer runs stay
    NaN. Returns (h, filled_mask).
    """
    was_nan = h.isna().any(axis=1)
    out = h.copy()
    for col in SNAPSHOT_VARS:
        s = out[col]
        run_id = s.notna().cumsum()
        run_len = s.isna().groupby(run_id).transform("sum")
        short = s.isna() & (run_len <= MAX_HOURLY_FILL)
        interp = s.interpolate(method="time", limit_direction="both")
        out.loc[short, col] = interp[short]
    # Rain: only fill hours whose other variables are now complete (i.e. the
    # hour sits in a short gap), so a long outage is not turned into "no rain".
    rain_nan = out["precipitation"].isna() & out[SNAPSHOT_VARS].notna().all(axis=1)
    out.loc[rain_nan, "precipitation"] = 0.0
    filled = was_nan & out.notna().all(axis=1)
    return out, filled


def to_daily(h: pd.DataFrame, hour_source: pd.Series) -> pd.DataFrame:
    """Hourly -> daily. A day with any NaN hour is left NaN (not averaged over the rest)."""
    h = h.copy()
    h["acq_date"] = h.index.tz_localize(None).floor("D")
    g = h.groupby("acq_date")
    daily = g[SNAPSHOT_VARS].mean()
    daily["precipitation"] = g["precipitation"].sum(min_count=1)
    incomplete = g[HOURLY_VARS].apply(lambda d: d.isna().any().any())
    daily.loc[incomplete, HOURLY_VARS] = np.nan

    # Day source: the "worst" source of any hour in it.
    rank = {"archive": 0, "forecast": 1, "interpolated": 2}
    src = hour_source.map(rank).groupby(h["acq_date"].values).max()
    inv = {v: k for k, v in rank.items()}
    daily["data_source"] = src.reindex(daily.index).map(inv)
    return daily[HOURLY_VARS + ["data_source"]]


def fill_short_day_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """Interpolates runs of at most MAX_DAILY_FILL missing days; rain = 0 on them."""
    nan_day = daily[HOURLY_VARS].isna().any(axis=1)
    if not nan_day.any():
        return daily
    run_id = (~nan_day).cumsum()
    run_len = nan_day.groupby(run_id).transform("sum")
    short = nan_day & (run_len <= MAX_DAILY_FILL)
    interp = daily[SNAPSHOT_VARS].interpolate(limit_direction="both")
    daily.loc[short, SNAPSHOT_VARS] = interp.loc[short]
    daily.loc[short, "precipitation"] = 0.0
    daily.loc[short, "data_source"] = "interpolated"
    return daily


# ==============================================================================
# One batch
# ==============================================================================
def process_batch(client, args, cells: list) -> dict:
    idx = full_hour_index(args.start, args.end)
    archive = fetch_hourly(client, args.archive_url, cells, args.start, args.end)

    series, sources = {}, {}
    for cell, raw in zip(cells, archive):
        series[cell] = on_full_index(raw, idx)
        sources[cell] = pd.Series("archive", index=idx)

    # Ocean check on the archive data (soil moisture never above 0).
    ocean = [c for c in cells
             if series[c]["soil_moisture_0_to_7cm"].isna().all()
             or series[c]["soil_moisture_0_to_7cm"].max() <= 0]
    land = [c for c in cells if c not in ocean]

    # Step 2: missing recent days -> Forecast API (one request for all such cells).
    fc_floor = max(pd.Timestamp(args.start).date(),
                   date.today() - timedelta(days=FORECAST_MAX_PAST_DAYS))
    need_fc = {c: [d for d in missing_days(series[c]) if d >= fc_floor] for c in land}
    need_fc = {c: d for c, d in need_fc.items() if d}
    if need_fc:
        fc_start = str(min(min(d) for d in need_fc.values()))
        fc_cells = list(need_fc)
        fc_frames = fetch_hourly(client, args.forecast_url, fc_cells, fc_start, args.end)
        for cell, fc in zip(fc_cells, fc_frames):
            series[cell], got = merge_forecast(series[cell], fc)
            sources[cell][got] = "forecast"

    # Steps 3-5.
    good, bad = [], []
    for cell in land:
        h, filled = fill_short_hour_gaps(series[cell])
        sources[cell][filled] = "interpolated"
        daily = fill_short_day_gaps(to_daily(h, sources[cell]))
        holes = daily[HOURLY_VARS].isna().any(axis=1)
        if holes.any():
            bad.append((cell, int(holes.sum()), str(daily.index[holes][0].date())))
            continue
        daily = daily.reset_index()
        daily["lat_round"], daily["lon_round"] = cell
        daily["datetime_utc"] = daily["acq_date"].dt.tz_localize("UTC")
        good.append(daily[["lat_round", "lon_round", "acq_date", "datetime_utc"]
                          + HOURLY_VARS + ["data_source"]])

    with _write_lock:
        if good:
            key = hashlib.md5(",".join(f"{a}_{o}" for a, o in cells).encode()).hexdigest()[:12]
            part = os.path.join(PARTS_DIR, f"part_{key}.csv")
            tmp = part + ".tmp"
            pd.concat(good, ignore_index=True).to_csv(tmp, index=False)
            os.replace(tmp, part)   # a part file is either complete or absent
        append_log(OCEAN_LOG, [f"{a},{o}" for a, o in ocean])
        append_log(INCOMPLETE_LOG, [f"{a},{o},{n} missing days,first {d}" for (a, o), n, d in bad])
        # Logged AFTER the data is on disk; incomplete cells are NOT logged,
        # so the next run retries them.
        done = [c for c in cells if c not in {b[0] for b in bad}]
        append_log(PROGRESS_LOG, [f"{a},{o}" for a, o in done])

    n_fc = sum(int((d["data_source"] == "forecast").sum()) for d in good)
    n_int = sum(int((d["data_source"] == "interpolated").sum()) for d in good)
    return {"land": len(good), "ocean": len(ocean), "bad": bad,
            "forecast_days": n_fc, "interp_days": n_int}


# ==============================================================================
# Combine + verify
# ==============================================================================
def combine_and_verify(start: str, end: str) -> bool:
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "part_*.csv")))
    if not parts:
        print("No part files to combine.")
        return False
    print(f"\nCombining {len(parts):,} part files...")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    # Only keep the requested window (older runs may have used other dates).
    df = df[(df["acq_date"] >= pd.Timestamp(start)) & (df["acq_date"] <= pd.Timestamp(end))]
    before = len(df)
    df = df.drop_duplicates(["lat_round", "lon_round", "acq_date"], keep="last")
    if len(df) < before:
        print(f"  removed {before - len(df):,} duplicate cell-days (from a re-fetched batch)")
    df = df.sort_values(["lat_round", "lon_round", "acq_date"]).reset_index(drop=True)

    expected_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    per_cell = df.groupby(["lat_round", "lon_round"]).agg(
        n=("acq_date", "size"), first=("acq_date", "min"), last=("acq_date", "max"))
    short_cells = per_cell[(per_cell["n"] != expected_days)
                           | (per_cell["first"] != pd.Timestamp(start))
                           | (per_cell["last"] != pd.Timestamp(end))]
    nan_rows = int(df[HOURLY_VARS].isna().any(axis=1).sum())

    print("\n" + "=" * 74)
    print("CONTINUITY CHECK")
    print("=" * 74)
    print(f"  window           : {start} -> {end} ({expected_days} days)")
    print(f"  land cells       : {len(per_cell):,}")
    print(f"  rows             : {len(df):,} (expected {len(per_cell) * expected_days:,})")
    print(f"  cells not full   : {len(short_cells):,}")
    print(f"  rows with NaN    : {nan_rows:,}")
    print("  data_source      : " + ", ".join(
        f"{k} {v:,}" for k, v in df["data_source"].value_counts().items()))
    ok = short_cells.empty and nan_rows == 0
    if not ok and not short_cells.empty:
        print("\n  Cells with missing days (first 10):")
        print(short_cells.head(10).to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  -> {'PASS' if ok else 'FAIL'}: saved {OUTPUT_CSV}")
    return ok


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD (default 2023-12-01)")
    parser.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD (default yesterday, UTC)")
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--forecast-url", default=DEFAULT_FORECAST_URL)
    parser.add_argument("--batch-size", type=int, default=10, help="Cells per request")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel requests (use 1 on the public archive API)")
    parser.add_argument("--limit", type=int, default=None, help="Only this many cells (test run)")
    parser.add_argument("--combine-only", action="store_true",
                        help="Skip fetching; just combine part files and verify")
    args = parser.parse_args()

    os.makedirs(PARTS_DIR, exist_ok=True)
    if args.combine_only:
        sys.exit(0 if combine_and_verify(args.start, args.end) else 1)

    grid = build_grid()
    done = read_cell_log(PROGRESS_LOG)
    todo = [c for c in grid if c not in done]
    if args.limit:
        todo = todo[: args.limit]
    if os.path.exists(INCOMPLETE_LOG):
        os.remove(INCOMPLETE_LOG)   # rebuilt each run from what is still incomplete

    print("=" * 74)
    print("FETCH FULL-GRID DAILY WEATHER HISTORY")
    print("=" * 74)
    print(f"Grid: {len(grid):,} cells | done: {len(done):,} | to fetch: {len(todo):,}")
    print(f"Dates: {args.start} -> {args.end}")
    print(f"Archive: {args.archive_url}\nForecast (recent days): {args.forecast_url}\n")

    if todo:
        batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        client = make_client()
        tot = {"cells": 0, "land": 0, "ocean": 0, "bad": 0, "fc": 0, "int": 0}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_batch, client, args, b): b for b in batches}
            try:
                for fut in as_completed(futures):
                    batch = futures[fut]
                    try:
                        r = fut.result()
                    except DailyLimitReached:
                        print("\nDaily API limit reached. Run again later -- it continues "
                              "from here.", flush=True)
                        pool.shutdown(wait=False, cancel_futures=True)
                        sys.exit(2)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  FAILED batch starting {batch[0]} (will retry next run): {exc}",
                              flush=True)
                        continue
                    tot["cells"] += len(batch)
                    tot["land"] += r["land"]
                    tot["ocean"] += r["ocean"]
                    tot["bad"] += len(r["bad"])
                    tot["fc"] += r["forecast_days"]
                    tot["int"] += r["interp_days"]
                    for cell, n, first in r["bad"]:
                        print(f"  INCOMPLETE {cell}: {n} day(s) still missing from {first} "
                              f"-- will retry next run", flush=True)
                    rate = tot["cells"] / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - tot["cells"]) / max(rate, 1e-9) / 60
                    print(f"  {tot['cells']:,}/{len(todo):,} cells | land {tot['land']:,} | "
                          f"ocean {tot['ocean']:,} | incomplete {tot['bad']:,} | "
                          f"~{eta:,.0f} min left", flush=True)
            except KeyboardInterrupt:
                print("\nStopped. Finished cells are saved; run again to continue.")
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

        print(f"\nThis run: {tot['land']:,} land cells saved, {tot['ocean']:,} ocean skipped, "
              f"{tot['bad']:,} incomplete.")
        print(f"Filled cell-days: {tot['fc']:,} from forecast, {tot['int']:,} interpolated.")

    remaining = [c for c in grid if c not in read_cell_log(PROGRESS_LOG)]
    if remaining and not args.limit:
        print(f"\n{len(remaining):,} cell(s) still not done -- run the script again. "
              f"See {INCOMPLETE_LOG} for cells with holes.")
        sys.exit(1)
    ok = combine_and_verify(args.start, args.end)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()