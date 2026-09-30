"""
================================================================================
fetch_weather_history.py  --  CONTINUOUS daily weather for EVERY grid cell
================================================================================
Downloads Open-Meteo weather for the full 0.5-degree Australia grid
(lat -44..-10, lon 112..154 = 5,865 cells) and saves ONE ROW PER (land cell,
day) -- every day, no gaps, no NaN. Replaces fetch_weather_data.py and
fetch_weather_history_local.py.

The output is the weather input of build_labeled_dataset.py:
    --weather       data/full_grid_daily_weather.csv
    --kbdi-weather  data/full_grid_daily_weather.csv

HOW CONTINUITY IS GUARANTEED
----------------------------
1. Archive API for the whole range.
2. Any missing days at the END are filled from the Forecast API (which keeps
   the last ~3 months of past data).
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
    precipitation -> SUM of the day,  everything else -> MEAN of the day,
plus the daily peaks (prediction_engine.DAILY_PEAK_COLS): max temperature,
min humidity, max wind / gust / VPD, daily evapotranspiration total, and the
temperature, humidity and wind at the hour with the highest FFDI.
Days are UTC days, same as the rest of the project; the Australian afternoon
falls inside one UTC day, so each day's peak is counted once.

The peak columns need the hourly data, so they are fetched into
data/full_grid_parts_v2/ with data/full_grid_progress_v2.log: the first run
after adding them fetches everything again. The old part files and log are
not touched.

WHY NO "cape"
-------------
The archive (ERA5) has no CAPE: it comes back NaN for every hour, on the
public API too. Requesting it made every historical day a hole. It was always
empty in the training data and is dropped by wildfire_data_processing.py.

DATABASE
--------
Each finished batch is also written to the `weather_daily` table as part of
one archive fetch run (source='archive'), before the batch is logged as
done -- so a batch whose DB write fails is retried. The archive is one row per
cell-day: re-fetched days that are already stored are skipped. After the
fetch, the KBDI features of the new days are built (feature_engineered),
continuing each cell from its last stored day; live forecasts continue from
there. --no-db writes the CSV files only. Part files fetched without the DB
can be loaded later with `Scripts/pipeline.py load-history`.

SELF-HOSTED vs PUBLIC API
-------------------------
Default is the self-hosted server (docker-compose.yml: `docker compose up -d`),
which has no quota. The public API counts a request as
    weight = locations * max(1, days / 14) * max(1, variables / 10)
and allows 600/min, 5,000/hour, 10,000/day. The full grid from 2023-12-01 is
~430,000 weight -- over 40 days of quota -- so use the public API only for
small tests (--bbox, short --start/--end). When the URL is public, requests
are paced to stay under those limits.

OCEAN CELLS
-----------
Most of the bounding box is sea. Before the first fetch, a cheap probe (one
week, 2 variables, LAND_PROBE_BATCH cells per request) classifies every cell
and caches it in data/full_grid_land_mask.csv; ocean cells are then never
fetched. The mask only grows (new --bbox cells are probed on demand);
delete the file to rebuild it. The per-batch ocean check still runs as a
backstop and logs any cell the probe missed.

RESUMABLE + INCREMENTAL: each batch is saved as its own part file, and the
progress log records every cell WITH the date range it covers
("lat,lon,start,end"). A run fetches only the days each cell is missing, so
extending --end by a week downloads a week, and moving --start earlier
downloads only the earlier days. Failed batches are retried in the same run
(MAX_RETRY_PASSES); stop any time (Ctrl+C) and run again -- it carries on.
Old logs without dates are converted once, using the dates found in the
part files (a .bak copy is kept).

USAGE
-----
    uv run python Scripts/fetch_weather_history.py --bbox -34 -33 150 152   # quick test
    uv run python Scripts/fetch_weather_history.py                           # full run + combine
    uv run python Scripts/fetch_weather_history.py --combine-only            # rebuild + verify
    uv run python Scripts/fetch_weather_history.py --end 2026-09-28
================================================================================
"""

import argparse
import glob
import hashlib
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
from retry_requests import retry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prediction_engine import DAILY_PEAK_COLS, daily_peaks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
LAT_MIN, LAT_MAX = -44.0, -10.0
LON_MIN, LON_MAX = 112.0, 154.0
GRID_SIZE = 0.5

# Training starts 2024-01-01; KBDI needs a run-in period before that.
TRAINING_START_DATE = "2024-01-01"
KBDI_SPINUP_DAYS = 31
DEFAULT_START = str((pd.Timestamp(TRAINING_START_DATE) - pd.Timedelta(days=KBDI_SPINUP_DAYS)).date())
# Default end = yesterday (UTC): today is not finished yet, so its "daily"
# values would be part observation, part forecast.
DEFAULT_END = str(datetime.now(timezone.utc).date() - timedelta(days=1))

DEFAULT_ARCHIVE_URL = "http://127.0.0.1:8080/v1/archive"    # self-hosted (docker-compose.yml)
DEFAULT_FORECAST_URL = "http://127.0.0.1:8080/v1/forecast"
# DEFAULT_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_MAX_PAST_DAYS = 90   # the forecast API keeps roughly 3 months of past data

DATA_DIR = os.path.join(REPO_ROOT, "data")
# "_v2": part files with the daily peak columns. The old full_grid_parts/ and
# full_grid_progress.log (means only) are left alone; a v2 run fetches again.
PARTS_DIR = os.path.join(DATA_DIR, "full_grid_parts_v2")
OUTPUT_CSV = os.path.join(DATA_DIR, "full_grid_daily_weather.csv")
PROGRESS_LOG = os.path.join(DATA_DIR, "full_grid_progress_v2.log")
OCEAN_LOG = os.path.join(DATA_DIR, "full_grid_ocean_cells.log")
INCOMPLETE_LOG = os.path.join(DATA_DIR, "full_grid_incomplete.log")
LAND_MASK_CSV = os.path.join(DATA_DIR, "full_grid_land_mask.csv")

LAND_PROBE_VARS = ["temperature_2m", "soil_moisture_0_to_7cm"]
LAND_PROBE_DAYS = 7
LAND_PROBE_BATCH = 100

# Same variables as the live pipeline, minus "cape" (see WHY NO "cape").
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "wind_gusts_10m",
    "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
]
SNAPSHOT_VARS = [v for v in HOURLY_VARS if v != "precipitation"]
# Every daily column: the means/sum above plus the daily peaks.
DAILY_COLS = HOURLY_VARS + DAILY_PEAK_COLS

MAX_HOURLY_FILL = 6      # longest run of missing hours filled by interpolation
MAX_DAILY_FILL = 3       # longest run of missing days filled by interpolation

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRY_PASSES = 3     # failed batches are retried within the same run
RETRY_BACKOFF_SECONDS = 30   # multiplied by the pass number

# Public API quota: (weight limit, window in seconds). Open-Meteo's 429 reasons
# are "Minutely/Hourly/Daily API request limit exceeded".
PUBLIC_LIMITS = {"minutely": (600, 60), "hourly": (5000, 3600), "daily": (10000, 86400)}

_write_lock = threading.Lock()


class DailyLimitReached(Exception):
    pass


# ==============================================================================
# Public API pacing
# ==============================================================================
def is_public(url: str) -> bool:
    return "open-meteo.com" in url and "customer-" not in url


def request_weight(n_locations: int, start: str, end: str, n_vars: int = len(HOURLY_VARS)) -> float:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return n_locations * max(1.0, days / 14) * max(1.0, n_vars / 10)


class WeightLimiter:
    """
    Keeps the weight sent to the public API under the minute and hour limits
    by sleeping; the daily limit raises DailyLimitReached (the run is
    resumable, so waiting a whole day is pointless). No-op for self-hosted.
    """

    def __init__(self):
        self.sent = deque()   # (timestamp, weight)
        self.lock = threading.Lock()

    def acquire(self, weight: float):
        if weight > PUBLIC_LIMITS["minutely"][0]:
            raise ValueError(
                f"one request weighs {weight:,.0f}, over the public per-minute limit "
                f"({PUBLIC_LIMITS['minutely'][0]}) -- lower --batch-size or shorten the dates")
        with self.lock:
            while True:
                now = time.time()
                while self.sent and now - self.sent[0][0] >= PUBLIC_LIMITS["daily"][1]:
                    self.sent.popleft()
                wait = 0.0
                for name, (limit, window) in PUBLIC_LIMITS.items():
                    in_window = [(t, w) for t, w in self.sent if now - t < window]
                    used = sum(w for _, w in in_window)
                    if used + weight <= limit:
                        continue
                    if name == "daily":
                        raise DailyLimitReached(f"local pacing: {used:,.0f} weight sent today")
                    # Wait until enough of the oldest requests leave the window.
                    for t, w in in_window:
                        used -= w
                        if used + weight <= limit:
                            wait = max(wait, window - (now - t))
                            break
                if wait <= 0:
                    self.sent.append((now, weight))
                    return
                print(f"  pacing public API: waiting {wait:,.0f}s...", flush=True)
                time.sleep(wait)


# ==============================================================================
# Grid and logs
# ==============================================================================
def round_to_grid(v: float, g: float = GRID_SIZE) -> float:
    """Double-round pattern used everywhere in the project."""
    return round(round(v / g) * g, 3)


def build_grid(lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX) -> list:
    lats = np.arange(lat_min, lat_max + GRID_SIZE / 2, GRID_SIZE)
    lons = np.arange(lon_min, lon_max + GRID_SIZE / 2, GRID_SIZE)
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


def append_log(path: str, lines: list):
    if lines:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")


ONE_DAY = pd.Timedelta(days=1)


def merge_intervals(intervals: list) -> list:
    """Merges overlapping or back-to-back (start, end) day ranges."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + ONE_DAY:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def read_progress() -> dict:
    """{cell: merged [(start, end), ...]} of the days already saved per cell."""
    if not os.path.exists(PROGRESS_LOG):
        return {}
    raw = {}
    with open(PROGRESS_LOG) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) != 4:
                continue
            cell = (round_to_grid(float(p[0])), round_to_grid(float(p[1])))
            raw.setdefault(cell, []).append((pd.Timestamp(p[2]), pd.Timestamp(p[3])))
    return {c: merge_intervals(v) for c, v in raw.items()}


def missing_windows(intervals: list, start: str, end: str) -> list:
    """The parts of [start, end] not covered by `intervals`, as (start, end) strings."""
    cur, stop = pd.Timestamp(start), pd.Timestamp(end)
    gaps = []
    for s, e in intervals:
        if e < cur:
            continue
        if s > stop:
            break
        if s > cur:
            gaps.append((cur, s - ONE_DAY))
        cur = max(cur, e + ONE_DAY)
        if cur > stop:
            break
    if cur <= stop:
        gaps.append((cur, stop))
    return [(str(a.date()), str(b.date())) for a, b in gaps]


def migrate_legacy_progress():
    """
    Converts old "lat,lon" progress lines (no dates) to "lat,lon,start,end",
    taking each cell's range from its rows in the part files. Ocean cells are
    dropped (the ocean log / land mask handle them), and so are cells whose
    saved days are not one continuous range -- those are simply re-fetched.
    """
    if not os.path.exists(PROGRESS_LOG):
        return
    with open(PROGRESS_LOG) as f:
        lines = [l.strip() for l in f if l.strip()]
    legacy = {(round_to_grid(float(a)), round_to_grid(float(b)))
              for a, b in (l.split(",") for l in lines if l.count(",") == 1)}
    if not legacy:
        return
    legacy -= read_cell_log(OCEAN_LOG)
    ranges = {}
    parts = glob.glob(os.path.join(PARTS_DIR, "part_*.csv"))
    if legacy and parts:
        df = pd.concat((pd.read_csv(p, usecols=["lat_round", "lon_round", "acq_date"])
                        for p in parts), ignore_index=True).drop_duplicates()
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        for (a, o), g in df.groupby(["lat_round", "lon_round"]):
            cell = (round_to_grid(a), round_to_grid(o))
            lo, hi = g["acq_date"].min(), g["acq_date"].max()
            if cell in legacy and len(g) == (hi - lo).days + 1:
                ranges[cell] = (lo.date(), hi.date())
    kept = [l for l in lines if l.count(",") == 3]
    kept += [f"{a},{o},{s},{e}" for (a, o), (s, e) in sorted(ranges.items())]
    os.replace(PROGRESS_LOG, PROGRESS_LOG + ".bak")
    with open(PROGRESS_LOG, "w") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    print(f"Progress log converted to 'lat,lon,start,end': {len(ranges):,} land cell(s) "
          f"kept with their dates, {len(legacy) - len(ranges):,} will be re-fetched "
          f"(backup: {os.path.basename(PROGRESS_LOG)}.bak)\n")


# ==============================================================================
# Fetch
# ==============================================================================
def make_client() -> openmeteo_requests.Client:
    # Plain session (no cache): multi-year responses are large and each is
    # needed once, so caching them only fills the disk.
    session = retry(requests.Session(), retries=5, backoff_factor=0.5)
    return openmeteo_requests.Client(session=session)


def fetch_hourly(client, limiter, url: str, cells: list, start: str, end: str,
                 variables: list = HOURLY_VARS) -> list:
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
        "hourly": variables,
        "timezone": "UTC",
    }
    if is_public(url):
        limiter.acquire(request_weight(len(cells), start, end, len(variables)))
    while True:
        try:
            responses = client.weather_api(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            break
        except Exception as exc:  # noqa: BLE001 - client wraps the HTTP error
            text = str(exc).lower()
            if "daily api request limit" in text:
                raise DailyLimitReached(text) from exc
            wait = next((PUBLIC_LIMITS[k][1] + 60 for k in ("minutely", "hourly")
                         if f"{k} api request limit" in text), None)
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
        for i, var in enumerate(variables):
            data[var] = hourly.Variables(i).ValuesAsNumpy()
        frames.append(pd.DataFrame(data))
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


def is_ocean(h: pd.DataFrame) -> bool:
    """
    No soil moisture for the whole series. A cell with no data at all (e.g. the
    server returned nothing) is NOT ocean -- it stays incomplete and is retried.
    """
    if h["temperature_2m"].isna().all():
        return False
    sm = h["soil_moisture_0_to_7cm"]
    return bool(sm.isna().all() or sm.max() <= 0)


def read_land_mask() -> dict:
    """{cell: True if land} from the cached mask (empty if not built yet)."""
    if not os.path.exists(LAND_MASK_CSV):
        return {}
    m = pd.read_csv(LAND_MASK_CSV)
    return {(round_to_grid(a), round_to_grid(o)): bool(land)
            for a, o, land in zip(m["lat_round"], m["lon_round"], m["is_land"])}


def build_land_mask(client, limiter, args, grid: list) -> dict:
    """
    Probes every grid cell not yet in the mask with one week of the two
    variables is_ocean() needs, and appends the result to LAND_MASK_CSV.
    Cells the server returned nothing for are left out (unknown), so they are
    fetched normally and the per-batch check decides.
    """
    mask = read_land_mask()
    todo = [c for c in grid if c not in mask]
    if not todo:
        return mask
    start = args.start
    end = str((pd.Timestamp(start) + pd.Timedelta(days=LAND_PROBE_DAYS - 1)).date())
    print(f"Land mask: probing {len(todo):,} cells ({start} -> {end})...", flush=True)
    is_new = not os.path.exists(LAND_MASK_CSV)
    for i in range(0, len(todo), LAND_PROBE_BATCH):
        batch = todo[i:i + LAND_PROBE_BATCH]
        frames = fetch_hourly(client, limiter, args.archive_url, batch, start, end,
                              variables=LAND_PROBE_VARS)
        rows = []
        for cell, f in zip(batch, frames):
            if f["temperature_2m"].isna().all():
                continue
            mask[cell] = not is_ocean(f)
            rows.append((*cell, int(mask[cell])))
        pd.DataFrame(rows, columns=["lat_round", "lon_round", "is_land"]).to_csv(
            LAND_MASK_CSV, mode="a", header=is_new, index=False)
        is_new = False
        print(f"  {min(i + LAND_PROBE_BATCH, len(todo)):,}/{len(todo):,} probed", flush=True)
    known = [c for c in grid if c in mask]
    print(f"Land mask: {sum(mask[c] for c in known):,} land, "
          f"{sum(not mask[c] for c in known):,} ocean, {len(grid) - len(known):,} unknown\n")
    return mask


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
    """
    Hourly -> daily means/sum plus the daily peaks (prediction_engine.
    daily_peaks). A day with any NaN hour is left NaN (not built from the rest).
    """
    peaks = daily_peaks(h.rename_axis("datetime_utc").reset_index(), []).set_index("acq_date")
    h = h.copy()
    h["acq_date"] = h.index.tz_localize(None).floor("D")
    g = h.groupby("acq_date")
    daily = g[SNAPSHOT_VARS].mean()
    daily["precipitation"] = g["precipitation"].sum(min_count=1)
    daily = daily.join(peaks)
    incomplete = g[HOURLY_VARS].apply(lambda d: d.isna().any().any())
    daily.loc[incomplete, DAILY_COLS] = np.nan

    # Day source: the "worst" source of any hour in it.
    rank = {"archive": 0, "forecast": 1, "interpolated": 2}
    src = hour_source.map(rank).groupby(h["acq_date"].values).max()
    inv = {v: k for k, v in rank.items()}
    daily["data_source"] = src.reindex(daily.index).map(inv)
    return daily[DAILY_COLS + ["data_source"]]


def fill_short_day_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """Interpolates runs of at most MAX_DAILY_FILL missing days; rain = 0 on them."""
    nan_day = daily[DAILY_COLS].isna().any(axis=1)
    if not nan_day.any():
        return daily
    run_id = (~nan_day).cumsum()
    run_len = nan_day.groupby(run_id).transform("sum")
    short = nan_day & (run_len <= MAX_DAILY_FILL)
    fill_cols = SNAPSHOT_VARS + DAILY_PEAK_COLS
    interp = daily[fill_cols].interpolate(limit_direction="both")
    interp["peak_fire_hour_utc"] = interp["peak_fire_hour_utc"].round()
    daily.loc[short, fill_cols] = interp.loc[short]
    daily.loc[short, "precipitation"] = 0.0
    daily.loc[short, "data_source"] = "interpolated"
    return daily


# ==============================================================================
# One batch
# ==============================================================================
def process_batch(client, limiter, args, cells: list, start: str, end: str) -> dict:
    """Fetches and saves [start, end] for `cells` (the days they are missing)."""
    idx = full_hour_index(start, end)
    archive = fetch_hourly(client, limiter, args.archive_url, cells, start, end)

    series, sources = {}, {}
    for cell, raw in zip(cells, archive):
        series[cell] = on_full_index(raw, idx)
        sources[cell] = pd.Series("archive", index=idx)

    ocean = [c for c in cells if is_ocean(series[c])]
    land = [c for c in cells if c not in ocean]

    # Step 2: missing recent days -> Forecast API (one request for all such cells).
    today = datetime.now(timezone.utc).date()
    fc_floor = max(pd.Timestamp(start).date(),
                   today - timedelta(days=FORECAST_MAX_PAST_DAYS))
    need_fc = {c: [d for d in missing_days(series[c]) if d >= fc_floor] for c in land}
    need_fc = {c: d for c, d in need_fc.items() if d}
    if need_fc:
        fc_start = str(min(min(d) for d in need_fc.values()))
        fc_cells = list(need_fc)
        fc_frames = fetch_hourly(client, limiter, args.forecast_url, fc_cells, fc_start, end)
        for cell, fc in zip(fc_cells, fc_frames):
            series[cell], got = merge_forecast(series[cell], fc)
            sources[cell][got] = "forecast"

    # Steps 3-5.
    good, bad = [], []
    for cell in land:
        h, filled = fill_short_hour_gaps(series[cell])
        sources[cell][filled] = "interpolated"
        daily = fill_short_day_gaps(to_daily(h, sources[cell]))
        holes = daily[DAILY_COLS].isna().any(axis=1)
        if holes.any():
            bad.append((cell, int(holes.sum()), str(daily.index[holes][0].date())))
            continue
        daily = daily.reset_index()
        daily["lat_round"], daily["lon_round"] = cell
        daily["datetime_utc"] = daily["acq_date"].dt.tz_localize("UTC")
        good.append(daily[["lat_round", "lon_round", "acq_date", "datetime_utc"]
                          + DAILY_COLS + ["data_source"]])

    # Before the progress log: if this raises, the batch is retried.
    if good and getattr(args, "run_id", None):
        from backend.api.weather import save_weather_daily
        save_weather_daily(pd.concat(good, ignore_index=True), args.run_id, "archive")

    with _write_lock:
        if good:
            # Dates are part of the key, so a run over another window never
            # overwrites this one's part file.
            key_src = f"{start}_{end}_" + ",".join(f"{a}_{o}" for a, o in cells)
            key = hashlib.md5(key_src.encode()).hexdigest()[:12]
            part = os.path.join(PARTS_DIR, f"part_{key}.csv")
            tmp = part + ".tmp"
            pd.concat(good, ignore_index=True).to_csv(tmp, index=False)
            os.replace(tmp, part)   # a part file is either complete or absent
        append_log(OCEAN_LOG, [f"{a},{o}" for a, o in ocean])
        append_log(INCOMPLETE_LOG, [f"{a},{o},{n} missing days,first {d}" for (a, o), n, d in bad])
        # Logged AFTER the data is on disk, with the dates it covers.
        # Incomplete and ocean cells are NOT logged: the next run retries the
        # former, and the ocean log / land mask skip the latter.
        done = [c for c in land if c not in {b[0] for b in bad}]
        append_log(PROGRESS_LOG, [f"{a},{o},{start},{end}" for a, o in done])

    n_fc = sum(int((d["data_source"] == "forecast").sum()) for d in good)
    n_int = sum(int((d["data_source"] == "interpolated").sum()) for d in good)
    return {"land": len(good), "ocean": len(ocean), "bad": bad,
            "forecast_days": n_fc, "interp_days": n_int}


def run_pass(client, limiter, args, batches: list, tot: dict, n_todo: int, t0: float) -> list:
    """
    Runs one pass over `batches` -- (start, end, cells) tuples. Returns the
    batches that raised (to retry).
    """
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_batch, client, limiter, args, cells, s, e): (s, e, cells)
                   for s, e, cells in batches}
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
                    print(f"  FAILED batch starting {batch[2][0]} ({batch[0]} -> {batch[1]}): {exc}",
                          flush=True)
                    failed.append(batch)
                    continue
                tot["cells"] += len(batch[2])
                tot["land"] += r["land"]
                tot["ocean"] += r["ocean"]
                tot["bad"] += len(r["bad"])
                tot["fc"] += r["forecast_days"]
                tot["int"] += r["interp_days"]
                for cell, n, first in r["bad"]:
                    print(f"  INCOMPLETE {cell}: {n} day(s) still missing from {first} "
                          f"-- will retry next run", flush=True)
                rate = tot["cells"] / max(time.time() - t0, 1e-9)
                eta = (n_todo - tot["cells"]) / max(rate, 1e-9) / 60
                print(f"  {tot['cells']:,}/{n_todo:,} cells | land {tot['land']:,} | "
                      f"ocean {tot['ocean']:,} | incomplete {tot['bad']:,} | "
                      f"~{eta:,.0f} min left", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. Finished cells are saved; run again to continue.")
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)
    return failed


# ==============================================================================
# Combine + verify
# ==============================================================================
def combine_and_verify(start: str, end: str, grid: list) -> bool:
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "part_*.csv")))
    if not parts:
        print("No part files to combine.")
        return False
    print(f"\nCombining {len(parts):,} part files...")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    # Only keep the requested window and grid (older runs may have used others).
    df = df[(df["acq_date"] >= pd.Timestamp(start)) & (df["acq_date"] <= pd.Timestamp(end))]
    in_grid = pd.MultiIndex.from_frame(df[["lat_round", "lon_round"]]).isin(grid)
    df = df[in_grid]
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
    # reindex: a part file without the peak columns counts as NaN (re-fetch it).
    nan_rows = int(df.reindex(columns=DAILY_COLS).isna().any(axis=1).sum())
    mask_ocean = {c for c, land in read_land_mask().items() if not land}
    ocean = (read_cell_log(OCEAN_LOG) | mask_ocean) & set(grid)
    missing_cells = len(grid) - len(per_cell) - len(ocean)

    print("\n" + "=" * 74)
    print("CONTINUITY CHECK")
    print("=" * 74)
    print(f"  window           : {start} -> {end} ({expected_days} days)")
    print(f"  land cells       : {len(per_cell):,} (+ {len(ocean):,} ocean of {len(grid):,})")
    print(f"  cells with no data: {max(missing_cells, 0):,}")
    print(f"  rows             : {len(df):,} (expected {len(per_cell) * expected_days:,})")
    print(f"  cells not full   : {len(short_cells):,}")
    print(f"  rows with NaN    : {nan_rows:,}")
    print("  data_source      : " + ", ".join(
        f"{k} {v:,}" for k, v in df["data_source"].value_counts().items()))
    ok = short_cells.empty and nan_rows == 0 and missing_cells <= 0
    if not short_cells.empty:
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
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"YYYY-MM-DD (default {DEFAULT_START}: training start minus KBDI spin-up)")
    parser.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD (default yesterday, UTC)")
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--forecast-url", default=DEFAULT_FORECAST_URL)
    parser.add_argument("--batch-size", type=int, default=10, help="Cells per request")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel requests (use 1 on the public API)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Only this part of the grid (test run), e.g. --bbox -34 -33 150 152")
    parser.add_argument("--limit", type=int, default=None, help="Only this many cells (test run)")
    parser.add_argument("--combine-only", action="store_true",
                        help="Skip fetching; just combine part files and verify")
    parser.add_argument("--no-db", dest="save_db", action="store_false",
                        help="Write the CSV files only, leave wildfire_db.db untouched")
    args = parser.parse_args()
    args.run_id = None

    grid = build_grid(*args.bbox) if args.bbox else build_grid()
    os.makedirs(PARTS_DIR, exist_ok=True)
    if args.combine_only:
        sys.exit(0 if combine_and_verify(args.start, args.end, grid) else 1)

    client = make_client()
    limiter = WeightLimiter()
    try:
        mask = build_land_mask(client, limiter, args, grid)
    except DailyLimitReached:
        print("\nDaily API limit reached while probing the land mask. Run again later.")
        sys.exit(2)
    mask_ocean = {c for c in grid if mask.get(c) is False}

    migrate_legacy_progress()
    skip = mask_ocean | (read_cell_log(OCEAN_LOG) & set(grid))
    progress = read_progress()
    jobs = {}   # (start, end) -> cells missing exactly that window
    for c in grid:
        if c not in skip:
            for w in missing_windows(progress.get(c, []), args.start, args.end):
                jobs.setdefault(w, []).append(c)
    if args.limit:
        keep = set(list(dict.fromkeys(c for cells in jobs.values() for c in cells))[: args.limit])
        jobs = {w: [c for c in cells if c in keep] for w, cells in jobs.items()}
        jobs = {w: cells for w, cells in jobs.items() if cells}
    todo = {c for cells in jobs.values() for c in cells}
    if os.path.exists(INCOMPLETE_LOG):
        os.remove(INCOMPLETE_LOG)   # rebuilt each run from what is still incomplete

    print("=" * 74)
    print("FETCH FULL-GRID DAILY WEATHER HISTORY")
    print("=" * 74)
    print(f"Grid: {len(grid):,} cells | ocean (skipped): {len(skip):,} | "
          f"up to date: {len(grid) - len(skip) - len(todo):,} | to fetch: {len(todo):,}")
    print(f"Dates: {args.start} -> {args.end}")
    for (s, e), cells in sorted(jobs.items(), key=lambda kv: -len(kv[1]))[:5]:
        days = (pd.Timestamp(e) - pd.Timestamp(s)).days + 1
        print(f"  missing {s} -> {e} ({days:,} day(s)): {len(cells):,} cell(s)")
    if len(jobs) > 5:
        print(f"  ... and {len(jobs) - 5:,} other window(s)")
    print(f"Archive: {args.archive_url}\nForecast (recent days): {args.forecast_url}")
    if is_public(args.archive_url):
        total = sum(request_weight(len(cells), s, e) for (s, e), cells in jobs.items())
        print(f"PUBLIC API: this run weighs ~{total:,.0f} "
              f"(~{total / PUBLIC_LIMITS['daily'][0]:,.1f} days of quota)")
    print()

    if args.save_db:
        sys.path.insert(0, REPO_ROOT)
        from backend.api.weather import finish_run, seed_grid_cells, start_run
        from backend.db_init import create_tables
        create_tables()
        seed_grid_cells({c: mask[c] for c in grid if c in mask})
        if todo:
            args.run_id = start_run("archive", args.start, args.end,
                                    note=f"bbox={args.bbox}" if args.bbox else "full grid")
            print(f"Archive run: {args.run_id}\n")

    if todo:
        batches = [(s, e, cells[i:i + args.batch_size]) for (s, e), cells in jobs.items()
                   for i in range(0, len(cells), args.batch_size)]
        n_todo = sum(len(cells) for cells in jobs.values())
        tot = {"cells": 0, "land": 0, "ocean": 0, "bad": 0, "fc": 0, "int": 0}
        t0 = time.time()
        try:
            for attempt in range(1, MAX_RETRY_PASSES + 1):
                if attempt > 1:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    print(f"\nRetry pass {attempt}/{MAX_RETRY_PASSES}: {len(batches)} failed "
                          f"batch(es), waiting {wait}s...", flush=True)
                    time.sleep(wait)
                batches = run_pass(client, limiter, args, batches, tot, n_todo, t0)
                if not batches:
                    break
        except SystemExit:
            # Stopped (Ctrl+C or daily limit): keep what was saved, and build
            # its features so the live forecast can continue from it.
            if args.run_id:
                finish_run(args.run_id, "partial")
                from backend.api.features import build_archive_features
                build_archive_features()
            raise

        print(f"\nThis run: {tot['land']:,} land cells saved, {tot['ocean']:,} ocean skipped, "
              f"{tot['bad']:,} incomplete, {sum(len(b[2]) for b in batches):,} failed.")
        print(f"Filled cell-days: {tot['fc']:,} from forecast, {tot['int']:,} interpolated.")

    if args.save_db:
        from backend.api.features import build_archive_features
        if args.run_id:
            status = "ok" if not batches and not tot["bad"] else "partial"
            n = finish_run(args.run_id, status)
            print(f"DB: archive run {args.run_id} {status}, {n:,} new weather rows.")
        build_archive_features()

    progress = read_progress()
    skip |= read_cell_log(OCEAN_LOG) & set(grid)
    remaining = [c for c in grid if c not in skip
                 and missing_windows(progress.get(c, []), args.start, args.end)]
    if args.bbox or args.limit:
        # Combining would overwrite the full-grid CSV with just the test cells.
        print(f"\nTest run (--bbox/--limit): {OUTPUT_CSV} left unchanged. "
              f"Run --combine-only for the full grid.")
        sys.exit(0)
    if remaining:
        print(f"\n{len(remaining):,} cell(s) still missing days -- run the script again. "
              f"See {INCOMPLETE_LOG} for cells with holes.")
        sys.exit(1)
    ok = combine_and_verify(args.start, args.end, grid)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
