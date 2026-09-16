"""
Warm-starts the per-cell KBDI recursion state (Scripts/fire_state.json) from
real weather history, so live predictions begin from an accumulated drought
value instead of a cold start at "soil fully saturated".

KBDI is a running deficit that takes months to reach a realistic level. The
live pipeline cold-starts every cell at 0 on the day it first appears, while
the model was trained on cells whose recursion had run since HISTORY_START --
so at serve time kbdi, drought_factor, ffdi, rate_of_spread and
days_since_cell_start all sit in a regime the model never saw. Seeding the
state from the same period the model trained on puts them back.

For every cell the prediction path currently serves, the full hourly series
from HISTORY_START to the day before the live window is fetched from the
Open-Meteo archive (one call per cell), aggregated to daily exactly as the
live and training pipelines do, and run through the recursion.

The archive is used rather than data/all_weather_data.csv on purpose: that
file holds 14-day windows around fire detections, not a continuous record,
and the recursion treats every gap as rain-free days -- a 200-day hole
inflates KBDI far more than a cold start understates it.

Idempotent: a cell whose state already begins on or before HISTORY_START is
warm and is skipped, so re-running after a wider live fetch only seeds the
newly added cells. --force redoes every cell. Cells that fail (rate limit,
network) are reported and the exit code is non-zero; run again to retry.

Do not run a prediction refresh at the same time: the API rewrites the state
file after each request and would overwrite the seeded values.

    uv run python Scripts/warm_start_kbdi.py [--force]
"""

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from backend.api.prediction import _fetch_weather_live
from Scripts.fetch_openmeteo_historical_data import ARCHIVE_URL, openmeteo
from Scripts.prediction_engine import (
    STATE_PATH,
    aggregate_to_daily,
    cell_key,
    kbdi_df_step,
    load_state,
)

# Start of the training weather series. Seeding from the same origin means
# days_since_cell_start at serve time lands in the range the model saw.
HISTORY_START = date(2024, 1, 1)

# Open-Meteo weights a request by variables x days, so a multi-year pull of
# the full ten-variable set costs ~28 call-units and trips the 600/minute
# limit after ~20 cells. The recursion only reads these two.
KBDI_VARS = ["temperature_2m", "precipitation"]
REQUEST_DELAY_SECONDS = 1.0
MINUTE_LIMIT_RETRIES = 5


def fetch_kbdi_history(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Hourly temperature and precipitation for one cell from the archive.

    A per-minute limit is waited out and retried; an hourly or daily limit
    is raised straight away so the run stops burning cells that will all
    fail the same way -- the caller re-runs later and only the unseeded
    cells are attempted.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start),
        "end_date": str(end),
        "hourly": KBDI_VARS,
        "timezone": "UTC",
    }
    for attempt in range(MINUTE_LIMIT_RETRIES + 1):
        try:
            response = openmeteo.weather_api(ARCHIVE_URL, params=params)[0]
            break
        except Exception as exc:  # noqa: BLE001 - client wraps the HTTP error
            text = str(exc).lower()
            if "minutely" in text and attempt < MINUTE_LIMIT_RETRIES:
                print("  minute limit hit; pausing 60s...")
                time.sleep(60)
                continue
            raise

    hourly = response.Hourly()
    data = {
        "datetime_utc": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, var in enumerate(KBDI_VARS):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["lat_round"] = lat
    df["lon_round"] = lon
    return df


def live_cells() -> dict:
    """{(lat, lon): first live day} for every cell the prediction path serves."""
    live = _fetch_weather_live(hours_back=168)
    if live.empty:
        return {}
    live = live.assign(_day=pd.to_datetime(live["datetime_utc"], utc=True).dt.date)
    return live.groupby(["lat_round", "lon_round"])["_day"].min().to_dict()


def run_recursion(daily: pd.DataFrame) -> dict:
    state = None
    for row in daily.sort_values("acq_date").itertuples(index=False):
        _, state = kbdi_df_step(state, row.acq_date, row.temperature_2m, row.precipitation)
    return state


def seed_cell(lat: float, lon: float, first_live: date) -> dict:
    end = first_live - timedelta(days=1)
    hourly = fetch_kbdi_history(lat, lon, HISTORY_START, end)
    time.sleep(REQUEST_DELAY_SECONDS)

    daily = aggregate_to_daily(hourly)
    if daily.empty:
        raise ValueError("no usable daily rows after aggregation")

    gaps = daily["acq_date"].diff().dt.days.dropna()
    if len(gaps) and gaps.max() > 1:
        raise ValueError(f"archive series has a {int(gaps.max())}-day gap")

    return run_recursion(daily)


def save_state_atomic(all_state: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("For every")[0])
    parser.add_argument("--force", action="store_true", help="Re-seed cells that are already warm.")
    args = parser.parse_args()

    cells = live_cells()
    print(f"{len(cells)} live cell(s) served by the prediction path.")
    if not cells:
        return

    existing = load_state()
    todo = {}
    for cell, first_live in cells.items():
        state = existing.get(cell_key(*cell))
        already_warm = (
            state is not None
            and date.fromisoformat(state["cell_start_date"]) <= HISTORY_START
        )
        if already_warm and not args.force:
            continue
        todo[cell] = first_live
    print(f"{len(todo)} to seed, {len(cells) - len(todo)} already warm.")
    if not todo:
        return

    seeded, failed = {}, []
    for i, ((lat, lon), first_live) in enumerate(todo.items(), start=1):
        try:
            seeded[cell_key(lat, lon)] = seed_cell(lat, lon, first_live)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not abort the batch
            failed.append(((lat, lon), str(exc)))
            print(f"  FAILED ({lat}, {lon}): {exc}")
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done ({len(failed)} failed)")

    # Re-read right before writing so a state save by the API in the
    # meantime is not clobbered for cells this run did not touch.
    all_state = load_state()
    before = {k: all_state.get(k, {}).get("q_prev") for k in seeded}
    all_state.update(seeded)
    save_state_atomic(all_state)

    print(f"\nSeeded {len(seeded)} cell(s) into {STATE_PATH}")
    if seeded:
        print("  cell            start        last         kbdi before -> after")
        for key, st in list(seeded.items())[:5]:
            prev = before[key]
            prev_txt = f"{prev:6.1f}" if prev is not None else "   n/a"
            print(f"  {key:<15} {st['cell_start_date']}   {st['last_date']}   {prev_txt} -> {st['q_prev']:6.1f}")
    if failed:
        print(f"\n{len(failed)} cell(s) failed; run again to retry them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
