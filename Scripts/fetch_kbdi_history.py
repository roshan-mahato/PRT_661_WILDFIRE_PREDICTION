"""
Fetches a CONTINUOUS daily temperature / precipitation series for every
training grid cell, for the KBDI recursion in build_labeled_dataset.py.

data/all_weather_data.csv cannot be used for this: it holds 14-day windows
around fire detections (median 33% of days present, median longest gap
~200 days), and the recursion treats every gap as rain-free days, so the
KBDI the model was trained on is badly inflated. Only the recursion needs
continuity -- every other feature is a same-day snapshot the existing CSV
already has -- so only these two variables are fetched, which also keeps
each multi-year request cheap against Open-Meteo's weighted call limits.

Output: one row per (cell, day) with temperature_2m (mean) and precipitation
(sum), aggregated exactly as the live and training pipelines do.

RESUMABLE: completed cells are logged and skipped on the next run. The free
tier allows roughly 1,600 of these requests per day, so a full run spans two
days. An hourly limit is waited out in place; the daily limit stops the
script with a message -- run it again the next day and it carries on.

    uv run python Scripts/fetch_kbdi_history.py
"""

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from Scripts.prediction_engine import aggregate_to_daily
from Scripts.warm_start_kbdi import HISTORY_START, fetch_kbdi_history

SOURCE_CSV = os.path.join(REPO_ROOT, "data", "all_weather_data.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "data", "kbdi_weather_continuous.csv")
PROGRESS_LOG = os.path.join(REPO_ROOT, "data", "kbdi_weather_continuous_progress.log")


def training_cells() -> list:
    cells = set()
    for chunk in pd.read_csv(SOURCE_CSV, usecols=["lat_round", "lon_round"], chunksize=2_000_000):
        cells |= set(map(tuple, chunk.drop_duplicates().itertuples(index=False)))
    return sorted(cells)


def completed_cells() -> set:
    if not os.path.exists(PROGRESS_LOG):
        return set()
    with open(PROGRESS_LOG) as f:
        return {tuple(map(float, line.split(","))) for line in f if line.strip()}


def main():
    cells = training_cells()
    done = completed_cells()
    todo = [c for c in cells if c not in done]
    end = date.today() - timedelta(days=1)
    print(f"{len(cells)} training cells; {len(done)} already fetched; {len(todo)} to go.")
    print(f"Series: {HISTORY_START} -> {end}")
    if not todo:
        print("Nothing to do.")
        return

    i = 0
    while i < len(todo):
        lat, lon = todo[i]
        try:
            hourly = fetch_kbdi_history(lat, lon, HISTORY_START, end)
        except Exception as exc:  # noqa: BLE001 - client wraps the HTTP error
            text = str(exc).lower()
            if "daily" in text:
                print(f"\nOpen-Meteo daily limit reached after {i} cell(s) this run. "
                      f"Run again tomorrow to continue.", flush=True)
                sys.exit(2)
            if "hourly" in text:
                print(f"  hourly limit hit after {i} cell(s); waiting 61 minutes...", flush=True)
                time.sleep(61 * 60)
                continue  # retry the same cell
            print(f"  FAILED ({lat}, {lon}): {exc}", flush=True)
            i += 1
            continue
        i += 1

        daily = aggregate_to_daily(hourly)
        gaps = daily["acq_date"].diff().dt.days.dropna()
        if len(gaps) and gaps.max() > 1:
            print(f"  SKIPPED ({lat}, {lon}): archive series has a {int(gaps.max())}-day gap")
            continue

        daily = daily[["lat_round", "lon_round", "acq_date", "temperature_2m", "precipitation"]]
        daily.to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)
        with open(PROGRESS_LOG, "a") as f:
            f.write(f"{lat},{lon}\n")

        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} this run ({len(done) + i}/{len(cells)} total)", flush=True)

    print(f"\nDone. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
