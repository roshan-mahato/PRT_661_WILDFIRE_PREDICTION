"""
================================================================================
pipeline.py  --  manage the weather -> features -> prediction tables
================================================================================
The fetch scripts fill the database themselves:

    fetch_weather_history.py   archive weather + archive features
    fetch_weather_live.py      forecast weather + features + predictions

This script covers everything else.

    init           create missing tables and load the land mask into grid_cell
                   (--drop-legacy also drops the old openmeteo / fire_prediction)
    load-history   load history part files (or a combined CSV) that were
                   fetched without the DB, then build their features
    features       build features for archive rows that have none
    predict        build features + predictions for a forecast run (default:
                   the latest) -- e.g. after retraining the model
    status         row counts, recent fetch runs, current predictions
    prune          delete forecast runs older than --keep-days (keeps the
                   rows behind current predictions; archive is never touched)

USAGE
-----
    uv run python Scripts/pipeline.py init
    uv run python Scripts/pipeline.py load-history                 # data/full_grid_parts/
    uv run python Scripts/pipeline.py load-history --csv data/full_grid_daily_weather.csv
    uv run python Scripts/pipeline.py predict
    uv run python Scripts/pipeline.py status
    uv run python Scripts/pipeline.py prune --keep-days 90
================================================================================
"""

import argparse
import glob
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.api import weather  # noqa: E402
from backend.api.features import build_archive_features, build_run_features  # noqa: E402
from backend.api.fire_prediction import predict_run, prune_forecasts  # noqa: E402
from backend.database import engine  # noqa: E402
from backend.db_init import init_db  # noqa: E402

DEFAULT_PARTS_DIR = os.path.join(REPO_ROOT, "data", "full_grid_parts")
CSV_CHUNK_ROWS = 200_000


def cmd_init(args):
    init_db(drop_legacy=args.drop_legacy)


def _history_frames(args):
    """Yields DataFrames of daily archive rows, NEWEST source first."""
    if args.csv:
        yield from pd.read_csv(args.csv, chunksize=CSV_CHUNK_ROWS)
        return
    parts = sorted(glob.glob(os.path.join(args.parts, "part_*.csv")),
                   key=os.path.getmtime, reverse=True)
    if not parts:
        sys.exit(f"No part files in {args.parts}")
    print(f"Loading {len(parts):,} part file(s) from {args.parts}")
    for p in parts:
        yield pd.read_csv(p)


def cmd_load_history(args):
    """
    Archive rows are unique per cell-day and duplicates are skipped, so the
    newest part file is loaded first: a re-fetched batch wins over an older
    one, as in fetch_weather_history.py's combine step.
    """
    init_db()
    t0 = time.time()
    run_id = weather.start_run("archive", note=f"loaded from {args.csv or args.parts}")
    total = seen = 0
    lo = hi = None
    try:
        for df in _history_frames(args):
            if df.empty:
                continue
            seen += len(df)
            total += weather.save_weather_daily(df, run_id, "archive")
            dates = pd.to_datetime(df["acq_date"])
            lo = dates.min() if lo is None else min(lo, dates.min())
            hi = dates.max() if hi is None else max(hi, dates.max())
            print(f"  {seen:,} rows read, {total:,} new ({time.time() - t0:,.0f}s)", flush=True)
    except BaseException:
        weather.finish_run(run_id, "failed")
        raise
    n = weather.finish_run(run_id, "ok")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE fetch_run SET window_start = ?, window_end = ? WHERE run_id = ?",
            (None if lo is None else str(lo.date()), None if hi is None else str(hi.date()), run_id),
        )
    print(f"Archive run {run_id}: {n:,} new weather rows ({seen - total:,} already stored).")
    build_archive_features()
    print(f"Done in {time.time() - t0:,.0f}s.")


def cmd_features(_args):
    build_archive_features()


def cmd_predict(args):
    run_id = args.run_id or weather.latest_forecast_run()
    if run_id is None:
        sys.exit("No forecast run in the database -- run Scripts/fetch_weather_live.py first.")
    print(f"Forecast run {run_id}: {build_run_features(run_id):,} new feature rows.")
    predict_run(run_id)


def cmd_status(_args):
    print("Rows:")
    for t, n in weather.table_counts().items():
        print(f"  {t:20s} {n:>12,}")
    print("\nRecent fetch runs:")
    runs = weather.list_runs(10)
    print(runs.to_string(index=False) if not runs.empty else "  none")
    with engine.connect() as conn:
        cur = pd.read_sql(
            "SELECT MIN(date) AS first_day, MAX(date) AS last_day, COUNT(*) AS rows, "
            "COUNT(DISTINCT cell_id) AS cells, MAX(run_id) AS latest_run "
            "FROM prediction WHERE is_current = 1", conn)
        arch = pd.read_sql(
            "SELECT MIN(date) AS first_day, MAX(date) AS last_day, "
            "COUNT(DISTINCT cell_id) AS cells FROM feature_engineered WHERE source = 'archive'",
            conn)
    print("\nArchive features:")
    print(arch.to_string(index=False))
    print("\nCurrent predictions:")
    print(cur.to_string(index=False))


def cmd_prune(args):
    prune_forecasts(keep_days=args.keep_days)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create tables and load the land mask")
    p.add_argument("--drop-legacy", action="store_true",
                   help="Also drop the old openmeteo and fire_prediction tables")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("load-history", help="Load history part files / CSV into weather_daily")
    p.add_argument("--parts", default=DEFAULT_PARTS_DIR, help="Directory of part_*.csv files")
    p.add_argument("--csv", default=None, help="A combined daily CSV instead of part files")
    p.set_defaults(func=cmd_load_history)

    sub.add_parser("features", help="Build missing archive features").set_defaults(func=cmd_features)

    p = sub.add_parser("predict", help="Features + predictions for a forecast run")
    p.add_argument("--run-id", type=int, default=None, help="Default: the latest forecast run")
    p.set_defaults(func=cmd_predict)

    sub.add_parser("status", help="Row counts and recent runs").set_defaults(func=cmd_status)

    p = sub.add_parser("prune", help="Delete old forecast runs")
    p.add_argument("--keep-days", type=int, default=90)
    p.set_defaults(func=cmd_prune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
