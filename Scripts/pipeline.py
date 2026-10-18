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
    setup          first run on a new machine, safe to repeat: init, load
                   the history CSV if the database has none, then daily.
                   setup.ps1 (repo root) runs it after the prerequisites.
    daily          the scheduled job: start Open-Meteo if it is down, run
                   fetch_weather_live.py (history to yesterday + forecast +
                   predictions), then prune. Logs to logs/daily_<date>.log.
                   Scripts/schedule_daily.ps1 registers it with Windows Task
                   Scheduler.

USAGE
-----
    uv run python Scripts/pipeline.py init
    uv run python Scripts/pipeline.py load-history                 # data/full_grid_parts/
    uv run python Scripts/pipeline.py load-history --csv data/full_grid_daily_weather.csv
    uv run python Scripts/pipeline.py predict
    uv run python Scripts/pipeline.py status
    uv run python Scripts/pipeline.py prune --keep-days 90
    uv run python Scripts/pipeline.py daily
    uv run python Scripts/pipeline.py setup
================================================================================
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from sqlalchemy import distinct, func, select, update

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.api import weather  
from backend.api.features import build_archive_features, build_run_features  
from backend.api.fire_prediction import predict_run, prune_forecasts  
from backend.database import engine  
from backend.db_init import init_db
from backend.db_model.feature_engineered import FeatureEngineered
from backend.db_model.fetch_run import FetchRun
from backend.db_model.prediction import Prediction

DEFAULT_PARTS_DIR = os.path.join(REPO_ROOT, "data", "full_grid_parts")
CSV_CHUNK_ROWS = 200_000
HISTORY_CSV = os.path.join(REPO_ROOT, "data", "full_grid_daily_weather.csv")
LAND_MASK_CSV = os.path.join(REPO_ROOT, "data", "full_grid_land_mask.csv")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
LIVE_SCRIPT = os.path.join(REPO_ROOT, "Scripts", "fetch_weather_live.py")
SERVER_POLL_SECONDS = 5


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
        conn.execute(
            update(FetchRun).where(FetchRun.run_id == run_id).values(
                window_start=None if lo is None else lo.date(),
                window_end=None if hi is None else hi.date())
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
    p, f = Prediction, FeatureEngineered
    with engine.connect() as conn:
        cur = pd.read_sql(
            select(func.min(p.date).label("first_day"), func.max(p.date).label("last_day"),
                   func.count().label("rows"), func.count(distinct(p.cell_id)).label("cells"),
                   func.max(p.run_id).label("latest_run"))
            .where(p.is_current), conn)
        arch = pd.read_sql(
            select(func.min(f.date).label("first_day"), func.max(f.date).label("last_day"),
                   func.count(distinct(f.cell_id)).label("cells"))
            .where(f.source == "archive"), conn)
    print("\nArchive features:")
    print(arch.to_string(index=False))
    print("\nCurrent predictions:")
    print(cur.to_string(index=False))


def cmd_prune(args):
    prune_forecasts(keep_days=args.keep_days)


# ==============================================================================
# daily -- the scheduled job
# ==============================================================================
class DailyLog:
    """Prints to the console and appends to logs/daily_<date>.log."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def server_up(url: str) -> bool:
    try:
        requests.get(url, timeout=5)   # any response, even a 400, means it is up
        return True
    except requests.exceptions.RequestException:
        return False


def ensure_server(url: str, wait_seconds: int, log) -> bool:
    """
    For the self-hosted server: if it is down, `docker compose up -d` and
    wait up to `wait_seconds` for it to answer. Returns whether it is up.
    """
    if server_up(url):
        return True
    log(f"Open-Meteo is not answering at {url} -- running `docker compose up -d`.")
    try:
        r = subprocess.run(["docker", "compose", "up", "-d"], cwd=REPO_ROOT,
                           capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            log(f"  docker: {line}")
    except FileNotFoundError:
        log("  docker is not installed or not on PATH.")
        return False
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(SERVER_POLL_SECONDS)
        if server_up(url):
            log("Open-Meteo is up.")
            return True
    log(f"Open-Meteo still not answering after {wait_seconds}s (is Docker Desktop running?).")
    return False


def run_logged(cmd: list, log) -> int:
    """Runs `cmd`, streaming its output to `log`. Returns the exit code."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace") as proc:
        for line in proc.stdout:
            log(line.rstrip("\n"))
    return proc.returncode


def cmd_daily(args):
    """
    Fetch (history to yesterday, 7-day forecast, predictions), then prune.
    The prune runs even if the fetch failed: it never touches the rows behind
    current predictions, so it is safe either way. Exits with the fetch's
    exit code (0 = every cell refreshed).
    """
    from Scripts.fetch_weather_history import is_public
    from Scripts.fetch_weather_live import FORECAST_URL

    log = DailyLog(os.path.join(LOG_DIR, f"daily_{datetime.now():%Y-%m-%d}.log"))
    t0 = time.time()
    log("=" * 74)
    log(f"DAILY RUN  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 74)
    try:
        if not is_public(FORECAST_URL) and not ensure_server(FORECAST_URL, args.wait, log):
            log("Daily run aborted: no weather server. Current predictions are unchanged.")
            sys.exit(2)

        rc = run_logged([sys.executable, LIVE_SCRIPT], log)
        if rc:
            log(f"Fetch exited {rc}: some or all cells were not refreshed (see above); "
                f"their previous predictions stay current.")
        prune_forecasts(keep_days=args.keep_days, log=log)
        log(f"Daily run finished in {(time.time() - t0) / 60:,.1f} min, exit {rc}.")
        log(f"Log: {log.path}\n")
        sys.exit(rc)
    finally:
        log.close()


def is_lfs_pointer(path: str) -> bool:
    """True if `path` is a Git LFS pointer (the real file was never pulled)."""
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def cmd_setup(args):
    """
    First run on a new machine, safe to repeat: tables and land mask, the
    weather history from the combined CSV when the database has none, then
    the daily job (history up to yesterday, 7-day forecast, predictions,
    prune). Exits like `daily`.
    """
    for path in (LAND_MASK_CSV, args.csv):
        if not os.path.exists(path) or is_lfs_pointer(path):
            sys.exit(f"{os.path.relpath(path, REPO_ROOT)} is missing or still a Git LFS pointer. "
                     f"Run `git lfs pull` in the repo, then try again.")
    init_db()
    last = weather.archive_last_dates()
    if last:
        print(f"Weather history already loaded: {len(last):,} cells, "
              f"up to {max(last.values()).date()}.")
    else:
        print(f"No weather history in the database yet -- loading "
              f"{os.path.relpath(args.csv, REPO_ROOT)} (several minutes).")
        cmd_load_history(argparse.Namespace(csv=args.csv, parts=DEFAULT_PARTS_DIR))
    cmd_daily(args)


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

    for name, func, help_text in (
        ("daily", cmd_daily, "Scheduled job: fetch + predict, then prune"),
        ("setup", cmd_setup, "First run: init, load the history CSV if needed, then daily"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--keep-days", type=int, default=90, help="Prune forecast runs older than this")
        p.add_argument("--wait", type=int, default=300,
                       help="Seconds to wait for Open-Meteo after starting it")
        p.set_defaults(func=func)
    p.add_argument("--csv", default=HISTORY_CSV, help="History to load when the database has none")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
