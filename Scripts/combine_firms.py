"""
================================================================================
combine_firms.py  --  merge NASA FIRMS VIIRS downloads into one file + the DB
================================================================================
Reads every FIRMS VIIRS CSV (archive and NRT, any of S-NPP / NOAA-20 /
NOAA-21), puts them into one table layout, and writes:

    data/firms_combined.csv          one row per detection, all satellites
    nasa_firms table (wildfire_db.db)

WHAT HAPPENS TO THE ROWS
------------------------
- Column names unified: brightness -> bright_ti4, bright_t31 -> bright_ti5,
  instrument = VIIRS, source = archive / nrt (from `version`), type = NULL on
  NRT rows (only the archive classifies detections), plus the 0.5-degree
  grid cell (lat_round, lon_round).
- Exact repeats (same position, date, time and satellite) are dropped.
- NRT rows on days that satellite's archive also covers are dropped: the
  archive is the reprocessed version of the same overpasses.
- Nothing is filtered by confidence or type -- every detection is kept and
  the labelling step decides what counts as a fire.

The DB load skips detections that are already stored, so running it again
(or with a newer download) only adds what is new; a newer archive also
replaces the stored NRT rows of the days it covers.

COVERAGE CHECK
--------------
Prints, per satellite, the days inside the date range with no detection at
all (satellite outages), and the days NO satellite covers. A day with no
detection anywhere in Australia is an outage, not a fire-free day.

USAGE
-----
    uv run python Scripts/combine_firms.py                     # data/DL_FIRE_*/fire_*.csv
    uv run python Scripts/combine_firms.py path/a.csv path/b.csv
    uv run python Scripts/combine_firms.py --no-db             # CSV only
    uv run python Scripts/combine_firms.py data/firms_combined.csv --no-csv   # load a combined file
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

from backend.api.firms import (  # noqa: E402
    KEY_COLS, drop_superseded_nrt, firms_summary, normalise_firms, save_firms,
)

DEFAULT_GLOB = os.path.join(REPO_ROOT, "data", "DL_FIRE_*", "fire_*.csv")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "data", "firms_combined.csv")


def read_all(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = normalise_firms(pd.read_csv(p))
        print(f"  {os.path.relpath(p, REPO_ROOT)}: {len(d):,} rows | "
              f"{', '.join(sorted(d['satellite'].unique()))} {', '.join(sorted(d['source'].unique()))} | "
              f"{d['acq_date'].min()} -> {d['acq_date'].max()}")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def combine(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    # Archive first, so a repeat of the same detection keeps the archive row.
    df = df.sort_values("source").drop_duplicates(KEY_COLS, keep="first")
    print(f"Duplicates removed: {n - len(df):,}")
    df, n_nrt = drop_superseded_nrt(df)
    print(f"NRT rows superseded by the archive: {n_nrt:,}")
    return df.sort_values(["acq_date", "acq_time", "satellite", "latitude", "longitude"]).reset_index(drop=True)


def coverage_report(df: pd.DataFrame) -> None:
    dates = pd.to_datetime(df["acq_date"])
    full = pd.date_range(dates.min(), dates.max())
    print(f"\nCoverage {full[0].date()} -> {full[-1].date()} ({len(full):,} days)")
    covered_any = set()
    for sat, g in df.groupby("satellite"):
        days = set(pd.to_datetime(g["acq_date"].unique()))
        covered_any |= days
        sat_full = pd.date_range(min(days), max(days))
        missing = sat_full.difference(pd.DatetimeIndex(sorted(days)))
        print(f"  {sat:5s} {min(days).date()} -> {max(days).date()} | {len(g):,} detections | "
              f"days with none: {len(missing)}"
              + (f" (first: {', '.join(missing.strftime('%Y-%m-%d')[:6])})" if len(missing) else ""))
    none = full.difference(pd.DatetimeIndex(sorted(covered_any)))
    print(f"  days no satellite covers: {len(none)}"
          + (f" -> {', '.join(none.strftime('%Y-%m-%d'))}" if len(none) else ""))
    print("\nBy source:")
    print(df.groupby(["satellite", "source"]).agg(
        rows=("acq_date", "size"), first_day=("acq_date", "min"), last_day=("acq_date", "max")
    ).to_string())
    print("\nConfidence:", df["confidence"].value_counts().to_dict(),
          "| type:", df["type"].value_counts(dropna=False).to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help=f"FIRMS CSVs (default: {os.path.relpath(DEFAULT_GLOB, REPO_ROOT)})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Combined CSV to write")
    parser.add_argument("--no-csv", dest="write_csv", action="store_false", help="Do not write the combined CSV")
    parser.add_argument("--no-db", dest="save_db", action="store_false", help="Do not load into the database")
    args = parser.parse_args()

    paths = args.files or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        sys.exit(f"No FIRMS files found ({DEFAULT_GLOB}).")

    t0 = time.time()
    print("=" * 74)
    print("COMBINE NASA FIRMS")
    print("=" * 74)
    df = combine(read_all(paths))
    print(f"Combined: {len(df):,} detections")
    coverage_report(df)

    if args.write_csv:
        tmp = args.output + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, args.output)
        print(f"\nCSV: {args.output}")

    if args.save_db:
        from backend.database import DATABASE_URL
        print(f"\nDB: {DATABASE_URL}")
        r = save_firms(df)
        print(f"DB: {r['inserted']:,} inserted, {r['skipped']:,} already stored, "
              f"{r['nrt_replaced']:,} old NRT rows replaced by archive.")
        print(firms_summary().to_string(index=False))
    print(f"\nDone in {time.time() - t0:,.0f}s.")


if __name__ == "__main__":
    main()
