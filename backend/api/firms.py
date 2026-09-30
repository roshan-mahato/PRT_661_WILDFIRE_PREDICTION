"""
Cleaning NASA FIRMS VIIRS downloads and storing them in the `nasa_firms` table.

Scripts/combine_firms.py reads the downloaded CSVs through
normalise_firms() and saves them with save_firms().
"""

from typing import Callable

import pandas as pd
from sqlalchemy import text

from backend.database import engine, write_lock
from backend.db_model.nasa_firms import NASAFirms

GRID_SIZE = 0.5
VIIRS_SATELLITES = {"SNPP", "N20", "N21"}

FIRMS_COLS = [
    "latitude", "longitude", "lat_round", "lon_round", "bright_ti4", "bright_ti5",
    "scan", "track", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "version", "frp", "daynight", "type", "source",
]
KEY_COLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
_REQUIRED = ["latitude", "longitude", "scan", "track", "acq_date", "acq_time", "satellite",
             "confidence", "version", "frp", "daynight"]
CHUNK_ROWS = 200_000


def normalise_firms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Puts one FIRMS VIIRS download (archive or NRT) into FIRMS_COLS form.
    Also accepts a file this function already produced, so a combined CSV
    can be loaded again as it is.

    - `brightness` / `bright_t31` -> `bright_ti4` / `bright_ti5` (the VIIRS
      channel names; the FIRMS download uses the MODIS names).
    - `instrument` = VIIRS (the S-NPP download says "SNPP" there).
    - `source` = nrt when `version` ends in NRT, else archive.
    - `type` is kept from the archive; NRT has no such column, so it is NULL.
    - `lat_round` / `lon_round`: the 0.5-degree grid cell, same rounding as
      the weather grid.
    """
    d = df.rename(columns={"brightness": "bright_ti4", "bright_t31": "bright_ti5"})
    missing = [c for c in _REQUIRED + ["bright_ti4", "bright_ti5"] if c not in d.columns]
    if missing:
        raise ValueError(f"Not a FIRMS VIIRS file, missing columns: {missing}")
    other = set(d["satellite"].astype(str).unique()) - VIIRS_SATELLITES
    if other:
        raise ValueError(f"Only VIIRS (SNPP, N20, N21) is supported, found satellite(s) {sorted(other)}")

    out = pd.DataFrame({
        "latitude": d["latitude"].astype(float),
        "longitude": d["longitude"].astype(float),
        "bright_ti4": d["bright_ti4"].astype(float),
        "bright_ti5": d["bright_ti5"].astype(float),
        "scan": d["scan"].astype(float),
        "track": d["track"].astype(float),
        "acq_date": pd.to_datetime(d["acq_date"]).dt.strftime("%Y-%m-%d"),
        "acq_time": d["acq_time"].astype(int),
        "satellite": d["satellite"].astype(str),
        "instrument": "VIIRS",
        "confidence": d["confidence"].astype(str),
        "version": d["version"].astype(str),
        "frp": d["frp"].astype(float),
        "daynight": d["daynight"].astype(str),
        "type": d["type"].astype("Int64") if "type" in d.columns else pd.array([pd.NA] * len(d), "Int64"),
    })
    out["source"] = (d["source"].astype(str) if "source" in d.columns else
                     out["version"].str.upper().str.endswith("NRT").map({True: "nrt", False: "archive"}))
    out["lat_round"] = ((out["latitude"] / GRID_SIZE).round() * GRID_SIZE).round(3)
    out["lon_round"] = ((out["longitude"] / GRID_SIZE).round() * GRID_SIZE).round(3)
    return out[FIRMS_COLS]


def archive_ranges(df: pd.DataFrame) -> dict:
    """{satellite: (first_date, last_date)} of the archive rows in `df`."""
    a = df[df["source"] == "archive"]
    return {s: (g["acq_date"].min(), g["acq_date"].max()) for s, g in a.groupby("satellite")}


def drop_superseded_nrt(df: pd.DataFrame) -> tuple:
    """
    Drops NRT rows for days that satellite's archive also covers: the archive
    is the reprocessed, classified version of the same overpasses (positions
    shift slightly, so the rows would not match as duplicates).
    Returns (df, rows dropped).
    """
    keep = pd.Series(True, index=df.index)
    for sat, (lo, hi) in archive_ranges(df).items():
        keep &= ~((df["source"] == "nrt") & (df["satellite"] == sat)
                  & (df["acq_date"] >= lo) & (df["acq_date"] <= hi))
    return df[keep], int((~keep).sum())


def ensure_firms_table(log: Callable[[str], None] = print):
    """
    Creates nasa_firms, or recreates it when it has the old column layout.
    A non-empty table with the old layout is left alone and raises: its rows
    would be lost.
    """
    with engine.begin() as conn:
        have = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(nasa_firms)")}
        want = {c.name for c in NASAFirms.__table__.columns}
        if have and not want <= have:
            n = conn.exec_driver_sql("SELECT COUNT(*) FROM nasa_firms").scalar()
            if n:
                raise RuntimeError(
                    f"nasa_firms has the old column layout and {n:,} rows; back it up and "
                    f"drop it, then run again.")
            conn.exec_driver_sql("DROP TABLE nasa_firms")
            log("nasa_firms: empty table with the old layout dropped.")
    NASAFirms.__table__.create(engine, checkfirst=True)


def save_firms(df: pd.DataFrame, log: Callable[[str], None] = print) -> dict:
    """
    Saves normalised detections. Duplicates of stored rows (same position,
    time and satellite) are skipped, and stored NRT rows on days a
    satellite's archive in `df` covers are deleted first, so loading a newer
    archive download upgrades the NRT days it overlaps.

    Returns {"inserted", "skipped", "nrt_replaced"}.
    """
    ensure_firms_table(log)
    replaced = 0
    with write_lock, engine.begin() as conn:
        for sat, (lo, hi) in archive_ranges(df).items():
            replaced += conn.execute(text(
                "DELETE FROM nasa_firms WHERE source = 'nrt' AND satellite = :s "
                "AND acq_date BETWEEN :lo AND :hi"), {"s": sat, "lo": lo, "hi": hi}).rowcount

    rows = df[FIRMS_COLS].astype(object).where(df[FIRMS_COLS].notna(), None)
    sql = (f"INSERT OR IGNORE INTO nasa_firms ({', '.join(FIRMS_COLS)}, created_at) "
           f"VALUES ({', '.join('?' * len(FIRMS_COLS))}, datetime('now'))")
    inserted = 0
    for i in range(0, len(rows), CHUNK_ROWS):
        chunk = list(rows.iloc[i:i + CHUNK_ROWS].itertuples(index=False, name=None))
        with write_lock, engine.begin() as conn:
            inserted += max(conn.exec_driver_sql(sql, chunk).rowcount, 0)
        log(f"  nasa_firms: {min(i + CHUNK_ROWS, len(rows)):,}/{len(rows):,} rows processed, "
            f"{inserted:,} new")
    return {"inserted": inserted, "skipped": len(rows) - inserted, "nrt_replaced": replaced}


def firms_summary() -> pd.DataFrame:
    """Rows and date range per satellite and source in the table."""
    with engine.connect() as conn:
        return pd.read_sql(text(
            "SELECT satellite, source, COUNT(*) AS rows, MIN(acq_date) AS first_day, "
            "MAX(acq_date) AS last_day, COUNT(DISTINCT acq_date) AS days "
            "FROM nasa_firms GROUP BY satellite, source ORDER BY satellite, source"), conn)
