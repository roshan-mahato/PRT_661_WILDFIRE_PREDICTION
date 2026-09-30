"""Combining FIRMS downloads and loading them into nasa_firms (temporary DB from conftest)."""

import pandas as pd
import pytest

from backend.api.firms import drop_superseded_nrt, firms_summary, normalise_firms, save_firms
from backend.database import engine
from backend.db_init import create_tables
from backend.db_model.base import Base


def download(satellite, dates, nrt=False, lat0=-20.0) -> pd.DataFrame:
    """A FIRMS-style download: MODIS column names, NRT without `type`."""
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "latitude": lat0 + i * 0.01, "longitude": 130.26, "brightness": 330.0, "scan": 0.4,
            "track": 0.4, "acq_date": d, "acq_time": 414, "satellite": satellite,
            "instrument": "SNPP" if satellite == "SNPP" else "VIIRS", "confidence": "n",
            "version": "2.0NRT" if nrt else "2", "bright_t31": 295.0, "frp": 5.0, "daynight": "D",
            **({} if nrt else {"type": 0}),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module", autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    create_tables()


def test_normalise():
    a = normalise_firms(download("SNPP", ["2026-06-30"]))
    n = normalise_firms(download("N20", ["2026-07-01"], nrt=True))
    assert a.loc[0, "bright_ti4"] == 330.0 and a.loc[0, "bright_ti5"] == 295.0
    assert a.loc[0, "instrument"] == "VIIRS" and a.loc[0, "source"] == "archive"
    assert (a.loc[0, "lat_round"], a.loc[0, "lon_round"]) == (-20.0, 130.5)
    assert n.loc[0, "source"] == "nrt" and pd.isna(n.loc[0, "type"])
    # A combined file loads back unchanged.
    pd.testing.assert_frame_equal(normalise_firms(a), a)


def test_non_viirs_rejected():
    with pytest.raises(ValueError):
        normalise_firms(download("Terra", ["2026-01-01"]))


def test_archive_supersedes_nrt():
    df = pd.concat([normalise_firms(download("SNPP", ["2026-07-01", "2026-07-02"])),
                    normalise_firms(download("SNPP", ["2026-07-02", "2026-07-03"], nrt=True, lat0=-21)),
                    normalise_firms(download("N20", ["2026-07-02"], nrt=True, lat0=-22))])
    kept, dropped = drop_superseded_nrt(df)
    assert dropped == 1                              # SNPP NRT on 07-02 only
    assert set(kept.loc[kept["source"] == "nrt", "acq_date"]) == {"2026-07-02", "2026-07-03"}


def test_save_is_idempotent_and_upgrades_nrt():
    nrt = normalise_firms(download("SNPP", ["2026-08-01", "2026-08-02", "2026-08-03"], nrt=True))
    assert save_firms(nrt)["inserted"] == 3
    assert save_firms(nrt)["inserted"] == 0          # same rows again: skipped

    # A later archive for 08-01..08-02 replaces those NRT days.
    arch = normalise_firms(download("SNPP", ["2026-08-01", "2026-08-02"], lat0=-25))
    r = save_firms(arch)
    assert r == {"inserted": 2, "skipped": 0, "nrt_replaced": 2}
    s = firms_summary().set_index("source")
    assert s.loc["archive", "rows"] == 2 and s.loc["nrt", "rows"] == 1


def test_old_empty_table_is_recreated():
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE nasa_firms")
        conn.exec_driver_sql("CREATE TABLE nasa_firms (id INTEGER PRIMARY KEY, latitude FLOAT)")
    assert save_firms(normalise_firms(download("N20", ["2026-01-01"])))["inserted"] == 1
