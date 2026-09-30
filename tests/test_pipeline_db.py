"""
End-to-end check of the weather -> features -> prediction tables, on the
temporary database set up in tests/conftest.py.
"""

import numpy as np
import pandas as pd
import pytest

from backend.api.features import build_archive_features, build_run_features
from backend.api.fire_prediction import get_current_predictions, predict_run, prune_forecasts
from backend.api.weather import WEATHER_VARS, finish_run, save_weather_daily, start_run
from backend.database import engine
from backend.db_init import create_tables
from backend.db_model.base import Base
from backend.routers.prediction import _to_response
from Scripts.prediction_engine import run_kbdi

CELL_A, CELL_B, CELL_COLD = (-33.0, 151.0), (-33.5, 151.0), (-34.0, 150.5)


def weather(cells, start, end, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for lat, lon in cells:
        for d in dates:
            rows.append({
                "lat_round": lat, "lon_round": lon, "acq_date": d,
                "temperature_2m": rng.normal(24, 5), "relative_humidity_2m": rng.uniform(15, 90),
                "wind_speed_10m": rng.uniform(2, 40), "wind_direction_10m": rng.uniform(0, 360),
                "precipitation": rng.gamma(1.2, 5) if rng.random() < 0.3 else 0.0,
                "wind_gusts_10m": rng.uniform(5, 70), "soil_moisture_0_to_7cm": rng.uniform(0.05, 0.4),
                "vapour_pressure_deficit": rng.uniform(0.2, 4), "et0_fao_evapotranspiration": rng.uniform(0, 0.8),
            })
    return pd.DataFrame(rows)


def with_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Adds afternoon peaks consistent with the daily means."""
    df = df.copy()
    df["temperature_2m_max"] = df["temperature_2m"] + 6
    df["relative_humidity_2m_min"] = (df["relative_humidity_2m"] - 20).clip(lower=5)
    df["wind_speed_10m_max"] = df["wind_speed_10m"] * 1.5
    df["wind_gusts_10m_max"] = df["wind_gusts_10m"] * 1.4
    df["vapour_pressure_deficit_max"] = df["vapour_pressure_deficit"] * 1.8
    df["et0_fao_evapotranspiration_sum"] = df["et0_fao_evapotranspiration"] * 24
    df["peak_fire_hour_utc"] = 5
    df["peak_fire_temperature_2m"] = df["temperature_2m"] + 5
    df["peak_fire_relative_humidity_2m"] = (df["relative_humidity_2m"] - 18).clip(lower=5)
    df["peak_fire_wind_speed_10m"] = df["wind_speed_10m"] * 1.3
    df["wind_direction_at_max_wind"] = df["wind_direction_10m"]
    return df


def query(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def load(source, df) -> int:
    run = start_run(source, df["acq_date"].min(), df["acq_date"].max())
    save_weather_daily(df, run, source)
    finish_run(run, "ok")
    return run


@pytest.fixture(scope="module", autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    create_tables()


def test_archive_features_continue_across_loads():
    first = weather([CELL_A, CELL_B], "2026-06-01", "2026-07-30", seed=1)
    load("archive", first)
    assert build_archive_features() == 120

    # One day overlaps the stored archive and must be skipped.
    second = weather([CELL_A, CELL_B], "2026-07-30", "2026-08-09", seed=2)
    run = start_run("archive")
    assert save_weather_daily(second, run, "archive") == 20
    finish_run(run, "ok")
    assert build_archive_features() == 20

    # Same values as one recursion over the whole stored series.
    stored = query("""SELECT w.cell_id, w.date, w.temperature_2m, w.precipitation, f.kbdi,
                             f.drought_factor, f.days_since_rain
                      FROM feature_engineered f JOIN weather_daily w USING (weather_id)
                      WHERE f.source = 'archive'""")
    expected = run_kbdi(stored[["cell_id", "date", "temperature_2m", "precipitation"]])
    for c in ("kbdi", "drought_factor", "days_since_rain"):
        np.testing.assert_allclose(stored[c], expected[c], rtol=1e-12)


def test_archive_peaks_filled_in_later():
    """History stored before the peak columns existed gets them on a re-load."""
    before = query("SELECT weather_id, temperature_2m FROM weather_daily WHERE source = 'archive'")
    again = with_peaks(weather([CELL_A, CELL_B], "2026-06-01", "2026-07-30", seed=1))
    run = start_run("archive")
    assert save_weather_daily(again, run, "archive") == 120      # filled in, not inserted
    finish_run(run, "ok")
    build_archive_features()

    after = query("""SELECT w.weather_id, w.temperature_2m, w.temperature_2m_max, f.ffdi, f.ffdi_max
                     FROM weather_daily w JOIN feature_engineered f USING (weather_id)
                     WHERE w.source = 'archive'""")
    assert len(after) == len(before) == 140                       # no new rows
    assert (after.set_index("weather_id")["temperature_2m"]
            == before.set_index("weather_id")["temperature_2m"]).all()   # means unchanged
    filled = after[after["temperature_2m_max"].notna()]
    assert len(filled) == 120 and filled["ffdi_max"].notna().all()
    assert (filled["ffdi_max"] >= filled["ffdi"]).all()
    assert after.loc[after["temperature_2m_max"].isna(), "ffdi_max"].isna().all()


def test_forecast_runs_and_current_predictions():
    # Overlaps the archive by one day (skipped), plus a cell with no history.
    f1 = load("forecast", weather([CELL_A, CELL_B, CELL_COLD], "2026-08-09", "2026-08-16", seed=3))
    assert build_run_features(f1) == 7 + 7 + 8
    assert predict_run(f1) == 22

    # The first forecast day continues from the last archive day.
    arch_last = query("""SELECT cell_id, kbdi FROM feature_engineered
                         WHERE source = 'archive' AND date = '2026-08-09'""")
    fc_first = query("""SELECT cell_id, kbdi, days_since_cell_start FROM feature_engineered
                        WHERE source = 'forecast' AND date = '2026-08-10'
                          AND cell_id IN (SELECT cell_id FROM feature_engineered WHERE source = 'archive')""")
    assert (fc_first["days_since_cell_start"] == 70).all()
    assert fc_first["kbdi"].notna().all() and len(arch_last) == 2

    # A later run where CELL_B failed: its old predictions stay current.
    f2 = load("forecast", with_peaks(weather([CELL_A, CELL_COLD], "2026-08-11", "2026-08-18", seed=4)))
    build_run_features(f2)
    f2_feats = query(f"""SELECT f.ffdi, f.ffdi_max FROM feature_engineered f
                         JOIN weather_daily w USING (weather_id) WHERE w.run_id = {f2}""")
    assert f2_feats["ffdi_max"].notna().all() and (f2_feats["ffdi_max"] >= f2_feats["ffdi"]).all()
    assert predict_run(f2) == 16

    cur = query("""SELECT g.lat_round, g.lon_round, p.date, p.run_id FROM prediction p
                   JOIN grid_cell g USING (cell_id) WHERE p.is_current = 1""")
    assert not cur.duplicated(["lat_round", "lon_round", "date"]).any()
    assert len(cur) == (1 + 8) + 7 + (2 + 8)
    is_b = (cur["lat_round"] == CELL_B[0]) & (cur["lon_round"] == CELL_B[1])
    expected_run = np.where(~is_b & (cur["date"] >= "2026-08-11"), f2, f1)
    assert (cur["run_id"] == expected_run).all()

    # Re-predicting the older run adds nothing and does not take "current" back.
    assert predict_run(f1) == 0
    assert query("SELECT COUNT(*) AS n FROM prediction WHERE is_current = 1 AND run_id = %d" % f2)["n"][0] == 16

    one_cell = get_current_predictions(latitude=-33.02, longitude=150.98)
    assert len(one_cell) == 9 and one_cell["acq_date"].is_unique
    assert _to_response(get_current_predictions(), hours_back=0).count == 26


def test_prune_keeps_current_predictions():
    before = query("SELECT COUNT(*) AS n FROM prediction WHERE is_current = 1")["n"][0]
    counts = prune_forecasts(keep_days=0)
    assert counts["prediction"] == 12            # the superseded F1 rows
    assert query("SELECT COUNT(*) AS n FROM prediction WHERE is_current = 1")["n"][0] == before
    assert query("SELECT COUNT(*) AS n FROM prediction WHERE is_current = 0")["n"][0] == 0
    assert query("SELECT COUNT(*) AS n FROM weather_daily WHERE source = 'archive'")["n"][0] == 140
