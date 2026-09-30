"""run_kbdi() must reproduce kbdi_df_step(), the recursion the model was trained on."""

import numpy as np
import pandas as pd
import pytest

from Scripts.prediction_engine import KBDI_RESULT_COLS, kbdi_df_step, run_kbdi

COMPARED = ["kbdi", "drought_factor", "days_since_rain", "days_since_cell_start",
            "kbdi_spinup_flag"]


def synthetic_weather(seed=0) -> pd.DataFrame:
    """Three cells with different date ranges, a gap, and rain events."""
    rng = np.random.default_rng(seed)
    frames = []
    for cell_id, (start, n) in enumerate([("2024-01-01", 90), ("2024-01-20", 60), ("2024-02-10", 45)]):
        dates = pd.date_range(start, periods=n, freq="D")
        if cell_id == 1:
            dates = dates.delete(slice(20, 26))   # six missing days
        rain = np.where(rng.random(len(dates)) < 0.3, rng.gamma(1.5, 6.0, len(dates)), 0.0)
        frames.append(pd.DataFrame({
            "cell_id": cell_id, "date": dates,
            "temperature_2m": rng.normal(25, 6, len(dates)),
            "precipitation": rain,
        }))
    # Shuffled: run_kbdi must not depend on row order.
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)


def reference(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.sort_values(["cell_id", "date"]).groupby("cell_id"):
        state = None
        for r in g.itertuples():
            res, state = kbdi_df_step(state, r.date, r.temperature_2m, r.precipitation)
            rows.append({"idx": r.Index, **res})
    return pd.DataFrame(rows).set_index("idx")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_run_kbdi_matches_reference(seed):
    df = synthetic_weather(seed)
    got = run_kbdi(df)
    ref = reference(df).loc[got.index]
    for c in COMPARED:
        np.testing.assert_allclose(got[c].astype(float), ref[c].astype(float), rtol=1e-12, err_msg=c)


def test_continuing_from_state_equals_one_run():
    df = synthetic_weather(3)
    full = run_kbdi(df)

    cut = pd.Timestamp("2024-02-20")
    first = run_kbdi(df[df["date"] < cut])
    state = first.sort_values("date").groupby("cell_id").tail(1)
    second = run_kbdi(df[df["date"] >= cut], state)

    both = pd.concat([first, second]).loc[full.index]
    for c in KBDI_RESULT_COLS:
        np.testing.assert_allclose(both[c].astype(float), full[c].astype(float), rtol=1e-12, err_msg=c)


def test_rows_before_state_are_rejected():
    df = synthetic_weather(4)
    state = run_kbdi(df).sort_values("date").groupby("cell_id").tail(1)
    with pytest.raises(ValueError):
        run_kbdi(df, state)


def test_daily_peaks_from_hourly():
    from Scripts.prediction_engine import aggregate_to_daily, compute_ffdi, daily_peaks

    hours = pd.date_range("2025-01-10", periods=48, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    h = pd.DataFrame({
        "datetime_utc": hours, "lat_round": -33.0, "lon_round": 151.0,
        "temperature_2m": rng.uniform(10, 35, 48), "relative_humidity_2m": rng.uniform(10, 90, 48),
        "wind_speed_10m": rng.uniform(0, 40, 48), "wind_gusts_10m": rng.uniform(5, 70, 48),
        "wind_direction_10m": rng.uniform(0, 360, 48), "precipitation": rng.uniform(0, 1, 48),
        "soil_moisture_0_to_7cm": 0.2, "vapour_pressure_deficit": rng.uniform(0, 5, 48),
        "et0_fao_evapotranspiration": rng.uniform(0, 0.7, 48),
    })
    peaks = daily_peaks(h, ["lat_round", "lon_round"])
    assert len(peaks) == 2

    day1 = h.iloc[:24]
    p = peaks.iloc[0]
    assert p["temperature_2m_max"] == day1["temperature_2m"].max()
    assert p["relative_humidity_2m_min"] == day1["relative_humidity_2m"].min()
    assert p["wind_gusts_10m_max"] == day1["wind_gusts_10m"].max()
    assert np.isclose(p["et0_fao_evapotranspiration_sum"], day1["et0_fao_evapotranspiration"].sum())
    assert p["wind_direction_at_max_wind"] == day1.loc[day1["wind_speed_10m"].idxmax(), "wind_direction_10m"]

    # The peak hour is the hour with the highest FFDI for a fixed Drought Factor.
    hourly_ffdi = compute_ffdi(day1["temperature_2m"], day1["relative_humidity_2m"],
                               day1["wind_speed_10m"], 6.0)
    best = day1.loc[hourly_ffdi.idxmax()]
    assert p["peak_fire_hour_utc"] == best["datetime_utc"].hour
    assert p["peak_fire_temperature_2m"] == best["temperature_2m"]

    # aggregate_to_daily keeps the means and adds the peaks.
    daily = aggregate_to_daily(h)
    assert np.isclose(daily.loc[0, "temperature_2m"], day1["temperature_2m"].mean())
    assert daily.loc[0, "temperature_2m_max"] == p["temperature_2m_max"]


def test_ffdi_max_is_never_below_ffdi():
    from Scripts.prediction_engine import aggregate_to_daily, engineer_features

    rng = np.random.default_rng(8)
    hours = pd.date_range("2025-01-01", periods=24 * 20, freq="h", tz="UTC")
    h = pd.DataFrame({
        "datetime_utc": hours, "lat_round": -20.0, "lon_round": 130.0,
        "temperature_2m": rng.uniform(15, 42, len(hours)), "relative_humidity_2m": rng.uniform(5, 95, len(hours)),
        "wind_speed_10m": rng.uniform(0, 45, len(hours)), "wind_gusts_10m": 30.0,
        "wind_direction_10m": 90.0, "precipitation": np.where(rng.random(len(hours)) < 0.05, 3.0, 0.0),
        "soil_moisture_0_to_7cm": 0.1, "vapour_pressure_deficit": 2.0, "et0_fao_evapotranspiration": 0.3,
    })
    daily = aggregate_to_daily(h).rename(columns={"acq_date": "date"})
    daily["cell_id"] = 1
    feats = engineer_features(daily)
    assert (feats["ffdi_max"] >= feats["ffdi"] - 1e-9).all()
    assert (feats["ffdi_max"] > feats["ffdi"]).mean() > 0.9
