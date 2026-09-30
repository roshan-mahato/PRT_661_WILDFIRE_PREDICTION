"""Labelling rules of Scripts/build_labeled_dataset.py."""

import numpy as np
import pandas as pd

from Scripts.build_labeled_dataset import fire_cell_days, label_all, observed_days

D = pd.Timestamp


def det(date, sat="SNPP", conf="n", typ=0.0, dn="D", lat=-20.0, lon=130.0):
    return {"lat_round": lat, "lon_round": lon, "acq_date": D(date), "satellite": sat,
            "confidence": conf, "type": typ, "daynight": dn}


def test_fire_cell_days_filters():
    f = pd.DataFrame([
        det("2025-01-01"),                         # counted
        det("2025-01-02", conf="l"),               # low confidence
        det("2025-01-03", typ=2.0),                # static industrial source
        det("2025-01-04", typ=np.nan),             # NRT, unclassified: counted
        det("2025-01-05", lat=-10.0),              # no weather for this cell
    ])
    keys = fire_cell_days(f, land={(-20.0, 130.0)})
    assert keys == {(-20.0, 130.0, D("2025-01-01")), (-20.0, 130.0, D("2025-01-04"))}


def test_observed_days_need_a_daytime_detection():
    f = pd.DataFrame([
        det("2025-01-01", "SNPP"), det("2025-01-01", "N20"),
        det("2025-01-02", "SNPP", dn="N"), det("2025-01-02", "N20"),   # SNPP night-only
        det("2025-01-03", "SNPP", dn="N"),                              # no daytime at all
        det("2025-01-04", "N20"),
    ])
    obs = observed_days(f)
    assert obs.to_dict() == {D("2025-01-01"): 2, D("2025-01-02"): 1,
                             D("2025-01-03"): 0, D("2025-01-04"): 1}


def test_label_all_drops_unobserved_days():
    dates = pd.date_range("2025-01-01", "2025-01-06")
    daily = pd.DataFrame({"lat_round": -20.0, "lon_round": 130.0, "acq_date": dates})
    obs = pd.Series([2, 2, 0, 1, 2], index=pd.date_range("2025-01-01", "2025-01-05"))
    fires = {(-20.0, 130.0, D("2025-01-02"))}
    out = label_all(daily, fires, obs)
    assert list(out["acq_date"].dt.day) == [1, 2, 4, 5]        # 03 unobserved, 06 outside window
    assert list(out["label"]) == [0, 1, 0, 0]
    assert list(out["days_to_nearest_fire"]) == [1, 0, 2, 3]
    assert list(out["satellites_observed"]) == [2, 2, 1, 2]
