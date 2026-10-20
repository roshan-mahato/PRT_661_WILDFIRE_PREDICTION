"""
POST /predictions/live: box handling, the one-fetch-at-a-time lock and error
mapping. The route function is called directly and the fetch itself is
replaced, so no weather server (and no HTTP test client) is needed.
"""

import threading

import pytest
from fastapi import HTTPException

import Scripts.fetch_weather_live as live
from backend.routers import prediction
from backend.routers.prediction import fetch_live_fire_risk

SUMMARY = {"run_id": 7, "status": "ok", "cells_requested": 9, "cells_saved": 9,
           "cells_failed": 0, "weather_rows": 63, "feature_rows": 63,
           "predictions_written": 63, "seconds": 1.0}
AUSTRALIA = dict(lat_min=-44.0, lat_max=-10.0, lon_min=112.0, lon_max=154.0)


@pytest.fixture
def calls(monkeypatch):
    """Replaces the fetch; records (bbox, keyword arguments) of every call."""
    seen = []

    def fake(bbox, **kwargs):
        seen.append((bbox, kwargs))
        return SUMMARY

    monkeypatch.setattr(live, "fetch_and_predict", fake)
    return seen


def test_box_is_widened_to_whole_cells(calls):
    out = fetch_live_fire_risk(lat_min=-35.92, lat_max=-35.12, lon_min=148.76, lon_max=149.40)

    assert out.predictions_written == 63
    assert calls[0][0] == (-36.0, -35.0, 148.5, 149.5)


def test_whole_country_box_is_unchanged(calls):
    fetch_live_fire_risk(**AUSTRALIA)
    assert calls[0][0] == (-44.0, -10.0, 112.0, 154.0)


def test_live_leaves_the_history_top_up_to_the_daily_job(calls):
    fetch_live_fire_risk(**AUSTRALIA)
    assert calls[0][1] == {"update_history": False}


def test_reversed_box_is_rejected():
    with pytest.raises(HTTPException) as exc:
        fetch_live_fire_risk(lat_min=-30.0, lat_max=-35.0, lon_min=112.0, lon_max=154.0)
    assert exc.value.status_code == 422


def test_weather_server_down_is_a_503(monkeypatch):
    def down(bbox, **kwargs):
        raise live.LiveFetchError("Cannot reach the self-hosted Open-Meteo server")
    monkeypatch.setattr(live, "fetch_and_predict", down)

    with pytest.raises(HTTPException) as exc:
        fetch_live_fire_risk(**AUSTRALIA)
    assert exc.value.status_code == 503 and "Open-Meteo" in exc.value.detail
    assert not prediction._live_fetch_lock.locked()  # released after a failure


def test_second_fetch_while_one_runs_is_refused(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow(bbox, **kwargs):
        started.set()
        release.wait(5)
        return SUMMARY

    monkeypatch.setattr(live, "fetch_and_predict", slow)
    first = {}
    t = threading.Thread(target=lambda: first.update(r=fetch_live_fire_risk(**AUSTRALIA)))
    t.start()
    started.wait(5)
    try:
        with pytest.raises(HTTPException) as exc:
            fetch_live_fire_risk(**AUSTRALIA)
    finally:
        release.set()
        t.join(5)

    assert exc.value.status_code == 409
    assert first["r"].status == "ok"
