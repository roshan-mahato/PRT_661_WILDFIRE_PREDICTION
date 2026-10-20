"""
End-to-end tests of the dashboard page (frontend/main.py) using Streamlit's
headless AppTest runner.

The backend is never contacted: the utils.api_client functions are swapped
for fakes before each run. main.py re-imports them on every rerun, so the
fakes are what the page actually calls. This checks the page wiring -- the
region selector, data-source toggle, refresh button and day strip -- and
that a dead API produces messages rather than a crash.
"""

from pathlib import Path

import pytest
from conftest import TODAY, TOMORROW
from streamlit.testing.v1 import AppTest
from utils import api_client

MAIN = Path(__file__).parents[2] / "frontend" / "main.py"
TIMEOUT = 30


class FakeApi:
    """Stands in for utils.api_client and records which endpoints were used."""

    def __init__(self, predictions, error=None, healthy=True):
        self.predictions = predictions
        self.error = error
        self.healthy = healthy
        self.calls = []
        self.bboxes = []
        self.live_error = None

    def stored(self, latest_only=True, risk_level=None):
        self.calls.append("stored")
        return self.predictions, self.error

    def live(self, hours_back=168):
        self.calls.append("live")
        return self.predictions, self.error

    def refresh(self, hours_back=168):
        self.calls.append("refresh")
        return self.predictions, None

    def live_fetch(self, bbox=None):
        self.calls.append("live_fetch")
        self.bboxes.append(bbox)
        if self.live_error:
            return None, self.live_error
        return {"run_id": 9, "status": "ok", "cells_requested": 4, "cells_saved": 4,
                "cells_failed": 0, "weather_rows": 28, "feature_rows": 28,
                "predictions_written": 28, "seconds": 3.0}, None

    def health(self):
        return (True, "API healthy") if self.healthy else (False, "API down")


def _install(monkeypatch, fake: FakeApi):
    monkeypatch.setattr(api_client, "fetch_stored_predictions", fake.stored)
    monkeypatch.setattr(api_client, "fetch_live_predictions", fake.live)
    monkeypatch.setattr(api_client, "refresh_predictions", fake.refresh)
    monkeypatch.setattr(api_client, "fetch_live_weather", fake.live_fetch)
    monkeypatch.setattr(api_client, "check_api_health", fake.health)


@pytest.fixture
def api(monkeypatch, predictions):
    fake = FakeApi(predictions)
    _install(monkeypatch, fake)
    return fake


def _run() -> AppTest:
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def _page_text(at: AppTest) -> str:
    """All custom HTML on the page (banner, tiles, panels) as one string."""
    return "\n".join(m.value for m in at.markdown)


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


# ----------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------
def test_page_renders_without_errors(api):
    at = _run()

    assert "Wildfire Prediction Platform" in _page_text(at)
    assert "How the forecast develops" in _page_text(at)
    assert not at.error
    assert not at.warning


def test_defaults_to_saved_predictions_for_whole_country(api):
    at = _run()

    assert "stored" in api.calls
    assert "live_fetch" not in api.calls
    assert at.selectbox[0].value == "Australia (National)"
    # Offshore cell included -- national view is unfiltered.
    assert "Showing 6 grid cell(s) for Australia (National)" in _page_text(at)
    assert "from the last saved prediction run" in _page_text(at)


def test_first_forecast_day_is_selected_by_default(api):
    at = _run()
    assert f"forecast day {TODAY}" in _page_text(at)
    assert _button(at, "Today").proto.type == "primary"


def test_page_does_not_call_the_model_endpoint(api):
    """Chart 4 is built from the predictions; the page must not depend on
    /model/feature-importance (which once broke it with a 501)."""
    at = _run()
    assert set(api.calls) <= {"stored", "live", "refresh", "live_fetch"}
    assert "Where the Danger Lasts" in "\n".join(m.value for m in at.markdown)


# ----------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------
def test_region_selector_filters_cells(api):
    at = _run()
    at.selectbox[0].select("Victoria").run()

    assert not at.exception
    assert "Showing 2 grid cell(s) for Victoria" in _page_text(at)


def test_region_without_cells_shows_empty_state(api):
    at = _run()
    at.selectbox[0].select("Tasmania").run()

    assert not at.exception
    assert "Showing" not in _page_text(at)
    assert "No forecast for this region yet" in _page_text(at)


def test_clicking_a_day_changes_the_forecast_day(api):
    at = _run()
    _button(at, "Tomorrow").click().run()

    assert not at.exception
    assert f"forecast day {TOMORROW}" in _page_text(at)


def test_live_fetches_the_latest_weather_and_predicts(api):
    at = _run()
    api.calls.clear()
    at.segmented_control[0].set_value("Live").run()

    assert not at.exception
    assert api.calls.count("live_fetch") == 1
    assert api.bboxes == [None]  # whole country
    assert "stored" in api.calls  # then shows what the fetch saved
    assert any("Fetched the latest forecast for 4 grid cells" in s.value for s in at.success)
    assert "fetched live at" in _page_text(at)


def test_live_fetches_only_the_selected_state(api):
    at = _run()
    at.selectbox[0].select("Victoria").run()
    at.segmented_control[0].set_value("Live").run()

    from core.geography import STATE_BOUNDS
    assert api.bboxes == [STATE_BOUNDS["Victoria"]]


def test_live_does_not_refetch_on_every_click(api):
    at = _run()
    at.segmented_control[0].set_value("Live").run()
    _button(at, "Tomorrow").click().run()

    assert api.calls.count("live_fetch") == 1
    assert f"forecast day {TOMORROW}" in _page_text(at)


def test_refresh_in_live_mode_fetches_again(api):
    at = _run()
    at.segmented_control[0].set_value("Live").run()
    _button(at, "Refresh").click().run()

    assert api.calls.count("live_fetch") == 2
    assert "refresh" not in api.calls


def test_failed_live_fetch_shows_error_and_is_not_retried_every_click(api):
    api.live_error = "Live fetch failed: Cannot reach the self-hosted Open-Meteo server"
    at = _run()
    at.segmented_control[0].set_value("Live").run()

    assert any("Open-Meteo" in e.value for e in at.error)
    assert "from the last saved prediction run" in _page_text(at)
    _button(at, "Tomorrow").click().run()
    assert api.calls.count("live_fetch") == 1


def test_refresh_in_saved_mode_repredicts_stored_forecast(api):
    at = _run()
    _button(at, "Refresh").click().run()

    assert "refresh" in api.calls
    assert "live_fetch" not in api.calls
    assert any("Predictions recomputed" in s.value for s in at.success)


# ----------------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------------
def test_api_down_shows_messages_not_a_crash(monkeypatch, empty_predictions):
    message = "Cannot reach the prediction API at http://127.0.0.1:8000."
    _install(monkeypatch, FakeApi(empty_predictions, error=message, healthy=False))

    at = _run()

    assert any(message in e.value for e in at.error)
    assert any("Prediction API unavailable" in w.value for w in at.warning)
    assert "Showing" not in _page_text(at)


def test_no_predictions_yet_explains_next_step(monkeypatch, empty_predictions):
    _install(monkeypatch, FakeApi(empty_predictions))

    at = _run()

    assert any("No predictions available yet" in i.value for i in at.info)
    assert not at.error
