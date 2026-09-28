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

import pandas as pd
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

    def stored(self, latest_only=True, risk_level=None):
        self.calls.append("stored")
        return self.predictions, self.error

    def live(self, hours_back=168):
        self.calls.append("live")
        return self.predictions, self.error

    def refresh(self, hours_back=168):
        self.calls.append("refresh")
        return self.predictions, None

    def importance(self):
        self.calls.append("importance")
        if self.error:
            return pd.DataFrame(columns=["feature", "importance"]), self.error
        return pd.DataFrame([{"feature": "kbdi", "importance": 100.0}]), None

    def health(self):
        return (True, "API healthy") if self.healthy else (False, "API down")


def _install(monkeypatch, fake: FakeApi):
    monkeypatch.setattr(api_client, "fetch_stored_predictions", fake.stored)
    monkeypatch.setattr(api_client, "fetch_live_predictions", fake.live)
    monkeypatch.setattr(api_client, "refresh_predictions", fake.refresh)
    monkeypatch.setattr(api_client, "fetch_feature_importance", fake.importance)
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


def _captions(at: AppTest) -> str:
    return "\n".join(c.value for c in at.caption)


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


# ----------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------
def test_page_renders_without_errors(api):
    at = _run()

    assert at.title[0].value == "Wildfire Prediction Platform"
    assert any(m.value == "## Insights" for m in at.markdown)
    assert not at.error
    assert not at.warning


def test_defaults_to_saved_predictions_for_whole_country(api):
    at = _run()

    assert "stored" in api.calls
    assert "live" not in api.calls
    assert at.selectbox[0].value == "Australia (National)"
    # Offshore cell included -- national view is unfiltered.
    assert "Showing 6 grid cell(s) for Australia (National)" in _captions(at)
    assert "from the last saved prediction run" in _captions(at)


def test_first_forecast_day_is_selected_by_default(api):
    at = _run()
    assert f"forecast day {TODAY}" in _captions(at)
    assert _button(at, "Today").proto.type == "primary"


def test_feature_importance_is_requested(api):
    _run()
    assert "importance" in api.calls


# ----------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------
def test_region_selector_filters_cells(api):
    at = _run()
    at.selectbox[0].select("Victoria").run()

    assert not at.exception
    assert "Showing 2 grid cell(s) for Victoria" in _captions(at)


def test_region_without_cells_shows_empty_state(api):
    at = _run()
    at.selectbox[0].select("Tasmania").run()

    assert not at.exception
    assert "Showing" not in _captions(at)
    assert any("No prediction data" in m.value for m in at.markdown)


def test_clicking_a_day_changes_the_forecast_day(api):
    at = _run()
    _button(at, "Tomorrow").click().run()

    assert not at.exception
    assert f"forecast day {TOMORROW}" in _captions(at)


def test_live_source_uses_live_endpoint(api):
    at = _run()
    api.calls.clear()
    at.radio[0].set_value("Live").run()

    assert "live" in api.calls
    assert "stored" not in api.calls
    assert "computed live from current weather" in _captions(at)


def test_refresh_button_triggers_refresh(api):
    at = _run()
    _button(at, "Refresh").click().run()

    assert "refresh" in api.calls
    assert any("Predictions refreshed" in s.value for s in at.success)


# ----------------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------------
def test_api_down_shows_messages_not_a_crash(monkeypatch, empty_predictions):
    message = "Cannot reach the prediction API at http://127.0.0.1:8000."
    _install(monkeypatch, FakeApi(empty_predictions, error=message, healthy=False))

    at = _run()

    assert any(message in e.value for e in at.error)
    assert any("Prediction API unavailable" in w.value for w in at.warning)
    assert "Showing" not in _captions(at)


def test_no_predictions_yet_explains_next_step(monkeypatch, empty_predictions):
    _install(monkeypatch, FakeApi(empty_predictions))

    at = _run()

    assert any("No predictions available yet" in i.value for i in at.info)
    assert not at.error
