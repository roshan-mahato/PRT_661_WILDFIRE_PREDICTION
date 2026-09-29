"""
Tests for utils.api_client -- the dashboard's only link to the backend.

No real server is needed: requests.get / requests.post are replaced with fakes,
so these tests check that every failure turns into an error message (never an
exception) and that successful payloads are parsed into the expected frames.
"""

import pandas as pd
import pytest
import requests
from utils import api_client


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


class FakeRequests:
    """Records every call and answers with a fixed response or exception."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def clear_caches():
    """st.cache_data would otherwise leak results between tests."""
    for fn in (
        api_client.fetch_stored_predictions,
        api_client.fetch_live_predictions,
        api_client.fetch_feature_importance,
        api_client.check_api_health,
    ):
        fn.clear()
    yield


def _payload(predictions: pd.DataFrame) -> dict:
    """Serialises the fixture frame the way the backend's JSON would look."""
    records = predictions.assign(
        acq_date=predictions["acq_date"].dt.strftime("%Y-%m-%d")
    ).to_dict(orient="records")
    return {"count": len(records), "hours_back": 0, "spinup_count": 0, "predictions": records}


def _mock_get(monkeypatch, result) -> FakeRequests:
    fake = FakeRequests(result)
    monkeypatch.setattr(api_client.requests, "get", fake)
    return fake


# ----------------------------------------------------------------------------
# _get error handling
# ----------------------------------------------------------------------------
def test_get_connection_error_suggests_starting_api(monkeypatch):
    _mock_get(monkeypatch, requests.exceptions.ConnectionError())
    payload, error = api_client._get("/health")
    assert payload is None
    assert "Cannot reach the prediction API" in error
    assert "uvicorn" in error


def test_get_timeout(monkeypatch):
    _mock_get(monkeypatch, requests.exceptions.Timeout())
    payload, error = api_client._get("/health")
    assert payload is None
    assert "did not respond" in error


def test_get_http_error_includes_json_detail(monkeypatch):
    _mock_get(monkeypatch, FakeResponse({"detail": "Model file not available"}, 503))
    payload, error = api_client._get("/predictions")
    assert payload is None
    assert error == "API error 503: Model file not available"


def test_get_http_error_with_non_json_body(monkeypatch):
    _mock_get(monkeypatch, FakeResponse(None, 500, text="Internal Server Error"))
    _, error = api_client._get("/predictions")
    assert error == "API error 500: Internal Server Error"


def test_get_unexpected_error_does_not_raise(monkeypatch):
    _mock_get(monkeypatch, RuntimeError("boom"))
    payload, error = api_client._get("/health")
    assert payload is None
    assert "boom" in error


# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------
def test_fetch_stored_predictions_parses_payload(monkeypatch, predictions):
    fake = _mock_get(monkeypatch, FakeResponse(_payload(predictions)))

    df, error = api_client.fetch_stored_predictions(latest_only=True)

    assert error is None
    assert len(df) == len(predictions)
    assert pd.api.types.is_datetime64_any_dtype(df["acq_date"])
    assert fake.calls[0]["url"].endswith("/predictions/stored")
    assert fake.calls[0]["params"] == {"latest_only": "true"}


def test_fetch_stored_predictions_passes_risk_level(monkeypatch, predictions):
    fake = _mock_get(monkeypatch, FakeResponse(_payload(predictions)))
    api_client.fetch_stored_predictions(latest_only=False, risk_level="High")
    assert fake.calls[0]["params"] == {"latest_only": "false", "risk_level": "High"}


def test_fetch_stored_predictions_empty_payload_keeps_columns(monkeypatch):
    _mock_get(monkeypatch, FakeResponse({"count": 0, "predictions": []}))
    df, error = api_client.fetch_stored_predictions()
    assert error is None
    assert df.empty
    assert list(df.columns) == api_client.PREDICTION_COLS


def test_fetch_stored_predictions_error_returns_empty_frame(monkeypatch):
    _mock_get(monkeypatch, requests.exceptions.ConnectionError())
    df, error = api_client.fetch_stored_predictions()
    assert df.empty
    assert list(df.columns) == api_client.PREDICTION_COLS
    assert error


def test_fetch_stored_predictions_is_cached(monkeypatch, predictions):
    fake = _mock_get(monkeypatch, FakeResponse(_payload(predictions)))
    api_client.fetch_stored_predictions()
    api_client.fetch_stored_predictions()
    assert len(fake.calls) == 1


def test_fetch_live_predictions_sends_hours_back(monkeypatch, predictions):
    fake = _mock_get(monkeypatch, FakeResponse(_payload(predictions)))
    df, error = api_client.fetch_live_predictions(hours_back=48)
    assert error is None
    assert len(df) == len(predictions)
    assert fake.calls[0]["url"].endswith("/predictions")
    assert fake.calls[0]["params"] == {"hours_back": 48}


def test_refresh_posts_and_clears_stored_cache(monkeypatch, predictions):
    get = _mock_get(monkeypatch, FakeResponse(_payload(predictions)))
    post = FakeRequests(FakeResponse(_payload(predictions)))
    monkeypatch.setattr(api_client.requests, "post", post)

    api_client.fetch_stored_predictions()
    df, error = api_client.refresh_predictions(hours_back=24)
    api_client.fetch_stored_predictions()

    assert error is None
    assert len(df) == len(predictions)
    assert post.calls[0]["url"].endswith("/predictions/refresh")
    assert post.calls[0]["params"] == {"hours_back": 24}
    # The second stored read must hit the API again, not the stale cache.
    assert len(get.calls) == 2


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (requests.exceptions.ConnectionError(), "Cannot reach"),
        (requests.exceptions.Timeout(), "timed out"),
        (RuntimeError("disk full"), "disk full"),
    ],
)
def test_refresh_errors_are_reported_not_raised(monkeypatch, exc, message):
    monkeypatch.setattr(api_client.requests, "post", FakeRequests(exc))
    df, error = api_client.refresh_predictions()
    assert df.empty
    assert message in error


# ----------------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------------
def test_fetch_feature_importance(monkeypatch):
    fake = _mock_get(
        monkeypatch,
        FakeResponse(
            {
                "model_name": "LightGBM",
                "importance_type": "gain",
                "trained_at": None,
                "features": [
                    {"feature": "kbdi", "importance": 60.0},
                    {"feature": "lat_round", "importance": 40.0},
                ],
            }
        ),
    )
    df, error = api_client.fetch_feature_importance()
    assert error is None
    assert list(df.columns) == ["feature", "importance"]
    assert df["importance"].sum() == pytest.approx(100.0)
    assert fake.calls[0]["url"].endswith("/model/feature-importance")


def test_fetch_feature_importance_error(monkeypatch):
    _mock_get(monkeypatch, FakeResponse({"detail": "no model"}, 503))
    df, error = api_client.fetch_feature_importance()
    assert df.empty
    assert list(df.columns) == ["feature", "importance"]
    assert "no model" in error


# ----------------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------------
def test_check_api_health_ok(monkeypatch):
    _mock_get(monkeypatch, FakeResponse({"status": "ok"}))
    assert api_client.check_api_health() == (True, "API healthy")


def test_check_api_health_degraded_reports_detail(monkeypatch):
    _mock_get(monkeypatch, FakeResponse({"status": "degraded", "detail": "model: missing"}))
    assert api_client.check_api_health() == (False, "model: missing")


def test_check_api_health_unreachable(monkeypatch):
    _mock_get(monkeypatch, requests.exceptions.ConnectionError())
    healthy, message = api_client.check_api_health()
    assert not healthy
    assert "Cannot reach" in message
