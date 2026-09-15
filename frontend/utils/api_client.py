"""
Client for the FastAPI backend.

Every network call is wrapped so the dashboard degrades gracefully: if the
backend is down, functions return an empty DataFrame plus an error message
rather than raising, and the UI shows a warning instead of a stack trace.
This matters for a live demo -- a dead API should not take the whole page
down with it.

Responses are cached so Streamlit's rerun-on-every-interaction model doesn't
hammer the backend each time a dropdown changes.
"""

import os
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("WILDFIRE_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30  # seconds; a full refresh can take a while

PREDICTION_COLS = [
    "lat_round", "lon_round", "acq_date", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "ffdi", "kbdi",
    "drought_factor", "kbdi_spinup_flag", "fire_probability",
    "fire_predicted", "risk_level",
]


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLS)


def _get(path: str, params: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    """Performs a GET and returns (payload, error_message). Never raises."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{path}", params=params or {}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the prediction API at {API_BASE_URL}. "
            "Start it with: uv run uvicorn backend.app:app --reload"
        )
    except requests.exceptions.Timeout:
        return None, f"The prediction API did not respond within {REQUEST_TIMEOUT}s."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:  # noqa: BLE001 - error body may not be JSON
            detail = exc.response.text[:200]
        return None, f"API error {exc.response.status_code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - the UI must never crash on a fetch
        return None, f"Unexpected error calling the API: {exc}"


def _payload_to_frame(payload: Optional[dict]) -> pd.DataFrame:
    if not payload or not payload.get("predictions"):
        return _empty_predictions()
    df = pd.DataFrame(payload["predictions"])
    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stored_predictions(
    latest_only: bool = True, risk_level: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Reads saved predictions from the backend. This is the default read path --
    it does no model inference, so it stays fast and still works when
    Open-Meteo is unavailable.

    Cached for 5 minutes, because Streamlit reruns the whole script on every
    widget interaction and predictions only change when a refresh runs.

    Returns:
        (DataFrame, error_message). The DataFrame is empty if the call failed.
    """
    params = {"latest_only": str(latest_only).lower()}
    if risk_level:
        params["risk_level"] = risk_level
    payload, error = _get("/predictions/stored", params)
    return _payload_to_frame(payload), error


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_predictions(hours_back: int = 168) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Computes predictions fresh from current weather data, without saving them.
    Slower than fetch_stored_predictions(); use when the user explicitly wants
    up-to-the-minute numbers.
    """
    payload, error = _get("/predictions", {"hours_back": hours_back})
    return _payload_to_frame(payload), error


def refresh_predictions(hours_back: int = 168) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Triggers a full recompute on the backend AND saves the results.

    Deliberately NOT cached -- this has a side effect (it writes to the
    database), and a cached write would silently do nothing on the second
    click, which looks like a broken button.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/predictions/refresh",
            params={"hours_back": hours_back},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.ConnectionError:
        return _empty_predictions(), f"Cannot reach the prediction API at {API_BASE_URL}."
    except requests.exceptions.Timeout:
        return _empty_predictions(), (
            f"Refresh timed out after {REQUEST_TIMEOUT}s. A full-grid refresh can be slow; "
            "it may still be running on the server."
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_predictions(), f"Refresh failed: {exc}"

    # The stored-prediction cache is now stale -- clear it so the next read
    # picks up what this refresh just wrote.
    fetch_stored_predictions.clear()
    return _payload_to_frame(payload), None


@st.cache_data(ttl=60, show_spinner=False)
def check_api_health() -> Tuple[bool, str]:
    """
    Returns (is_healthy, message). Used to show a banner when the model or
    database is unavailable, so the user knows why numbers are missing.
    """
    payload, error = _get("/health")
    if error:
        return False, error
    if payload.get("status") == "ok":
        return True, "API healthy"
    return False, payload.get("detail") or "API reports degraded status"
