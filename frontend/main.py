"""
Wildfire Prediction Platform — Streamlit dashboard.

Reads predictions from the FastAPI backend (backend/app.py). Start the API
first, then run this:

    uv run uvicorn backend.app:app --reload
    uv run streamlit run frontend/main.py
"""

import pandas as pd
import streamlit as st

from components.charts import render_insights_section
from components.day_strip import render_day_strip
from components.headers import render_header
from components.insights import render_insights_row
from components.map_section import render_map_with_insights
from utils.api_client import (
    check_api_health,
    fetch_live_predictions,
    fetch_stored_predictions,
    refresh_predictions,
)
from utils.regions import (
    build_daily_trend,
    day_options,
    filter_by_region,
    summarise_region,
)

st.set_page_config(
    page_title="Wildfire Prediction Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

controls = render_header()
region = controls["region"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
if controls["refresh"]:
    with st.spinner("Recomputing predictions from the latest weather data..."):
        _, refresh_error = refresh_predictions()
    if refresh_error:
        st.error(refresh_error)
    else:
        st.success("Predictions refreshed and saved.")

healthy, health_message = check_api_health()
if not healthy:
    st.warning(f"Prediction API unavailable — {health_message}")

if controls["use_live"]:
    predictions, load_error = fetch_live_predictions()
    source_label = "computed live from current weather"
else:
    predictions, load_error = fetch_stored_predictions(latest_only=True)
    source_label = "from the last saved prediction run"

if load_error:
    st.error(load_error)

if predictions.empty and not load_error:
    st.info(
        "No predictions available yet. Fetch live weather "
        "(`uv run python Scripts/fetch_openmeteo_live.py`), then press Refresh."
    )

# ---------------------------------------------------------------------------
# Region slice + derived figures
# ---------------------------------------------------------------------------
regional = filter_by_region(predictions, region)

# The strip selects one day; the cards and map describe that day, while the
# trend chart keeps the whole window so the shape of the forecast stays visible.
selected_day = render_day_strip(day_options(regional))
if selected_day is None:
    day_regional = regional
else:
    day_regional = regional[
        pd.to_datetime(regional["acq_date"]).dt.date == selected_day
    ].reset_index(drop=True)

summary = summarise_region(day_regional, region)
trend = build_daily_trend(regional)

if summary["has_data"]:
    st.caption(
        f"Showing {summary['cell_count']} grid cell(s) for {region} — "
        f"forecast day {summary['forecast_date']}, {source_label}."
    )

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
render_insights_row(summary)
render_map_with_insights(day_regional, region, summary)
render_insights_section(trend, summary)
