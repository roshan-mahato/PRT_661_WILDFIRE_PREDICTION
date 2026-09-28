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
from components.day_strip import render_day_strip, resolve_selected_day
from components.headers import render_header
from components.hero import render_hero
from components.insights import render_insights_row
from components.map_section import render_map_with_insights
from utils.api_client import (
    check_api_health,
    fetch_feature_importance,
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
from utils.styles import inject_global_css

st.set_page_config(
    page_title="Wildfire Prediction Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_css()

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

# The strip selects one day; the banner, cards and map describe that day,
# while the trend charts keep the whole window so the shape of the forecast
# stays visible. The selection is resolved before anything is drawn because
# the banner sits above the strip.
options = day_options(regional)
selected_day = resolve_selected_day(options)
if selected_day is None:
    day_regional = regional
else:
    day_regional = regional[
        pd.to_datetime(regional["acq_date"]).dt.date == selected_day
    ].reset_index(drop=True)

summary = summarise_region(day_regional, region)
trend = build_daily_trend(regional)

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
render_hero(summary, region, source_label)
render_day_strip(options)
render_insights_row(summary)
render_map_with_insights(day_regional, region, summary)
importance, importance_error = fetch_feature_importance()
render_insights_section(trend, summary, regional, importance, importance_error)
