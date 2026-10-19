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
from components.hero import render_hero, render_town_hero
from components.insights import render_insights_row
from components.map_section import render_map_with_insights
from components.town_search import render_town_search
from config import NEARBY_TOWNS_MIN_KM
from core.focus import cells_within, make_focus, nearby_towns, town_day_options, town_rows
from utils.api_client import (
    check_api_health,
    fetch_live_predictions,
    fetch_stored_predictions,
    refresh_predictions,
)
from core.geography import filter_by_region
from core.summaries import day_options, summarise_region
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
search = render_town_search(region)   # (town, radius_km) or None

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
        "No predictions available yet. Fetch the forecast "
        "(`uv run python Scripts/fetch_weather_live.py`), which also predicts."
    )

# ---------------------------------------------------------------------------
# Scope: a searched town and the area around it, or else the selected region
# ---------------------------------------------------------------------------
focus = make_focus(predictions, *search) if search else None
if focus is not None and focus.cell is None and not predictions.empty:
    st.warning(f"No forecast area near {focus.town} in the current data; showing {region} instead.")
if focus is not None and focus.cell is None:
    focus = None


def _for_day(frame: pd.DataFrame, day) -> pd.DataFrame:
    if day is None:
        return frame
    return frame[pd.to_datetime(frame["acq_date"]).dt.date == day].reset_index(drop=True)


if focus is None:
    regional = filter_by_region(predictions, region)

    # The strip selects one day; the banner, cards and map describe that day,
    # while the charts keep the whole window so the shape of the forecast
    # stays visible. The selection is resolved before anything is drawn
    # because the banner sits above the strip.
    options = day_options(regional)
    selected_day = resolve_selected_day(options)
    day_regional = _for_day(regional, selected_day)
    summary = summarise_region(day_regional, region)

    render_hero(summary, region, source_label)
    render_day_strip(options)
    render_insights_row(summary)
    render_map_with_insights(day_regional, region, summary)
    render_insights_section(regional, day_regional)
else:
    # Town focus works on the national predictions by distance, ignoring the
    # state boxes, so border towns find their forecast and neighbours on both
    # sides of the border.
    area = cells_within(predictions, focus)
    options = town_day_options(area, focus)
    selected_day = resolve_selected_day(options)
    day_area = _for_day(area, selected_day)
    summary = summarise_region(day_area, focus.town)
    town_today = town_rows(day_area, focus)
    town_row = town_today.iloc[0] if not town_today.empty else None
    nearby_km = max(focus.radius_km, NEARBY_TOWNS_MIN_KM)
    nearby = nearby_towns(_for_day(predictions, selected_day), focus, nearby_km)

    render_town_hero(focus, town_row, summary, source_label)
    render_day_strip(options)
    render_insights_row(summary)
    # The map draws every area (heat map, like the region view), centred on
    # the town; the panel beside it summarises only the area around the town.
    render_map_with_insights(day_area, region, summary, focus=focus, focus_row=town_row,
                             map_predictions=_for_day(predictions, selected_day))
    render_insights_section(area, day_area, focus=focus, nearby=nearby, nearby_km=nearby_km)
