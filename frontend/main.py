"""
Wildfire Prediction Platform — Streamlit dashboard.

Reads predictions from the FastAPI backend (backend/app.py). Start the API
first, then run this:

    uv run uvicorn backend.app:app --reload
    uv run streamlit run frontend/main.py
"""

import time
from datetime import datetime

import pandas as pd
import streamlit as st
from components.charts import render_insights_section
from components.day_strip import render_day_strip, resolve_selected_day
from components.headers import render_header
from components.hero import render_hero, render_town_hero
from components.insights import render_insights_row
from components.map_section import render_map_with_insights
from components.town_search import render_town_search
from config import LIVE_MAX_AGE, NEARBY_TOWNS_MIN_KM
from core.focus import cells_within, make_focus, nearby_towns, town_day_options, town_rows
from utils.api_client import (
    check_api_health,
    fetch_live_weather,
    fetch_stored_predictions,
    refresh_predictions,
)
from core.geography import filter_by_region, region_bbox
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
def run_live_fetch(region: str) -> None:
    """Downloads the latest forecast for the region and predicts it."""
    # Remember the attempt even if it fails, so a server that is down is not
    # asked again on every click -- Refresh retries straight away.
    st.session_state.setdefault("live_tried", {})[region] = time.time()
    with st.spinner(f"Fetching the latest weather for {region} and predicting..."):
        summary, error = fetch_live_weather(region_bbox(region))
    if error:
        st.error(error)
        return
    st.session_state.setdefault("live_fetched", {})[region] = time.time()
    message = (
        f"Fetched the latest forecast for {summary['cells_saved']:,} grid cells and "
        f"saved {summary['predictions_written']:,} predictions in {summary['seconds']:.0f}s."
    )
    if summary["cells_failed"]:
        st.warning(f"{message} {summary['cells_failed']:,} cells could not be fetched "
                   "and keep their previous predictions.")
    else:
        st.success(message)


if controls["use_live"]:
    # Fetch when asked to, or when this region has no recent live fetch.
    # Streamlit reruns the script on every click, so without the age check
    # each click on the day strip would download the weather again.
    last = st.session_state.get("live_tried", {}).get(region)
    if controls["refresh"] or last is None or time.time() - last > LIVE_MAX_AGE:
        run_live_fetch(region)
elif controls["refresh"]:
    with st.spinner("Re-running the model on the stored forecast..."):
        _, refresh_error = refresh_predictions()
    if refresh_error:
        st.error(refresh_error)
    else:
        st.success("Predictions recomputed and saved.")

healthy, health_message = check_api_health()
if not healthy:
    st.warning(f"Prediction API unavailable — {health_message}")

# Both modes read the stored predictions; Live has just written new ones.
predictions, load_error = fetch_stored_predictions(latest_only=True)
fetched_at = st.session_state.get("live_fetched", {}).get(region)
if controls["use_live"] and fetched_at:
    source_label = f"fetched live at {datetime.fromtimestamp(fetched_at):%H:%M}"
else:
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
