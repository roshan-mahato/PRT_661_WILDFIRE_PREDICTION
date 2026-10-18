"""
The Insights section: four charts, each answering a different question so
nothing on the page is said twice.

  1. fire_weather.render_fire_weather              WHY does danger change?
  2. risk_outlook.render_risk_outlook              HOW WIDESPREAD is it each day?
  3. places_affected.render_places_affected        WHO is affected today?
  4. hotspots.render_persistent_hotspots           WHERE and WHEN does it last?

They speak in everyday units -- named danger bands, % of the region, towns,
days -- rather than the model's grid cells and probabilities. Their data is
prepared in core.chart_data; this package only draws it.
"""

import pandas as pd
import streamlit as st
from utils.styles import section_header

from components.charts.fire_weather import render_fire_weather
from components.charts.hotspots import render_persistent_hotspots
from components.charts.places_affected import render_places_affected
from components.charts.risk_outlook import render_risk_outlook

__all__ = [
    "render_insights_section",
    "render_fire_weather",
    "render_risk_outlook",
    "render_places_affected",
    "render_persistent_hotspots",
]


def render_insights_section(predictions: pd.DataFrame, day_predictions: pd.DataFrame) -> None:
    """
    Renders the full 'Insights' section: a 2x2 grid of charts.

    Args:
        predictions: the region's prediction rows across every forecast day.
        day_predictions: the same rows narrowed to the day selected in the
                         day strip (chart 3 describes that day).
    """
    section_header(
        "Insights",
        "How the forecast develops",
        "The weather behind the risk, how widespread it is, and which places are affected.",
    )

    row1_col1, row1_col2 = st.columns(2, gap="medium")
    with row1_col1:
        with st.container(key="card-weather"):
            render_fire_weather(predictions)
    with row1_col2:
        with st.container(key="card-outlook"):
            render_risk_outlook(predictions)

    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        with st.container(key="card-places"):
            render_places_affected(day_predictions)
    with row2_col2:
        with st.container(key="card-hotspots"):
            render_persistent_hotspots(predictions)
