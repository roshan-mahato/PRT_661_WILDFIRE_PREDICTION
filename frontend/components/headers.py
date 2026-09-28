import streamlit as st

from utils.regions import NATIONAL
from utils.styles import html, icon

REGIONS = [
    NATIONAL,
    "New South Wales",
    "Victoria",
    "Queensland",
    "Western Australia",
    "South Australia",
    "Tasmania",
    "Northern Territory",
    "Australian Capital Territory",
]

SOURCES = ["Saved", "Live"]


def render_header():
    """
    Renders the top bar: brand on the left, controls on the right.

    Returns:
        dict with keys `region` (str), `use_live` (bool) and `refresh`
        (bool, True on the run where the refresh button was clicked).
    """
    with st.container(key="topbar"):
        brand_col, region_col, source_col, refresh_col = st.columns(
            [5, 2.4, 1.7, 1.1], vertical_alignment="bottom"
        )

        with brand_col:
            html(
                f"""
                <div class="brand">
                  <div class="brand-mark">{icon("flame", 24)}</div>
                  <div>
                    <h1 class="brand-title" style="margin:0;padding:0;">Wildfire Prediction Platform</h1>
                    <div class="brand-sub">Next-day fire risk for Australia, forecast on a 0.5&deg; grid</div>
                  </div>
                </div>
                """
            )

        with region_col:
            region = st.selectbox("Region", options=REGIONS, index=0)

        with source_col:
            source = st.segmented_control(
                "Data source",
                options=SOURCES,
                default="Saved",
                required=True,
                width="stretch",
                help=(
                    "Saved reads stored predictions — fast, and works if the weather "
                    "API is down. Live recomputes from current weather data."
                ),
            )

        with refresh_col:
            refresh = st.button(
                "Refresh",
                icon=":material/refresh:",
                width="stretch",
                help="Recompute predictions from the latest weather and save them.",
            )

    return {"region": region, "use_live": source == "Live", "refresh": refresh}
