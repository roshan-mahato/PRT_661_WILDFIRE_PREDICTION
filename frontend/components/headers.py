import streamlit as st

from utils.regions import NATIONAL

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


def render_header():
    """
    Renders the top header, region selector, and data-source controls.

    Returns:
        dict with keys `region` (str), `use_live` (bool) and `refresh`
        (bool, True on the run where the refresh button was clicked).
    """
    title_col, source_col, refresh_col, region_col = st.columns([6, 2, 1.4, 2.4])

    with title_col:
        st.title("Wildfire Prediction Platform")

    with source_col:
        source = st.radio(
            "Data source",
            options=["Saved", "Live"],
            index=0,
            horizontal=True,
            help=(
                "Saved reads stored predictions — fast, and works if the weather "
                "API is down. Live recomputes from current weather data."
            ),
        )

    with refresh_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        refresh = st.button(
            "Refresh",
            use_container_width=True,
            help="Recompute predictions from the latest weather and save them.",
        )

    with region_col:
        region = st.selectbox("Region", options=REGIONS, index=0)

    return {"region": region, "use_live": source == "Live", "refresh": refresh}
