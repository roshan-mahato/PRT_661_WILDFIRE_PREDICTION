"""
Town search row under the top bar.

The town list follows the region selector: all of Australia lists every
town, a state lists only its own towns. Choosing a town switches the page to
"town focus" (see core.focus); a radius control and a Clear button appear
next to the search box while a town is selected.
"""

from typing import Optional, Tuple

import streamlit as st
from config import DEFAULT_RADIUS_KM, RADIUS_OPTIONS_KM
from core.geography import is_national
from core.places import towns_for_region

SEARCH_KEY = "town_search"
RADIUS_KEY = "focus_radius"


def _clear() -> None:
    st.session_state.pop(SEARCH_KEY, None)


def render_town_search(region: str) -> Optional[Tuple[str, int]]:
    """
    Renders the search row and returns (town, radius_km), or None when no
    town is selected.
    """
    towns = towns_for_region(region)
    # A town chosen under another region may not be in this region's list
    # (Kingaroy chosen, then the region switched to NSW): drop it, so the page
    # goes back to the region view instead of showing a town that's not listed.
    if st.session_state.get(SEARCH_KEY) not in (None, *towns):
        _clear()

    place = "Australia" if is_national(region) else region
    with st.container(key="townsearch"):
        search_col, radius_col, clear_col = st.columns([5.6, 2.6, 1.1], vertical_alignment="bottom")
        with search_col:
            town = st.selectbox(
                "Find a town",
                options=towns,
                index=None,
                key=SEARCH_KEY,
                placeholder=f"Search {len(towns)} towns in {place}…",
                help="Centre the page on a town: its 7-day forecast and the area around it.",
            )
        if town is None:
            return None
        with radius_col:
            radius = st.segmented_control(
                "Area around it",
                options=list(RADIUS_OPTIONS_KM),
                default=DEFAULT_RADIUS_KM,
                format_func=lambda km: f"{km} km",
                required=True,
                key=RADIUS_KEY,
                width="stretch",
            )
        with clear_col:
            st.button("Clear", icon=":material/close:", on_click=_clear, width="stretch")
    return town, int(radius)
