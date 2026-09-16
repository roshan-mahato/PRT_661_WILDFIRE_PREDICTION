"""
Forecast day selector -- one tile per day across the forecast window, in the
style of a weather app.

The buttons are rendered in their own row of columns, above the tiles, because
a click is only reported on the rerun it triggers: handling every button first
means the tiles below are drawn from the selection the user just made rather
than the previous one.
"""

from datetime import date

import streamlit as st

from utils.theme import COLORS, SEVERITY_STYLE

SESSION_KEY = "selected_forecast_day"


def _day_label(day: date) -> str:
    """Short button label: Today / Tomorrow / weekday abbreviation."""
    delta = (day - date.today()).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Tomorrow"
    return day.strftime("%a")


def _tile_html(option: dict, selected: bool) -> str:
    colour, bg = SEVERITY_STYLE.get(option["peak_level"], (COLORS["border"], COLORS["bg"]))
    border = colour if selected else COLORS["border"]
    background = bg if selected else COLORS["surface"]
    weight = "2px" if selected else "1px"

    return f"""
    <div style="
        border: {weight} solid {border};
        border-radius: 10px;
        background: {background};
        padding: 10px 8px;
        text-align: center;
        margin-bottom: 10px;
    ">
        <div style="font-size: 0.72rem; color: {COLORS['text_dim']};">
            {option['date'].strftime('%d %b')}
        </div>
        <div style="font-size: 1.35rem; font-weight: 600; color: {colour}; margin: 2px 0;">
            {option['peak_probability']:.0%}
        </div>
        <div style="font-size: 0.78rem; font-weight: 600; color: {colour};">
            {option['peak_level']}
        </div>
        <div style="font-size: 0.7rem; color: {COLORS['text_muted']}; margin-top: 4px;">
            {option['at_risk_count']} of {option['cell_count']} cells
        </div>
        <div style="font-size: 0.7rem; color: {COLORS['text_muted']};">
            FFDI {option['max_ffdi']:.1f}
        </div>
    </div>
    """


def render_day_strip(options: list):
    """
    Renders the forecast day strip and returns the selected `date`.

    Args:
        options: output of utils.regions.day_options(), oldest day first.

    Returns:
        The selected date, or None when there are no forecast days.
    """
    if not options:
        return None

    dates = [option["date"] for option in options]

    # The available days change with the region, the data source and every
    # refresh, so a previously selected day may no longer exist.
    if st.session_state.get(SESSION_KEY) not in dates:
        st.session_state[SESSION_KEY] = dates[0]

    st.markdown(
        f"<div style='font-size:0.8rem; color:{COLORS['text_muted']}; "
        f"margin-bottom:6px;'>Forecast day</div>",
        unsafe_allow_html=True,
    )

    button_cols = st.columns(len(options))
    for index, col in enumerate(button_cols):
        with col:
            is_selected = dates[index] == st.session_state[SESSION_KEY]
            if st.button(
                _day_label(dates[index]),
                key=f"forecast_day_{dates[index]}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state[SESSION_KEY] = dates[index]

    selected = st.session_state[SESSION_KEY]

    tile_cols = st.columns(len(options))
    for index, col in enumerate(tile_cols):
        with col:
            st.markdown(
                _tile_html(options[index], dates[index] == selected),
                unsafe_allow_html=True,
            )

    return selected
