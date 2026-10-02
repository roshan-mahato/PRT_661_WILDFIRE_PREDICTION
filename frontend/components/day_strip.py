"""
Forecast day selector -- one tile per day across the forecast window, in the
style of a weather app.

Each day is a button (the day name) sitting on top of a tile with that day's
headline figures. The buttons use an on_click callback, which Streamlit runs
*before* the rerun, so every tile is drawn from the selection the user just
made rather than the previous one.
"""

from datetime import date, timedelta

import streamlit as st

from utils.glossary import term
from utils.regions import week_highlights
from utils.styles import html, icon
from utils.theme import COLORS, SEVERITY_STYLE

SESSION_KEY = "selected_forecast_day"
SLOTS = 7  # a full forecast week; fewer days leave placeholders, not giant tiles


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
    classes = "day-tile selected" if selected else "day-tile"
    return (
        f'<div class="{classes}" style="--sev:{colour};--sev-bg:{bg};">'
        f'<div class="date">{option["date"].strftime("%d %b")}</div>'
        f'<div class="prob">{option["peak_probability"]:.0%}</div>'
        f'<div class="lvl">{option["peak_level"]}</div>'
        f'<div class="meta">{option["at_risk_count"]} of {option["cell_count"]} cells</div>'
        f'<div class="meta" style="margin-top:0">{term("ffdi")} {option["max_ffdi"]:.1f}</div>'
        "</div>"
    )


def _day_name(day: date) -> str:
    """'Thursday 24 Sep', or 'Today' / 'Tomorrow' with the date."""
    label = _day_label(day)
    full = day.strftime("%A")
    return f"{label if label in ('Today', 'Tomorrow') else full} {day.strftime('%d %b')}"


def _at_risk_text(option: dict) -> str:
    return f'{option["at_risk_count"]} of {option["cell_count"]} cells at risk'


def week_summary_html(options: list) -> str:
    """
    One line naming the riskiest and calmest day, e.g.
    "Riskiest: Thursday 24 Sep (Extreme, 40 of 137 cells at risk)
     · Lowest risk: Monday 21 Sep (3 of 137 cells at risk)".
    Empty when there are fewer than two days to compare.
    """
    highlights = week_highlights(options)
    if highlights is None:
        return ""
    worst, best = highlights["worst"], highlights["best"]
    if highlights["flat"]:
        return (
            f'<div class="week-line"><span>{icon("minus", 15)}<span>Risk stays about the same '
            f"all week: peak level <b>{worst['peak_level']}</b>, with "
            f"{best['at_risk_count']}–{worst['at_risk_count']} of {worst['cell_count']} "
            "cells at risk each day.</span></span></div>"
        )
    return (
        '<div class="week-line">'
        f'<span><span class="up">{icon("trend_up", 15)}</span><span>Riskiest day: '
        f"<b>{_day_name(worst['date'])}</b> ({worst['peak_level']} peak, {_at_risk_text(worst)})</span></span>"
        f'<span><span class="down">{icon("trend_down", 15)}</span><span>Lowest risk: '
        f"<b>{_day_name(best['date'])}</b> ({_at_risk_text(best)})</span></span>"
        "</div>"
    )


def _select(day: date) -> None:
    st.session_state[SESSION_KEY] = day


def resolve_selected_day(options: list):
    """
    Returns the currently selected date, falling back to the first day.

    Separate from render_day_strip() so the page can use the selection (for
    the status banner) before the strip itself is drawn further down.
    """
    if not options:
        return None
    dates = [option["date"] for option in options]
    # The available days change with the region, the data source and every
    # refresh, so a previously selected day may no longer exist.
    if st.session_state.get(SESSION_KEY) not in dates:
        st.session_state[SESSION_KEY] = dates[0]
    return st.session_state[SESSION_KEY]


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
    selected = resolve_selected_day(options)

    if len(options) == 1:
        hint = "Only one day in this run. Switch to Live for the full outlook."
    else:
        hint = "Select a day to update the cards and map"

    with st.container(key="daystrip"):
        html(
            f"""
            <div class="day-strip-head">
              <span class="title">Forecast days</span>
              <span class="hint">{hint}</span>
            </div>
            {week_summary_html(options)}
            """
        )

        cols = st.columns(max(SLOTS, len(options)), gap="small")
        for index, col in enumerate(cols):
            with col:
                if index < len(options):
                    day = dates[index]
                    st.button(
                        _day_label(day),
                        key=f"forecast_day_{day}",
                        width="stretch",
                        type="primary" if day == selected else "secondary",
                        on_click=_select,
                        args=(day,),
                    )
                    html(_tile_html(options[index], day == selected))
                else:
                    missing = dates[-1] + timedelta(days=index - len(options) + 1)
                    html(
                        f'<div class="day-placeholder">{missing.strftime("%a %d %b")}'
                        "<br>not in this run</div>"
                    )

    return selected
