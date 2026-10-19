"""
Chart 4 -- Where the Danger Lasts: which places, and on which days?

A day-by-day timeline for the places that stay at High or Extreme risk the
longest: one row per place, one slot per day, filled on its dangerous days.
The only view that combines where AND when.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import HOTSPOT_ROWS
from core.chart_data import day_dates, hotspot_labels
from core.hotspots import persistent_hotspots
from utils.theme import COLORS

from components.charts._common import base_layout, chart_title, legend_top, show_chart

LOWER_RISK_TINT = "#E9E5DE"  # Low / Moderate days

# (legend name, risk levels drawn in that colour, colour)
SLOT_GROUPS = (
    ("Extreme", ["Extreme"], COLORS["extreme"]),
    ("High", ["High"], COLORS["high"]),
    ("Lower risk", ["Moderate", "Low"], LOWER_RISK_TINT),
)


def _headline(all_hotspots: pd.DataFrame, n_days: int) -> str:
    every_day = int((all_hotspots["days_elevated"] == n_days).sum())
    if n_days == 1:
        return f"<b>{len(all_hotspots)}</b> places are at High or Extreme risk on this forecast day."
    if every_day:
        noun = "place is" if every_day == 1 else "places are"
        return f"<b>{every_day}</b> {noun} at High or Extreme risk on all {n_days} days."
    return (f"<b>{len(all_hotspots)}</b> places reach High or Extreme risk at least once; "
            f"none stays there all {n_days} days.")


def render_persistent_hotspots(predictions: pd.DataFrame) -> None:
    days_all = sorted(day_dates(predictions).unique()) if predictions is not None and not predictions.empty else []
    all_hotspots = persistent_hotspots(predictions, top=None)
    hotspots = all_hotspots.head(HOTSPOT_ROWS)

    if hotspots.empty:
        chart_title("Where the Danger Lasts", "Places at High or Extreme risk, day by day")
        st.info(
            "No place reaches High or Extreme risk in this forecast."
            if days_all else "No forecast data available for this region."
        )
        return

    n_days = len(days_all)
    chart_title("Where the Danger Lasts",
                f"{_headline(all_hotspots, n_days)} The {len(hotspots)} most persistent, day by day.")

    labels = hotspot_labels(hotspots)
    label_of = {(r.lat_round, r.lon_round): lab
                for r, lab in zip(hotspots.itertuples(index=False), labels)}
    cells = predictions.assign(day=day_dates(predictions))
    cells = cells[[k in label_of for k in zip(cells["lat_round"], cells["lon_round"])]]
    day_index = {d: i for i, d in enumerate(days_all)}

    fig = go.Figure()
    for name, levels, colour in SLOT_GROUPS:
        part = cells[cells["risk_level"].astype(str).isin(levels)]
        fig.add_trace(go.Bar(
            name=name, orientation="h",
            y=[label_of[k] for k in zip(part["lat_round"], part["lon_round"])],
            x=[0.86] * len(part),
            base=[day_index[d] + 0.07 for d in part["day"]],
            marker=dict(color=colour, line=dict(width=0)),
            customdata=list(zip(part["day"].dt.strftime("%a %d %b"),
                                part["risk_level"].astype(str),
                                (part["fire_probability"] * 100).round(0))),
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]}: %{customdata[1]}"
                           " (%{customdata[2]:.0f}% chance)<extra></extra>"),
        ))
    fig.add_trace(go.Scatter(
        x=[n_days + 0.1] * len(hotspots), y=labels, mode="text",
        text=[f"{d} of {n_days}" for d in hotspots["days_elevated"]],
        textposition="middle right", textfont=dict(color=COLORS["text"], size=11),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_traces(cliponaxis=False)
    fig.update_layout(**base_layout(
        height=max(170, 30 * len(hotspots) + 80),
        margin=dict(l=8, r=16, t=28, b=28),
        barmode="overlay", showlegend=True, legend=legend_top(traceorder="normal"),
        xaxis=dict(range=[0, n_days + 1.1], showgrid=False, zeroline=False,
                   tickvals=[i + 0.5 for i in range(n_days)],
                   ticktext=[pd.Timestamp(d).strftime("%a<br>%d") for d in days_all]),
        yaxis=dict(autorange="reversed", automargin=True, categoryorder="array",
                   categoryarray=labels),
        bargap=0.3,
    ))
    show_chart(fig)
