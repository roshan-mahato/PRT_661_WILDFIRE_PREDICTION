"""
Chart 1 -- Fire Weather This Week: WHY does the danger change?

The weather behind the forecast. Every other chart shows the model's
prediction; this shows the fire danger index (FFDI) that the weather
produces, on bands named in words, so a change in risk from one day to the
next has a visible reason.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.chart_data import capitalise, daily_fire_weather, daily_town_fire_weather
from core.focus import town_rows
from core.danger import FFDI_BANDS, ffdi_band_name
from core.places import describe_location
from utils.glossary import term
from utils.theme import COLORS

from components.charts._common import base_layout, chart_title, legend_top, show_chart

# One warm tint per FFDI band, light to dark. A single hue on purpose -- the
# model's risk colours (green/gold/orange/red) mean something else, and
# reusing them here would blur the two scales.
BAND_TINTS = ["#FAF7F2", "#FBEBD8", "#F8D9BC", "#F3BFA0", "#EC9F82", "#E07F66"]

# Candidate tops for the y axis: the first one comfortably above the peak.
Y_AXIS_STEPS = (25, 50, 75, 100, 150)


def _band_shapes(y_max: float) -> tuple:
    """Shaded FFDI bands up to y_max, with their names in the right margin
    (outside the plot, so they never collide with the lines)."""
    shapes, labels = [], []
    bounds = [lower for lower, _ in FFDI_BANDS] + [float("inf")]
    for i, (lower, name) in enumerate(FFDI_BANDS):
        if lower >= y_max:
            break
        upper = min(bounds[i + 1], y_max)
        shapes.append(dict(
            type="rect", xref="paper", x0=0, x1=1, yref="y", y0=lower, y1=upper,
            fillcolor=BAND_TINTS[i], line=dict(width=0), layer="below",
        ))
        labels.append(dict(
            xref="paper", x=1.01, xanchor="left", yref="y", y=(lower + upper) / 2,
            text=name, showarrow=False, font=dict(size=10, color=COLORS["text_dim"]),
        ))
    return shapes, labels


def render_fire_weather(predictions: pd.DataFrame, focus=None) -> None:
    """
    Region mode: the worst place (solid) and a typical area (dashed).
    Town focus: the searched town's own area (solid) and the worst place
    within the radius (dashed).
    """
    if focus is None:
        subtitle = (f"The weather behind the forecast: fire danger ({term('ffdi')}) at the worst "
                    "place and in a typical area &middot; shaded by danger band")
    else:
        subtitle = (f"Fire danger ({term('ffdi')}) in {focus.name}'s area and at the worst "
                    f"place within {focus.radius_km} km &middot; shaded by danger band")
    chart_title("Fire Weather This Week", subtitle)
    if predictions is None or predictions.empty:
        st.info("No forecast data available for this region.")
        return

    weather = daily_fire_weather(predictions)
    days = [d.strftime("%a %d %b") for d in weather["date"]]
    marker_size = 8 if len(days) > 1 else 14

    worst_places = [
        capitalise(describe_location(lat, lon))
        for lat, lon in zip(weather["worst_lat"], weather["worst_lon"])
    ]
    fig = go.Figure()
    if focus is not None:
        _add_town_traces(fig, weather, days, worst_places, marker_size, focus, town_rows(predictions, focus))
        _finish(fig, weather)
        return
    fig.add_trace(go.Scatter(
        x=days, y=weather["worst_ffdi"], name="Worst place",
        mode="lines+markers",
        line=dict(color=COLORS["ember"], width=2.5, shape="spline", smoothing=0.6),
        marker=dict(size=marker_size, line=dict(color=COLORS["surface"], width=2)),
        customdata=list(zip(
            [ffdi_band_name(v) for v in weather["worst_ffdi"]], worst_places,
            weather["worst_weather"],
        )),
        hovertemplate=(
            "%{x}<br><b>Worst: %{customdata[0]} fire weather</b> (FFDI %{y:.0f})"
            "<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=weather["typical_ffdi"], name="Typical area",
        mode="lines+markers",
        line=dict(color=COLORS["text_muted"], width=1.6, dash="dash"),
        marker=dict(size=marker_size - 2, line=dict(color=COLORS["surface"], width=2)),
        customdata=list(zip(
            [ffdi_band_name(v) for v in weather["typical_ffdi"]], weather["typical_weather"],
        )),
        hovertemplate=(
            "%{x}<br><b>Typical: %{customdata[0]} fire weather</b> (FFDI %{y:.0f})"
            "<br>Regional averages: %{customdata[1]}<extra></extra>"
        ),
    ))

    _finish(fig, weather)


def _add_town_traces(fig, weather, days, worst_places, marker_size, focus, town) -> None:
    """Town focus: the town's own area (solid) and the worst place nearby (dashed)."""
    own = daily_town_fire_weather(town).set_index("date").reindex(weather["date"])
    fig.add_trace(go.Scatter(
        x=days, y=own["town_ffdi"], name=f"{focus.name}'s area",
        mode="lines+markers",
        line=dict(color=COLORS["ember"], width=2.5, shape="spline", smoothing=0.6),
        marker=dict(size=marker_size, line=dict(color=COLORS["surface"], width=2)),
        customdata=list(zip(
            [ffdi_band_name(v) if pd.notna(v) else "No forecast" for v in own["town_ffdi"]],
            own["town_weather"].fillna(""),
        )),
        hovertemplate=(
            f"%{{x}}<br><b>{focus.name}: %{{customdata[0]}} fire weather</b> (FFDI %{{y:.0f}})"
            "<br>%{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=weather["worst_ffdi"], name=f"Worst within {focus.radius_km} km",
        mode="lines+markers",
        line=dict(color=COLORS["text_muted"], width=1.6, dash="dash"),
        marker=dict(size=marker_size - 2, line=dict(color=COLORS["surface"], width=2)),
        customdata=list(zip(
            [ffdi_band_name(v) for v in weather["worst_ffdi"]], worst_places,
            weather["worst_weather"],
        )),
        hovertemplate=(
            "%{x}<br><b>Worst nearby: %{customdata[0]} fire weather</b> (FFDI %{y:.0f})"
            "<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
        ),
    ))


def _finish(fig, weather) -> None:
    """Danger bands, axes and layout shared by both modes, then draw."""
    peak = float(weather["worst_ffdi"].max())
    y_max = next((b for b in Y_AXIS_STEPS if b > peak * 1.15), peak * 1.15)
    shapes, band_labels = _band_shapes(y_max)
    fig.update_layout(**base_layout(
        showlegend=True, legend=legend_top(),
        shapes=shapes, annotations=band_labels,
        margin=dict(l=44, r=86, t=8, b=36),
        yaxis=dict(range=[0, y_max], showgrid=False, zeroline=False,
                   title="Fire danger index"),
        xaxis=dict(showgrid=False),
    ))
    show_chart(fig)
