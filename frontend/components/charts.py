"""
Plotly chart components for the wildfire dashboard.

Four charts, each a small function that renders directly into the current
Streamlit container:

  1. render_risk_trend           — predicted fire probability per forecast day (line/area)
  2. render_historical_incidents — seasonal incident count (bar) [reference data]
  3. render_risk_distribution    — grid cells by risk level (donut)
  4. render_factor_weights       — model feature importance (horizontal bar) [reference data]

Charts 1 and 3 are driven by live prediction data from the backend. Charts 2
and 4 still use static reference values from utils.mock_data and should be
replaced with real figures before submission.

They all read colors from utils.theme.COLORS so they stay visually
consistent with the rest of the app.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.mock_data import FACTOR_WEIGHTS, HISTORICAL_INCIDENTS, MONTHS
from utils.theme import COLORS

PLOT_FONT = dict(family="Inter, sans-serif", color=COLORS["text_muted"], size=12)
GRID_COLOR = COLORS["border"]


def _base_layout(height: int = 280, **overrides) -> dict:
    """Shared Plotly layout so every chart has the same transparent
    background, margins, and font — override anything per-chart via kwargs."""
    layout = dict(
        height=height,
        margin=dict(l=36, r=16, t=8, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        showlegend=False,
    )
    layout.update(overrides)
    return layout


# ----------------------------------------------------------------------------
# 1. RISK TREND
# ----------------------------------------------------------------------------
def render_risk_trend(trend: "pd.DataFrame") -> None:
    """
    Forecast fire probability over the prediction window.

    Shows the regional mean alongside the peak cell: the mean tells you the
    general trend, while the peak is what actually matters operationally --
    a low average hides a single cell at extreme risk.
    """
    if trend is None or trend.empty:
        st.markdown(
            "<div class='chart-title'>Fire Risk Forecast</div>", unsafe_allow_html=True
        )
        st.info("No forecast data available for this region.")
        return

    days = [d.strftime("%a %d %b") for d in pd.to_datetime(trend["acq_date"])]
    mean_pct = (trend["mean_probability"] * 100).round(1)
    max_pct = (trend["max_probability"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=max_pct,
            name="Peak cell",
            mode="lines+markers",
            line=dict(color=COLORS["flame"], width=3),
            fill="tozeroy",
            fillcolor="rgba(242, 84, 91, 0.15)",
            hovertemplate="%{x}: %{y:.1f}%<extra>Peak cell</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=mean_pct,
            name="Regional mean",
            mode="lines+markers",
            line=dict(color=COLORS["text_muted"], width=1.6, dash="dash"),
            hovertemplate="%{x}: %{y:.1f}%<extra>Regional mean</extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=11),
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor=GRID_COLOR,
                title="Fire probability (%)",
                zeroline=False,
            ),
            xaxis=dict(showgrid=False),
        )
    )

    st.markdown(
        "<div class='chart-title'>Fire Risk Forecast</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='chart-sub'>Predicted fire probability per forecast day</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ----------------------------------------------------------------------------
# 2. HISTORICAL INCIDENTS
# ----------------------------------------------------------------------------
def render_historical_incidents() -> None:
    """Generic seasonal incident-count bar chart, colored by intensity."""
    vmax = max(HISTORICAL_INCIDENTS)

    def bar_color(v: int) -> str:
        t = v / vmax
        if t < 0.35:
            return COLORS["teal"]
        if t < 0.7:
            return COLORS["gold"]
        return COLORS["flame"]

    colors = [bar_color(v) for v in HISTORICAL_INCIDENTS]

    fig = go.Figure(
        go.Bar(
            x=MONTHS,
            y=HISTORICAL_INCIDENTS,
            marker=dict(color=colors, line_width=0),
            hovertemplate="%{x}: %{y} incidents<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            yaxis=dict(
                showgrid=True, gridcolor=GRID_COLOR, title="Incidents", zeroline=False
            ),
            xaxis=dict(showgrid=False),
            bargap=0.35,
        )
    )

    st.markdown(
        "<div class='chart-title'>Historical Incidents by Month</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='chart-sub'>Typical seasonal pattern, national reference data "
        "&middot; warmer colors = higher-incident months</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ----------------------------------------------------------------------------
# 3. RISK DISTRIBUTION
# ----------------------------------------------------------------------------
def render_risk_distribution(summary: Dict) -> None:
    """
    How the region's grid cells split across risk levels.

    This replaced a containment-progress donut: containment describes fires
    already burning, which this system does not track -- it predicts risk
    ahead of ignition, so cells-by-risk-level is the figure the model can
    actually support.
    """
    st.markdown(
        "<div class='chart-title'>Risk Distribution</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='chart-sub'>Grid cells by predicted risk level &middot; latest forecast day</div>",
        unsafe_allow_html=True,
    )

    if not summary.get("has_data"):
        st.info("No prediction data available for this region.")
        return

    counts = summary["risk_counts"]
    labels = ["Low", "Moderate", "High", "Extreme"]
    values = [counts[label] for label in labels]
    colors = [COLORS["low"], COLORS["moderate"], COLORS["high"], COLORS["extreme"]]

    total = sum(values)
    at_risk = summary["at_risk_count"]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.74,
            marker=dict(colors=colors, line=dict(width=0)),
            textinfo="none",
            sort=False,
            direction="clockwise",
            hovertemplate="%{label}: %{value} cells<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            height=230,
            annotations=[
                dict(
                    text=(
                        f"<b>{total}</b><br>"
                        f"<span style='font-size:11px;color:{COLORS['text_muted']}'>cells</span>"
                    ),
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=26, color=COLORS["text"]),
                )
            ],
        )
    )

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown(
            f"""
            <div class="donut-facts" style="margin-top:26px;">
              Cells flagged at risk: <b>{at_risk}</b><br>
              Peak probability: <b>{summary["max_probability"]:.1%}</b><br>
              Peak FFDI: <b>{summary["max_ffdi"]:.1f}</b><br>
              Peak KBDI: <b>{summary["max_kbdi"]:.1f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# 4. MODEL FACTOR WEIGHTS
# ----------------------------------------------------------------------------
def render_factor_weights() -> None:
    """Model feature-importance ranking as a horizontal bar chart. This is a
    property of the trained model, not the region, so it doesn't change when
    the region selector changes."""
    labels = [f[0] for f in FACTOR_WEIGHTS]
    values = [f[1] for f in FACTOR_WEIGHTS]
    colors = [COLORS[f[2]] for f in FACTOR_WEIGHTS]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line_width=0),
            text=[f"{v}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x}%<extra></extra>",
        )
    )
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        **_base_layout(
            height=240,
            xaxis=dict(range=[0, max(values) * 1.3], showgrid=False, visible=False),
            yaxis=dict(autorange="reversed"),
        )
    )

    st.markdown(
        "<div class='chart-title'>Model Factor Weights</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='chart-sub'>Feature importance is a property of the trained model, "
        "consistent across regions</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ----------------------------------------------------------------------------
# LAYOUT: put all four in a 2x2 grid
# ----------------------------------------------------------------------------
def render_insights_section(trend: "pd.DataFrame", summary: Dict) -> None:
    """
    Renders the full 'Insights' section: a 2x2 grid of charts.

    Args:
        trend: per-day aggregates from utils.regions.build_daily_trend().
        summary: region summary from utils.regions.summarise_region().
    """
    st.markdown("## Insights")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        with st.container(border=True):
            render_risk_trend(trend)
    with row1_col2:
        with st.container(border=True):
            render_historical_incidents()

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True):
            render_risk_distribution(summary)
    with row2_col2:
        with st.container(border=True):
            render_factor_weights()
