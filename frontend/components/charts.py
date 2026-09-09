"""
Plotly chart components for the wildfire dashboard.

Four charts, each a small function that renders directly into the current
Streamlit container:

  1. render_risk_trend           — predicted vs. observed risk index (line/area)
  2. render_historical_incidents — seasonal incident count (bar)
  3. render_containment_donut    — containment progress (donut)
  4. render_factor_weights       — model feature importance (horizontal bar)

They all read colors from styles.theme.COLORS so they stay visually
consistent with the rest of the app, and take plain dicts/lists as input
(no Streamlit-specific data types) so they're easy to test or reuse.
"""

from typing import Dict

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
def render_risk_trend(data: Dict) -> None:
    """Predicted vs. observed 7-day risk index, as a filled line chart."""
    pred = data["risk_trend_pred"]
    obs = data["risk_trend_obs"]
    days = [f"Day {i + 1}" for i in range(len(pred))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=pred,
            name="Predicted",
            mode="lines",
            line=dict(color=COLORS["flame"], width=3),
            fill="tozeroy",
            fillcolor="rgba(242, 84, 91, 0.15)",
            hovertemplate="%{x}: %{y}<extra>Predicted</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=obs,
            name="Observed",
            mode="lines",
            line=dict(color=COLORS["text_muted"], width=1.6, dash="dash"),
            hovertemplate="%{x}: %{y}<extra>Observed</extra>",
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
                range=[0, 100], gridcolor=GRID_COLOR, title="Risk index", zeroline=False
            ),
            xaxis=dict(showgrid=False),
        )
    )

    st.markdown(
        "<div class='chart-title'>Risk Trend (7-day)</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='chart-sub'>Predicted vs. observed risk index &middot; updates with region</div>",
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
# 3. CONTAINMENT DONUT
# ----------------------------------------------------------------------------
def render_containment_donut(data: Dict) -> None:
    """Containment progress for the primary cluster, as a donut chart with
    supporting stats alongside it."""
    pct = data["containment_pct"]

    fig = go.Figure(
        go.Pie(
            values=[pct, 100 - pct],
            hole=0.74,
            marker=dict(colors=[COLORS["teal"], "#EFEDE7"], line=dict(width=0)),
            textinfo="none",
            sort=False,
            direction="clockwise",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        **_base_layout(
            height=230,
            annotations=[
                dict(
                    text=f"<b>{pct}%</b><br><span style='font-size:11px;color:{COLORS['text_muted']}'>contained</span>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=26, color=COLORS["text"]),
                )
            ],
        )
    )

    st.markdown(
        "<div class='chart-title'>Containment Progress</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='chart-sub'>Primary cluster &middot; updates with region</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown(
            f"""
            <div class="donut-facts" style="margin-top:26px;">
              Est. time remaining: <b>{data["est_hours_to_containment"]}h</b><br>
              Model confidence: <b>{data["confidence_pct"]}%</b><br>
              Growth rate: <b>+{data["growth_rate_km2_hr"]} km&sup2;/hr</b>
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
def render_insights_section(data: Dict) -> None:
    """Render the full 'Insights' section: a 2x2 grid of the four charts
    above. Call this once from app.py with the currently selected region's
    data dict."""
    st.markdown("## Insights")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        with st.container(border=True):
            render_risk_trend(data)
    with row1_col2:
        with st.container(border=True):
            render_historical_incidents()

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True):
            render_containment_donut(data)
    with row2_col2:
        with st.container(border=True):
            render_factor_weights()
