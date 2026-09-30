"""
Plotly chart components for the wildfire dashboard.

Four charts, each a small function that renders directly into the current
Streamlit container:

  1. render_risk_trend        — predicted fire probability per forecast day (line/area)
  2. render_risk_outlook      — grid cells per risk level, per forecast day (stacked bar)
  3. render_risk_distribution — grid cells by risk level, selected day (donut)
  4. render_factor_weights    — trained model's feature importance (horizontal bar)

Charts 1-3 are driven by the region's prediction rows; chart 4 by the
/model/feature-importance endpoint, which reads the trained model directly.

They all read colors from utils.theme.COLORS so they stay visually
consistent with the rest of the app.
"""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.regions import RISK_ORDER
from utils.styles import html, section_header
from utils.theme import COLORS, FONT_BODY

PLOT_FONT = dict(family=FONT_BODY, color=COLORS["text_muted"], size=12)
GRID_COLOR = COLORS["border"]
CHART_CONFIG = {"displayModeBar": False}


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
        hoverlabel=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            font=dict(family=FONT_BODY, color=COLORS["text"], size=12),
        ),
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
    # A line needs two points; with a single day, larger markers keep the
    # values visible instead of leaving two lonely dots.
    marker_size = 8 if len(days) > 1 else 14

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=max_pct,
            name="Peak cell",
            mode="lines+markers",
            line=dict(color=COLORS["flame"], width=2.5, shape="spline", smoothing=0.6),
            marker=dict(size=marker_size, line=dict(color=COLORS["surface"], width=2)),
            fill="tozeroy",
            fillcolor="rgba(242, 84, 91, 0.12)",
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
            marker=dict(size=marker_size - 2, line=dict(color=COLORS["surface"], width=2)),
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
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ----------------------------------------------------------------------------
# 2. RISK OUTLOOK BY DAY
# ----------------------------------------------------------------------------
def render_risk_outlook(predictions: pd.DataFrame) -> None:
    """
    How many of the region's grid cells fall in each risk level, per forecast
    day.

    The donut (chart 3) shows this split for the selected day only; this puts
    every day side by side, so the user can see which days the danger spreads
    across the region and plan around them. The label above each bar is the
    share of cells at High or Extreme -- the figure that matters for action.
    """
    st.markdown(
        "<div class='chart-title'>Risk Outlook by Day</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='chart-sub'>Grid cells per risk level &middot; label = % of cells "
        "at High or Extreme</div>",
        unsafe_allow_html=True,
    )

    if predictions is None or predictions.empty:
        st.info("No forecast data available for this region.")
        return

    dates = pd.to_datetime(predictions["acq_date"]).dt.normalize()
    counts = (
        pd.crosstab(dates, predictions["risk_level"])
        .reindex(columns=RISK_ORDER, fill_value=0)
        .sort_index()
    )
    totals = counts.sum(axis=1)
    days = [d.strftime("%a %d %b") for d in counts.index]

    fig = go.Figure()
    # Extreme sits on the baseline so the most dangerous share is the easiest
    # to compare between days; calmer levels stack on top.
    for level in reversed(RISK_ORDER):
        share = counts[level] / totals
        fig.add_trace(
            go.Bar(
                x=days,
                y=counts[level],
                name=level,
                customdata=share,
                marker=dict(
                    color=COLORS[level.lower()],
                    line=dict(color=COLORS["surface"], width=2),
                ),
                hovertemplate=(
                    f"%{{x}}<br>{level}: %{{y}} cells (%{{customdata:.0%}})"
                    "<extra></extra>"
                ),
            )
        )

    high_share = (counts["High"] + counts["Extreme"]) / totals * 100
    fig.add_trace(
        go.Scatter(
            x=days,
            y=totals,
            mode="text",
            text=[f"{v:.0f}%" for v in high_share],
            textposition="top center",
            textfont=dict(color=COLORS["text"], size=11),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        **_base_layout(
            barmode="stack",
            bargap=0.35,
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
                range=[0, totals.max() * 1.15],
                gridcolor=GRID_COLOR,
                title="Grid cells",
                zeroline=False,
            ),
            xaxis=dict(showgrid=False),
        )
    )
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


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
        "<div class='chart-sub'>Grid cells by predicted risk level &middot; selected forecast day</div>",
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
            marker=dict(colors=colors, line=dict(color=COLORS["surface"], width=2)),
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
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    with col2:
        html(
            f"""
            <div class="donut-facts">
              <div>Cells flagged at risk<b>{at_risk} of {total}</b></div>
              <div>Peak probability<b>{summary["max_probability"]:.1%}</b></div>
              <div>Peak FFDI<b>{summary["max_ffdi"]:.1f}</b></div>
              <div>Peak KBDI<b>{summary["max_kbdi"]:.1f} / 203.2</b></div>
            </div>
            """
        )


# ----------------------------------------------------------------------------
# 4. MODEL FACTOR WEIGHTS
# ----------------------------------------------------------------------------
# The model uses ~27 engineered inputs; most are unreadable to a non-specialist
# ("et0_fao_evapotranspiration", "month_cos"). They are grouped into the
# handful of factors a user would recognise, and the chart shows each group's
# summed share. Hovering a bar lists the underlying features.
FACTOR_GROUPS = [
    ("Location", ["lat_round", "lon_round"]),
    (
        "Fuel & soil dryness",
        [
            "soil_moisture_0_to_7cm",
            "days_since_rain",
            "kbdi",
            "drought_factor",
            "precipitation",
            "et0_fao_evapotranspiration",
            "et0_fao_evapotranspiration_sum",
        ],
    ),
    (
        "Heat & air dryness",
        [
            "temperature_2m",
            "temperature_2m_max",
            "relative_humidity_2m",
            "relative_humidity_2m_min",
            "vapour_pressure_deficit",
            "vapour_pressure_deficit_max",
            "emc",
        ],
    ),
    ("Season", ["month_sin", "month_cos"]),
    (
        "Wind",
        [
            "wind_speed_10m",
            "wind_speed_10m_max",
            "wind_gusts_10m",
            "wind_gusts_10m_max",
            "wind_direction_10m",
            "wind_direction_at_max_wind",
        ],
    ),
    ("Fire danger indices", ["ffdi", "ffdi_max", "rate_of_spread"]),
]

FEATURE_LABELS = {
    "lat_round": "Latitude",
    "lon_round": "Longitude",
    "soil_moisture_0_to_7cm": "Topsoil moisture",
    "days_since_rain": "Days since rain",
    "kbdi": "Drought index (KBDI)",
    "drought_factor": "Drought factor",
    "precipitation": "Rainfall",
    "et0_fao_evapotranspiration": "Evapotranspiration",
    "et0_fao_evapotranspiration_sum": "Evapotranspiration (daily total)",
    "temperature_2m": "Temperature",
    "temperature_2m_max": "Max temperature",
    "relative_humidity_2m": "Humidity",
    "relative_humidity_2m_min": "Min humidity",
    "vapour_pressure_deficit": "Vapour pressure deficit",
    "vapour_pressure_deficit_max": "Max vapour pressure deficit",
    "emc": "Fuel moisture (EMC)",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
    "wind_speed_10m": "Wind speed",
    "wind_speed_10m_max": "Max wind speed",
    "wind_gusts_10m": "Wind gusts",
    "wind_gusts_10m_max": "Strongest gust",
    "wind_direction_10m": "Wind direction",
    "wind_direction_at_max_wind": "Wind direction (strongest wind)",
    "ffdi": "FFDI",
    "ffdi_max": "Peak FFDI",
    "rate_of_spread": "Rate of spread",
}


def _group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Sums per-feature importance into FACTOR_GROUPS; anything unmapped
    (e.g. a feature added at retraining) lands in 'Other' rather than
    silently disappearing."""
    lookup = {f: group for group, feats in FACTOR_GROUPS for f in feats}
    df = importance.assign(
        group=importance["feature"].map(lookup).fillna("Other"),
        label=importance["feature"].map(FEATURE_LABELS).fillna(importance["feature"]),
    ).sort_values("importance", ascending=False)

    rows = []
    for group, members in df.groupby("group", sort=False):
        detail = "<br>".join(
            f"  {r.label}: {r.importance:.1f}%" for r in members.itertuples()
        )
        rows.append(
            {"group": group, "importance": members["importance"].sum(), "detail": detail}
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def render_factor_weights(
    importance: pd.DataFrame, error: Optional[str] = None
) -> None:
    """Model feature-importance ranking as a horizontal bar chart. This is a
    property of the trained model, not the region, so it doesn't change when
    the region selector changes."""
    st.markdown(
        "<div class='chart-title'>What Drives the Prediction</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='chart-sub'>Share of the model's decisions per factor (LightGBM gain) "
        "&middot; same for every region</div>",
        unsafe_allow_html=True,
    )

    if importance is None or importance.empty:
        st.info(error or "Feature importance is not available.")
        return

    grouped = _group_importance(importance)
    values = grouped["importance"].round(1)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=grouped["group"],
            orientation="h",
            customdata=grouped["detail"],
            marker=dict(color=COLORS["spread"], line_width=0),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        **_base_layout(
            height=236,
            margin=dict(l=8, r=16, t=4, b=8),
            xaxis=dict(range=[0, values.max() * 1.25], showgrid=False, visible=False),
            yaxis=dict(autorange="reversed"),
            bargap=0.35,
        )
    )
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ----------------------------------------------------------------------------
# LAYOUT: put all four in a 2x2 grid
# ----------------------------------------------------------------------------
def render_insights_section(
    trend: pd.DataFrame,
    summary: Dict,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    importance_error: Optional[str] = None,
) -> None:
    """
    Renders the full 'Insights' section: a 2x2 grid of charts.

    Args:
        trend: per-day aggregates from utils.regions.build_daily_trend().
        summary: region summary from utils.regions.summarise_region().
        predictions: the region's prediction rows across every forecast day.
        importance: per-feature importance from api_client.fetch_feature_importance().
        importance_error: error message to show if importance could not be fetched.
    """
    section_header(
        "Insights",
        "How the forecast develops",
        "Trends across the forecast window, and what the model bases its predictions on.",
    )

    row1_col1, row1_col2 = st.columns(2, gap="medium")
    with row1_col1:
        with st.container(key="card-trend"):
            render_risk_trend(trend)
    with row1_col2:
        with st.container(key="card-outlook"):
            render_risk_outlook(predictions)

    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        with st.container(key="card-distribution"):
            render_risk_distribution(summary)
    with row2_col2:
        with st.container(key="card-factors"):
            render_factor_weights(importance, importance_error)
