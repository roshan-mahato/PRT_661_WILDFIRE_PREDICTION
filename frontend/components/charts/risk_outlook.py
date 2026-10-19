"""
Chart 2 -- How Widespread Is the Risk?: how much of the region, each day?

The only chart about how widespread the danger is. Each bar is one day and
shows the share of the region's land at each risk level, with a "1 in N"
label for High or Extreme.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.chart_data import daily_risk_shares
from core.danger import RISK_ORDER
from core.plain import area_comparison, format_area, one_in
from utils.theme import COLORS

from components.charts._common import GRID_COLOR, base_layout, chart_title, legend_top, show_chart


def render_risk_outlook(predictions: pd.DataFrame) -> None:
    chart_title(
        "How Widespread Is the Risk?",
        "Share of the region at each risk level, by day &middot; "
        "label = share at High or Extreme",
    )
    if predictions is None or predictions.empty:
        st.info("No forecast data available for this region.")
        return

    shares, area, counts = daily_risk_shares(predictions)
    days = [d.strftime("%a %d %b") for d in shares.index]

    fig = go.Figure()
    # Extreme sits on the baseline so the most dangerous share is the easiest
    # to compare between days; calmer levels stack on top.
    for level in reversed(RISK_ORDER):
        comparison = [area_comparison(a) for a in area[level]]
        fig.add_trace(go.Bar(
            x=days, y=shares[level] * 100, name=level,
            customdata=list(zip(
                [format_area(a) for a in area[level]],
                [f" · {c}" if c else "" for c in comparison],
                counts[level],
            )),
            marker=dict(color=COLORS[level.lower()],
                        line=dict(color=COLORS["surface"], width=2)),
            hovertemplate=(
                f"%{{x}}<br><b>{level}</b>: %{{y:.0f}}% of the region"
                "<br>about %{customdata[0]}%{customdata[1]}"
                "<br>%{customdata[2]} areas<extra></extra>"
            ),
        ))

    elevated = shares["High"] + shares["Extreme"]
    fig.add_trace(go.Scatter(
        x=days, y=[100] * len(days), mode="text",
        text=[f"<b>{s:.0%}</b><br>{one_in(s)}" for s in elevated],
        textposition="top center", textfont=dict(color=COLORS["text"], size=11),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(**base_layout(
        barmode="stack", bargap=0.35, showlegend=True, legend=legend_top(),
        margin=dict(l=44, r=16, t=8, b=36),
        yaxis=dict(range=[0, 128], tickvals=[0, 25, 50, 75, 100], ticksuffix="%",
                   gridcolor=GRID_COLOR, zeroline=False, title="Share of region"),
        xaxis=dict(showgrid=False),
    ))
    show_chart(fig)
