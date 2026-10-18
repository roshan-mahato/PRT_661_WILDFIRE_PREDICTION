"""
Shared Plotly styling for the four Insights charts, so they all have the same
transparent background, margins, fonts, legend placement and title markup.
"""

import streamlit as st
from utils.theme import COLORS, FONT_BODY

PLOT_FONT = dict(family=FONT_BODY, color=COLORS["text_muted"], size=12)
GRID_COLOR = COLORS["border"]
CHART_CONFIG = {"displayModeBar": False}


def base_layout(height: int = 280, **overrides) -> dict:
    """Shared Plotly layout -- override anything per chart via kwargs."""
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


def legend_top(**extra) -> dict:
    """Horizontal legend along the top-left of the plot."""
    return dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11), **extra)


def chart_title(title: str, subtitle: str) -> None:
    """Card heading: title plus a one-line explanation underneath."""
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chart-sub'>{subtitle}</div>", unsafe_allow_html=True)


def show_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
