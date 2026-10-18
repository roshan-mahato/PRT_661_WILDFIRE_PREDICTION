"""
Chart 3 -- Places Affected: which places, on the selected day?

Towns by risk level -- the people-and-places view. Each listed town takes
the forecast for the grid area it sits in.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import TOP_PLACES
from core.chart_data import day_dates, names_preview
from core.danger import RISK_ORDER
from core.places import towns_at_risk
from utils.glossary import term
from utils.styles import html
from utils.theme import COLORS, SEVERITY_STYLE

from components.charts._common import base_layout, chart_title, show_chart


def render_places_affected(day_predictions: pd.DataFrame) -> None:
    day_label = (
        day_dates(day_predictions).max().strftime("%a %d %b")
        if day_predictions is not None and not day_predictions.empty else "selected day"
    )
    chart_title("Places Affected", f"Towns by risk level &middot; {day_label}")

    towns = towns_at_risk(day_predictions)
    if towns.empty:
        st.info("No listed towns fall inside the forecast areas for this region.")
        return

    by_level = {lvl: towns.loc[towns["risk_level"] == lvl, "town"].tolist() for lvl in RISK_ORDER}
    values = [len(by_level[lvl]) for lvl in RISK_ORDER]
    elevated = values[2] + values[3]
    noun = "town" if elevated == 1 else "towns"

    fig = go.Figure(go.Pie(
        labels=RISK_ORDER, values=values, hole=0.72,
        marker=dict(colors=[COLORS[lvl.lower()] for lvl in RISK_ORDER],
                    line=dict(color=COLORS["surface"], width=2)),
        customdata=[names_preview(by_level[lvl]) for lvl in RISK_ORDER],
        textinfo="none", sort=False, direction="clockwise",
        hovertemplate="<b>%{label}</b>: %{value} towns<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        height=230,
        annotations=[dict(
            text=(f"<b>{elevated}</b><br><span style='font-size:11px;color:{COLORS['text_muted']}'>"
                  f"{noun} at High<br>or Extreme risk</span>"),
            x=0.5, y=0.5, showarrow=False, font=dict(size=26, color=COLORS["text"]),
        )],
    ))

    top = towns.head(TOP_PLACES)
    rows = "".join(
        f'<div class="place-row"><span>{name}</span>'
        f'<span class="pill" style="color:{SEVERITY_STYLE[lvl][0]};background:{SEVERITY_STYLE[lvl][1]}">'
        f"{lvl}</span></div>"
        for name, lvl in zip(top["town"], top["risk_level"].astype(str))
    )
    col1, col2 = st.columns([1.05, 1])
    with col1:
        show_chart(fig)
    with col2:
        html(
            f"""
            <div class="donut-facts">
              <div>Most at-risk places<span class="place-list">{rows}</span></div>
            </div>
            <div class="why-note">{len(towns)} listed towns in this view. Each takes the
              forecast for its {term('grid_cell', '55 km area')}.</div>
            """
        )
