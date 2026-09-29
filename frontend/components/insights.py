"""
Row of four headline stat tiles for the selected region and day.
"""

import streamlit as st

from utils.regions import ffdi_band
from utils.styles import html, icon
from utils.theme import COLORS, SEVERITY_STYLE


def _tile(
    label: str,
    icon_name: str,
    body: str,
    accent: str = COLORS["text_muted"],
    accent_bg: str = COLORS["bg"],
) -> None:
    html(
        f"""
        <div class="stat-tile" style="--accent:{accent};--accent-bg:{accent_bg};">
          <div class="stat-head">
            <div class="stat-icon">{icon(icon_name, 17)}</div>
            <div class="stat-label">{label}</div>
          </div>
          {body}
        </div>
        """
    )


def render_insights_row(summary: dict):
    """
    Renders the four headline tiles from a region summary produced by
    utils.regions.summarise_region().

    When there is no data the tiles still render, saying so plainly -- an
    empty row would look like a rendering bug rather than missing data.
    """
    with st.container(key="stats"):
        cols = st.columns(4)
        tiles = [
            ("Cells at Risk", "grid"),
            ("Risk Level", "flame"),
            ("Peak FFDI", "gauge"),
            ("Conditions", "sun"),
        ]

        if not summary.get("has_data"):
            for col, (label, icon_name) in zip(cols, tiles):
                with col:
                    _tile(
                        label,
                        icon_name,
                        '<div class="stat-context" style="margin-top:14px">'
                        "No prediction data available for this region.</div>",
                    )
            return

        cells, at_risk = summary["cell_count"], summary["at_risk_count"]
        share = at_risk / cells if cells else 0
        level = summary["max_risk_level"]
        risk_colour, risk_bg = SEVERITY_STYLE.get(level, (COLORS["border"], COLORS["bg"]))
        ffdi = summary["max_ffdi"]

        with cols[0]:
            _tile(
                "Cells at Risk",
                "grid",
                f"""
                <div class="stat-value">{at_risk} <small>of {cells}</small></div>
                <div class="stat-context">{at_risk} of {cells} grid cells predicted to
                  see fire activity ({share:.0%}).</div>
                <div class="bar"><span style="width:{share * 100:.1f}%"></span></div>
                """,
                accent=COLORS["ember"],
                accent_bg=COLORS["ember_bg"],
            )

        with cols[1]:
            _tile(
                "Risk Level",
                "flame",
                f"""
                <div class="stat-value" style="color:{risk_colour}">{level}</div>
                <div class="stat-context">{level} — peak fire probability
                  {summary["max_probability"]:.0%} across the region.</div>
                <div class="bar"><span style="width:{summary["max_probability"] * 100:.1f}%"></span></div>
                """,
                accent=risk_colour,
                accent_bg=risk_bg,
            )

        with cols[2]:
            _tile(
                "Peak FFDI",
                "gauge",
                f"""
                <div class="stat-value">{ffdi:.1f} <small>{ffdi_band(ffdi)}</small></div>
                <div class="stat-context">{ffdi:.1f} ({ffdi_band(ffdi)}) — regional
                  average {summary["mean_ffdi"]:.1f}.</div>
                <div class="bar"><span style="width:{min(ffdi, 100):.1f}%"></span></div>
                """,
                accent=COLORS["high"],
                accent_bg=COLORS["high_bg"],
            )

        with cols[3]:
            _tile(
                "Conditions",
                "sun",
                f"""
                <div class="conditions">
                  <div>{icon("thermometer", 13)} Temp<b>{summary["mean_temp"]:.0f}°C</b></div>
                  <div>{icon("droplet", 13)} Humidity<b>{summary["mean_humidity"]:.0f}%</b></div>
                  <div>{icon("wind", 13)} Wind<b>{summary["mean_wind"]:.0f}<small style="font-size:.7rem"> km/h</small></b></div>
                </div>
                <div class="stat-context">Regional averages: {summary["mean_temp"]:.0f}°C,
                  {summary["mean_humidity"]:.0f}% humidity, wind {summary["mean_wind"]:.0f} km/h.</div>
                """,
                accent=COLORS["spread"],
                accent_bg=COLORS["spread_bg"],
            )
