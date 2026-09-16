"""
Map panel: plots predicted fire risk per grid cell, with a region threat
summary alongside it.
"""

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from utils.regions import ffdi_band, get_region_view
from utils.theme import COLORS, SEVERITY_STYLE

# Heat gradient reuses the severity palette so the map agrees with the cards.
HEAT_GRADIENT = {
    0.25: COLORS["low"],
    0.50: COLORS["moderate"],
    0.75: COLORS["high"],
    1.00: COLORS["extreme"],
}


def _marker_radius(probability: float) -> float:
    """Scales marker size with probability so high-risk cells read first."""
    return 4 + (probability * 8)


def render_map_with_insights(predictions: pd.DataFrame, region: str, summary: dict):
    """
    Renders the Folium map (80%) and the region threat summary (20%).

    Args:
        predictions: prediction rows for this region, already narrowed to the
                     forecast day chosen in the day strip. One row per cell --
                     passing several days would stack markers on the same
                     coordinates.
        region: selected region name.
        summary: output of utils.regions.summarise_region().
    """
    map_col, right_col = st.columns([8, 2])

    with map_col:
        center, zoom = get_region_view(region)

        attr = (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, '
            '&copy; <a href="https://carto.com/attributions">CARTO</a>'
        )
        tiles = (
            "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
            "?key=cb1_2s99_1_503a3a7bf96ca39604de30b4"
        )

        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=tiles,
            control_scale=False,
            attr=attr,
        )

        day = predictions

        if not day.empty:
            heat_data = [
                [row.lat_round, row.lon_round, float(row.fire_probability)]
                for row in day.itertuples(index=False)
            ]
            HeatMap(
                heat_data,
                radius=18,
                blur=22,
                max_zoom=1,
                gradient=HEAT_GRADIENT,
            ).add_to(m)

            for row in day.itertuples(index=False):
                colour, _ = SEVERITY_STYLE.get(row.risk_level, (COLORS["low"], ""))
                spinup_note = (
                    "<br><i>Low confidence: cell has under 30 days of history.</i>"
                    if getattr(row, "kbdi_spinup_flag", False)
                    else ""
                )
                popup_html = (
                    f"<b>{row.risk_level} risk</b><br>"
                    f"Fire probability: {row.fire_probability:.1%}<br>"
                    f"FFDI: {row.ffdi:.1f} ({ffdi_band(row.ffdi)})<br>"
                    f"KBDI: {row.kbdi:.1f}<br>"
                    f"Drought factor: {row.drought_factor:.1f}<br>"
                    f"Temp: {row.temperature_2m:.1f}&deg;C, "
                    f"RH: {row.relative_humidity_2m:.0f}%<br>"
                    f"Wind: {row.wind_speed_10m:.0f} km/h<br>"
                    f"Cell: {row.lat_round}, {row.lon_round}"
                    f"{spinup_note}"
                )
                folium.CircleMarker(
                    location=[row.lat_round, row.lon_round],
                    radius=_marker_radius(float(row.fire_probability)),
                    color=colour,
                    weight=1,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.75,
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f"{row.risk_level} — {row.fire_probability:.0%}",
                ).add_to(m)

        st_folium(
            m,
            use_container_width=True,
            height=800,
            returned_objects=[],
        )

        if day.empty:
            st.info(
                "No predictions to map. Fetch live weather, then run a refresh "
                "to generate predictions."
            )

    with right_col:
        _render_threat_panel(summary)


def _render_threat_panel(summary: dict):
    """Right-hand summary panel, driven by the region summary."""
    if not summary.get("has_data"):
        st.markdown(
            f"""
            <div style="border: 1px solid {COLORS['border']}; border-radius: 12px;
                        padding: 16px; background: {COLORS['surface']};">
                <h4 style="margin-top:0; color: {COLORS['text_muted']}; font-size: 1.05rem;">
                    Region Threat Summary
                </h4>
                <hr style="border-color: {COLORS['border']}; margin: 10px 0;">
                <p style="font-size: 0.85rem; color: {COLORS['text_muted']};">
                    No prediction data for this region yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    colour, bg = SEVERITY_STYLE.get(
        summary["max_risk_level"], (COLORS["border"], COLORS["bg"])
    )
    lat, lon = summary["hottest_cell"]
    counts = summary["risk_counts"]
    forecast_date = summary["forecast_date"]

    spinup_note = ""
    if summary["spinup_count"]:
        spinup_note = (
            f"<div style='border-left: 3px solid {COLORS['moderate']}; padding: 8px 10px;"
            f" border-radius: 4px; font-size: 0.78rem; margin-top: 10px;"
            f" background: {COLORS['moderate_bg']};'>"
            f"<b>{summary['spinup_count']}</b> cell(s) have under 30 days of history — "
            f"their drought values are cold-start estimates.</div>"
        )

    st.markdown(
        f"""
        <div style="border: 1px solid {COLORS['border']}; border-radius: 12px;
                    padding: 16px; background: {COLORS['surface']};">
            <h4 style="margin-top:0; color: {colour}; font-size: 1.05rem;">
                Region Threat Summary
            </h4>
            <hr style="border-color: {COLORS['border']}; margin: 10px 0;">
            <p style="font-size: 0.85rem; margin-bottom: 12px;">
                <b>Forecast day:</b><br>{forecast_date}
            </p>
            <p style="font-size: 0.85rem; margin-bottom: 12px;">
                <b>Highest risk cell:</b><br>{lat}, {lon}
            </p>
            <p style="font-size: 0.85rem; margin-bottom: 12px;">
                <b>Peak probability:</b><br>{summary['max_probability']:.1%}
            </p>
            <p style="font-size: 0.85rem; margin-bottom: 12px;">
                <b>Peak FFDI:</b><br>{summary['max_ffdi']:.1f} ({ffdi_band(summary['max_ffdi'])})
            </p>
            <p style="font-size: 0.85rem; margin-bottom: 12px;">
                <b>Peak KBDI:</b><br>{summary['max_kbdi']:.1f} / 203.2
            </p>
            <p style="font-size: 0.85rem; margin-bottom: 4px;"><b>Cells by risk level:</b></p>
            <p style="font-size: 0.8rem; margin: 0 0 12px 0; color: {COLORS['text_muted']};">
                Extreme {counts['Extreme']} &middot; High {counts['High']} &middot;
                Moderate {counts['Moderate']} &middot; Low {counts['Low']}
            </p>
            <div style="border-left: 3px solid {colour}; padding: 8px 10px;
                        border-radius: 4px; font-size: 0.8rem; background: {bg};">
                <b>Status:</b> {summary['max_risk_level']} risk across
                {summary['cell_count']} monitored cell(s).
            </div>
            {spinup_note}
        </div>
        """,
        unsafe_allow_html=True,
    )
