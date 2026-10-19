"""
Map panel: plots predicted fire risk per grid cell, with a region threat
summary alongside it.
"""

import math
from typing import Optional

import folium
from folium import JsCode
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
import streamlit.components.v1 as components
from config import FOCUS_MAP_VIEW_KM, MAP_HEIGHT
from core.danger import RISK_ORDER, ffdi_band
from core.explain import explain_cell, short_reason
from core.geography import STATE_BOUNDS, get_region_view, is_national
from core.places import describe_location, format_coords
from utils.glossary import term
from utils.styles import html, icon, section_header
from utils.theme import COLORS, SEVERITY_STYLE

# Heat gradient reuses the severity palette so the map agrees with the cards.
HEAT_GRADIENT = {
    0.25: COLORS["low"],
    0.50: COLORS["moderate"],
    0.75: COLORS["high"],
    1.00: COLORS["extreme"],
}


# Colour of a "why" item by how strongly it pushes fire danger up.
DRIVER_COLOURS = {3: COLORS["extreme"], 2: COLORS["high"], 1: COLORS["moderate"]}


def _marker_radius(probability: float) -> float:
    """Scales marker size with probability so high-risk cells read first."""
    return 4 + (probability * 8)


def _pill(level: str) -> str:
    colour, bg = SEVERITY_STYLE.get(level, (COLORS["text_muted"], COLORS["bg"]))
    return f'<span class="pill" style="color:{colour};background:{bg}"><span class="dot"></span>{level}</span>'


def _focus_bounds(focus) -> list:
    """South-west and north-east corners of the view around a searched town:
    the chosen radius, but never less than FOCUS_MAP_VIEW_KM, so the heat
    of the neighbouring areas stays visible."""
    view_km = max(focus.radius_km, FOCUS_MAP_VIEW_KM)
    dlat = view_km / 111.32
    dlon = view_km / (111.32 * max(math.cos(math.radians(focus.lat)), 0.1))
    return [[focus.lat - dlat, focus.lon - dlon], [focus.lat + dlat, focus.lon + dlon]]


def _add_town_pin(m: folium.Map, focus) -> None:
    """A pin on the searched town, so it is easy to find on the heat map."""
    folium.Marker(
        location=[focus.lat, focus.lon], tooltip=focus.town,
        icon=folium.Icon(color="darkred", icon="map-marker", prefix="fa"),
    ).add_to(m)


def _popup_fields(row) -> dict:
    """
    The text pieces of one area's popup and tooltip, formatted in Python.
    The browser assembles them into the popup (_POPUP_JS) only when it is
    opened, instead of the page carrying 2,861 ready-made popups.
    """
    colour, _ = SEVERITY_STYLE.get(row.risk_level, (COLORS["low"], ""))
    place = describe_location(row.lat_round, row.lon_round)
    return {
        "c": colour,
        "r": _marker_radius(float(row.fire_probability)),
        "lv": str(row.risk_level),
        "pl": place[0].upper() + place[1:],
        "pr": f"{row.fire_probability:.1%}",
        "why": short_reason(row._asdict()),
        "ff": f"{row.ffdi:.1f} ({ffdi_band(row.ffdi)})",
        "kb": f"{row.kbdi:.1f}",
        "dfc": f"{row.drought_factor:.1f}",
        "t": f"{row.temperature_2m:.1f}",
        "rh": f"{row.relative_humidity_2m:.0f}",
        "w": f"{row.wind_speed_10m:.0f}",
        "cell": format_coords(row.lat_round, row.lon_round),
        "sp": 1 if bool(getattr(row, "kbdi_spinup_flag", False)) else 0,
        "tip": f"{row.risk_level} {row.fire_probability:.0%} — {place}",
    }


def _cell_features(predictions: pd.DataFrame) -> dict:
    """Every area as one GeoJSON point, so the map holds a single layer."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row.lon_round), float(row.lat_round)]},
                "properties": _popup_fields(row),
            }
            for row in predictions.itertuples(index=False)
        ],
    }


# The popup an area shows when clicked, from its _popup_fields(). Same markup
# and wording as before the map was made a single layer.
_POPUP_JS = """function (p) {
    return "<div style='font-family:Inter,sans-serif;font-size:12px;line-height:1.55'>"
        + "<b style='color:" + p.c + ";font-size:13px'>" + p.lv + " risk</b> "
        + "<span style='color:#57534E'>" + p.pl + "</span><br>"
        + "Fire probability: <b>" + p.pr + "</b><br>"
        + "<b>Why:</b> " + p.why + "<br>"
        + "FFDI: " + p.ff + "<br>"
        + "KBDI: " + p.kb + " &middot; Drought factor: " + p.dfc + "<br>"
        + "Temp: " + p.t + "&deg;C &middot; "
        + "RH: " + p.rh + "% &middot; "
        + "Wind: " + p.w + " km/h<br>"
        + "<span style='color:#8A847B'>Cell " + p.cell + "</span>"
        + (p.sp ? "<br><i>Low confidence: cell has under 30 days of history.</i>" : "")
        + "</div>";
}"""

# Styles each area's circle and attaches its hover label and (lazily built) popup.
_ON_EACH_CELL_JS = """function (feature, layer) {
    var p = feature.properties;
    var popup = """ + _POPUP_JS + """;
    layer.setStyle({radius: p.r, color: "#FFFFFF", weight: 1, fill: true,
                    fillColor: p.c, fillOpacity: 0.85});
    layer.bindTooltip("<div>" + p.tip + "</div>", {sticky: true});
    layer.bindPopup(function () { return popup(p); }, {maxWidth: 280});
}"""


def _build_map(predictions: pd.DataFrame, region: str, focus=None) -> folium.Map:
    center, zoom = get_region_view(region)
    if focus is not None:
        center = [focus.lat, focus.lon]
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        # Canvas draws thousands of circles far faster than one SVG shape each.
        prefer_canvas=True,
        tiles=(
            "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
            "?key=cb1_2s99_1_503a3a7bf96ca39604de30b4"
        ),
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, '
            '&copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        control_scale=True,
    )
    # A fixed zoom can't suit every state (Tasmania vs Western Australia), so
    # frame the state's own bounding box instead.
    if focus is not None:
        m.fit_bounds(_focus_bounds(focus), padding=(12, 12))
        _add_town_pin(m, focus)
    elif not is_national(region) and region in STATE_BOUNDS:
        lat_min, lat_max, lon_min, lon_max = STATE_BOUNDS[region]
        m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], padding=(12, 12))

    if predictions.empty:
        return m

    HeatMap(
        [
            [row.lat_round, row.lon_round, float(row.fire_probability)]
            for row in predictions.itertuples(index=False)
        ],
        radius=18,
        blur=22,
        max_zoom=1,
        gradient=HEAT_GRADIENT,
    ).add_to(m)

    folium.GeoJson(
        _cell_features(predictions),
        name="Risk by area",
        marker=folium.CircleMarker(radius=4, color="#FFFFFF", weight=1, fill=True, fill_opacity=0.85),
        on_each_feature=JsCode(_ON_EACH_CELL_JS),
        zoom_on_click=False,
    ).add_to(m)
    return m


@st.cache_data(max_entries=24, show_spinner=False)
def map_html(predictions: pd.DataFrame, region: str, focus=None) -> str:
    """
    The finished map as an HTML page, cached by its inputs, so going back to
    a region, town or day seen recently doesn't rebuild it. The map is shown
    read-only (nothing is sent back from it), so a plain HTML component does
    the same job as st_folium without re-rendering the map on every rerun.
    """
    return _build_map(predictions, region, focus).get_root().render()


def _legend() -> str:
    items = "".join(
        f'<span><span class="sw" style="background:{COLORS[level.lower()]}"></span>{level}</span>'
        for level in RISK_ORDER
    )
    return (
        f'<div class="legend"><b style="color:{COLORS["text"]}">{term("risk_level")}</b>{items}'
        '<span class="note">Bigger circle = higher probability · click a cell for details</span></div>'
    )


def render_map_with_insights(predictions: pd.DataFrame, region: str, summary: dict,
                             focus=None, focus_row=None, map_predictions=None):
    """
    Renders the map card and the region threat summary beside it.

    Args:
        predictions: prediction rows for this region, already narrowed to the
                     forecast day chosen in the day strip. One row per cell --
                     passing several days would stack markers on the same
                     coordinates.
        region: selected region name.
        summary: output of core.summaries.summarise_region().
        focus: core.focus.Focus when a town is searched -- the map centres on it.
        focus_row: the town's forecast row for the day, explained in the panel.
        map_predictions: rows to draw on the map, if different from `predictions`
                         (town focus draws every area, like the region view,
                         while the panel describes only the area around the town).
    """
    if focus is None:
        subtitle = f"Each circle is a 0.5° {term('grid_cell')}, coloured by its predicted risk level."
    else:
        subtitle = (f"Each circle is a 0.5° {term('grid_cell')}, coloured by its predicted risk "
                    f"level &middot; centred on {focus.town}.")
    section_header("Where", "Risk map", subtitle)
    with st.container(key="maprow"):
        map_col, right_col = st.columns([3, 1], gap="medium")

    with map_col:
        with st.container(key="card-map"):
            components.html(
                map_html(predictions if map_predictions is None else map_predictions, region, focus),
                height=MAP_HEIGHT,
            )
            if predictions.empty:
                st.info(
                    "No predictions to map. Fetch live weather, then run a refresh "
                    "to generate predictions."
                )
            html(_legend())

    with right_col:
        with st.container(key="card-threat"):
            _render_threat_panel(summary, predictions, focus, focus_row)


def _why_block(predictions: pd.DataFrame, focus=None, focus_row=None) -> str:
    """Plain-English reasons for the highest-risk cell's rating -- or, in town
    focus, for the searched town's own forecast area."""
    if focus is not None and focus_row is not None:
        row = focus_row
        info = explain_cell(row)
        level = str(row["risk_level"])
        info["title"] = (f"Why is {focus.name} at {level} risk?" if level in ("High", "Extreme")
                         else f"What's affecting the risk in {focus.name}?")
        where = f"In {focus.name}'s forecast area"
    else:
        row = predictions.loc[predictions["fire_probability"].idxmax()]
        info = explain_cell(row)
        where = f"At the highest-risk cell, {describe_location(row['lat_round'], row['lon_round'])}"
    items = "".join(
        f'<div class="why-item" style="--c:{DRIVER_COLOURS[d.strength]}">'
        f"{icon(d.icon, 14)}<span>{d.text}</span></div>"
        for d in info["drivers"]
    )
    calm = (
        f'<div class="why-calm">Working against fire: {", ".join(info["calming"])}.</div>'
        if info["calming"] else ""
    )
    note = f'<div class="why-note">{info["note"]}</div>' if info["note"] else ""
    return (
        f'<div class="mix-label">{info["title"]}<span class="place">{where}</span></div>'
        f'<div class="why-list">{items}</div>{calm}{note}'
    )


def _render_threat_panel(summary: dict, predictions: Optional[pd.DataFrame] = None,
                         focus=None, focus_row=None):
    """Right-hand summary panel, driven by the region (or town-area) summary."""
    title = ("Region Threat Summary" if focus is None
             else f'Area Summary<span class="place">Within {focus.radius_km} km of {focus.town}</span>')
    if not summary.get("has_data"):
        html(
            f"""
            <div class="panel-title">{title}</div>
            <p class="empty-note">{icon("info", 14)} No prediction data for this region yet.</p>
            """
        )
        return

    lat, lon = summary["hottest_cell"]
    counts = summary["risk_counts"]
    total = sum(counts.values()) or 1

    mix = "".join(
        f'<span style="width:{counts[level] / total * 100:.2f}%;background:{COLORS[level.lower()]}"'
        f' title="{level}: {counts[level]}"></span>'
        for level in reversed(RISK_ORDER)
        if counts[level]
    )
    mix_key = "".join(
        f'<span><span class="sw" style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{COLORS[level.lower()]};margin-right:5px"></span>'
        f"{level} <b>{counts[level]}</b></span>"
        for level in reversed(RISK_ORDER)
    )

    spinup_note = ""
    if summary["spinup_count"]:
        spinup_note = (
            f'<div class="callout">{icon("info", 14)}<div><b>{summary["spinup_count"]}</b> '
            "cell(s) have under 30 days of history — their drought values are "
            "cold-start estimates.</div></div>"
        )

    has_cells = predictions is not None and not predictions.empty
    why = _why_block(predictions, focus, focus_row) if has_cells else ""

    html(
        f"""
        <div class="panel-title">{title}</div>
        <div style="margin-bottom:6px">{_pill(summary["max_risk_level"])}
          <span style="font-size:.78rem;color:{COLORS["text_muted"]};margin-left:6px">
          {summary["max_risk_level"]} risk across {summary["cell_count"]} monitored cell(s).</span></div>
        <div class="panel-row">Forecast day<b>{summary["forecast_date"]}</b></div>
        <div class="panel-row">Highest risk cell<b>{describe_location(lat, lon)}<span class="place">{lat}, {lon}</span></b></div>
        <div class="panel-row"><span>Peak {term("probability", "probability")}</span><b>{summary["max_probability"]:.1%}</b></div>
        <div class="panel-row"><span>Peak {term("ffdi")}</span><b>{summary["max_ffdi"]:.1f} <small>({ffdi_band(summary["max_ffdi"])})</small></b></div>
        <div class="panel-row"><span>Peak {term("kbdi")}</span><b>{summary["max_kbdi"]:.1f} <small>/ 203.2</small></b></div>
        <div class="mix-label">Cells by risk level</div>
        <div class="mix-bar">{mix}</div>
        <div class="mix-key">{mix_key}</div>
        {why}
        {spinup_note}
        """
    )
