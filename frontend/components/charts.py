"""
Plotly chart components for the wildfire dashboard.

Four charts, each answering a different question so nothing on the page is
said twice:

  1. render_fire_weather       — WHY does danger change? Fire weather per day (line)
  2. render_risk_outlook       — HOW WIDESPREAD is it each day? Share of region (stacked bar)
  3. render_places_affected    — WHO is affected today? Towns by risk level (donut)
  4. render_persistent_hotspots — WHERE and WHEN does it last? Day-by-day timeline (bars)

They speak in everyday units -- named danger bands, % of the region, towns,
days -- rather than the model's grid cells and probabilities, and read
colours from utils.theme.COLORS so they match the rest of the app.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.glossary import term
from utils.places import describe_location, towns_at_risk
from utils.plain import (
    FFDI_BANDS,
    area_comparison,
    cell_area_km2,
    ffdi_band_name,
    format_area,
    one_in,
    weather_words,
)
from utils.regions import RISK_ORDER, persistent_hotspots
from utils.styles import format_coords, html, section_header
from utils.theme import COLORS, FONT_BODY, SEVERITY_STYLE

PLOT_FONT = dict(family=FONT_BODY, color=COLORS["text_muted"], size=12)
GRID_COLOR = COLORS["border"]
CHART_CONFIG = {"displayModeBar": False}
HOTSPOT_ROWS = 8      # chart 4: places shown
TOP_PLACES = 5        # chart 3: places named beside the donut

# Chart 1 background: one warm tint per FFDI band, light to dark. A single
# hue on purpose -- the model's risk colours (green/gold/orange/red) mean
# something else, and reusing them here would blur the two scales.
BAND_TINTS = ["#FAF7F2", "#FBEBD8", "#F8D9BC", "#F3BFA0", "#EC9F82", "#E07F66"]
LOWER_RISK_TINT = "#E9E5DE"  # chart 4: Low / Moderate days


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


def _legend_top(**extra) -> dict:
    return dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11), **extra)


def _title(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chart-sub'>{subtitle}</div>", unsafe_allow_html=True)


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _day_dates(predictions: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(predictions["acq_date"]).dt.normalize()


# ----------------------------------------------------------------------------
# 1. FIRE WEATHER
# ----------------------------------------------------------------------------
def daily_fire_weather(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Per forecast day: the typical (regional mean) FFDI and weather, and the
    worst FFDI with the place and weather where it occurs.
    """
    rows = []
    for day, group in predictions.groupby(_day_dates(predictions), sort=True):
        worst = group.loc[group["ffdi"].idxmax()]
        rows.append({
            "date": day,
            "typical_ffdi": float(group["ffdi"].mean()),
            "worst_ffdi": float(worst["ffdi"]),
            "worst_lat": float(worst["lat_round"]),
            "worst_lon": float(worst["lon_round"]),
            "worst_weather": weather_words(worst),
            "typical_weather": weather_words(group[
                ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]].mean()),
        })
    return pd.DataFrame(rows)


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


def render_fire_weather(predictions: pd.DataFrame) -> None:
    """
    The weather behind the forecast. Every other chart shows the model's
    prediction; this shows the fire danger index (FFDI) that the weather
    produces, on bands named in words, so a change in risk from one day to
    the next has a visible reason.
    """
    _title(
        "Fire Weather This Week",
        f"The weather behind the forecast: fire danger ({term('ffdi')}) at the worst "
        "place and in a typical area &middot; shaded by danger band",
    )
    if predictions is None or predictions.empty:
        st.info("No forecast data available for this region.")
        return

    weather = daily_fire_weather(predictions)
    days = [d.strftime("%a %d %b") for d in weather["date"]]
    marker_size = 8 if len(days) > 1 else 14

    worst_places = [
        _capitalise(describe_location(lat, lon))
        for lat, lon in zip(weather["worst_lat"], weather["worst_lon"])
    ]
    fig = go.Figure()
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

    peak = float(weather["worst_ffdi"].max())
    # Show bands up to the first boundary comfortably above the worst value.
    y_max = next((b for b in (25, 50, 75, 100, 150) if b > peak * 1.15), peak * 1.15)
    shapes, band_labels = _band_shapes(y_max)
    fig.update_layout(**_base_layout(
        showlegend=True, legend=_legend_top(),
        shapes=shapes, annotations=band_labels,
        margin=dict(l=44, r=86, t=8, b=36),
        yaxis=dict(range=[0, y_max], showgrid=False, zeroline=False,
                   title="Fire danger index"),
        xaxis=dict(showgrid=False),
    ))
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ----------------------------------------------------------------------------
# 2. RISK OUTLOOK BY DAY
# ----------------------------------------------------------------------------
def daily_risk_shares(predictions: pd.DataFrame) -> tuple:
    """
    Share of the region's land area at each risk level, per day (rows: days,
    columns: RISK_ORDER, values 0-1), plus the matching areas in km^2 and
    the number of areas (grid cells). Area-weighted, because a cell near
    Darwin covers ~30% more ground than one in Tasmania.
    """
    data = predictions.assign(
        day=_day_dates(predictions),
        area=predictions["lat_round"].map(cell_area_km2),
        level=predictions["risk_level"].astype(str),
    )
    area = data.pivot_table(index="day", columns="level", values="area", aggfunc="sum",
                            fill_value=0).reindex(columns=RISK_ORDER, fill_value=0).sort_index()
    counts = pd.crosstab(data["day"], data["level"]).reindex(
        columns=RISK_ORDER, fill_value=0).sort_index()
    shares = area.div(area.sum(axis=1), axis=0)
    return shares, area, counts


def render_risk_outlook(predictions: pd.DataFrame) -> None:
    """
    How much of the region is at each risk level, day by day -- the only
    chart about how widespread the danger is. Shown as a share of the
    region's land, with a "1 in N" label for High or Extreme.
    """
    _title(
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
    fig.update_layout(**_base_layout(
        barmode="stack", bargap=0.35, showlegend=True, legend=_legend_top(),
        margin=dict(l=44, r=16, t=8, b=36),
        yaxis=dict(range=[0, 128], tickvals=[0, 25, 50, 75, 100], ticksuffix="%",
                   gridcolor=GRID_COLOR, zeroline=False, title="Share of region"),
        xaxis=dict(showgrid=False),
    ))
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ----------------------------------------------------------------------------
# 3. PLACES AFFECTED
# ----------------------------------------------------------------------------
def _names_preview(names: list, limit: int = 6) -> str:
    if not names:
        return "none"
    shown = ", ".join(n.split(",")[0] for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def render_places_affected(day_predictions: pd.DataFrame) -> None:
    """
    Towns by risk level on the selected day -- the people-and-places view.
    Each listed town takes the forecast for the grid area it sits in.
    """
    day_label = (
        _day_dates(day_predictions).max().strftime("%a %d %b")
        if day_predictions is not None and not day_predictions.empty else "selected day"
    )
    _title("Places Affected", f"Towns by risk level &middot; {day_label}")

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
        customdata=[_names_preview(by_level[lvl]) for lvl in RISK_ORDER],
        textinfo="none", sort=False, direction="clockwise",
        hovertemplate="<b>%{label}</b>: %{value} towns<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
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
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
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


# ----------------------------------------------------------------------------
# 4. WHERE AND WHEN THE DANGER LASTS
# ----------------------------------------------------------------------------
def _hotspot_labels(hotspots: pd.DataFrame) -> list:
    """Place names for the y axis. Two cells can share a description ("near
    Kingaroy, QLD"), and Plotly merges identical category labels into one
    row, so duplicates get their coordinates appended."""
    names = [
        _capitalise(describe_location(r.lat_round, r.lon_round))
        for r in hotspots.itertuples(index=False)
    ]
    counts = pd.Series(names).value_counts()
    return [
        f"{n} ({format_coords(r.lat_round, r.lon_round)})" if counts[n] > 1 else n
        for n, r in zip(names, hotspots.itertuples(index=False))
    ]


def render_persistent_hotspots(predictions: pd.DataFrame) -> None:
    """
    A day-by-day timeline for the places that stay at High or Extreme risk
    the longest: one row per place, one slot per day, filled on its
    dangerous days. The only view that combines where AND when.
    """
    days_all = sorted(_day_dates(predictions).unique()) if predictions is not None and not predictions.empty else []
    all_hotspots = persistent_hotspots(predictions, top=None)
    hotspots = all_hotspots.head(HOTSPOT_ROWS)

    if hotspots.empty:
        _title("Where the Danger Lasts", "Places at High or Extreme risk, day by day")
        st.info(
            "No place reaches High or Extreme risk in this forecast."
            if days_all else "No forecast data available for this region."
        )
        return

    n_days = len(days_all)
    every_day = int((all_hotspots["days_elevated"] == n_days).sum())
    if n_days == 1:
        headline = f"<b>{len(all_hotspots)}</b> places are at High or Extreme risk on this forecast day."
    elif every_day:
        noun = "place is" if every_day == 1 else "places are"
        headline = f"<b>{every_day}</b> {noun} at High or Extreme risk on all {n_days} days."
    else:
        headline = (f"<b>{len(all_hotspots)}</b> places reach High or Extreme risk at least once; "
                    f"none stays there all {n_days} days.")
    _title("Where the Danger Lasts",
           f"{headline} The {len(hotspots)} most persistent, day by day.")

    labels = _hotspot_labels(hotspots)
    label_of = {(r.lat_round, r.lon_round): lab
                for r, lab in zip(hotspots.itertuples(index=False), labels)}
    cells = predictions.assign(day=_day_dates(predictions))
    cells = cells[[k in label_of for k in zip(cells["lat_round"], cells["lon_round"])]]
    day_index = {d: i for i, d in enumerate(days_all)}

    groups = (("Extreme", ["Extreme"], COLORS["extreme"]),
              ("High", ["High"], COLORS["high"]),
              ("Lower risk", ["Moderate", "Low"], LOWER_RISK_TINT))
    fig = go.Figure()
    for name, levels, colour in groups:
        part = cells[cells["risk_level"].astype(str).isin(levels)]
        fig.add_trace(go.Bar(
            name=name, orientation="h",
            y=[label_of[k] for k in zip(part["lat_round"], part["lon_round"])],
            x=[0.86] * len(part),
            base=[day_index[d] + 0.07 for d in part["day"]],
            marker=dict(color=colour, line=dict(width=0)),
            customdata=list(zip(part["day"].dt.strftime("%a %d %b"),
                                part["risk_level"].astype(str),
                                (part["fire_probability"] * 100).round(0))),
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]}: %{customdata[1]}"
                           " (%{customdata[2]:.0f}% chance)<extra></extra>"),
        ))
    fig.add_trace(go.Scatter(
        x=[n_days + 0.1] * len(hotspots), y=labels, mode="text",
        text=[f"{d} of {n_days}" for d in hotspots["days_elevated"]],
        textposition="middle right", textfont=dict(color=COLORS["text"], size=11),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_traces(cliponaxis=False)
    fig.update_layout(**_base_layout(
        height=max(170, 30 * len(hotspots) + 80),
        margin=dict(l=8, r=16, t=28, b=28),
        barmode="overlay", showlegend=True, legend=_legend_top(traceorder="normal"),
        xaxis=dict(range=[0, n_days + 1.1], showgrid=False, zeroline=False,
                   tickvals=[i + 0.5 for i in range(n_days)],
                   ticktext=[pd.Timestamp(d).strftime("%a<br>%d") for d in days_all]),
        yaxis=dict(autorange="reversed", automargin=True, categoryorder="array",
                   categoryarray=labels),
        bargap=0.3,
    ))
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# ----------------------------------------------------------------------------
# LAYOUT: put all four in a 2x2 grid
# ----------------------------------------------------------------------------
def render_insights_section(predictions: pd.DataFrame, day_predictions: pd.DataFrame) -> None:
    """
    Renders the full 'Insights' section: a 2x2 grid of charts.

    Args:
        predictions: the region's prediction rows across every forecast day.
        day_predictions: the same rows narrowed to the day selected in the
                         day strip (chart 3 describes that day).
    """
    section_header(
        "Insights",
        "How the forecast develops",
        "The weather behind the risk, how widespread it is, and which places are affected.",
    )

    row1_col1, row1_col2 = st.columns(2, gap="medium")
    with row1_col1:
        with st.container(key="card-weather"):
            render_fire_weather(predictions)
    with row1_col2:
        with st.container(key="card-outlook"):
            render_risk_outlook(predictions)

    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        with st.container(key="card-places"):
            render_places_affected(day_predictions)
    with row2_col2:
        with st.container(key="card-hotspots"):
            render_persistent_hotspots(predictions)
