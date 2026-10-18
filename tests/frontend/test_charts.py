"""
Tests for components.charts.

Streamlit output is intercepted: st.plotly_chart hands us the Plotly figure,
so the tests can check the numbers each chart actually plots, and
st.info / st.markdown are recorded to check the empty-state messages.
"""

import pandas as pd
import pytest
from components import charts
from conftest import TODAY, TOMORROW
from utils.places import describe_location, towns_at_risk
from utils.plain import (
    area_comparison,
    cell_area_km2,
    ffdi_band_name,
    format_area,
    one_in,
    weather_words,
)
from utils.regions import persistent_hotspots


@pytest.fixture
def st_calls(monkeypatch):
    """Captures what the charts send to Streamlit."""
    calls = {"figures": [], "info": [], "markdown": []}
    monkeypatch.setattr(charts.st, "plotly_chart", lambda fig, **_: calls["figures"].append(fig))
    monkeypatch.setattr(charts.st, "info", lambda msg, **_: calls["info"].append(msg))
    monkeypatch.setattr(
        charts.st, "markdown", lambda body, **_: calls["markdown"].append(body)
    )
    return calls


def _tomorrow(predictions):
    return predictions[predictions["acq_date"].dt.date == TOMORROW].reset_index(drop=True)


def _place(lat, lon):
    text = describe_location(lat, lon)
    return text[0].upper() + text[1:]


# ----------------------------------------------------------------------------
# Plain-language helpers
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("share", "text"),
    [(0.0, "none"), (0.128, "1 in 8"), (0.17, "1 in 6"), (0.45, "1 in 2"), (0.6, "6 in 10"), (1.0, "10 in 10")],
)
def test_one_in(share, text):
    assert one_in(share) == text


def test_cell_area_shrinks_away_from_the_equator():
    assert 2_900 < cell_area_km2(-12.5) < 3_100  # Top End
    assert 2_200 < cell_area_km2(-42.0) < 2_400  # Tasmania


@pytest.mark.parametrize(
    ("km2", "text"),
    [(0, "0 km²"), (2_812, "2,800 km²"), (337_000, "340,000 km²"), (1_034_000, "1.0 million km²")],
)
def test_format_area(km2, text):
    assert format_area(km2) == text


@pytest.mark.parametrize(
    ("km2", "text"),
    [
        (1_000, ""),
        (230_000, "about the size of Victoria"),
        (1_000_000, "about the size of South Australia"),
        (900_000, "about the size of South Australia"),  # within 10% of SA
        (883_000, "larger than New South Wales"),  # >10% above NSW, >10% below SA
        (3_000_000, "larger than Western Australia"),
    ],
)
def test_area_comparison(km2, text):
    assert area_comparison(km2) == text


@pytest.mark.parametrize(
    ("ffdi", "band"),
    [(0, "Low–Moderate"), (12, "High"), (30, "Very High"), (60, "Severe"), (80, "Extreme"), (120, "Catastrophic")],
)
def test_ffdi_band_name(ffdi, band):
    assert ffdi_band_name(ffdi) == band


def test_weather_words():
    row = {"temperature_2m": 31, "relative_humidity_2m": 14, "wind_speed_10m": 25}
    assert weather_words(row) == "very hot (31°C), very dry air (14%), fresh wind (25 km/h)"


# ----------------------------------------------------------------------------
# 1. Fire weather
# ----------------------------------------------------------------------------
def test_fire_weather_plots_worst_and_typical_ffdi(st_calls, predictions):
    charts.render_fire_weather(predictions)

    (fig,) = st_calls["figures"]
    worst, typical = fig.data
    assert (worst.name, typical.name) == ("Worst place", "Typical area")
    assert list(worst.y) == pytest.approx([30.0, 60.0])
    assert list(typical.y) == pytest.approx([65 / 6, 187 / 6])


def test_fire_weather_hover_names_band_place_and_weather(st_calls, predictions):
    charts.render_fire_weather(predictions)

    worst, typical = st_calls["figures"][0].data
    band, place, words = worst.customdata[1]
    assert band == "Severe"
    assert place == _place(-33.0, 147.0)
    assert words == "hot (25°C), dry air (30%), breezy (20 km/h)"
    assert typical.customdata[0][0] == "Low–Moderate"


def test_fire_weather_shades_named_bands_up_past_the_peak(st_calls, predictions):
    charts.render_fire_weather(predictions)

    fig = st_calls["figures"][0]
    # Peak 60 -> axis to 75: Low–Moderate, High, Very High and Severe bands.
    assert fig.layout.yaxis.range == (0, 75)
    assert [s.y0 for s in fig.layout.shapes] == [0, 12, 25, 50]
    assert fig.layout.shapes[-1].y1 == 75
    labels = fig.layout.annotations
    assert [a.text for a in labels] == ["Low–Moderate", "High", "Very High", "Severe"]
    # Band names sit in the right margin, outside the plot area.
    assert all(a.x > 1 and a.xanchor == "left" for a in labels)


def test_fire_weather_empty(st_calls, empty_predictions):
    charts.render_fire_weather(empty_predictions)
    assert st_calls["figures"] == []
    assert st_calls["info"]


# ----------------------------------------------------------------------------
# 2. How widespread
# ----------------------------------------------------------------------------
def test_risk_shares_are_area_weighted():
    """A cell near the equator covers more ground than one far south, so it
    must count for more of the region than a plain cell count would give."""
    df = pd.DataFrame([
        {"lat_round": -10.0, "lon_round": 130.0, "acq_date": TODAY, "risk_level": "Extreme", "fire_probability": 0.9},
        {"lat_round": -43.0, "lon_round": 146.0, "acq_date": TODAY, "risk_level": "Low", "fire_probability": 0.1},
    ])
    shares, area, counts = charts.daily_risk_shares(df)
    assert shares.loc[pd.Timestamp(TODAY), "Extreme"] > 0.5
    assert shares.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert list(counts.iloc[0]) == [1, 0, 0, 1]
    assert area.loc[pd.Timestamp(TODAY), "Extreme"] == pytest.approx(cell_area_km2(-10.0))


def test_outlook_stacks_percent_of_region(st_calls, predictions):
    charts.render_risk_outlook(predictions)

    (fig,) = st_calls["figures"]
    bars = [t for t in fig.data if t.type == "bar"]
    assert [b.name for b in bars] == ["Extreme", "High", "Moderate", "Low"]
    totals = [sum(b.y[i] for b in bars) for i in range(2)]
    assert totals == pytest.approx([100, 100])
    assert fig.layout.yaxis.ticksuffix == "%"
    assert fig.layout.barmode == "stack"


def test_outlook_labels_share_and_one_in_n(st_calls, predictions):
    charts.render_risk_outlook(predictions)

    shares, _, _ = charts.daily_risk_shares(predictions)
    elevated = shares["High"] + shares["Extreme"]
    (labels,) = [t for t in st_calls["figures"][0].data if t.type == "scatter"]
    assert list(labels.text) == [f"<b>{s:.0%}</b><br>{one_in(s)}" for s in elevated]


def test_outlook_hover_gives_area_comparison_and_count(st_calls, predictions):
    charts.render_risk_outlook(predictions)

    extreme = st_calls["figures"][0].data[0]
    area_text, comparison, count = extreme.customdata[1]  # tomorrow
    _, area, _ = charts.daily_risk_shares(predictions)
    tomorrow_extreme = area["Extreme"].iloc[1]
    assert area_text == format_area(tomorrow_extreme)
    assert comparison == f" · {area_comparison(tomorrow_extreme)}"
    assert count == 2


def test_outlook_empty(st_calls, empty_predictions):
    charts.render_risk_outlook(empty_predictions)
    assert st_calls["figures"] == []
    assert st_calls["info"]


# ----------------------------------------------------------------------------
# 3. Places affected
# ----------------------------------------------------------------------------
def test_towns_take_the_forecast_of_their_grid_area(predictions):
    towns = towns_at_risk(_tomorrow(predictions))
    assert dict(zip(towns["town"], towns["risk_level"])) == {
        "Condobolin, NSW": "Extreme",
        "Albury, NSW": "Extreme",
        "Wodonga, VIC": "Extreme",
        "Canberra, ACT": "Moderate",
    }
    assert towns["fire_probability"].is_monotonic_decreasing


def test_places_donut_counts_towns_per_level(st_calls, predictions):
    charts.render_places_affected(_tomorrow(predictions))

    (fig,) = st_calls["figures"]
    pie = fig.data[0]
    assert list(pie.labels) == ["Low", "Moderate", "High", "Extreme"]
    assert list(pie.values) == [0, 1, 0, 3]
    assert pie.customdata[3] == "Condobolin, Albury, Wodonga"
    assert "<b>3</b>" in fig.layout.annotations[0].text
    assert "towns at High" in fig.layout.annotations[0].text


def test_places_side_list_names_the_riskiest_towns(st_calls, predictions):
    charts.render_places_affected(_tomorrow(predictions))

    side = st_calls["markdown"][-1]
    assert side.index("Condobolin, NSW") < side.index("Canberra, ACT")
    assert "4 listed towns in this view" in side


def test_places_names_preview_truncates():
    names = [f"Town{i}, NSW" for i in range(9)]
    assert charts._names_preview(names) == "Town0, Town1, Town2, Town3, Town4, Town5 and 3 more"
    assert charts._names_preview([]) == "none"


def test_places_no_towns(st_calls, predictions):
    offshore = predictions[predictions["lat_round"] == -20.0]
    charts.render_places_affected(offshore)
    assert st_calls["figures"] == []
    assert st_calls["info"] == ["No listed towns fall inside the forecast areas for this region."]


# ----------------------------------------------------------------------------
# 4. Where and when the danger lasts
# ----------------------------------------------------------------------------
def test_persistent_hotspots_ranks_by_days_then_probability(predictions):
    hotspots = persistent_hotspots(predictions)

    cells = list(zip(hotspots["lat_round"], hotspots["lon_round"]))
    # (-36, 147) is High today and Extreme tomorrow: 2 days. The other two
    # are elevated on one day only, ordered by Extreme days then probability.
    assert cells == [(-36.0, 147.0), (-33.0, 147.0), (-37.5, 145.0)]
    assert list(hotspots["days_elevated"]) == [2, 1, 1]
    assert list(hotspots["worst_level"]) == ["Extreme", "Extreme", "High"]
    assert set(hotspots["days_total"]) == {2}
    assert hotspots["peak_probability"].iloc[0] == pytest.approx(0.8)


def test_persistent_hotspots_ignores_low_and_moderate_cells(predictions):
    calm = predictions[predictions["risk_level"].isin(["Low", "Moderate"])]
    assert persistent_hotspots(calm).empty


def test_persistent_hotspots_respects_top(predictions):
    assert len(persistent_hotspots(predictions, top=2)) == 2


def test_persistent_hotspots_ranks_extreme_days_before_probability():
    """Equal elevated days: more Extreme days wins even with a lower peak."""
    rows = [
        # Cell A: one Extreme day with a very high peak, then High.
        {"lat_round": -30.0, "lon_round": 140.0, "acq_date": TODAY, "risk_level": "Extreme", "fire_probability": 0.90},
        {"lat_round": -30.0, "lon_round": 140.0, "acq_date": TOMORROW, "risk_level": "High", "fire_probability": 0.40},
        # Cell B: Extreme on both days, but a lower peak than A.
        {"lat_round": -31.0, "lon_round": 141.0, "acq_date": TODAY, "risk_level": "Extreme", "fire_probability": 0.65},
        {"lat_round": -31.0, "lon_round": 141.0, "acq_date": TOMORROW, "risk_level": "Extreme", "fire_probability": 0.66},
    ]
    ranked = persistent_hotspots(pd.DataFrame(rows))
    assert list(ranked["lat_round"]) == [-31.0, -30.0]
    assert list(ranked["days_extreme"]) == [2, 1]
    assert list(ranked["days_high"]) == [0, 1]


def test_timeline_places_each_day_in_its_slot(st_calls, predictions):
    charts.render_persistent_hotspots(predictions)

    (fig,) = st_calls["figures"]
    extreme, high, lower, totals = fig.data
    assert [extreme.name, high.name, lower.name] == ["Extreme", "High", "Lower risk"]
    assert fig.layout.barmode == "overlay"

    first = _place(-36.0, 147.0)
    # (-36,147): High today (slot 0), Extreme tomorrow (slot 1).
    assert (first, pytest.approx(1.07)) in list(zip(extreme.y, extreme.base))
    assert (first, pytest.approx(0.07)) in list(zip(high.y, high.base))
    # (-33,147) was Moderate today -> a grey "lower risk" slot.
    assert (_place(-33.0, 147.0), pytest.approx(0.07)) in list(zip(lower.y, lower.base))
    assert list(totals.text) == ["2 of 2", "1 of 2", "1 of 2"]
    assert list(fig.layout.yaxis.categoryarray)[0] == first
    assert list(fig.layout.xaxis.ticktext) == [
        pd.Timestamp(TODAY).strftime("%a<br>%d"), pd.Timestamp(TOMORROW).strftime("%a<br>%d"),
    ]


def test_timeline_hover_gives_day_level_and_chance(st_calls, predictions):
    charts.render_persistent_hotspots(predictions)
    extreme = st_calls["figures"][0].data[0]
    day, level, chance = extreme.customdata[0]
    assert level == "Extreme"
    assert day == pd.Timestamp(TOMORROW).strftime("%a %d %b")
    assert chance in (80, 90)


def test_timeline_headline_counts_places_elevated_every_day(st_calls, predictions):
    charts.render_persistent_hotspots(predictions)
    subtitle = st_calls["markdown"][1]
    assert "<b>1</b> place is at High or Extreme risk on all 2 days." in subtitle
    assert "The 3 most persistent, day by day." in subtitle


def test_timeline_rows_stay_unique(st_calls, predictions):
    """Two cells near the same town must not merge into one row."""
    twin = predictions[predictions["lat_round"] == -33.0].copy()
    twin["lon_round"] = 147.1  # same nearest town, different cell
    charts.render_persistent_hotspots(pd.concat([predictions, twin]))
    rows = list(st_calls["figures"][0].layout.yaxis.categoryarray)
    assert len(rows) == len(set(rows))


def test_timeline_no_elevated_cells(st_calls, predictions):
    calm = predictions[predictions["risk_level"] == "Low"]
    charts.render_persistent_hotspots(calm)
    assert st_calls["figures"] == []
    assert st_calls["info"] == ["No place reaches High or Extreme risk in this forecast."]


def test_timeline_no_data(st_calls, empty_predictions):
    charts.render_persistent_hotspots(empty_predictions)
    assert st_calls["info"] == ["No forecast data available for this region."]
