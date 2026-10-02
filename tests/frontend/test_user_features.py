"""
Tests for the user-facing helpers: place names, plain-English risk reasons,
best/worst day of the week, and glossary tooltips.
"""

from datetime import timedelta

import pandas as pd
import pytest
from components import day_strip, map_section
from conftest import TODAY, TOMORROW
from utils import explain, glossary, places
from utils.regions import STATE_BOUNDS, day_options, summarise_region, week_highlights

STATE_ABBR = {
    "New South Wales": "NSW", "Victoria": "VIC", "Queensland": "QLD",
    "Western Australia": "WA", "South Australia": "SA", "Tasmania": "TAS",
    "Northern Territory": "NT", "Australian Capital Territory": "ACT",
}


# ----------------------------------------------------------------------------
# Place names
# ----------------------------------------------------------------------------
def test_close_cell_reads_as_near_a_town():
    assert places.describe_location(-23.70, 133.88) == "near Alice Springs, NT"


def test_distant_cell_gives_distance_and_direction():
    # About 1 degree due north of Alice Springs.
    text = places.describe_location(-22.70, 133.88)
    assert text.endswith("N of Alice Springs, NT") or "Ti Tree" in text
    assert " km " in text


@pytest.mark.parametrize(
    ("dlat", "dlon", "direction"),
    [(1, 0, "N"), (-1, 0, "S"), (0, 1, "E"), (0, -1, "W"), (1, 1, "NE"), (-1, -1, "SW")],
)
def test_bearing(dlat, dlon, direction):
    assert places._bearing(-30.0, 140.0, -30.0 + dlat, 140.0 + dlon) == direction


def test_distances_are_rounded_to_5_km():
    text = places.describe_location(-25.0, 125.0)
    km = int(text.split(" km ")[0])
    assert km % 5 == 0


def test_every_town_lies_in_its_own_state():
    """Guards against a typo in a hand-entered coordinate (a sign flip or a
    swapped lat/lon would put a town in the wrong state or the ocean)."""
    by_abbr = {abbr: STATE_BOUNDS[name] for name, abbr in STATE_ABBR.items()}
    for name, state, lat, lon in places.TOWNS:
        lat_min, lat_max, lon_min, lon_max = by_abbr[state]
        margin = 2.0 if state == "VIC" else 0.3  # VIC box stops at -35.8; Mildura is further north
        assert lat_min - margin <= lat <= lat_max + margin, name
        assert lon_min - 0.3 <= lon <= lon_max + 0.3, name


def test_no_duplicate_towns():
    keys = [(name, state) for name, state, _, _ in places.TOWNS]
    assert len(keys) == len(set(keys))


# ----------------------------------------------------------------------------
# Plain-English reasons
# ----------------------------------------------------------------------------
def _row(**overrides):
    row = {
        "temperature_2m": 20.0, "relative_humidity_2m": 50.0, "wind_speed_10m": 10.0,
        "drought_factor": 4.0, "kbdi": 30.0, "ffdi": 5.0,
        "kbdi_spinup_flag": False, "risk_level": "Low",
    }
    row.update(overrides)
    return row


def test_fire_weather_drivers_strongest_first():
    row = _row(temperature_2m=33, relative_humidity_2m=12, wind_speed_10m=25,
               drought_factor=9.5, kbdi=160, ffdi=55, risk_level="Extreme")
    drivers = explain.fire_drivers(row)
    texts = [d.text for d in drivers]

    assert "Very hot (33°C average)" in texts
    assert "Very dry air (12% humidity)" in texts
    assert "Fresh wind (25 km/h average)" in texts
    assert "Fuel extremely dry (drought factor 9.5/10)" in texts
    assert "Severe fire danger (FFDI 55)" in texts
    assert [d.strength for d in drivers] == sorted((d.strength for d in drivers), reverse=True)


def test_explain_cell_limits_drivers_and_uses_high_risk_title():
    row = _row(temperature_2m=33, relative_humidity_2m=12, wind_speed_10m=40,
               drought_factor=9.5, kbdi=160, ffdi=55, risk_level="Extreme")
    info = explain.explain_cell(row)
    assert info["title"] == "Why is the risk high here?"
    assert len(info["drivers"]) == 4
    assert info["note"] == ""


def test_high_risk_with_mild_weather_says_so():
    """The model also weighs location and season; when the weather is
    unremarkable the explanation must not invent a weather cause."""
    info = explain.explain_cell(_row(risk_level="High"))
    assert "history of fires and the time of year" in info["note"]


def test_low_risk_explains_calming_conditions():
    info = explain.explain_cell(_row(relative_humidity_2m=85, temperature_2m=12,
                                     drought_factor=1.5, wind_speed_10m=5))
    assert info["title"] == "What's affecting the risk here?"
    assert "humid air (85%)" in info["calming"]
    assert "cool (12°C average)" in info["calming"]
    assert "No fire-weather warning signs" in info["note"]


def test_spinup_cells_get_a_confidence_note():
    info = explain.explain_cell(_row(kbdi_spinup_flag=True))
    assert "early estimates" in info["note"]


def test_short_reason():
    assert explain.short_reason(_row(relative_humidity_2m=12, temperature_2m=33)) == "Very hot, Very dry air"
    assert explain.short_reason(_row(relative_humidity_2m=85)).startswith("Mild: humid air")


def test_threat_panel_explains_the_highest_risk_cell(monkeypatch, predictions):
    import streamlit as st

    bodies = []
    monkeypatch.setattr(st, "markdown", lambda body, **_: bodies.append(body))
    tomorrow = predictions[predictions["acq_date"].dt.date == TOMORROW]
    map_section._render_threat_panel(summarise_region(tomorrow, "Australia (National)"), tomorrow)

    html = bodies[0]
    assert "Why is the risk high here?" in html
    assert places.describe_location(-33.0, 147.0) in html
    assert "Severe fire danger (FFDI 60)" in html


# ----------------------------------------------------------------------------
# Best / worst day
# ----------------------------------------------------------------------------
def test_week_highlights_picks_riskiest_and_calmest_day(predictions):
    highlights = week_highlights(day_options(predictions))
    assert highlights["worst"]["date"] == TOMORROW
    assert highlights["best"]["date"] == TODAY
    assert not highlights["flat"]


def test_week_highlights_needs_two_days(predictions):
    today = predictions[predictions["acq_date"].dt.date == TODAY]
    assert week_highlights(day_options(today)) is None
    assert day_strip.week_summary_html(day_options(today)) == ""


def test_week_highlights_flat_when_days_are_alike(predictions):
    today = predictions[predictions["acq_date"].dt.date == TODAY]
    copy = today.copy()
    copy["acq_date"] = copy["acq_date"] + timedelta(days=1)
    options = day_options(pd.concat([today, copy]))
    assert week_highlights(options)["flat"]
    assert "Risk stays about the same all week" in day_strip.week_summary_html(options)


def test_week_summary_names_the_days(predictions):
    text = day_strip.week_summary_html(day_options(predictions))
    assert "Riskiest day: <b>Tomorrow" in text
    assert "3 of 6 cells at risk" in text
    assert "Lowest risk: <b>Today" in text


# ----------------------------------------------------------------------------
# Glossary tooltips
# ----------------------------------------------------------------------------
def test_term_markup_is_accessible():
    html = glossary.term("ffdi")
    assert 'class="term"' in html
    assert 'tabindex="0"' in html  # reachable by keyboard and by tapping
    assert "data-tip=" in html and "aria-label=" in html
    assert html.endswith(">FFDI</span>")


def test_term_custom_label():
    assert glossary.term("grid_cell", "grid cells").endswith(">grid cells</span>")


def test_term_escapes_quotes_in_definitions(monkeypatch):
    """A quote in a definition must not end the data-tip attribute early."""
    monkeypatch.setitem(glossary.GLOSSARY, "demo", ("Demo", 'Says "hi" & <b>bye</b>'))
    html = glossary.term("demo")
    assert 'data-tip="Says &quot;hi&quot; &amp; &lt;b&gt;bye&lt;/b&gt;"' in html


def test_every_glossary_entry_has_a_label_and_definition():
    for key, (label, definition) in glossary.GLOSSARY.items():
        assert label and len(definition) > 40, key


def _option(day, at_risk, cells=2861, mean=0.08):
    return {"date": day, "peak_level": "Extreme", "peak_probability": 0.9,
            "mean_probability": mean, "at_risk_count": at_risk, "cell_count": cells,
            "max_ffdi": 20.0}


def test_week_is_not_flat_when_at_risk_counts_differ_by_a_third():
    """Regression: nationally ~11-15% of cells are at risk; 320 vs 420 cells is
    a real difference and must name a riskiest day, not say 'about the same'."""
    options = [_option(TODAY + timedelta(days=i), n) for i, n in enumerate([366, 320, 420, 378])]
    highlights = week_highlights(options)
    assert not highlights["flat"]
    assert highlights["worst"]["at_risk_count"] == 420
    assert highlights["best"]["at_risk_count"] == 320


def test_flat_week_message_shows_the_range():
    options = [_option(TODAY + timedelta(days=i), n) for i, n in enumerate([400, 410, 405])]
    text = day_strip.week_summary_html(options)
    assert "Risk stays about the same all week" in text
    assert "400–410 of 2861 cells at risk each day" in text
