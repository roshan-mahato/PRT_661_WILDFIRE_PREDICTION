"""
Tests for the HTML-card components: the insight row, the day strip tiles and
the map's threat panel. They render with st.markdown, so the tests capture
the HTML and check the figures in it.
"""

from datetime import timedelta

import pytest
from components import day_strip, insights, map_section
from conftest import TODAY, TOMORROW
from utils.regions import day_options, summarise_region


@pytest.fixture
def markdown(monkeypatch):
    """Records every st.markdown body, for any module that renders HTML."""
    bodies = []
    import streamlit as st

    monkeypatch.setattr(st, "markdown", lambda body, **_: bodies.append(body))
    return bodies


# ----------------------------------------------------------------------------
# Insight cards
# ----------------------------------------------------------------------------
def test_insights_row_shows_summary_figures(markdown, predictions):
    insights.render_insights_row(summarise_region(predictions, "Australia (National)"))

    html = "\n".join(markdown)
    assert "3 of 6 grid cells" in html
    assert "Extreme — peak fire probability 90%" in html
    assert "60.0 (Severe)" in html
    assert "25°C, 30% humidity, wind 20 km/h" in html


def test_insights_row_no_data_still_renders_four_cards(markdown, empty_predictions):
    insights.render_insights_row(summarise_region(empty_predictions, "Victoria"))

    assert len(markdown) == 4
    for title in ("Cells at Risk", "Risk Level", "Peak FFDI", "Conditions"):
        assert any(title in body for body in markdown)
    assert all("No prediction data available" in body for body in markdown)


# ----------------------------------------------------------------------------
# Day strip
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("offset", "label"),
    [(0, "Today"), (1, "Tomorrow")],
)
def test_day_label_relative_days(offset, label):
    assert day_strip._day_label(TODAY + timedelta(days=offset)) == label


def test_day_label_weekday_for_later_days():
    later = TODAY + timedelta(days=3)
    assert day_strip._day_label(later) == later.strftime("%a")


def test_tile_html_shows_day_figures(predictions):
    _, tomorrow = day_options(predictions)
    html = day_strip._tile_html(tomorrow, selected=False)

    assert TOMORROW.strftime("%d %b") in html
    assert "90%" in html
    assert "Extreme" in html
    assert "3 of 6 cells" in html
    assert "FFDI 60.0" in html


def test_tile_html_highlights_selected_day(predictions):
    today, _ = day_options(predictions)
    assert "2px solid" in day_strip._tile_html(today, selected=True)
    assert "1px solid" in day_strip._tile_html(today, selected=False)


def test_render_day_strip_without_days_returns_none():
    assert day_strip.render_day_strip([]) is None


# ----------------------------------------------------------------------------
# Map threat panel
# ----------------------------------------------------------------------------
def test_threat_panel_shows_region_summary(markdown, predictions):
    map_section._render_threat_panel(summarise_region(predictions, "Australia (National)"))

    (html,) = markdown
    assert str(TOMORROW) in html
    assert "-33.0, 147.0" in html
    assert "90.0%" in html
    assert "Extreme 2 &middot; High 1" in html
    assert "Extreme risk across" in html


def test_threat_panel_warns_about_spinup_cells(markdown, predictions):
    today = predictions[predictions["acq_date"].dt.date == TODAY]
    map_section._render_threat_panel(summarise_region(today, "Australia (National)"))
    assert "<b>1</b> cell(s) have under 30 days of history" in markdown[0]


def test_threat_panel_no_data(markdown, empty_predictions):
    map_section._render_threat_panel(summarise_region(empty_predictions, "Victoria"))
    assert "No prediction data for this region yet." in markdown[0]


def test_marker_radius_grows_with_probability():
    assert map_section._marker_radius(0.0) == 4
    assert map_section._marker_radius(1.0) == 12
    assert map_section._marker_radius(0.9) > map_section._marker_radius(0.1)
