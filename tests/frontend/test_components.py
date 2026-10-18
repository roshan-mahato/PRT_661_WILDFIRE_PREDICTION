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
    assert ">FFDI</span> 60.0" in html


def test_tile_html_highlights_selected_day(predictions):
    today, _ = day_options(predictions)
    assert 'class="day-tile selected"' in day_strip._tile_html(today, selected=True)
    assert 'class="day-tile"' in day_strip._tile_html(today, selected=False)


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
    assert "Extreme <b>2</b>" in html
    assert "High <b>1</b>" in html
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


# ----------------------------------------------------------------------------
# Status banner
# ----------------------------------------------------------------------------
from components import hero  # noqa: E402
from utils import styles  # noqa: E402
from utils.places import describe_location  # noqa: E402


def test_hero_headline_and_facts(markdown, predictions):
    summary = summarise_region(predictions, "Australia (National)")
    hero.render_hero(summary, "Australia (National)", "from the last saved prediction run")

    (html,) = markdown
    assert "Extreme</span> fire risk" in html
    assert "3 of 6</b> monitored" in html
    assert ">grid cells</span> (50%)" in html
    assert "33.0°S, 147.0°E" in html
    assert describe_location(-33.0, 147.0) in html
    assert ">90<small>%</small>" in html
    assert "Showing 6 grid cell(s) for Australia (National)" in html
    assert hero.ADVICE["Extreme"] in html


@pytest.mark.parametrize(
    ("level", "left"),
    [("Low", "12.5"), ("Moderate", "37.5"), ("High", "62.5"), ("Extreme", "87.5")],
)
def test_hero_scale_marker_sits_in_the_level_step(level, left):
    """Risk bands are scaled around the model threshold, not evenly spaced in
    probability, so the marker goes in the middle of the level's own step."""
    assert f'class="scale-marker" style="left:{left}%"' in hero._scale(level)


@pytest.mark.parametrize(
    ("region", "authority"),
    [
        ("Australia (National)", "Australian Warning System"),
        ("Victoria", "Country Fire Authority"),
        ("New South Wales", "NSW Rural Fire Service"),
    ],
)
def test_hero_links_the_right_fire_authority(markdown, predictions, region, authority):
    hero.render_hero(summarise_region(predictions, region), region, "")
    assert authority in markdown[0]


def test_every_state_has_a_fire_authority():
    from components.headers import REGIONS
    from utils.regions import is_national

    for region in REGIONS:
        if not is_national(region):
            assert region in hero.FIRE_AUTHORITIES


def test_hero_empty_state(markdown, empty_predictions):
    hero.render_hero(summarise_region(empty_predictions, "Tasmania"), "Tasmania", "")
    assert "No forecast for this region yet" in markdown[0]
    assert "Tasmania" in markdown[0]


# ----------------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------------
def test_html_helper_keeps_words_apart_and_removes_indentation(markdown):
    """Regression: joining lines with "" glued words together ('flaggedfor')
    and four-space indents turned HTML into Markdown code blocks."""
    styles.html(
        """
        <div>
            flagged
            for fire
        </div>
        """
    )
    assert markdown == ["<div> flagged for fire </div>"]


@pytest.mark.parametrize(
    ("lat", "lon", "text"),
    [(-33.5, 151.0, "33.5°S, 151.0°E"), (1.0, -2.25, "1.0°N, 2.2°W")],
)
def test_format_coords(lat, lon, text):
    assert styles.format_coords(lat, lon) == text


def test_map_frames_selected_state(predictions):
    m = map_section._build_map(predictions, "Tasmania")
    rendered = m.get_root().render()
    assert "fitBounds" in rendered
    assert "fitBounds" not in map_section._build_map(predictions, "Australia (National)").get_root().render()
