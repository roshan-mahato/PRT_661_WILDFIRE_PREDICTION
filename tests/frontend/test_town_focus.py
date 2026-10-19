"""
Tests for town search and "town focus": the search list follows the region
selector, and a chosen town re-centres the page on itself and its
surroundings, by distance, across state borders.
"""

from pathlib import Path

import pandas as pd
import pytest
from conftest import TODAY, TOMORROW
from core import focus as F
from core.geography import STATE_CODES
from core.places import TOWNS, towns_for_region
from streamlit.testing.v1 import AppTest
from utils import api_client

MAIN = Path(__file__).parents[2] / "frontend" / "main.py"


# ----------------------------------------------------------------------------
# The search list follows the region
# ----------------------------------------------------------------------------
def test_national_lists_every_town():
    assert len(towns_for_region("Australia (National)")) == len(TOWNS) == 405


@pytest.mark.parametrize("region", list(STATE_CODES))
def test_a_state_lists_only_its_own_towns(region):
    towns = towns_for_region(region)
    assert towns
    assert all(t.endswith(f", {STATE_CODES[region]}") for t in towns)


def test_state_lists_cover_every_town_exactly_once():
    listed = [t for region in STATE_CODES for t in towns_for_region(region)]
    assert sorted(listed) == sorted(towns_for_region("Australia (National)"))


def test_border_town_is_listed_under_its_real_state():
    """Albury's forecast area falls in the Victoria *box*, but it is a NSW town."""
    assert "Albury, NSW" in towns_for_region("New South Wales")
    assert "Albury, NSW" not in towns_for_region("Victoria")


# ----------------------------------------------------------------------------
# Which forecast area a town uses
# ----------------------------------------------------------------------------
def test_town_uses_its_own_area(predictions):
    focus = F.make_focus(predictions, "Condobolin, NSW", 50)
    assert focus.cell == (-33.0, 147.0)
    assert focus.own_area


def test_border_town_finds_its_forecast(predictions):
    """By distance, not by state box: Albury's area is found even though the
    region filter counts it as Victoria."""
    focus = F.make_focus(predictions, "Albury, NSW", 50)
    assert focus.cell == (-36.0, 147.0)
    assert focus.own_area


def test_town_without_its_own_area_uses_the_nearest(predictions):
    focus = F.make_focus(predictions, "Yass, NSW", 50)   # its own area isn't in the data
    assert focus.cell == (-35.5, 149.0)
    assert not focus.own_area
    assert 60 < focus.offset_km < 90
    assert focus.offset_direction == "S"


def test_town_with_nothing_nearby_has_no_forecast(predictions):
    focus = F.make_focus(predictions, "Perth, WA", 50)
    assert focus.cell is None


def test_focus_name_and_state():
    focus = F.Focus("Kingaroy, QLD", -26.54, 151.84, 50, None)
    assert (focus.name, focus.state) == ("Kingaroy", "QLD")


# ----------------------------------------------------------------------------
# The surroundings
# ----------------------------------------------------------------------------
def test_radius_selects_areas_by_distance(predictions):
    focus = F.make_focus(predictions, "Albury, NSW", 50)
    cells = lambda frame: set(zip(frame["lat_round"], frame["lon_round"]))  # noqa: E731
    assert cells(F.cells_within(predictions, focus)) == {(-36.0, 147.0)}
    # Victoria's (-37.5, 145.0) is ~240 km away; Canberra's (-35.5, 149.0) ~190 km.
    assert cells(F.cells_within(predictions, focus, 200)) == {(-36.0, 147.0), (-35.5, 149.0)}
    assert (-37.5, 145.0) in cells(F.cells_within(predictions, focus, 260))


def test_town_area_is_always_included(predictions):
    """With a small radius the area's centre can be further than the radius
    from the town; the town's own area must still be in scope."""
    focus = F.make_focus(predictions, "Yass, NSW", 25)
    assert (-35.5, 149.0) in set(zip(*[F.cells_within(predictions, focus)[c] for c in ("lat_round", "lon_round")]))


def test_day_strip_shows_the_towns_own_level(predictions):
    focus = F.make_focus(predictions, "Albury, NSW", 200)
    options = F.town_day_options(F.cells_within(predictions, focus), focus)
    assert [(o["date"], o["peak_level"], o["peak_probability"]) for o in options] == [
        (TODAY, "High", 0.6), (TOMORROW, "Extreme", 0.8),
    ]
    # ...while the at-risk counts describe the whole area (2 areas within 200 km)
    assert [o["cell_count"] for o in options] == [2, 2]


def test_nearby_towns_are_nearest_first_with_direction(predictions):
    focus = F.make_focus(predictions, "Albury, NSW", 50)
    tomorrow = predictions[predictions["acq_date"].dt.date == TOMORROW]
    near = F.nearby_towns(tomorrow, focus, 250)
    assert "Albury, NSW" not in list(near["town"])
    assert near.iloc[0]["town"] == "Wodonga, VIC"     # across the border, ~5 km away
    assert near.iloc[0]["distance_km"] < 10
    assert near["distance_km"].is_monotonic_increasing
    assert "Canberra, ACT" in list(near["town"])


def test_timeline_puts_the_town_first(predictions):
    focus = F.make_focus(predictions, "Canberra, ACT", 300)
    rows = F.town_timeline(F.cells_within(predictions, focus), focus, 8)
    assert (rows.iloc[0]["lat_round"], rows.iloc[0]["lon_round"]) == (-35.5, 149.0)
    # Canberra's area is never High or Extreme here, but it still comes first.
    assert rows.iloc[0]["days_elevated"] == 0
    assert len(rows) > 1


# ----------------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------------
@pytest.fixture
def page(monkeypatch, predictions):
    monkeypatch.setattr(api_client, "fetch_stored_predictions", lambda **_: (predictions, None))
    monkeypatch.setattr(api_client, "fetch_live_predictions", lambda **_: (predictions, None))
    monkeypatch.setattr(api_client, "refresh_predictions", lambda **_: (predictions, None))
    monkeypatch.setattr(api_client, "check_api_health", lambda: (True, "API healthy"))
    at = AppTest.from_file(str(MAIN), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _town_box(at):
    return at.selectbox[1]


def test_search_box_lists_all_towns_nationally(page):
    assert len(_town_box(page).options) == 405
    assert _town_box(page).value is None
    assert "Search 405 towns in Australia" in _town_box(page).proto.placeholder


def test_search_box_follows_the_region(page):
    page.selectbox[0].select("New South Wales").run()
    assert len(_town_box(page).options) == len(towns_for_region("New South Wales"))
    assert "Albury, NSW" in _town_box(page).options


def test_choosing_a_town_focuses_the_page(page):
    _town_box(page).select("Albury, NSW").run()
    assert not page.exception
    text = _text(page)
    assert "fire risk in Albury" in text
    assert "within 50 km of Albury, NSW" in text            # banner meta + map subtitle
    assert "Why is Albury at High risk?" in text            # side panel explains the town (today: High)
    assert "Area Summary" in text
    assert page.segmented_control[1].value == 50           # radius control appears
    assert any(b.label == "Clear" for b in page.button)


def test_border_town_works_under_its_own_state(page):
    """NSW selected + Albury: the forecast comes from the national data, so the
    page doesn't wrongly say there is no forecast."""
    page.selectbox[0].select("New South Wales").run()
    _town_box(page).select("Albury, NSW").run()
    assert "fire risk in Albury" in _text(page)
    assert "No forecast near" not in _text(page)


def test_changing_the_radius_widens_the_area(page):
    _town_box(page).select("Albury, NSW").run()
    page.segmented_control[1].set_value(100).run()
    assert "within 100 km of Albury, NSW" in _text(page)


def test_switching_to_a_region_without_the_town_clears_it(page):
    _town_box(page).select("Albury, NSW").run()
    page.selectbox[0].select("Victoria").run()
    assert _town_box(page).value is None
    assert "fire risk in Albury" not in _text(page)
    assert "Region Threat Summary" in _text(page)


def test_clear_returns_to_the_region_view(page):
    _town_box(page).select("Albury, NSW").run()
    next(b for b in page.button if b.label == "Clear").click().run()
    assert _town_box(page).value is None
    assert "Region Threat Summary" in _text(page)


def test_town_with_no_forecast_falls_back_to_the_region(page):
    _town_box(page).select("Perth, WA").run()
    assert not page.exception
    assert any("No forecast area near Perth, WA" in w.value for w in page.warning)
    assert "Region Threat Summary" in _text(page)


def test_fallback_area_is_explained(page):
    _town_box(page).select("Yass, NSW").run()
    assert "Yass's own area has no forecast" in _text(page)


def test_charts_switch_to_the_town(page):
    _town_box(page).select("Albury, NSW").run()
    charts = [c for c in page.get("plotly_chart")]
    assert charts, "expected the insight charts to render"
    text = _text(page)
    assert "Fire danger" in text and "in Albury's area" in text            # chart 1
    assert "the area within 50 km of Albury" in text                        # chart 2
    assert "Towns within 100 km of Albury" in text                          # chart 3
    assert "Albury first, then the most persistent places" in text          # chart 4


def test_nearby_town_counts_by_its_own_distance():
    """Regression: Gladstone is ~90 km from Rockhampton, but the centre of its
    forecast area is ~120 km away. Within 100 km it must still be listed."""
    from conftest import _row

    data = pd.DataFrame([
        _row(-23.5, 150.5, TODAY, 0.53, "High"),    # Rockhampton's area
        _row(-24.0, 151.5, TODAY, 0.20, "Moderate"),  # Gladstone's area
    ])
    focus = F.make_focus(data, "Rockhampton, QLD", 50)
    near = F.nearby_towns(data, focus, 100)
    assert "Gladstone, QLD" in list(near["town"])
    row = near[near["town"] == "Gladstone, QLD"].iloc[0]
    assert 80 < row["distance_km"] < 100
    assert row["risk_level"] == "Moderate"
    assert row["direction"] == "SE"


def test_town_map_is_the_full_heat_map_centred_on_the_town(predictions):
    """The map in town focus looks like the region view -- heat layer and a
    circle for every area -- just centred on the town with a pin; no radius
    circle and no cut-down set of areas."""
    from components import map_section

    tomorrow = predictions[predictions["acq_date"].dt.date == TOMORROW]
    focus = F.make_focus(predictions, "Albury, NSW", 50)
    html = map_section._build_map(tomorrow, "Australia (National)", focus).get_root().render()

    assert html.count('"type": "Point"') == len(tomorrow)     # every area, not just nearby
    assert "HeatLayer" in html or "heatLayer" in html
    assert "L.marker(" in html                                # pin on the town
    assert "L.circle(" not in html                            # no radius circle...
    assert '"dashArray": "' not in html                       # ...and no dashed line anywhere
    assert "fitBounds" in html
