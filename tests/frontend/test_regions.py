"""
Tests for core.geography, core.summaries and core.danger -- the layer that
turns per-cell predictions into the region-level numbers every card, chart
and the map display.
"""

import pandas as pd
import pytest
from conftest import CELLS, TODAY, TOMORROW
from core import danger, geography, summaries


# ----------------------------------------------------------------------------
# Region naming and assignment
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label",
    ["Australia (National)", "Australia(national)", "australia", "National", "", None],
)
def test_is_national_accepts_label_variants(label):
    assert geography.is_national(label)


@pytest.mark.parametrize("label", ["Victoria", "New South Wales", "Tasmania"])
def test_is_national_rejects_states(label):
    assert not geography.is_national(label)


@pytest.mark.parametrize(("lat", "lon"), list(CELLS))
def test_assign_state(lat, lon):
    assert geography.assign_state(lat, lon) == CELLS[(lat, lon)]


def test_each_cell_is_assigned_to_exactly_one_state(predictions):
    """Per-state counts must add up to the national total (minus offshore
    cells), otherwise a border cell is being double counted."""
    states = [s for s in geography.STATE_BOUNDS]
    per_state = sum(len(geography.filter_by_region(predictions, s)) for s in states)
    offshore = sum(
        1
        for row in predictions.itertuples()
        if geography.assign_state(row.lat_round, row.lon_round) is None
    )
    assert per_state + offshore == len(predictions)


def test_filter_by_region_national_returns_everything(predictions):
    out = geography.filter_by_region(predictions, geography.NATIONAL)
    assert len(out) == len(predictions)


def test_filter_by_region_state(predictions):
    out = geography.filter_by_region(predictions, "Victoria")
    # Two Victorian cells (one is the NSW/VIC border cell) x two days.
    assert len(out) == 4
    assert set(zip(out["lat_round"], out["lon_round"])) == {(-37.5, 145.0), (-36.0, 147.0)}
    assert list(out.index) == list(range(len(out)))


def test_filter_by_region_unknown_region_returns_everything(predictions):
    assert len(geography.filter_by_region(predictions, "Atlantis")) == len(predictions)


def test_filter_by_region_empty_frame(empty_predictions):
    assert geography.filter_by_region(empty_predictions, "Victoria").empty


def test_get_region_view_falls_back_to_national():
    assert geography.get_region_view("Atlantis") == geography.REGION_VIEWS[geography.NATIONAL]
    assert geography.get_region_view("Tasmania") == geography.REGION_VIEWS["Tasmania"]


def test_every_selectable_region_has_a_view_and_bounds():
    from components.headers import REGIONS

    for region in REGIONS:
        assert region in geography.REGION_VIEWS
        if not geography.is_national(region):
            assert region in geography.STATE_BOUNDS


# ----------------------------------------------------------------------------
# Day handling
# ----------------------------------------------------------------------------
def test_latest_day_only(predictions):
    out = summaries.latest_day_only(predictions)
    assert set(out["acq_date"].dt.date) == {TOMORROW}
    assert len(out) == 6


def test_day_options_one_entry_per_day_oldest_first(predictions):
    options = summaries.day_options(predictions)
    assert [o["date"] for o in options] == [TODAY, TOMORROW]


def test_day_options_headline_figures(predictions):
    today, tomorrow = summaries.day_options(predictions)

    assert today["peak_level"] == "High"
    assert today["peak_probability"] == pytest.approx(0.60)
    assert today["at_risk_count"] == 1
    assert today["cell_count"] == 6
    assert today["max_ffdi"] == pytest.approx(30.0)

    assert tomorrow["peak_level"] == "Extreme"
    assert tomorrow["at_risk_count"] == 3


def test_day_options_empty(empty_predictions):
    assert summaries.day_options(empty_predictions) == []


# ----------------------------------------------------------------------------
# Region summary
# ----------------------------------------------------------------------------
def test_summarise_region_uses_latest_day_only(predictions):
    summary = summaries.summarise_region(predictions, geography.NATIONAL)

    assert summary["has_data"]
    assert summary["forecast_date"] == TOMORROW
    assert summary["cell_count"] == 6
    assert summary["at_risk_count"] == 3
    assert summary["max_risk_level"] == "Extreme"
    assert summary["max_probability"] == pytest.approx(0.90)
    assert summary["max_ffdi"] == pytest.approx(60.0)
    assert summary["hottest_cell"] == (-33.0, 147.0)
    assert summary["risk_counts"] == {"Low": 1, "Moderate": 2, "High": 1, "Extreme": 2}


def test_summarise_region_counts_spinup_cells(predictions):
    today = predictions[predictions["acq_date"].dt.date == TODAY]
    assert summaries.summarise_region(today, geography.NATIONAL)["spinup_count"] == 1


def test_summarise_region_empty_has_same_keys(predictions, empty_predictions):
    """The UI indexes the summary without guards, so the no-data summary must
    carry every key the real one does."""
    full = summaries.summarise_region(predictions, "Victoria")
    empty = summaries.summarise_region(empty_predictions, "Victoria")

    assert not empty["has_data"]
    assert empty.keys() == full.keys()
    assert empty["risk_counts"] == {level: 0 for level in danger.RISK_ORDER}


# ----------------------------------------------------------------------------
# FFDI bands
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ffdi", "band"),
    [
        (0, "Low-Moderate"),
        (11.9, "Low-Moderate"),
        (12, "High"),
        (24.9, "High"),
        (25, "Very High"),
        (50, "Severe"),
        (75, "Extreme"),
        (100, "Catastrophic"),
        (150, "Catastrophic"),
    ],
)
def test_ffdi_band_boundaries(ffdi, band):
    assert danger.ffdi_band(ffdi) == band


def test_vectorised_state_assignment_matches_row_by_row():
    """filter_by_region() uses assign_states() on whole columns for speed; it
    must give exactly the same state as assign_state() for every cell,
    including the overlapping border boxes and offshore cells."""
    lats = [x / 2 for x in range(-90, -18)]   # -45.0 .. -9.5, the whole grid
    lons = [x / 2 for x in range(222, 310)]   # 111.0 .. 154.5
    grid = pd.DataFrame([(la, lo) for la in lats for lo in lons], columns=["lat", "lon"])
    expected = [geography.assign_state(la, lo) for la, lo in zip(grid["lat"], grid["lon"])]
    assert list(geography.assign_states(grid["lat"], grid["lon"])) == expected
