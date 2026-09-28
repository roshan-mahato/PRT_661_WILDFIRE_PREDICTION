"""
Tests for components.charts.

Streamlit output is intercepted: st.plotly_chart hands us the Plotly figure,
so the tests can check the numbers each chart actually plots, and
st.info / st.markdown are recorded to check the empty-state messages.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from components import charts
from conftest import TODAY, TOMORROW
from utils.regions import build_daily_trend, summarise_region

METADATA_PATH = Path(__file__).parents[2] / "models" / "best_fire_model_metadata.json"


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


# ----------------------------------------------------------------------------
# 1. Risk trend
# ----------------------------------------------------------------------------
def test_risk_trend_plots_peak_and_mean_as_percent(st_calls, predictions):
    trend = build_daily_trend(predictions)
    charts.render_risk_trend(trend)

    (fig,) = st_calls["figures"]
    peak, mean = fig.data
    assert peak.name == "Peak cell"
    assert mean.name == "Regional mean"
    assert list(peak.y) == pytest.approx([60.0, 90.0])
    assert list(mean.y) == pytest.approx((trend["mean_probability"] * 100).round(1).tolist())
    assert fig.layout.yaxis.range == (0, 100)


def test_risk_trend_empty_shows_message(st_calls, predictions):
    charts.render_risk_trend(build_daily_trend(predictions.iloc[0:0]))
    assert st_calls["figures"] == []
    assert st_calls["info"]


def test_risk_trend_none(st_calls):
    charts.render_risk_trend(None)
    assert st_calls["figures"] == []
    assert st_calls["info"]


# ----------------------------------------------------------------------------
# 2. Risk outlook by day
# ----------------------------------------------------------------------------
def _bars(fig):
    return {trace.name: list(trace.y) for trace in fig.data if trace.type == "bar"}


def test_risk_outlook_counts_cells_per_level_per_day(st_calls, predictions):
    charts.render_risk_outlook(predictions)

    (fig,) = st_calls["figures"]
    bars = _bars(fig)
    assert bars == {
        "Extreme": [0, 2],
        "High": [1, 1],
        "Moderate": [1, 2],
        "Low": [4, 1],
    }
    assert fig.layout.barmode == "stack"


def test_risk_outlook_extreme_is_stacked_first(st_calls, predictions):
    """Extreme sits on the baseline so it's comparable across days."""
    charts.render_risk_outlook(predictions)
    names = [t.name for t in st_calls["figures"][0].data if t.type == "bar"]
    assert names == ["Extreme", "High", "Moderate", "Low"]


def test_risk_outlook_labels_show_high_or_extreme_share(st_calls, predictions):
    charts.render_risk_outlook(predictions)

    (fig,) = st_calls["figures"]
    (labels,) = [t for t in fig.data if t.type == "scatter"]
    # Today: 1 of 6 cells High+; tomorrow: 3 of 6.
    assert list(labels.text) == ["17%", "50%"]
    assert list(labels.y) == [6, 6]
    assert list(labels.x) == [
        TODAY.strftime("%a %d %b"),
        TOMORROW.strftime("%a %d %b"),
    ]


def test_risk_outlook_missing_levels_are_zero(st_calls, predictions):
    only_low = predictions[predictions["risk_level"] == "Low"]
    charts.render_risk_outlook(only_low)

    bars = _bars(st_calls["figures"][0])
    assert bars["Extreme"] == [0, 0]
    assert bars["Low"] == [4, 1]


def test_risk_outlook_hover_shows_share(st_calls, predictions):
    charts.render_risk_outlook(predictions)
    extreme = st_calls["figures"][0].data[0]
    assert list(extreme.customdata) == pytest.approx([0.0, 2 / 6])


def test_risk_outlook_empty_shows_message(st_calls, empty_predictions):
    charts.render_risk_outlook(empty_predictions)
    assert st_calls["figures"] == []
    assert st_calls["info"]


# ----------------------------------------------------------------------------
# 3. Risk distribution
# ----------------------------------------------------------------------------
def test_risk_distribution_matches_summary(st_calls, predictions):
    summary = summarise_region(predictions, "Australia (National)")
    charts.render_risk_distribution(summary)

    (fig,) = st_calls["figures"]
    pie = fig.data[0]
    assert list(pie.labels) == ["Low", "Moderate", "High", "Extreme"]
    assert list(pie.values) == [1, 2, 1, 2]
    assert "<b>6</b>" in fig.layout.annotations[0].text
    facts = st_calls["markdown"][-1]
    assert "Cells flagged at risk<b>3 of 6</b>" in facts
    assert "Peak probability<b>90.0%</b>" in facts


def test_risk_distribution_no_data(st_calls, empty_predictions):
    charts.render_risk_distribution(summarise_region(empty_predictions, "Victoria"))
    assert st_calls["figures"] == []
    assert st_calls["info"]


# ----------------------------------------------------------------------------
# 4. Factor weights
# ----------------------------------------------------------------------------
def test_group_importance_sums_features_into_factors(feature_importance):
    grouped = charts._group_importance(feature_importance).set_index("group")

    assert grouped.loc["Location", "importance"] == pytest.approx(35.0)
    assert grouped.loc["Fuel & soil dryness", "importance"] == pytest.approx(35.0)
    assert grouped.loc["Heat & air dryness", "importance"] == pytest.approx(12.0)
    assert grouped.loc["Wind", "importance"] == pytest.approx(8.0)
    assert grouped["importance"].sum() == pytest.approx(100.0)


def test_group_importance_sorted_descending(feature_importance):
    grouped = charts._group_importance(feature_importance)
    assert grouped["importance"].is_monotonic_decreasing


def test_group_importance_hover_uses_readable_names(feature_importance):
    grouped = charts._group_importance(feature_importance).set_index("group")
    detail = grouped.loc["Fuel & soil dryness", "detail"]
    assert "Topsoil moisture: 25.0%" in detail
    assert "Drought index (KBDI): 10.0%" in detail
    assert "soil_moisture_0_to_7cm" not in detail


def test_group_importance_unknown_feature_goes_to_other():
    df = pd.DataFrame([{"feature": "new_feature", "importance": 100.0}])
    grouped = charts._group_importance(df)
    assert list(grouped["group"]) == ["Other"]
    assert "new_feature" in grouped["detail"].iloc[0]


def test_every_model_feature_has_a_group_and_label():
    """Guards against a retrain adding a feature that would fall into
    'Other' with a raw column name as its label."""
    features = json.loads(METADATA_PATH.read_text())["features"]
    grouped = {f for _, feats in charts.FACTOR_GROUPS for f in feats}
    assert set(features) <= grouped
    assert set(features) <= set(charts.FEATURE_LABELS)


def test_factor_weights_chart(st_calls, feature_importance):
    charts.render_factor_weights(feature_importance)

    (fig,) = st_calls["figures"]
    bar = fig.data[0]
    assert bar.orientation == "h"
    # Largest factor at the top (y axis is reversed).
    assert list(bar.x) == sorted(bar.x, reverse=True)
    assert set(list(bar.y)[:2]) == {"Location", "Fuel & soil dryness"}
    assert list(bar.text)[:2] == ["35.0%", "35.0%"]
    assert fig.layout.yaxis.autorange == "reversed"


def test_factor_weights_shows_api_error(st_calls):
    charts.render_factor_weights(pd.DataFrame(), "Cannot reach the prediction API")
    assert st_calls["figures"] == []
    assert st_calls["info"] == ["Cannot reach the prediction API"]


def test_factor_weights_default_message(st_calls):
    charts.render_factor_weights(None)
    assert st_calls["info"] == ["Feature importance is not available."]
