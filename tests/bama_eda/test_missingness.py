"""Missingness attribution and unusable-field detection."""

from __future__ import annotations

import pandas as pd

from bama_eda.missingness import (
    field_completeness,
    missingness_by_group,
    missingness_patterns,
    summarise,
    unusable_fields,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "platform_ad_id": ["a", "b", "c", "d"],
            "current_status": ["active", "likely_removed", "active", "likely_removed"],
            "snapshot_scraped_at": ["2026-06-01T11:00:00+00:00", None, None, None],
            "snap_brand": ["dena", None, None, None],
            "deep_brand": ["dena", "peugeot", None, None],
            "always_null": [None, None, None, None],
            "constant": [1, 1, 1, 1],
            "complete": [1, 2, 3, 4],
            "battery": [None, None, None, None],
            "deep_battery_capacity_kwh": [None, None, None, 40.0],
        }
    )


class TestFieldCompleteness:
    def test_counts_and_percentages(self) -> None:
        out = field_completeness(_frame()).set_index("column")
        assert out.loc["complete", "missing_count"] == 0
        assert out.loc["always_null", "missing_pct"] == 100.0
        assert out.loc["deep_brand", "missing_count"] == 2

    def test_always_missing_and_constant_are_flagged(self) -> None:
        out = field_completeness(_frame()).set_index("column")
        assert bool(out.loc["always_null", "always_missing"])
        assert bool(out.loc["constant", "near_zero_variance"])
        assert not bool(out.loc["complete", "near_zero_variance"])

    def test_unsuitable_fields_are_excluded_from_modelling(self) -> None:
        out = field_completeness(_frame()).set_index("column")
        assert not bool(out.loc["always_null", "modelling_suitable"])
        assert not bool(out.loc["constant", "modelling_suitable"])
        assert bool(out.loc["complete", "modelling_suitable"])


class TestCauseAttribution:
    def test_snapshot_columns_missing_without_a_snapshot(self) -> None:
        """Not a parser bug: no detail page was scraped for those listings."""
        out = field_completeness(_frame()).set_index("column")
        assert out.loc["snap_brand", "missing_cause"] == "snapshot_unavailable"

    def test_conditional_fields_are_not_applicable_rather_than_missing(self) -> None:
        out = field_completeness(_frame()).set_index("column")
        assert out.loc["deep_battery_capacity_kwh", "missing_cause"] == "not_applicable"
        assert out.loc["deep_battery_capacity_kwh", "conditional_note"]

    def test_disappearance_before_a_deep_scrape_is_its_own_cause(self) -> None:
        """The bias that matters: the shortest-lived listings are the least scraped."""
        frame = _frame()
        frame["deep_price"] = [1, None, None, None]
        frame["current_status"] = ["active", "likely_removed", "likely_removed", "likely_removed"]
        out = field_completeness(frame).set_index("column")
        assert out.loc["deep_price", "missing_cause"] == "disappeared_before_deep_scrape"

    def test_summary_states_the_bias(self) -> None:
        completeness = field_completeness(_frame())
        summary = summarise(_frame(), completeness)
        assert "not missing at random" in summary["interpretation"]
        assert summary["columns_always_missing"] >= 1


class TestPatterns:
    def test_joint_patterns_are_grouped(self) -> None:
        patterns = missingness_patterns(_frame())
        assert not patterns.empty
        assert patterns["count"].sum() == 4
        assert "missing_fields" in patterns.columns

    def test_by_group_reports_sample_sizes(self) -> None:
        out = missingness_by_group(_frame(), "current_status", ["deep_brand"], min_group_size=3)
        assert set(out["group_value"]) == {"active", "likely_removed"}
        assert "sufficient_sample" in out.columns
        assert not out["sufficient_sample"].all()


class TestUnusableFields:
    def test_categorised(self) -> None:
        groups = unusable_fields(field_completeness(_frame()))
        assert "always_null" in groups["always_missing"]
        assert "constant" in groups["constant"]
        assert "snap_brand" in groups["mostly_missing"]
