"""Censoring classes, duration bounds and delayed-entry Kaplan-Meier."""

from __future__ import annotations

import pandas as pd
import pytest

from bama_eda.duration_analysis import (
    build_duration_table,
    check_bound_ordering,
    classify_censoring,
    duration_summary,
    kaplan_meier,
    survival_by_group,
)
from bama_eda.models import CensoringClass


def _row(**fields) -> pd.Series:
    base = {
        "current_status": "active",
        "left_truncated": True,
        "filter_exit_reason": None,
    }
    base.update(fields)
    return pd.Series(base)


class TestCensoringClassification:
    def test_still_present_is_right_censored(self) -> None:
        assert classify_censoring(_row(current_status="active")) == str(
            CensoringClass.RIGHT_CENSORED
        )

    def test_missing_once_is_still_right_censored(self) -> None:
        """One absence is not a disappearance."""
        assert classify_censoring(_row(current_status="missing_once")) == str(
            CensoringClass.RIGHT_CENSORED
        )

    def test_disappearance_with_known_entry_is_interval_censored(self) -> None:
        assert classify_censoring(
            _row(current_status="likely_removed", left_truncated=False)
        ) == str(CensoringClass.INTERVAL_CENSORED)

    def test_disappearance_with_unknown_entry_is_left_truncated(self) -> None:
        assert classify_censoring(
            _row(current_status="likely_removed", left_truncated=True)
        ) == str(CensoringClass.LEFT_TRUNCATED)

    def test_filter_exit_wins_over_absence(self) -> None:
        """A listing that left the search has not left the market."""
        assert classify_censoring(
            _row(current_status="likely_removed", filter_exit_reason="price_below_filter")
        ) == str(CensoringClass.ACTIVE_OUTSIDE_FILTER)

    def test_reappeared_and_reposted_have_their_own_classes(self) -> None:
        assert classify_censoring(_row(current_status="reappeared")) == str(
            CensoringClass.REAPPEARED
        )
        assert classify_censoring(_row(current_status="reposted")) == str(CensoringClass.REPOSTED)


class TestDurationBounds:
    def _frame(self) -> pd.DataFrame:
        t0 = pd.Timestamp("2026-06-01T11:00:00Z")
        return pd.DataFrame(
            [
                {
                    "platform_ad_id": "gone",
                    "current_status": "likely_removed",
                    "first_seen_at": t0.isoformat(),
                    "last_seen_at": (t0 + pd.Timedelta(days=3)).isoformat(),
                    "first_missing_at": (t0 + pd.Timedelta(days=4)).isoformat(),
                    "left_truncated": False,
                    "eligible_for_duration_ranking": True,
                    "published_at": (t0 - pd.Timedelta(hours=6)).isoformat(),
                },
                {
                    "platform_ad_id": "here",
                    "current_status": "active",
                    "first_seen_at": t0.isoformat(),
                    "last_seen_at": (t0 + pd.Timedelta(days=3)).isoformat(),
                    "first_missing_at": None,
                    "left_truncated": True,
                    "eligible_for_duration_ranking": False,
                    "published_at": None,
                },
            ]
        )

    def test_bounds_are_ordered(self) -> None:
        table = build_duration_table(self._frame())
        check = check_bound_ordering(table)
        assert check["ok"], check

    def test_a_present_listing_has_no_upper_bound(self) -> None:
        table = build_duration_table(self._frame())
        row = table[table["platform_ad_id"] == "here"].iloc[0]
        assert pd.isna(row["maximum_disappearance_duration"])
        assert row["right_censored"] == 1
        assert row["event_observed"] == 0

    def test_market_duration_only_for_eligible_rows(self) -> None:
        table = build_duration_table(self._frame())
        eligible = table[table["platform_ad_id"] == "gone"].iloc[0]
        ineligible = table[table["platform_ad_id"] == "here"].iloc[0]
        assert not pd.isna(eligible["estimated_market_duration"])
        assert pd.isna(ineligible["estimated_market_duration"])

    def test_market_duration_exceeds_observed_by_the_entry_delay(self) -> None:
        table = build_duration_table(self._frame())
        row = table[table["platform_ad_id"] == "gone"].iloc[0]
        assert row["estimated_market_duration"] > row["observed_monitoring_duration"]
        assert row["estimated_market_duration"] == pytest.approx(3.25, abs=0.01)

    def test_inverted_bounds_are_detected(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "minimum_disappearance_duration": 5.0,
                    "estimated_disappearance_duration": 2.0,
                    "maximum_disappearance_duration": 6.0,
                }
            ]
        )
        check = check_bound_ordering(table)
        assert not check["ok"] and check["violations"] == 1


class TestKaplanMeier:
    def _cohort(self, n: int, events: int) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append(
                {
                    "observed_monitoring_duration": float(1 + i % 20),
                    "event_observed": 1 if i < events else 0,
                    "entry_time_days": 0.0,
                    "deep_brand": "dena" if i % 2 else "peugeot",
                }
            )
        return pd.DataFrame(rows)

    def test_refuses_below_the_thresholds(self) -> None:
        result = kaplan_meier(self._cohort(20, 3), min_subjects=30, min_events=10)
        assert not result["available"]
        assert result["reason"] == "insufficient_sample"
        assert "No curve is estimated" in result["note"]

    def test_estimates_above_the_thresholds(self) -> None:
        result = kaplan_meier(self._cohort(120, 40), min_subjects=30, min_events=10)
        assert result["available"]
        curve = result["curve"]
        assert (curve["survival"].diff().dropna() <= 1e-9).all(), "survival must not rise"
        assert curve["survival"].between(0, 1).all()
        assert (curve["ci_lower"] <= curve["survival"] + 1e-9).all()
        assert (curve["ci_upper"] >= curve["survival"] - 1e-9).all()

    def test_event_definition_is_disappearance_not_sale(self) -> None:
        result = kaplan_meier(self._cohort(120, 40))
        assert "NOT a confirmed sale" in result["event_definition"]

    def test_delayed_entry_changes_the_risk_set(self) -> None:
        """A subject that entered late must not be at risk before it entered."""
        cohort = self._cohort(120, 40)
        late = cohort.copy()
        late["entry_time_days"] = 10.0
        plain = kaplan_meier(cohort)
        delayed = kaplan_meier(late)
        assert plain["available"] and delayed["available"]
        assert delayed["delayed_entry_applied"]
        assert not plain["delayed_entry_applied"]
        # With everyone entering at t=10, early risk sets are empty, so the curves
        # cannot be identical.
        assert not plain["curve"]["survival"].equals(delayed["curve"]["survival"])

    def test_group_comparison_skips_small_groups_visibly(self) -> None:
        cohort = self._cohort(120, 40)
        cohort.loc[cohort.index[:5], "deep_brand"] = "rare"
        result = survival_by_group(cohort, "deep_brand", min_subjects=30, min_events=10)
        skipped = {s["group"] for s in result["groups_skipped"]}
        assert "rare" in skipped
        assert "rare" not in result["curves"]


class TestDurationSummary:
    def test_classes_are_counted_separately(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "platform_ad_id": "a",
                    "current_status": "active",
                    "first_seen_at": "2026-06-01T11:00:00+00:00",
                    "last_seen_at": "2026-06-03T11:00:00+00:00",
                    "first_missing_at": None,
                    "left_truncated": True,
                    "eligible_for_duration_ranking": False,
                    "published_at": None,
                },
                {
                    "platform_ad_id": "b",
                    "current_status": "active_outside_filter",
                    "first_seen_at": "2026-06-01T11:00:00+00:00",
                    "last_seen_at": "2026-06-02T11:00:00+00:00",
                    "first_missing_at": "2026-06-03T11:00:00+00:00",
                    "filter_exit_reason": "price_below_filter",
                    "left_truncated": True,
                    "eligible_for_duration_ranking": False,
                    "published_at": None,
                },
            ]
        )
        summary = duration_summary(build_duration_table(frame))
        assert summary["filter_exits"] == 1
        assert summary["observed_disappearances"] == 0
        assert "not a confirmed sale" in summary["interpretation"].lower()
