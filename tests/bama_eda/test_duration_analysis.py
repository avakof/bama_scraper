"""Eligibility gating for duration rankings and publication coverage."""

from __future__ import annotations

import pandas as pd

from bama_eda.temporal_analysis import (
    duration_ranking_feasibility,
    entry_delay_distribution,
    observation_intervals,
    publication_coverage,
    publication_volume,
)


def _frame(eligible: int, total: int, disappeared: int = 0) -> pd.DataFrame:
    rows = []
    for i in range(total):
        is_eligible = i < eligible
        rows.append(
            {
                "platform_ad_id": f"ad{i}",
                "published_at": "2026-06-01T01:00:00+00:00" if is_eligible else None,
                "published_at_reliable": is_eligible,
                "published_at_source": "detail_relative" if is_eligible else None,
                "left_truncated": not is_eligible,
                "eligible_for_duration_ranking": is_eligible,
                "entry_delay_seconds": 3600.0 if is_eligible else 40 * 86400.0,
                "current_status": "likely_removed" if i < disappeared else "active",
            }
        )
    return pd.DataFrame(rows)


class TestPublicationCoverage:
    def test_counts_and_shares(self) -> None:
        out = publication_coverage(_frame(6, 100))
        assert out["advertisements_total"] == 100
        assert out["eligible_for_duration_ranking"] == 6
        assert out["eligible_share_pct"] == 6.0
        assert out["left_truncated"] == 94

    def test_interpretation_names_the_limit(self) -> None:
        out = publication_coverage(_frame(6, 100))
        assert "left-truncated" in out["interpretation"]
        assert "lower bounds" in out["interpretation"]

    def test_source_breakdown_is_reported(self) -> None:
        out = publication_coverage(_frame(6, 100))
        assert out["publication_source_counts"]["detail_relative"] == 6


class TestFeasibilityGate:
    def test_refuses_a_ranking_on_a_tiny_eligible_sample(self) -> None:
        """The live database has six eligible listings; a top-10 would be a list."""
        verdict = duration_ranking_feasibility(_frame(6, 2819, disappeared=0), min_eligible=30)
        assert not verdict["feasible"]
        assert verdict["verdict"] == "insufficient_data_for_duration_ranking"
        assert "a ranking of this many rows would be a list" in verdict["note"]

    def test_allows_a_ranking_when_the_sample_supports_it(self) -> None:
        verdict = duration_ranking_feasibility(_frame(50, 100, disappeared=20), min_eligible=30)
        assert verdict["feasible"]
        assert verdict["verdict"] == "duration_ranking_available"
        assert verdict["eligible_with_observed_disappearance"] == 20

    def test_eligible_but_no_events_is_still_a_refusal(self) -> None:
        verdict = duration_ranking_feasibility(_frame(50, 100, disappeared=0), min_eligible=30)
        assert not verdict["feasible"]


class TestObservationIntervals:
    def test_interval_is_reported_as_the_resolution_limit(self) -> None:
        panel = pd.DataFrame(
            {
                "advertisement_id": [1, 1, 1],
                "obs_observed_at": [
                    "2026-06-01T11:00:00+00:00",
                    "2026-06-02T11:00:00+00:00",
                    "2026-06-03T11:00:00+00:00",
                ],
            }
        )
        out = observation_intervals(panel)
        assert out["median_interval_hours"] == 24.0
        assert "never a time of sale" in out["interpretation"]

    def test_single_observation_yields_no_interval(self) -> None:
        panel = pd.DataFrame(
            {"advertisement_id": [1], "obs_observed_at": ["2026-06-01T11:00:00+00:00"]}
        )
        assert not observation_intervals(panel)["available"]


class TestEntryDelay:
    def test_bands_mark_which_qualify(self) -> None:
        out = entry_delay_distribution(_frame(6, 100))
        assert not out.empty
        assert set(out.columns) >= {"band", "advertisements", "eligible_band"}
        assert out["advertisements"].sum() == 100

    def test_publication_volume_needs_known_dates(self) -> None:
        assert publication_volume(_frame(0, 10)).empty
        assert not publication_volume(_frame(6, 10)).empty
