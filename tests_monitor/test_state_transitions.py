"""State-machine transitions, tested as pure functions.

These cover the transition table directly. The database-level equivalents live in
``test_simulation.py``; keeping both means a regression in the rules is caught
without a database, and a regression in persistence is caught with one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bama_monitor.models import AdStatus, EventType
from bama_monitor.state_machine import (
    TRANSITION_TABLE,
    AdState,
    is_comparable,
    mark_reposted,
    on_missing,
    on_seen,
    promote_to_likely_sold,
)

T0 = datetime(2026, 7, 1, 11, 0, tzinfo=UTC)
T1 = T0 + timedelta(days=1)
T2 = T0 + timedelta(days=2)
T3 = T0 + timedelta(days=3)


def event_types(outcome) -> list[EventType]:
    return [event for event, _ in outcome.events]


class TestFirstDiscovery:
    def test_unknown_advertisement_becomes_new(self) -> None:
        outcome = on_seen(None, observed_at=T0, is_first_discovery=True)
        assert outcome.status is AdStatus.NEW
        assert outcome.state.first_seen_at == T0
        assert outcome.state.total_seen_runs == 1

    def test_discovery_emits_both_discovered_and_seen(self) -> None:
        outcome = on_seen(None, observed_at=T0, is_first_discovery=True)
        assert event_types(outcome) == [EventType.DISCOVERED, EventType.SEEN]


class TestNewToActive:
    def test_second_sighting_promotes_to_active(self) -> None:
        first = on_seen(None, observed_at=T0, is_first_discovery=True)
        second = on_seen(first.state, observed_at=T1)
        assert second.status is AdStatus.ACTIVE
        assert second.state.total_seen_runs == 2
        assert second.state.first_seen_at == T0  # preserved

    def test_active_stays_active(self) -> None:
        state = AdState(status=AdStatus.ACTIVE, total_seen_runs=5, first_seen_at=T0)
        assert on_seen(state, observed_at=T1).status is AdStatus.ACTIVE


class TestActiveToMissing:
    def test_one_miss_is_missing_once_not_removed(self) -> None:
        """The single most important guard: one absence is not a removal."""
        state = AdState(status=AdStatus.ACTIVE, first_seen_at=T0, last_seen_at=T0)
        outcome = on_missing(state, run_started_at=T1, removal_confirmation_misses=2)
        assert outcome.status is AdStatus.MISSING_ONCE
        assert outcome.status is not AdStatus.LIKELY_REMOVED
        assert outcome.state.consecutive_misses == 1
        assert outcome.state.first_missing_at == T1
        assert event_types(outcome) == [EventType.MISSING_FIRST_TIME]

    def test_missing_event_states_it_is_not_a_sale(self) -> None:
        state = AdState(status=AdStatus.ACTIVE, first_seen_at=T0)
        _, evidence = on_missing(state, run_started_at=T1).events[0]
        assert "not a removal or a sale" in evidence["note"]

    def test_last_seen_is_not_overwritten_by_a_miss(self) -> None:
        state = AdState(status=AdStatus.ACTIVE, first_seen_at=T0, last_seen_at=T0)
        outcome = on_missing(state, run_started_at=T1)
        assert outcome.state.last_seen_at == T0


class TestMissingToLikelyRemoved:
    def test_second_consecutive_miss_confirms_removal(self) -> None:
        state = AdState(
            status=AdStatus.MISSING_ONCE,
            consecutive_misses=1,
            first_seen_at=T0,
            last_seen_at=T0,
            first_missing_at=T1,
        )
        outcome = on_missing(state, run_started_at=T2, removal_confirmation_misses=2)
        assert outcome.status is AdStatus.LIKELY_REMOVED
        assert outcome.state.confirmed_removed_at == T2
        assert outcome.state.first_missing_at == T1  # first miss time preserved
        assert event_types(outcome) == [EventType.REMOVAL_CONFIRMED]

    def test_threshold_of_three_needs_three_misses(self) -> None:
        state = AdState(status=AdStatus.ACTIVE, first_seen_at=T0)
        first = on_missing(state, run_started_at=T1, removal_confirmation_misses=3)
        second = on_missing(first.state, run_started_at=T2, removal_confirmation_misses=3)
        third = on_missing(second.state, run_started_at=T3, removal_confirmation_misses=3)
        assert first.status is AdStatus.MISSING_ONCE
        assert second.status is AdStatus.MISSING_ONCE
        assert third.status is AdStatus.LIKELY_REMOVED

    def test_further_misses_stay_removed_without_duplicate_events(self) -> None:
        state = AdState(
            status=AdStatus.LIKELY_REMOVED,
            consecutive_misses=2,
            first_seen_at=T0,
            first_missing_at=T1,
            confirmed_removed_at=T2,
        )
        outcome = on_missing(state, run_started_at=T3)
        assert outcome.status is AdStatus.LIKELY_REMOVED
        assert outcome.state.consecutive_misses == 3
        assert event_types(outcome) == []  # already reported
        assert outcome.state.confirmed_removed_at == T2  # not moved


class TestReappearance:
    def test_missing_once_reappears(self) -> None:
        state = AdState(status=AdStatus.MISSING_ONCE, consecutive_misses=1, first_seen_at=T0)
        outcome = on_seen(state, observed_at=T2)
        assert outcome.status is AdStatus.REAPPEARED
        assert outcome.state.consecutive_misses == 0
        assert outcome.state.reappeared_at == T2

    def test_likely_removed_reappears(self) -> None:
        state = AdState(
            status=AdStatus.LIKELY_REMOVED,
            consecutive_misses=2,
            first_seen_at=T0,
            first_missing_at=T1,
            confirmed_removed_at=T2,
        )
        outcome = on_seen(state, observed_at=T3)
        assert outcome.status is AdStatus.REAPPEARED
        assert event_types(outcome)[0] is EventType.REAPPEARED

    @pytest.mark.parametrize("start", [AdStatus.LIKELY_SOLD, AdStatus.REPOSTED])
    def test_observation_supersedes_any_inference(self, start: AdStatus) -> None:
        """An inference must never outrank a fresh observation."""
        state = AdState(status=start, consecutive_misses=2, first_seen_at=T0, first_missing_at=T1)
        outcome = on_seen(state, observed_at=T3)
        assert outcome.status is AdStatus.REAPPEARED

    def test_absence_history_is_preserved_on_reappearance(self) -> None:
        state = AdState(
            status=AdStatus.LIKELY_REMOVED,
            consecutive_misses=2,
            first_seen_at=T0,
            first_missing_at=T1,
            confirmed_removed_at=T2,
        )
        outcome = on_seen(state, observed_at=T3)
        assert outcome.state.first_missing_at == T1
        assert outcome.state.confirmed_removed_at == T2

    def test_reappearance_event_explains_the_supersession(self) -> None:
        state = AdState(status=AdStatus.LIKELY_REMOVED, consecutive_misses=2, first_seen_at=T0)
        _, evidence = on_seen(state, observed_at=T3).events[0]
        assert "superseded by this observation" in evidence["note"]

    def test_reappeared_then_seen_again_becomes_active(self) -> None:
        state = AdState(status=AdStatus.REAPPEARED, first_seen_at=T0, reappeared_at=T2)
        assert on_seen(state, observed_at=T3).status is AdStatus.ACTIVE


class TestLikelySold:
    def test_promotion_only_from_likely_removed(self) -> None:
        state = AdState(status=AdStatus.LIKELY_REMOVED, consecutive_misses=2, first_seen_at=T0)
        outcome = promote_to_likely_sold(state, confidence=0.7, evidence={}, at=T3)
        assert outcome.status is AdStatus.LIKELY_SOLD

    @pytest.mark.parametrize(
        "start",
        [AdStatus.ACTIVE, AdStatus.NEW, AdStatus.MISSING_ONCE, AdStatus.REAPPEARED],
    )
    def test_cannot_promote_from_any_other_state(self, start: AdStatus) -> None:
        """A visible or one-day-absent listing can never be called sold."""
        state = AdState(status=start, first_seen_at=T0)
        assert promote_to_likely_sold(state, confidence=0.99, evidence={}, at=T3).status is start

    def test_event_disclaims_confirmation(self) -> None:
        state = AdState(status=AdStatus.LIKELY_REMOVED, consecutive_misses=2)
        _, evidence = promote_to_likely_sold(
            state, confidence=0.8, evidence={"x": 1}, at=T3
        ).events[0]
        assert "NOT a confirmed sale" in evidence["note"]


class TestRepost:
    def test_marks_parent_reposted(self) -> None:
        state = AdState(status=AdStatus.LIKELY_REMOVED, consecutive_misses=2)
        outcome = mark_reposted(state, child_platform_ad_id="new1", score=0.8, evidence={})
        assert outcome.status is AdStatus.REPOSTED
        assert event_types(outcome) == [EventType.POSSIBLE_REPOST]

    def test_event_states_it_is_not_a_sale(self) -> None:
        state = AdState(status=AdStatus.LIKELY_REMOVED, consecutive_misses=2)
        _, evidence = mark_reposted(
            state, child_platform_ad_id="new1", score=0.8, evidence={}
        ).events[0]
        assert "not a sale" in evidence["note"]

    def test_active_advertisement_is_not_marked_reposted(self) -> None:
        state = AdState(status=AdStatus.ACTIVE)
        assert mark_reposted(state, child_platform_ad_id="n", score=0.9, evidence={}).status is (
            AdStatus.ACTIVE
        )


class TestComparablePopulation:
    @pytest.mark.parametrize(
        "status",
        [AdStatus.NEW, AdStatus.ACTIVE, AdStatus.REAPPEARED, AdStatus.MISSING_ONCE],
    )
    def test_included(self, status: AdStatus) -> None:
        assert is_comparable(status)

    @pytest.mark.parametrize(
        "status",
        [AdStatus.LIKELY_REMOVED, AdStatus.LIKELY_SOLD, AdStatus.REPOSTED, AdStatus.UNKNOWN],
    )
    def test_excluded(self, status: AdStatus) -> None:
        # Excluded so a long-removed listing is not re-counted as missing daily.
        assert not is_comparable(status)


class TestTransitionTableIsDocumented:
    def test_table_covers_every_status(self) -> None:
        text = " ".join(f"{a} {b} {c}" for a, b, c in TRANSITION_TABLE)
        for status in AdStatus:
            if status is AdStatus.UNKNOWN:
                continue
            assert str(status) in text, f"{status} missing from the documented table"
