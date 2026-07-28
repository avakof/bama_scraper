"""Interval-censored durations, sale scoring, and timezone handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bama_monitor.config import MonitorConfig, SaleScoringWeights
from bama_monitor.duration_estimation import estimate_duration, survival_row
from bama_monitor.models import AdStatus, DetailVerdict, SaleLabel
from bama_monitor.sale_scoring import ScoringInputs, label_for, score_sale
from bama_monitor.scheduler import local_slot, next_scheduled_slot, slot_range

DAY = 86400.0
T0 = datetime(2026, 7, 1, 11, 0, tzinfo=UTC)


class TestDurationBounds:
    def test_seen_then_missing_on_consecutive_days(self) -> None:
        """First seen day 1, last seen day 1, missing day 2."""
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0,
            first_missing_at=T0 + timedelta(days=1),
            status=AdStatus.LIKELY_REMOVED,
        )
        assert estimate.minimum_active_seconds == 0.0
        assert estimate.maximum_active_seconds == DAY
        assert estimate.estimated_active_seconds == DAY / 2
        assert estimate.observation_interval_hours == 24.0

    def test_multi_week_listing(self) -> None:
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=20),
            first_missing_at=T0 + timedelta(days=21),
            status=AdStatus.LIKELY_REMOVED,
        )
        days = estimate.as_days()
        assert days["minimum_active_days"] == 20.0
        assert days["maximum_active_days"] == 21.0
        assert days["estimated_active_days"] == 20.5

    def test_bounds_always_ordered(self) -> None:
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=5),
            first_missing_at=T0 + timedelta(days=6),
            status=AdStatus.LIKELY_REMOVED,
        )
        assert (
            estimate.minimum_active_seconds
            <= estimate.estimated_active_seconds
            <= estimate.maximum_active_seconds
        )

    def test_still_active_is_right_censored_with_no_upper_bound(self) -> None:
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=3),
            first_missing_at=None,
            status=AdStatus.ACTIVE,
            now=T0 + timedelta(days=3),
        )
        assert estimate.censored
        assert estimate.maximum_active_seconds is None
        assert estimate.estimated_active_seconds is None
        assert estimate.minimum_active_seconds == 3 * DAY

    def test_missing_once_is_still_censored(self) -> None:
        # One miss is not a removal, so the duration is not yet closed.
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=1),
            first_missing_at=T0 + timedelta(days=2),
            status=AdStatus.MISSING_ONCE,
        )
        assert estimate.censored

    def test_missing_first_seen_yields_nothing(self) -> None:
        estimate = estimate_duration(
            first_seen_at=None,
            last_seen_at=None,
            first_missing_at=None,
            status=AdStatus.UNKNOWN,
        )
        assert estimate.minimum_active_seconds is None
        assert estimate.censored

    def test_reappearance_extends_the_lower_bound(self) -> None:
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=10),
            first_missing_at=None,
            status=AdStatus.REAPPEARED,
            now=T0 + timedelta(days=10),
        )
        assert estimate.minimum_active_seconds == 10 * DAY

    def test_observation_interval_reflects_daily_cadence(self) -> None:
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=1),
            first_missing_at=T0 + timedelta(days=2),
            status=AdStatus.LIKELY_REMOVED,
        )
        # With one scrape per day the uncertainty approaches 24 hours.
        assert estimate.observation_interval_hours == 24.0


class TestSurvivalRow:
    def _record(self, status: AdStatus, **kw) -> dict:
        base = {
            "platform_ad_id": "a1",
            "current_status": str(status),
            "first_seen_at": T0.isoformat(),
            "last_seen_at": (T0 + timedelta(days=2)).isoformat(),
            "first_missing_at": (T0 + timedelta(days=3)).isoformat(),
            "sale_confidence": 0.7,
            "sale_label": "likely_sold",
        }
        base.update(kw)
        return base

    def test_event_observed_means_disappearance_not_sale(self) -> None:
        row = survival_row(self._record(AdStatus.LIKELY_REMOVED))
        assert row["event_observed"] == 1
        assert row["event_type"] == "disappearance_observed"
        # The sale inference is carried separately, never merged in.
        assert row["sale_evidence_score"] == 0.7
        # The old name implied a calibrated probability; the export must not carry
        # it, so a downstream chart cannot label 0.7 as "70% sold".
        assert "sale_confidence" not in row
        assert row["sale_label"] == "likely_sold"

    def test_active_listing_is_censored_not_an_event(self) -> None:
        row = survival_row(self._record(AdStatus.ACTIVE, first_missing_at=None))
        assert row["event_observed"] == 0
        assert row["right_censored"] == 1

    def test_reposted_is_not_an_event(self) -> None:
        row = survival_row(self._record(AdStatus.REPOSTED))
        assert row["event_observed"] == 0
        assert row["event_type"] == "reposted"
        assert row["reposted"] == 1

    def test_bounds_present(self) -> None:
        row = survival_row(self._record(AdStatus.LIKELY_REMOVED))
        assert row["duration_lower_bound_days"] == 2.0
        assert row["duration_upper_bound_days"] == 3.0
        assert row["duration_estimate_days"] == 2.5


class TestSaleScoring:
    weights = SaleScoringWeights()

    def test_two_misses_alone_is_not_enough_for_likely_sold(self) -> None:
        """Absence plus 'no repost found' must not reach the likely_sold band."""
        score = score_sale(
            ScoringInputs(consecutive_misses=2, removal_confirmation_misses=2), self.weights
        )
        assert score.confidence == pytest.approx(0.40)
        assert score.label is SaleLabel.POSSIBLY_SOLD
        assert score.label is not SaleLabel.LIKELY_SOLD

    def test_404_adds_evidence(self) -> None:
        score = score_sale(
            ScoringInputs(
                consecutive_misses=2,
                removal_confirmation_misses=2,
                detail_verdict=DetailVerdict.HTTP_404,
            ),
            self.weights,
        )
        assert score.confidence > 0.5
        assert any(c["rule"] == "detail_persistent_404_410" for c in score.components)

    def test_accessible_detail_page_reduces_confidence(self) -> None:
        """A live page means it left the filter, not the market."""
        absent_only = score_sale(
            ScoringInputs(consecutive_misses=2, removal_confirmation_misses=2), self.weights
        )
        with_page = score_sale(
            ScoringInputs(
                consecutive_misses=2,
                removal_confirmation_misses=2,
                detail_verdict=DetailVerdict.STILL_ACTIVE,
            ),
            self.weights,
        )
        assert with_page.confidence < absent_only.confidence

    def test_reappearance_collapses_confidence(self) -> None:
        score = score_sale(
            ScoringInputs(
                consecutive_misses=2,
                removal_confirmation_misses=2,
                detail_verdict=DetailVerdict.HTTP_404,
                reappeared=True,
            ),
            self.weights,
        )
        assert score.label is SaleLabel.UNKNOWN or score.confidence < 0.65

    def test_repost_reduces_confidence_and_is_never_a_sale(self) -> None:
        score = score_sale(
            ScoringInputs(
                consecutive_misses=2,
                removal_confirmation_misses=2,
                detail_verdict=DetailVerdict.HTTP_404,
                reposted=True,
                seller_reposted_same_vehicle=True,
            ),
            self.weights,
        )
        assert any(c["rule"] == "vehicle_reposted" for c in score.components)
        assert score.confidence < 0.65

    def test_confidence_is_clamped(self) -> None:
        score = score_sale(
            ScoringInputs(
                consecutive_misses=5,
                removal_confirmation_misses=2,
                detail_verdict=DetailVerdict.HTTP_410,
                persistent_unavailable_checks=4,
                had_price_reduction=True,
                first_seen_at=T0,
                first_missing_at=T0 + timedelta(days=1),
                now=T0 + timedelta(days=30),
            ),
            self.weights,
        )
        assert 0.0 <= score.confidence <= 1.0

    def test_components_are_auditable(self) -> None:
        score = score_sale(
            ScoringInputs(consecutive_misses=2, removal_confirmation_misses=2), self.weights
        )
        for component in score.components:
            assert {"rule", "weight", "reason"} <= set(component)
        assert "not a confirmed sale" in score.as_evidence()["disclaimer"]

    def test_confirmed_sold_is_never_produced(self) -> None:
        for confidence in (0.0, 0.4, 0.65, 0.85, 1.0):
            assert label_for(confidence, self.weights) is not SaleLabel.CONFIRMED_SOLD

    @pytest.mark.parametrize(
        "confidence,expected",
        [
            (0.0, SaleLabel.UNKNOWN),
            (0.39, SaleLabel.UNKNOWN),
            (0.40, SaleLabel.POSSIBLY_SOLD),
            (0.64, SaleLabel.POSSIBLY_SOLD),
            (0.65, SaleLabel.LIKELY_SOLD),
            (0.84, SaleLabel.LIKELY_SOLD),
            (0.85, SaleLabel.HIGHLY_LIKELY_SOLD),
            (1.0, SaleLabel.HIGHLY_LIKELY_SOLD),
        ],
    )
    def test_label_boundaries(self, confidence: float, expected: SaleLabel) -> None:
        assert label_for(confidence, self.weights) is expected


class TestTimezoneAndDst:
    cfg = MonitorConfig()

    def test_local_hour_is_always_13(self) -> None:
        zone = ZoneInfo("Europe/Paris")
        for month in range(1, 13):
            slot = local_slot(datetime(2026, month, 15, tzinfo=UTC), self.cfg)
            assert slot.astimezone(zone).hour == 13

    def test_utc_offset_shifts_across_spring_forward(self) -> None:
        before = local_slot(datetime(2026, 3, 28, tzinfo=UTC), self.cfg)
        after = local_slot(datetime(2026, 3, 29, tzinfo=UTC), self.cfg)
        assert before.hour == 12  # CET
        assert after.hour == 11  # CEST
        assert (before + timedelta(days=1)) - after == timedelta(hours=1)

    def test_utc_offset_shifts_across_fall_back(self) -> None:
        before = local_slot(datetime(2026, 10, 24, tzinfo=UTC), self.cfg)
        after = local_slot(datetime(2026, 10, 26, tzinfo=UTC), self.cfg)
        assert before.hour == 11  # CEST
        assert after.hour == 12  # CET

    def test_exactly_one_slot_per_calendar_day_across_dst(self) -> None:
        slots = slot_range(
            self.cfg,
            datetime(2026, 3, 27, tzinfo=UTC),
            datetime(2026, 3, 31, tzinfo=UTC),
        )
        assert len(slots) == 5
        assert len(set(slots)) == 5

    def test_slots_are_all_timezone_aware_utc(self) -> None:
        for slot in slot_range(
            self.cfg, datetime(2026, 10, 23, tzinfo=UTC), datetime(2026, 10, 28, tzinfo=UTC)
        ):
            assert slot.tzinfo is not None
            assert slot.utcoffset() == timedelta(0)

    def test_next_slot_rolls_to_tomorrow_after_the_time_passes(self) -> None:
        after_run = datetime(2026, 7, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris")).astimezone(UTC)
        nxt = next_scheduled_slot(self.cfg, reference=after_run)
        assert nxt.astimezone(ZoneInfo("Europe/Paris")).day == 16

    def test_allow_past_attaches_to_todays_slot(self) -> None:
        after_run = datetime(2026, 7, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris")).astimezone(UTC)
        slot = next_scheduled_slot(self.cfg, reference=after_run, allow_past=True)
        assert slot.astimezone(ZoneInfo("Europe/Paris")).day == 15


class TestDurationOrderingInvariant:
    """min <= est <= max must hold for every shape, or a report can contradict itself."""

    def test_reappeared_listing_has_no_upper_bound(self) -> None:
        # Seen day 1, missed day 2, seen again day 3: the absence is history, and
        # closing the interval on it would put the upper bound below the lower one.
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=3),
            first_missing_at=T0 + timedelta(days=2),
            status=AdStatus.REAPPEARED,
            now=T0 + timedelta(days=3),
        )
        assert estimate.censored
        assert estimate.maximum_active_seconds is None
        assert estimate.estimated_active_seconds is None
        assert estimate.minimum_active_seconds == 3 * DAY

    @pytest.mark.parametrize(
        "status,last_offset,missing_offset",
        [
            (AdStatus.LIKELY_REMOVED, 1, 2),
            (AdStatus.LIKELY_REMOVED, 0, 1),
            (AdStatus.LIKELY_SOLD, 5, 7),
            (AdStatus.REAPPEARED, 3, 2),
            (AdStatus.MISSING_ONCE, 1, 2),
            (AdStatus.ACTIVE, 4, None),
        ],
    )
    def test_bounds_never_invert(
        self, status: AdStatus, last_offset: int, missing_offset: int | None
    ) -> None:
        estimate = estimate_duration(
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=last_offset),
            first_missing_at=None
            if missing_offset is None
            else T0 + timedelta(days=missing_offset),
            status=status,
            now=T0 + timedelta(days=max(last_offset, missing_offset or 0)),
        )
        low = estimate.minimum_active_seconds
        est = estimate.estimated_active_seconds
        high = estimate.maximum_active_seconds
        assert low is not None
        if est is not None:
            assert low <= est, f"{status}: min {low} > est {est}"
        if est is not None and high is not None:
            assert est <= high, f"{status}: est {est} > max {high}"
        if high is not None:
            assert low <= high, f"{status}: min {low} > max {high}"
