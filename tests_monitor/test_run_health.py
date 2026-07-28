"""Run-health gating.

The property under test throughout: when a run is anything other than ``valid``,
**no advertisement's missing counter moves**. That is the difference between a
dataset that can be trusted and one that silently records live listings as removed.
"""

from __future__ import annotations

import pytest
from monitor_helpers import make_discovery, misses_of, run_day, status_of

from bama_monitor.config import MonitorConfig
from bama_monitor.db import Database
from bama_monitor.models import AdStatus, EventType, RunHealth
from bama_monitor.run_health import anomaly_alerts, evaluate_run


class TestHealthClassification:
    def test_healthy_run_is_valid(self, cfg: MonitorConfig) -> None:
        report = evaluate_run(make_discovery(["a", "b", "c"]), cfg, baseline_counts=[3, 3, 3])
        assert report.status is RunHealth.VALID
        assert report.valid

    def test_blocked_run_is_blocked(self, cfg: MonitorConfig) -> None:
        result = make_discovery(["a"], blocked=True, termination_reason="blocked_http_429")
        report = evaluate_run(result, cfg)
        assert report.status is RunHealth.BLOCKED
        assert not report.valid

    def test_unverified_termination_is_partial(self, cfg: MonitorConfig) -> None:
        report = evaluate_run(make_discovery(["a", "b"], healthy=False), cfg)
        assert report.status is RunHealth.PARTIAL

    def test_failed_initial_page_is_failed(self, cfg: MonitorConfig) -> None:
        result = make_discovery(["a"], initial_page_ok=False)
        assert evaluate_run(result, cfg).status is RunHealth.FAILED

    def test_unpersisted_inventory_is_failed(self, cfg: MonitorConfig) -> None:
        report = evaluate_run(make_discovery(["a", "b"]), cfg, persisted=False)
        assert report.status is RunHealth.FAILED

    def test_count_collapse_against_median_is_invalid(self, cfg: MonitorConfig) -> None:
        # 2 against a median of 100 is a 98% drop; far past the 70% threshold.
        report = evaluate_run(make_discovery(["a", "b"]), cfg, baseline_counts=[100, 100, 100])
        assert not report.valid
        assert any(c.name == "count_vs_recent_median" and not c.passed for c in report.checks)

    def test_absolute_floor_rejects_tiny_runs(self) -> None:
        cfg = MonitorConfig(health={"min_absolute_count": 50})
        report = evaluate_run(make_discovery(["a", "b"]), cfg)
        assert not report.valid

    def test_too_many_failed_pages_is_partial(self, cfg: MonitorConfig) -> None:
        result = make_discovery(["a", "b"], pages_fetched=5, pages_failed=5)
        report = evaluate_run(result, cfg)
        assert not report.valid
        assert report.status is RunHealth.PARTIAL

    def test_empty_titles_signal_a_selector_change(self, cfg: MonitorConfig) -> None:
        result = make_discovery(["a", "b", "c"], titles={"a": "", "b": "", "c": ""})
        report = evaluate_run(result, cfg)
        assert not report.valid
        failed = [c for c in report.checks if not c.passed]
        assert any("selector change" in c.detail for c in failed)

    def test_mass_disappearance_is_vetoed(self, cfg: MonitorConfig) -> None:
        # 100 known active, only 10 seen: too destructive to trust.
        report = evaluate_run(
            make_discovery([f"a{i}" for i in range(10)]),
            cfg,
            baseline_counts=[10, 10, 10],
            known_active_count=100,
            expected_missing_count=90,
        )
        assert not report.valid
        assert any(c.name == "plausible_missing_ratio" and not c.passed for c in report.checks)

    def test_first_run_has_no_baseline_and_still_passes(self, cfg: MonitorConfig) -> None:
        report = evaluate_run(make_discovery(["a", "b"]), cfg, baseline_counts=[])
        assert report.valid
        skipped = next(c for c in report.checks if c.name == "count_vs_recent_median")
        assert skipped.passed and not skipped.blocking

    def test_reason_lists_failed_checks(self, cfg: MonitorConfig) -> None:
        report = evaluate_run(make_discovery(["a"], blocked=True), cfg)
        assert "not_blocked" in report.reason()

    def test_alerts_are_produced_for_failures(self, cfg: MonitorConfig) -> None:
        result = make_discovery(["a"], blocked=True)
        alerts = anomaly_alerts(evaluate_run(result, cfg), result)
        assert alerts and all("alert_type" in a for a in alerts)


class TestInvalidRunsDoNotTouchCounters:
    """End-to-end: the counters must be untouched for every unhealthy shape."""

    def _baseline(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["a", "b", "c"])
        run_day(db, cfg, day=2, keys=["a", "b", "c"])
        assert status_of(db, "c") == str(AdStatus.ACTIVE)

    def test_partial_run_does_not_mark_missing(self, db: Database, cfg: MonitorConfig) -> None:
        self._baseline(db, cfg)
        _, health, comparison = run_day(db, cfg, day=3, keys=["a"], healthy=False)
        assert health.status is RunHealth.PARTIAL
        assert not comparison.applied
        assert status_of(db, "c") == str(AdStatus.ACTIVE)
        assert misses_of(db, "c") == 0

    def test_blocked_run_does_not_mark_missing(self, db: Database, cfg: MonitorConfig) -> None:
        self._baseline(db, cfg)
        _, health, comparison = run_day(
            db, cfg, day=3, keys=["a"], blocked=True, termination_reason="blocked_http_403"
        )
        assert health.status is RunHealth.BLOCKED
        assert not comparison.applied
        assert misses_of(db, "b") == 0

    def test_collapsed_count_does_not_mark_missing(self, db: Database, cfg: MonitorConfig) -> None:
        cfg = cfg.model_copy(
            update={"health": cfg.health.model_copy(update={"min_absolute_count": 3})}
        )
        self._baseline(db, cfg)
        _, health, comparison = run_day(db, cfg, day=3, keys=["a"])
        assert not health.valid
        assert not comparison.applied
        assert misses_of(db, "b") == 0

    def test_failed_page_does_not_mark_missing(self, db: Database, cfg: MonitorConfig) -> None:
        self._baseline(db, cfg)
        _, health, comparison = run_day(db, cfg, day=3, keys=["a"], initial_page_ok=False)
        assert health.status is RunHealth.FAILED
        assert not comparison.applied
        assert misses_of(db, "b") == 0

    def test_parser_failure_does_not_mark_missing(self, db: Database, cfg: MonitorConfig) -> None:
        self._baseline(db, cfg)
        _, health, comparison = run_day(db, cfg, day=3, keys=["a", "b"], titles={"a": "", "b": ""})
        assert not health.valid
        assert not comparison.applied
        assert misses_of(db, "c") == 0

    def test_invalid_run_still_persists_its_observations(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Evidence is kept even when the run cannot be compared."""
        self._baseline(db, cfg)
        context, health, comparison = run_day(db, cfg, day=3, keys=["a"], healthy=False)
        assert not comparison.applied
        seen = db.scalar(
            "SELECT COUNT(*) FROM daily_ad_observations WHERE run_id=? AND was_seen=?",
            [context.run_id, True],
        )
        assert seen == 1

    def test_invalid_run_writes_no_missing_observations(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        self._baseline(db, cfg)
        context, _, _ = run_day(db, cfg, day=3, keys=["a"], healthy=False)
        absent = db.scalar(
            "SELECT COUNT(*) FROM daily_ad_observations WHERE run_id=? AND was_seen=?",
            [context.run_id, False],
        )
        assert absent == 0

    def test_invalid_run_is_not_used_as_a_baseline(self, db: Database, cfg: MonitorConfig) -> None:
        """A rejected run must not become the previous-valid-run reference."""
        self._baseline(db, cfg)
        run_day(db, cfg, day=3, keys=["a"], healthy=False)
        from bama_monitor.repository import Repository

        latest_valid = Repository(db).last_valid_run(cfg.configuration_hash())
        assert latest_valid is not None
        run = Repository(db).get_run(int(latest_valid["id"]))
        assert run is not None and run["status"] == str(RunHealth.VALID)

    def test_recovery_after_invalid_day(self, db: Database, cfg: MonitorConfig) -> None:
        """A good run after a bad one behaves as if the bad one never happened."""
        self._baseline(db, cfg)
        run_day(db, cfg, day=3, keys=["a"], healthy=False)
        run_day(db, cfg, day=4, keys=["a", "b"])
        # c was genuinely absent in the day-4 valid run: exactly one miss.
        assert misses_of(db, "c") == 1
        assert status_of(db, "c") == str(AdStatus.MISSING_ONCE)

    def test_listing_first_seen_in_an_invalid_run_is_discovered_by_the_next_valid_one(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A row created by a refused run must still be *counted* as a discovery later.

        The invalid run persists ``z`` as an observed fact but applies no transition,
        so it is not new there. The next valid run is where it enters the tracked set,
        and `new_ids` has to say so -- otherwise the counter reports zero while the
        event log records a discovery, and the ad is skipped for detail scraping and
        repost matching.
        """
        self._baseline(db, cfg)
        _, health3, cmp3 = run_day(db, cfg, day=3, keys=["a", "b", "c", "z"], healthy=False)
        assert not health3.valid and not cmp3.applied
        assert cmp3.new_ids == []  # nothing applied, so nothing counted
        assert cmp3.observed_new_ids == ["z"]  # but the row creation is on record
        assert cmp3.counters()["new_count"] == 0

        _, health4, cmp4 = run_day(db, cfg, day=4, keys=["a", "b", "c", "z"])
        assert health4.valid and cmp4.applied
        assert cmp4.new_ids == ["z"]
        assert cmp4.observed_new_ids == []  # the row already existed
        assert status_of(db, "z") == str(AdStatus.NEW)

        # The counter agrees with the immutable event log, which is what a replay
        # rebuilds itself from.
        events = db.fetchall(
            "SELECT a.platform_ad_id FROM advertisement_events e"
            " JOIN advertisements a ON a.id = e.advertisement_id"
            " WHERE e.event_type = ? ORDER BY e.id",
            [str(EventType.DISCOVERED)],
        )
        assert [str(r["platform_ad_id"]) for r in events].count("z") == 1


class TestNoPreviousValidRun:
    def test_first_run_marks_nothing_missing(self, db: Database, cfg: MonitorConfig) -> None:
        _, health, comparison = run_day(db, cfg, day=1, keys=["a", "b"])
        assert health.valid
        assert comparison.applied
        assert comparison.missing_ids == []
        assert len(comparison.new_ids) == 2

    def test_invalid_first_run_creates_no_advertisement_statuses(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        _, health, comparison = run_day(db, cfg, day=1, keys=["a"], blocked=True)
        assert not health.valid
        assert not comparison.applied
        # The advertisement row exists as evidence but stays `new`, never active.
        assert status_of(db, "a") == str(AdStatus.NEW)


class TestLiveAdapterFidelity:
    """The live discovery adapter must not invent observations.

    `pages_failed` feeds a blocking health check and the daily alerts, so a
    truncated-but-clean run must report zero failed pages. Truncation is already
    reported by `reached_verified_end=False`; counting it twice, once as a fault
    that never happened, would make every bounded run look broken.
    """

    def _result(self, termination: str, pages: int = 3):
        from bama_monitor.scrapers import build_discovery_result

        outcome = {
            "termination_reason": termination,
            "pages_fetched": pages,
            "duplicate_urls": 0,
            "elapsed_seconds": 1.0,
        }
        return build_discovery_result(
            search_url="https://bama.ir/car?x=1",
            cards=[make_discovery(["a", "b"]).cards[0]],
            outcome=outcome,
            evidence_dir="/tmp/x",
        )

    @pytest.mark.parametrize(
        "termination",
        ["max_pages_reached", "max_runtime_reached", "api_exhausted_after_2_empty_pages"],
    )
    def test_clean_truncation_reports_no_failed_pages(self, termination: str) -> None:
        result = self._result(termination)
        assert result.pages_failed == 0
        assert result.errors == []

    @pytest.mark.parametrize(
        "termination,blocked",
        [("http_error_500", False), ("network_error", False), ("blocked_http_429", True)],
    )
    def test_real_page_errors_are_reported(self, termination: str, blocked: bool) -> None:
        result = self._result(termination)
        assert result.pages_failed == 1
        assert result.errors == [termination]
        assert result.blocked is blocked

    def test_truncated_run_is_partial_not_page_failure(self, cfg: MonitorConfig) -> None:
        """The reason a bounded run is refused must be truncation, not a fake fault."""
        report = evaluate_run(self._result("max_pages_reached", pages=2), cfg, baseline_counts=[])
        assert not report.valid
        failed = {c.name for c in report.checks if not c.passed}
        assert "verified_termination" in failed
        assert "failed_page_ratio" not in failed
