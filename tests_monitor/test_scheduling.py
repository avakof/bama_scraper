"""Scheduling, provenance, slot ownership and daily detail coverage.

These tests defend the distinction the whole deployment rests on: a run that a
human started is not the day's scheduled observation, however good its data is.
"""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from monitor_helpers import integrity_error, run_day

from bama_monitor.config import MonitorConfig
from bama_monitor.daily_snapshot import (
    build_daily_dataset,
    export_daily_snapshot,
    genuine_history,
    read_latest_pointer,
    validate_run,
)
from bama_monitor.db import Database, utcnow
from bama_monitor.inventory_comparison import skipped_run
from bama_monitor.models import (
    ACCOUNTED_OUTCOMES,
    DetailCheckOutcome,
    RunHealth,
)
from bama_monitor.repository import Repository
from bama_monitor.scheduler_daemon import SchedulerDaemon

PARIS = ZoneInfo("Europe/Paris")
REPO = Path(__file__).resolve().parents[1]


def _daemon(cfg: MonitorConfig) -> SchedulerDaemon:
    return SchedulerDaemon(cfg)


# ---------------------------------------------------------------------------
# Slot ownership
# ---------------------------------------------------------------------------


class TestSlotOwnership:
    def _slot(self) -> datetime:
        return datetime(2026, 8, 3, 11, 0, tzinfo=UTC)

    def _create(self, repo: Repository, cfg: MonitorConfig, **kw) -> tuple[int, bool]:
        payload = {
            "search_url": cfg.search_url,
            "scheduled_for": self._slot(),
            "timezone": cfg.timezone,
            "configuration_hash": cfg.configuration_hash(),
            "scraper_version": "test",
            "monitor_version": "test",
            "previous_valid_run_id": None,
            "production_schedule_name": cfg.production_schedule_name,
        }
        payload.update(kw)
        return repo.create_or_get_run(**payload)

    def test_a_manual_run_does_not_claim_the_scheduled_slot(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        manual_id, _ = self._create(repo, cfg, trigger_type="manual")
        scheduled_id, created = self._create(repo, cfg, trigger_type="scheduled")
        assert created, "the scheduled run must be created, not resumed from the manual one"
        assert scheduled_id != manual_id

    def test_a_test_run_does_not_claim_the_scheduled_slot(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        test_id, _ = self._create(repo, cfg, trigger_type="deployment_test")
        scheduled_id, created = self._create(repo, cfg, trigger_type="scheduled")
        assert created and scheduled_id != test_id

    def test_a_synthetic_run_does_not_claim_the_scheduled_slot(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        synthetic_id, _ = self._create(repo, cfg, trigger_type="scheduled", is_synthetic=True)
        real_id, created = self._create(repo, cfg, trigger_type="scheduled")
        assert created and real_id != synthetic_id

    def test_one_production_run_per_genuine_slot(self, db: Database, cfg: MonitorConfig) -> None:
        repo = Repository(db)
        first, created_first = self._create(repo, cfg, trigger_type="scheduled")
        second, created_second = self._create(repo, cfg, trigger_type="scheduled")
        assert created_first and not created_second
        assert first == second, "the second call must resume, not fork history"

    def test_a_catch_up_run_resumes_the_same_production_slot(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        scheduled, _ = self._create(repo, cfg, trigger_type="scheduled")
        catch_up, created = self._create(repo, cfg, trigger_type="catch_up")
        assert not created and catch_up == scheduled

    def test_a_skipped_run_keeps_its_real_trigger_type(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A declined scheduled trigger is not somebody running the pipeline by hand."""
        repo = Repository(db)
        run_id = skipped_run(
            db,
            search_url=cfg.search_url,
            scheduled_for=self._slot(),
            cfg=cfg,
            holder="run 41 (pid 9999)",
            trigger_type="scheduled",
            production_schedule_name=cfg.production_schedule_name,
        )
        row = repo.get_run(run_id)
        assert row["trigger_type"] == "scheduled"
        assert row["status"] == str(RunHealth.SKIPPED_DUE_TO_EXISTING_RUN)

    def test_a_skipped_run_does_not_fulfil_the_slot(self, db: Database, cfg: MonitorConfig) -> None:
        """It records that the trigger fired, not that the market was observed."""
        repo = Repository(db)
        skipped_run(
            db,
            search_url=cfg.search_url,
            scheduled_for=self._slot(),
            cfg=cfg,
            holder="run 41 (pid 9999)",
            trigger_type="scheduled",
            production_schedule_name=cfg.production_schedule_name,
        )
        assert (
            repo.genuine_production_run_for_slot(
                scheduled_for=self._slot(),
                configuration_hash=cfg.configuration_hash(),
                production_schedule_name=cfg.production_schedule_name,
            )
            is None
        ), "reconciliation must still see this slot as needing a run"

    def test_a_skipped_run_does_not_block_a_later_real_run(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """The database index, not only the lookup, has to allow the retry."""
        repo = Repository(db)
        skipped = skipped_run(
            db,
            search_url=cfg.search_url,
            scheduled_for=self._slot(),
            cfg=cfg,
            holder="run 41 (pid 9999)",
            trigger_type="scheduled",
            production_schedule_name=cfg.production_schedule_name,
        )
        real, created = self._create(repo, cfg, trigger_type="catch_up")
        assert created and real != skipped

    def test_two_declined_attempts_at_one_slot_are_both_recordable(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        ids = [
            skipped_run(
                db,
                search_url=cfg.search_url,
                scheduled_for=self._slot(),
                cfg=cfg,
                holder=f"run {n}",
                trigger_type="scheduled",
                production_schedule_name=cfg.production_schedule_name,
            )
            for n in (41, 42)
        ]
        assert len(set(ids)) == 2
        assert all(repo.get_run(i)["trigger_type"] == "scheduled" for i in ids)

    def test_manual_runs_never_resume_each_other(self, db: Database, cfg: MonitorConfig) -> None:
        """Two manual runs are two events, not one slot."""
        repo = Repository(db)
        first, _ = self._create(repo, cfg, trigger_type="manual")
        second, created = self._create(repo, cfg, trigger_type="manual")
        assert created and first != second

    def test_genuine_lookup_ignores_manual_runs(self, db: Database, cfg: MonitorConfig) -> None:
        repo = Repository(db)
        self._create(repo, cfg, trigger_type="manual")
        found = repo.genuine_production_run_for_slot(
            scheduled_for=self._slot(),
            configuration_hash=cfg.configuration_hash(),
            production_schedule_name=cfg.production_schedule_name,
        )
        assert found is None

    def test_production_slot_uniqueness_is_enforced_by_the_database(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Not only by application logic: the index is the last line of defence."""
        repo = Repository(db)
        self._create(repo, cfg, trigger_type="scheduled")
        with pytest.raises(integrity_error(db)):
            db.execute(
                "INSERT INTO monitoring_runs (search_url, started_at, scheduled_for,"
                " timezone, status, configuration_hash, trigger_type, is_synthetic,"
                " production_schedule_name, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    cfg.search_url,
                    utcnow(),
                    self._slot(),
                    cfg.timezone,
                    "running",
                    cfg.configuration_hash(),
                    "scheduled",
                    False,
                    cfg.production_schedule_name,
                    utcnow(),
                ],
            )


# ---------------------------------------------------------------------------
# Timezone and DST
# ---------------------------------------------------------------------------


class TestTimezone:
    def test_slot_is_paris_regardless_of_the_system_timezone(self, cfg: MonitorConfig) -> None:
        """A scheduler that inherits the machine's zone drifts on any other machine."""
        daemon = _daemon(cfg)
        reference = datetime(2026, 8, 3, 6, 0, tzinfo=UTC)
        expected = daemon.slot_for_date(reference)

        original = os.environ.get("TZ")
        try:
            for zone in ("Asia/Tehran", "UTC", "America/New_York", "Europe/Paris"):
                os.environ["TZ"] = zone
                time_module = __import__("time")
                time_module.tzset()
                assert daemon.slot_for_date(reference) == expected
        finally:
            if original is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original
            __import__("time").tzset()

    def test_local_wall_clock_is_always_thirteen_hundred(self, cfg: MonitorConfig) -> None:
        daemon = _daemon(cfg)
        for month in range(1, 13):
            reference = datetime(2026, month, 15, 6, 0, tzinfo=UTC)
            slot = daemon.slot_for_date(reference)
            assert slot.astimezone(PARIS).strftime("%H:%M") == "13:00"

    def test_march_dst_transition(self, cfg: MonitorConfig) -> None:
        """Europe/Paris springs forward on 2027-03-28: 12:00Z becomes 11:00Z."""
        daemon = _daemon(cfg)
        before = daemon.slot_for_date(datetime(2027, 3, 27, 6, 0, tzinfo=UTC))
        after = daemon.slot_for_date(datetime(2027, 3, 28, 6, 0, tzinfo=UTC))
        assert before.strftime("%H:%M") == "12:00"
        assert after.strftime("%H:%M") == "11:00"
        assert before.astimezone(PARIS).hour == after.astimezone(PARIS).hour == 13

    def test_october_dst_transition(self, cfg: MonitorConfig) -> None:
        """And falls back on 2026-10-25: 11:00Z becomes 12:00Z."""
        daemon = _daemon(cfg)
        before = daemon.slot_for_date(datetime(2026, 10, 24, 6, 0, tzinfo=UTC))
        after = daemon.slot_for_date(datetime(2026, 10, 25, 6, 0, tzinfo=UTC))
        assert before.strftime("%H:%M") == "11:00"
        assert after.strftime("%H:%M") == "12:00"
        assert before.astimezone(PARIS).hour == after.astimezone(PARIS).hour == 13

    def test_next_trigger_is_reported_in_three_clocks(self, cfg: MonitorConfig) -> None:
        described = _daemon(cfg).describe_next()
        assert set(described) == {"europe_paris", "utc", "local_machine"}
        assert "+02:00" in described["europe_paris"] or "+01:00" in described["europe_paris"]


# ---------------------------------------------------------------------------
# Startup reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    def _at(self, cfg: MonitorConfig, when: datetime) -> dict:
        return _daemon(cfg).reconcile(now=when)

    def test_before_the_slot_it_waits(self, db: Database, cfg: MonitorConfig) -> None:
        decision = self._at(cfg, datetime(2026, 8, 3, 9, 0, tzinfo=UTC))  # 11:00 Paris
        assert decision["action"] == "wait"

    def test_after_the_slot_within_grace_it_catches_up(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        decision = self._at(cfg, datetime(2026, 8, 3, 12, 30, tzinfo=UTC))  # 90 min late
        assert decision["action"] == "catch_up"
        assert decision["minutes_late"] == pytest.approx(90, abs=1)

    def test_beyond_the_grace_window_it_records_a_miss(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A missed observation is recorded, never fabricated."""
        decision = self._at(cfg, datetime(2026, 8, 3, 20, 0, tzinfo=UTC))  # 9 h late
        assert decision["action"] == "record_missed"
        assert "will not be fabricated" in decision["reason"]

    def test_an_existing_genuine_run_needs_nothing(self, db: Database, cfg: MonitorConfig) -> None:
        repo = Repository(db)
        slot = _daemon(cfg).slot_for_date(datetime(2026, 8, 3, 12, 0, tzinfo=UTC))
        repo.create_or_get_run(
            search_url=cfg.search_url,
            scheduled_for=slot,
            timezone=cfg.timezone,
            configuration_hash=cfg.configuration_hash(),
            scraper_version="t",
            monitor_version="t",
            previous_valid_run_id=None,
            trigger_type="scheduled",
            production_schedule_name=cfg.production_schedule_name,
        )
        decision = self._at(cfg, datetime(2026, 8, 3, 12, 30, tzinfo=UTC))
        assert decision["action"] == "none"

    def test_a_manual_run_does_not_satisfy_the_slot(self, db: Database, cfg: MonitorConfig) -> None:
        """The bug this deployment exists to fix."""
        repo = Repository(db)
        slot = _daemon(cfg).slot_for_date(datetime(2026, 8, 3, 12, 0, tzinfo=UTC))
        repo.create_or_get_run(
            search_url=cfg.search_url,
            scheduled_for=slot,
            timezone=cfg.timezone,
            configuration_hash=cfg.configuration_hash(),
            scraper_version="t",
            monitor_version="t",
            previous_valid_run_id=None,
            trigger_type="manual",
        )
        decision = self._at(cfg, datetime(2026, 8, 3, 12, 30, tzinfo=UTC))
        assert decision["action"] == "catch_up", "a manual run must not satisfy the slot"

    def test_a_catch_up_run_keeps_the_intended_slot_and_the_real_start(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        slot = datetime(2026, 8, 3, 11, 0, tzinfo=UTC)
        run_id, _ = repo.create_or_get_run(
            search_url=cfg.search_url,
            scheduled_for=slot,
            timezone=cfg.timezone,
            configuration_hash=cfg.configuration_hash(),
            scraper_version="t",
            monitor_version="t",
            previous_valid_run_id=None,
            trigger_type="catch_up",
            production_schedule_name=cfg.production_schedule_name,
        )
        run = repo.get_run(run_id)
        from bama_monitor.db import parse_ts

        assert parse_ts(run["scheduled_for"]) == slot
        assert parse_ts(run["started_at"]) != slot, (
            "started_at must be the real execution time, never a fabricated 13:00"
        )


# ---------------------------------------------------------------------------
# Daily detail coverage
# ---------------------------------------------------------------------------


class TestDailyDetailCoverage:
    def test_every_discovered_advertisement_gets_a_check(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        context, _, _ = run_day(db, cfg, day=1, keys=["a", "b", "c"])
        repo = Repository(db)
        for row in db.fetchall("SELECT id FROM advertisements"):
            repo.record_detail_check(
                {
                    "run_id": context.run_id,
                    "advertisement_id": int(row["id"]),
                    "checked_at": utcnow(),
                    "outcome": str(DetailCheckOutcome.COMPLETED),
                    "detail_availability": "available",
                    "parser_status": "ok",
                }
            )
        counts = repo.detail_check_counts(context.run_id)
        assert counts["total"] == 3

    def test_a_check_is_idempotent_per_run_and_advertisement(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Replaying a run must not inflate the coverage count."""
        context, _, _ = run_day(db, cfg, day=1, keys=["a"])
        repo = Repository(db)
        ad_id = int(db.fetchone("SELECT id FROM advertisements")["id"])
        for _ in range(3):
            repo.record_detail_check(
                {
                    "run_id": context.run_id,
                    "advertisement_id": ad_id,
                    "checked_at": utcnow(),
                    "outcome": str(DetailCheckOutcome.COMPLETED),
                }
            )
        assert repo.detail_check_counts(context.run_id)["total"] == 1

    def test_unchanged_content_references_the_existing_snapshot(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Checking daily must not mean storing the payload daily."""
        context, _, _ = run_day(db, cfg, day=1, keys=["a"])
        repo = Repository(db)
        ad_id = int(db.fetchone("SELECT id FROM advertisements")["id"])
        snapshot_id = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": context.run_id,
                "scraped_at": utcnow(),
                "content_hash": "hash-1",
                "title": "t",
            }
        )
        found = repo.latest_snapshot_for(ad_id, "hash-1")
        assert found is not None and int(found["id"]) == snapshot_id

    def test_a_different_hash_finds_no_existing_snapshot(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        context, _, _ = run_day(db, cfg, day=1, keys=["a"])
        repo = Repository(db)
        ad_id = int(db.fetchone("SELECT id FROM advertisements")["id"])
        repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": context.run_id,
                "scraped_at": utcnow(),
                "content_hash": "hash-1",
                "title": "t",
            }
        )
        assert repo.latest_snapshot_for(ad_id, "hash-2") is None

    def test_accounted_outcomes_exclude_retryable_failures(self) -> None:
        assert DetailCheckOutcome.RETRYABLE_FAILURE not in ACCOUNTED_OUTCOMES
        assert DetailCheckOutcome.COMPLETED in ACCOUNTED_OUTCOMES
        assert DetailCheckOutcome.GONE in ACCOUNTED_OUTCOMES
        assert DetailCheckOutcome.PERMANENT_FAILURE in ACCOUNTED_OUTCOMES


# ---------------------------------------------------------------------------
# Validity and the latest-valid pointer
# ---------------------------------------------------------------------------


class TestSnapshotValidity:
    def _run_with_checks(
        self, db: Database, cfg: MonitorConfig, *, outcome: str, keys: list[str]
    ) -> int:
        context, _, _ = run_day(db, cfg, day=1, keys=keys)
        repo = Repository(db)
        for row in db.fetchall("SELECT id FROM advertisements"):
            repo.record_detail_check(
                {
                    "run_id": context.run_id,
                    "advertisement_id": int(row["id"]),
                    "checked_at": utcnow(),
                    "outcome": outcome,
                }
            )
        repo.update_run(
            context.run_id,
            status="valid",
            comparison_applied=True,
            termination_reason="api_exhausted_after_2_empty_pages",
            finished_at=utcnow(),
        )
        return context.run_id

    def test_a_complete_run_is_valid(self, db: Database, cfg: MonitorConfig) -> None:
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.COMPLETED), keys=["a", "b"]
        )
        assert validate_run(db, run_id).valid

    def test_an_unresolved_retryable_failure_blocks_validity(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.RETRYABLE_FAILURE), keys=["a", "b"]
        )
        validation = validate_run(db, run_id)
        assert not validation.valid
        assert "no_unresolved_retryable_failures" in validation.failures()

    def test_a_gone_page_still_counts_as_accounted(self, db: Database, cfg: MonitorConfig) -> None:
        """404/410 is a complete answer, not a failure to look."""
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.GONE), keys=["a", "b"]
        )
        assert validate_run(db, run_id).valid

    def test_every_check_is_reported_including_passing_ones(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.COMPLETED), keys=["a"]
        )
        validation = validate_run(db, run_id)
        assert len(validation.checks) >= 7
        assert any(c["passed"] for c in validation.checks)

    def test_a_partial_run_cannot_update_the_latest_valid_pointer(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        cfg = cfg.model_copy(update={"daily_snapshot_dir": tmp_path / "snapshots"})
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.RETRYABLE_FAILURE), keys=["a", "b"]
        )
        result = export_daily_snapshot(db, cfg, run_id, require_valid=True)
        assert not result["exported"]
        assert read_latest_pointer(cfg) is None

    def test_a_valid_run_publishes_and_advances_the_pointer(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        cfg = cfg.model_copy(update={"daily_snapshot_dir": tmp_path / "snapshots"})
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.COMPLETED), keys=["a", "b"]
        )
        result = export_daily_snapshot(db, cfg, run_id, require_valid=True)
        assert result["exported"]
        pointer = read_latest_pointer(cfg)
        assert pointer is not None and pointer["run_id"] == run_id

    def test_the_export_directory_is_partitioned_by_slot(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        cfg = cfg.model_copy(update={"daily_snapshot_dir": tmp_path / "snapshots"})
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.COMPLETED), keys=["a"]
        )
        result = export_daily_snapshot(db, cfg, run_id, require_valid=True)
        path = Path(result["destination"])
        assert path.name == f"run_{run_id}"
        assert "Europe-Paris" in path.parent.name
        assert len(path.parent.parent.name) == 10  # YYYY-MM-DD

    def test_no_staging_directory_survives_a_successful_export(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        cfg = cfg.model_copy(update={"daily_snapshot_dir": tmp_path / "snapshots"})
        run_id = self._run_with_checks(
            db, cfg, outcome=str(DetailCheckOutcome.COMPLETED), keys=["a"]
        )
        export_daily_snapshot(db, cfg, run_id, require_valid=True)
        assert not list((tmp_path / "snapshots").rglob(".staging_*"))


# ---------------------------------------------------------------------------
# History boundary
# ---------------------------------------------------------------------------


class TestGenuineHistory:
    def test_synthetic_and_manual_runs_are_excluded_from_history(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        base = datetime(2026, 8, 3, 11, 0, tzinfo=UTC)
        for index, (trigger, synthetic) in enumerate(
            [
                ("scheduled", False),
                ("manual", False),
                ("simulation", True),
                ("deployment_test", False),
                ("catch_up", False),
            ]
        ):
            repo.create_or_get_run(
                search_url=cfg.search_url,
                scheduled_for=base + timedelta(days=index),
                timezone=cfg.timezone,
                configuration_hash=cfg.configuration_hash(),
                scraper_version="t",
                monitor_version="t",
                previous_valid_run_id=None,
                trigger_type=trigger,
                is_synthetic=synthetic,
                production_schedule_name=cfg.production_schedule_name,
            )
        history = genuine_history(db)
        assert set(history["trigger_type"]) == {"scheduled", "catch_up"}
        assert len(history) == 2

    def test_synthetic_runs_can_be_included_explicitly(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        repo = Repository(db)
        repo.create_or_get_run(
            search_url=cfg.search_url,
            scheduled_for=datetime(2026, 8, 3, 11, tzinfo=UTC),
            timezone=cfg.timezone,
            configuration_hash=cfg.configuration_hash(),
            scraper_version="t",
            monitor_version="t",
            previous_valid_run_id=None,
            trigger_type="scheduled",
            is_synthetic=True,
        )
        assert len(genuine_history(db)) == 0
        assert len(genuine_history(db, include_synthetic=True)) == 1

    def test_a_skipped_run_is_not_history(self, db: Database, cfg: MonitorConfig) -> None:
        """It carries a genuine trigger, but it observed nothing."""
        skipped_run(
            db,
            search_url=cfg.search_url,
            scheduled_for=datetime(2026, 8, 3, 11, tzinfo=UTC),
            cfg=cfg,
            holder="run 41",
            trigger_type="scheduled",
            production_schedule_name=cfg.production_schedule_name,
        )
        history = genuine_history(db)
        assert len(history) == 0, "a declined attempt must not appear as a monitored day"

    def test_the_boundary_needs_a_published_day_not_just_a_finished_run(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A genuine run whose snapshot is refused is real history, but it is not
        the first day of the mother dataset — nothing was published.

        The run below finishes healthy yet records no detail checks, so the
        validity gate blocks the export. If the boundary tracked run status alone
        it would point at a day no reader can load.
        """
        context, _health, _cmp = run_day(db, cfg, day=0, keys=["a", "b"])
        repo = Repository(db)
        repo.update_run(context.run_id, trigger_type="scheduled", is_synthetic=False)

        daemon = _daemon(cfg)
        outcome = SimpleNamespace(run_id=context.run_id, status="valid")
        daemon._after_run(db, outcome)

        assert not validate_run(db, context.run_id).valid, "gate must refuse this run"
        schedule = repo.get_production_schedule(cfg.production_schedule_name) or {}
        assert schedule.get("first_genuine_scheduled_run_id") is None


# ---------------------------------------------------------------------------
# Daily dataset reconstruction
# ---------------------------------------------------------------------------


class TestDailyDataset:
    def test_reconstructs_run_and_per_advertisement_timestamps(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        context, _, _ = run_day(db, cfg, day=1, keys=["a", "b"])
        repo = Repository(db)
        for row in db.fetchall("SELECT id FROM advertisements"):
            repo.record_detail_check(
                {
                    "run_id": context.run_id,
                    "advertisement_id": int(row["id"]),
                    "checked_at": utcnow(),
                    "outcome": str(DetailCheckOutcome.COMPLETED),
                }
            )
        frame = build_daily_dataset(db, context.run_id)
        assert len(frame) == 2
        for column in (
            "run_id",
            "scheduled_for",
            "run_started_at",
            "advertisement_id",
            "canonical_url",
            "card_observed_at",
            "detail_checked_at",
            "snapshot_id",
            "current_status",
            "detail_availability",
            "sale_label",
            "sale_evidence_score",
        ):
            assert column in frame.columns, column

    def test_no_snapshot_from_after_the_run_is_attached(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Reached through the run's own check, so a later snapshot cannot leak in."""
        context, _, _ = run_day(db, cfg, day=1, keys=["a"])
        repo = Repository(db)
        ad_id = int(db.fetchone("SELECT id FROM advertisements")["id"])
        today = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": context.run_id,
                "scraped_at": utcnow(),
                "content_hash": "today",
                "title": "today",
            }
        )
        repo.record_detail_check(
            {
                "run_id": context.run_id,
                "advertisement_id": ad_id,
                "checked_at": utcnow(),
                "outcome": str(DetailCheckOutcome.COMPLETED),
                "snapshot_id": today,
            }
        )
        repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": context.run_id,
                "scraped_at": utcnow() + timedelta(days=5),
                "content_hash": "future",
                "title": "future",
            }
        )
        frame = build_daily_dataset(db, context.run_id)
        assert frame.iloc[0]["title"] == "today"

    def test_private_contact_columns_are_absent(self, db: Database, cfg: MonitorConfig) -> None:
        context, _, _ = run_day(db, cfg, day=1, keys=["a"])
        frame = build_daily_dataset(db, context.run_id)
        assert not {"phone", "seller_phone", "dealer_address"} & set(frame.columns)


# ---------------------------------------------------------------------------
# macOS deployment artefacts
# ---------------------------------------------------------------------------


class TestLaunchdArtefacts:
    def test_the_plist_template_is_valid(self) -> None:
        template = REPO / "deploy" / "macos" / "com.bama.monitor.scheduler.plist"
        assert template.exists()
        result = subprocess.run(
            ["plutil", "-lint", str(template)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_every_deployment_script_exists_and_parses(self) -> None:
        macos = REPO / "deploy" / "macos"
        for name in (
            "install_launchd.sh",
            "uninstall_launchd.sh",
            "status_launchd.sh",
            "run_once.sh",
            "scheduler_wrapper.sh",
        ):
            path = macos / name
            assert path.exists(), name
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, f"{name}: {result.stderr}"

    def test_the_readme_exists(self) -> None:
        assert (REPO / "deploy" / "macos" / "README_MACOS.md").exists()

    @pytest.mark.skipif(
        not (Path.home() / "Library/LaunchAgents/com.bama.monitor.scheduler.plist").exists(),
        reason="the LaunchAgent is not installed on this machine",
    )
    def test_the_installed_plist_contains_only_absolute_paths(self) -> None:
        """A relative path in a plist fails silently at login."""
        installed = Path.home() / "Library/LaunchAgents/com.bama.monitor.scheduler.plist"
        payload = plistlib.loads(installed.read_bytes())

        for key in ("WorkingDirectory", "StandardOutPath", "StandardErrorPath"):
            assert payload[key].startswith("/"), f"{key} is not absolute: {payload[key]}"
        for argument in payload["ProgramArguments"]:
            assert argument.startswith("/"), f"argument is not absolute: {argument}"
        import re

        # Only @PLACEHOLDER@ markers matter; the template's own comment mentions
        # them in prose.
        body = installed.read_text()
        leftovers = re.findall(r"@[A-Z_]+@", body)
        assert not leftovers, f"unsubstituted placeholder(s): {leftovers}"

    @pytest.mark.skipif(
        not (Path.home() / "Library/LaunchAgents/com.bama.monitor.scheduler.plist").exists(),
        reason="the LaunchAgent is not installed on this machine",
    )
    def test_the_installed_logs_are_writable(self) -> None:
        installed = Path.home() / "Library/LaunchAgents/com.bama.monitor.scheduler.plist"
        payload = plistlib.loads(installed.read_bytes())
        for key in ("StandardOutPath", "StandardErrorPath"):
            path = Path(payload[key])
            assert path.parent.is_dir(), f"{key} directory missing"
            assert os.access(path.parent, os.W_OK), f"{key} directory not writable"


class TestSnapshotAndHistoryCommands:
    """The two commands an operator uses to answer "is the dataset current?"."""

    def _run_cli(self, argv: list[str], db: Database, cfg: MonitorConfig) -> tuple[int, str]:
        import io
        from contextlib import redirect_stdout

        from bama_monitor import cli

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main([*argv, "--database-url", db.url])
        return code, buf.getvalue()

    def test_validate_only_writes_nothing_and_exits_nonzero_when_invalid(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path, monkeypatch
    ) -> None:
        context, _h, _c = run_day(db, cfg, day=0, keys=["a"])
        monkeypatch.setenv("BAMA_MONITOR_DAILY_SNAPSHOT_DIR", str(tmp_path / "snaps"))
        code, out = self._run_cli(
            ["snapshot", "--run-id", str(context.run_id), "--validate-only"], db, cfg
        )
        assert code == 1, "an invalid run must not report success"
        assert '"valid": false' in out
        assert not (tmp_path / "snaps").exists(), "validate-only must write nothing"

    def test_history_reports_both_boundaries_separately(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """The first genuine run and the first published day are different facts."""
        code, out = self._run_cli(["history"], db, cfg)
        assert code == 0
        payload = json.loads(out)
        assert "first_genuine_run_id" in payload
        assert "first_published_run_id" in payload

    def test_snapshot_command_is_registered(self) -> None:
        from bama_monitor.cli import build_parser

        choices = build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
        assert "snapshot" in choices and "history" in choices


class TestMigrationDialectSelection:
    """A dialect-suffixed migration must reach only its own backend.

    This is not theoretical. The scheduler daemon had been running since before
    the dialect filter was written, so its in-memory module still walked every
    file, and on reconnecting it executed the PostgreSQL-only migration against
    SQLite. That instance was harmless, which is exactly why it is worth a test:
    the next PostgreSQL-only statement would not be.
    """

    def test_each_dialect_sees_only_its_own_suffixed_files(self) -> None:
        from bama_monitor.db import _migration_files

        for dialect in ("sqlite", "postgres"):
            other = "postgres" if dialect == "sqlite" else "sqlite"
            names = [p.name for p in _migration_files(dialect)]
            assert not any(f".{other}.sql" in n for n in names), (dialect, names)

    def test_unsuffixed_migrations_reach_both(self) -> None:
        from bama_monitor.db import _migration_files

        sqlite_names = {p.name for p in _migration_files("sqlite")}
        pg_names = {p.name for p in _migration_files("postgres")}
        shared = sqlite_names & pg_names
        assert all(n.count(".") == 1 for n in shared), shared
        assert len(shared) >= 4

    def test_a_suffixed_pair_covers_the_same_number(self) -> None:
        """If one dialect has NNN_x.sqlite.sql, the other needs its own NNN."""
        from bama_monitor.db import _migration_files

        for dialect in ("sqlite", "postgres"):
            numbers = [p.name.split("_")[0] for p in _migration_files(dialect)]
            assert len(numbers) == len(set(numbers)), (dialect, numbers)


class TestSnapshotFileIntegrity:
    """An export must be readable without guessing what an empty file means."""

    def test_an_empty_event_export_still_has_its_header(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        """A zero-row day and a broken writer produced identical files before.

        Live check: `filter_exits.csv` came out as three bytes of byte-order mark
        and nothing else.
        """
        import pandas as pd

        from bama_monitor.daily_snapshot import EVENT_COLUMNS, _write_frame

        target = tmp_path / "filter_exits.csv"
        written = _write_frame(pd.DataFrame(), target, columns=EVENT_COLUMNS)
        assert written == 0
        header = target.read_text(encoding="utf-8-sig").strip()
        assert header == ",".join(EVENT_COLUMNS)
        assert list(pd.read_csv(target).columns) == list(EVENT_COLUMNS)

    def test_a_missing_slot_serialises_as_json_null(self) -> None:
        """`str(None)` gives "None", which a reader parses as a present value."""
        from bama_monitor.daily_snapshot import _iso

        assert _iso(None) is None
        assert _iso(datetime(2026, 7, 28, 11, tzinfo=UTC)) == "2026-07-28T11:00:00+00:00"
        assert json.loads(json.dumps({"scheduled_for": _iso(None)}))["scheduled_for"] is None
