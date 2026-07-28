"""Locking, overlap prevention, alerting and report generation."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from monitor_helpers import make_context, run_day

from bama_monitor.alerts import Alert, AlertManager, AlertSink, LogSink
from bama_monitor.config import MonitorConfig
from bama_monitor.db import Database, utcnow
from bama_monitor.locking import DatabaseLock, FileLock, LockUnavailable, run_lock
from bama_monitor.models import AlertSeverity, RunHealth
from bama_monitor.reports import generate_report
from bama_monitor.repository import Repository


class TestFileLock:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock = FileLock(tmp_path / "x.lock")
        lock.acquire()
        lock.release()
        lock.acquire()  # reacquirable after release
        lock.release()

    def test_second_holder_is_refused(self, tmp_path: Path) -> None:
        first = FileLock(tmp_path / "x.lock")
        first.acquire()
        try:
            with pytest.raises(LockUnavailable) as exc:
                FileLock(tmp_path / "x.lock").acquire()
            assert exc.value.layer == "filesystem"
        finally:
            first.release()

    def test_holder_is_recorded_for_diagnosis(self, tmp_path: Path) -> None:
        lock = FileLock(tmp_path / "x.lock")
        lock.acquire()
        try:
            assert (tmp_path / "x.lock").read_text(encoding="utf-8").strip()
        finally:
            lock.release()


class TestDatabaseLock:
    def test_acquire_and_release(self, db: Database) -> None:
        lock = DatabaseLock(db)
        lock.acquire(run_id=None)
        assert db.scalar("SELECT COUNT(*) FROM run_locks") == 1
        lock.release()
        assert db.scalar("SELECT COUNT(*) FROM run_locks") == 0

    def test_unexpired_lock_blocks_a_second_run(self, db: Database) -> None:
        first = DatabaseLock(db)
        first.acquire()
        try:
            # A second holder in the same process shares the advisory lock on
            # PostgreSQL, so assert on the row-level refusal explicitly.
            db.execute("UPDATE run_locks SET holder='someone-else'")
            with pytest.raises(LockUnavailable):
                DatabaseLock(db).acquire()
        finally:
            db.execute("DELETE FROM run_locks")

    def test_expired_lock_is_taken_over(self, db: Database) -> None:
        """A crashed run must not block the schedule forever."""
        lock = DatabaseLock(db)
        lock.acquire()
        db.execute(
            "UPDATE run_locks SET holder='dead-process', expires_at=?",
            [utcnow() - timedelta(hours=1)],
        )
        DatabaseLock(db).acquire()  # must not raise
        row = db.fetchone("SELECT holder FROM run_locks")
        assert row is not None and row["holder"] != "dead-process"
        db.execute("DELETE FROM run_locks")

    def test_heartbeat_extends_the_lease(self, db: Database) -> None:
        lock = DatabaseLock(db)
        lock.acquire()
        before = db.fetchone("SELECT expires_at FROM run_locks")
        db.execute("UPDATE run_locks SET expires_at=?", [utcnow() + timedelta(minutes=1)])
        lock.heartbeat()
        after = db.fetchone("SELECT expires_at FROM run_locks")
        assert before is not None and after is not None
        lock.release()


class TestRunLockContext:
    def test_context_acquires_and_releases_both_layers(self, db: Database, tmp_path: Path) -> None:
        with run_lock(db, tmp_path):
            assert db.scalar("SELECT COUNT(*) FROM run_locks") == 1
        assert db.scalar("SELECT COUNT(*) FROM run_locks") == 0

    def test_overlap_is_refused_not_queued(self, db: Database, tmp_path: Path) -> None:
        """A blocked run must skip immediately, never wait and compare stale data."""
        with run_lock(db, tmp_path):
            with pytest.raises(LockUnavailable):
                with run_lock(db, tmp_path):
                    pass

    def test_file_lock_is_released_when_db_lock_fails(self, db: Database, tmp_path: Path) -> None:
        db.execute(
            "INSERT INTO run_locks (lock_name, holder, acquired_at, expires_at) VALUES (?,?,?,?)",
            ["bama_monitor_daily", "other", utcnow(), utcnow() + timedelta(hours=1)],
        )
        with pytest.raises(LockUnavailable):
            with run_lock(db, tmp_path):
                pass
        # The filesystem layer must not be left held after the failure.
        lock = FileLock(tmp_path / "bama_monitor_daily.lock")
        lock.acquire()
        lock.release()


class TestSkippedRunRecording:
    def test_skipped_run_is_recorded_with_its_status(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        from bama_monitor.inventory_comparison import skipped_run

        run_id = skipped_run(
            db, search_url=cfg.search_url, scheduled_for=utcnow(), cfg=cfg, holder="host:123"
        )
        run = Repository(db).get_run(run_id)
        assert run is not None
        assert str(run["status"]) == str(RunHealth.SKIPPED_DUE_TO_EXISTING_RUN)
        alerts = db.fetchall("SELECT alert_type FROM alerts WHERE run_id=?", [run_id])
        assert any(str(a["alert_type"]) == "overlapping_run_skipped" for a in alerts)


class RecordingSink(AlertSink):
    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return True


class TestAlerts:
    def test_alert_is_persisted_and_dispatched(self, db: Database, cfg: MonitorConfig) -> None:
        sink = RecordingSink()
        manager = AlertManager(Repository(db), cfg.alerts, extra_sinks=[sink])
        manager.raise_alert("daily_run_failed", "boom", severity=AlertSeverity.CRITICAL)
        assert db.scalar("SELECT COUNT(*) FROM alerts") == 1
        assert len(sink.sent) == 1

    def test_alert_is_persisted_even_if_a_sink_fails(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Delivery failure must not lose the fact that the condition occurred."""

        class Broken(AlertSink):
            def send(self, alert: Alert) -> bool:
                return False

        manager = AlertManager(Repository(db), cfg.alerts, extra_sinks=[Broken()])
        manager.raise_alert("database_unavailable", "cannot reach db")
        assert db.scalar("SELECT COUNT(*) FROM alerts") == 1

    def test_below_threshold_is_recorded_but_not_dispatched(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        sink = RecordingSink()
        alerts_cfg = cfg.alerts.model_copy(update={"min_severity": "critical"})
        manager = AlertManager(Repository(db), alerts_cfg, extra_sinks=[sink])
        manager.raise_alert("x", "informational", severity=AlertSeverity.INFO)
        assert db.scalar("SELECT COUNT(*) FROM alerts") == 1
        assert sink.sent == []

    def test_log_sink_always_present(self, db: Database, cfg: MonitorConfig) -> None:
        manager = AlertManager(Repository(db), cfg.alerts)
        assert any(isinstance(s, LogSink) for s in manager.sinks)


class TestReports:
    def _simulate(self, db: Database, cfg: MonitorConfig) -> None:
        from monitor_helpers import run_day

        run_day(db, cfg, day=1, keys=["A", "B", "C", "D"])
        run_day(db, cfg, day=2, keys=["A", "B", "D", "E"], prices={"A": 900_000_000})
        run_day(db, cfg, day=3, keys=["A", "D", "E", "F"])
        run_day(db, cfg, day=4, keys=["A", "B", "D", "E", "F"])

    def test_all_required_files_are_written(self, db: Database, cfg: MonitorConfig) -> None:
        self._simulate(db, cfg)
        manifest = generate_report(db, cfg)
        out = cfg.reports_dir / manifest["date"]
        for name in (
            "run_summary.json",
            "daily_inventory.csv",
            "new_ads.csv",
            "missing_ads.csv",
            "likely_removed_ads.csv",
            "reappeared_ads.csv",
            "possible_reposts.csv",
            "likely_sold_ads.csv",
            "price_changes.csv",
            "scrape_errors.csv",
            "dashboard.html",
        ):
            assert (out / name).exists(), name

    def test_summary_separates_observation_from_inference(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        self._simulate(db, cfg)
        manifest = generate_report(db, cfg)
        summary = json.loads(
            (cfg.reports_dir / manifest["date"] / "run_summary.json").read_text(encoding="utf-8")
        )
        interpretation = summary["interpretation"]
        assert "INFERRED" in interpretation["likely_sale"]
        assert "not produced by this system" in interpretation["confirmed_sale"]
        assert "INFERENCE" in summary["disclaimer"]

    def test_dashboard_states_the_distinction(self, db: Database, cfg: MonitorConfig) -> None:
        self._simulate(db, cfg)
        manifest = generate_report(db, cfg)
        html = (cfg.reports_dir / manifest["date"] / "dashboard.html").read_text(encoding="utf-8")
        assert "observed disappearance" in html
        assert "confirmed sale (never inferred)" in html
        assert "INFERENCE" in html

    def test_invalid_run_is_flagged_in_the_dashboard(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        from monitor_helpers import run_day

        run_day(db, cfg, day=1, keys=["A", "B"])
        run_day(db, cfg, day=2, keys=["A"], healthy=False)
        manifest = generate_report(db, cfg)
        html = (cfg.reports_dir / manifest["date"] / "dashboard.html").read_text(encoding="utf-8")
        assert "NO status transitions were applied" in html

    def test_reappeared_report_contains_b(self, db: Database, cfg: MonitorConfig) -> None:
        self._simulate(db, cfg)
        manifest = generate_report(db, cfg)
        text = (cfg.reports_dir / manifest["date"] / "reappeared_ads.csv").read_text(
            encoding="utf-8-sig"
        )
        assert "B" in text

    def test_price_change_report_has_the_reduction(self, db: Database, cfg: MonitorConfig) -> None:
        self._simulate(db, cfg)
        # The price change happened in the day-2 run.
        run = db.fetchone("SELECT id FROM monitoring_runs ORDER BY scheduled_for LIMIT 1 OFFSET 1")
        assert run is not None
        manifest = generate_report(db, cfg, run_id=int(run["id"]))
        text = (cfg.reports_dir / manifest["date"] / "price_changes.csv").read_text(
            encoding="utf-8-sig"
        )
        assert "price_decrease" in text

    def test_analytics_files_are_written(self, db: Database, cfg: MonitorConfig) -> None:
        self._simulate(db, cfg)
        manifest = generate_report(db, cfg)
        out = cfg.reports_dir / manifest["date"]
        for name in (
            # Named for what it measures: a disappearance, not a sale.
            "time_to_disappearance.csv",
            "survival_dataset.csv",
            "cohorts.csv",
            "market_by_brand.csv",
            # Exclusions are published, not silently dropped.
            "left_truncated_excluded.csv",
            # The vehicle grain, where a repost chain is one life.
            "vehicle_durations.csv",
            # The false-removal ledger.
            "filter_exits.csv",
        ):
            assert (out / name).exists(), name
        assert not (out / "time_to_sale.csv").exists()

    def test_survival_export_labels_event_observed_correctly(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        self._simulate(db, cfg)
        manifest = generate_report(db, cfg)
        text = (cfg.reports_dir / manifest["date"] / "survival_dataset.csv").read_text(
            encoding="utf-8-sig"
        )
        header = text.splitlines()[0]
        assert "event_observed" in header
        assert "sale_evidence_score" in header  # inference kept in its own column
        assert "sale_confidence" not in header  # never named like a probability


class TestDialectPortability:
    """Every query must be legal on *both* backends, not just the one used in dev.

    These are the statements that use a portable operator token rather than plain
    SQL: PostgreSQL rejects `x IS ?` as a syntax error while SQLite accepts it as
    null-safe equality, so a query written the SQLite way passes every local test
    and then fails in production.
    """

    def _ad(self, db: Database, cfg: MonitorConfig) -> tuple[int, int]:
        repo = Repository(db)
        context = make_context(repo, cfg, day=1)
        ad_id = repo.upsert_advertisement(
            platform_ad_id="snap-1",
            canonical_url="https://bama.ir/car/detail-snap1",
            search_configuration_hash=cfg.configuration_hash(),
        )
        return context.run_id, ad_id

    def test_snapshot_dedup_with_a_null_content_hash(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_id, ad_id = self._ad(db, cfg)
        repo = Repository(db)
        first = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": run_id,
                "content_hash": None,
                "title": "t",
                "scraped_at": utcnow(),
            }
        )
        second = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": run_id,
                "content_hash": None,
                "title": "t",
                "scraped_at": utcnow(),
            }
        )
        # NULL = NULL is unknown in SQL, so a plain `=` would insert twice.
        assert first is not None and first == second

    def test_snapshot_dedup_with_a_real_content_hash(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_id, ad_id = self._ad(db, cfg)
        repo = Repository(db)
        a = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": run_id,
                "content_hash": "h1",
                "title": "t",
                "scraped_at": utcnow(),
            }
        )
        b = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": run_id,
                "content_hash": "h1",
                "title": "t",
                "scraped_at": utcnow(),
            }
        )
        c = repo.insert_snapshot(
            {
                "advertisement_id": ad_id,
                "run_id": run_id,
                "content_hash": "h2",
                "title": "t2",
                "scraped_at": utcnow(),
            }
        )
        assert a == b
        assert c != a

    def test_no_query_uses_the_non_portable_is_placeholder(self) -> None:
        """Guard the whole package, not just the one query that was found broken."""
        import re
        from pathlib import Path

        offenders = []
        pattern = re.compile(r"\bIS\s+\?")
        for path in Path("src/bama_monitor").rglob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#"):  # prose, incl. this rule's own docs
                    continue
                if pattern.search(line):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
        assert not offenders, "use {{EQ}} for null-safe equality:\n" + "\n".join(offenders)


def manifest_run_id(db: Database) -> int:
    row = db.fetchone("SELECT id FROM monitoring_runs ORDER BY id DESC LIMIT 1")
    assert row is not None
    return int(row["id"])


class TestReportSchemaStability:
    """A quiet day must not change any file's columns."""

    def test_empty_event_files_keep_their_real_header(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        from bama_monitor.reports import EVENT_COLUMNS, PROVENANCE_COLUMNS

        # One run in which nothing goes missing, reappears, reposts or sells.
        run_day(db, cfg, day=1, keys=["a", "b"])
        generate_report(db, cfg)

        out = cfg.reports_dir / "2026-07-01"
        expected = PROVENANCE_COLUMNS + EVENT_COLUMNS
        for name in ("missing_ads", "likely_removed_ads", "reappeared_ads", "likely_sold_ads"):
            text = (out / f"{name}.csv").read_text(encoding="utf-8-sig")
            header = text.splitlines()[0].split(",")
            assert header == expected, f"{name}.csv lost its schema when empty"
            assert len(text.splitlines()) == 1  # header only, no fabricated rows

    def test_populated_and_empty_files_share_one_schema(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=["a", "b"])
        run_day(db, cfg, day=2, keys=["a"])  # b goes missing
        generate_report(db, cfg)
        out = cfg.reports_dir / "2026-07-02"
        missing = (out / "missing_ads.csv").read_text(encoding="utf-8-sig").splitlines()
        empty = (out / "reappeared_ads.csv").read_text(encoding="utf-8-sig").splitlines()
        assert missing[0] == empty[0]
        assert len(missing) == 2
        # Every row is prefixed with the run identifiers, so a count can never be
        # quoted without the run that produced it.
        assert missing[1].split(",")[0] == str(manifest_run_id(db))
        assert ",b," in missing[1]
