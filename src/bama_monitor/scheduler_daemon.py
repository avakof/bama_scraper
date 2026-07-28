"""Long-running scheduler: fires the daily production run at 13:00 Europe/Paris.

Design decisions worth stating, because each one is a way this could quietly do
the wrong thing:

**The timezone is explicit, never the machine's.** ``CronTrigger`` is constructed
with ``ZoneInfo("Europe/Paris")``. If the Mac is set to Tehran, London or UTC, the
job still fires at 13:00 Paris. A scheduler that inherits the system timezone
looks correct on the developer's laptop and silently drifts on any other machine.

**A missed slot is reconciled, not invented.** On startup the daemon asks whether
today's slot already has a *genuine production* run. If not, and the slot has
passed, it starts a ``catch_up`` run whose ``scheduled_for`` is the intended 13:00
and whose ``started_at`` is the real time — never a fabricated 13:00 observation.
Past the grace window it records a missed slot and alerts instead.

**Manual runs do not count.** The reconciliation lookup is scoped to production
triggers, so a manual run attributed to today's slot neither satisfies nor blocks
the scheduled execution.

**Overlap is prevented at four layers.** APScheduler's ``max_instances=1``, the
filesystem lock, the database lock row, and (on PostgreSQL) a session advisory
lock. Any one of them can fail; all four failing at once is the scenario this does
not defend against.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
import uuid
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from bama_scraper.logging_config import get_logger

from .alerts import AlertManager
from .config import MonitorConfig, load_config
from .db import connect, parse_ts, utcnow
from .models import AlertSeverity, TriggerType
from .repository import Repository
from .scheduler import next_scheduled_slot, parse_run_time

log = get_logger("bama_monitor.scheduler_daemon")


class SchedulerDaemon:
    """Owns the production schedule for one search configuration."""

    def __init__(self, cfg: MonitorConfig) -> None:
        self.cfg = cfg
        self.zone = ZoneInfo(cfg.timezone)
        # Identifies this daemon process in every run it creates, so a run can be
        # traced back to the scheduler instance that fired it.
        self.instance_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.scheduler: BlockingScheduler | None = None
        self._stopping = False

    # -- slot arithmetic ---------------------------------------------------

    def slot_for_date(self, day: datetime) -> datetime:
        """The 13:00 Paris instant for the Paris date of ``day``, as UTC.

        Built by localising a naive wall-clock time in the target zone, so the
        UTC instant moves across DST while the local time stays fixed. ``fold=0``
        resolves the ambiguous repeated hour in autumn to the first occurrence.
        """
        target = parse_run_time(self.cfg.run_at)
        local_date = day.astimezone(self.zone).date()
        naive = datetime.combine(local_date, target)
        return naive.replace(tzinfo=self.zone, fold=0).astimezone(UTC)

    def todays_slot(self, *, now: datetime | None = None) -> datetime:
        return self.slot_for_date(now or utcnow())

    def next_slot(self, *, now: datetime | None = None) -> datetime:
        return next_scheduled_slot(self.cfg, reference=now or utcnow())

    def describe_next(self, *, now: datetime | None = None) -> dict[str, str]:
        instant = self.next_slot(now=now)
        return {
            "europe_paris": instant.astimezone(self.zone).isoformat(),
            "utc": instant.astimezone(UTC).isoformat(),
            "local_machine": instant.astimezone().isoformat(),
        }

    # -- reconciliation ----------------------------------------------------

    def reconcile(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Decide what, if anything, today's slot still needs.

        Returns a decision rather than acting, so the same logic is testable
        without a database write or a network call.
        """
        now = now or utcnow()
        slot = self.todays_slot(now=now)
        db = connect(self.cfg.database_url)
        try:
            repo = Repository(db)
            existing = repo.genuine_production_run_for_slot(
                scheduled_for=slot,
                configuration_hash=self.cfg.configuration_hash(),
                production_schedule_name=self.cfg.production_schedule_name,
            )
            if existing is not None:
                return {
                    "action": "none",
                    "slot": slot.isoformat(),
                    "reason": (
                        f"genuine production run {existing['id']} already exists for this slot"
                    ),
                    "existing_run_id": int(existing["id"]),
                }

            if now < slot:
                return {
                    "action": "wait",
                    "slot": slot.isoformat(),
                    "reason": (
                        f"slot has not arrived; {(slot - now).total_seconds() / 60:.0f} "
                        "minute(s) to go"
                    ),
                }

            late_minutes = (now - slot).total_seconds() / 60.0
            if late_minutes <= self.cfg.catch_up_grace_minutes:
                return {
                    "action": "catch_up",
                    "slot": slot.isoformat(),
                    "minutes_late": round(late_minutes, 1),
                    "reason": (
                        f"slot passed {late_minutes:.0f} min ago, inside the "
                        f"{self.cfg.catch_up_grace_minutes} min grace window"
                    ),
                }
            return {
                "action": "record_missed",
                "slot": slot.isoformat(),
                "minutes_late": round(late_minutes, 1),
                "reason": (
                    f"slot passed {late_minutes:.0f} min ago, beyond the "
                    f"{self.cfg.catch_up_grace_minutes} min grace window; the "
                    "observation cannot be recovered and will not be fabricated"
                ),
            }
        finally:
            db.close()

    def record_missed_slot(self, decision: dict[str, Any]) -> None:
        """Persist the fact that a slot was missed, and alert."""
        db = connect(self.cfg.database_url)
        try:
            repo = Repository(db)
            slot = parse_ts(decision["slot"])
            repo.record_missed_slot(
                {
                    "production_schedule_name": self.cfg.production_schedule_name,
                    "scheduled_for": slot,
                    "scheduled_date_paris": (
                        slot.astimezone(self.zone).date().isoformat() if slot else None
                    ),
                    "detected_at": utcnow(),
                    "minutes_late": decision.get("minutes_late"),
                    "resolution": "outside_grace_window",
                    "note": decision.get("reason"),
                }
            )
            AlertManager(repo, self.cfg.alerts).raise_alert(
                "scheduled_slot_missed",
                f"the {decision['slot']} slot was missed: {decision.get('reason')}",
                severity=AlertSeverity.WARNING,
            )
            log.warning("scheduler.slot_missed", slot=decision["slot"])
        finally:
            db.close()

    # -- execution ---------------------------------------------------------

    def execute(
        self, *, trigger_type: TriggerType, scheduled_for: datetime | None = None
    ) -> dict[str, Any]:
        """Run one production job. Called by the trigger and by reconciliation."""
        from .runner import DailyRunner

        slot = scheduled_for or self.todays_slot()
        log.info(
            "scheduler.job_start",
            trigger=str(trigger_type),
            slot=slot.isoformat(),
            instance=self.instance_id,
        )
        db = connect(self.cfg.database_url)
        runner = DailyRunner(self.cfg, db=db)
        try:
            outcome = asyncio.run(
                runner.run_daily(
                    scheduled_for=slot,
                    trigger_type=str(trigger_type),
                    is_synthetic=False,
                    production_schedule_name=self.cfg.production_schedule_name,
                    scheduler_instance_id=self.instance_id,
                )
            )
            payload = outcome.as_dict()
            self._after_run(db, outcome)
            log.info(
                "scheduler.job_finished",
                run_id=outcome.run_id,
                status=str(outcome.status),
                trigger=str(trigger_type),
            )
            return payload
        except Exception as exc:  # noqa: BLE001 - a scheduler must not die on a job
            log.error("scheduler.job_failed", error=repr(exc))
            try:
                AlertManager(Repository(db), self.cfg.alerts).raise_alert(
                    "scheduled_run_failed",
                    f"{trigger_type} run raised {exc!r}",
                    severity=AlertSeverity.CRITICAL,
                )
            except Exception:  # noqa: BLE001
                pass
            return {"error": repr(exc), "trigger_type": str(trigger_type)}
        finally:
            db.close()

    def _after_run(self, db: Any, outcome: Any) -> None:
        """Export the daily snapshot and advance the history marker."""
        from .daily_snapshot import export_daily_snapshot

        if outcome.run_id is None:
            return
        published = False
        try:
            result = export_daily_snapshot(db, self.cfg, outcome.run_id, require_valid=True)
            published = bool(result.get("exported"))
            log.info(
                "scheduler.snapshot",
                run_id=outcome.run_id,
                exported=published,
                latest_valid_updated=result.get("latest_valid_updated"),
            )
        except Exception as exc:  # noqa: BLE001 - a failed export must not lose the run
            log.error("scheduler.snapshot_failed", run_id=outcome.run_id, error=repr(exc))

        repo = Repository(db)
        schedule = repo.get_production_schedule(self.cfg.production_schedule_name)
        first_id = schedule.get("first_genuine_scheduled_run_id") if schedule else None
        # The boundary marks the first day a reader can actually use, so it needs
        # the snapshot to have been published — not merely a run that finished.
        # A genuine run that fails its validity gate is real history and stays in
        # `monitoring_runs`, but it is not the start of the mother dataset.
        if first_id is None and published and str(outcome.status) == "valid":
            repo.upsert_production_schedule(
                {
                    "name": self.cfg.production_schedule_name,
                    "search_configuration_hash": self.cfg.configuration_hash(),
                    "timezone": self.cfg.timezone,
                    "run_at_local": self.cfg.run_at,
                    "first_genuine_scheduled_run_id": outcome.run_id,
                    "host_name": socket.gethostname(),
                    "notes": "first genuine scheduled execution",
                }
            )

    # -- lifecycle ---------------------------------------------------------

    def register_schedule(self) -> None:
        """Record that this machine owns the production schedule."""
        db = connect(self.cfg.database_url)
        try:
            repo = Repository(db)
            existing = repo.get_production_schedule(self.cfg.production_schedule_name)
            payload: dict[str, Any] = {
                "name": self.cfg.production_schedule_name,
                "search_configuration_hash": self.cfg.configuration_hash(),
                "timezone": self.cfg.timezone,
                "run_at_local": self.cfg.run_at,
                "host_name": socket.gethostname(),
                "installed_by": os.environ.get("USER"),
            }
            if not existing or not existing.get("production_schedule_started_at"):
                payload["production_schedule_started_at"] = utcnow()
                payload["notes"] = (
                    "genuine scheduled history starts here; earlier runs are manual "
                    "or test executions"
                )
            repo.upsert_production_schedule(payload)
        finally:
            db.close()

    def start(self) -> int:
        """Install the trigger, reconcile, then block."""
        self.register_schedule()
        hour, minute = self.cfg.run_at.split(":")[0], self.cfg.run_at.split(":")[1]
        trigger = CronTrigger(hour=int(hour), minute=int(minute), timezone=self.zone)
        self.scheduler = BlockingScheduler(
            timezone=self.zone,
            executors={"default": ThreadPoolExecutor(1)},
            job_defaults={
                # One at a time, and a run that fires while another is going is
                # collapsed rather than queued: two discoveries of the same slot
                # would double-count observations.
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": self.cfg.catch_up_grace_minutes * 60,
            },
        )
        self.scheduler.add_job(
            self._scheduled_job,
            trigger=trigger,
            id="bama-daily-production",
            name=f"Bama daily production run at {self.cfg.run_at} {self.cfg.timezone}",
            replace_existing=True,
        )

        log.info(
            "scheduler.started",
            instance=self.instance_id,
            timezone=self.cfg.timezone,
            run_at=self.cfg.run_at,
            next_fire=json.dumps(self.describe_next()),
            system_timezone=str(datetime.now().astimezone().tzinfo),
        )
        print(
            json.dumps(
                {
                    "event": "scheduler_started",
                    "instance_id": self.instance_id,
                    "pid": os.getpid(),
                    "timezone": self.cfg.timezone,
                    "run_at": self.cfg.run_at,
                    "next_trigger": self.describe_next(),
                    "database": _redact(self.cfg.database_url),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        self._reconcile_on_startup()
        self._install_signal_handlers()
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("scheduler.stopped")
        return 0

    def _scheduled_job(self) -> None:
        self.execute(trigger_type=TriggerType.SCHEDULED, scheduled_for=self.todays_slot())

    def _reconcile_on_startup(self) -> None:
        decision = self.reconcile()
        log.info("scheduler.reconcile", **{k: str(v) for k, v in decision.items()})
        print(json.dumps({"event": "reconcile", **decision}, ensure_ascii=False), flush=True)

        if decision["action"] == "catch_up":
            slot = parse_ts(decision["slot"])
            outcome = self.execute(trigger_type=TriggerType.CATCH_UP, scheduled_for=slot)
            db = connect(self.cfg.database_url)
            try:
                Repository(db).record_missed_slot(
                    {
                        "production_schedule_name": self.cfg.production_schedule_name,
                        "scheduled_for": slot,
                        "scheduled_date_paris": (
                            slot.astimezone(self.zone).date().isoformat() if slot else None
                        ),
                        "detected_at": utcnow(),
                        "minutes_late": decision.get("minutes_late"),
                        "resolution": "caught_up",
                        "catch_up_run_id": outcome.get("run_id"),
                        "note": decision.get("reason"),
                    }
                )
            finally:
                db.close()
        elif decision["action"] == "record_missed":
            self.record_missed_slot(decision)

    def _install_signal_handlers(self) -> None:
        def handle(signum: int, _frame: Any) -> None:
            if self._stopping:
                return
            self._stopping = True
            log.info("scheduler.signal", signal=signum)
            if self.scheduler is not None:
                # wait=True: let an in-flight run finish rather than leaving a
                # half-applied comparison and a held lock behind.
                self.scheduler.shutdown(wait=True)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handle)


def _redact(url: str) -> str:
    if url.startswith(("postgres://", "postgresql://")):
        return "postgresql://<redacted>"
    return url


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--run-at", default=None, help="HH:MM local to --timezone")
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--production-schedule-name", default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="reconcile and exit without installing the trigger (for testing)",
    )
    parser.add_argument(
        "--print-next",
        action="store_true",
        help="print the next trigger instant and exit",
    )
    args = parser.parse_args(argv)

    cfg = load_config(
        args.config,
        database_url=args.database_url,
        run_at=args.run_at,
        timezone=args.timezone,
        production_schedule_name=args.production_schedule_name,
    )
    cfg.ensure_dirs()
    daemon = SchedulerDaemon(cfg)

    if args.print_next:
        print(json.dumps(daemon.describe_next(), ensure_ascii=False, indent=2))
        return 0
    if args.once:
        decision = daemon.reconcile()
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    return daemon.start()


if __name__ == "__main__":
    sys.exit(main())
