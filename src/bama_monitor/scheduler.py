"""Scheduling at 13:00 Europe/Paris.

The wall-clock time is the contract, not a fixed UTC offset: Paris is UTC+1 in
winter and UTC+2 in summer, so a schedule expressed in UTC would drift by an hour
twice a year. Every slot is therefore computed in the target zone and only then
converted to UTC for storage.

Two daylight-saving edge cases are handled explicitly:

* **Spring forward** — 13:00 always exists (the gap is at 02:00-03:00), so nothing
  special is needed for this time, but the helper is written generally.
* **Fall back** — an ambiguous local time resolves to the *first* occurrence
  (``fold=0``), which keeps exactly one run per calendar day.

systemd with ``OnCalendar=`` and ``Timezone=Europe/Paris`` is the preferred
deployment; :func:`run_forever` exists only for a continuously running process.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .config import MonitorConfig


def parse_run_time(value: str) -> time:
    """Parse ``"HH:MM"`` into a :class:`datetime.time`."""
    hour, _, minute = value.partition(":")
    return time(hour=int(hour), minute=int(minute or 0))


def local_slot(day: datetime, cfg: MonitorConfig) -> datetime:
    """The scheduled instant for ``day``, as an aware UTC datetime.

    ``day`` may be in any zone; only its date in the target zone matters.
    """
    zone = ZoneInfo(cfg.timezone)
    target = parse_run_time(cfg.run_at)
    local_date = day.astimezone(zone).date()
    # fold=0 picks the earlier of two identical local times on a fall-back day.
    naive = datetime.combine(local_date, target).replace(fold=0)
    return naive.replace(tzinfo=zone).astimezone(UTC)


def next_scheduled_slot(
    cfg: MonitorConfig, *, reference: datetime | None = None, allow_past: bool = False
) -> datetime:
    """Next (or current) scheduled slot in UTC.

    ``allow_past=True`` returns today's slot even if it has already passed, which
    is what a manually triggered or catch-up run wants: it should attach to today's
    slot rather than tomorrow's.
    """
    now = (reference or datetime.now(UTC)).astimezone(UTC)
    today = local_slot(now, cfg)
    if allow_past or today > now:
        return today
    return local_slot(now + timedelta(days=1), cfg)


def slot_range(cfg: MonitorConfig, start: datetime, end: datetime) -> list[datetime]:
    """Every scheduled slot between two dates inclusive, for backfill."""
    zone = ZoneInfo(cfg.timezone)
    cursor = start.astimezone(zone).date()
    last = end.astimezone(zone).date()
    slots: list[datetime] = []
    while cursor <= last:
        slots.append(local_slot(datetime.combine(cursor, time(12, 0), tzinfo=zone), cfg))
        cursor += timedelta(days=1)
    return slots


def describe_schedule(
    cfg: MonitorConfig, *, reference: datetime | None = None
) -> dict[str, object]:
    """Human-checkable description of the schedule, including the UTC offset.

    Printed by ``bama_monitor validate`` so an operator can confirm the timer will
    fire when they expect, including across a DST boundary.
    """
    now = (reference or datetime.now(UTC)).astimezone(UTC)
    zone = ZoneInfo(cfg.timezone)
    upcoming = [local_slot(now + timedelta(days=offset), cfg) for offset in range(0, 5)]
    return {
        "timezone": cfg.timezone,
        "run_at_local": cfg.run_at,
        "current_utc_offset": now.astimezone(zone).strftime("%z"),
        "next_slot_utc": next_scheduled_slot(cfg, reference=now).isoformat(),
        "next_slot_local": next_scheduled_slot(cfg, reference=now).astimezone(zone).isoformat(),
        "upcoming_slots_local": [s.astimezone(zone).isoformat() for s in upcoming],
        "upcoming_slots_utc": [s.isoformat() for s in upcoming],
        "note": (
            "the local wall-clock time is fixed; the UTC instant shifts by one hour "
            "across daylight-saving transitions"
        ),
    }


def run_forever(cfg: MonitorConfig) -> None:  # pragma: no cover - long-running
    """APScheduler loop for a continuously running process.

    Only appropriate when the application is already a long-lived service. For a
    scheduled batch job, systemd's timer is more robust: it survives process
    death, records the outcome, and does not need this process to stay alive.
    """
    import asyncio

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    from .runner import DailyRunner

    target = parse_run_time(cfg.run_at)
    scheduler = BlockingScheduler(timezone=ZoneInfo(cfg.timezone))

    def job() -> None:
        runner = DailyRunner(cfg)
        try:
            asyncio.run(runner.run_daily())
        finally:
            runner.close()

    scheduler.add_job(
        job,
        CronTrigger(hour=target.hour, minute=target.minute, timezone=ZoneInfo(cfg.timezone)),
        id="bama_monitor_daily",
        # If a previous run overran, do not stack another on top.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
