"""Command-line interface for the monitoring system."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bama_scraper.logging_config import configure_logging, get_logger

from .analytics import (
    cohort_analysis,
    create_views,
    daily_inventory,
    fastest_disappearing,
    filter_exit_report,
    left_truncated_disappearances,
    market_statistics,
    survival_dataset,
    time_to_disappearance,
    truncation_summary,
)
from .config import MONITOR_VERSION, MonitorConfig, load_config
from .db import connect, migrate, utcnow
from .models import AdStatus, RunHealth
from .reports import generate_report
from .repository import Repository
from .scheduler import describe_schedule, next_scheduled_slot, slot_range
from .validation import VALID_OUTCOMES
from .vehicle_grain import rebuild_vehicle_entities, vehicle_durations

log = get_logger("bama_monitor.cli")

COMMANDS = (
    ("run-daily", "full daily pipeline: discover, compare, verify, score, report"),
    ("discover-only", "run discovery and persist the inventory without comparing"),
    ("compare", "re-run the comparison for an existing run id"),
    ("verify-missing", "verify detail pages of advertisements missing in a run"),
    ("refresh-details", "run the deep scraper for advertisements due a refresh"),
    ("detect-reposts", "run repost detection over recent removals"),
    ("report", "generate the daily report tree"),
    ("backfill", "create runs for a date range (simulation/catch-up)"),
    ("validate", "check configuration, schedule, database and scraper wiring"),
    ("migrate", "apply database migrations"),
    ("analytics", "print an analytics view as JSON"),
    ("status", "summarise current inventory state"),
    ("validation-sample", "draw a stratified sample of disappearances to label by hand"),
    ("validation-ingest", "load a completed labelling worksheet"),
    ("calibration", "report the measured sale rate per evidence-score band"),
    ("snapshot", "validate a run and export its daily snapshot"),
    ("history", "where genuine scheduled history starts, and what came before it"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bama_monitor",
        description="Longitudinal monitoring of Bama.ir vehicle listings.",
    )
    parser.add_argument("--config", default=None, help="YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in COMMANDS:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--database-url", default=None)
        p.add_argument("--search-url", default=None)
        p.add_argument("--timezone", default=None)
        p.add_argument("--run-at", default=None)
        p.add_argument("--removal-confirmation-misses", type=int, default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument("--reports-dir", default=None)
        p.add_argument("--log-level", default=None)
        p.add_argument("--dry-run", action="store_true", default=None)
        if name in ("compare", "verify-missing", "report", "snapshot"):
            p.add_argument("--run-id", type=int, default=None)
        if name == "snapshot":
            p.add_argument(
                "--validate-only",
                action="store_true",
                help="report the validity checks without writing anything",
            )
            p.add_argument(
                "--allow-invalid",
                action="store_true",
                help=(
                    "export the run's own directory even though it failed validation. "
                    "`latest_valid.json` is still NOT advanced."
                ),
            )
        if name == "report":
            p.add_argument("--date", default=None, help="YYYY-MM-DD")
        if name == "backfill":
            p.add_argument("--from-date", required=True)
            p.add_argument("--to-date", required=True)
        if name == "run-daily":
            p.add_argument(
                "--scheduled-for",
                default=None,
                help="ISO instant to attribute this run to (production triggers only)",
            )
            p.add_argument(
                "--trigger-type",
                default="manual",
                choices=[
                    "scheduled",
                    "catch_up",
                    "manual",
                    "deployment_test",
                    "backfill",
                    "simulation",
                ],
                help=(
                    "how this run is triggered. `manual` (the default) takes NO "
                    "scheduled slot, so running by hand can never satisfy or block "
                    "the day's scheduled execution"
                ),
            )
            p.add_argument("--is-synthetic", action="store_true", default=False)
            p.add_argument("--production-schedule-name", default=None)
            p.add_argument(
                "--export-snapshot",
                action="store_true",
                default=False,
                help="write the daily snapshot export after the run",
            )
        if name == "analytics":
            p.add_argument(
                "--view",
                default="daily_inventory",
                choices=[
                    "daily_inventory",
                    "time_to_disappearance",
                    "fastest_disappearing",
                    "left_truncated_excluded",
                    "vehicle_durations",
                    "filter_exits",
                    "truncation_summary",
                    "market",
                    "cohorts",
                    "survival",
                ],
            )
            p.add_argument("--dimension", default="brand")
            p.add_argument("--limit", type=int, default=50)
        if name == "validation-sample":
            p.add_argument("--per-band", type=int, default=25)
            p.add_argument("--out", default=None, help="worksheet path")
        if name == "validation-ingest":
            p.add_argument("--worksheet", required=True)
            p.add_argument("--labelled-by", default="manual")
    return parser


def _config(args: argparse.Namespace) -> MonitorConfig:
    return load_config(
        args.config,
        database_url=args.database_url,
        search_url=args.search_url,
        timezone=args.timezone,
        run_at=args.run_at,
        removal_confirmation_misses=args.removal_confirmation_misses,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        log_level=args.log_level,
        dry_run=args.dry_run,
    )


def _dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    configure_logging(cfg.log_level)
    cfg.ensure_dirs()
    command = args.command

    if command == "migrate":
        db = connect(cfg.database_url, run_migrations=False)
        try:
            applied = migrate(db, verbose=True)
            create_views(db)
            _dump({"applied": applied, "dialect": db.dialect, "views": list(create_views(db))})
        finally:
            db.close()
        return 0

    if command == "validate":
        return _cmd_validate(cfg)

    db = connect(cfg.database_url)
    repo = Repository(db)
    try:
        if command == "run-daily":
            from .runner import DailyRunner

            slot = (
                datetime.fromisoformat(args.scheduled_for).astimezone(UTC)
                if getattr(args, "scheduled_for", None)
                else None
            )
            trigger = getattr(args, "trigger_type", "manual")
            runner = DailyRunner(cfg, db=db)
            outcome = asyncio.run(
                runner.run_daily(
                    scheduled_for=slot,
                    trigger_type=trigger,
                    is_synthetic=bool(getattr(args, "is_synthetic", False)),
                    production_schedule_name=getattr(args, "production_schedule_name", None)
                    or cfg.production_schedule_name,
                )
            )
            payload = outcome.as_dict()
            if outcome.run_id and not outcome.skipped:
                generate_report(db, cfg, run_id=outcome.run_id)
                if getattr(args, "export_snapshot", False):
                    from .daily_snapshot import export_daily_snapshot

                    payload["snapshot"] = export_daily_snapshot(
                        db, cfg, outcome.run_id, require_valid=True
                    )
            _dump(payload)
            # Non-zero only for genuine failures; a skipped or invalid run is a
            # recorded outcome, not a crash, but must be visible to the caller.
            return 0 if outcome.status is RunHealth.VALID else 1

        if command == "discover-only":
            from .inventory_comparison import InventoryComparator
            from .models import RunContext
            from .scrapers import BamaDiscoveryScraper

            slot = next_scheduled_slot(cfg, allow_past=True)
            previous = repo.last_valid_run(cfg.configuration_hash())
            run_id, _ = repo.create_or_get_run(
                search_url=cfg.search_url,
                scheduled_for=slot,
                timezone=cfg.timezone,
                configuration_hash=cfg.configuration_hash(),
                scraper_version="discover-only",
                monitor_version=MONITOR_VERSION,
                previous_valid_run_id=int(previous["id"]) if previous else None,
            )
            context = RunContext(
                run_id=run_id,
                scheduled_for=slot,
                started_at=utcnow(),
                search_url=cfg.search_url,
                timezone=cfg.timezone,
                scraper_version="discover-only",
                configuration_hash=cfg.configuration_hash(),
            )
            evidence = cfg.output_dir / "runs" / f"run_{run_id}"
            scraper = BamaDiscoveryScraper(cfg, evidence)
            result = asyncio.run(scraper.discover(cfg.search_url))
            comparator = InventoryComparator(db, cfg)
            persist = comparator.persist_inventory(result, context)
            health = comparator.validate(result, context, persisted=persist.persisted)
            repo.update_run(
                run_id,
                status=str(health.status),
                finished_at=utcnow(),
                health_reason=health.reason(),
            )
            _dump(
                {
                    "run_id": run_id,
                    "discovered": result.discovered_count,
                    "termination_reason": result.termination_reason,
                    "health": health.as_dict(),
                    "comparison_applied": False,
                    "note": "discover-only never applies status transitions",
                }
            )
            return 0

        if command == "compare":
            _dump(_cmd_compare(db, cfg, args.run_id))
            return 0

        if command == "verify-missing":
            _dump(asyncio.run(_cmd_verify_missing(db, cfg, args.run_id)))
            return 0

        if command == "refresh-details":
            _dump(asyncio.run(_cmd_refresh_details(db, cfg)))
            return 0

        if command == "detect-reposts":
            _dump(_cmd_detect_reposts(db, cfg))
            return 0

        if command == "report":
            _dump(generate_report(db, cfg, run_id=args.run_id, date=args.date))
            return 0

        if command == "backfill":
            slots = slot_range(
                cfg,
                datetime.fromisoformat(args.from_date).replace(tzinfo=UTC),
                datetime.fromisoformat(args.to_date).replace(tzinfo=UTC),
            )
            _dump(
                {
                    "slots": [s.isoformat() for s in slots],
                    "count": len(slots),
                    "note": (
                        "backfill lists the scheduled slots; historical inventory cannot "
                        "be reconstructed after the fact because the site only exposes "
                        "its current state. Use run-daily --scheduled-for per slot to "
                        "attribute a live run to a specific slot."
                    ),
                }
            )
            return 0

        if command == "analytics":
            return _cmd_analytics(db, cfg, args)

        if command == "snapshot":
            from .daily_snapshot import export_daily_snapshot, validate_run

            snapshot_run_id: int | None = args.run_id
            if snapshot_run_id is None:
                latest = db.fetchone("SELECT id FROM monitoring_runs ORDER BY id DESC LIMIT 1")
                snapshot_run_id = int(latest["id"]) if latest else None
            if snapshot_run_id is None:
                _dump({"error": "no runs exist"})
                return 2
            if args.validate_only:
                validation = validate_run(db, snapshot_run_id)
                _dump({"run_id": snapshot_run_id, **validation.as_dict()})
                return 0 if validation.valid else 1
            exported = export_daily_snapshot(
                db, cfg, snapshot_run_id, require_valid=not args.allow_invalid
            )
            _dump(exported)
            return 0 if exported.get("exported") else 1

        if command == "history":
            from .daily_snapshot import describe_history_boundary, read_latest_pointer

            _dump(
                {
                    **describe_history_boundary(db, cfg),
                    "latest_valid_pointer": read_latest_pointer(cfg),
                }
            )
            return 0

        if command == "status":
            _dump(
                {
                    "status_totals": repo.count_by_status(),
                    "runs": db.fetchall(
                        "SELECT id, scheduled_for, status, discovered_count, new_count,"
                        " missing_count, removed_count, reappeared_count, comparison_applied"
                        " FROM monitoring_runs ORDER BY id DESC LIMIT 15"
                    ),
                    "open_alerts": db.scalar(
                        "SELECT COUNT(*) FROM alerts WHERE resolved_at IS NULL"
                    ),
                }
            )
            return 0

        if command == "validation-sample":
            from .validation import draw_sample, write_worksheet

            latest = db.fetchone("SELECT id FROM monitoring_runs ORDER BY id DESC LIMIT 1")
            sample = draw_sample(
                db,
                per_band=args.per_band,
                run_id=int(latest["id"]) if latest else None,
            )
            path = Path(args.out) if args.out else cfg.reports_dir / "validation_worksheet.csv"
            written = write_worksheet(sample, path)
            _dump(
                {
                    "worksheet": str(path),
                    "rows": written,
                    "per_band": args.per_band,
                    "next_step": (
                        "fill in `observed_outcome` for each row, then run "
                        "`validation-ingest --worksheet <path>`"
                    ),
                    "valid_outcomes": list(VALID_OUTCOMES),
                }
            )
            return 0

        if command == "validation-ingest":
            from .validation import ingest_worksheet

            ingested = ingest_worksheet(db, Path(args.worksheet), labelled_by=args.labelled_by)
            _dump(ingested)
            # Non-zero when a label could not be accepted: a rejected row means the
            # worksheet and the database disagree, which must not pass silently.
            return 0 if not ingested["rejected"] else 1

        if command == "calibration":
            from .validation import calibration

            _dump(calibration(db))
            return 0

        print(f"unknown command: {command}", file=sys.stderr)
        return 2
    finally:
        db.close()


def _cmd_validate(cfg: MonitorConfig) -> int:
    """Check everything an operator would otherwise discover at 13:00."""
    report: dict[str, Any] = {
        "monitor_version": MONITOR_VERSION,
        "search_url": cfg.search_url,
        "configuration_hash": cfg.configuration_hash(),
        "removal_confirmation_misses": cfg.removal_confirmation_misses,
        "schedule": describe_schedule(cfg),
        "checks": [],
    }

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})

    try:
        db = connect(cfg.database_url)
        tables = ("monitoring_runs", "advertisements", "daily_ad_observations")
        for table in tables:
            db.scalar(f"SELECT COUNT(*) FROM {table}")
        check("database", True, f"{db.dialect} reachable; core tables present")
        report["dialect"] = db.dialect
        report["status_totals"] = Repository(db).count_by_status()
        db.close()
    except Exception as exc:  # noqa: BLE001
        check("database", False, repr(exc))

    try:
        from .scrapers import BamaDetailScraper, BamaDiscoveryScraper  # noqa: F401

        check("scraper_wiring", True, "discovery and detail adapters import cleanly")
    except Exception as exc:  # noqa: BLE001
        check("scraper_wiring", False, repr(exc))

    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(cfg.timezone)
        check("timezone", True, f"{cfg.timezone} resolves")
    except Exception as exc:  # noqa: BLE001
        check("timezone", False, repr(exc))

    ok = all(c["ok"] for c in report["checks"])
    report["ok"] = ok
    _dump(report)
    return 0 if ok else 1


def _cmd_compare(db: Any, cfg: MonitorConfig, run_id: int | None) -> dict[str, Any]:
    """Re-evaluate health and comparison for a stored run.

    Recomputes from the persisted observations, so it is safe to call after a
    failure without re-scraping.
    """
    repo = Repository(db)
    run = (
        repo.get_run(run_id)
        if run_id
        else db.fetchone("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT 1")
    )
    if not run:
        return {"error": "no such run"}
    observations = db.fetchall(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN was_seen THEN 1 ELSE 0 END) AS seen"
        " FROM daily_ad_observations WHERE run_id=?",
        [int(run["id"])],
    )
    return {
        "run_id": int(run["id"]),
        "status": run.get("status"),
        "comparison_applied": bool(run.get("comparison_applied")),
        "observations": observations[0] if observations else {},
        "counters": {
            k: run.get(k)
            for k in (
                "discovered_count",
                "new_count",
                "missing_count",
                "removed_count",
                "reappeared_count",
                "reposted_count",
            )
        },
        "note": (
            "comparison is applied during run-daily inside a transaction. This "
            "command reports the stored outcome rather than re-applying transitions, "
            "so it cannot double-count misses."
        ),
    }


async def _cmd_verify_missing(db: Any, cfg: MonitorConfig, run_id: int | None) -> dict[str, Any]:
    from .detail_verification import select_verification_targets, verify_missing
    from .models import RunContext
    from .scrapers import BamaDetailScraper

    repo = Repository(db)
    run = (
        repo.get_run(run_id)
        if run_id
        else db.fetchone("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT 1")
    )
    if not run:
        return {"error": "no such run"}
    rows = repo.advertisements_by_status([AdStatus.MISSING_ONCE, AdStatus.LIKELY_REMOVED])
    targets = select_verification_targets(rows, cfg)
    if not targets:
        return {"verified": 0, "note": "no missing advertisements to verify"}

    context = RunContext(
        run_id=int(run["id"]),
        scheduled_for=utcnow(),
        started_at=utcnow(),
        search_url=cfg.search_url,
        timezone=cfg.timezone,
        scraper_version="verify",
        configuration_hash=cfg.configuration_hash(),
    )
    async with BamaDetailScraper(cfg) as scraper:
        outcomes = await verify_missing(scraper, targets, cfg, context)
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[str(outcome.verdict)] = counts.get(str(outcome.verdict), 0) + 1
        repo.record_verification(
            advertisement_id=outcome.advertisement_id,
            run_id=context.run_id,
            url=outcome.url,
            verdict=str(outcome.verdict),
            http_status=outcome.http_status,
            evidence=outcome.evidence,
            checked_at=context.started_at,
        )
    return {"verified": len(outcomes), "verdicts": counts}


async def _cmd_refresh_details(db: Any, cfg: MonitorConfig) -> dict[str, Any]:
    from .runner import DailyRunner

    runner = DailyRunner(cfg, db=db)
    run = db.fetchone("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT 1")
    if not run:
        return {"error": "no runs yet"}
    from .models import ComparisonResult, RunContext

    context = RunContext(
        run_id=int(run["id"]),
        scheduled_for=utcnow(),
        started_at=utcnow(),
        search_url=cfg.search_url,
        timezone=cfg.timezone,
        scraper_version="refresh",
        configuration_hash=cfg.configuration_hash(),
    )
    outcome_holder = type(
        "O", (), {"detail_success": 0, "detail_failure": 0, "notes": [], "verifications": 0}
    )()
    comparison = ComparisonResult(
        run_id=context.run_id, previous_run_id=None, applied=True, reason="refresh"
    )
    await runner._detail_phase(context, comparison, outcome_holder)  # noqa: SLF001
    return {
        "detail_success": outcome_holder.detail_success,
        "detail_failure": outcome_holder.detail_failure,
        "notes": outcome_holder.notes,
    }


def _cmd_detect_reposts(db: Any, cfg: MonitorConfig) -> dict[str, Any]:
    from .models import ComparisonResult, RunContext
    from .runner import DailyRunner

    run = db.fetchone("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT 1")
    if not run:
        return {"error": "no runs yet"}
    runner = DailyRunner(cfg, db=db)
    context = RunContext(
        run_id=int(run["id"]),
        scheduled_for=utcnow(),
        started_at=utcnow(),
        search_url=cfg.search_url,
        timezone=cfg.timezone,
        scraper_version="reposts",
        configuration_hash=cfg.configuration_hash(),
    )
    new_ids = [
        str(r["platform_ad_id"])
        for r in db.fetchall(
            "SELECT platform_ad_id FROM advertisements WHERE first_run_id=?", [context.run_id]
        )
    ]
    comparison = ComparisonResult(
        run_id=context.run_id,
        previous_run_id=None,
        applied=True,
        reason="repost pass",
        new_ids=new_ids,
    )
    outcome = type("O", (), {"reposts": 0})()
    runner._repost_phase(context, comparison, outcome)  # noqa: SLF001
    return {"reposts_detected": outcome.reposts, "candidates_considered": len(new_ids)}


def _cmd_analytics(db: Any, cfg: MonitorConfig, args: argparse.Namespace) -> int:
    view = args.view
    if view == "daily_inventory":
        _dump(daily_inventory(db, limit=args.limit))
    elif view == "time_to_disappearance":
        _dump(time_to_disappearance(db)[: args.limit])
    elif view == "fastest_disappearing":
        _dump(
            fastest_disappearing(
                db, min_confidence=cfg.scoring.ranking_min_confidence, limit=args.limit
            )
        )
    elif view == "left_truncated_excluded":
        _dump(
            left_truncated_disappearances(
                db, min_confidence=cfg.scoring.ranking_min_confidence, limit=args.limit
            )
        )
    elif view == "vehicle_durations":
        rebuild_vehicle_entities(db)
        _dump(vehicle_durations(db)[: args.limit])
    elif view == "filter_exits":
        _dump(filter_exit_report(db)[: args.limit])
    elif view == "truncation_summary":
        _dump(truncation_summary(db))
    elif view == "market":
        _dump(market_statistics(db, args.dimension))
    elif view == "cohorts":
        _dump(cohort_analysis(db))
    elif view == "survival":
        _dump(survival_dataset(db)[: args.limit])
    return 0


if __name__ == "__main__":
    sys.exit(main())
