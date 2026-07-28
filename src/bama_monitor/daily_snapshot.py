"""Daily snapshot: reconstruct one run's complete state and export it atomically.

The database stays the source of truth. These files are **immutable exports of a
selected valid run** — a convenience for analysis, never an authority.

Two properties matter more than the file format:

**No future content.** Each row references the content snapshot the run's own
detail check pointed at. A later snapshot, however much better it looks, describes
a different day.

**Atomic publication.** Everything is written into a temporary directory and
renamed into place only after validation passes. A reader that finds the directory
finds a complete, validated export; a failed run leaves no half-written day behind,
and `latest_valid.json` keeps pointing at the last good one.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .config import MonitorConfig
from .db import Database, parse_ts, utcnow
from .models import ACCOUNTED_OUTCOMES, RunHealth
from .repository import Repository

#: Files a complete daily snapshot contains.
EXPECTED_FILES: tuple[str, ...] = (
    "run_summary.json",
    "inventory.csv",
    "inventory.parquet",
    "detail_checks.csv",
    "advertisement_snapshots.parquet",
    "media.csv",
    "new_ads.csv",
    "missing_ads.csv",
    "removed_ads.csv",
    "reappeared_ads.csv",
    "filter_exits.csv",
    "reposts.csv",
    "errors.csv",
    "manifest.json",
)


@dataclass
class SnapshotValidation:
    """Why a run may or may not publish a daily snapshot."""

    valid: bool
    checks: list[dict[str, Any]]

    def failures(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["passed"]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": self.checks,
            "failed": self.failures(),
        }


def validate_run(db: Database, run_id: int) -> SnapshotValidation:
    """Decide whether a run's state is complete enough to publish.

    Every condition is checked and reported, including the ones that passed: a
    report listing only failures cannot be told from one where nothing ran.
    """
    repo = Repository(db)
    run = repo.get_run(run_id)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, blocking: bool = True) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "detail": detail, "blocking": blocking}
        )

    if run is None:
        return SnapshotValidation(
            False,
            [
                {
                    "name": "run_exists",
                    "passed": False,
                    "detail": f"run {run_id} not found",
                    "blocking": True,
                }
            ],
        )

    status = str(run.get("status"))
    add("run_status_valid", status == str(RunHealth.VALID), f"status={status}")

    discovered = int(
        db.scalar(
            "SELECT COUNT(*) FROM daily_ad_observations WHERE run_id=? AND was_seen=?",
            [run_id, True],
        )
        or 0
    )
    counts = repo.detail_check_counts(run_id)
    accounted = sum(counts.get(str(o), 0) for o in ACCOUNTED_OUTCOMES)
    retryable = counts.get("retryable_failure", 0)

    add(
        "every_advertisement_checked",
        counts.get("total", 0) >= discovered and discovered > 0,
        f"{counts.get('total', 0)} detail check(s) for {discovered} discovered advertisement(s)",
    )
    add(
        "detail_accounted_equals_discovered",
        accounted == discovered and discovered > 0,
        f"accounted {accounted} vs discovered {discovered}; a run may only publish "
        "when every advertisement's check ended in a known state",
    )
    add(
        "no_unresolved_retryable_failures",
        retryable == 0,
        f"{retryable} retryable detail failure(s) outstanding",
    )
    add(
        "comparison_applied",
        bool(run.get("comparison_applied")),
        "inventory comparison completed transactionally",
    )
    add(
        "termination_verified",
        str(run.get("termination_reason") or "").startswith("api_exhausted"),
        f"termination_reason={run.get('termination_reason')}",
    )
    add(
        "provenance_complete",
        all(run.get(f) for f in ("trigger_type", "started_at", "configuration_hash")),
        "trigger_type, started_at and configuration_hash present",
    )
    blocked = str(run.get("termination_reason") or "").startswith("blocked_http")
    add("not_blocked", not blocked, "no blocking response during discovery")

    valid = all(c["passed"] for c in checks if c["blocking"])
    return SnapshotValidation(valid, checks)


def build_daily_dataset(db: Database, run_id: int) -> pd.DataFrame:
    """One complete row per advertisement for this run.

    Deterministic: ordered by advertisement id, and every join is scoped to the
    run. The snapshot is reached **through the run's own detail check**, so a
    snapshot written after this run — by a later run, or by a manual refresh —
    can never be attached to it.
    """
    frame = pd.DataFrame(
        db.fetchall(
            "SELECT"
            "  r.id AS run_id, r.scheduled_for, r.started_at AS run_started_at,"
            "  r.finished_at AS run_finished_at, r.trigger_type, r.is_synthetic,"
            "  r.scheduled_date_paris, r.scheduled_hour_paris,"
            "  r.configuration_hash AS search_configuration_hash,"
            "  r.scraper_version, r.monitor_version,"
            "  a.platform_ad_id AS advertisement_id, a.canonical_url,"
            "  a.current_status, a.detail_availability, a.sale_label,"
            "  a.sale_confidence AS sale_evidence_score, a.vehicle_entity_id,"
            "  a.repost_parent_ad_id, a.first_seen_at, a.last_seen_at,"
            "  a.first_missing_at, a.consecutive_misses, a.published_at,"
            "  a.published_at_source, a.left_truncated,"
            "  a.eligible_for_duration_ranking, a.filter_exit_reason,"
            "  o.observed_at AS card_observed_at, o.was_seen,"
            "  o.card_title, o.card_price_raw, o.card_price_normalized,"
            "  o.card_year, o.card_mileage_normalized, o.card_location,"
            "  c.checked_at AS detail_checked_at, c.detail_http_status,"
            "  c.parser_status, c.outcome AS detail_outcome, c.content_hash,"
            "  c.content_changed, c.snapshot_id,"
            "  s.scraped_at AS snapshot_created_at, s.title, s.brand, s.model,"
            "  s.trim, s.year, s.price_normalized, s.mileage_normalized,"
            "  s.description, s.seller_type, s.seller_name, s.city, s.province,"
            "  s.body_color, s.interior_color, s.transmission, s.fuel_type,"
            "  s.engine, s.body_condition, s.chassis_condition, s.insurance,"
            "  s.published_at_raw, s.detail_page_status, s.parser_version"
            " FROM monitoring_runs r"
            " JOIN daily_ad_observations o ON o.run_id = r.id"
            " JOIN advertisements a ON a.id = o.advertisement_id"
            " LEFT JOIN advertisement_detail_checks c"
            "   ON c.run_id = r.id AND c.advertisement_id = a.id"
            " LEFT JOIN advertisement_snapshots s ON s.id = c.snapshot_id"
            " WHERE r.id = ?"
            " ORDER BY a.id",
            [run_id],
        )
    )
    return frame


#: Columns an event export always carries, so a day with no removals writes a
#: header rather than an empty file. A zero-byte CSV cannot be told apart from a
#: broken export, and "no filter exits today" is a real finding worth stating.
EVENT_COLUMNS: tuple[str, ...] = (
    "platform_ad_id",
    "canonical_url",
    "event_type",
    "previous_status",
    "new_status",
    "event_at",
    "evidence_json",
)


def _iso(value: Any) -> str | None:
    """ISO text, or a real JSON null.

    ``str(None)`` yields the string ``"None"``, which a downstream reader parses
    as a present value. A manual run genuinely has no slot, and the export has to
    say so in a way that survives the round trip.
    """
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _write_frame(
    frame: pd.DataFrame,
    path: Path,
    *,
    parquet: bool = False,
    columns: tuple[str, ...] | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns:
        # An empty query result has no columns at all, so the schema is pinned
        # here instead of being inferred from rows that do not exist.
        frame = pd.DataFrame(columns=list(columns))
    if frame.empty and not parquet:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return 0
    if parquet:
        safe = frame.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                types = {type(v) for v in safe[column].dropna().head(200)}
                if len(types) > 1:
                    safe[column] = safe[column].astype(str).where(safe[column].notna())
        safe.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return int(len(frame))


def _events_frame(db: Database, run_id: int, event_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        db.fetchall(
            "SELECT a.platform_ad_id, a.canonical_url, e.event_type,"
            " e.previous_status, e.new_status, e.event_at, e.evidence_json"
            " FROM advertisement_events e"
            " JOIN advertisements a ON a.id = e.advertisement_id"
            " WHERE e.run_id=? AND e.event_type=? ORDER BY a.platform_ad_id",
            [run_id, event_type],
        )
    )


def snapshot_directory(cfg: MonitorConfig, run: dict[str, Any]) -> Path:
    """``daily_snapshots/YYYY-MM-DD/HH-MM_Europe-Paris/run_<id>/``.

    Partitioned by the **scheduled slot** in Paris terms, because that is the
    dataset's identity. The real observation timestamps live inside the files;
    the directory name is a label, not a claim that anything happened at 13:00.
    """
    paris = ZoneInfo(cfg.timezone)
    slot = parse_ts(run.get("scheduled_for"))
    date_label = run.get("scheduled_date_paris")
    hour_label = run.get("scheduled_hour_paris")
    if slot is not None:
        local = slot.astimezone(paris)
        date_label = date_label or local.date().isoformat()
        hour_label = hour_label or local.strftime("%H-%M")
    else:
        started = parse_ts(run.get("started_at")) or utcnow()
        local = started.astimezone(paris)
        date_label = date_label or local.date().isoformat()
        hour_label = hour_label or f"manual-{local.strftime('%H-%M')}"
    zone_label = cfg.timezone.replace("/", "-")
    return (
        cfg.daily_snapshot_dir
        / str(date_label)
        / f"{hour_label}_{zone_label}"
        / f"run_{int(run['id'])}"
    )


def export_daily_snapshot(
    db: Database,
    cfg: MonitorConfig,
    run_id: int,
    *,
    require_valid: bool = True,
) -> dict[str, Any]:
    """Write the day's export, atomically, and update the latest-valid pointer.

    Writes into a temporary directory beside the destination and renames it in
    only after validation passes. ``latest_valid.json`` is advanced only for a run
    that passed every blocking check — a partial run never replaces the mother
    dataset.
    """
    repo = Repository(db)
    run = repo.get_run(run_id)
    if run is None:
        raise ValueError(f"run {run_id} does not exist")

    validation = validate_run(db, run_id)
    destination = snapshot_directory(cfg, run)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if require_valid and not validation.valid:
        return {
            "exported": False,
            "run_id": run_id,
            "validation": validation.as_dict(),
            "reason": (
                "run did not pass validation; no snapshot published and the "
                "latest-valid pointer is unchanged"
            ),
            "destination_would_be": str(destination),
        }

    staging = Path(tempfile.mkdtemp(prefix=f".staging_run{run_id}_", dir=destination.parent))
    rows: dict[str, int] = {}
    try:
        dataset = build_daily_dataset(db, run_id)
        rows["inventory"] = _write_frame(dataset, staging / "inventory.csv")
        _write_frame(dataset, staging / "inventory.parquet", parquet=True)

        checks = pd.DataFrame(repo.detail_checks_for_run(run_id))
        rows["detail_checks"] = _write_frame(checks, staging / "detail_checks.csv")

        snapshots = pd.DataFrame(
            db.fetchall(
                "SELECT s.* FROM advertisement_snapshots s"
                " JOIN advertisement_detail_checks c ON c.snapshot_id = s.id"
                " WHERE c.run_id=? ORDER BY s.id",
                [run_id],
            )
        )
        if not snapshots.empty:
            snapshots = snapshots.drop(columns=["attributes_json"], errors="ignore")
        rows["advertisement_snapshots"] = _write_frame(
            snapshots, staging / "advertisement_snapshots.parquet", parquet=True
        )

        media = pd.DataFrame(
            db.fetchall(
                "SELECT a.platform_ad_id, s.media_json, s.scraped_at"
                " FROM advertisement_snapshots s"
                " JOIN advertisement_detail_checks c ON c.snapshot_id = s.id"
                " JOIN advertisements a ON a.id = s.advertisement_id"
                " WHERE c.run_id=? ORDER BY a.platform_ad_id",
                [run_id],
            )
        )
        rows["media"] = _write_frame(media, staging / "media.csv")

        for name, event_type in (
            ("new_ads", "discovered"),
            ("missing_ads", "missing_first_time"),
            ("removed_ads", "removal_confirmed"),
            ("reappeared_ads", "reappeared"),
            ("filter_exits", "filter_exit"),
            ("reposts", "possible_repost"),
        ):
            rows[name] = _write_frame(
                _events_frame(db, run_id, event_type),
                staging / f"{name}.csv",
                columns=EVENT_COLUMNS,
            )

        errors = pd.DataFrame(
            db.fetchall(
                "SELECT stage, url, error_type, error_message, retryable, attempt,"
                " created_at FROM scrape_errors WHERE run_id=? ORDER BY id",
                [run_id],
            )
        )
        rows["errors"] = _write_frame(errors, staging / "errors.csv")

        summary = _run_summary(db, run, validation, rows)
        (staging / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        manifest = {
            "run_id": run_id,
            "generated_at": utcnow().isoformat(),
            "destination": str(destination),
            "rows": rows,
            "files": sorted(p.name for p in staging.iterdir()),
            "validation": validation.as_dict(),
            "monitor_version": run.get("monitor_version"),
            "scraper_version": run.get("scraper_version"),
            "note": (
                "immutable export of one run. The database remains the source of "
                "truth; these files are a convenience, not an authority."
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        missing = [f for f in EXPECTED_FILES if not (staging / f).exists()]
        if missing:
            raise RuntimeError(f"export incomplete, missing: {missing}")

        # Atomic publication: the destination appears complete or not at all.
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    repo.update_run(run_id, snapshot_export_path=str(destination))
    pointer = None
    if validation.valid and not bool(run.get("is_synthetic")):
        pointer = _update_latest_pointer(cfg, run, destination, rows)

    return {
        "exported": True,
        "run_id": run_id,
        "destination": str(destination),
        "rows": rows,
        "validation": validation.as_dict(),
        "latest_valid_updated": pointer is not None,
        "latest_valid_path": str(pointer) if pointer else None,
    }


def _run_summary(
    db: Database, run: dict[str, Any], validation: SnapshotValidation, rows: dict[str, int]
) -> dict[str, Any]:
    repo = Repository(db)
    counts = repo.detail_check_counts(int(run["id"]))
    return {
        "run_id": int(run["id"]),
        "trigger_type": run.get("trigger_type"),
        "is_synthetic": bool(run.get("is_synthetic")),
        "production_schedule_name": run.get("production_schedule_name"),
        "status": run.get("status"),
        "timestamps": {
            "scheduled_for": _iso(run.get("scheduled_for")),
            "scheduled_date_paris": run.get("scheduled_date_paris"),
            "scheduled_hour_paris": run.get("scheduled_hour_paris"),
            "run_started_at": _iso(run.get("started_at")),
            "discovery_started_at": _iso(run.get("discovery_started_at")),
            "discovery_finished_at": _iso(run.get("discovery_finished_at")),
            "detail_started_at": _iso(run.get("detail_started_at")),
            "detail_finished_at": _iso(run.get("detail_finished_at")),
            "run_finished_at": _iso(run.get("finished_at")),
            "note": (
                "a daily snapshot is not one instant: discovery and detail scraping "
                "span the phases above. The slot is the partition label; the real "
                "observation times are on every row."
            ),
        },
        "host": {
            "host_name": run.get("host_name"),
            "process_id": run.get("process_id"),
            "scheduler_instance_id": run.get("scheduler_instance_id"),
        },
        "counters": {
            key: run.get(key)
            for key in (
                "discovered_count",
                "new_count",
                "active_count",
                "missing_count",
                "removed_count",
                "reappeared_count",
                "reposted_count",
                "detail_accounted_count",
                "detail_success_count",
                "detail_gone_count",
                "detail_permanent_failure_count",
                "detail_retryable_failure_count",
                "detail_coverage_rate",
                "duplicate_count",
                "error_count",
            )
        },
        "detail_check_outcomes": counts,
        "exported_rows": rows,
        "validation": validation.as_dict(),
        "search_configuration_hash": run.get("configuration_hash"),
    }


def _update_latest_pointer(
    cfg: MonitorConfig, run: dict[str, Any], destination: Path, rows: dict[str, int]
) -> Path:
    """Advance ``latest_valid.json`` — only ever called for a validated run."""
    pointer = cfg.daily_snapshot_dir / "latest_valid.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": int(run["id"]),
        "trigger_type": run.get("trigger_type"),
        "scheduled_for": _iso(run.get("scheduled_for")),
        "scheduled_date_paris": run.get("scheduled_date_paris"),
        "path": str(destination),
        "advertisements": rows.get("inventory"),
        "detail_checks": rows.get("detail_checks"),
        "updated_at": utcnow().isoformat(),
        "in_genuine_history": str(run.get("trigger_type")) in ("scheduled", "catch_up")
        and not run.get("is_synthetic"),
        "note": (
            "points at the most recent run that passed every blocking validity "
            "check. A partial or failed run never advances this pointer. Check "
            "`in_genuine_history`: a manual or test run can be complete and valid "
            "and still not be part of the daily scheduled series."
        ),
    }
    # Write-then-rename so a reader never sees a half-written pointer.
    staging = pointer.with_suffix(".json.tmp")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, pointer)
    return pointer


def read_latest_pointer(cfg: MonitorConfig) -> dict[str, Any] | None:
    pointer = cfg.daily_snapshot_dir / "latest_valid.json"
    if not pointer.exists():
        return None
    return json.loads(pointer.read_text(encoding="utf-8"))


def genuine_history(db: Database, *, include_synthetic: bool = False) -> pd.DataFrame:
    """Runs that count as daily history.

    Synthetic runs and non-production triggers are excluded by default: a manual
    run at 10:30 is real work, but it is not the day's scheduled observation, and
    counting it as one would overstate how much history exists.

    Runs declined for lock contention are excluded on the same principle. They
    carry a genuine trigger — the schedule really did fire — but they observed
    nothing, so listing one as a monitored day would claim a day of data that was
    never collected.
    """
    clause = "" if include_synthetic else " AND is_synthetic = {{FALSE}}"
    clause += " AND status <> 'skipped_due_to_existing_run'"
    return pd.DataFrame(
        db.fetchall(
            "SELECT id, scheduled_for, scheduled_date_paris, started_at, finished_at,"
            " trigger_type, is_synthetic, status, discovered_count,"
            " detail_accounted_count, detail_coverage_rate"
            " FROM monitoring_runs"
            f" WHERE trigger_type IN ('scheduled','catch_up'){clause}"
            " ORDER BY scheduled_for",
        )
    )


def first_genuine_run(db: Database) -> dict[str, Any] | None:
    return db.fetchone(
        "SELECT * FROM monitoring_runs"
        " WHERE trigger_type IN ('scheduled','catch_up')"
        " AND is_synthetic = {{FALSE}}"
        " AND status <> 'skipped_due_to_existing_run'"
        " ORDER BY scheduled_for LIMIT 1"
    )


def describe_history_boundary(db: Database, cfg: MonitorConfig) -> dict[str, Any]:
    """Where genuine scheduled history starts, and what came before it.

    Two boundaries are reported because they can legitimately differ:

    ``first_genuine_run_id``
        the earliest run with a production trigger — when automation began.
    ``first_published_run_id``
        the run recorded in ``production_schedule`` when a day first passed every
        validity check — when the mother dataset began.

    A genuine run that failed its gate advances the first but not the second, and
    collapsing them into one number would hide exactly that case.
    """
    repo = Repository(db)
    schedule = repo.get_production_schedule(cfg.production_schedule_name)
    first = first_genuine_run(db)
    pre = db.fetchall(
        "SELECT id, trigger_type, is_synthetic, started_at FROM monitoring_runs"
        " WHERE trigger_type NOT IN ('scheduled','catch_up') OR is_synthetic = {{TRUE}}"
        " ORDER BY id"
    )
    return {
        "production_schedule_name": cfg.production_schedule_name,
        "production_schedule_started_at": (
            schedule.get("production_schedule_started_at") if schedule else None
        ),
        "first_genuine_run_id": int(first["id"]) if first else None,
        "first_genuine_slot": str(first.get("scheduled_for")) if first else None,
        "first_published_run_id": (
            schedule.get("first_genuine_scheduled_run_id") if schedule else None
        ),
        "genuine_scheduled_runs": int(len(genuine_history(db))),
        "pre_deployment_runs": [
            {
                "run_id": int(r["id"]),
                "trigger_type": r.get("trigger_type"),
                "is_synthetic": bool(r.get("is_synthetic")),
                "started_at": str(r.get("started_at")),
            }
            for r in pre
        ],
        "interpretation": (
            "runs listed under pre_deployment_runs are real work but not daily "
            "history: they were executed by hand or for testing, and none of them "
            "is an observation of a scheduled slot."
        ),
    }


def latest_valid_run_id(db: Database) -> int | None:
    row = db.fetchone(
        "SELECT id FROM monitoring_runs WHERE status=? AND is_synthetic = {{FALSE}}"
        " ORDER BY COALESCE(finished_at, started_at) DESC LIMIT 1",
        [str(RunHealth.VALID)],
    )
    return int(row["id"]) if row else None


def utc_and_local(instant: datetime, timezone: str) -> dict[str, str]:
    """One instant expressed three ways, for reports that humans read."""
    return {
        "utc": instant.astimezone(ZoneInfo("UTC")).isoformat(),
        timezone.lower().replace("/", "_"): instant.astimezone(ZoneInfo(timezone)).isoformat(),
        "local_machine": instant.astimezone().isoformat(),
    }
