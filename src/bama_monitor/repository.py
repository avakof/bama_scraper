"""Persistence for monitoring entities.

All SQL lives here so the workflow modules stay readable and the idempotency
guarantees are enforced in one place. Every write that a replayed run could
duplicate goes through an ``ON CONFLICT`` path keyed on the unique constraints
declared in the migration.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .db import Database, parse_ts, utcnow
from .models import (
    AdStatus,
    DiscoveredCard,
    EventType,
    RunHealth,
)


class Repository:
    """Data access for runs, advertisements, observations, events and evidence."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # -- runs --------------------------------------------------------------

    def create_or_get_run(
        self,
        *,
        search_url: str,
        scheduled_for: datetime | None,
        timezone: str,
        configuration_hash: str,
        scraper_version: str,
        monitor_version: str,
        previous_valid_run_id: int | None,
        trigger_type: str = "manual",
        is_synthetic: bool = False,
        production_schedule_name: str | None = None,
        host_name: str | None = None,
        process_id: int | None = None,
        scheduler_instance_id: str | None = None,
        scheduled_date_paris: str | None = None,
        scheduled_hour_paris: str | None = None,
    ) -> tuple[int, bool]:
        """Create the run row, or resume the existing **production** run for a slot.

        The idempotency key is ``(production_schedule_name, configuration_hash,
        scheduled_for)`` and it applies **only to genuine production triggers**.

        This scoping is the fix for a real defect. The key used to be
        ``(scheduled_for, configuration_hash)`` for every run, so a manual
        execution attributed to today's slot was indistinguishable from the day's
        scheduled run: the scheduler would find it, "resume" it, and the slot would
        appear satisfied by a run that fired two and a half hours early. Manual,
        test, backfill and simulation runs now never match a production lookup and
        never occupy a slot, whatever their ``scheduled_for`` says.

        Returns ``(run_id, created)``.
        """
        is_production = trigger_type in ("scheduled", "catch_up") and not is_synthetic

        if is_production and scheduled_for is not None:
            existing = self.db.fetchone(
                "SELECT id FROM monitoring_runs"
                " WHERE scheduled_for=? AND configuration_hash=?"
                "   AND trigger_type IN ('scheduled','catch_up')"
                "   AND is_synthetic = ?"
                "   AND status <> 'skipped_due_to_existing_run'"
                "   AND production_schedule_name {{EQ}} ?",
                [scheduled_for, configuration_hash, False, production_schedule_name],
            )
            if existing:
                return int(existing["id"]), False

        now = utcnow()
        run_id = self.db.insert_returning_id(
            "INSERT INTO monitoring_runs (search_url, started_at, scheduled_for, timezone,"
            " status, configuration_hash, scraper_version, monitor_version,"
            " previous_valid_run_id, trigger_type, is_synthetic,"
            " production_schedule_name, host_name, process_id, scheduler_instance_id,"
            " scheduled_date_paris, scheduled_hour_paris, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                search_url,
                now,
                scheduled_for,
                timezone,
                str(RunHealth.RUNNING),
                configuration_hash,
                scraper_version,
                monitor_version,
                previous_valid_run_id,
                trigger_type,
                is_synthetic,
                production_schedule_name if is_production else None,
                host_name,
                process_id,
                scheduler_instance_id,
                scheduled_date_paris,
                scheduled_hour_paris,
                now,
            ],
        )
        return run_id, True

    def genuine_production_run_for_slot(
        self,
        *,
        scheduled_for: datetime,
        configuration_hash: str,
        production_schedule_name: str | None,
    ) -> dict[str, Any] | None:
        """The genuine production run for a slot, if one exists.

        Used by startup reconciliation to decide whether a slot still needs a run.
        A manual run attributed to the same slot deliberately does not count, and
        neither does a run that was declined for lock contention: it recorded that
        the trigger fired, not that the market was observed.
        """
        return self.db.fetchone(
            "SELECT * FROM monitoring_runs"
            " WHERE scheduled_for=? AND configuration_hash=?"
            "   AND trigger_type IN ('scheduled','catch_up')"
            "   AND is_synthetic = ?"
            "   AND status <> 'skipped_due_to_existing_run'"
            "   AND production_schedule_name {{EQ}} ?",
            [scheduled_for, configuration_hash, False, production_schedule_name],
        )

    def update_run(self, run_id: int, **fields: Any) -> None:
        if not fields:
            return
        json_columns = {"evidence_path"}
        assignments = ",".join(f"{k}=?" for k in fields)
        params: list[Any] = []
        for key, value in fields.items():
            if key not in json_columns and isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, default=str)
            params.append(value)
        params.append(run_id)
        self.db.execute(f"UPDATE monitoring_runs SET {assignments} WHERE id=?", params)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM monitoring_runs WHERE id=?", [run_id])

    def last_valid_run(self, configuration_hash: str) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM monitoring_runs WHERE status=? AND configuration_hash=?"
            " ORDER BY scheduled_for DESC LIMIT 1",
            [str(RunHealth.VALID), configuration_hash],
        )

    def running_runs(self, exclude_run_id: int | None = None) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT id, started_at FROM monitoring_runs WHERE status=?", [str(RunHealth.RUNNING)]
        )
        return [r for r in rows if exclude_run_id is None or int(r["id"]) != exclude_run_id]

    # -- daily detail checks -----------------------------------------------

    def record_detail_check(self, record: dict[str, Any]) -> int | None:
        """Record that one advertisement was checked on one run.

        Idempotent on ``(run_id, advertisement_id)``: replaying a run updates the
        existing row rather than inflating the coverage count.

        This row is the *proof of checking*. It is deliberately separate from a
        content snapshot, which is written only when content changes — without it,
        "no snapshot today" cannot be distinguished from "never looked at today",
        and a daily-coverage guarantee would be unverifiable.
        """
        payload = dict(record)
        payload.setdefault("created_at", utcnow())
        columns = list(payload)
        self.db.upsert(
            "advertisement_detail_checks",
            ["run_id", "advertisement_id"],
            payload,
            update_columns=[
                c for c in columns if c not in ("run_id", "advertisement_id", "created_at")
            ],
        )
        row = self.db.fetchone(
            "SELECT id FROM advertisement_detail_checks WHERE run_id=? AND advertisement_id=?",
            [payload["run_id"], payload["advertisement_id"]],
        )
        return int(row["id"]) if row else None

    def detail_check_counts(self, run_id: int) -> dict[str, int]:
        """Per-outcome tallies for one run, for the validity gate."""
        rows = self.db.fetchall(
            "SELECT outcome, COUNT(*) AS n FROM advertisement_detail_checks"
            " WHERE run_id=? GROUP BY outcome",
            [run_id],
        )
        counts = {str(r["outcome"]): int(r["n"]) for r in rows}
        counts["total"] = sum(counts.values())
        return counts

    def latest_snapshot_for(
        self, advertisement_id: int, content_hash: str | None
    ) -> dict[str, Any] | None:
        """An existing snapshot with this content hash, if one exists.

        Lets an unchanged advertisement's daily check reference the immutable
        snapshot it already matches instead of storing the payload again.
        """
        if not content_hash:
            return None
        return self.db.fetchone(
            "SELECT * FROM advertisement_snapshots"
            " WHERE advertisement_id=? AND content_hash=?"
            " ORDER BY scraped_at DESC, id DESC LIMIT 1",
            [advertisement_id, content_hash],
        )

    def detail_checks_for_run(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT c.*, a.platform_ad_id, a.canonical_url"
            " FROM advertisement_detail_checks c"
            " JOIN advertisements a ON a.id = c.advertisement_id"
            " WHERE c.run_id=? ORDER BY a.platform_ad_id",
            [run_id],
        )

    # -- production schedule -----------------------------------------------

    def upsert_production_schedule(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        now = utcnow()
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        self.db.upsert(
            "production_schedule",
            ["name"],
            payload,
            update_columns=[c for c in payload if c not in ("name", "created_at")],
        )

    def get_production_schedule(self, name: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM production_schedule WHERE name=?", [name])

    def record_missed_slot(self, record: dict[str, Any]) -> None:
        """Record a slot the machine was not available for.

        Written rather than fabricated: the absence of an observation is itself a
        fact about the pipeline, and inventing the missing snapshot would be worse
        than having a gap.
        """
        payload = dict(record)
        payload.setdefault("created_at", utcnow())
        self.db.upsert(
            "missed_schedule_slots",
            ["production_schedule_name", "scheduled_for"],
            payload,
            update_columns=[
                c
                for c in payload
                if c not in ("production_schedule_name", "scheduled_for", "created_at")
            ],
        )

    # -- advertisements ----------------------------------------------------

    def get_advertisement(
        self, platform_ad_id: str, platform: str = "bama"
    ) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM advertisements WHERE platform=? AND platform_ad_id=?",
            [platform, platform_ad_id],
        )

    def upsert_advertisement(
        self, *, platform_ad_id: str, canonical_url: str, platform: str = "bama", **fields: Any
    ) -> int:
        """Insert or update one advertisement, returning its surrogate id."""
        existing = self.get_advertisement(platform_ad_id, platform)
        now = utcnow()
        if existing is None:
            payload = {
                "platform": platform,
                "platform_ad_id": platform_ad_id,
                "canonical_url": canonical_url,
                "current_status": str(fields.pop("current_status", AdStatus.NEW)),
                "created_at": now,
                "updated_at": now,
            }
            payload.update(self._normalize_ad_fields(fields))
            columns = list(payload)
            sql = (
                f"INSERT INTO advertisements ({','.join(columns)})"
                f" VALUES ({','.join('?' * len(columns))})"
            )
            return self.db.insert_returning_id(sql, [payload[c] for c in columns])

        updates = self._normalize_ad_fields(fields)
        updates["updated_at"] = now
        updates["canonical_url"] = canonical_url
        assignments = ",".join(f"{k}=?" for k in updates)
        self.db.execute(
            f"UPDATE advertisements SET {assignments} WHERE id=?",
            [*updates.values(), existing["id"]],
        )
        return int(existing["id"])

    @staticmethod
    def _normalize_ad_fields(fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, AdStatus):
                out[key] = str(value)
            elif isinstance(value, (dict, list)):
                out[key] = json.dumps(value, ensure_ascii=False, default=str)
            else:
                out[key] = value
        return out

    def comparison_population(self, configuration_hash: str) -> list[dict[str, Any]]:
        """Advertisements eligible for an absence check.

        Restricted to the same search configuration and to states that are still
        expected inside the filter. Terminal states are excluded so a long-removed
        listing is not re-counted as missing every day.
        """
        statuses = [
            str(AdStatus.NEW),
            str(AdStatus.ACTIVE),
            str(AdStatus.REAPPEARED),
            str(AdStatus.MISSING_ONCE),
        ]
        placeholders = ",".join("?" * len(statuses))
        return self.db.fetchall(
            "SELECT * FROM advertisements"
            f" WHERE current_status IN ({placeholders})"
            " AND (search_configuration_hash IS NULL OR search_configuration_hash = ?)",
            [*statuses, configuration_hash],
        )

    def advertisements_by_status(self, statuses: list[AdStatus]) -> list[dict[str, Any]]:
        values = [str(s) for s in statuses]
        placeholders = ",".join("?" * len(values))
        return self.db.fetchall(
            f"SELECT * FROM advertisements WHERE current_status IN ({placeholders})", values
        )

    def count_by_status(self) -> dict[str, int]:
        rows = self.db.fetchall(
            "SELECT current_status, COUNT(*) AS n FROM advertisements GROUP BY current_status"
        )
        return {str(r["current_status"]): int(r["n"]) for r in rows}

    # -- observations ------------------------------------------------------

    def record_observation(
        self,
        *,
        run_id: int,
        advertisement_id: int,
        was_seen: bool,
        observed_at: datetime,
        card: DiscoveredCard | None = None,
    ) -> None:
        """Append one observation. Unique on ``(run_id, advertisement_id)``.

        Existing rows are never overwritten with different content: a replay of the
        same run writes the same values, and history is append-only by design.
        """
        record: dict[str, Any] = {
            "run_id": run_id,
            "advertisement_id": advertisement_id,
            "was_seen": was_seen,
            "observed_at": observed_at,
        }
        if card is not None:
            record.update(
                {
                    "position": card.position,
                    "page_number": card.page_number,
                    "scroll_cycle": card.scroll_cycle,
                    "card_title": card.card_title,
                    "card_price_raw": card.card_price_raw,
                    "card_price_normalized": card.card_price_normalized,
                    "card_year": card.card_year,
                    "card_mileage_raw": card.card_mileage_raw,
                    "card_mileage_normalized": card.card_mileage_normalized,
                    "card_location": card.card_location,
                    "card_image_url": card.card_image_url,
                    "card_hash": card.card_hash(),
                }
            )
        self.db.upsert(
            "daily_ad_observations",
            ["run_id", "advertisement_id"],
            record,
            update_columns=[],  # DO NOTHING: observations are immutable
        )

    def last_observation(self, advertisement_id: int) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM daily_ad_observations WHERE advertisement_id=?"
            " ORDER BY observed_at DESC, id DESC LIMIT 1",
            [advertisement_id],
        )

    def previous_seen_observation(
        self, advertisement_id: int, before_run_id: int
    ) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM daily_ad_observations WHERE advertisement_id=? AND was_seen=?"
            " AND run_id <> ? ORDER BY observed_at DESC, id DESC LIMIT 1",
            [advertisement_id, True, before_run_id],
        )

    # -- events ------------------------------------------------------------

    def record_event(
        self,
        *,
        advertisement_id: int,
        run_id: int | None,
        event_type: EventType,
        previous_status: AdStatus | None,
        new_status: AdStatus | None,
        event_at: datetime,
        evidence: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> None:
        """Append an immutable event; idempotent per (ad, run, event_type)."""
        self.db.upsert(
            "advertisement_events",
            ["advertisement_id", "run_id", "event_type"],
            {
                "advertisement_id": advertisement_id,
                "run_id": run_id,
                "event_type": str(event_type),
                "previous_status": str(previous_status) if previous_status else None,
                "new_status": str(new_status) if new_status else None,
                "event_at": event_at,
                "evidence_json": self.db.json_dump(evidence),
                "confidence": confidence,
                "created_at": utcnow(),
            },
            update_columns=[],
        )

    def events_for_run(
        self, run_id: int, event_type: EventType | None = None
    ) -> list[dict[str, Any]]:
        if event_type is None:
            return self.db.fetchall(
                "SELECT e.*, a.platform_ad_id, a.canonical_url FROM advertisement_events e"
                " JOIN advertisements a ON a.id = e.advertisement_id WHERE e.run_id=?",
                [run_id],
            )
        return self.db.fetchall(
            "SELECT e.*, a.platform_ad_id, a.canonical_url FROM advertisement_events e"
            " JOIN advertisements a ON a.id = e.advertisement_id"
            " WHERE e.run_id=? AND e.event_type=?",
            [run_id, str(event_type)],
        )

    # -- snapshots ---------------------------------------------------------

    def latest_snapshot(self, advertisement_id: int) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM advertisement_snapshots WHERE advertisement_id=?"
            " ORDER BY scraped_at DESC, id DESC LIMIT 1",
            [advertisement_id],
        )

    def insert_snapshot(self, record: dict[str, Any]) -> int | None:
        """Insert a snapshot unless an identical one already exists for this run."""
        payload = dict(record)
        payload.setdefault("created_at", utcnow())
        for key in ("attributes_json", "media_json"):
            if key in payload and not isinstance(payload[key], str):
                payload[key] = self.db.json_dump(payload[key])
        existing = self.db.fetchone(
            "SELECT id FROM advertisement_snapshots"
            " WHERE advertisement_id=? AND run_id {{EQ}} ? AND content_hash {{EQ}} ?",
            [payload.get("advertisement_id"), payload.get("run_id"), payload.get("content_hash")],
        )
        if existing:
            return int(existing["id"])
        columns = list(payload)
        sql = (
            f"INSERT INTO advertisement_snapshots ({','.join(columns)})"
            f" VALUES ({','.join('?' * len(columns))})"
        )
        return self.db.insert_returning_id(sql, [payload[c] for c in columns])

    # -- price history -----------------------------------------------------

    def record_price_change(
        self,
        *,
        advertisement_id: int,
        run_id: int | None,
        change_type: str,
        old_price: int | None,
        new_price: int | None,
        old_price_raw: str | None,
        new_price_raw: str | None,
        changed_at: datetime,
    ) -> None:
        absolute = None
        percentage = None
        if old_price is not None and new_price is not None:
            absolute = new_price - old_price
            if old_price:
                percentage = round(100.0 * absolute / old_price, 4)
        self.db.upsert(
            "price_changes",
            ["advertisement_id", "run_id", "change_type"],
            {
                "advertisement_id": advertisement_id,
                "run_id": run_id,
                "change_type": change_type,
                "old_price": old_price,
                "new_price": new_price,
                "absolute_change": absolute,
                "percentage_change": percentage,
                "old_price_raw": old_price_raw,
                "new_price_raw": new_price_raw,
                "changed_at": changed_at,
                "created_at": utcnow(),
            },
            update_columns=[],
        )

    def price_changes_for_run(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT p.*, a.platform_ad_id, a.canonical_url FROM price_changes p"
            " JOIN advertisements a ON a.id = p.advertisement_id WHERE p.run_id=?",
            [run_id],
        )

    def recent_price_reduction(self, advertisement_id: int) -> bool:
        row = self.db.fetchone(
            "SELECT 1 AS hit FROM price_changes WHERE advertisement_id=?"
            " AND change_type='price_decrease' LIMIT 1",
            [advertisement_id],
        )
        return bool(row)

    # -- verification ------------------------------------------------------

    def record_verification(
        self,
        *,
        advertisement_id: int,
        run_id: int | None,
        url: str,
        verdict: str,
        http_status: int | None,
        evidence: dict[str, Any],
        checked_at: datetime,
    ) -> None:
        self.db.upsert(
            "detail_verifications",
            ["advertisement_id", "run_id"],
            {
                "advertisement_id": advertisement_id,
                "run_id": run_id,
                "url": url,
                "verdict": verdict,
                "http_status": http_status,
                "evidence_json": self.db.json_dump(evidence),
                "checked_at": checked_at,
            },
        )

    def consecutive_unavailable_checks(self, advertisement_id: int) -> int:
        """How many of the most recent checks in a row said 404/410."""
        rows = self.db.fetchall(
            "SELECT verdict FROM detail_verifications WHERE advertisement_id=?"
            " ORDER BY checked_at DESC LIMIT 10",
            [advertisement_id],
        )
        count = 0
        for row in rows:
            if str(row["verdict"]) in ("http_404", "http_410"):
                count += 1
            else:
                break
        return count

    # -- reposts -----------------------------------------------------------

    def record_repost_link(
        self,
        *,
        parent_ad_id: int,
        child_ad_id: int,
        run_id: int | None,
        score: float,
        matched_on: list[str],
        evidence: dict[str, Any],
        vehicle_entity_id: str,
    ) -> None:
        self.db.upsert(
            "repost_links",
            ["parent_ad_id", "child_ad_id"],
            {
                "parent_ad_id": parent_ad_id,
                "child_ad_id": child_ad_id,
                "run_id": run_id,
                "score": score,
                "matched_on_json": self.db.json_dump(matched_on),
                "evidence_json": self.db.json_dump(evidence),
                "vehicle_entity_id": vehicle_entity_id,
                "created_at": utcnow(),
            },
        )

    def repost_links_for_run(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT r.*, p.platform_ad_id AS parent_platform_ad_id,"
            " c.platform_ad_id AS child_platform_ad_id FROM repost_links r"
            " JOIN advertisements p ON p.id = r.parent_ad_id"
            " JOIN advertisements c ON c.id = r.child_ad_id WHERE r.run_id=?",
            [run_id],
        )

    # -- errors and alerts -------------------------------------------------

    def record_error(
        self,
        *,
        run_id: int | None,
        stage: str,
        error_type: str,
        error_message: str,
        advertisement_id: int | None = None,
        url: str | None = None,
        retryable: bool = True,
        attempt: int = 1,
        screenshot_path: str | None = None,
        html_path: str | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO scrape_errors (run_id, advertisement_id, stage, url, error_type,"
            " error_message, retryable, attempt, screenshot_path, html_path, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                run_id,
                advertisement_id,
                stage,
                url,
                error_type,
                error_message[:4000],
                retryable,
                attempt,
                screenshot_path,
                html_path,
                utcnow(),
            ],
        )

    def errors_for_run(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM scrape_errors WHERE run_id=? ORDER BY id", [run_id])

    def record_alert(
        self,
        *,
        run_id: int | None,
        severity: str,
        alert_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.db.insert_returning_id(
            "INSERT INTO alerts (run_id, severity, alert_type, message, metadata_json, created_at)"
            " VALUES (?,?,?,?,?,?)",
            [run_id, severity, alert_type, message, self.db.json_dump(metadata), utcnow()],
        )

    def alerts_for_run(self, run_id: int) -> list[dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM alerts WHERE run_id=? ORDER BY id", [run_id])

    # -- helpers -----------------------------------------------------------

    def ad_timestamps(self, advertisement_id: int) -> dict[str, datetime | None]:
        row = self.db.fetchone(
            "SELECT first_seen_at, last_seen_at, first_missing_at, confirmed_removed_at"
            " FROM advertisements WHERE id=?",
            [advertisement_id],
        )
        if not row:
            return {}
        return {k: parse_ts(v) for k, v in row.items()}
