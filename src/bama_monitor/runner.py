"""Daily run orchestration.

The sequence is fixed and each stage is separately recoverable:

    lock -> create/resume run -> discover -> persist -> validate health
         -> compare (only if valid) -> detail scrapes -> verify missing
         -> repost detection -> sale scoring -> finalize -> report

Everything after ``persist`` is conditional on health. Everything before it is
evidence collection, which happens regardless so a failed run still leaves a
diagnosable trail.
"""

from __future__ import annotations

import os
import random
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from bama_scraper.config import SCRAPER_VERSION
from bama_scraper.logging_config import get_logger

from .alerts import AlertManager
from .config import MONITOR_VERSION, MonitorConfig
from .db import Database, connect, parse_ts, utcnow
from .detail_verification import select_verification_targets, verify_missing
from .inventory_comparison import (
    ComparisonResult,
    InventoryComparator,
    finalize_run,
    skipped_run,
)
from .locking import LockUnavailable, run_lock
from .models import (
    ACCOUNTED_OUTCOMES,
    AVAILABILITY_BY_VERDICT,
    STRONG_REMOVAL_VERDICTS,
    AdStatus,
    AlertSeverity,
    DetailAvailability,
    DetailCheckOutcome,
    DetailVerdict,
    DiscoveryResult,
    EventType,
    FilterExitReason,
    RunContext,
    RunHealth,
)
from .publication import classify_published_source, to_datetime, truncation_status
from .repository import Repository
from .repost_detection import VehicleProfile, find_reposts, fingerprint, vehicle_entity_id
from .run_health import HealthReport, anomaly_alerts
from .sale_scoring import ScoringInputs, score_sale
from .scheduler import next_scheduled_slot
from .search_filter import assert_bounds_parsed, classify_absence, parse_search_filter
from .state_machine import AdState, mark_reposted, promote_to_likely_sold

log = get_logger("bama_monitor.runner")


@dataclass
class RunOutcome:
    run_id: int | None
    status: RunHealth
    health: HealthReport | None = None
    comparison: ComparisonResult | None = None
    discovery: DiscoveryResult | None = None
    detail_success: int = 0
    detail_failure: int = 0
    verifications: int = 0
    reposts: int = 0
    promoted_sold: int = 0
    #: Live listings that left the monitored search rather than the market.
    filter_exits: int = 0
    #: Absences whose live detail page matched every bound -- explained by
    #: nothing, and deliberately left that way.
    unexplained_absences: int = 0
    #: Detail checks whose page was 404/410 -- a complete answer, so accounted for.
    detail_gone: int = 0
    #: Detail checks that failed permanently with evidence -- also accounted for.
    detail_permanent_failure: int = 0
    #: completed + gone + permanent failure. Must equal the discovered count.
    detail_accounted: int = 0
    detail_coverage_rate: float | None = None
    skipped: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": str(self.status),
            "skipped": self.skipped,
            "health": self.health.as_dict() if self.health else None,
            "comparison": {
                "applied": self.comparison.applied if self.comparison else False,
                "reason": self.comparison.reason if self.comparison else None,
                **(self.comparison.counters() if self.comparison else {}),
            },
            "discovered_count": self.discovery.discovered_count if self.discovery else 0,
            "termination_reason": self.discovery.termination_reason if self.discovery else None,
            "detail_success": self.detail_success,
            "detail_failure": self.detail_failure,
            "detail_gone": self.detail_gone,
            "detail_permanent_failure": self.detail_permanent_failure,
            "detail_accounted": self.detail_accounted,
            "detail_coverage_rate": self.detail_coverage_rate,
            "missing_verifications": self.verifications,
            "filter_exits": self.filter_exits,
            "unexplained_absences": self.unexplained_absences,
            "reposts_detected": self.reposts,
            "promoted_to_likely_sold": self.promoted_sold,
            "error": self.error,
            "notes": self.notes,
        }


class DailyRunner:
    """Executes one monitoring run."""

    def __init__(
        self,
        cfg: MonitorConfig,
        *,
        db: Database | None = None,
        discovery_scraper: Any = None,
        detail_scraper_factory: Any = None,
    ) -> None:
        self.cfg = cfg
        self.cfg.ensure_dirs()
        self._own_db = db is None
        self.db = db or connect(cfg.database_url)
        self.repo = Repository(self.db)
        self.alerts = AlertManager(self.repo, cfg.alerts)
        self._discovery_scraper = discovery_scraper
        self._detail_scraper_factory = detail_scraper_factory
        # Parsed once: the bounds an absent-but-live listing is re-checked against.
        self.search_filter = parse_search_filter(cfg.search_url)
        for warning in assert_bounds_parsed(cfg.search_url, self.search_filter):
            log.warning("monitor.filter.bound_not_parsed", detail=warning)

    def close(self) -> None:
        if self._own_db:
            self.db.close()

    # -- public entry point ------------------------------------------------

    async def run_daily(
        self,
        *,
        scheduled_for: datetime | None = None,
        trigger_type: str = "manual",
        is_synthetic: bool = False,
        production_schedule_name: str | None = None,
        scheduler_instance_id: str | None = None,
    ) -> RunOutcome:
        """Run the full daily pipeline under the execution lock.

        ``trigger_type`` decides whether this run may own a production slot. A
        manual run is given no slot at all unless one is passed explicitly, so
        running the pipeline by hand can never satisfy — or block — the day's
        scheduled execution.
        """
        is_production = trigger_type in ("scheduled", "catch_up") and not is_synthetic
        slot: datetime | None = scheduled_for
        if slot is None and is_production:
            slot = next_scheduled_slot(self.cfg, reference=utcnow(), allow_past=True)
        try:
            with run_lock(self.db, self.cfg.lock_dir):
                return await self._run(
                    slot,
                    trigger_type=trigger_type,
                    is_synthetic=is_synthetic,
                    production_schedule_name=production_schedule_name if is_production else None,
                    scheduler_instance_id=scheduler_instance_id,
                )
        except LockUnavailable as exc:
            run_id = skipped_run(
                self.db,
                search_url=self.cfg.search_url,
                scheduled_for=slot,
                cfg=self.cfg,
                holder=exc.holder,
                trigger_type=trigger_type,
                is_synthetic=is_synthetic,
                production_schedule_name=production_schedule_name if is_production else None,
            )
            log.warning("monitor.run.skipped", holder=exc.holder, run_id=run_id)
            return RunOutcome(
                run_id=run_id,
                status=RunHealth.SKIPPED_DUE_TO_EXISTING_RUN,
                skipped=True,
                error=str(exc),
            )

    # -- internals ---------------------------------------------------------

    async def _run(
        self,
        slot: datetime | None,
        *,
        trigger_type: str = "manual",
        is_synthetic: bool = False,
        production_schedule_name: str | None = None,
        scheduler_instance_id: str | None = None,
    ) -> RunOutcome:
        previous = self.repo.last_valid_run(self.cfg.configuration_hash())
        paris = ZoneInfo(self.cfg.timezone)
        run_id, created = self.repo.create_or_get_run(
            search_url=self.cfg.search_url,
            scheduled_for=slot,
            timezone=self.cfg.timezone,
            configuration_hash=self.cfg.configuration_hash(),
            scraper_version=SCRAPER_VERSION,
            monitor_version=MONITOR_VERSION,
            previous_valid_run_id=int(previous["id"]) if previous else None,
            trigger_type=trigger_type,
            is_synthetic=is_synthetic,
            production_schedule_name=production_schedule_name,
            host_name=socket.gethostname(),
            process_id=os.getpid(),
            scheduler_instance_id=scheduler_instance_id,
            scheduled_date_paris=(slot.astimezone(paris).date().isoformat() if slot else None),
            scheduled_hour_paris=(slot.astimezone(paris).strftime("%H-%M") if slot else None),
        )
        # Observations are timestamped with the *scheduled slot*, not wall-clock
        # now. With a daily cadence this puts every observation on a regular
        # 24-hour grid, so duration bounds and the censoring interval are exact
        # rather than jittered by however late the run actually fired (systemd's
        # RandomizedDelaySec, or a Persistent=true catch-up after downtime). The
        # real start and finish times are still recorded on the run row, so the
        # attribution is visible rather than hidden.
        real_start = utcnow()
        # With a slot, observations sit on the 24-hour grid so duration bounds are
        # exact. Without one (a manual run), the only honest stamp is the real
        # start: there is no slot to attribute the observation to.
        observation_stamp = slot if slot is not None else real_start
        context = RunContext(
            run_id=run_id,
            scheduled_for=slot if slot is not None else real_start,
            started_at=observation_stamp,
            search_url=self.cfg.search_url,
            timezone=self.cfg.timezone,
            scraper_version=SCRAPER_VERSION,
            configuration_hash=self.cfg.configuration_hash(),
            previous_valid_run_id=int(previous["id"]) if previous else None,
        )
        outcome = RunOutcome(run_id=run_id, status=RunHealth.RUNNING)
        if not created:
            outcome.notes.append(
                "run row already existed for this scheduled slot; resuming idempotently"
            )
        self.repo.update_run(
            run_id,
            status=str(RunHealth.RUNNING),
            started_at=real_start,
            discovery_started_at=real_start,
        )

        evidence_dir = self.cfg.output_dir / "runs" / f"run_{run_id}"
        self.repo.update_run(run_id, evidence_path=str(evidence_dir))

        # --- discovery ----------------------------------------------------
        try:
            scraper = self._discovery_scraper or self._default_discovery(evidence_dir)
            discovery = await scraper.discover(self.cfg.search_url)
        except Exception as exc:  # noqa: BLE001
            self.repo.record_error(
                run_id=run_id, stage="discovery", error_type="exception", error_message=repr(exc)
            )
            self.repo.update_run(
                run_id,
                status=str(RunHealth.FAILED),
                finished_at=utcnow(),
                health_reason=f"discovery raised: {exc!r}",
            )
            self.alerts.raise_alert(
                "daily_run_failed",
                f"discovery failed: {exc!r}",
                severity=AlertSeverity.CRITICAL,
                run_id=run_id,
            )
            return RunOutcome(run_id=run_id, status=RunHealth.FAILED, error=repr(exc))

        outcome.discovery = discovery
        self.repo.update_run(run_id, discovery_finished_at=utcnow())
        comparator = InventoryComparator(self.db, self.cfg)

        # --- persist (evidence first) -------------------------------------
        persist = comparator.persist_inventory(discovery, context)
        if not persist.persisted:
            self.repo.record_error(
                run_id=run_id,
                stage="persist",
                error_type="exception",
                error_message=persist.error or "unknown",
            )

        # --- health -------------------------------------------------------
        health = comparator.validate(discovery, context, persisted=persist.persisted)
        outcome.health = health
        for payload in anomaly_alerts(health, discovery):
            self.alerts.raise_alert(
                payload["alert_type"],
                payload["message"],
                severity=payload["severity"],
                run_id=run_id,
                metadata=payload["metadata"],
            )
        if discovery.blocked:
            self.alerts.raise_alert(
                "access_blocked",
                "Bama returned a blocking response; run will not drive status changes",
                severity=AlertSeverity.CRITICAL,
                run_id=run_id,
            )
        if discovery.discovered_count == 0:
            self.alerts.raise_alert(
                "no_advertisements_found",
                "discovery returned zero advertisements",
                severity=AlertSeverity.CRITICAL,
                run_id=run_id,
            )

        # --- comparison (gated on health) ---------------------------------
        comparison = comparator.compare(discovery, context, health, persist)
        outcome.comparison = comparison
        if not comparison.applied:
            log.warning("monitor.comparison.skipped", run_id=run_id, reason=comparison.reason)
            outcome.notes.append(comparison.reason)

        # --- detail work (only meaningful for a valid run) ----------------
        if comparison.applied:
            await self._detail_phase(context, comparison, outcome)
            self._repost_phase(context, comparison, outcome)
            self._scoring_phase(context, outcome)
            self._removal_rate_alert(context, comparison)

        outcome.status = health.status
        finalize_run(
            self.db,
            context,
            health,
            comparison,
            detail_success=outcome.detail_success,
            detail_failure=outcome.detail_failure,
        )
        self.repo.update_run(
            run_id,
            likely_sold_count=outcome.promoted_sold,
            reposted_count=outcome.reposts,
        )
        return outcome

    def _default_discovery(self, evidence_dir: Path) -> Any:
        from .scrapers import BamaDiscoveryScraper

        return BamaDiscoveryScraper(self.cfg, evidence_dir)

    # -- detail phase ------------------------------------------------------

    def _default_detail_factory(self) -> Any:
        """The production detail scraper, as an async context manager."""
        from .scrapers import BamaDetailScraper

        return BamaDetailScraper(self.cfg)

    def _select_detail_targets(
        self, context: RunContext, comparison: ComparisonResult
    ) -> list[tuple[int, str, str, str]]:
        """Pick which advertisements deserve a detail fetch today.

        Scraping every listing daily would be ~2,800 requests for almost no new
        information, so the policy targets change: new listings, changed cards,
        newly missing listings, previous failures, a periodic refresh of the rest
        and a small random QC sample.
        """
        policy = self.cfg.detail
        chosen: dict[str, tuple[int, str, str, str]] = {}

        def add(row: dict[str, Any], reason: str) -> None:
            key = str(row["platform_ad_id"])
            if key not in chosen:
                chosen[key] = (
                    int(row["id"]),
                    key,
                    str(row["canonical_url"]),
                    reason,
                )

        if policy.scrape_new:
            for platform_ad_id in comparison.new_ids:
                row = self.repo.get_advertisement(platform_ad_id)
                if row:
                    add(row, "new")

        if policy.scrape_card_changed:
            changed = self.db.fetchall(
                "SELECT a.* FROM advertisements a JOIN price_changes p"
                " ON p.advertisement_id = a.id AND p.run_id = ?",
                [context.run_id],
            )
            for row in changed:
                add(row, "card_changed")

        if policy.scrape_missing:
            for platform_ad_id in comparison.newly_missing_ids:
                row = self.repo.get_advertisement(platform_ad_id)
                if row:
                    add(row, "missing")

        if policy.scrape_previous_failures:
            for row in self.db.fetchall(
                "SELECT * FROM advertisements WHERE detail_failure_count > 0"
                " AND current_status IN (?,?,?)",
                [str(AdStatus.NEW), str(AdStatus.ACTIVE), str(AdStatus.REAPPEARED)],
            ):
                add(row, "previous_failure")

        # Periodic refresh: anything whose last snapshot is older than the window.
        cutoff = context.started_at - timedelta(days=policy.active_refresh_days_max)
        stale = self.db.fetchall(
            "SELECT a.* FROM advertisements a"
            " WHERE a.current_status IN (?,?,?)"
            " AND (a.latest_snapshot_id IS NULL OR NOT EXISTS ("
            "   SELECT 1 FROM advertisement_snapshots s"
            "   WHERE s.advertisement_id = a.id AND s.scraped_at > ?))",
            [str(AdStatus.NEW), str(AdStatus.ACTIVE), str(AdStatus.REAPPEARED), cutoff],
        )
        for row in stale:
            add(row, "periodic_refresh")

        # Random QC sample of everything else still active.
        if policy.qc_sample_ratio > 0:
            active = self.db.fetchall(
                "SELECT * FROM advertisements WHERE current_status IN (?,?)",
                [str(AdStatus.ACTIVE), str(AdStatus.REAPPEARED)],
            )
            pool = [r for r in active if str(r["platform_ad_id"]) not in chosen]
            sample_size = int(len(pool) * policy.qc_sample_ratio)
            for row in random.sample(pool, min(sample_size, len(pool))):
                add(row, "qc_sample")

        targets = list(chosen.values())
        if len(targets) > policy.max_details_per_run:
            # Truncation is bounded and reported rather than silent.
            priority = {
                "new": 0,
                "missing": 1,
                "card_changed": 2,
                "previous_failure": 3,
                "periodic_refresh": 4,
                "qc_sample": 5,
            }
            targets.sort(key=lambda t: priority.get(t[3], 9))
            targets = targets[: policy.max_details_per_run]
        return targets

    async def _detail_phase(
        self, context: RunContext, comparison: ComparisonResult, outcome: RunOutcome
    ) -> None:
        """Check the detail page of **every** advertisement in today's inventory.

        This is a daily census, not an incremental refresh. The policy-driven
        selection (new / changed / missing / due-a-refresh) is still available and
        is what a manual or exploratory run uses, but a production daily run must
        be able to state that every listing was looked at *today*. Without that,
        "no change recorded" is ambiguous between "unchanged" and "not examined",
        and every downstream duration and disappearance claim inherits the
        ambiguity.

        Storage is still deduplicated: an unchanged advertisement's check
        references the existing immutable snapshot through ``snapshot_id`` rather
        than writing the payload again. Checking and storing are separate concepts.
        """
        full_census = self.cfg.detail.check_every_advertisement_daily
        targets = (
            self._all_inventory_targets(context)
            if full_census
            else self._select_detail_targets(context, comparison)
        )
        if not targets:
            return

        self.repo.update_run(context.run_id, detail_started_at=utcnow())
        factory = self._detail_scraper_factory or self._default_detail_factory
        cap = self.cfg.detail.max_details_per_run
        truncated = not full_census and len(targets) >= cap
        if truncated:
            outcome.notes.append(f"detail targets capped at {cap}")
        if full_census:
            outcome.notes.append(
                f"daily census: checking every one of {len(targets)} discovered advertisements"
            )

        blocked = False
        async with factory() as scraper:
            for advertisement_id, _platform_ad_id, url, reason in targets:
                started = utcnow()
                result = await scraper.scrape(url)
                elapsed_ms = int((utcnow() - started).total_seconds() * 1000)

                outcome_kind, snapshot_id, changed = self._apply_detail_result(
                    advertisement_id, context, result, reason
                )
                self.repo.record_detail_check(
                    {
                        "run_id": context.run_id,
                        "advertisement_id": advertisement_id,
                        "checked_at": started,
                        "detail_http_status": result.http_status,
                        "detail_availability": str(
                            AVAILABILITY_BY_VERDICT.get(
                                str(result.verdict), DetailAvailability.UNKNOWN
                            )
                        ),
                        # "parse_failed" is a claim about our parser, so it is
                        # reserved for the case where parsing was actually tried.
                        # A 404, a block or a timeout never reached the parser.
                        "parser_status": "ok"
                        if result.ok
                        else "parse_failed"
                        if result.verdict is DetailVerdict.PARSE_ERROR
                        else "not_attempted",
                        "content_hash": result.content_hash,
                        "snapshot_id": snapshot_id,
                        "content_changed": changed,
                        "attempt_count": 1,
                        "duration_ms": elapsed_ms,
                        "outcome": str(outcome_kind),
                        "error_type": None if result.ok else str(result.verdict),
                        "error_message": result.error,
                    }
                )

                if outcome_kind is DetailCheckOutcome.COMPLETED:
                    outcome.detail_success += 1
                elif outcome_kind is DetailCheckOutcome.GONE:
                    outcome.detail_gone += 1
                elif outcome_kind is DetailCheckOutcome.PERMANENT_FAILURE:
                    outcome.detail_permanent_failure += 1
                else:
                    outcome.detail_failure += 1

                if result.verdict is DetailVerdict.BLOCKED:
                    blocked = True
                    outcome.notes.append("detail phase stopped early: blocked response")
                    self.alerts.raise_alert(
                        "access_blocked",
                        "detail scraping hit a blocking response and stopped",
                        severity=AlertSeverity.CRITICAL,
                        run_id=context.run_id,
                    )
                    break

            # Verification of missing listings shares the fetcher.
            await self._verify_phase(context, comparison, outcome, scraper)

        self.repo.update_run(context.run_id, detail_finished_at=utcnow())
        self._record_detail_coverage(context, outcome, blocked=blocked)

        attempted = (
            outcome.detail_success
            + outcome.detail_failure
            + outcome.detail_gone
            + outcome.detail_permanent_failure
        )
        if attempted and outcome.detail_failure / attempted > 0.30:
            self.alerts.raise_alert(
                "detail_failure_rate_exceeded",
                f"detail failure rate {outcome.detail_failure}/{attempted}",
                severity=AlertSeverity.WARNING,
                run_id=context.run_id,
            )

    def _all_inventory_targets(self, context: RunContext) -> list[tuple[int, str, str, str]]:
        """Every advertisement observed as present in this run.

        Ordered by id so a resumed run walks the same sequence and the coverage
        count is reproducible.
        """
        rows = self.db.fetchall(
            "SELECT a.id, a.platform_ad_id, a.canonical_url"
            " FROM daily_ad_observations o"
            " JOIN advertisements a ON a.id = o.advertisement_id"
            " WHERE o.run_id = ? AND o.was_seen = ?"
            " ORDER BY a.id",
            [context.run_id, True],
        )
        return [
            (int(r["id"]), str(r["platform_ad_id"]), str(r["canonical_url"]), "daily_census")
            for r in rows
        ]

    def _apply_detail_result(
        self, advertisement_id: int, context: RunContext, result: Any, reason: str
    ) -> tuple[DetailCheckOutcome, int | None, bool]:
        """Persist one detail outcome and classify it for coverage accounting."""
        if result.ok:
            existing = self.repo.latest_snapshot_for(advertisement_id, result.content_hash)
            if existing is not None:
                # Unchanged: point today's check at the snapshot it already matches
                # rather than storing a duplicate payload.
                self.db.execute(
                    "UPDATE advertisements SET last_detail_verdict=?,"
                    " last_detail_checked_at=?, detail_failure_count=0 WHERE id=?",
                    [str(result.verdict), context.started_at, advertisement_id],
                )
                return DetailCheckOutcome.COMPLETED, int(existing["id"]), False
            snapshot_id = self._store_snapshot(advertisement_id, context, result, reason)
            return DetailCheckOutcome.COMPLETED, snapshot_id, True

        self.db.execute(
            "UPDATE advertisements SET detail_failure_count = detail_failure_count + 1,"
            " last_detail_verdict=?, last_detail_checked_at=? WHERE id=?",
            [str(result.verdict), context.started_at, advertisement_id],
        )
        retryable = result.verdict in (DetailVerdict.TEMPORARY_ERROR, DetailVerdict.BLOCKED)
        self.repo.record_error(
            run_id=context.run_id,
            advertisement_id=advertisement_id,
            stage="detail",
            url=getattr(result, "url", None),
            error_type=str(result.verdict),
            error_message=result.error or "detail scrape failed",
            retryable=retryable,
        )
        self.repo.record_event(
            advertisement_id=advertisement_id,
            run_id=context.run_id,
            event_type=EventType.DETAIL_UNAVAILABLE,
            previous_status=None,
            new_status=None,
            event_at=context.started_at,
            evidence={"verdict": str(result.verdict), "error": result.error},
        )
        if result.verdict in STRONG_REMOVAL_VERDICTS:
            # 404/410 is a complete answer: the page is gone. Accounted for.
            return DetailCheckOutcome.GONE, None, False
        if retryable:
            return DetailCheckOutcome.RETRYABLE_FAILURE, None, False
        return DetailCheckOutcome.PERMANENT_FAILURE, None, False

    def _record_detail_coverage(
        self, context: RunContext, outcome: RunOutcome, *, blocked: bool
    ) -> None:
        """Compute and store the coverage figures the validity gate reads."""
        counts = self.repo.detail_check_counts(context.run_id)
        accounted = sum(counts.get(str(o), 0) for o in ACCOUNTED_OUTCOMES)
        discovered = int(
            self.db.scalar(
                "SELECT COUNT(*) FROM daily_ad_observations WHERE run_id=? AND was_seen=?",
                [context.run_id, True],
            )
            or 0
        )
        rate = round(accounted / discovered, 6) if discovered else None
        outcome.detail_accounted = accounted
        outcome.detail_coverage_rate = rate
        self.repo.update_run(
            context.run_id,
            detail_accounted_count=accounted,
            detail_gone_count=counts.get(str(DetailCheckOutcome.GONE), 0),
            detail_permanent_failure_count=counts.get(str(DetailCheckOutcome.PERMANENT_FAILURE), 0),
            detail_retryable_failure_count=counts.get(str(DetailCheckOutcome.RETRYABLE_FAILURE), 0),
            detail_coverage_rate=rate,
        )
        if discovered and accounted < discovered and not blocked:
            outcome.notes.append(
                f"detail coverage incomplete: {accounted}/{discovered} accounted for"
            )

    def _store_snapshot(
        self, advertisement_id: int, context: RunContext, result: Any, reason: str
    ) -> int | None:
        """Write a snapshot only when the content actually changed.

        Returns the snapshot id the caller should reference — the newly written one
        when content changed, or the existing one when it did not, so today's
        detail check can point at immutable content without duplicating it.
        """
        latest = self.repo.latest_snapshot(advertisement_id)
        if latest and latest.get("content_hash") == result.content_hash:
            # Nothing changed; record that we checked without inflating history.
            self.db.execute(
                "UPDATE advertisements SET last_detail_checked_at=?, last_detail_verdict=?,"
                " detail_failure_count=0 WHERE id=?",
                [context.started_at, str(result.verdict), advertisement_id],
            )
            return int(latest["id"])

        fields = result.fields
        # The seller's publication time, kept distinct from `first_seen_at`, which
        # is only when *we* started looking.
        published_at = to_datetime(fields.get("published_ts"))
        published_source = classify_published_source(fields.get("published_text"), published_at)
        snapshot_id = self.repo.insert_snapshot(
            {
                "advertisement_id": advertisement_id,
                "run_id": context.run_id,
                "scraped_at": context.started_at,
                "title": fields.get("title"),
                "brand": fields.get("brand_fa") or fields.get("brand"),
                "model": fields.get("model_fa") or fields.get("model"),
                "trim": fields.get("trim_fa") or fields.get("trim"),
                "year": fields.get("year_text"),
                "price_raw": fields.get("price_text"),
                "price_normalized": fields.get("price_toman"),
                "mileage_raw": fields.get("mileage_text"),
                "mileage_normalized": fields.get("mileage_km"),
                "description": fields.get("description"),
                "seller_type": fields.get("seller_type"),
                "seller_name": fields.get("dealer_name"),
                "city": fields.get("city"),
                "province": fields.get("province"),
                "body_color": fields.get("body_color"),
                "interior_color": fields.get("inside_color"),
                "transmission": fields.get("transmission"),
                "fuel_type": fields.get("fuel_type"),
                "engine": fields.get("engine"),
                "body_condition": fields.get("body_status"),
                "chassis_condition": fields.get("chassis_condition"),
                "insurance": fields.get("insurance_text"),
                "published_at_raw": fields.get("published_text"),
                "published_at_normalized": published_at,
                "attributes_json": {k: v for k, v in fields.items() if not k.endswith("_json")},
                "media_json": result.media,
                "content_hash": result.content_hash,
                "detail_page_status": str(result.verdict),
                "parser_version": result.parser_version,
            }
        )
        updates: dict[str, Any] = {
            "latest_snapshot_id": snapshot_id,
            "last_detail_checked_at": context.started_at,
            "last_detail_verdict": str(result.verdict),
            "detail_failure_count": 0,
        }
        fingerprint_value = self._fingerprint_from_fields(fields, result.media)
        if fingerprint_value:
            updates["vehicle_fingerprint"] = fingerprint_value

        # Publication time and the left-truncation verdict that follows from it.
        # Only ever written when the detail page actually produced one, so a later
        # failed scrape cannot erase a good value.
        if published_at is not None:
            row = self.db.fetchone(
                "SELECT first_seen_at FROM advertisements WHERE id=?", [advertisement_id]
            )
            truncation = truncation_status(
                published_at=published_at,
                published_source=published_source,
                first_seen_at=parse_ts(row.get("first_seen_at")) if row else None,
                tolerance=timedelta(hours=self.cfg.entry_tolerance_hours),
            )
            updates.update(truncation)
        assignments = ",".join(f"{k}=?" for k in updates)
        self.db.execute(
            f"UPDATE advertisements SET {assignments} WHERE id=?",
            [*updates.values(), advertisement_id],
        )
        if latest is not None:
            self.repo.record_event(
                advertisement_id=advertisement_id,
                run_id=context.run_id,
                event_type=EventType.ATTRIBUTES_CHANGED,
                previous_status=None,
                new_status=None,
                event_at=context.started_at,
                evidence={
                    "reason": reason,
                    "old_content_hash": latest.get("content_hash"),
                    "new_content_hash": result.content_hash,
                },
            )
            self._detail_price_change(advertisement_id, context, latest, fields)
        return snapshot_id

    def _detail_price_change(
        self,
        advertisement_id: int,
        context: RunContext,
        latest: dict[str, Any],
        fields: dict[str, Any],
    ) -> None:
        old_price = latest.get("price_normalized")
        new_price = fields.get("price_toman")
        if old_price == new_price:
            return
        change_type = (
            "price_increase"
            if old_price is not None and new_price is not None and new_price > old_price
            else "price_decrease"
            if old_price is not None and new_price is not None
            else "price_hidden"
            if new_price is None
            else "price_revealed"
        )
        self.repo.record_price_change(
            advertisement_id=advertisement_id,
            run_id=context.run_id,
            change_type=f"detail_{change_type}",
            old_price=old_price,
            new_price=new_price,
            old_price_raw=latest.get("price_raw"),
            new_price_raw=fields.get("price_text"),
            changed_at=context.started_at,
        )

    def _fingerprint_from_fields(
        self, fields: dict[str, Any], media: list[dict[str, Any]]
    ) -> str | None:
        profile = _profile_from_snapshot_fields("tmp", fields, media)
        return fingerprint(profile, self.cfg.repost) if profile.taxonomy_key() else None

    # -- verification ------------------------------------------------------

    async def _verify_phase(
        self,
        context: RunContext,
        comparison: ComparisonResult,
        outcome: RunOutcome,
        scraper: Any,
    ) -> None:
        if not comparison.missing_ids:
            return
        rows = [
            row
            for row in (self.repo.get_advertisement(pid) for pid in comparison.missing_ids)
            if row
        ]
        targets = select_verification_targets(rows, self.cfg)
        if not targets:
            return
        results = await verify_missing(scraper, targets, self.cfg, context)
        outcome.verifications = len(results)
        for verification in results:
            self.repo.record_verification(
                advertisement_id=verification.advertisement_id,
                run_id=context.run_id,
                url=verification.url,
                verdict=str(verification.verdict),
                http_status=verification.http_status,
                evidence=verification.evidence,
                checked_at=context.started_at,
            )
            availability = AVAILABILITY_BY_VERDICT.get(
                str(verification.verdict), DetailAvailability.UNKNOWN
            )
            self.db.execute(
                "UPDATE advertisements SET last_detail_verdict=?, last_detail_checked_at=?,"
                " detail_availability=? WHERE id=?",
                [
                    str(verification.verdict),
                    context.started_at,
                    str(availability),
                    verification.advertisement_id,
                ],
            )
            if verification.verdict is DetailVerdict.STILL_ACTIVE:
                # Important negative evidence: the listing exists. Whether it also
                # still matches the search decides between "left the filter" and
                # "we cannot explain this absence".
                self.repo.record_event(
                    advertisement_id=verification.advertisement_id,
                    run_id=context.run_id,
                    event_type=EventType.DETAIL_STILL_ACTIVE,
                    previous_status=None,
                    new_status=None,
                    event_at=context.started_at,
                    evidence=verification.evidence,
                )
                self._apply_filter_exit(context, verification, outcome)

    def _apply_filter_exit(
        self, context: RunContext, verification: Any, outcome: RunOutcome
    ) -> None:
        """A live listing that no longer matches the search has not disappeared.

        Leaving it at ``missing_once`` would let it march to ``likely_removed`` and
        then feed a sale inference, on evidence that says the opposite. So it moves
        to a terminal-for-comparison status, its miss counter is reset, and the
        reason is recorded. An absence that cannot be explained by any bound stays
        exactly as it was — "unexplained" must not be dressed up as "explained".
        """
        reason, evidence = classify_absence(verification.fields, self.search_filter)
        if reason is FilterExitReason.SEARCH_INDEX_INCONSISTENCY:
            # Honest non-explanation: keep the observed status, record the finding.
            self.repo.record_event(
                advertisement_id=verification.advertisement_id,
                run_id=context.run_id,
                event_type=EventType.FILTER_EXIT,
                previous_status=None,
                new_status=None,
                event_at=context.started_at,
                evidence={**evidence, "reason": str(reason), "status_changed": False},
            )
            outcome.unexplained_absences += 1
            return

        row = self.db.fetchone(
            "SELECT current_status FROM advertisements WHERE id=?",
            [verification.advertisement_id],
        )
        previous = AdStatus(str(row["current_status"])) if row else None
        self.db.execute(
            "UPDATE advertisements SET current_status=?, consecutive_misses=0,"
            " detail_availability=?, filter_exit_reason=?, filter_exit_at=?,"
            " filter_exit_evidence_json=?, updated_at=? WHERE id=?",
            [
                str(AdStatus.ACTIVE_OUTSIDE_FILTER),
                str(DetailAvailability.OUTSIDE_FILTER),
                str(reason),
                context.started_at,
                self.db.json_dump(evidence),
                utcnow(),
                verification.advertisement_id,
            ],
        )
        self.repo.record_event(
            advertisement_id=verification.advertisement_id,
            run_id=context.run_id,
            event_type=EventType.FILTER_EXIT,
            previous_status=previous,
            new_status=AdStatus.ACTIVE_OUTSIDE_FILTER,
            event_at=context.started_at,
            evidence={
                **evidence,
                "reason": str(reason),
                "status_changed": True,
                "interpretation": (
                    "the listing is still for sale but no longer matches the monitored "
                    "search; this is not a disappearance and not evidence of a sale"
                ),
            },
        )
        outcome.filter_exits += 1

    # -- repost detection --------------------------------------------------

    def _repost_phase(
        self, context: RunContext, comparison: ComparisonResult, outcome: RunOutcome
    ) -> None:
        if not self.cfg.repost.enabled or not comparison.new_ids:
            return
        cutoff = context.started_at - timedelta(days=self.cfg.repost.lookback_days)
        parents = self.db.fetchall(
            "SELECT a.*, s.brand, s.model, s.trim, s.year, s.mileage_normalized,"
            " s.price_normalized, s.city, s.seller_name, s.seller_type, s.body_color,"
            " s.interior_color, s.description, s.media_json"
            " FROM advertisements a LEFT JOIN advertisement_snapshots s"
            " ON s.id = a.latest_snapshot_id"
            " WHERE a.current_status IN (?,?) AND (a.first_missing_at IS NULL OR a.first_missing_at >= ?)",
            [str(AdStatus.LIKELY_REMOVED), str(AdStatus.MISSING_ONCE), cutoff],
        )
        children = self.db.fetchall(
            "SELECT a.*, s.brand, s.model, s.trim, s.year, s.mileage_normalized,"
            " s.price_normalized, s.city, s.seller_name, s.seller_type, s.body_color,"
            " s.interior_color, s.description, s.media_json"
            " FROM advertisements a LEFT JOIN advertisement_snapshots s"
            " ON s.id = a.latest_snapshot_id WHERE a.first_run_id = ?",
            [context.run_id],
        )
        if not parents or not children:
            return

        parent_profiles = [_profile_from_row(r, self.db) for r in parents]
        child_profiles = [_profile_from_row(r, self.db) for r in children]
        matches = find_reposts(child_profiles, parent_profiles, self.cfg.repost)
        if not matches:
            return

        by_id = {str(r["platform_ad_id"]): r for r in [*parents, *children]}
        for match in matches:
            parent = by_id.get(match.parent_platform_ad_id)
            child = by_id.get(match.new_platform_ad_id)
            if not parent or not child:
                continue
            entity = vehicle_entity_id(
                match.parent_platform_ad_id,
                match.new_platform_ad_id,
                existing=parent.get("vehicle_entity_id"),
            )
            self.repo.record_repost_link(
                parent_ad_id=int(parent["id"]),
                child_ad_id=int(child["id"]),
                run_id=context.run_id,
                score=match.score,
                matched_on=match.matched_on,
                evidence=match.evidence,
                vehicle_entity_id=entity,
            )
            self.db.execute(
                "UPDATE advertisements SET vehicle_entity_id=? WHERE id IN (?,?)",
                [entity, int(parent["id"]), int(child["id"])],
            )
            self.db.execute(
                "UPDATE advertisements SET repost_parent_ad_id=? WHERE id=?",
                [int(parent["id"]), int(child["id"])],
            )
            transition = mark_reposted(
                AdState(status=AdStatus(str(parent["current_status"]))),
                child_platform_ad_id=match.new_platform_ad_id,
                score=match.score,
                evidence={"matched_on": match.matched_on, **match.evidence},
            )
            self.db.execute(
                "UPDATE advertisements SET current_status=? WHERE id=?",
                [str(transition.status), int(parent["id"])],
            )
            for event_type, evidence in transition.events:
                self.repo.record_event(
                    advertisement_id=int(parent["id"]),
                    run_id=context.run_id,
                    event_type=event_type,
                    previous_status=AdStatus(str(parent["current_status"])),
                    new_status=transition.status,
                    event_at=context.started_at,
                    evidence=evidence,
                    confidence=match.score,
                )
            outcome.reposts += 1
            comparison.reposted_ids.append(match.parent_platform_ad_id)

    # -- sale scoring ------------------------------------------------------

    def _scoring_phase(self, context: RunContext, outcome: RunOutcome) -> None:
        """Re-score every advertisement that is currently absent.

        Scoring is separate from the state machine on purpose: confidence can rise
        or fall as evidence accumulates without rewriting the observed history.
        """
        rows = self.db.fetchall(
            "SELECT * FROM advertisements WHERE current_status IN (?,?,?,?)",
            [
                str(AdStatus.MISSING_ONCE),
                str(AdStatus.LIKELY_REMOVED),
                str(AdStatus.LIKELY_SOLD),
                str(AdStatus.REPOSTED),
            ],
        )
        for row in rows:
            advertisement_id = int(row["id"])
            status = AdStatus(str(row["current_status"]))
            verdict_value = row.get("last_detail_verdict")
            inputs = ScoringInputs(
                consecutive_misses=int(row.get("consecutive_misses") or 0),
                removal_confirmation_misses=self.cfg.removal_confirmation_misses,
                detail_verdict=DetailVerdict(verdict_value) if verdict_value else None,
                persistent_unavailable_checks=self.repo.consecutive_unavailable_checks(
                    advertisement_id
                ),
                had_price_reduction=self.repo.recent_price_reduction(advertisement_id),
                first_seen_at=parse_ts(row.get("first_seen_at")),
                first_missing_at=parse_ts(row.get("first_missing_at")),
                reappeared=bool(row.get("reappeared_at"))
                and status in (AdStatus.REAPPEARED, AdStatus.ACTIVE),
                reposted=status is AdStatus.REPOSTED or bool(row.get("repost_parent_ad_id")),
                seller_reposted_same_vehicle=bool(
                    self.db.scalar(
                        "SELECT 1 FROM repost_links WHERE parent_ad_id=? LIMIT 1",
                        [advertisement_id],
                    )
                ),
                now=context.started_at,
            )
            score = score_sale(inputs, self.cfg.scoring)
            self.db.execute(
                "UPDATE advertisements SET sale_confidence=?, sale_label=? WHERE id=?",
                [score.confidence, str(score.label), advertisement_id],
            )

            # Promotion to likely_sold is a *status* change, so it needs the
            # threshold AND the state machine's guard (only from likely_removed).
            if (
                status is AdStatus.LIKELY_REMOVED
                and score.confidence >= self.cfg.scoring.status_promotion_at
            ):
                transition = promote_to_likely_sold(
                    AdState(status=status),
                    confidence=score.confidence,
                    evidence=score.as_evidence(),
                    at=context.started_at,
                )
                if transition.status is AdStatus.LIKELY_SOLD:
                    self.db.execute(
                        "UPDATE advertisements SET current_status=?, suspected_sold_at=? WHERE id=?",
                        [str(AdStatus.LIKELY_SOLD), context.started_at, advertisement_id],
                    )
                    for event_type, evidence in transition.events:
                        self.repo.record_event(
                            advertisement_id=advertisement_id,
                            run_id=context.run_id,
                            event_type=event_type,
                            previous_status=status,
                            new_status=AdStatus.LIKELY_SOLD,
                            event_at=context.started_at,
                            evidence=evidence,
                            confidence=score.confidence,
                        )
                    outcome.promoted_sold += 1

    def _removal_rate_alert(self, context: RunContext, comparison: ComparisonResult) -> None:
        total = len(comparison.seen_ids) + len(comparison.missing_ids)
        if not total:
            return
        rate = len(comparison.likely_removed_ids) / total
        if rate > 0.20:
            self.alerts.raise_alert(
                "unusually_high_removal_rate",
                f"{len(comparison.likely_removed_ids)} of {total} advertisements reached"
                f" likely_removed in one run ({rate:.1%})",
                severity=AlertSeverity.WARNING,
                run_id=context.run_id,
                metadata={"rate": round(rate, 4)},
            )


def _profile_from_row(row: dict[str, Any], db: Database) -> VehicleProfile:
    media = Database.json_load(row.get("media_json")) or []
    return _profile_from_snapshot_fields(
        str(row["platform_ad_id"]),
        {
            "brand_fa": row.get("brand"),
            "model_fa": row.get("model"),
            "trim_fa": row.get("trim"),
            "year_text": row.get("year"),
            "mileage_km": row.get("mileage_normalized"),
            "price_toman": row.get("price_normalized"),
            "city": row.get("city"),
            "dealer_name": row.get("seller_name"),
            "seller_type": row.get("seller_type"),
            "body_color": row.get("body_color"),
            "inside_color": row.get("interior_color"),
            "description": row.get("description"),
        },
        media if isinstance(media, list) else [],
    )


def _profile_from_snapshot_fields(
    platform_ad_id: str, fields: dict[str, Any], media: list[dict[str, Any]]
) -> VehicleProfile:
    """Build a matching profile, deriving image hashes from media URLs.

    Bama serves the same photo from a stable CDN path, so the path component acts
    as a usable perceptual-hash substitute without downloading the bytes. A true
    perceptual hash would be better and is noted as a limitation.
    """
    import hashlib
    from urllib.parse import urlparse

    hashes: list[str] = []
    for item in media or []:
        if not isinstance(item, dict):
            continue
        url = item.get("original_url") or item.get("large_url") or item.get("url")
        if not url:
            continue
        path = urlparse(str(url)).path
        hashes.append(hashlib.sha1(path.encode("utf-8")).hexdigest()[:16])

    return VehicleProfile(
        platform_ad_id=platform_ad_id,
        brand=fields.get("brand_fa") or fields.get("brand"),
        model=fields.get("model_fa") or fields.get("model"),
        trim=fields.get("trim_fa") or fields.get("trim"),
        year=str(fields.get("year_text") or fields.get("year") or "") or None,
        mileage_km=fields.get("mileage_km") or fields.get("mileage_normalized"),
        price_toman=fields.get("price_toman") or fields.get("price_normalized"),
        city=fields.get("city"),
        seller_name=fields.get("dealer_name") or fields.get("seller_name"),
        seller_type=fields.get("seller_type"),
        body_color=fields.get("body_color"),
        interior_color=fields.get("inside_color") or fields.get("interior_color"),
        description=fields.get("description"),
        image_hashes=tuple(hashes[:6]),
    )
