"""Run-health validation: the gate that protects the historical record.

The single most damaging failure mode in longitudinal monitoring is a *partial*
scrape. If the scraper is blocked, stops early, or the site returns a maintenance
page, the listings it failed to see look identical to listings that were genuinely
removed. Comparing such a run would push hundreds of live advertisements toward
``likely_removed`` and there is no way to distinguish that corruption afterwards.

So health is evaluated *before* any comparison, and only :attr:`RunHealth.VALID`
permits status transitions. Every other outcome preserves the collected evidence,
leaves missing counters untouched, and raises an alert.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .config import MonitorConfig
from .db import Database
from .models import DiscoveryResult, RunHealth


@dataclass
class HealthCheck:
    """One named check and its outcome."""

    name: str
    passed: bool
    detail: str
    #: A failed blocking check makes the run unusable for comparison. A failed
    #: non-blocking check is recorded and alerted but does not veto the run.
    blocking: bool = True
    observed: Any = None
    threshold: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "blocking": self.blocking,
            "detail": self.detail,
            "observed": self.observed,
            "threshold": self.threshold,
        }


@dataclass
class HealthReport:
    status: RunHealth
    checks: list[HealthCheck] = field(default_factory=list)
    baseline_median: float | None = None

    @property
    def valid(self) -> bool:
        return self.status is RunHealth.VALID

    @property
    def failed_blocking(self) -> list[HealthCheck]:
        return [c for c in self.checks if c.blocking and not c.passed]

    def reason(self) -> str:
        failed = self.failed_blocking
        if not failed:
            return "all blocking checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in failed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "valid": self.valid,
            "reason": self.reason(),
            "baseline_median": self.baseline_median,
            "checks": [c.as_dict() for c in self.checks],
        }


def recent_valid_counts(db: Database, config_hash: str, limit: int) -> list[int]:
    """Discovered counts from recent valid runs of the *same* configuration.

    Filtering on the configuration hash matters: a threshold or filter change
    makes older counts incomparable, and using them would produce a false anomaly.
    """
    rows = db.fetchall(
        "SELECT discovered_count FROM monitoring_runs"
        " WHERE status = ? AND configuration_hash = ? AND discovered_count IS NOT NULL"
        " ORDER BY scheduled_for DESC LIMIT ?",
        [str(RunHealth.VALID), config_hash, limit],
    )
    return [int(r["discovered_count"]) for r in rows if r["discovered_count"] is not None]


def evaluate_run(
    result: DiscoveryResult,
    cfg: MonitorConfig,
    *,
    baseline_counts: list[int] | None = None,
    known_active_count: int | None = None,
    expected_missing_count: int | None = None,
    persisted: bool = True,
) -> HealthReport:
    """Classify a discovery result's health.

    ``baseline_counts`` are prior valid counts for the median anomaly check.
    ``known_active_count`` and ``expected_missing_count`` come from the caller,
    which has both id sets, and together veto an implausible mass-disappearance.
    """
    checks: list[HealthCheck] = []
    thresholds = cfg.health

    checks.append(
        HealthCheck(
            "target_url_loaded",
            result.initial_page_ok,
            "initial page loaded" if result.initial_page_ok else "initial page request failed",
            observed=result.initial_page_ok,
        )
    )

    checks.append(
        HealthCheck(
            "not_blocked",
            not result.blocked,
            "no blocking or CAPTCHA page detected"
            if not result.blocked
            else "blocked/CAPTCHA response observed",
            observed=result.blocked,
        )
    )

    if thresholds.require_verified_termination:
        checks.append(
            HealthCheck(
                "verified_termination",
                result.reached_verified_end,
                f"termination_reason={result.termination_reason!r}",
                observed=result.termination_reason,
            )
        )

    if thresholds.require_stabilization:
        checks.append(
            HealthCheck(
                "stabilization_completed",
                result.stabilization_completed,
                "final stabilization pass completed"
                if result.stabilization_completed
                else "stabilization pass did not complete",
                observed=result.stabilization_completed,
            )
        )

    checks.append(
        HealthCheck(
            "inventory_persisted",
            persisted,
            "inventory persisted" if persisted else "inventory was not persisted",
            observed=persisted,
        )
    )

    # -- plausibility of the count ----------------------------------------
    count = result.discovered_count
    checks.append(
        HealthCheck(
            "minimum_absolute_count",
            count >= thresholds.min_absolute_count,
            f"discovered {count} advertisements",
            observed=count,
            threshold=thresholds.min_absolute_count,
        )
    )

    median: float | None = None
    if baseline_counts:
        median = float(statistics.median(baseline_counts))
        floor = median * thresholds.min_count_ratio_vs_median
        checks.append(
            HealthCheck(
                "count_vs_recent_median",
                count >= floor,
                f"count {count} vs median {median:.0f} (floor {floor:.0f})",
                observed=count,
                threshold=round(floor, 1),
            )
        )
    else:
        # No baseline yet. This is normal on the first run and must not be
        # treated as an anomaly, but it is recorded so the report is honest.
        checks.append(
            HealthCheck(
                "count_vs_recent_median",
                True,
                "no prior valid run of this configuration; baseline check skipped",
                blocking=False,
                observed=count,
            )
        )

    # -- page-level failures ----------------------------------------------
    attempted = result.pages_fetched + result.pages_failed
    if attempted:
        ratio = result.pages_failed / attempted
        checks.append(
            HealthCheck(
                "failed_page_ratio",
                ratio <= thresholds.max_failed_page_ratio,
                f"{result.pages_failed}/{attempted} pages failed ({ratio:.1%})",
                observed=round(ratio, 4),
                threshold=thresholds.max_failed_page_ratio,
            )
        )

    # -- parser sanity ----------------------------------------------------
    if count:
        empty_titles = sum(1 for c in result.cards if not (c.card_title or "").strip())
        ratio = empty_titles / count
        checks.append(
            HealthCheck(
                "card_titles_present",
                ratio <= thresholds.max_empty_title_ratio,
                f"{empty_titles}/{count} cards have no title ({ratio:.1%})"
                " — a selector change is the usual cause",
                observed=round(ratio, 4),
                threshold=thresholds.max_empty_title_ratio,
            )
        )
        missing_urls = sum(1 for c in result.cards if not c.canonical_url)
        checks.append(
            HealthCheck(
                "card_urls_present",
                missing_urls == 0,
                f"{missing_urls} cards have no canonical URL",
                observed=missing_urls,
                threshold=0,
            )
        )

    # -- mass-disappearance veto ------------------------------------------
    # The caller supplies the exact figure because only it knows both id sets;
    # deriving it from counts alone would be wrong whenever the inventory both
    # gained and lost listings on the same day.
    if known_active_count and expected_missing_count is not None:
        ratio = expected_missing_count / known_active_count
        checks.append(
            HealthCheck(
                "plausible_missing_ratio",
                ratio <= thresholds.max_missing_ratio,
                f"this run would mark {expected_missing_count}/{known_active_count}"
                f" advertisements missing ({ratio:.1%})",
                observed=round(ratio, 4),
                threshold=thresholds.max_missing_ratio,
            )
        )

    if result.errors:
        checks.append(
            HealthCheck(
                "no_critical_errors",
                False,
                f"{len(result.errors)} error(s): {result.errors[0][:160]}",
                observed=len(result.errors),
            )
        )

    status = _classify(checks, result)
    return HealthReport(status=status, checks=checks, baseline_median=median)


def _classify(checks: list[HealthCheck], result: DiscoveryResult) -> RunHealth:
    """Map failed checks onto the most specific health status.

    The distinction matters operationally: ``blocked`` means back off, ``partial``
    means retry may complete it, ``invalid`` means the data is unusable as-is.
    """
    failed = {c.name for c in checks if c.blocking and not c.passed}
    if not failed:
        return RunHealth.VALID
    if "not_blocked" in failed:
        return RunHealth.BLOCKED
    if "target_url_loaded" in failed or "inventory_persisted" in failed:
        return RunHealth.FAILED
    if failed & {"verified_termination", "stabilization_completed", "failed_page_ratio"}:
        # The scrape ran but cannot be trusted to be complete.
        return RunHealth.PARTIAL
    return RunHealth.INVALID


def anomaly_alerts(report: HealthReport, result: DiscoveryResult) -> list[dict[str, Any]]:
    """Alert payloads for each failed check, for the alerting layer to dispatch."""
    alerts: list[dict[str, Any]] = []
    for check in report.checks:
        if check.passed:
            continue
        alerts.append(
            {
                "alert_type": f"health_{check.name}",
                "severity": "critical" if check.blocking else "warning",
                "message": f"run health check failed: {check.name} — {check.detail}",
                "metadata": {
                    "check": check.as_dict(),
                    "run_status": str(report.status),
                    "discovered_count": result.discovered_count,
                    "termination_reason": result.termination_reason,
                },
            }
        )
    return alerts
