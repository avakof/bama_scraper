"""Configuration for the monitoring system.

Every threshold that decides whether an advertisement is considered removed, or
whether a run is trusted at all, is a named setting here rather than a literal
buried in logic — those are the values an operator needs to tune, and the values
a reviewer needs to audit.

Credentials are never stored in this file. ``database_url`` and alert
destinations come from the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

MONITOR_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEARCH_URL = (
    "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian"
)


class AlertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: Structured logging is always on; these are optional extra sinks.
    webhook_url_env: str = "BAMA_MONITOR_WEBHOOK_URL"
    email_to_env: str = "BAMA_MONITOR_ALERT_EMAIL"
    smtp_url_env: str = "BAMA_MONITOR_SMTP_URL"
    min_severity: Literal["info", "warning", "critical"] = "warning"


class HealthThresholds(BaseModel):
    """Anomaly thresholds that decide whether a run may be compared."""

    model_config = ConfigDict(extra="forbid")

    #: A run is anomalous if its count falls below this fraction of the recent
    #: median. 0.30 means "70% lower than the median" per the specification.
    min_count_ratio_vs_median: float = 0.30
    #: How many prior valid runs form the median baseline.
    median_window_runs: int = 7
    #: Absolute floor: a run finding fewer than this is implausible for a filter
    #: that has been returning thousands.
    min_absolute_count: int = 50
    #: Share of result pages allowed to fail before the run is untrustworthy.
    max_failed_page_ratio: float = 0.20
    #: Share of cards allowed to have an empty title.
    max_empty_title_ratio: float = 0.20
    #: A single run may not push more than this share of the known inventory into
    #: "missing" — a spike that large is far more likely to be our fault.
    max_missing_ratio: float = 0.50
    require_verified_termination: bool = True
    require_stabilization: bool = True


class SaleScoringWeights(BaseModel):
    """Rule weights for the sale-likelihood inference.

    Positive weights are evidence *consistent with* a sale; negative weights are
    evidence against. Nothing here can produce a confirmed sale.
    """

    model_config = ConfigDict(extra="forbid")

    absent_two_valid_runs: float = 0.30
    detail_explicitly_unavailable: float = 0.15
    detail_persistent_404_410: float = 0.15
    recent_price_reduction: float = 0.10
    disappeared_soon_after_publication: float = 0.10
    seller_did_not_repost: float = 0.10
    disappearance_persists_seven_days: float = 0.10

    reappeared: float = -0.30
    vehicle_reposted: float = -0.25
    detail_remains_accessible: float = -0.20
    appears_outside_original_filter: float = -0.20

    #: Label boundaries, inclusive lower bounds.
    possibly_sold_at: float = 0.40
    likely_sold_at: float = 0.65
    highly_likely_sold_at: float = 0.85
    #: A listing must reach this confidence before it appears in
    #: "fastest disappearing" rankings.
    ranking_min_confidence: float = 0.65
    #: Confidence at which the observed status may become ``likely_sold``.
    status_promotion_at: float = 0.65
    #: "Soon after publication" cutoff, in days.
    soon_after_publication_days: float = 3.0


class RepostConfig(BaseModel):
    """Fingerprint and matching tolerances for repost detection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: Only consider parents removed within this window.
    lookback_days: int = 45
    #: Mileage may drift a little between a removal and a repost.
    mileage_tolerance_km: int = 5_000
    mileage_bucket_km: int = 10_000
    price_tolerance_ratio: float = 0.12
    price_bucket_toman: int = 50_000_000
    #: Minimum description similarity (0-1) to count as textual evidence.
    description_similarity_min: float = 0.72
    #: Total score needed to record a possible repost.
    match_threshold: float = 0.70
    #: Weights per matched signal; deliberately spread so no single field decides.
    weight_brand_model_year: float = 0.30
    weight_same_seller: float = 0.20
    weight_mileage_close: float = 0.15
    weight_price_close: float = 0.10
    weight_city: float = 0.05
    weight_image_hash: float = 0.20
    weight_description_similar: float = 0.15
    weight_colors: float = 0.05


class DetailPolicy(BaseModel):
    """When the deep scraper runs. Not every listing, not every day."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int = 2
    scrape_new: bool = True
    scrape_card_changed: bool = True
    scrape_missing: bool = True
    scrape_previous_failures: bool = True
    #: Periodic refresh window for unchanged active listings.
    active_refresh_days_min: int = 3
    active_refresh_days_max: int = 7
    #: Random quality-control sample of otherwise-skipped listings.
    qc_sample_ratio: float = 0.01
    #: Hard cap so one run cannot balloon into thousands of detail fetches.
    #: Ignored when `check_every_advertisement_daily` is on: a census has no cap,
    #: because a partial census is not a census.
    max_details_per_run: int = 600
    #: Check EVERY discovered advertisement on every run.
    #:
    #: On for production. The selective policy above is an optimisation that makes
    #: "no snapshot today" ambiguous between "unchanged" and "not examined"; a
    #: daily census removes the ambiguity and is what lets a run assert coverage.
    #: Storage stays deduplicated -- an unchanged check references the existing
    #: snapshot rather than copying it.
    check_every_advertisement_daily: bool = True
    delay_min: float = 0.4
    delay_max: float = 1.2
    request_timeout: float = 45.0


class MonitorConfig(BaseModel):
    """Top-level settings."""

    model_config = ConfigDict(extra="forbid")

    search_url: str = DEFAULT_SEARCH_URL
    timezone: str = "Europe/Paris"
    run_at: str = "13:00"

    #: SQLite for development, PostgreSQL for production. Read from the
    #: environment so credentials never live in a config file.
    database_url: str = Field(
        default_factory=lambda: os.environ.get(
            "BAMA_MONITOR_DATABASE_URL",
            f"sqlite:///{REPO_ROOT / 'monitor_data' / 'bama_monitor.sqlite'}",
        )
    )

    #: Consecutive missed valid runs before removal is considered confirmed.
    removal_confirmation_misses: int = 2
    #: Days of continued absence that add the "persists" evidence component.
    persistent_absence_days: int = 7
    #: How close first observation must be to publication for a listing's whole
    #: life to count as observed. Beyond this it is left-truncated and excluded
    #: from duration rankings.
    entry_tolerance_hours: float = 36.0

    health: HealthThresholds = Field(default_factory=HealthThresholds)
    scoring: SaleScoringWeights = Field(default_factory=SaleScoringWeights)
    repost: RepostConfig = Field(default_factory=RepostConfig)
    detail: DetailPolicy = Field(default_factory=DetailPolicy)
    alerts: AlertConfig = Field(default_factory=AlertConfig)

    output_dir: Path = REPO_ROOT / "monitor_data"
    #: Immutable daily exports, partitioned by scheduled slot.
    daily_snapshot_dir: Path = REPO_ROOT / "daily_snapshots"

    #: Names the production schedule. Slot uniqueness is scoped to it, so a second
    #: schedule (another search, another hour) cannot collide with this one.
    production_schedule_name: str = "bama-daily-1300-paris"
    #: How late a missed slot may still be run as `catch_up`. Beyond this the slot
    #: is recorded as missed rather than fabricated.
    catch_up_grace_minutes: int = 240
    reports_dir: Path = REPO_ROOT / "reports"
    lock_dir: Path = REPO_ROOT / "monitor_data" / "locks"

    #: Bound the discovery pass so a pathological run cannot spin forever.
    max_discovery_pages: int = 5000
    max_run_seconds: float = 10800.0
    dry_run: bool = False
    log_level: str = "INFO"

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://"))

    @property
    def sqlite_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("database_url is not a SQLite URL")
        return Path(self.database_url[len(prefix) :])

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.reports_dir, self.lock_dir, self.daily_snapshot_dir):
            path.mkdir(parents=True, exist_ok=True)

    def configuration_hash(self) -> str:
        """Stable hash of everything that affects comparability of two runs.

        Only the *search definition* and the decision thresholds are included:
        changing them means today's inventory is not comparable with yesterday's,
        which is exactly what this hash is used to detect. Paths and log levels
        are excluded because they change nothing about the data.
        """
        payload = {
            "search_url": self.search_url,
            "timezone": self.timezone,
            "removal_confirmation_misses": self.removal_confirmation_misses,
            "persistent_absence_days": self.persistent_absence_days,
            "health": self.health.model_dump(),
            "scoring": self.scoring.model_dump(),
            "repost": self.repost.model_dump(),
            "monitor_version": MONITOR_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None, **overrides: Any) -> MonitorConfig:
    """Build a config from an optional YAML file plus overrides."""
    data: dict[str, Any] = {}
    if path:
        candidate = Path(path)
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("output_dir", "reports_dir", "lock_dir"):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return MonitorConfig(**data)
