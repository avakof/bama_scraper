"""EDA configuration.

Thresholds live here rather than in the analysis code because they are editorial
decisions, not implementation details: "how many listings before a group is a
market segment" is a judgement the reader is entitled to see and change.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the monitoring database lives by default. Deliberately the same path the
#: monitor writes to, so the EDA analyses the real thing rather than an export.
DEFAULT_MONITOR_SQLITE = REPO_ROOT / "monitor_data" / "bama_monitor.sqlite"

#: Optional attribute source. See `AttributeEnrichment` below for why this exists
#: and why it is opt-in.
DEFAULT_DEEP_SQLITE = REPO_ROOT / "deep_scraper" / "data" / "bama_deep.sqlite"


class Thresholds(BaseModel):
    """Suppression thresholds. Below these, a statistic is noise with a label."""

    model_config = ConfigDict(extra="forbid")

    #: Groups smaller than this are exported but marked `insufficient_sample`
    #: rather than presented as market segments.
    min_group_size: int = 10
    #: Survival curves need both enough subjects and enough actual events; a curve
    #: over 40 subjects with 2 events is a straight line with a confidence band
    #: wider than the plot.
    min_survival_group_size: int = 30
    min_survival_events: int = 10
    #: A detail snapshot older than this is flagged stale: its price and mileage
    #: may no longer describe what the search showed on the analysis run.
    snapshot_stale_hours: float = 72.0
    #: Correlations below this |rho| are reported but not narrated.
    weak_correlation: float = 0.2
    #: Minimum decided human labels before a calibrated sale rate may be quoted.
    min_calibration_labels: int = 40
    #: Minimum pairs for a correlation to be computed at all.
    min_correlation_pairs: int = 30


class AttributeEnrichment(BaseModel):
    """Optional join to the deep scraper's per-vehicle attributes.

    Why this is needed: the monitoring schema stores vehicle attributes in
    ``advertisement_snapshots``, which is written only when a detail page is
    actually scraped. Detail scraping is rate-limited and policy-driven, so on a
    young installation the snapshot coverage is a few percent, and questions about
    brand, model, condition or seller type simply cannot be answered from the
    monitoring database alone.

    Why it is **opt-in and provenanced**: joining a second database is exactly the
    kind of convenience that destroys auditability if it happens silently. When
    enabled, every enriched column is recorded in the manifest with its source
    table, its coverage, and the temporal rule applied.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    database_path: Path | None = None
    table: str = "ad_deep"
    join_left: str = "platform_ad_id"
    join_right: str = "ad_id"
    #: The same temporal rule as for snapshots: nothing scraped after the analysis
    #: run's finish time may describe that run.
    timestamp_column: str = "scraped_at"

    #: Take every analysable column rather than a hand-picked list. A curated list
    #: silently decides what the analysis is allowed to notice; the exclusions
    #: below are the only editorial judgement, and they are about blobs and privacy
    #: rather than about which findings are interesting.
    include_all_columns: bool = True
    #: Drop a column that is this empty. It cannot support a distribution.
    min_column_coverage: float = 0.01

    #: Trim-level technical specifications (sunroof, ABS, airbag count, dimensions,
    #: performance), joined through the trim key.
    #:
    #: These are attributes of a MODEL-TRIM, not observations of an individual
    #: listing: every listing of the same trim carries identical values. So
    #: "sunroof by city" describes the trim mix in that city, not a per-car
    #: measurement, and `feature_analysis` labels them accordingly.
    include_specs: bool = True
    spec_view: str = "v_trim_specs_wide"
    spec_join_column: str = "review_key"


class EdaConfig(BaseModel):
    """Top-level EDA settings."""

    model_config = ConfigDict(extra="forbid")

    #: Resolution order: explicit CLI value, then environment, then the monitor's
    #: own default SQLite path. Credentials are never written to a config file.
    database_url: str = Field(
        default_factory=lambda: (
            os.environ.get("BAMA_EDA_DATABASE_URL")
            or os.environ.get("BAMA_DATABASE_URL")
            or os.environ.get("BAMA_MONITOR_DATABASE_URL")
            or f"sqlite:///{DEFAULT_MONITOR_SQLITE}"
        )
    )
    run_id: str = "latest-valid"
    output_dir: Path = REPO_ROOT / "reports" / "eda"
    cache_dir: Path = REPO_ROOT / "reports" / "eda" / ".cache"

    thresholds: Thresholds = Field(default_factory=Thresholds)
    enrichment: AttributeEnrichment = Field(default_factory=AttributeEnrichment)

    #: Default False: an estimated market duration for a listing whose earlier life
    #: was never observed is a lower bound masquerading as a measurement.
    include_left_truncated: bool = False
    html: bool = True
    export_csv: bool = True
    export_parquet: bool = True
    #: Fixed so a rerun reproduces the same samples and jitter.
    random_seed: int = 20260728
    max_rows_in_memory: int = 500_000
    chunk_size: int = 50_000
    log_level: str = "INFO"

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://"))

    @property
    def backend(self) -> str:
        return "postgres" if self.is_postgres else "sqlite"

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def fingerprint(self) -> str:
        """Hash of everything that changes the numbers.

        The database URL is deliberately excluded (it may carry credentials) but
        the backend is included, because SQLite and PostgreSQL can legitimately
        disagree on ordering.
        """
        payload = {
            "backend": self.backend,
            "run_id": self.run_id,
            "thresholds": self.thresholds.model_dump(),
            "include_left_truncated": self.include_left_truncated,
            "enrichment": {
                "enabled": self.enrichment.enabled,
                "table": self.enrichment.table,
            },
            "random_seed": self.random_seed,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path | None = None, **overrides: Any) -> EdaConfig:
    """Build a config from an optional YAML file plus CLI overrides."""
    data: dict[str, Any] = {}
    if path:
        candidate = Path(path)
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}

    # Nested overrides arrive flat from the CLI; route them to their sections.
    threshold_keys = set(Thresholds.model_fields)
    thresholds = dict(data.get("thresholds") or {})
    for key in list(overrides):
        if key in threshold_keys and overrides[key] is not None:
            thresholds[key] = overrides.pop(key)
    if thresholds:
        data["thresholds"] = thresholds

    data.update({k: v for k, v in overrides.items() if v is not None})
    return EdaConfig(**data)
