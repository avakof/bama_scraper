"""Vocabulary for the EDA package.

This module exists mainly to make the *analytical* vocabulary explicit and
mechanically checkable. The monitoring package already separates observation from
inference; an analysis layer is where that separation usually gets quietly lost,
because a chart title is much easier to write than a caveat.

So the forbidden phrasings are listed here as data, and a test asserts that no
generated artefact contains them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd

#: Analysis-layer version, recorded in every manifest and every dataset row.
ANALYSIS_VERSION = "1.0.0"


class Severity(StrEnum):
    """Integrity-finding severity. ``CRITICAL`` stops the analysis."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFORMATIONAL = "informational"


class AnalyticalRole(StrEnum):
    """What a column may legitimately be used for."""

    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    RAW_TEXT = "raw_text"
    NORMALIZED_NUMERIC = "normalized_numeric"
    CATEGORY = "category"
    #: A recorded fact about what was seen.
    OUTCOME_OBSERVATION = "outcome_observation"
    #: A derived guess. Never to be reported as a measurement.
    HEURISTIC_INFERENCE = "heuristic_inference"
    PROVENANCE = "provenance"
    #: Excluded from every output: private, or a large blob with no analytical use.
    PRIVATE_OR_EXCLUDED = "private_or_excluded"


class CensoringClass(StrEnum):
    """How a duration observation is censored.

    These are kept as distinct classes rather than a boolean because they need
    different statistical handling and mixing them silently is the single easiest
    way to produce a confident wrong answer about time on market.
    """

    #: Publication and disappearance both observed: usable as an event.
    FULLY_OBSERVED = "fully_observed"
    #: Still present at the end of observation.
    RIGHT_CENSORED = "right_censored"
    #: Already on the market when observation began; earlier life unknown.
    LEFT_TRUNCATED = "left_truncated"
    #: Disappearance happened somewhere inside a known interval.
    INTERVAL_CENSORED = "interval_censored"
    #: Left the search but not the market: a censoring event, not a disappearance.
    ACTIVE_OUTSIDE_FILTER = "active_outside_filter"
    REAPPEARED = "reappeared"
    REPOSTED = "reposted"
    UNKNOWN = "unknown"


#: Phrasings that misstate what this data can support. Asserted absent from every
#: generated CSV, JSON and HTML artefact by ``tests/bama_eda/test_provenance.py``.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "time_to_sale",
    "sale_probability",
    "probability_of_sale",
    "confirmed_sold",
    "sold_vehicles",
    "exact_sale_date",
    "sale_date",
)

#: Terms this package uses instead.
APPROVED_TERMS: tuple[str, ...] = (
    "time_to_disappearance",
    "observed_monitoring_duration",
    "estimated_market_duration",
    "sale_evidence_score",
    "likely_sold",
    "active_outside_filter",
    "left_truncated",
    "right_censored",
    "interval_censored",
    "vehicle_entity_id",
)

#: Columns never copied into an analytical output, whatever the source table says.
#: `phone` is stripped at scrape time already; this is a second, independent gate,
#: because a privacy guarantee with one enforcement point is a privacy hope.
PRIVATE_COLUMNS: frozenset[str] = frozenset(
    {
        "phone",
        "phone_number",
        "mobile",
        "seller_phone",
        "contact",
        "contact_number",
        "telephone",
        "dealer_address",
        "address",
        "street_address",
        "exact_address",
    }
)

#: Large blobs that would bloat memory with no analytical use at this stage.
EXCLUDED_BLOB_COLUMNS: frozenset[str] = frozenset(
    {
        "api_raw_json",
        "html_payload_json",
        "json_ld_json",
        "attributes_json",
        "raw_html",
        "html",
        "authenticity_json",
        "icon_json",
        "options_json",
        "life_styles_json",
        "breadcrumb_json",
        "provenance_json",
    }
)


@dataclass(frozen=True)
class AnalysisContext:
    """Immutable identity of one analysis: which run, which code, which database.

    Every dataset row and every output file carries these fields. A count from a
    live marketplace is meaningless without them.
    """

    run_id: int
    scheduled_for: datetime | None
    run_started_at: datetime | None
    run_finished_at: datetime | None
    search_url: str | None
    search_configuration_hash: str | None
    scraper_version: str | None
    monitor_version: str | None
    database_backend: str
    analysis_started_at: datetime
    analysis_version: str = ANALYSIS_VERSION

    def as_dict(self) -> dict[str, Any]:
        def iso(value: datetime | None) -> str | None:
            return value.isoformat() if value else None

        return {
            "run_id": self.run_id,
            "scheduled_for": iso(self.scheduled_for),
            "run_started_at": iso(self.run_started_at),
            "run_finished_at": iso(self.run_finished_at),
            "search_url": self.search_url,
            "search_configuration_hash": self.search_configuration_hash,
            "scraper_version": self.scraper_version,
            "monitor_version": self.monitor_version,
            "database_backend": self.database_backend,
            "analysis_started_at": iso(self.analysis_started_at),
            "analysis_version": self.analysis_version,
        }

    def provenance_columns(self) -> dict[str, Any]:
        """The subset stamped onto every dataset row."""
        payload = self.as_dict()
        return {
            "run_id": payload["run_id"],
            "scheduled_for": payload["scheduled_for"],
            "run_started_at": payload["run_started_at"],
            "run_finished_at": payload["run_finished_at"],
            "search_configuration_hash": payload["search_configuration_hash"],
            "scraper_version": payload["scraper_version"],
            "monitor_version": payload["monitor_version"],
            "analysis_version": payload["analysis_version"],
            "database_backend": payload["database_backend"],
        }


@dataclass
class Finding:
    """One integrity finding."""

    check: str
    severity: Severity
    count: int
    detail: str
    sample: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": str(self.severity),
            "count": self.count,
            "detail": self.detail,
            "sample": ", ".join(str(s) for s in self.sample[:5]),
        }


@dataclass
class AnalysisWarning:
    """A non-blocking analytical caveat.

    Collected rather than logged-and-forgotten, and exported to
    ``analysis_warnings.csv`` so a reader of the report sees the same caveats the
    analyst did.
    """

    area: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"area": self.area, "message": self.message}


def is_texty(series: pd.Series) -> bool:
    """Whether a column holds text.

    NOT ``dtype == object``: pandas 3 gives string columns a dedicated ``str``
    dtype, so the object test silently matches nothing and every text-scanning
    guard -- including the privacy sweep -- passes by scanning zero columns.
    """
    return bool(pd.api.types.is_string_dtype(series) or series.dtype == object)
