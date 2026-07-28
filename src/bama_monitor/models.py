"""Domain model for longitudinal listing monitoring.

The central discipline of this package is encoded here: an advertisement
*disappearing from the observed inventory* is a fact, while a *sale* is an
inference. The two are represented by different fields — :class:`AdStatus`
records what was observed, :class:`SaleLabel` and ``sale_confidence`` record what
is inferred — and no code path collapses one into the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class AdStatus(StrEnum):
    """Observed lifecycle state of an advertisement listing."""

    NEW = "new"
    ACTIVE = "active"
    MISSING_ONCE = "missing_once"
    LIKELY_REMOVED = "likely_removed"
    REAPPEARED = "reappeared"
    LIKELY_SOLD = "likely_sold"
    REPOSTED = "reposted"
    #: Alive, but no longer inside the monitored search. A seller who drops the
    #: price below ``priceFrom`` leaves the result set without leaving the market;
    #: counting that as a disappearance would be a false removal.
    ACTIVE_OUTSIDE_FILTER = "active_outside_filter"
    UNKNOWN = "unknown"


#: States whose advertisements are still part of the comparison population.
#: ``ACTIVE_OUTSIDE_FILTER`` is deliberately absent: the listing is known to be
#: alive and known to be outside the search, so its continued absence carries no
#: information and must not accumulate misses.
COMPARABLE_STATUSES: frozenset[AdStatus] = frozenset(
    {AdStatus.NEW, AdStatus.ACTIVE, AdStatus.REAPPEARED, AdStatus.MISSING_ONCE}
)

#: States that are terminal for comparison purposes -- they are not re-tested for
#: absence, though a reappearance is still honoured if one is observed.
TERMINAL_STATUSES: frozenset[AdStatus] = frozenset(
    {
        AdStatus.LIKELY_REMOVED,
        AdStatus.LIKELY_SOLD,
        AdStatus.REPOSTED,
        AdStatus.ACTIVE_OUTSIDE_FILTER,
    }
)

#: Statuses that mean "not on the market any more, as far as we can tell".
#: ``ACTIVE_OUTSIDE_FILTER`` is NOT one of them -- the car is still for sale.
DISAPPEARED_STATUSES: frozenset[AdStatus] = frozenset(
    {AdStatus.LIKELY_REMOVED, AdStatus.LIKELY_SOLD}
)


class RunHealth(StrEnum):
    """Health of a single monitoring run.

    Only :attr:`VALID` permits status transitions. Anything else means the run's
    absence signal is untrustworthy and must not touch missing counters.
    """

    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"
    RUNNING = "running"
    SKIPPED_DUE_TO_EXISTING_RUN = "skipped_due_to_existing_run"


class TriggerType(StrEnum):
    """How a run came to exist.

    Only :attr:`SCHEDULED` and :attr:`CATCH_UP` form genuine daily history. The
    others describe real work that must never be counted as a daily observation —
    a manual run at 10:30 is not the day's 13:00 measurement, however good its
    data is.
    """

    SCHEDULED = "scheduled"
    CATCH_UP = "catch_up"
    MANUAL = "manual"
    DEPLOYMENT_TEST = "deployment_test"
    BACKFILL = "backfill"
    SIMULATION = "simulation"


#: Trigger types that may own a production scheduled slot.
PRODUCTION_TRIGGERS: frozenset[TriggerType] = frozenset(
    {TriggerType.SCHEDULED, TriggerType.CATCH_UP}
)

#: Trigger types whose runs count as genuine daily history.
GENUINE_HISTORY_TRIGGERS: frozenset[TriggerType] = PRODUCTION_TRIGGERS


class DetailCheckOutcome(StrEnum):
    """How one advertisement's daily detail check ended.

    ``COMPLETED``, ``GONE`` and ``PERMANENT_FAILURE`` are all *accounted for*: the
    advertisement was looked at and the result is known. ``RETRYABLE_FAILURE`` is
    not, and a run with any of those left over cannot be valid.
    """

    COMPLETED = "completed"
    GONE = "gone"
    PERMANENT_FAILURE = "permanent_failure"
    RETRYABLE_FAILURE = "retryable_failure"
    SKIPPED = "skipped"


#: Outcomes that count towards `detail_accounted_count`.
ACCOUNTED_OUTCOMES: frozenset[DetailCheckOutcome] = frozenset(
    {
        DetailCheckOutcome.COMPLETED,
        DetailCheckOutcome.GONE,
        DetailCheckOutcome.PERMANENT_FAILURE,
    }
)


class EventType(StrEnum):
    """Immutable audit events."""

    DISCOVERED = "discovered"
    SEEN = "seen"
    PRICE_CHANGED = "price_changed"
    DESCRIPTION_CHANGED = "description_changed"
    ATTRIBUTES_CHANGED = "attributes_changed"
    MISSING_FIRST_TIME = "missing_first_time"
    REMOVAL_CONFIRMED = "removal_confirmed"
    REAPPEARED = "reappeared"
    POSSIBLE_REPOST = "possible_repost"
    LIKELY_SOLD = "likely_sold"
    DETAIL_UNAVAILABLE = "detail_unavailable"
    #: The listing left the monitored search but its page shows it is still for
    #: sale outside the filter. An explanation for an absence, not a removal.
    FILTER_EXIT = "filter_exit"
    #: It came back inside the filter (e.g. the price was raised again).
    FILTER_REENTRY = "filter_reentry"
    #: The detail page loaded normally while the listing was absent from search.
    #: Named for what was observed -- the page was never down, so "restored" would
    #: put an event in the audit trail that did not happen.
    DETAIL_STILL_ACTIVE = "detail_still_active"


class DetailVerdict(StrEnum):
    """Outcome of verifying a missing advertisement's detail page.

    ``STILL_ACTIVE`` deliberately *reduces* sale confidence: a page that still
    loads while the listing is absent from search points at a filter or indexing
    effect, not a sale.
    """

    STILL_ACTIVE = "detail_page_still_active"
    EXPLICITLY_UNAVAILABLE = "detail_page_reports_unavailable"
    HTTP_404 = "http_404"
    HTTP_410 = "http_410"
    REDIRECTED_TO_SEARCH = "redirected_to_search"
    TEMPORARY_ERROR = "temporary_server_error"
    BLOCKED = "blocked_response"
    #: Our parser raised on a response the site actually returned. This is a
    #: defect here, not a statement about the server, and it is not retryable:
    #: the same payload will fail the same way until the code is fixed.
    PARSE_ERROR = "local_parse_error"
    UNKNOWN = "unknown"


class DetailAvailability(StrEnum):
    """What the detail page says about itself.

    Kept strictly apart from :class:`AdStatus` (what the search showed) and
    :class:`SaleLabel` (what we infer). A 410 means the page is gone; it does not
    say why, and on its own it is not a status transition.
    """

    AVAILABLE = "available"
    GONE = "gone"
    REPORTS_UNAVAILABLE = "reports_unavailable"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    BLOCKED = "blocked"
    #: Page loads and the listing is live, but it no longer matches the search.
    OUTSIDE_FILTER = "outside_filter"
    UNKNOWN = "unknown"


#: Verdict -> availability. The mapping is the only place the two vocabularies
#: touch, so neither can silently absorb the other's meaning.
AVAILABILITY_BY_VERDICT: dict[str, DetailAvailability] = {
    "detail_page_still_active": DetailAvailability.AVAILABLE,
    "detail_page_reports_unavailable": DetailAvailability.REPORTS_UNAVAILABLE,
    "http_404": DetailAvailability.GONE,
    "http_410": DetailAvailability.GONE,
    "redirected_to_search": DetailAvailability.REPORTS_UNAVAILABLE,
    "temporary_server_error": DetailAvailability.TEMPORARILY_UNAVAILABLE,
    "blocked_response": DetailAvailability.BLOCKED,
    #: A parse failure tells us nothing about the page, so it must not resolve to
    #: any availability claim — least of all "temporarily unavailable".
    "local_parse_error": DetailAvailability.UNKNOWN,
    "unknown": DetailAvailability.UNKNOWN,
}


class FilterExitReason(StrEnum):
    """Why a still-live listing no longer matches the monitored search."""

    PRICE_BELOW_FILTER = "price_below_filter"
    PRICE_ABOVE_FILTER = "price_above_filter"
    YEAR_OUTSIDE_FILTER = "year_outside_filter"
    CATEGORY_CHANGED = "category_changed"
    COUNTRY_CLASSIFICATION_CHANGED = "country_classification_changed"
    #: Matches every criterion we can check, yet was absent from the results --
    #: an indexing effect rather than a listing change.
    SEARCH_INDEX_INCONSISTENCY = "search_index_inconsistency"


class PublishedAtSource(StrEnum):
    """Provenance of ``published_at``, which decides whether it can be trusted."""

    DETAIL_ABSOLUTE = "detail_absolute"
    DETAIL_RELATIVE = "detail_relative"
    DETAIL_COARSE = "detail_coarse"
    UNKNOWN = "unknown"


#: Sources precise enough to compute a market duration from.
RELIABLE_PUBLISHED_SOURCES: frozenset[PublishedAtSource] = frozenset(
    {PublishedAtSource.DETAIL_ABSOLUTE, PublishedAtSource.DETAIL_RELATIVE}
)


#: Verdicts that constitute evidence of permanent removal.
STRONG_REMOVAL_VERDICTS: frozenset[DetailVerdict] = frozenset(
    {DetailVerdict.HTTP_404, DetailVerdict.HTTP_410}
)


class SaleLabel(StrEnum):
    """Human-facing label for a sale *inference*, never a confirmation."""

    UNKNOWN = "unknown"
    POSSIBLY_SOLD = "possibly_sold"
    LIKELY_SOLD = "likely_sold"
    HIGHLY_LIKELY_SOLD = "highly_likely_sold"
    #: Reserved for an explicit statement by the platform or seller. Nothing in
    #: this codebase assigns it from absence alone.
    CONFIRMED_SOLD = "confirmed_sold"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Scraper interface payloads
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiscoveredCard:
    """One advertisement card as observed on the search surface."""

    platform_ad_id: str
    canonical_url: str
    position: int | None = None
    page_number: int | None = None
    scroll_cycle: int | None = None
    card_title: str | None = None
    card_price_raw: str | None = None
    card_price_normalized: int | None = None
    card_year: str | None = None
    card_mileage_raw: str | None = None
    card_mileage_normalized: int | None = None
    card_location: str | None = None
    card_image_url: str | None = None
    is_promoted: bool | None = None

    def card_hash(self) -> str:
        """Stable hash of the card's volatile content.

        A change here is the cheap trigger for a detail re-scrape, so it covers
        exactly the fields the search surface can change without a new ad id.
        """
        import hashlib

        parts = (
            self.card_title or "",
            self.card_price_raw or "",
            self.card_mileage_raw or "",
            self.card_year or "",
            self.card_location or "",
        )
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DiscoveryResult:
    """Everything one discovery pass produced, plus its own health evidence."""

    search_url: str
    cards: list[DiscoveredCard]
    termination_reason: str
    pages_fetched: int = 0
    pages_failed: int = 0
    duplicate_count: int = 0
    initial_page_ok: bool = True
    reached_verified_end: bool = False
    stabilization_completed: bool = False
    blocked: bool = False
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    raw_evidence_path: str | None = None

    @property
    def discovered_count(self) -> int:
        return len(self.cards)

    def unique_ids(self) -> set[str]:
        return {c.platform_ad_id for c in self.cards}


@dataclass(slots=True)
class DetailResult:
    """Parsed detail page for one advertisement."""

    platform_ad_id: str
    url: str
    ok: bool
    verdict: DetailVerdict = DetailVerdict.UNKNOWN
    http_status: int | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    media: list[dict[str, Any]] = field(default_factory=list)
    content_hash: str | None = None
    parser_version: str | None = None
    error: str | None = None


@dataclass(slots=True)
class StatusTransition:
    """A single state change, with the evidence that justified it."""

    platform_ad_id: str
    advertisement_id: int | None
    previous_status: AdStatus | None
    new_status: AdStatus
    event_type: EventType
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None


@dataclass(slots=True)
class ComparisonResult:
    """Outcome of comparing one run's inventory against prior state."""

    run_id: int
    previous_run_id: int | None
    applied: bool
    reason: str
    #: Advertisements that received a first-discovery *transition* in this run.
    #: Populated from the emitted ``discovered`` events, so it always agrees with
    #: the event log -- including a replay, and including a listing whose row was
    #: created by an earlier run whose comparison was refused.
    new_ids: list[str] = field(default_factory=list)
    #: Advertisement rows physically created while persisting this run's inventory.
    #: An observed fact, recorded even when transitions are refused. Normally equal
    #: to ``new_ids``; it diverges after an unhealthy run created rows.
    observed_new_ids: list[str] = field(default_factory=list)
    seen_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    newly_missing_ids: list[str] = field(default_factory=list)
    likely_removed_ids: list[str] = field(default_factory=list)
    reappeared_ids: list[str] = field(default_factory=list)
    reposted_ids: list[str] = field(default_factory=list)
    transitions: list[StatusTransition] = field(default_factory=list)

    def counters(self) -> dict[str, int]:
        return {
            "new_count": len(self.new_ids),
            "active_count": len(self.seen_ids),
            "missing_count": len(self.missing_ids),
            "removed_count": len(self.likely_removed_ids),
            "reappeared_count": len(self.reappeared_ids),
            "reposted_count": len(self.reposted_ids),
        }


@dataclass(slots=True)
class DurationEstimate:
    """Interval-censored active duration.

    Removal is never observed directly -- it happened somewhere between the last
    sighting and the first miss -- so a single number would be a fabrication. All
    three bounds are stored and the midpoint is explicitly an estimate.
    """

    minimum_active_seconds: float | None
    maximum_active_seconds: float | None
    estimated_active_seconds: float | None
    observation_interval_hours: float | None
    censored: bool

    def as_days(self) -> dict[str, float | None]:
        def days(value: float | None) -> float | None:
            return None if value is None else round(value / 86400.0, 4)

        return {
            "minimum_active_days": days(self.minimum_active_seconds),
            "maximum_active_days": days(self.maximum_active_seconds),
            "estimated_active_days": days(self.estimated_active_seconds),
        }


@dataclass(slots=True)
class SaleScore:
    """Auditable sale-likelihood inference."""

    confidence: float
    label: SaleLabel
    components: list[dict[str, Any]] = field(default_factory=list)

    def as_evidence(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "label": str(self.label),
            "components": self.components,
            "disclaimer": (
                "Confidence expresses inferred likelihood of sale from observed "
                "disappearance. It is not a confirmed sale."
            ),
        }


@dataclass(slots=True)
class RepostMatch:
    """A candidate link between a removed advertisement and a new one."""

    new_platform_ad_id: str
    parent_platform_ad_id: str
    score: float
    matched_on: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunContext:
    """Identity and timing of the run currently executing."""

    run_id: int
    scheduled_for: datetime
    started_at: datetime
    search_url: str
    timezone: str
    scraper_version: str
    configuration_hash: str
    previous_valid_run_id: int | None = None
