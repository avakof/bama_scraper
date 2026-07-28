"""Publication time, and what it does and does not license you to compute.

``first_seen_at`` is when **monitoring** first observed a listing. ``published_at``
is when the **seller** published it. For every advertisement that already existed
on the first monitoring day these differ by an unknown amount: monitoring starting
on 1 August and seeing a listing disappear on 3 August has observed two days, but
the listing may have been on the market since June.

Such rows are **left-truncated** (delayed entry). Ranking them by observed duration
would systematically flatter long-standing listings, because their earlier life is
invisible. So two durations are kept apart:

    observed_monitoring_duration = disappearance interval - first_seen_at
    estimated_market_duration    = disappearance interval - published_at

and the second is computed *only* when the publication time is reliable and the
listing entered observation close enough to publication for its whole life to have
been seen.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from .models import RELIABLE_PUBLISHED_SOURCES, PublishedAtSource

#: How close first observation must be to publication for the listing's full life
#: to count as observed. One daily cycle plus half a cycle of slack.
DEFAULT_ENTRY_TOLERANCE = timedelta(hours=36)

#: Phrases that give an order of magnitude but not a usable instant.
_COARSE_MARKERS = (
    "لحظاتی پیش",
    "دقایقی پیش",
    "هم اکنون",
    "همین الان",
    "چند لحظه پیش",
)


def classify_published_source(raw_text: str | None, parsed: Any) -> PublishedAtSource:
    """How precise is this publication time?

    ``لحظاتی پیش`` ("moments ago") is *coarse*: it pins the day but not the hour,
    which is fine for "published today" and useless for an hours-scale duration.
    An absolute Jalali date or an explicit relative offset is usable.
    """
    if parsed is None:
        return PublishedAtSource.UNKNOWN
    text = (raw_text or "").strip()
    if not text:
        return PublishedAtSource.UNKNOWN
    if any(marker in text for marker in _COARSE_MARKERS):
        return PublishedAtSource.DETAIL_COARSE
    if "/" in text or any(ch.isdigit() for ch in text) and "پیش" not in text:
        return PublishedAtSource.DETAIL_ABSOLUTE
    if "پیش" in text or "دیروز" in text or "پریروز" in text:
        return PublishedAtSource.DETAIL_RELATIVE
    return PublishedAtSource.DETAIL_COARSE


def to_datetime(value: Any) -> datetime | None:
    """Accept the UNIX timestamp the detail parsers produce, or a datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def truncation_status(
    *,
    published_at: datetime | None,
    published_source: PublishedAtSource | str | None,
    first_seen_at: datetime | None,
    tolerance: timedelta = DEFAULT_ENTRY_TOLERANCE,
) -> dict[str, Any]:
    """Decide whether this listing's full time on the market was observed.

    Returns the four stored fields. ``left_truncated`` defaults to *True* whenever
    the answer is unknown: an unverifiable listing must not be silently promoted
    into a ranking it does not qualify for.
    """
    source = (
        PublishedAtSource(str(published_source))
        if published_source and str(published_source) in {s.value for s in PublishedAtSource}
        else PublishedAtSource.UNKNOWN
    )
    reliable = source in RELIABLE_PUBLISHED_SOURCES and published_at is not None

    if published_at is None or first_seen_at is None:
        return {
            "published_at": published_at,
            "published_at_source": str(source),
            "published_at_reliable": False,
            "left_truncated": True,
            "eligible_for_duration_ranking": False,
            "entry_delay_seconds": None,
        }

    delay = (first_seen_at - published_at).total_seconds()
    # A negative delay means the listing was seen before it was published, which
    # cannot happen: treat it as an unusable publication time rather than as a
    # zero delay that would sneak the row into the rankings.
    usable_delay = delay >= -3600
    within = usable_delay and delay <= tolerance.total_seconds()

    return {
        "published_at": published_at,
        "published_at_source": str(source),
        "published_at_reliable": bool(reliable and usable_delay),
        "left_truncated": not within,
        "eligible_for_duration_ranking": bool(reliable and within),
        "entry_delay_seconds": delay if usable_delay else None,
    }


def market_duration_bounds(
    *,
    published_at: datetime | None,
    last_seen_at: datetime | None,
    first_missing_at: datetime | None,
    eligible: bool,
) -> dict[str, float | None]:
    """Time on the market measured from publication, when that is legitimate.

    Returns all-``None`` for an ineligible listing rather than a number with a
    caveat attached, because a number with a caveat is what ends up in a chart
    without the caveat.
    """
    if not eligible or published_at is None:
        return {
            "market_minimum_days": None,
            "market_maximum_days": None,
            "market_estimate_days": None,
        }

    def days(delta: float) -> float:
        return round(delta / 86400.0, 4)

    anchor = last_seen_at or published_at
    minimum = max(0.0, (anchor - published_at).total_seconds())
    if first_missing_at is None:
        return {
            "market_minimum_days": days(minimum),
            "market_maximum_days": None,
            "market_estimate_days": None,
        }
    maximum = max(minimum, (first_missing_at - published_at).total_seconds())
    midpoint = anchor + (first_missing_at - anchor) / 2
    estimate = max(0.0, (midpoint - published_at).total_seconds())
    return {
        "market_minimum_days": days(minimum),
        "market_maximum_days": days(maximum),
        "market_estimate_days": days(estimate),
    }
