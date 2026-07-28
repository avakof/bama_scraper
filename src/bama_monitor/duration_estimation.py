"""Interval-censored active-duration estimation.

Removal is never observed. With one scrape per day, all that is known is that the
listing was present at ``last_seen_at`` and absent at ``first_missing_at``; the
actual removal happened somewhere in between. Reporting a single duration would
be a fabrication of up to the full observation interval.

So three quantities are stored, and the midpoint is labelled an estimate rather
than a time of sale:

    minimum_active = last_seen_at    - first_seen_at
    maximum_active = first_missing_at - first_seen_at
    estimated      = midpoint(last_seen, first_missing) - first_seen_at

Advertisements still present are **right-censored**: their duration so far is a
lower bound only, which is what survival analysis needs to handle them correctly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import AdStatus, DurationEstimate

#: Statuses whose duration is right-censored (still on the market as far as we know).
_CENSORED_STATUSES = frozenset(
    {AdStatus.NEW, AdStatus.ACTIVE, AdStatus.REAPPEARED, AdStatus.MISSING_ONCE}
)


def estimate_duration(
    *,
    first_seen_at: datetime | None,
    last_seen_at: datetime | None,
    first_missing_at: datetime | None,
    status: AdStatus | str,
    now: datetime | None = None,
) -> DurationEstimate:
    """Compute the three duration bounds plus the observation interval."""
    status = AdStatus(str(status))
    if first_seen_at is None:
        return DurationEstimate(None, None, None, None, censored=True)

    anchor_last = last_seen_at or first_seen_at
    minimum = max(0.0, (anchor_last - first_seen_at).total_seconds())

    censored = status in _CENSORED_STATUSES or first_missing_at is None

    # A listing that was seen again *after* its first miss is back on the market:
    # the recorded absence is history, not the end of its life. Closing the
    # interval on the old first_missing_at would produce an upper bound below the
    # lower bound, so it is treated as right-censored instead.
    returned_after_absence = first_missing_at is not None and first_missing_at <= anchor_last

    if first_missing_at is None or returned_after_absence:
        # Never observed absent: only a lower bound exists. `now` extends the
        # lower bound to the present for reporting, but the value stays a bound.
        reference = now or anchor_last
        minimum = max(minimum, (reference - first_seen_at).total_seconds())
        return DurationEstimate(
            minimum_active_seconds=minimum,
            maximum_active_seconds=None,
            estimated_active_seconds=None,
            observation_interval_hours=None,
            censored=True,
        )

    maximum = max(minimum, (first_missing_at - first_seen_at).total_seconds())
    interval_seconds = max(0.0, (first_missing_at - anchor_last).total_seconds())
    midpoint = anchor_last + (first_missing_at - anchor_last) / 2
    estimated = max(0.0, (midpoint - first_seen_at).total_seconds())

    return DurationEstimate(
        minimum_active_seconds=minimum,
        maximum_active_seconds=maximum,
        estimated_active_seconds=estimated,
        observation_interval_hours=round(interval_seconds / 3600.0, 4),
        censored=censored,
    )


def observed_vs_market(record: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Both duration concepts side by side, never merged.

    ``observed_monitoring_*`` measures from ``first_seen_at`` and is available for
    every listing. ``market_*`` measures from ``published_at`` and exists only for
    listings whose whole life was observed. Reporting the first as time-on-market
    would understate every listing that predates monitoring, without bound.
    """
    from .db import parse_ts
    from .publication import market_duration_bounds

    first_seen = parse_ts(record.get("first_seen_at"))
    last_seen = parse_ts(record.get("last_seen_at"))
    first_missing = parse_ts(record.get("first_missing_at"))
    published = parse_ts(record.get("published_at"))
    eligible = bool(record.get("eligible_for_duration_ranking"))

    observed = estimate_duration(
        first_seen_at=first_seen,
        last_seen_at=last_seen,
        first_missing_at=first_missing,
        status=AdStatus(str(record.get("current_status") or AdStatus.UNKNOWN)),
        now=now,
    ).as_days()

    market = market_duration_bounds(
        published_at=published,
        last_seen_at=last_seen,
        first_missing_at=first_missing,
        eligible=eligible,
    )

    return {
        "observed_monitoring_minimum_days": observed["minimum_active_days"],
        "observed_monitoring_maximum_days": observed["maximum_active_days"],
        "observed_monitoring_estimate_days": observed["estimated_active_days"],
        "estimated_market_minimum_days": market["market_minimum_days"],
        "estimated_market_maximum_days": market["market_maximum_days"],
        "estimated_market_estimate_days": market["market_estimate_days"],
        "published_at": published.isoformat() if published else None,
        "published_at_source": record.get("published_at_source"),
        "published_at_reliable": int(bool(record.get("published_at_reliable"))),
        "left_truncated": int(bool(record.get("left_truncated", 1))),
        "eligible_for_duration_ranking": int(eligible),
        "entry_delay_days": (
            round(float(record["entry_delay_seconds"]) / 86400.0, 4)
            if record.get("entry_delay_seconds") is not None
            else None
        ),
    }


def survival_row(record: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Build one survival-analysis row from an advertisement record.

    ``event_observed = 1`` means **the disappearance was observed**, not that a
    sale was confirmed. The distinction is carried in the column names and in
    ``event_type`` so no downstream model can silently reinterpret it.
    """
    from .db import parse_ts
    from .publication import market_duration_bounds

    first_seen = parse_ts(record.get("first_seen_at"))
    last_seen = parse_ts(record.get("last_seen_at"))
    first_missing = parse_ts(record.get("first_missing_at"))
    status = AdStatus(str(record.get("current_status") or AdStatus.UNKNOWN))
    published = parse_ts(record.get("published_at"))

    estimate = estimate_duration(
        first_seen_at=first_seen,
        last_seen_at=last_seen,
        first_missing_at=first_missing,
        status=status,
        now=now,
    )
    days = estimate.as_days()

    # Disappearance is "observed" once the listing has been absent from a valid
    # run; reappearance and reposting both mean it did not leave the market.
    disappearance_observed = status in (
        AdStatus.LIKELY_REMOVED,
        AdStatus.LIKELY_SOLD,
    )
    if disappearance_observed:
        event_type = "disappearance_observed"
    elif status is AdStatus.REPOSTED:
        event_type = "reposted"
    elif status is AdStatus.ACTIVE_OUTSIDE_FILTER:
        # Still for sale, just not inside the monitored search. Observation stops
        # here, so this is a censoring event -- treating it as a disappearance
        # would count a live car as gone.
        event_type = "left_search_filter"
    else:
        event_type = "still_present_or_returned"

    return {
        "vehicle_entity_id": record.get("vehicle_entity_id") or record.get("platform_ad_id"),
        "advertisement_id": record.get("platform_ad_id"),
        "first_seen_at": first_seen.isoformat() if first_seen else None,
        "last_seen_at": last_seen.isoformat() if last_seen else None,
        "first_missing_at": first_missing.isoformat() if first_missing else None,
        # Measured from FIRST OBSERVATION, not from publication. For a listing
        # that predates monitoring this is a lower bound on time on market.
        "duration_measured_from": "first_observation",
        "duration_lower_bound_days": days["minimum_active_days"],
        "duration_upper_bound_days": days["maximum_active_days"],
        "duration_estimate_days": days["estimated_active_days"],
        # Measured from PUBLICATION. Populated only for listings whose whole life
        # was observed; None everywhere else, by design.
        **market_duration_bounds(
            published_at=parse_ts(record.get("published_at")),
            last_seen_at=last_seen,
            first_missing_at=first_missing,
            eligible=bool(record.get("eligible_for_duration_ranking")),
        ),
        "observation_interval_hours": estimate.observation_interval_hours,
        # 1 = disappearance observed. NOT a confirmed sale.
        "event_observed": int(disappearance_observed),
        "event_type": event_type,
        "right_censored": int(estimate.censored),
        # Heuristic evidence score, NOT a calibrated probability. 0.65 does not
        # mean "65% of these sold"; see sale_scoring.METHODOLOGY.
        "sale_evidence_score": record.get("sale_confidence"),
        "sale_label": record.get("sale_label"),
        "status": str(status),
        # Delayed entry. A row with left_truncated = 1 was already on the market
        # when observation began, so its duration is a lower bound on the true
        # time on market and must be handled as left-truncated in any model.
        "left_truncated": int(bool(record.get("left_truncated", 1))),
        "eligible_for_duration_ranking": int(bool(record.get("eligible_for_duration_ranking"))),
        "published_at": published.isoformat() if published else None,
        "entry_delay_days": (
            round(float(record["entry_delay_seconds"]) / 86400.0, 4)
            if record.get("entry_delay_seconds") is not None
            else None
        ),
        "detail_availability": record.get("detail_availability"),
        "filter_exit_reason": record.get("filter_exit_reason"),
        "brand": record.get("brand"),
        "model": record.get("model"),
        "year": record.get("year"),
        "price": record.get("price_normalized"),
        "mileage": record.get("mileage_normalized"),
        "seller_type": record.get("seller_type"),
        "reposted": int(status is AdStatus.REPOSTED or bool(record.get("repost_parent_ad_id"))),
    }
