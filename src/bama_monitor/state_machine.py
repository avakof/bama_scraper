"""Advertisement state machine.

Kept as pure functions over plain values so every transition is unit-testable
without a database, and so the transition table itself is inspectable.

Two invariants are enforced here rather than left to callers:

* **One miss is never a removal.** ``consecutive_misses == 1`` yields
  ``missing_once``; only ``removal_confirmation_misses`` (default 2) consecutive
  *valid-run* misses yield ``likely_removed``.
* **A reappearance always wins.** Being seen again returns an advertisement to the
  active population from any state, including ``likely_sold`` and ``reposted`` —
  the observation overrides the earlier inference, and the inference's confidence
  is reduced by the scoring layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .models import (
    COMPARABLE_STATUSES,
    AdStatus,
    EventType,
)


@dataclass(frozen=True)
class AdState:
    """The mutable fields of an advertisement that transitions depend on."""

    status: AdStatus
    consecutive_misses: int = 0
    total_seen_runs: int = 0
    total_missing_runs: int = 0
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    first_missing_at: datetime | None = None
    confirmed_removed_at: datetime | None = None
    reappeared_at: datetime | None = None


@dataclass(frozen=True)
class TransitionOutcome:
    """Result of applying one observation to one advertisement."""

    state: AdState
    events: tuple[tuple[EventType, dict[str, Any]], ...] = ()

    @property
    def status(self) -> AdStatus:
        return self.state.status


#: Human-readable transition table, also used by the tests and the docs so the
#: three cannot drift apart.
TRANSITION_TABLE: tuple[tuple[str, str, str], ...] = (
    ("(unknown to us)", "seen in a valid run", "new"),
    ("new", "seen in a valid run", "active"),
    ("active", "seen in a valid run", "active"),
    ("reappeared", "seen in a valid run", "active"),
    ("active | new | reappeared", "absent for 1 valid run", "missing_once"),
    ("missing_once", "absent for another valid run", "likely_removed"),
    ("likely_removed", "still absent", "likely_removed (confidence rises)"),
    ("missing_once | likely_removed", "seen again", "reappeared"),
    ("likely_removed", "strong sale evidence", "likely_sold"),
    ("likely_removed", "matching new advertisement found", "reposted"),
    ("likely_sold | reposted", "seen again", "reappeared"),
    ("missing_once | likely_removed", "live page fails a filter bound", "active_outside_filter"),
    ("active_outside_filter", "seen in the search again", "active"),
)


def on_seen(
    state: AdState | None,
    *,
    observed_at: datetime,
    is_first_discovery: bool = False,
) -> TransitionOutcome:
    """Apply a positive observation.

    A previously unknown advertisement becomes ``new``; an already-known one
    becomes ``active``, except that an advertisement returning from absence is
    routed through ``reappeared`` so the round trip is visible in its history
    rather than being quietly overwritten.
    """
    if state is None or is_first_discovery:
        new_state = AdState(
            status=AdStatus.NEW,
            consecutive_misses=0,
            total_seen_runs=(state.total_seen_runs + 1) if state else 1,
            total_missing_runs=state.total_missing_runs if state else 0,
            first_seen_at=(state.first_seen_at if state and state.first_seen_at else observed_at),
            last_seen_at=observed_at,
        )
        return TransitionOutcome(
            new_state,
            (
                (EventType.DISCOVERED, {"observed_at": observed_at.isoformat()}),
                (EventType.SEEN, {"observed_at": observed_at.isoformat()}),
            ),
        )

    was_absent = state.status in (
        AdStatus.MISSING_ONCE,
        AdStatus.LIKELY_REMOVED,
        AdStatus.LIKELY_SOLD,
        AdStatus.REPOSTED,
    )

    if was_absent:
        # The observation contradicts the earlier inference. Record it explicitly:
        # this is the single most important guard against calling a live listing sold.
        new_state = AdState(
            status=AdStatus.REAPPEARED,
            consecutive_misses=0,
            total_seen_runs=state.total_seen_runs + 1,
            total_missing_runs=state.total_missing_runs,
            first_seen_at=state.first_seen_at or observed_at,
            last_seen_at=observed_at,
            # The absence really happened, so its timestamps are preserved as
            # history; they are simply no longer the end of the story.
            first_missing_at=state.first_missing_at,
            confirmed_removed_at=state.confirmed_removed_at,
            reappeared_at=observed_at,
        )
        return TransitionOutcome(
            new_state,
            (
                (
                    EventType.REAPPEARED,
                    {
                        "observed_at": observed_at.isoformat(),
                        "previous_status": str(state.status),
                        "misses_before_reappearance": state.consecutive_misses,
                        "note": (
                            "advertisement was observed again; any prior removal or "
                            "sale inference is superseded by this observation"
                        ),
                    },
                ),
                (EventType.SEEN, {"observed_at": observed_at.isoformat()}),
            ),
        )

    if state.status is AdStatus.ACTIVE_OUTSIDE_FILTER:
        # It matched the search again -- the seller raised the price back over the
        # bound, or corrected the year. Recorded as its own event so the round trip
        # out of and back into the filter is visible in the audit trail.
        new_state = AdState(
            status=AdStatus.ACTIVE,
            consecutive_misses=0,
            total_seen_runs=state.total_seen_runs + 1,
            total_missing_runs=state.total_missing_runs,
            first_seen_at=state.first_seen_at or observed_at,
            last_seen_at=observed_at,
            first_missing_at=state.first_missing_at,
            confirmed_removed_at=state.confirmed_removed_at,
            reappeared_at=state.reappeared_at,
        )
        return TransitionOutcome(
            new_state,
            (
                (
                    EventType.FILTER_REENTRY,
                    {
                        "observed_at": observed_at.isoformat(),
                        "note": (
                            "listing matches the monitored search again; it never "
                            "left the market, so this is not a reappearance from "
                            "absence"
                        ),
                    },
                ),
                (EventType.SEEN, {"observed_at": observed_at.isoformat()}),
            ),
        )

    # new -> active on the second sighting; active/other stay active.
    new_state = AdState(
        status=AdStatus.ACTIVE,
        consecutive_misses=0,
        total_seen_runs=state.total_seen_runs + 1,
        total_missing_runs=state.total_missing_runs,
        first_seen_at=state.first_seen_at or observed_at,
        last_seen_at=observed_at,
        first_missing_at=state.first_missing_at,
        confirmed_removed_at=state.confirmed_removed_at,
        reappeared_at=state.reappeared_at,
    )
    return TransitionOutcome(
        new_state, ((EventType.SEEN, {"observed_at": observed_at.isoformat()}),)
    )


def on_missing(
    state: AdState,
    *,
    run_started_at: datetime,
    removal_confirmation_misses: int = 2,
) -> TransitionOutcome:
    """Apply a negative observation from a **valid** run.

    Callers must not invoke this for an invalid run: doing so is the corruption
    this package exists to prevent, and :mod:`bama_monitor.inventory_comparison`
    gates it on :class:`~bama_monitor.run_health.HealthReport`.
    """
    misses = state.consecutive_misses + 1
    events: list[tuple[EventType, dict[str, Any]]] = []

    first_missing_at = state.first_missing_at or run_started_at
    confirmed_removed_at = state.confirmed_removed_at
    status = state.status

    if misses == 1:
        status = AdStatus.MISSING_ONCE
        events.append(
            (
                EventType.MISSING_FIRST_TIME,
                {
                    "run_started_at": run_started_at.isoformat(),
                    "consecutive_misses": misses,
                    "note": (
                        "absent from one valid run; this is an observed "
                        "disappearance from inventory, not a removal or a sale"
                    ),
                },
            )
        )
    elif misses >= removal_confirmation_misses:
        already_removed = state.status is AdStatus.LIKELY_REMOVED
        status = AdStatus.LIKELY_REMOVED
        if confirmed_removed_at is None:
            confirmed_removed_at = run_started_at
        if not already_removed:
            events.append(
                (
                    EventType.REMOVAL_CONFIRMED,
                    {
                        "run_started_at": run_started_at.isoformat(),
                        "consecutive_misses": misses,
                        "threshold": removal_confirmation_misses,
                        "note": (
                            "absent from the required number of consecutive valid "
                            "runs; removal is inferred, cause unknown"
                        ),
                    },
                )
            )
    else:
        # Threshold above 2: intermediate misses stay missing_once.
        status = AdStatus.MISSING_ONCE

    new_state = AdState(
        status=status,
        consecutive_misses=misses,
        total_seen_runs=state.total_seen_runs,
        total_missing_runs=state.total_missing_runs + 1,
        first_seen_at=state.first_seen_at,
        last_seen_at=state.last_seen_at,
        first_missing_at=first_missing_at,
        confirmed_removed_at=confirmed_removed_at,
        reappeared_at=state.reappeared_at,
    )
    return TransitionOutcome(new_state, tuple(events))


def promote_to_likely_sold(
    state: AdState, *, confidence: float, evidence: dict[str, Any], at: datetime
) -> TransitionOutcome:
    """Move ``likely_removed`` to ``likely_sold`` on strong inferred evidence.

    Only reachable from ``likely_removed``: a listing that has been absent for
    one run, or is currently visible, can never be labelled sold.
    """
    if state.status is not AdStatus.LIKELY_REMOVED:
        return TransitionOutcome(state)
    new_state = AdState(
        status=AdStatus.LIKELY_SOLD,
        consecutive_misses=state.consecutive_misses,
        total_seen_runs=state.total_seen_runs,
        total_missing_runs=state.total_missing_runs,
        first_seen_at=state.first_seen_at,
        last_seen_at=state.last_seen_at,
        first_missing_at=state.first_missing_at,
        confirmed_removed_at=state.confirmed_removed_at,
        reappeared_at=state.reappeared_at,
    )
    payload = dict(evidence)
    payload["note"] = "inferred from disappearance and supporting evidence; NOT a confirmed sale"
    return TransitionOutcome(
        new_state, ((EventType.LIKELY_SOLD, payload | {"at": at.isoformat()}),)
    )


def mark_reposted(
    state: AdState, *, child_platform_ad_id: str, score: float, evidence: dict[str, Any]
) -> TransitionOutcome:
    """Mark an absent advertisement as reposted under a new id.

    A repost is explicitly *not* a sale: the vehicle is still on the market, so
    the scoring layer subtracts confidence when this link exists.
    """
    if state.status not in (AdStatus.LIKELY_REMOVED, AdStatus.MISSING_ONCE, AdStatus.LIKELY_SOLD):
        return TransitionOutcome(state)
    new_state = AdState(
        status=AdStatus.REPOSTED,
        consecutive_misses=state.consecutive_misses,
        total_seen_runs=state.total_seen_runs,
        total_missing_runs=state.total_missing_runs,
        first_seen_at=state.first_seen_at,
        last_seen_at=state.last_seen_at,
        first_missing_at=state.first_missing_at,
        confirmed_removed_at=state.confirmed_removed_at,
        reappeared_at=state.reappeared_at,
    )
    return TransitionOutcome(
        new_state,
        (
            (
                EventType.POSSIBLE_REPOST,
                {
                    "child_platform_ad_id": child_platform_ad_id,
                    "score": score,
                    "note": "same vehicle appears under a new advertisement id; not a sale",
                    **evidence,
                },
            ),
        ),
    )


def is_comparable(status: AdStatus) -> bool:
    """Whether an advertisement still belongs to the absence-checking population."""
    return status in COMPARABLE_STATUSES
