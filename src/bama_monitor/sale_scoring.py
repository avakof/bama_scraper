"""Sale-likelihood inference.

This module produces a *confidence*, never a verdict. Its output answers "how
consistent is the observed evidence with this vehicle having been sold?", and the
answer is always accompanied by the individual components that produced it so a
reviewer can disagree with any one of them.

The scoring is deliberately rule-based rather than learned: there is no labelled
ground truth for sales on this platform, so a model would be fitting to its own
assumptions. Weights live in configuration and every component is stored.

``SaleLabel.CONFIRMED_SOLD`` exists in the enum but is never produced here. It is
reserved for an explicit statement by the platform or seller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .config import SaleScoringWeights
from .models import DetailVerdict, SaleLabel, SaleScore


@dataclass(slots=True)
class ScoringInputs:
    """Everything the rules need, gathered by the caller."""

    consecutive_misses: int
    removal_confirmation_misses: int
    detail_verdict: DetailVerdict | None = None
    #: Number of consecutive verification checks that returned 404/410.
    persistent_unavailable_checks: int = 0
    had_price_reduction: bool = False
    first_seen_at: datetime | None = None
    first_missing_at: datetime | None = None
    reappeared: bool = False
    reposted: bool = False
    #: True when the seller has another live listing for the same vehicle.
    seller_reposted_same_vehicle: bool = False
    #: True when the listing was found outside the monitored filter.
    seen_outside_filter: bool = False
    now: datetime | None = None


def score_sale(inputs: ScoringInputs, weights: SaleScoringWeights) -> SaleScore:
    """Compute confidence in ``[0, 1]`` with an auditable component list."""
    components: list[dict[str, Any]] = []

    def add(name: str, weight: float, why: str) -> None:
        components.append({"rule": name, "weight": round(weight, 4), "reason": why})

    # --- evidence consistent with a sale ---------------------------------
    if inputs.consecutive_misses >= inputs.removal_confirmation_misses:
        add(
            "absent_two_valid_runs",
            weights.absent_two_valid_runs,
            f"absent from {inputs.consecutive_misses} consecutive valid runs "
            f"(threshold {inputs.removal_confirmation_misses})",
        )

    verdict = inputs.detail_verdict
    if verdict is DetailVerdict.EXPLICITLY_UNAVAILABLE:
        add(
            "detail_explicitly_unavailable",
            weights.detail_explicitly_unavailable,
            "detail page states the listing is unavailable",
        )
    if verdict in (DetailVerdict.HTTP_404, DetailVerdict.HTTP_410):
        add(
            "detail_persistent_404_410",
            weights.detail_persistent_404_410,
            f"detail page returns {verdict}"
            + (
                f" on {inputs.persistent_unavailable_checks} consecutive checks"
                if inputs.persistent_unavailable_checks > 1
                else ""
            ),
        )

    if inputs.had_price_reduction:
        add(
            "recent_price_reduction",
            weights.recent_price_reduction,
            "price was reduced before the listing disappeared",
        )

    if inputs.first_seen_at and inputs.first_missing_at:
        age_days = (inputs.first_missing_at - inputs.first_seen_at).total_seconds() / 86400.0
        if age_days <= weights.soon_after_publication_days:
            add(
                "disappeared_soon_after_publication",
                weights.disappeared_soon_after_publication,
                f"disappeared {age_days:.1f} days after first being seen",
            )

    if not inputs.seller_reposted_same_vehicle and not inputs.reposted:
        add(
            "seller_did_not_repost",
            weights.seller_did_not_repost,
            "no matching repost found for this vehicle",
        )

    if inputs.first_missing_at:
        reference = inputs.now or datetime.now(inputs.first_missing_at.tzinfo)
        if reference - inputs.first_missing_at >= timedelta(days=7):
            add(
                "disappearance_persists_seven_days",
                weights.disappearance_persists_seven_days,
                "still absent seven or more days after the first miss",
            )

    # --- evidence against a sale -----------------------------------------
    # These are the guards that keep a live listing from being labelled sold.
    if inputs.reappeared:
        add(
            "reappeared",
            weights.reappeared,
            "listing was observed again after being absent",
        )
    if inputs.reposted or inputs.seller_reposted_same_vehicle:
        add(
            "vehicle_reposted",
            weights.vehicle_reposted,
            "the same vehicle appears under another advertisement id",
        )
    if verdict is DetailVerdict.STILL_ACTIVE:
        add(
            "detail_remains_accessible",
            weights.detail_remains_accessible,
            "detail page still loads normally, so absence is probably a filter or "
            "indexing effect rather than a removal",
        )
    if inputs.seen_outside_filter:
        add(
            "appears_outside_original_filter",
            weights.appears_outside_original_filter,
            "listing was found outside the monitored search filter",
        )

    raw = sum(c["weight"] for c in components)
    confidence = max(0.0, min(1.0, raw))
    return SaleScore(
        confidence=round(confidence, 4), label=label_for(confidence, weights), components=components
    )


def label_for(confidence: float, weights: SaleScoringWeights) -> SaleLabel:
    """Bucket a confidence value. Never returns ``CONFIRMED_SOLD``."""
    if confidence >= weights.highly_likely_sold_at:
        return SaleLabel.HIGHLY_LIKELY_SOLD
    if confidence >= weights.likely_sold_at:
        return SaleLabel.LIKELY_SOLD
    if confidence >= weights.possibly_sold_at:
        return SaleLabel.POSSIBLY_SOLD
    return SaleLabel.UNKNOWN


METHODOLOGY = """\
Sale-confidence methodology
===========================

The system observes one fact: an advertisement stopped appearing in the monitored
search inventory. It does not observe sales. Any statement about a sale is an
inference, and this module makes that inference explicit, bounded and auditable.

Inputs are the observed record only: how many consecutive *valid* runs the listing
was absent from, what its detail page returned when checked, whether its price had
been reduced, how soon after publication it vanished, whether a matching vehicle
was reposted, and whether it later reappeared.

Positive components raise confidence; negative components lower it. The two
strongest negative signals exist specifically to prevent false positives:

* a detail page that still loads means the listing probably left the *filter*,
  not the market;
* a matching repost means the vehicle is still for sale under a new id.

Confidence is clamped to [0, 1] and bucketed into unknown / possibly_sold /
likely_sold / highly_likely_sold. The bucket names are inferences. The separate
label `confirmed_sold` is never assigned by this module; it is reserved for an
explicit statement from the platform or the seller, which this data source does
not provide.

Every component -- its rule name, weight and reason -- is persisted with the
advertisement and in the event log, so any score can be recomputed or contested
after the fact, and changing a weight does not rewrite history.
"""
