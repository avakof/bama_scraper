"""Verification of missing advertisements' detail pages.

When a listing vanishes from search, its detail page is the only independent
evidence available about *why*. The mapping from outcome to interpretation is the
whole point of this module, and it is deliberately asymmetric:

* page still loads normally  -> evidence **against** removal (filter or indexing
  effect); this must not be read as a sale;
* page says unavailable      -> evidence for removal;
* HTTP 404 / 410             -> stronger evidence of permanent removal;
* temporary error            -> no evidence either way;
* blocked                    -> no evidence, and a signal to back off.

Raw outcomes are persisted separately from the inference so a verdict can be
re-scored later without re-fetching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from .config import MonitorConfig
from .models import DetailVerdict, RunContext

#: Interpretation attached to each verdict, stored with the evidence so a reader
#: of the database does not have to reconstruct the reasoning.
VERDICT_INTERPRETATION: dict[DetailVerdict, str] = {
    DetailVerdict.STILL_ACTIVE: (
        "detail page loads normally; absence from search is more consistent with a "
        "filter change, promotion change or indexing delay than with removal"
    ),
    DetailVerdict.EXPLICITLY_UNAVAILABLE: (
        "detail page states the advertisement is unavailable; evidence of removal, cause unknown"
    ),
    DetailVerdict.HTTP_404: "detail page returns 404; evidence of permanent removal",
    DetailVerdict.HTTP_410: "detail page returns 410 Gone; strong evidence of permanent removal",
    DetailVerdict.REDIRECTED_TO_SEARCH: (
        "detail page redirected to search; consistent with removal but not conclusive"
    ),
    DetailVerdict.TEMPORARY_ERROR: "temporary server error; no evidence about removal",
    DetailVerdict.BLOCKED: "access blocked; no evidence about removal",
    DetailVerdict.UNKNOWN: "outcome could not be classified; no evidence about removal",
}


@dataclass(slots=True)
class VerificationOutcome:
    platform_ad_id: str
    advertisement_id: int
    url: str
    verdict: DetailVerdict
    http_status: int | None
    evidence: dict[str, Any]
    #: Parsed detail fields, carried so the caller can re-check the listing
    #: against the search bounds. A live page is only half the answer: whether it
    #: still matches the filter decides between "left the search" and "left the
    #: market".
    fields: dict[str, Any] = dataclass_field(default_factory=dict)


async def verify_missing(
    detail_scraper: Any,
    targets: Sequence[tuple[int, str, str]],
    cfg: MonitorConfig,
    context: RunContext,
) -> list[VerificationOutcome]:
    """Check each missing advertisement's detail page.

    ``targets`` is ``(advertisement_id, platform_ad_id, url)``. Verification stops
    early on a blocked response: continuing would neither yield evidence nor be
    polite.
    """
    outcomes: list[VerificationOutcome] = []
    for advertisement_id, platform_ad_id, url in targets:
        result = await detail_scraper.scrape(url)
        verdict = result.verdict
        evidence = {
            "verdict": str(verdict),
            "http_status": result.http_status,
            "interpretation": VERDICT_INTERPRETATION.get(verdict, ""),
            "checked_at": context.started_at.isoformat(),
            "run_id": context.run_id,
            "error": result.error,
            # Explicit so no consumer can mistake this for a sale confirmation.
            "is_sale_confirmation": False,
        }
        outcomes.append(
            VerificationOutcome(
                platform_ad_id=platform_ad_id,
                advertisement_id=advertisement_id,
                url=url,
                verdict=verdict,
                http_status=result.http_status,
                evidence=evidence,
                fields=dict(result.fields or {}),
            )
        )
        if verdict is DetailVerdict.BLOCKED:
            evidence["stopped_early"] = True
            break
    return outcomes


def select_verification_targets(
    missing: Sequence[dict[str, Any]], cfg: MonitorConfig, *, limit: int | None = None
) -> list[tuple[int, str, str]]:
    """Choose which missing advertisements to verify.

    Newly missing listings are prioritised: their verdict is what distinguishes a
    filter effect from a real removal, and it is most informative on the first day
    of absence. Long-absent listings are re-checked far less often.
    """
    ranked = sorted(
        missing,
        key=lambda r: (int(r.get("consecutive_misses") or 0), r.get("platform_ad_id") or ""),
    )
    cap = limit if limit is not None else cfg.detail.max_details_per_run
    return [(int(r["id"]), str(r["platform_ad_id"]), str(r["canonical_url"])) for r in ranked[:cap]]
