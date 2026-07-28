"""Does this listing still match the monitored search?

An advertisement can vanish from the results while remaining perfectly alive: a
seller drops the asking price under ``priceFrom``, corrects the production year,
or the listing is recategorised. Absence then says nothing about the market, and
counting it as a disappearance produces a false removal — and, downstream, a
false sale.

So absence is explained before it is interpreted. When the detail page still
loads, the listing's own fields are re-checked against the filter bounds:

* fails a bound  -> ``filter_exit`` with the specific reason;
* passes them all -> ``search_index_inconsistency`` (it should have been in the
  results and was not, which is an indexing effect, not a listing change).

The bounds come from the search URL itself, parsed by the same
``bama_scraper.discovery.parse_filters`` used during discovery, so the monitor
cannot drift from what was actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import FilterExitReason

#: Prices this far below the bound are treated as a genuine change rather than a
#: rounding or currency-unit artefact (1 million Toman).
PRICE_EPSILON = 1_000_000


@dataclass(frozen=True)
class SearchFilter:
    """The bounds a listing must satisfy to appear in the monitored search.

    Every field is optional: an absent bound is not a constraint. ``year_from``
    and ``price_from`` are *lower* bounds — verified against the live endpoint,
    which maps ``year=1397-2018,`` to ``yearFrom`` and ``price=1000000000`` to
    ``priceFrom``.
    """

    year_from: int | None = None
    year_to: int | None = None
    price_from: int | None = None
    price_to: int | None = None
    body: str | None = None
    country: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any((self.year_from, self.year_to, self.price_from, self.price_to))

    def describe(self) -> dict[str, Any]:
        return {
            "year_from": self.year_from,
            "year_to": self.year_to,
            "price_from": self.price_from,
            "price_to": self.price_to,
            "body": self.body,
            "country": self.country,
        }


def parse_search_filter(search_url: str) -> SearchFilter:
    """Read the filter bounds out of the monitored search URL."""
    try:
        from bama_scraper.discovery import parse_filters
    except Exception:  # pragma: no cover - the scraper package is always present
        return SearchFilter(raw={})

    described = parse_filters(search_url) or {}
    # `parse_filters` returns {api_params, interpretation, raw_query}; the bounds
    # actually sent to the endpoint live under `api_params`. Reading the top level
    # instead would yield no bounds at all and silently turn every filter-exit
    # check into "inconclusive".
    parsed: dict[str, Any] = dict(described.get("api_params") or {})
    if not parsed:
        parsed = {k: v for k, v in described.items() if isinstance(v, str)}

    def as_int(value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        # `1397-2018` encodes one year in both calendars; the Jalali part leads.
        head = text.split("-", 1)[0].replace(",", "")
        try:
            return int(head)
        except ValueError:
            return None

    return SearchFilter(
        year_from=as_int(parsed.get("yearFrom") or parsed.get("year_from")),
        year_to=as_int(parsed.get("yearTo") or parsed.get("year_to")),
        price_from=as_int(parsed.get("priceFrom") or parsed.get("price_from")),
        price_to=as_int(parsed.get("priceTo") or parsed.get("price_to")),
        body=(parsed.get("body") or None),
        country=(parsed.get("country") or None),
        raw=dict(parsed),
    )


def assert_bounds_parsed(search_url: str, search: SearchFilter) -> list[str]:
    """Warn when the URL declares a bound that did not survive parsing.

    A filter that parses to nothing makes every absence look inconclusive, which
    is indistinguishable from "no listing ever left the filter". This turns that
    silent degradation into a visible warning.
    """
    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(search_url).query)
    warnings: list[str] = []
    if "price" in query and search.price_from is None and search.price_to is None:
        warnings.append(
            f"search URL declares price={query['price'][0]!r} but no price bound parsed"
        )
    if "year" in query and search.year_from is None and search.year_to is None:
        warnings.append(f"search URL declares year={query['year'][0]!r} but no year bound parsed")
    return warnings


def _year_of(fields: dict[str, Any]) -> int | None:
    """Jalali production year from whichever field carries it."""
    for key in ("year_jalali", "year", "year_text", "production_year"):
        value = fields.get(key)
        if value is None:
            continue
        try:
            year = int(str(value).strip()[:4])
        except (ValueError, TypeError):
            continue
        # A Gregorian year given where a Jalali one is expected: convert rather
        # than reject, so a parser change cannot manufacture filter exits.
        if year > 1500:
            year -= 621
        return year
    return None


def _price_of(fields: dict[str, Any]) -> int | None:
    for key in ("price_toman", "price_normalized", "price"):
        value = fields.get(key)
        if value is None:
            continue
        try:
            price = int(value)
        except (ValueError, TypeError):
            continue
        # 0 is Bama's encoding for "negotiable"/"call for price", not a real
        # price, and must never be read as "below the minimum".
        return price if price > 0 else None
    return None


def evaluate(
    fields: dict[str, Any], search: SearchFilter
) -> tuple[FilterExitReason | None, dict[str, Any]]:
    """Check one listing's own fields against the search bounds.

    Returns ``(reason, evidence)``. ``reason is None`` means every bound that
    could be checked was satisfied. The evidence dict always records what was
    compared, including which checks had to be skipped for want of a field —
    silence about a skipped check would make an unverifiable listing look
    verified.
    """
    evidence: dict[str, Any] = {
        "filter": search.describe(),
        "observed": {},
        "checked": [],
        "skipped": [],
    }

    price = _price_of(fields)
    year = _year_of(fields)
    evidence["observed"] = {
        "price_toman": price,
        "year_jalali": year,
        "body": fields.get("body_type_fa") or fields.get("body_type"),
        "country": fields.get("country") or fields.get("origin"),
    }

    if search.price_from is not None:
        if price is None:
            evidence["skipped"].append("price_from")
        else:
            evidence["checked"].append("price_from")
            if price < search.price_from - PRICE_EPSILON:
                evidence["reason_detail"] = (
                    f"asking price {price:,} is below the monitored minimum {search.price_from:,}"
                )
                return FilterExitReason.PRICE_BELOW_FILTER, evidence

    if search.price_to is not None:
        if price is None:
            evidence["skipped"].append("price_to")
        else:
            evidence["checked"].append("price_to")
            if price > search.price_to + PRICE_EPSILON:
                evidence["reason_detail"] = (
                    f"asking price {price:,} is above the monitored maximum {search.price_to:,}"
                )
                return FilterExitReason.PRICE_ABOVE_FILTER, evidence

    if search.year_from is not None:
        if year is None:
            evidence["skipped"].append("year_from")
        else:
            evidence["checked"].append("year_from")
            if year < search.year_from:
                evidence["reason_detail"] = (
                    f"production year {year} is older than the monitored minimum {search.year_from}"
                )
                return FilterExitReason.YEAR_OUTSIDE_FILTER, evidence

    if search.year_to is not None and year is not None:
        evidence["checked"].append("year_to")
        if year > search.year_to:
            evidence["reason_detail"] = (
                f"production year {year} is newer than the monitored maximum {search.year_to}"
            )
            return FilterExitReason.YEAR_OUTSIDE_FILTER, evidence

    return None, evidence


def classify_absence(
    fields: dict[str, Any], search: SearchFilter
) -> tuple[FilterExitReason, dict[str, Any]]:
    """Explain why a *still-live* listing was absent from the results.

    Only call this when the detail page loaded and the listing is alive. If no
    bound is violated the absence is attributed to the search index rather than
    to the listing, which keeps "we cannot explain this" visible instead of
    letting it masquerade as a removal.
    """
    reason, evidence = evaluate(fields, search)
    if reason is not None:
        return reason, evidence
    evidence["reason_detail"] = (
        "the listing satisfies every bound that could be checked yet was absent "
        "from the results; attributed to the search index, not to the listing"
    )
    return FilterExitReason.SEARCH_INDEX_INCONSISTENCY, evidence
