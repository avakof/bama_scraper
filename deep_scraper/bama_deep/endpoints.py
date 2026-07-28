"""URL builders and access guards.

Every request the deep scraper makes goes through a builder here, and every URL is
checked against :data:`FORBIDDEN_PATTERNS` *before* it is issued. The guard is
mechanical rather than a matter of discipline, because two classes of endpoint
must never be touched:

* **Write endpoints** (``/cad/api/log/visit``, ``/log/impression``, ``/log/share``,
  ``/event/api/v1/events``) mutate Bama's own view and impression counters.
  Calling them from a scraper would inflate a seller's statistics.
* **Authenticated endpoints** (``/cad/api/carad/*``, ``/cad/api/price?``,
  ``/cad/api/Corporation/phone/*``) return 401 unauthenticated; the phone one in
  particular exists to reveal personal contact data.

``robots.txt`` disallows exactly one path prefix,
``/uploads/BamaImages/CampaignBanner/`` (advertising banners), which
:func:`is_allowed_asset` refuses. Vehicle photos under
``/uploads/BamaImages/VehicleCarImages/`` are permitted.
"""

from __future__ import annotations

import re
from typing import Final
from urllib.parse import quote, urlparse

BASE: Final[str] = "https://bama.ir"

#: Path fragments that must never be requested, with the reason.
FORBIDDEN_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"/cad/api/log/"), "write-only telemetry: inflates Bama's counters"),
    (re.compile(r"/event/api/v1/events"), "analytics write endpoint"),
    (re.compile(r"/nws/api/(?:carReview/report|comment/(?:post|like|dislike))"), "write endpoint"),
    (re.compile(r"/cad/api/carad/"), "authenticated ad-owner view (401)"),
    (re.compile(r"/cad/api/price\?"), "authenticated paginated price list (401)"),
    (re.compile(r"/cad/api/Corporation/phone/"), "personal contact data (401)"),
    (re.compile(r"/prf/api/"), "user account endpoints"),
)

#: robots.txt Disallow.
DISALLOWED_ASSET_PREFIXES: Final[tuple[str, ...]] = ("/uploads/BamaImages/CampaignBanner/",)


class ForbiddenEndpointError(RuntimeError):
    """Raised before issuing a request that must never be made."""


def assert_allowed(url: str) -> None:
    """Raise :class:`ForbiddenEndpointError` if this URL must not be requested."""
    for pattern, reason in FORBIDDEN_PATTERNS:
        if pattern.search(url):
            raise ForbiddenEndpointError(f"{url} is forbidden: {reason}")


def is_allowed_asset(url: str) -> bool:
    """Whether a media asset may be downloaded, per robots.txt."""
    path = urlparse(url).path
    return not any(path.startswith(prefix) for prefix in DISALLOWED_ASSET_PREFIXES)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def ad_api_url(code: str) -> str:
    """Public per-advertisement JSON endpoint (the SSR source of the detail page)."""
    return f"{BASE}/cad/api/detail/{quote(code, safe='')}"


def ad_html_url(path_or_url: str) -> str:
    """Absolute detail-page URL from a stored path or URL."""
    if path_or_url.startswith("http"):
        return path_or_url
    return f"{BASE}/{path_or_url.lstrip('/')}"


def review_detail_url(review_url: str) -> str:
    """CarReview metadata for a model-trim (``review_url`` is a site-relative path)."""
    return f"{BASE}/nws/api/CarReview/carreviewdetail?url={quote(review_url, safe='/')}"


def review_specification_url(research_id: int | str, trim_name: str | None) -> str:
    """The ~110-item technical specification list for a model-trim."""
    trim = quote(trim_name or "", safe="")
    return f"{BASE}/nws/api/CarReview/getspecification?id={research_id}&trimname={trim}"


def review_seo_url(research_id: int | str, trim_name: str | None) -> str:
    """Pre-typed numeric subset of the specifications."""
    trim = quote(trim_name or "", safe="")
    return f"{BASE}/nws/api/CarReview/carreviewseo?id={research_id}&trimName={trim}"


def price_detail_url(brand: str, model: str, trim: str) -> str:
    """Daily market/factory price history for a model-trim."""
    return (
        f"{BASE}/cad/api/price/detail?brand={quote(brand, safe='')}"
        f"&model={quote(model, safe='')}&trim={quote(trim, safe='')}"
    )


def price_brand_url() -> str:
    return f"{BASE}/cad/api/price/brand"


def price_hierarchy_url() -> str:
    return f"{BASE}/cad/api/price/hierarchy"


def dealer_profile_url(dealer_id: int | str) -> str:
    return f"{BASE}/cad/api/Corporation/{dealer_id}"


def dealer_ads_url(dealer_id: int | str) -> str:
    """A dealer's full live inventory in one response (also a coverage cross-check)."""
    return f"{BASE}/cad/api/Corporation/ads/{dealer_id}"


# ---------------------------------------------------------------------------
# Join-key normalization
# ---------------------------------------------------------------------------

_REVIEW_RE: Final[re.Pattern[str]] = re.compile(r"^/car-reviews/(?P<brand>[^/]+)/(?P<rest>.+)$")
_SPECS_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<model>.+?)-specs-(?P<ids>\d+(?:-\d+)*)(?:-(?P<trim>.+))?$"
)


def normalize_review_key(review_url: str | None) -> str | None:
    """Canonicalize a review URL into a stable join key.

    Lowercased, path-only, no trailing slash, so the same model-trim always maps
    to one key regardless of how the URL was written.

    >>> normalize_review_key("/car-reviews/Dena/plusef7p-specs-1481-6mt/")
    '/car-reviews/dena/plusef7p-specs-1481-6mt'
    """
    if not review_url:
        return None
    path = urlparse(review_url.strip()).path or review_url.strip()
    path = path.rstrip("/").lower()
    return path or None


def parse_review_key(review_key: str) -> dict[str, str | None]:
    """Split a review key into its parts.

    Three shapes occur in the wild and all must parse::

        /car-reviews/dena/plusef7p-specs-1481-6mt         -> trim '6mt'
        /car-reviews/samand/soren-specs-1113-459-plusxu7p -> extra id segment
        /car-reviews/shahin/g-specs-1360                  -> no trim suffix
    """
    out: dict[str, str | None] = {
        "brand_slug": None,
        "model_slug": None,
        "trim_slug": None,
        "extra_segment": None,
    }
    match = _REVIEW_RE.match(review_key or "")
    if not match:
        return out
    out["brand_slug"] = match.group("brand")
    specs = _SPECS_RE.match(match.group("rest"))
    if not specs:
        out["model_slug"] = match.group("rest")
        return out
    out["model_slug"] = specs.group("model")
    out["trim_slug"] = specs.group("trim")
    out["extra_segment"] = specs.group("ids")
    return out


def normalize_price_key(price_url: str | None) -> tuple[str | None, dict[str, str | None]]:
    """Turn ``/price/dena_plusef7p_6mt`` into ``("dena|plusef7p|6mt", parts)``.

    Two-segment URLs are legitimate: models with a single trim are published as
    ``/price/shahin_g`` and the endpoint accepts an empty ``trim`` parameter,
    returning the full series set (verified: ``shahin_g`` -> 5 series x 94 points).
    Requiring three segments silently dropped 63 advertisements.

    Returns ``(None, ...)`` only when there is no ``brand_model`` pair at all.
    """
    empty: dict[str, str | None] = {"brand": None, "model": None, "trim": None}
    if not price_url:
        return None, empty
    path = urlparse(price_url.strip()).path or price_url.strip()
    tail = path.rstrip("/").rsplit("/", 1)[-1].lower()
    parts = [p for p in tail.split("_") if p]
    if len(parts) < 2:
        return None, empty
    brand, model = parts[0], parts[1]
    trim = "_".join(parts[2:])  # "" when the model has a single trim
    return f"{brand}|{model}|{trim}", {"brand": brand, "model": model, "trim": trim}


def dealer_id_from_url(url: str | None) -> int | None:
    """Extract the numeric dealer id from ``/dealer/6007``."""
    if not url:
        return None
    match = re.search(r"/dealer/(\d+)", url)
    return int(match.group(1)) if match else None
