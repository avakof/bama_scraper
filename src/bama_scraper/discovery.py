"""Search-result discovery.

Two independent mechanisms, both producing the same :class:`DiscoveredAd`
records so their inventories can be compared during the completeness audit:

* :func:`discover_via_api` — the public JSON endpoint the page itself calls.
* :func:`discover_via_scroll` — Playwright infinite-scroll with adaptive
  termination, used as fallback and as the independent verification pass.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import re
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Config
from .logging_config import get_logger
from .models import DiscoveredAd, NetworkCandidate
from .normalization import normalize_mileage, normalize_price, normalize_text, parse_year
from .storage import Storage

log = get_logger(__name__)

BASE = "https://bama.ir"
SEARCH_API = f"{BASE}/cad/api/search"

#: Detail URLs look like ``/car/detail-<code>-<slug>``.
AD_URL_RE = re.compile(r"/car/detail-([A-Za-z0-9]+)(?:-[^/?#]*)?")

#: Query parameters that carry no addressing meaning and break deduplication.
TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "yclid",
        "msclkid",
        "ref",
        "referrer",
        "from",
        "source",
        "_ga",
        "_gl",
        "adid",
        "campaign",
        "cid",
    }
)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def canonical_ad_url(href: str, base: str = BASE) -> str | None:
    """Normalize any advertisement href into a canonical absolute URL.

    Drops fragments and tracking parameters, forces https + the bare host, and
    strips trailing slashes so the same ad always yields one key.

    >>> canonical_ad_url("/car/detail-abc123-pride-1400?utm_source=x#gallery")
    'https://bama.ir/car/detail-abc123-pride-1400'
    """
    if not href:
        return None
    absolute = urljoin(base + "/", href.strip())
    parts = urlparse(absolute)
    if not AD_URL_RE.search(parts.path):
        return None
    host = (parts.netloc or "bama.ir").lower()
    if host.startswith("www."):
        host = host[4:]
    if host and not host.endswith("bama.ir"):
        return None
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/")
    return urlunparse(("https", "bama.ir", path, "", urlencode(kept), ""))


def extract_ad_id(url: str) -> str | None:
    """Extract the stable advertisement code from a detail URL.

    >>> extract_ad_id("https://bama.ir/car/detail-lanch67j-dena-plus-1403")
    'lanch67j'
    """
    if not url:
        return None
    match = AD_URL_RE.search(urlparse(url).path if "//" in url else url)
    return match.group(1) if match else None


def parse_filters(search_url: str) -> dict[str, Any]:
    """Translate the public search URL into the API parameters the site uses.

    Verified against live traffic: the UI's ``year=<from>,<to>`` and
    ``price=<from>`` become ``yearFrom``/``yearTo`` and ``priceFrom``. Both are
    *lower bounds*, not maxima.
    """
    query = dict(parse_qsl(urlparse(search_url).query, keep_blank_values=True))
    params: dict[str, Any] = {}
    interpretation: dict[str, str] = {}

    for key, value in query.items():
        if key == "year":
            frm, _, to = value.partition(",")
            if frm:
                params["yearFrom"] = frm
                interpretation["year"] = (
                    f"production year from {frm} (inclusive) with no upper bound "
                    "-> 'that year and newer'"
                )
            if to:
                params["yearTo"] = to
                interpretation["year"] = f"production year between {frm} and {to}"
        elif key == "price":
            frm, _, to = value.partition(",")
            if frm:
                params["priceFrom"] = frm
                interpretation["price"] = (
                    f"minimum price {frm} Toman (priceFrom) - a lower bound, not a maximum"
                )
            if to:
                params["priceTo"] = to
        else:
            params[key] = value
            interpretation[key] = f"passed through unchanged as '{key}={value}'"
    return {"api_params": params, "interpretation": interpretation, "raw_query": query}


# ---------------------------------------------------------------------------
# Card mapping
# ---------------------------------------------------------------------------


def card_to_ad(
    entry: dict[str, Any], search_url: str, position: int, cycle: int
) -> DiscoveredAd | None:
    """Map one API ``ads[]`` entry to a :class:`DiscoveredAd`.

    Entries with ``type != "ad"`` (banners, campaign slots) are skipped.
    """
    if entry.get("type") != "ad":
        return None
    detail = entry.get("detail")
    if not isinstance(detail, dict):
        return None
    url = canonical_ad_url(detail.get("url") or "")
    if not url:
        return None
    ad_id = detail.get("code") or extract_ad_id(url)
    if not ad_id:
        return None

    price = entry.get("price") or {}
    price_text = price.get("price")
    # "0" is Bama's sentinel for negotiable/installment, never a real price.
    price_toman = normalize_price(price_text) if price.get("type") == "lumpsum" else None
    year_text = detail.get("year")
    year_jalali, year_gregorian = parse_year(year_text)

    return DiscoveredAd(
        ad_id=ad_id,
        url=url,
        source_search_url=search_url,
        card_title=normalize_text(detail.get("title")),
        card_subtitle=normalize_text(detail.get("subtitle")),
        price_text=normalize_text(str(price_text)) if price_text not in (None, "") else None,
        price_toman=price_toman,
        price_type=price.get("type"),
        year_text=normalize_text(str(year_text)) if year_text else None,
        year_jalali=year_jalali,
        year_gregorian=year_gregorian,
        mileage_text=normalize_text(detail.get("mileage")),
        mileage_km=normalize_mileage(detail.get("mileage")),
        location_text=normalize_text(detail.get("location")),
        thumbnail_url=detail.get("image"),
        is_promoted=bool(detail.get("pin") or detail.get("badge")),
        discovery_position=position,
        discovery_cycle=cycle,
        discovered_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# ---------------------------------------------------------------------------
# API discovery
# ---------------------------------------------------------------------------


def _save_raw_page(cfg: Config, run_id: str, page: int, body: dict[str, Any]) -> None:
    """Persist each raw endpoint response (gzipped) for later debugging.

    Best-effort: a disk problem here must never abort discovery.
    """
    try:
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.raw_dir / f"search_{run_id}_p{page:05d}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(body, fh, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("discovery.raw_save_failed", page=page, error=repr(exc))


def _headers(search_url: str, cfg: Config) -> dict[str, str]:
    return {
        "User-Agent": cfg.user_agent,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
        "Referer": search_url,
        "X-Requested-With": "XMLHttpRequest",
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_page(client: httpx.Client, params: dict[str, Any], page: int) -> dict[str, Any]:
    response = client.get(SEARCH_API, params={**params, "pageIndex": page})
    response.raise_for_status()
    return response.json()


def discover_via_api(
    cfg: Config,
    storage: Storage,
    run_id: str,
    *,
    start_page: int = 0,
    max_pages: int = 5000,
    on_cycle: Any = None,
) -> dict[str, Any]:
    """Page through the public search endpoint until it is exhausted.

    Termination is driven by the *observed* data, not by the endpoint's own
    counters. ``total_count`` is a running "delivered so far" figure that keeps
    climbing past the real total (2850 then 2880 for a ~2849-ad set), and
    ``total_pages`` tracks it. ``has_next`` does eventually flip to ``False``,
    but it is one flag sitting beside two demonstrably wrong counters, so it is
    recorded as corroborating evidence rather than trusted alone. We stop after
    :attr:`Config.api_stale_pages` consecutive pages that contribute no
    advertisements.
    """
    filters = parse_filters(cfg.url)
    params = {**filters["api_params"], "pageSize": cfg.api_page_size}
    started = time.time()
    page = start_page
    stale = 0
    total_new = 0
    pages_fetched = 0
    dup_urls = 0
    seen_urls: set[str] = set()
    termination = "unknown"
    raw_meta: dict[str, Any] = {}

    with httpx.Client(
        headers=_headers(cfg.url, cfg), timeout=cfg.request_timeout, follow_redirects=True
    ) as client:
        while page < max_pages:
            if time.time() - started > cfg.max_runtime:
                termination = "max_runtime_reached"
                break
            try:
                body = _fetch_page(client, params, page)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                storage.log_error(
                    run_id, None, str(exc.request.url), "discovery", "http_error", str(exc)
                )
                if status in (403, 429):
                    termination = f"blocked_http_{status}"
                    log.error("discovery.blocked", status=status)
                    break
                termination = f"http_error_{status}"
                break
            except httpx.HTTPError as exc:
                storage.log_error(run_id, None, cfg.url, "discovery", "network_error", str(exc))
                termination = "network_error"
                break

            pages_fetched += 1
            _save_raw_page(cfg, run_id, page, body)
            entries = (body.get("data") or {}).get("ads") or []
            raw_meta = body.get("metadata") or {}
            ads: list[DiscoveredAd] = []
            for i, entry in enumerate(entries):
                ad = card_to_ad(entry, cfg.url, position=page * cfg.api_page_size + i, cycle=page)
                if ad is None:
                    continue
                if ad.url in seen_urls:
                    dup_urls += 1
                    continue
                seen_urls.add(ad.url)
                ads.append(ad)

            new_count = storage.upsert_discovered(ads, run_id)
            total_new += new_count
            storage.set_checkpoint(run_id, "api_last_page", page)
            storage.set_checkpoint(run_id, "api_unique_total", len(seen_urls))

            real_ads = sum(1 for e in entries if e.get("type") == "ad")
            if real_ads == 0:
                stale += 1
            else:
                stale = 0

            print(
                f"page={page} entries={len(entries)} ads={real_ads} new={new_count} "
                f"unique_total={storage.count_discovered()} "
                f"total_count_field={raw_meta.get('total_count')} stale={stale}",
                flush=True,
            )
            if on_cycle:
                on_cycle(page, len(seen_urls))

            if stale >= cfg.api_stale_pages:
                termination = f"api_exhausted_after_{stale}_empty_pages"
                break
            page += 1
            time.sleep(random.uniform(cfg.delay_min, cfg.delay_max))
        else:
            termination = "max_pages_reached"

    return {
        "mode": "api",
        "pages_fetched": pages_fetched,
        "cycles": pages_fetched,
        "unique_urls": len(seen_urls),
        "new_ads": total_new,
        "duplicate_urls": dup_urls,
        "stale_at_end": stale,
        "termination_reason": termination,
        "elapsed_seconds": time.time() - started,
        "last_page": page,
        "endpoint": SEARCH_API,
        "api_params": params,
        "last_metadata": raw_meta,
    }


# ---------------------------------------------------------------------------
# Browser scroll discovery
# ---------------------------------------------------------------------------

_EXTRACT_CARDS_JS = """
() => {
  const seen = new Set();
  const out = [];
  document.querySelectorAll('a[href*="/car/detail-"]').forEach((a, i) => {
    const href = a.getAttribute('href');
    if (!href || seen.has(href)) return;
    seen.add(href);
    const card = a.closest('article') || a;
    const text = (card.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
    const img = card.querySelector('img');
    out.push({
      href,
      title: text[0] || null,
      subtitle: text[1] || null,
      lines: text,
      image: img ? (img.getAttribute('src') || img.getAttribute('data-src')) : null,
      promoted: /نردبان|ویژه|پین/.test(card.innerText || ''),
      position: i,
    });
  });
  return out;
}
"""


def _dom_card_to_ad(card: dict[str, Any], search_url: str, cycle: int) -> DiscoveredAd | None:
    """Map a DOM-extracted card to a :class:`DiscoveredAd`.

    DOM cards expose far less structure than the API, so most numeric fields
    are recovered heuristically from the card's text lines.
    """
    url = canonical_ad_url(card.get("href") or "")
    if not url:
        return None
    ad_id = extract_ad_id(url)
    if not ad_id:
        return None
    lines: list[str] = card.get("lines") or []
    price_line = next((x for x in lines if "تومان" in x or "،" in x and x.count("0") > 4), None)
    mileage_line = next((x for x in lines if "km" in x or "کیلومتر" in x), None)
    year_line = next((x for x in lines if re.fullmatch(r"\s*1[34]\d{2}\s*", x)), None)
    return DiscoveredAd(
        ad_id=ad_id,
        url=url,
        source_search_url=search_url,
        card_title=normalize_text(card.get("title")),
        card_subtitle=normalize_text(card.get("subtitle")),
        price_text=normalize_text(price_line),
        price_toman=normalize_price(price_line) if price_line else None,
        year_text=normalize_text(year_line),
        year_jalali=parse_year(year_line)[0],
        year_gregorian=parse_year(year_line)[1],
        mileage_text=normalize_text(mileage_line),
        mileage_km=normalize_mileage(mileage_line) if mileage_line else None,
        thumbnail_url=card.get("image"),
        is_promoted=bool(card.get("promoted")),
        discovery_position=card.get("position"),
        discovery_cycle=cycle,
        discovered_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


async def discover_via_scroll(
    cfg: Config,
    storage: Storage,
    run_id: str,
    *,
    collect_network: bool = True,
) -> dict[str, Any]:
    """Adaptive infinite-scroll discovery.

    A cycle extracts every card currently in the DOM *before* scrolling, so
    virtualized cards that get unmounted are still captured. A cycle counts as
    stale only when **all** of these hold: no new unique URL, no listing XHR,
    no document-height growth, no loading indicator, no enabled "load more"
    control, and the viewport is already at the bottom.
    """
    from playwright.async_api import async_playwright  # imported lazily: heavy dependency

    started = time.time()
    stale = 0
    cycle = 0
    seen: set[str] = set()
    xhr_since_cycle = {"n": 0}
    candidates: dict[str, NetworkCandidate] = {}
    termination = "unknown"
    height = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not cfg.headed)
        context = await browser.new_context(
            user_agent=cfg.user_agent,
            locale=cfg.locale,
            viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
        )
        page = await context.new_page()
        page.set_default_timeout(cfg.navigation_timeout * 1000)

        async def on_response(response: Any) -> None:
            try:
                request = response.request
                if request.resource_type not in ("xhr", "fetch"):
                    return
                if "/cad/api/search" in response.url:
                    xhr_since_cycle["n"] += 1
                if collect_network and "bama.ir" in response.url:
                    body = None
                    if "json" in (response.headers.get("content-type") or ""):
                        try:
                            body = await response.json()
                        except Exception:
                            body = None
                    ads = None
                    if isinstance(body, dict) and isinstance(body.get("data"), dict):
                        raw = body["data"].get("ads")
                        ads = len(raw) if isinstance(raw, list) else None
                    candidates[response.url] = NetworkCandidate(
                        url=response.url,
                        method=request.method,
                        status=response.status,
                        resource_type=request.resource_type,
                        is_listing_endpoint="/cad/api/search" in response.url,
                        top_level_keys=list(body.keys()) if isinstance(body, dict) else [],
                        ad_count=ads,
                    )
            except Exception:  # never let instrumentation break the scrape
                return

        page.on("response", on_response)
        await page.goto(
            cfg.url, wait_until="domcontentloaded", timeout=int(cfg.navigation_timeout * 1000)
        )
        await page.wait_for_timeout(cfg.settle_ms)

        try:
            while cycle < cfg.max_scrolls:
                if time.time() - started > cfg.max_runtime:
                    termination = "max_runtime_reached"
                    break

                # 1-2. extract everything currently rendered, then persist it
                cards = await page.evaluate(_EXTRACT_CARDS_JS)
                ads = [a for a in (_dom_card_to_ad(c, cfg.url, cycle) for c in cards) if a]
                fresh = [a for a in ads if a.url not in seen]
                for a in fresh:
                    seen.add(a.url)
                new_count = storage.upsert_discovered(fresh, run_id)
                storage.set_checkpoint(run_id, "scroll_cycle", cycle)
                storage.set_checkpoint(run_id, "scroll_unique", len(seen))

                prev_height = height
                height = await page.evaluate("document.body.scrollHeight")
                xhr_since_cycle["n"] = 0

                # 5-6. nudge, then go to the true bottom
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.92)")
                await page.wait_for_timeout(250)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(cfg.scroll_pause_ms)

                # 8. click a "load more" control if one is present and enabled
                clicked = await page.evaluate(
                    """() => {
                        const btns = [...document.querySelectorAll('button, a[role=button]')]
                          .filter(b => /بیشتر|بارگذاری|load more/i.test(b.textContent || ''));
                        const b = btns.find(x => !x.disabled &&
                                   x.offsetParent !== null);
                        if (b) { b.click(); return true; }
                        return false;
                    }"""
                )
                if clicked:
                    await page.wait_for_timeout(cfg.settle_ms)

                loading = await page.evaluate(
                    """() => !!document.querySelector(
                         '[class*=spinner], [class*=loading], [aria-busy=true]')"""
                )
                at_bottom = await page.evaluate(
                    "() => (window.innerHeight + window.scrollY) >= "
                    "(document.body.scrollHeight - 250)"
                )
                new_height = await page.evaluate("document.body.scrollHeight")

                progressed = (
                    new_count > 0
                    or xhr_since_cycle["n"] > 0
                    or new_height > max(prev_height, height)
                    or loading
                    or clicked
                    or not at_bottom
                )
                stale = 0 if progressed else stale + 1
                height = new_height

                print(
                    f"cycle={cycle} visible={len(cards)} new={new_count} "
                    f"unique_total={len(seen)} height={new_height} stale={stale}",
                    flush=True,
                )

                if stale >= cfg.stale_cycles:
                    # 11. final stabilization check before declaring the end
                    await page.wait_for_timeout(cfg.settle_ms * 2)
                    final_cards = await page.evaluate(_EXTRACT_CARDS_JS)
                    final_ads = [
                        a for a in (_dom_card_to_ad(c, cfg.url, cycle) for c in final_cards) if a
                    ]
                    extra = [a for a in final_ads if a.url not in seen]
                    if extra:
                        for a in extra:
                            seen.add(a.url)
                        storage.upsert_discovered(extra, run_id)
                        stale = 0
                        log.info("scroll.stabilization_found_more", n=len(extra))
                    else:
                        termination = f"stable_after_{stale}_stale_cycles"
                        break
                cycle += 1
            else:
                termination = "max_scrolls_reached"
        except KeyboardInterrupt:
            termination = "interrupted_by_user"
            log.warning("scroll.interrupted", cycle=cycle, unique=len(seen))
        finally:
            if termination in ("unknown", "max_scrolls_reached"):
                cfg.ensure_dirs()
                try:
                    await page.screenshot(
                        path=str(cfg.debug_dir / f"scroll_end_{run_id}.png"), full_page=False
                    )
                    (cfg.debug_dir / f"scroll_end_{run_id}.html").write_text(
                        await page.content(), encoding="utf-8"
                    )
                except Exception:
                    pass
            if candidates:
                storage.save_network_candidates(candidates.values())
            await context.close()
            await browser.close()

    return {
        "mode": "scroll",
        "cycles": cycle,
        "unique_urls": len(seen),
        "stale_at_end": stale,
        "termination_reason": termination,
        "elapsed_seconds": time.time() - started,
        "final_height": height,
        "network_candidates": len(candidates),
    }


def merge_inventories(first: Iterable[str], second: Iterable[str]) -> dict[str, Any]:
    """Compare two discovery inventories for the completeness audit."""
    a, b = set(first), set(second)
    return {
        "first_pass": len(a),
        "second_pass": len(b),
        "only_in_first": len(a - b),
        "only_in_second": len(b - a),
        "new_urls_found_by_second_pass": sorted(b - a)[:50],
        "stable": not (b - a),
    }


async def _amain(cfg: Config, storage: Storage, run_id: str) -> dict[str, Any]:
    return await discover_via_scroll(cfg, storage, run_id)


def run_discovery(cfg: Config, storage: Storage, run_id: str) -> dict[str, Any]:
    """Dispatch discovery according to ``cfg.mode``."""
    mode = cfg.mode
    if mode in ("auto", "api", "pagination"):
        result = discover_via_api(cfg, storage, run_id)
        found = result["unique_urls"]
        if mode == "auto" and found == 0:
            log.warning("discovery.api_empty_falling_back_to_scroll")
            return asyncio.run(_amain(cfg, storage, run_id))
        return result
    if mode == "scroll":
        return asyncio.run(_amain(cfg, storage, run_id))
    raise ValueError(f"unknown mode: {mode}")
