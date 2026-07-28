"""Adapters over the existing scrapers.

Neither the discovery logic nor the detail parsers are reimplemented here. The
discovery adapter drives ``bama_scraper.discovery.discover_via_api`` — the
implementation whose termination behaviour was verified against the live site
(it stops on consecutive empty pages rather than on the endpoint's own counters,
which grow indefinitely) — and reads back the inventory it persisted. The detail
adapter drives the deep scraper's ``parse_ad_api`` / ``parse_ad_html`` / ``merge_ad``
pipeline.

Each run gets its own scratch discovery database under the run's evidence
directory, so the raw pages and card-level inventory of every run are retained
even when the run is later judged invalid.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Protocol

from bama_scraper.logging_config import get_logger

from .config import MonitorConfig
from .models import DetailResult, DetailVerdict, DiscoveredCard, DiscoveryResult

log = get_logger("bama_monitor.scrapers")

# The deep scraper lives in a sibling top-level folder rather than in src/.
_DEEP_ROOT = Path(__file__).resolve().parents[2] / "deep_scraper"
if str(_DEEP_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEEP_ROOT))


class DiscoveryScraper(Protocol):
    async def discover(self, search_url: str) -> DiscoveryResult: ...


class DetailScraper(Protocol):
    async def scrape(self, advertisement_url: str) -> DetailResult: ...


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def build_discovery_result(
    *,
    search_url: str,
    cards: list[DiscoveredCard],
    outcome: dict[str, Any],
    evidence_dir: str,
) -> DiscoveryResult:
    """Translate a discovery outcome into the health-gate's input vocabulary.

    Kept separate from the network call so the classification — which decides
    whether a run may change any advertisement's status — is unit-testable.
    """
    reason = str(outcome.get("termination_reason") or "unknown")
    # The discovery implementation reports a verified end only when it saw the
    # configured number of consecutive empty pages.
    verified_end = reason.startswith("api_exhausted")
    blocked = reason.startswith("blocked_http")

    # Only a genuine page error counts as a failed page. Truncation by the page or
    # runtime ceiling is *not* a failure: it is already reported by
    # `reached_verified_end=False`, and inventing a failed page here would put a
    # fabricated observation into the health report and the daily alerts.
    page_failed = reason.startswith(("http_error", "network_error", "blocked_http"))
    pages_fetched = int(outcome.get("pages_fetched") or 0)

    return DiscoveryResult(
        search_url=search_url,
        cards=cards,
        termination_reason=reason,
        pages_fetched=pages_fetched,
        pages_failed=1 if page_failed else 0,
        duplicate_count=int(outcome.get("duplicate_urls") or 0),
        initial_page_ok=pages_fetched > 0,
        reached_verified_end=verified_end,
        # The empty-page confirmation *is* the stabilization pass for the API path:
        # it re-queries past the apparent end before stopping.
        stabilization_completed=verified_end,
        blocked=blocked,
        errors=[reason] if page_failed else [],
        elapsed_seconds=float(outcome.get("elapsed_seconds") or 0.0),
        raw_evidence_path=evidence_dir,
    )


class BamaDiscoveryScraper:
    """Wraps the verified API-based discovery pass."""

    def __init__(self, cfg: MonitorConfig, evidence_dir: Path) -> None:
        self.cfg = cfg
        self.evidence_dir = evidence_dir

    async def discover(self, search_url: str) -> DiscoveryResult:
        """Run discovery to its verified end and return the card inventory."""
        return await asyncio.to_thread(self._discover_sync, search_url)

    def _discover_sync(self, search_url: str) -> DiscoveryResult:
        from bama_scraper.config import Config as ScraperConfig
        from bama_scraper.discovery import discover_via_api
        from bama_scraper.storage import Storage

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        scraper_cfg = ScraperConfig(
            url=search_url,
            output_dir=self.evidence_dir,
            delay_min=self.cfg.detail.delay_min,
            delay_max=self.cfg.detail.delay_max,
            request_timeout=self.cfg.detail.request_timeout,
            max_runtime=self.cfg.max_run_seconds,
        )
        scraper_cfg.ensure_dirs()
        storage = Storage(scraper_cfg.db_path)
        try:
            storage.start_run("monitor", search_url, {}, "api", "monitor")
            outcome = discover_via_api(
                scraper_cfg, storage, "monitor", max_pages=self.cfg.max_discovery_pages
            )
            rows = storage.conn.execute(
                "SELECT ad_id, url, card_title, price_text, price_toman, year_text,"
                " mileage_text, mileage_km, location_text, thumbnail_url, is_promoted,"
                " discovery_position, discovery_cycle FROM discovered_ads"
                " ORDER BY discovery_cycle, discovery_position"
            ).fetchall()
        finally:
            storage.close()

        cards = [
            DiscoveredCard(
                platform_ad_id=row["ad_id"],
                canonical_url=row["url"],
                position=row["discovery_position"],
                page_number=row["discovery_cycle"],
                scroll_cycle=row["discovery_cycle"],
                card_title=row["card_title"],
                card_price_raw=row["price_text"],
                card_price_normalized=row["price_toman"],
                card_year=row["year_text"],
                card_mileage_raw=row["mileage_text"],
                card_mileage_normalized=row["mileage_km"],
                card_location=row["location_text"],
                card_image_url=row["thumbnail_url"],
                is_promoted=bool(row["is_promoted"]) if row["is_promoted"] is not None else None,
            )
            for row in rows
        ]

        return build_discovery_result(
            search_url=search_url,
            cards=cards,
            outcome=outcome,
            evidence_dir=str(self.evidence_dir),
        )


class StaticDiscoveryScraper:
    """Deterministic discovery for tests and multi-day simulation.

    Exists so the whole monitoring workflow — including health gating, state
    transitions and reporting — can be exercised end to end without touching the
    network, which is what makes the multi-day simulation reproducible.
    """

    def __init__(self, plan: list[DiscoveryResult]) -> None:
        self._plan = list(plan)
        self.calls = 0

    async def discover(self, search_url: str) -> DiscoveryResult:
        if not self._plan:
            raise RuntimeError("StaticDiscoveryScraper exhausted")
        self.calls += 1
        return self._plan.pop(0)


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------

#: Persian phrases a removed listing renders instead of the vehicle.
_UNAVAILABLE_MARKERS = (
    "آگهی حذف شده",
    "آگهی منقضی",
    "این آگهی وجود ندارد",
    "یافت نشد",
    "حذف شده است",
)


class BamaDetailScraper:
    """Wraps the deep scraper's JSON-API + HTML merge pipeline.

    Also classifies *why* a page could not be read, because for a missing
    advertisement that classification is the evidence: a page that still loads
    argues against a sale, while 404/410 argues for removal.
    """

    def __init__(self, cfg: MonitorConfig) -> None:
        self.cfg = cfg
        self._fetcher: Any = None
        self._deep_cfg: Any = None

    async def __aenter__(self) -> BamaDetailScraper:
        from bama_deep.config import DeepConfig
        from bama_deep.net import DeepFetcher

        self._deep_cfg = DeepConfig(
            concurrency=self.cfg.detail.concurrency,
            delay_min=self.cfg.detail.delay_min,
            delay_max=self.cfg.detail.delay_max,
            request_timeout=self.cfg.detail.request_timeout,
            output_dir=self.cfg.output_dir / "deep",
            referer=self.cfg.search_url,
        )
        self._fetcher = await DeepFetcher(self._deep_cfg).__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._fetcher is not None:
            await self._fetcher.__aexit__(*exc)
            self._fetcher = None

    async def scrape(self, advertisement_url: str) -> DetailResult:
        """Fetch and parse one detail page, classifying failures precisely."""
        import time

        from bama_deep.ad_api import parse_ad_api
        from bama_deep.ad_html import parse_ad_html
        from bama_deep.endpoints import ad_api_url
        from bama_deep.merge import merge_ad
        from bama_deep.net import (
            BlockedError,
            BudgetExhaustedError,
            PermanentFetchError,
        )

        from bama_scraper.discovery import extract_ad_id

        ad_id = extract_ad_id(advertisement_url) or advertisement_url
        if self._fetcher is None:
            raise RuntimeError("BamaDetailScraper must be used as an async context manager")

        now = time.time()
        api_result = None
        html_result = None
        verdict = DetailVerdict.UNKNOWN
        status_code: int | None = None
        error: str | None = None

        try:
            payload = await self._fetcher.fetch_json(ad_api_url(ad_id))
        except PermanentFetchError as exc:
            message = str(exc)
            status_code = 410 if "410" in message else 404 if "404" in message else None
            verdict = (
                DetailVerdict.HTTP_410
                if status_code == 410
                else DetailVerdict.HTTP_404
                if status_code == 404
                else DetailVerdict.EXPLICITLY_UNAVAILABLE
            )
            error = message
        except BlockedError as exc:
            verdict, error = DetailVerdict.BLOCKED, str(exc)
        except BudgetExhaustedError as exc:
            verdict, error = DetailVerdict.UNKNOWN, str(exc)
        except Exception as exc:  # noqa: BLE001
            verdict, error = DetailVerdict.TEMPORARY_ERROR, repr(exc)
        else:
            # The site answered. Anything that fails from here is our parser, and
            # calling that a "temporary server error" would blame the wrong party
            # and mark it retryable — a payload we cannot parse today will not
            # parse tomorrow either.
            try:
                api_result = parse_ad_api(payload, now_ts=now)
                verdict = DetailVerdict.STILL_ACTIVE
            except Exception as exc:  # noqa: BLE001
                verdict, error = DetailVerdict.PARSE_ERROR, repr(exc)
                log.error("monitor.detail.parse_failed", ad_id=ad_id, error=repr(exc))

        if verdict is DetailVerdict.STILL_ACTIVE:
            try:
                html = await self._fetcher.fetch_html(advertisement_url)
                if any(marker in html for marker in _UNAVAILABLE_MARKERS):
                    verdict = DetailVerdict.EXPLICITLY_UNAVAILABLE
                else:
                    html_result = parse_ad_html(html, now_ts=now)
            except PermanentFetchError:
                # The API answered but the page did not: treat as still-active
                # rather than removed, and record the inconsistency.
                error = "html unavailable while api responded"
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)

        if api_result is None and html_result is None:
            return DetailResult(
                platform_ad_id=ad_id,
                url=advertisement_url,
                ok=False,
                verdict=verdict,
                http_status=status_code,
                error=error,
            )

        merged = merge_ad(
            api_result,
            html_result,
            ad_id=ad_id,
            url=advertisement_url,
            api_url=ad_api_url(ad_id),
            now_ts=now,
        )
        return DetailResult(
            platform_ad_id=ad_id,
            url=advertisement_url,
            ok=True,
            verdict=verdict,
            http_status=status_code or 200,
            fields=merged.record,
            media=merged.media,
            content_hash=_content_hash(merged.record),
            parser_version=str(merged.record.get("deep_version") or "1.0.0"),
            error=error,
        )


def _content_hash(record: dict[str, Any]) -> str:
    """Hash the fields whose change should trigger a new snapshot.

    Evidence blobs and timestamps are excluded: they change on every scrape and
    would make every run look like a content change.
    """
    import hashlib
    import json

    volatile = {
        "scraped_at",
        "api_json_sha256",
        "html_sha256",
        "api_raw_json",
        "html_payload_json",
        "json_ld_json",
        "provenance_json",
        "numeric_agreement_json",
        "published_text",
        "published_ts",
        "stats_model_ad_count",
        "stats_trim_ad_count",
        "meta_title",
        "conflict_count",
        "parse_warnings_json",
    }
    payload = {k: v for k, v in sorted(record.items()) if k not in volatile}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class StaticDetailScraper:
    """Scripted detail responses for tests and simulation."""

    def __init__(self, responses: dict[str, DetailResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def __aenter__(self) -> StaticDetailScraper:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def scrape(self, advertisement_url: str) -> DetailResult:
        self.calls.append(advertisement_url)
        if advertisement_url in self.responses:
            return self.responses[advertisement_url]
        return DetailResult(
            platform_ad_id=advertisement_url,
            url=advertisement_url,
            ok=False,
            verdict=DetailVerdict.UNKNOWN,
            error="no scripted response",
        )
