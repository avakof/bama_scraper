"""Fetching layer: pooled HTTP client plus an optional Playwright fallback.

Detail pages were verified to be fully server-rendered (the ``__NUXT_DATA__``
payload is present in the raw HTTP response), so the default path is httpx —
far cheaper than a browser per page. A Playwright fallback exists for pages
that come back without the payload, e.g. if the site starts client-rendering.
"""

from __future__ import annotations

import asyncio
import random
from contextlib import asynccontextmanager
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .logging_config import get_logger

log = get_logger(__name__)


class BlockedError(Exception):
    """Raised when the site actively refuses automated access (403/429/CAPTCHA)."""


class RetryableFetchError(Exception):
    """Transient failure that is worth another bounded attempt."""


class PermanentFetchError(Exception):
    """The page will not become available by retrying (404/410)."""


class AbortedError(Exception):
    """The fetch was skipped because the run is shutting down.

    Distinct from a failure: the request was never issued, so the
    advertisement must stay claimable rather than burn a retry attempt.
    """


def detail_headers(cfg: Config, referer: str) -> dict[str, str]:
    return {
        "User-Agent": cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
        "Referer": referer,
        "Upgrade-Insecure-Requests": "1",
    }


class DetailFetcher:
    """Concurrency-limited fetcher with bounded retries and polite delays."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._client: httpx.AsyncClient | None = None
        #: Clients replaced by recycling; closed together at shutdown so that
        #: requests still in flight on them are never cut off mid-response.
        self._retired: list[httpx.AsyncClient] = []
        self._fetched = 0
        self._aborted = False
        self._browser: Any = None
        self._pw: Any = None
        self._context: Any = None

    async def __aenter__(self) -> DetailFetcher:
        self._client = self._make_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for client in (*self._retired, self._client):
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # shutdown must not mask the real error
                    pass
        self._retired.clear()
        await self._close_browser()

    async def _close_browser(self) -> None:
        for obj in (self._context, self._browser, self._pw):
            try:
                if obj is None:
                    continue
                await (obj.close() if hasattr(obj, "close") else obj.stop())
            except Exception:
                pass
        self._context = self._browser = self._pw = None

    def _make_client(self) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=max(4, self.cfg.concurrency * 2),
            max_keepalive_connections=self.cfg.concurrency,
        )
        return httpx.AsyncClient(
            headers=detail_headers(self.cfg, self.cfg.url),
            timeout=self.cfg.request_timeout,
            follow_redirects=True,
            limits=limits,
        )

    async def _recycle_if_needed(self) -> None:
        """Recycle the pooled connections periodically to avoid stale sockets.

        The outgoing client is *retired*, not closed: a concurrent worker may
        still be mid-request on it, and closing it underneath would abort that
        request with "client has been closed". Retired clients are closed once
        in :meth:`__aexit__`.

        Blocking here until peers go idle would deadlock instead -- each worker
        holds a semaphore slot while waiting for the others to release theirs.
        """
        self._fetched += 1
        if self._fetched % self.cfg.recycle_context_every != 0:
            return
        log.info("fetcher.recycle", fetched=self._fetched)
        old, self._client = self._client, self._make_client()
        if old is not None:
            self._retired.append(old)
        await self._close_browser()

    @retry(
        retry=retry_if_exception_type(RetryableFetchError),
        wait=wait_exponential(multiplier=1, min=2, max=45),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _get(self, url: str) -> str:
        # Bind once: a concurrent recycle may swap self._client mid-request, and
        # this request must keep using the client it started on.
        client = self._client
        assert client is not None
        try:
            response = await client.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableFetchError(f"transport: {exc!r}") from exc

        status = response.status_code
        if status in (403, 429):
            raise BlockedError(f"HTTP {status} for {url}")
        if status in (404, 410):
            raise PermanentFetchError(f"HTTP {status} for {url}")
        if status >= 500:
            raise RetryableFetchError(f"HTTP {status} for {url}")
        if status >= 400:
            raise PermanentFetchError(f"HTTP {status} for {url}")
        return response.text

    def abort(self) -> None:
        """Stop issuing new requests.

        Callers awaiting the semaphore are already past any check made before
        ``fetch`` was entered, so the decisive re-check happens *inside*
        :meth:`fetch` once the slot is acquired.
        """
        self._aborted = True

    async def fetch(self, url: str) -> tuple[str, str]:
        """Return ``(html, source)`` for a detail URL, respecting concurrency."""
        async with self._sem:
            # Re-checked here, not by the caller: a whole batch of coroutines
            # queues on this semaphore, and without this they would each still
            # fire a request after the site has already told us to back off.
            if self._aborted:
                raise AbortedError(url)
            await asyncio.sleep(random.uniform(self.cfg.delay_min, self.cfg.delay_max))
            html = await self._get(url)
            await self._recycle_if_needed()
            if "__NUXT_DATA__" not in html:
                log.warning("fetch.payload_missing_trying_browser", url=url)
                rendered = await self._fetch_with_browser(url)
                if rendered:
                    return rendered, "browser"
            return html, "httpx"

    async def download(self, url: str) -> bytes | None:
        """Fetch a binary asset (image/video). Returns ``None`` on failure.

        Asset downloads are best-effort: a missing image must never fail the
        advertisement it belongs to.
        """
        async with self._sem:
            await asyncio.sleep(random.uniform(self.cfg.delay_min, self.cfg.delay_max))
            client = self._client
            if client is None:
                return None
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.content
                log.warning("download.bad_status", url=url, status=response.status_code)
            except Exception as exc:  # noqa: BLE001
                log.warning("download.failed", url=url, error=repr(exc))
            return None

    async def _ensure_browser(self) -> Any:
        if self._context is not None:
            return self._context
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=not self.cfg.headed)
        self._context = await self._browser.new_context(
            user_agent=self.cfg.user_agent,
            locale=self.cfg.locale,
            viewport={"width": self.cfg.viewport_width, "height": self.cfg.viewport_height},
        )
        return self._context

    async def _fetch_with_browser(self, url: str) -> str | None:
        try:
            context = await self._ensure_browser()
            page = await context.new_page()
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=int(self.cfg.navigation_timeout * 1000),
                )
                await page.wait_for_timeout(2000)
                return await page.content()
            finally:
                await page.close()
        except Exception as exc:  # browser fallback is best-effort
            log.warning("fetch.browser_fallback_failed", url=url, error=repr(exc))
            return None


@asynccontextmanager
async def playwright_page(cfg: Config):
    """Standalone Playwright page, used by the ``inspect`` command."""
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=not cfg.headed)
    context = await browser.new_context(
        user_agent=cfg.user_agent,
        locale=cfg.locale,
        viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
    )
    page = await context.new_page()
    page.set_default_timeout(cfg.navigation_timeout * 1000)
    try:
        yield page
    finally:
        await context.close()
        await browser.close()
        await pw.stop()
