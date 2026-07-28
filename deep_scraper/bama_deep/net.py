"""Concurrency-limited fetcher shared by every phase.

Deliberately mirrors ``bama_scraper.browser.DetailFetcher``, including the two
behaviours that were bug-fixed there:

* ``abort()`` is re-checked **inside** :meth:`fetch_text` after the semaphore is
  acquired. A whole batch of coroutines queues on that semaphore, so a flag
  checked before the call would still let every queued request fire after the
  site has told us to back off.
* Recycling **retires** the outgoing client instead of closing it. A concurrent
  worker may still be mid-request on it, and blocking until peers go idle
  deadlocks (each worker holds a slot while waiting for the others).

One instance is shared across all four phases, so the single semaphore bounds
the global request rate even when phases interleave.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import DeepConfig
from .endpoints import assert_allowed, is_allowed_asset


class BlockedError(Exception):
    """The site actively refused access (403/429)."""


class RetryableFetchError(Exception):
    """Transient failure worth another bounded attempt."""


class PermanentFetchError(Exception):
    """Will not succeed by retrying (404/410/4xx)."""


class AbortedError(Exception):
    """Skipped because the run is shutting down; the request was never sent."""


class BudgetExhaustedError(Exception):
    """The configured request ceiling was reached."""


def classify_status(status: int, url: str) -> None:
    """Raise the right exception for a non-success status.

    Kept in one place so every phase treats the site's signals identically.
    """
    if status in (403, 429):
        raise BlockedError(f"HTTP {status} for {url}")
    if status in (404, 410):
        raise PermanentFetchError(f"HTTP {status} for {url}")
    if status >= 500:
        raise RetryableFetchError(f"HTTP {status} for {url}")
    if status >= 400:
        raise PermanentFetchError(f"HTTP {status} for {url}")


class DeepFetcher:
    """Polite HTTP client with bounded retries, an abort switch and a budget."""

    def __init__(self, cfg: DeepConfig) -> None:
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._client: httpx.AsyncClient | None = None
        self._retired: list[httpx.AsyncClient] = []
        self._aborted = False
        self.requests_issued = 0
        self.blocked_reason: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def _make_client(self) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=max(4, self.cfg.concurrency * 2),
            max_keepalive_connections=self.cfg.concurrency,
        )
        return httpx.AsyncClient(
            headers={
                "User-Agent": self.cfg.user_agent,
                "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
                "Referer": self.cfg.referer,
            },
            timeout=self.cfg.request_timeout,
            follow_redirects=True,
            limits=limits,
        )

    async def __aenter__(self) -> DeepFetcher:
        self._client = self._make_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for client in (*self._retired, self._client):
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass
        self._retired.clear()
        self._client = None

    def abort(self, reason: str | None = None) -> None:
        """Stop issuing new requests; queued callers raise :class:`AbortedError`."""
        self._aborted = True
        if reason and not self.blocked_reason:
            self.blocked_reason = reason

    @property
    def aborted(self) -> bool:
        return self._aborted

    async def _recycle_if_needed(self) -> None:
        self.requests_issued += 1
        every = self.cfg.recycle_context_every
        if every and self.requests_issued % every == 0:
            old, self._client = self._client, self._make_client()
            if old is not None:
                self._retired.append(old)

    # -- fetching ----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(RetryableFetchError),
        wait=wait_exponential(multiplier=1, min=2, max=45),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _request(self, url: str, *, accept: str) -> httpx.Response:
        client = self._client
        if client is None:
            raise RetryableFetchError("client not initialised")
        try:
            response = await client.get(url, headers={"Accept": accept})
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableFetchError(f"transport: {exc!r}") from exc
        classify_status(response.status_code, url)
        return response

    async def fetch_text(self, url: str, *, accept: str = "*/*") -> str:
        """Fetch a URL as text, respecting the guard list, budget and abort flag."""
        assert_allowed(url)  # raises before any socket is opened
        async with self._sem:
            if self._aborted:
                raise AbortedError(url)
            if self.cfg.max_requests and self.requests_issued >= self.cfg.max_requests:
                raise BudgetExhaustedError(f"request budget of {self.cfg.max_requests} reached")
            await asyncio.sleep(random.uniform(self.cfg.delay_min, self.cfg.delay_max))
            response = await self._request(url, accept=accept)
            await self._recycle_if_needed()
            return response.text

    async def fetch_json(self, url: str) -> Any:
        """Fetch and decode JSON; a non-JSON body is a permanent failure."""
        import json

        body = await self.fetch_text(url, accept="application/json, text/plain, */*")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise PermanentFetchError(f"non-JSON body from {url}: {exc}") from exc

    async def fetch_html(self, url: str) -> str:
        return await self.fetch_text(
            url, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        )

    async def download_asset(self, url: str) -> bytes | None:
        """Fetch a binary asset, refusing anything robots.txt disallows."""
        if not is_allowed_asset(url):
            return None
        assert_allowed(url)
        async with self._sem:
            if self._aborted:
                raise AbortedError(url)
            await asyncio.sleep(random.uniform(self.cfg.delay_min, self.cfg.delay_max))
            client = self._client
            if client is None:
                return None
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    await self._recycle_if_needed()
                    return response.content
            except Exception:
                return None
            return None
