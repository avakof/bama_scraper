"""Fetcher behaviour and endpoint guards against a local server.

A request log records every path the server was actually asked for, so tests can
assert not only what happened but what was **never requested**.
"""

from __future__ import annotations

import asyncio
import http.server
import socket
import threading

import pytest
from bama_deep.config import DeepConfig
from bama_deep.endpoints import ForbiddenEndpointError
from bama_deep.net import (
    AbortedError,
    BlockedError,
    BudgetExhaustedError,
    DeepFetcher,
    PermanentFetchError,
    RetryableFetchError,
    classify_status,
)

REQUEST_LOG: list[str] = []
_STATE = {"flaky_hits": 0}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        REQUEST_LOG.append(self.path)
        path = self.path
        if path.startswith("/403"):
            code, body = 403, b"forbidden"
        elif path.startswith("/429"):
            code, body = 429, b"slow down"
        elif path.startswith("/404"):
            code, body = 404, b"gone"
        elif path.startswith("/500"):
            code, body = 500, b"boom"
        elif path.startswith("/flaky"):
            _STATE["flaky_hits"] += 1
            code, body = (500, b"boom") if _STATE["flaky_hits"] <= 2 else (200, b'{"ok":true}')
        elif path.startswith("/notjson"):
            code, body = 200, b"<html>not json</html>"
        else:
            code, body = 200, b'{"ok":true}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture(scope="module")
def base_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def _reset() -> None:
    REQUEST_LOG.clear()
    _STATE["flaky_hits"] = 0


def cfg(**kw) -> DeepConfig:
    return DeepConfig(delay_min=0.0, delay_max=0.0, request_timeout=10.0, **kw)


def run(coro_factory, config: DeepConfig | None = None):
    async def main():
        async with DeepFetcher(config or cfg()) as fetcher:
            return await coro_factory(fetcher)

    return asyncio.run(main())


class TestForbiddenEndpoints:
    """These must fail before a socket is opened, not after."""

    FORBIDDEN = (
        "https://bama.ir/cad/api/log/visit",
        "https://bama.ir/cad/api/log/impression",
        "https://bama.ir/cad/api/log/share",
        "https://bama.ir/event/api/v1/events",
        "https://bama.ir/cad/api/carad/abc123",
        "https://bama.ir/cad/api/carad/abc123/details",
        "https://bama.ir/cad/api/price?pageIndex=0",
        "https://bama.ir/cad/api/Corporation/phone/921",
        "https://bama.ir/prf/api/user/userinfo",
    )

    @pytest.mark.parametrize("url", FORBIDDEN)
    def test_raises_and_issues_no_request(self, url: str) -> None:
        with pytest.raises(ForbiddenEndpointError):
            run(lambda f: f.fetch_json(url))
        assert REQUEST_LOG == [], f"a request was issued for {url}"

    def test_asset_download_refuses_disallowed_prefix(self) -> None:
        banner = "https://cdn-sth1.bama.ir/uploads/BamaImages/CampaignBanner/1/x.jpg"
        assert run(lambda f: f.download_asset(banner)) is None
        assert REQUEST_LOG == []


class TestStatusClassification:
    @pytest.mark.parametrize("status", [403, 429])
    def test_block(self, status: int) -> None:
        with pytest.raises(BlockedError):
            classify_status(status, "u")

    @pytest.mark.parametrize("status", [404, 410, 400, 401, 422])
    def test_permanent(self, status: int) -> None:
        with pytest.raises(PermanentFetchError):
            classify_status(status, "u")

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_retryable(self, status: int) -> None:
        with pytest.raises(RetryableFetchError):
            classify_status(status, "u")

    @pytest.mark.parametrize("status", [200, 201, 204, 301])
    def test_success_passes(self, status: int) -> None:
        classify_status(status, "u")  # must not raise


class TestFetching:
    def test_json(self, base_url: str) -> None:
        assert run(lambda f: f.fetch_json(f"{base_url}/ok")) == {"ok": True}

    def test_non_json_body_is_permanent(self, base_url: str) -> None:
        with pytest.raises(PermanentFetchError):
            run(lambda f: f.fetch_json(f"{base_url}/notjson"))

    def test_429_blocks(self, base_url: str) -> None:
        with pytest.raises(BlockedError):
            run(lambda f: f.fetch_json(f"{base_url}/429"))

    def test_404_permanent(self, base_url: str) -> None:
        with pytest.raises(PermanentFetchError):
            run(lambda f: f.fetch_json(f"{base_url}/404"))

    def test_transient_500_is_retried_then_succeeds(self, base_url: str) -> None:
        assert run(lambda f: f.fetch_json(f"{base_url}/flaky")) == {"ok": True}
        assert len(REQUEST_LOG) == 3  # two failures, then success

    def test_request_counter(self, base_url: str) -> None:
        async def body(f):
            await f.fetch_json(f"{base_url}/a")
            await f.fetch_json(f"{base_url}/b")
            return f.requests_issued

        assert run(body) == 2


class TestAbortAndBudget:
    def test_abort_prevents_further_requests(self, base_url: str) -> None:
        async def body(f):
            await f.fetch_json(f"{base_url}/first")
            f.abort("blocked upstream")
            with pytest.raises(AbortedError):
                await f.fetch_json(f"{base_url}/second")
            return True

        assert run(body)
        # Only the pre-abort request reached the server.
        assert [p for p in REQUEST_LOG if "second" in p] == []

    def test_queued_batch_stops_after_a_block(self, base_url: str) -> None:
        """The regression the parent project hit: a whole batch kept firing.

        Every coroutine is already past any pre-call check by the time the first
        response lands, so the abort has to be enforced inside the fetch.
        """

        async def body(f):
            async def one(i: int):
                try:
                    if i == 0:
                        await f.fetch_json(f"{base_url}/429")
                    else:
                        await f.fetch_json(f"{base_url}/ok{i}")
                except BlockedError:
                    f.abort("429")
                except AbortedError:
                    pass

            await asyncio.gather(*(one(i) for i in range(40)), return_exceptions=True)
            return len(REQUEST_LOG)

        issued = run(body, cfg(concurrency=1))
        assert issued < 40, f"{issued} requests issued after a block"

    def test_budget_ceiling_is_enforced(self, base_url: str) -> None:
        async def body(f):
            for i in range(3):
                await f.fetch_json(f"{base_url}/ok{i}")
            with pytest.raises(BudgetExhaustedError):
                await f.fetch_json(f"{base_url}/over")
            return True

        assert run(body, cfg(max_requests=3))
        assert [p for p in REQUEST_LOG if "over" in p] == []


class TestConcurrencyAndRecycling:
    def test_concurrency_is_capped(self, base_url: str) -> None:
        async def body(f):
            peak = {"now": 0, "max": 0}
            original = f._request

            async def counting(url, *, accept):
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
                try:
                    return await original(url, accept=accept)
                finally:
                    peak["now"] -= 1

            f._request = counting  # type: ignore[method-assign]
            await asyncio.gather(*(f.fetch_json(f"{base_url}/ok{i}") for i in range(12)))
            return peak["max"]

        assert run(body, cfg(concurrency=2)) <= 2

    def test_recycling_does_not_abort_concurrent_requests(self, base_url: str) -> None:
        # Retiring rather than closing the outgoing client is what makes this safe.
        async def body(f):
            results = await asyncio.gather(*(f.fetch_json(f"{base_url}/ok{i}") for i in range(24)))
            return len(results)

        assert run(body, cfg(concurrency=2, recycle_context_every=2)) == 24
