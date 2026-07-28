"""DetailFetcher behaviour against a local server: retries, errors, recycling."""

from __future__ import annotations

import asyncio
import http.server
import socket
import threading

import pytest

from bama_scraper.browser import (
    BlockedError,
    DetailFetcher,
    PermanentFetchError,
)
from bama_scraper.config import Config

PAGE = (
    '<html><body><script type="application/json" id="__NUXT_DATA__">'
    '[{"data":1},{"x":2},"ok"]</script></body></html>'
)


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves 200 normally; /403, /429, /404 and /500 exercise the error paths."""

    flaky_hits: dict[str, int] = {}

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if path.startswith("/403"):
            code, body = 403, b"forbidden"
        elif path.startswith("/429"):
            code, body = 429, b"slow down"
        elif path.startswith("/404"):
            code, body = 404, b"gone"
        elif path.startswith("/flaky"):
            # fail twice, then succeed -- exercises bounded retry
            n = self.flaky_hits.get(path, 0) + 1
            self.flaky_hits[path] = n
            if n <= 2:
                code, body = 500, b"boom"
            else:
                code, body = 200, PAGE.encode()
        else:
            code, body = 200, PAGE.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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


def _cfg(**kw) -> Config:
    return Config(delay_min=0.0, delay_max=0.0, request_timeout=10.0, **kw)


class TestFetching:
    def test_successful_fetch(self, base_url: str) -> None:
        async def run() -> tuple[str, str]:
            async with DetailFetcher(_cfg()) as f:
                return await f.fetch(f"{base_url}/ok")

        html, source = asyncio.run(run())
        assert "__NUXT_DATA__" in html and source == "httpx"

    @pytest.mark.parametrize("code", ["403", "429"])
    def test_block_signals_raise_blocked_error(self, base_url: str, code: str) -> None:
        async def run() -> None:
            async with DetailFetcher(_cfg()) as f:
                await f.fetch(f"{base_url}/{code}")

        # A block must surface distinctly so the pipeline can stop gracefully
        # rather than hammering the site harder.
        with pytest.raises(BlockedError):
            asyncio.run(run())

    def test_404_is_permanent(self, base_url: str) -> None:
        async def run() -> None:
            async with DetailFetcher(_cfg()) as f:
                await f.fetch(f"{base_url}/404")

        with pytest.raises(PermanentFetchError):
            asyncio.run(run())

    def test_transient_500_is_retried_then_succeeds(self, base_url: str) -> None:
        async def run() -> str:
            async with DetailFetcher(_cfg()) as f:
                html, _ = await f.fetch(f"{base_url}/flaky-a")
                return html

        assert "__NUXT_DATA__" in asyncio.run(run())


class TestRecycling:
    def test_recycle_does_not_abort_concurrent_requests(self, base_url: str) -> None:
        """Regression: swapping the client mid-flight killed in-flight requests.

        With recycling every 2 fetches and 2 workers, a naive implementation
        closes the client while the other worker is still using it.
        """

        async def run() -> list[str]:
            cfg = _cfg(concurrency=2, recycle_context_every=2)
            async with DetailFetcher(cfg) as f:
                results = await asyncio.gather(*(f.fetch(f"{base_url}/ok{i}") for i in range(24)))
                return [src for _, src in results]

        sources = asyncio.run(run())
        assert len(sources) == 24
        assert all(s == "httpx" for s in sources)

    def test_concurrency_limit_is_respected(self, base_url: str) -> None:
        async def run() -> int:
            cfg = _cfg(concurrency=2)
            peak = {"now": 0, "max": 0}
            async with DetailFetcher(cfg) as f:
                original = f._get

                async def counting(url: str) -> str:
                    peak["now"] += 1
                    peak["max"] = max(peak["max"], peak["now"])
                    try:
                        return await original(url)
                    finally:
                        peak["now"] -= 1

                f._get = counting  # type: ignore[method-assign]
                await asyncio.gather(*(f.fetch(f"{base_url}/ok{i}") for i in range(12)))
            return peak["max"]

        assert asyncio.run(run()) <= 2


class TestImageDownload:
    """`--download-images` must actually write files, and skip existing ones."""

    def test_downloads_and_skips_existing(self, base_url: str, tmp_path) -> None:
        from bama_scraper.pipeline import _download_images

        async def run() -> tuple[int, int]:
            cfg = _cfg(output_dir=tmp_path, download_images=True)
            urls = [f"{base_url}/a.jpg", f"{base_url}/b.jpg"]
            async with DetailFetcher(cfg) as f:
                first = await _download_images(cfg, f, "ad123", urls)
                second = await _download_images(cfg, f, "ad123", urls)
            return first, second

        first, second = asyncio.run(run())
        assert first == 2
        assert second == 0  # already on disk -> resumed runs do not refetch
        files = sorted((tmp_path / "images" / "ad123").iterdir())
        assert [p.name for p in files] == ["000.jpg", "001.jpg"]
        assert all(p.stat().st_size > 0 for p in files)

    def test_failed_asset_does_not_raise(self, base_url: str, tmp_path) -> None:
        from bama_scraper.pipeline import _download_images

        async def run() -> int:
            cfg = _cfg(output_dir=tmp_path, download_images=True)
            async with DetailFetcher(cfg) as f:
                return await _download_images(cfg, f, "ad404", [f"{base_url}/404.jpg"])

        assert asyncio.run(run()) == 0  # best-effort: missing image is not fatal
