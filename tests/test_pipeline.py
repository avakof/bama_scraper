"""Pipeline orchestration: interruption, blocking, resume and status transitions.

Runs against a local server so no live site is involved.
"""

from __future__ import annotations

import asyncio
import http.server
import socket
import threading
from pathlib import Path

import pytest

from bama_scraper.config import Config
from bama_scraper.models import DiscoveredAd
from bama_scraper.pipeline import scrape_details
from bama_scraper.storage import Storage

NUXT = (
    '[{"data":1},{"get-ad-pdp-car_x":2},{"data":3},'
    '{"code":4,"content":5,"vehicle":7},"code0001",'
    '{"title":6},"\\u067e\\u0631\\u0627\\u06cc\\u062f",{"year":8},{"value":1400}]'
)
PAGE = (
    f'<html><body><script type="application/json" id="__NUXT_DATA__">{NUXT}</script></body></html>'
)

_state = {"block_after": None, "served": 0}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        _state["served"] += 1
        limit = _state["block_after"]
        if limit is not None and _state["served"] > limit:
            code, body = 429, b"rate limited"
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


@pytest.fixture(autouse=True)
def _reset() -> None:
    _state["block_after"] = None
    _state["served"] = 0


def seed(store: Storage, base_url: str, n: int) -> None:
    ads = [
        DiscoveredAd(
            ad_id=f"code{i:04d}",
            url=f"{base_url}/car/detail-code{i:04d}-pride-1400",
            source_search_url=base_url,
            discovery_position=i,
            discovery_cycle=0,
        )
        for i in range(n)
    ]
    store.upsert_discovered(ads, "run1")


def make(tmp_path: Path, base_url: str, n: int, **kw) -> tuple[Config, Storage]:
    cfg = Config(url=base_url, output_dir=tmp_path, delay_min=0.0, delay_max=0.0, **kw)
    store = Storage(cfg.db_path)
    store.start_run("run1", base_url, {}, "api", "test")
    seed(store, base_url, n)
    return cfg, store


class TestHappyPath:
    def test_all_ads_completed(self, tmp_path: Path, base_url: str) -> None:
        cfg, store = make(tmp_path, base_url, 6)
        stats = asyncio.run(scrape_details(cfg, store, "run1"))
        assert stats["completed"] == 6
        assert store.stats()["completed"] == 6
        assert store.stats()["pending"] == 0
        store.close()

    def test_limit_leaves_the_rest_claimable(self, tmp_path: Path, base_url: str) -> None:
        cfg, store = make(tmp_path, base_url, 10)
        stats = asyncio.run(scrape_details(cfg, store, "run1", limit=4))
        assert stats["completed"] == 4
        # the untouched ads must remain claimable, not parked in a dead status
        assert len(store.claim_pending(50)) == 6
        store.close()

    def test_rerun_is_a_noop_when_everything_is_done(self, tmp_path: Path, base_url: str) -> None:
        cfg, store = make(tmp_path, base_url, 5)
        asyncio.run(scrape_details(cfg, store, "run1"))
        again = asyncio.run(scrape_details(cfg, store, "run1"))
        assert again["attempted"] == 0  # resume skips completed work
        store.close()

    def test_refresh_rescrapes_every_ad_not_just_one_batch(
        self, tmp_path: Path, base_url: str
    ) -> None:
        # Regression: --refresh used to break after the first batch.
        cfg, store = make(tmp_path, base_url, 120, concurrency=4)
        asyncio.run(scrape_details(cfg, store, "run1"))
        refreshed = asyncio.run(
            scrape_details(cfg.model_copy(update={"refresh": True}), store, "run1")
        )
        assert refreshed["attempted"] == 120
        store.close()


class TestInterruption:
    def test_ctrl_c_persists_progress_and_reports_it(
        self, tmp_path: Path, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+C must save what was done rather than losing the run."""
        cfg, store = make(tmp_path, base_url, 200)

        import bama_scraper.pipeline as pipeline

        real_stats = Storage.stats
        calls = {"n": 0}

        def exploding_stats(self):  # raise KeyboardInterrupt between batches
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
            return real_stats(self)

        monkeypatch.setattr(Storage, "stats", exploding_stats)
        stats = asyncio.run(pipeline.scrape_details(cfg, store, "run1"))
        monkeypatch.undo()

        assert stats["interrupted"] is True
        # work completed before the interrupt is durably stored
        done = store.stats()["completed"]
        assert done > 0
        assert done == store.conn.execute("SELECT COUNT(*) FROM ad_details").fetchone()[0]
        store.close()

    def test_progress_survives_reopen_and_resume_finishes(
        self, tmp_path: Path, base_url: str
    ) -> None:
        cfg, store = make(tmp_path, base_url, 20)
        asyncio.run(scrape_details(cfg, store, "run1", limit=8))
        completed_before = store.stats()["completed"]
        store.close()

        reopened = Storage(cfg.db_path)
        assert reopened.stats()["completed"] == completed_before
        asyncio.run(scrape_details(cfg, reopened, "run1"))
        assert reopened.stats()["completed"] == 20
        assert reopened.stats()["pending"] == 0
        reopened.close()


class TestBlocking:
    def test_429_stops_the_run_gracefully(self, tmp_path: Path, base_url: str) -> None:
        """Regression: a whole batch used to keep firing after the first 429.

        Every coroutine in a batch passes any pre-fetch check before the first
        request completes, so the abort must be enforced *inside* the fetch,
        after the concurrency slot is acquired.
        """
        _state["block_after"] = 3
        cfg, store = make(tmp_path, base_url, 40, concurrency=1, max_retries=1)
        stats = asyncio.run(scrape_details(cfg, store, "run1"))

        assert stats["blocked"] is True
        assert stats["blocked_reason"] and "429" in stats["blocked_reason"]
        # Stops immediately: only the request that hit the block is attempted
        # beyond the successful ones; the rest are skipped without a request.
        assert stats["attempted"] <= 5
        assert stats["skipped_after_block"] >= 30
        # No request is sent for the skipped ads.
        assert _state["served"] <= 5
        # The block is recorded exactly once, not once per queued ad.
        errors = store.conn.execute(
            "SELECT COUNT(*) FROM scrape_errors WHERE error_type='blocked'"
        ).fetchone()[0]
        assert errors == 1
        store.close()

    def test_blocked_ads_stay_claimable_without_burning_retries(
        self, tmp_path: Path, base_url: str
    ) -> None:
        _state["block_after"] = 3
        cfg, store = make(tmp_path, base_url, 40, concurrency=1, max_retries=1)
        asyncio.run(scrape_details(cfg, store, "run1"))
        # Ads we never sent must remain retryable on the next run: a block is
        # the site's problem, not the advertisement's.
        remaining = store.claim_pending(100, max_attempts=1)
        assert len(remaining) >= 30
        assert all(r["attempts"] == 0 for r in remaining)
        store.close()

    def test_progress_before_the_block_is_kept(self, tmp_path: Path, base_url: str) -> None:
        _state["block_after"] = 5
        cfg, store = make(tmp_path, base_url, 40, concurrency=1, max_retries=1)
        asyncio.run(scrape_details(cfg, store, "run1"))
        assert store.stats()["completed"] >= 1
        store.close()
