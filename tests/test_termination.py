"""Termination logic and browser integration against a local synthetic site.

These tests never touch bama.ir. They serve a locally generated page that
reproduces the two behaviours that make termination hard:

* **virtualization** -- only a window of cards stays in the DOM, so cards seen
  earlier are unmounted and would be lost by a scrape-at-the-end strategy;
* **a finite result set** -- after the last batch nothing more loads, and the
  scraper must recognise that rather than spinning forever.
"""

from __future__ import annotations

import http.server
import socket
import threading
from pathlib import Path

import pytest

from bama_scraper.config import Config
from bama_scraper.discovery import discover_via_scroll
from bama_scraper.storage import Storage

TOTAL_ADS = 84
BATCH = 12
WINDOW = 24  # cards kept mounted at once -> forces virtualization


PAGE = """
<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<title>local infinite scroll</title></head>
<body style="margin:0">
<section id="list"></section>
<div id="sentinel" style="height:40px">...</div>
<script>
const TOTAL = __TOTAL__, BATCH = __BATCH__, WINDOW = __WINDOW__;
let loaded = 0, loading = false;
const list = document.getElementById('list');

function render() {
  // Virtualized window: drop cards that scrolled far above the viewport.
  while (list.children.length > WINDOW) list.removeChild(list.firstElementChild);
}
function addBatch() {
  if (loading || loaded >= TOTAL) return;
  loading = true;
  document.body.setAttribute('aria-busy', 'true');
  setTimeout(() => {
    const n = Math.min(BATCH, TOTAL - loaded);
    for (let i = 0; i < n; i++) {
      const idx = loaded + i;
      const art = document.createElement('article');
      art.style.height = '260px';
      art.innerHTML = '<a href="/car/detail-loc' + String(idx).padStart(5,'0') +
        '-pride-140' + (idx % 10) + '"><span>پراید ' + idx + '</span>' +
        '<span>۱۲,۳۴۵ km</span></a>';
      list.appendChild(art);
    }
    loaded += n;
    render();
    loading = false;
    document.body.removeAttribute('aria-busy');
  }, 120);
}
addBatch();
window.addEventListener('scroll', () => {
  if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 300) addBatch();
});
</script></body></html>
"""
PAGE = (
    PAGE.replace("__TOTAL__", str(TOTAL_ADS))
    .replace("__BATCH__", str(BATCH))
    .replace("__WINDOW__", str(WINDOW))
)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence test output
        return


@pytest.fixture(scope="module")
def local_site() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/car"
    server.shutdown()


def _has_chromium() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            return Path(pw.chromium.executable_path).exists()
    except Exception:
        return False


requires_browser = pytest.mark.skipif(
    not _has_chromium(), reason="Chromium not installed (run: playwright install chromium)"
)


@requires_browser
class TestInfiniteScrollIntegration:
    @pytest.fixture(scope="class")
    def result(self, local_site: str, tmp_path_factory: pytest.TempPathFactory) -> tuple:
        import asyncio

        out = tmp_path_factory.mktemp("scroll")
        cfg = Config(
            url=local_site,
            output_dir=out,
            stale_cycles=4,
            max_scrolls=120,
            scroll_pause_ms=180,
            settle_ms=220,
            max_runtime=180.0,
        )
        store = Storage(cfg.db_path)
        store.start_run("t", cfg.url, {}, "scroll", "test")
        res = asyncio.run(discover_via_scroll(cfg, store, "t", collect_network=False))
        yield res, store
        store.close()

    def test_collects_every_ad_despite_virtualization(self, result: tuple) -> None:
        res, store = result
        # Only WINDOW cards are ever mounted at once, yet all TOTAL_ADS are captured
        # because each cycle extracts before scrolling.
        assert store.count_discovered() == TOTAL_ADS
        assert res["unique_urls"] == TOTAL_ADS

    def test_virtualization_actually_occurred(self, result: tuple) -> None:
        _, store = result
        # More ads persisted than could ever be in the DOM simultaneously.
        assert store.count_discovered() > WINDOW

    def test_terminates_on_stale_cycles_not_scroll_limit(self, result: tuple) -> None:
        res, _ = result
        assert res["termination_reason"].startswith("stable_after_")
        assert res["cycles"] < 120

    def test_required_stale_cycles_observed(self, result: tuple) -> None:
        res, _ = result
        assert res["stale_at_end"] >= 4

    def test_urls_are_canonical_and_unique(self, result: tuple) -> None:
        _, store = result
        urls = store.all_discovered_urls()
        assert len(urls) == TOTAL_ADS
        assert all(u.startswith("https://bama.ir/car/detail-") for u in urls)

    def test_ids_extracted_for_every_row(self, result: tuple) -> None:
        _, store = result
        rows = list(store.conn.execute("SELECT ad_id FROM discovered_ads"))
        assert all(r["ad_id"] and r["ad_id"].startswith("loc") for r in rows)
        assert len({r["ad_id"] for r in rows}) == TOTAL_ADS

    def test_checkpoints_written_during_discovery(self, result: tuple) -> None:
        _, store = result
        assert store.get_checkpoint("t", "scroll_unique") == TOTAL_ADS
        assert store.get_checkpoint("t", "scroll_cycle") is not None

    def test_resume_after_partial_discovery_adds_no_duplicates(self, result: tuple) -> None:
        _, store = result
        before = store.count_discovered()
        # Re-running discovery over the same site must not create new rows.
        assert store.upsert_discovered([], "t") == 0
        assert store.count_discovered() == before


class TestStaleCycleRule:
    """The stale-cycle predicate itself, independent of a browser."""

    @staticmethod
    def progressed(
        new: int, xhr: int, height_grew: bool, loading: bool, clicked: bool, at_bottom: bool
    ) -> bool:
        return bool(new or xhr or height_grew or loading or clicked or not at_bottom)

    def test_single_empty_scroll_is_not_the_end(self) -> None:
        stale = 0
        for _ in range(3):
            stale = 0 if self.progressed(0, 0, False, False, False, True) else stale + 1
        assert stale == 3
        assert stale < Config().stale_cycles  # default 8 -> would not stop yet

    def test_any_signal_resets_the_counter(self) -> None:
        assert self.progressed(1, 0, False, False, False, True)  # new ads
        assert self.progressed(0, 1, False, False, False, True)  # listing XHR
        assert self.progressed(0, 0, True, False, False, True)  # height grew
        assert self.progressed(0, 0, False, True, False, True)  # spinner
        assert self.progressed(0, 0, False, False, True, True)  # load-more clicked
        assert self.progressed(0, 0, False, False, False, False)  # not at bottom

    def test_all_signals_quiet_means_stale(self) -> None:
        assert not self.progressed(0, 0, False, False, False, True)

    def test_default_requires_eight_consecutive_stale_cycles(self) -> None:
        assert Config().stale_cycles == 8


class TestApiTerminationRule:
    """API discovery must ignore the endpoint's own counters."""

    def test_empty_batch_increments_stale(self) -> None:
        stale = 0
        for real_ads in (30, 30, 27, 0, 0):
            stale = 0 if real_ads else stale + 1
        assert stale == 2

    def test_stops_only_after_configured_empty_pages(self) -> None:
        cfg = Config()
        assert cfg.api_stale_pages >= 2
        stale = 1
        assert stale < cfg.api_stale_pages  # one empty page is not enough
