"""Detail-page scraping orchestration: claim work, fetch, parse, persist."""

from __future__ import annotations

import asyncio
import gzip
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .browser import (
    AbortedError,
    BlockedError,
    DetailFetcher,
    PermanentFetchError,
    RetryableFetchError,
)
from .config import SCRAPER_VERSION, Config
from .detail_parser import parse_detail
from .logging_config import get_logger
from .storage import Storage

log = get_logger(__name__)


async def scrape_details(
    cfg: Config, storage: Storage, run_id: str, *, limit: int | None = None
) -> dict[str, Any]:
    """Fetch and parse every advertisement still awaiting a detail page.

    Work is claimed in batches so an interrupted run resumes cleanly: rows move
    ``pending -> scraping -> completed`` and any row left in ``scraping`` by a
    crashed process is recovered on the next start.

    ``limit`` caps how many advertisements this invocation processes (used by
    smoke tests). It only bounds the claim loop -- it never mutates the status
    of ads it does not process, so a later full run still picks them up.
    """
    cfg.ensure_dirs()
    recovered = storage.recover_stuck()
    if recovered:
        log.info("details.recovered_stuck_rows", n=recovered)

    started = time.time()
    stats: dict[str, Any] = {
        "completed": 0,
        "failed": 0,
        "retryable": 0,
        "permanent": 0,
        "blocked": False,
        "warnings": 0,
        "attempted": 0,
        "images_downloaded": 0,
        "skipped_after_block": 0,
    }
    stop_flag = {"stop": False, "reason": ""}
    interrupted = False

    async with DetailFetcher(cfg) as fetcher:

        async def handle(url: str, ad_id: str) -> None:
            try:
                html, source = await fetcher.fetch(url)
            except AbortedError:
                # Never sent: keep the ad claimable and do not count an attempt.
                storage.mark_status(url, "pending")
                stats["skipped_after_block"] += 1
                return
            except BlockedError as exc:
                stats["attempted"] += 1
                if not stop_flag["stop"]:
                    stop_flag["stop"] = True
                    stop_flag["reason"] = str(exc)
                    stats["blocked"] = True
                    fetcher.abort()  # drain the rest of the batch without requests
                    storage.log_error(run_id, ad_id, url, "detail", "blocked", str(exc))
                    log.error("details.blocked_stopping", url=url, error=str(exc))
                storage.mark_status(url, "pending")
                return
            except PermanentFetchError as exc:
                stats["attempted"] += 1
                stats["permanent"] += 1
                stats["failed"] += 1
                storage.mark_status(url, "permanent_error", str(exc), bump_attempt=True)
                storage.log_error(run_id, ad_id, url, "detail", "permanent", str(exc))
                return
            except (RetryableFetchError, Exception) as exc:  # noqa: BLE001
                stats["attempted"] += 1
                stats["retryable"] += 1
                stats["failed"] += 1
                storage.mark_status(url, "retryable_error", repr(exc), bump_attempt=True)
                storage.log_error(run_id, ad_id, url, "detail", "retryable", repr(exc))
                return

            stats["attempted"] += 1
            try:
                detail, media = parse_detail(html, url, ad_id, scraper_version=SCRAPER_VERSION)
                detail.fetch_source = source
                storage.save_detail(detail, media)
                if cfg.download_images and detail.image_urls:
                    stats["images_downloaded"] += await _download_images(
                        cfg, fetcher, ad_id, detail.image_urls
                    )
                storage.mark_status(url, "completed", None, bump_attempt=True)
                stats["completed"] += 1
                stats["warnings"] += len(detail.parse_warnings)
                if detail.parse_status != "ok" and cfg.save_failed_html:
                    _dump_html(cfg, ad_id, html)
            except Exception as exc:  # noqa: BLE001
                stats["failed"] += 1
                storage.mark_status(url, "retryable_error", f"parse: {exc!r}", bump_attempt=True)
                storage.log_error(run_id, ad_id, url, "parse", "parse_error", repr(exc))
                if cfg.save_failed_html:
                    _dump_html(cfg, ad_id, html)

        try:
            processed = 0
            # --refresh walks a fixed snapshot of every ad; the normal path is
            # status-driven and shrinks as rows complete.
            refresh_queue = storage.all_ads() if cfg.refresh else None
            while True:
                batch_size = max(cfg.concurrency * 25, 50)
                if limit is not None:
                    if processed >= limit:
                        break
                    batch_size = min(batch_size, limit - processed)
                if refresh_queue is not None:
                    batch = refresh_queue[:batch_size]
                    refresh_queue = refresh_queue[batch_size:]
                else:
                    batch = storage.claim_pending(
                        limit=batch_size,
                        refresh=False,
                        max_attempts=cfg.max_retries,
                    )
                if not batch:
                    break
                processed += len(batch)
                urls = [r["url"] for r in batch]
                storage.mark_many_scraping(urls)
                await asyncio.gather(
                    *(handle(r["url"], r["ad_id"]) for r in batch),
                    return_exceptions=True,
                )
                done = storage.stats()
                print(
                    f"details completed={done['completed']} pending={done['pending']} "
                    f"retryable={done['retryable']} permanent={done['permanent']} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
                if stop_flag["stop"]:
                    log.error("details.aborting_due_to_block", reason=stop_flag["reason"])
                    break
                if time.time() - started > cfg.max_runtime:
                    log.warning("details.max_runtime_reached")
                    break
        except KeyboardInterrupt:
            interrupted = True
            log.warning("details.interrupted_progress_saved", **storage.stats())

    stats["elapsed_seconds"] = time.time() - started
    stats["blocked_reason"] = stop_flag["reason"] or None
    stats["interrupted"] = interrupted
    return stats


async def _download_images(cfg: Config, fetcher: DetailFetcher, ad_id: str, urls: list[str]) -> int:
    """Download an advertisement's images when ``--download-images`` is set.

    Files already on disk are skipped so a resumed run does not refetch them.
    """
    target = cfg.output_dir / "images" / ad_id
    target.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, url in enumerate(urls):
        suffix = Path(urlparse(url).path).suffix or ".jpg"
        path = target / f"{i:03d}{suffix}"
        if path.exists():
            continue
        data = await fetcher.download(url)
        if data:
            path.write_bytes(data)
            saved += 1
    return saved


def _dump_html(cfg: Config, ad_id: str, html: str) -> None:
    """Persist compressed HTML for failed or ambiguous parses only."""
    try:
        path = cfg.debug_dir / f"{ad_id}.html.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(html)
    except Exception:
        pass
