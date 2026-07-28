"""Phase drivers: claim work in batches, fetch, parse, merge, persist.

Every phase follows the same restartable shape as
``bama_scraper.pipeline.scrape_details``: claim a batch, mark it in flight,
gather, commit each outcome, print a progress line, and stop on a block, the
request budget or the runtime ceiling.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .ad_api import parse_ad_api
from .ad_html import parse_ad_html
from .config import DEEP_VERSION, DeepConfig
from .merge import merge_ad
from .net import (
    AbortedError,
    BlockedError,
    BudgetExhaustedError,
    DeepFetcher,
    PermanentFetchError,
)
from .storage import DeepStorage


class PhaseOutcome(dict):
    """Plain dict with attribute-free access; kept simple for JSON summaries."""


def _batch_size(cfg: DeepConfig) -> int:
    return max(cfg.concurrency * 25, 50)


async def run_phase_ads(
    cfg: DeepConfig,
    storage: DeepStorage,
    fetcher: DeepFetcher,
    run_id: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Phase A: fetch both views of every queued advertisement and merge them.

    The two requests per advertisement are issued **sequentially inside one
    worker task**, which keeps the instantaneous rate at ``concurrency`` rather
    than double it, and lets a 404 from the API skip the HTML fetch entirely.
    """
    recovered = storage.recover_stuck()
    if cfg.retry_delisted:
        storage.requeue_delisted()

    stats: dict[str, Any] = {
        "phase": "ads",
        "recovered_stuck": recovered,
        "attempted": 0,
        "completed": 0,
        "partial": 0,
        "delisted": 0,
        "permanent": 0,
        "retryable": 0,
        "skipped_after_block": 0,
        "conflicts": 0,
        "critical_conflicts": 0,
        "blocked": False,
        "budget_exhausted": False,
        "interrupted": False,
    }
    started = time.time()
    stop = {"stop": False, "reason": ""}

    async def handle(ad_id: str, url: str, api_url: str) -> None:
        api_result: dict[str, Any] | None = None
        html_result: dict[str, Any] | None = None
        api_status = html_status = "skipped"
        now = time.time()

        # --- JSON API ---
        if not cfg.skip_api:
            try:
                payload = await fetcher.fetch_json(api_url)
                api_result = parse_ad_api(payload, now_ts=now)
                api_status = "ok"
            except AbortedError:
                stats["skipped_after_block"] += 1
                storage.set_ad_status(ad_id, "pending")
                return
            except BudgetExhaustedError as exc:
                stats["budget_exhausted"] = True
                stop.update(stop=True, reason=str(exc))
                storage.set_ad_status(ad_id, "pending")
                return
            except BlockedError as exc:
                _register_block(storage, fetcher, stats, stop, run_id, ad_id, api_url, exc)
                storage.set_ad_status(ad_id, "pending")
                return
            except PermanentFetchError as exc:
                api_status = "404"
                storage.log_error(
                    run_id,
                    phase="ads",
                    entity_kind="ad",
                    entity_key=ad_id,
                    url=api_url,
                    stage="api",
                    error_type="permanent",
                    message=str(exc),
                )
            except Exception as exc:  # noqa: BLE001
                api_status = "error"
                storage.log_error(
                    run_id,
                    phase="ads",
                    entity_kind="ad",
                    entity_key=ad_id,
                    url=api_url,
                    stage="api",
                    error_type="retryable",
                    message=repr(exc),
                )

        # --- HTML page ---
        if not cfg.skip_html:
            try:
                html = await fetcher.fetch_html(url)
                html_result = parse_ad_html(html, now_ts=now)
                html_status = "ok" if html_result.get("ok") else "no_payload"
            except AbortedError:
                stats["skipped_after_block"] += 1
                storage.set_ad_status(ad_id, "pending")
                return
            except BudgetExhaustedError as exc:
                stats["budget_exhausted"] = True
                stop.update(stop=True, reason=str(exc))
                storage.set_ad_status(ad_id, "pending")
                return
            except BlockedError as exc:
                _register_block(storage, fetcher, stats, stop, run_id, ad_id, url, exc)
                storage.set_ad_status(ad_id, "pending")
                return
            except PermanentFetchError as exc:
                html_status = "404"
                storage.log_error(
                    run_id,
                    phase="ads",
                    entity_kind="ad",
                    entity_key=ad_id,
                    url=url,
                    stage="html",
                    error_type="permanent",
                    message=str(exc),
                )
            except Exception as exc:  # noqa: BLE001
                html_status = "error"
                storage.log_error(
                    run_id,
                    phase="ads",
                    entity_kind="ad",
                    entity_key=ad_id,
                    url=url,
                    stage="html",
                    error_type="retryable",
                    message=repr(exc),
                )

        stats["attempted"] += 1

        # Both sources unavailable and at least one said "gone" -> delisted.
        if api_result is None and html_result is None:
            if "404" in (api_status, html_status):
                merged = merge_ad(
                    None,
                    None,
                    ad_id=ad_id,
                    url=url,
                    api_url=api_url,
                    now_ts=now,
                    deep_version=DEEP_VERSION,
                )
                storage.save_ad(merged.record, [], [], [])
                storage.set_ad_status(
                    ad_id, "delisted", api_status=api_status, html_status=html_status
                )
                stats["delisted"] += 1
            else:
                stats["retryable"] += 1
                storage.set_ad_status(
                    ad_id,
                    "retryable_error",
                    api_status=api_status,
                    html_status=html_status,
                    error="both sources failed",
                    bump_attempt=True,
                )
            return

        try:
            merged = merge_ad(
                api_result,
                html_result,
                ad_id=ad_id,
                url=url,
                api_url=api_url,
                now_ts=now,
                deep_version=DEEP_VERSION,
            )
            storage.save_ad(merged.record, merged.media, merged.raw, merged.conflicts)
            stats["conflicts"] += len(merged.conflicts)
            stats["critical_conflicts"] += sum(
                1 for c in merged.conflicts if c["severity"] == "critical"
            )
            status = "completed" if merged.record.get("parse_status") == "ok" else "partial"
            stats["completed" if status == "completed" else "partial"] += 1
            storage.set_ad_status(ad_id, status, api_status=api_status, html_status=html_status)
        except Exception as exc:  # noqa: BLE001
            stats["retryable"] += 1
            storage.set_ad_status(
                ad_id,
                "retryable_error",
                api_status=api_status,
                html_status=html_status,
                error=f"merge/persist: {exc!r}",
                bump_attempt=True,
            )
            storage.log_error(
                run_id,
                phase="ads",
                entity_kind="ad",
                entity_key=ad_id,
                url=url,
                stage="merge",
                error_type="parse_error",
                message=repr(exc),
            )

    # -- batch loop --------------------------------------------------------
    processed = 0
    snapshot = storage.all_queue() if cfg.refresh else None
    try:
        while True:
            size = _batch_size(cfg)
            if limit is not None:
                if processed >= limit:
                    break
                size = min(size, limit - processed)
            if snapshot is not None:
                batch = snapshot[:size]
                snapshot = snapshot[size:]
            else:
                batch = storage.claim_ads(size, max_attempts=cfg.max_retries)
            if not batch:
                break
            processed += len(batch)
            storage.mark_ads_fetching([r["ad_id"] for r in batch])
            await asyncio.gather(
                *(handle(r["ad_id"], r["url"], r["api_url"]) for r in batch),
                return_exceptions=True,
            )
            breakdown = storage.status_breakdown("deep_queue")
            print(
                f"ads completed={breakdown.get('completed', 0)} "
                f"partial={breakdown.get('partial', 0)} "
                f"delisted={breakdown.get('delisted', 0)} "
                f"pending={breakdown.get('pending', 0) + breakdown.get('fetching', 0)} "
                f"conflicts={stats['conflicts']} "
                f"requests={fetcher.requests_issued} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            storage.set_checkpoint(run_id, "ads_processed", processed)
            if stop["stop"]:
                break
            if time.time() - started > cfg.max_runtime:
                stats["max_runtime_reached"] = True
                break
    except KeyboardInterrupt:
        stats["interrupted"] = True

    stats["elapsed_seconds"] = round(time.time() - started, 1)
    stats["requests_issued"] = fetcher.requests_issued
    stats["stop_reason"] = stop["reason"] or None
    stats["queue"] = storage.status_breakdown("deep_queue")
    return stats


def _register_block(
    storage: DeepStorage,
    fetcher: DeepFetcher,
    stats: dict[str, Any],
    stop: dict[str, Any],
    run_id: str,
    entity: str,
    url: str,
    exc: Exception,
) -> None:
    """Record a block once and stop the whole run from issuing more requests."""
    if stats.get("blocked"):
        return
    stats["blocked"] = True
    stop.update(stop=True, reason=str(exc))
    fetcher.abort(str(exc))
    storage.log_error(
        run_id,
        phase="ads",
        entity_kind="ad",
        entity_key=entity,
        url=url,
        stage="fetch",
        error_type="blocked",
        message=str(exc),
    )


def print_summary(name: str, stats: dict[str, Any]) -> None:
    payload = {k: v for k, v in stats.items() if k != "queue"}
    print(f"[{name}] {json.dumps(payload, ensure_ascii=False, default=str)}", flush=True)
