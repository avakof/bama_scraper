"""Command-line interface for the Bama scraper."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from typing import Any

from .backfill import backfill_published_ts
from .config import SCRAPER_VERSION, Config, load_config
from .discovery import parse_filters, run_discovery
from .export import export_all, export_sqlite_copy
from .logging_config import configure_logging, get_logger
from .models import NetworkCandidate
from .pipeline import scrape_details
from .storage import Storage
from .validation import audit, verification_pass, write_summary

log = get_logger(__name__)


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bama_scraper",
        description="Scrape Bama.ir filtered car search results and detail pages.",
    )
    parser.add_argument("--config", help="path to a YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--url", default=None, help="target search URL")
        p.add_argument("--mode", choices=["auto", "api", "pagination", "scroll"], default=None)
        p.add_argument("--headed", dest="headed", action="store_true", default=None)
        p.add_argument("--headless", dest="headed", action="store_false", default=None)
        p.add_argument("--max-scrolls", type=int, default=None)
        p.add_argument("--stale-cycles", type=int, default=None)
        p.add_argument("--max-runtime", type=float, default=None)
        p.add_argument("--concurrency", type=int, default=None)
        p.add_argument("--delay-min", type=float, default=None)
        p.add_argument("--delay-max", type=float, default=None)
        p.add_argument("--resume", action="store_true", default=None)
        p.add_argument("--refresh", action="store_true", default=None)
        p.add_argument("--download-images", action="store_true", default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument("--log-level", default=None)

    for name, help_text in [
        ("inspect", "inspect the live page and record network candidates"),
        ("discover", "discover advertisements from the search results"),
        ("scrape-details", "fetch and parse every discovered detail page"),
        ("run", "discover + scrape-details + validate + export"),
        ("export", "write CSV/JSONL/Parquet/SQLite exports"),
        ("validate", "run the completeness audit"),
        ("backfill", "recompute derived fields on stored records (no refetching)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        common(p)
        if name == "validate":
            p.add_argument(
                "--skip-second-pass",
                action="store_true",
                help="audit stored data without re-querying the site",
            )
        if name == "scrape-details":
            p.add_argument(
                "--limit",
                type=int,
                default=None,
                help="stop after this many advertisements (smoke tests)",
            )
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    return load_config(
        getattr(args, "config", None),
        url=args.url,
        mode=args.mode,
        headed=args.headed,
        max_scrolls=args.max_scrolls,
        stale_cycles=args.stale_cycles,
        max_runtime=args.max_runtime,
        concurrency=args.concurrency,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        resume=args.resume,
        refresh=args.refresh,
        download_images=args.download_images,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def _cmd_inspect(cfg: Config, storage: Storage, run_id: str) -> dict[str, Any]:
    """Open the live page, record XHR/fetch traffic, and report the mechanism."""
    from .browser import playwright_page

    candidates: dict[str, NetworkCandidate] = {}
    findings: dict[str, Any] = {"url": cfg.url, "filters": parse_filters(cfg.url)}

    async with playwright_page(cfg) as page:

        async def on_response(response: Any) -> None:
            try:
                request = response.request
                if request.resource_type not in ("xhr", "fetch"):
                    return
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
                    notes=json.dumps(body.get("metadata"), ensure_ascii=False)
                    if isinstance(body, dict) and body.get("metadata")
                    else None,
                )
            except Exception:
                return

        page.on("response", on_response)
        await page.goto(
            cfg.url, wait_until="domcontentloaded", timeout=int(cfg.navigation_timeout * 1000)
        )
        await page.wait_for_timeout(cfg.settle_ms)

        findings["initial"] = await page.evaluate(
            """() => ({
                 links: document.querySelectorAll('a[href*="/car/detail-"]').length,
                 height: document.body.scrollHeight,
                 hasNuxt: !!document.getElementById('__NUXT_DATA__'),
                 loadMoreControls: [...document.querySelectorAll('button,a[role=button]')]
                    .filter(b => /بیشتر|بارگذاری|load more/i.test(b.textContent||''))
                    .map(b => b.textContent.trim()).slice(0, 5),
                 jsonLd: document.querySelectorAll('script[type="application/ld+json"]').length,
               })"""
        )
        observations = []
        for _ in range(5):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(cfg.scroll_pause_ms + 800)
            observations.append(
                await page.evaluate(
                    """() => ({
                     links: document.querySelectorAll('a[href*="/car/detail-"]').length,
                     height: document.body.scrollHeight })"""
                )
            )
        findings["scroll_observations"] = observations
        cfg.ensure_dirs()
        await page.screenshot(path=str(cfg.debug_dir / f"inspect_{run_id}.png"))
        (cfg.debug_dir / f"inspect_{run_id}.html").write_text(
            await page.content(), encoding="utf-8"
        )

    storage.save_network_candidates(candidates.values())
    listing = [c for c in candidates.values() if c.is_listing_endpoint]
    findings["listing_endpoints"] = [c.url for c in listing]
    findings["mechanism"] = (
        "public JSON endpoint driven by infinite scroll"
        if listing
        else "no listing XHR observed; DOM scrolling required"
    )
    findings["network_candidate_count"] = len(candidates)
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    write_summary(findings, cfg.output_dir / "inspect_report.json")
    return findings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    configure_logging(cfg.log_level)
    cfg.ensure_dirs()
    run_id = _new_run_id()

    storage = Storage(cfg.db_path)
    filters = parse_filters(cfg.url)
    storage.start_run(run_id, cfg.url, filters, cfg.mode, SCRAPER_VERSION)
    started = time.time()
    command = args.command

    try:
        if command == "inspect":
            asyncio.run(_cmd_inspect(cfg, storage, run_id))
            return 0

        if command == "backfill":
            result = backfill_published_ts(storage, only_missing=not cfg.refresh)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if command == "export":
            manifest = export_all(storage, cfg.output_dir)
            export_sqlite_copy(storage, cfg.output_dir / "bama_ads.sqlite")
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0

        if command == "validate":
            extra: dict[str, Any] = {}
            if not getattr(args, "skip_second_pass", False):
                extra["verification_pass"] = verification_pass(cfg, storage, run_id)
            # Carry termination evidence forward even when discovery ran in an
            # earlier invocation, so the summary is self-contained.
            discovery = storage.latest_discovery_result()
            if discovery:
                extra["discovery"] = discovery
                extra["termination_reason"] = discovery.get("termination_reason")
                extra["api_pages_or_cycles"] = discovery.get("cycles")
                extra["stale_cycles_at_termination"] = discovery.get("stale_at_end")
                extra["discovery_elapsed_seconds"] = discovery.get("elapsed_seconds")
            report = audit(storage, cfg, run_id, extra)
            report["applied_filters"] = filters
            report["audited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_summary(report, cfg.output_dir / "run_summary.json")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["checks_passed"] else 1

        discovery_result: dict[str, Any] = {}
        if command in ("discover", "run"):
            log.info(
                "discovery.start", url=cfg.url, mode=cfg.mode, filters=filters["interpretation"]
            )
            discovery_result = run_discovery(cfg, storage, run_id)
            storage.set_checkpoint(run_id, "discovery_result", discovery_result)
            log.info(
                "discovery.done",
                **{
                    k: v
                    for k, v in discovery_result.items()
                    if k not in ("api_params", "last_metadata")
                },
            )

        detail_stats: dict[str, Any] = {}
        if command in ("scrape-details", "run"):
            limit = getattr(args, "limit", None)
            detail_stats = asyncio.run(scrape_details(cfg, storage, run_id, limit=limit))
            log.info("details.done", **detail_stats)

        if command == "run":
            verification = verification_pass(cfg, storage, run_id)
            if not verification["stable"]:
                log.warning("verification.new_ads_found_rescraping", n=verification["newly_added"])
                detail_stats = asyncio.run(scrape_details(cfg, storage, run_id))
                verification = verification_pass(cfg, storage, run_id)

            manifest = export_all(storage, cfg.output_dir)
            export_sqlite_copy(storage, cfg.output_dir / "bama_ads.sqlite")
            report = audit(
                storage,
                cfg,
                run_id,
                {
                    "run_id": run_id,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "elapsed_seconds": round(time.time() - started, 1),
                    "target_url": cfg.url,
                    "applied_filters": filters,
                    "scraper_version": SCRAPER_VERSION,
                    "discovery": discovery_result,
                    "detail_stats": detail_stats,
                    "verification_pass": verification,
                    "export_manifest": manifest,
                },
            )
            write_summary(report, cfg.output_dir / "run_summary.json")
            storage.finish_run(
                run_id,
                discovery_result.get("termination_reason", "n/a"),
                discovery_result.get("cycles", 0),
                discovery_result.get("stale_at_end", 0),
                report,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return 0 if report["checks_passed"] else 1

        print(json.dumps(storage.stats(), ensure_ascii=False, indent=2))
        return 0

    except KeyboardInterrupt:
        log.warning("interrupted.progress_persisted", **storage.stats())
        return 130
    finally:
        storage.close()


if __name__ == "__main__":
    sys.exit(main())
