"""Command-line interface for the deep scraper."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from typing import Any

from bama_scraper.logging_config import configure_logging, get_logger

from .config import DEEP_VERSION, DeepConfig, load_config
from .net import DeepFetcher
from .phases import print_summary, run_phase_ads
from .seed import SeedError, file_sha256, seed_from_source
from .storage import DeepStorage

log = get_logger("bama_deep")

COMMANDS = (
    ("seed", "import the discovered-ad inventory from the existing database (read-only)"),
    ("ads", "phase A: per-ad JSON API + HTML, merged and cross-validated"),
    ("specs", "phase B: per-model-trim technical specifications"),
    ("prices", "phase C: per-model-trim daily price history"),
    ("dealers", "phase D: dealer profiles and inventories"),
    ("run", "seed -> ads -> specs -> prices -> dealers -> audit -> export"),
    ("audit", "completeness, privacy and cross-validation report"),
    ("export", "write CSV / JSONL / Parquet / snapshot"),
    ("reindex", "rebuild the spec key catalog and wide view"),
    ("stats", "row counts and per-phase status"),
    ("plan", "print the request budget and issue nothing"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python deep_scraper/run_deep.py",
        description="Deep per-car scraper for Bama.ir advertisements.",
    )
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in COMMANDS:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--source-db", default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument("--concurrency", type=int, default=None)
        p.add_argument("--delay-min", type=float, default=None)
        p.add_argument("--delay-max", type=float, default=None)
        p.add_argument("--max-retries", type=int, default=None)
        p.add_argument("--request-timeout", type=float, default=None)
        p.add_argument("--max-runtime", type=float, default=None)
        p.add_argument("--max-requests", type=int, default=None)
        p.add_argument("--phases", default=None, help="comma list, e.g. a,b,c,d")
        p.add_argument("--limit", type=int, default=None, help="cap entities this invocation")
        p.add_argument("--refresh", action="store_true", default=None)
        p.add_argument("--retry-delisted", action="store_true", default=None)
        p.add_argument("--skip-api", action="store_true", default=None)
        p.add_argument("--skip-html", action="store_true", default=None)
        p.add_argument("--download-images", action="store_true", default=None)
        p.add_argument("--dry-run", action="store_true", default=None)
        p.add_argument("--no-dealer-address", action="store_true", default=None)
        p.add_argument("--log-level", default=None)
        if name == "seed":
            p.add_argument(
                "--only-completed",
                action="store_true",
                help="seed only ads the original run completed",
            )
        if name == "audit":
            p.add_argument("--strict", action="store_true", help="fail on any warning")
    return parser


def _config(args: argparse.Namespace) -> DeepConfig:
    return load_config(
        args.config,
        source_db=args.source_db,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        max_runtime=args.max_runtime,
        max_requests=args.max_requests,
        phases=args.phases,
        refresh=args.refresh,
        retry_delisted=args.retry_delisted,
        skip_api=args.skip_api,
        skip_html=args.skip_html,
        download_images=args.download_images,
        dry_run=args.dry_run,
        no_dealer_address=args.no_dealer_address,
        log_level=args.log_level,
    )


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


async def _run_ads(cfg: DeepConfig, storage: DeepStorage, run_id: str, limit: int | None) -> dict:
    async with DeepFetcher(cfg) as fetcher:
        return await run_phase_ads(cfg, storage, fetcher, run_id, limit=limit)


def cmd_plan(cfg: DeepConfig, storage: DeepStorage) -> dict[str, Any]:
    """Report the request budget without issuing anything."""
    pending = sum(
        count
        for status, count in storage.status_breakdown("deep_queue").items()
        if status in ("pending", "fetching", "retryable_error")
    )
    per_ad = (0 if cfg.skip_api else 1) + (0 if cfg.skip_html else 1)
    reviews = storage.count("trim_reviews", "status != 'completed'")
    prices = storage.count("price_keys", "status != 'completed'")
    dealers = storage.count("dealers", "status != 'completed'")
    plan = {
        "ads_pending": pending,
        "requests_per_ad": per_ad,
        "phase_a_requests": pending * per_ad,
        "phase_b_requests": reviews * 3,
        "phase_c_requests": prices + 2,
        "phase_d_requests": dealers * 2,
        "max_requests_ceiling": cfg.max_requests,
    }
    plan["total_estimated_requests"] = (
        plan["phase_a_requests"]
        + plan["phase_b_requests"]
        + plan["phase_c_requests"]
        + plan["phase_d_requests"]
    )
    return plan


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    configure_logging(cfg.log_level)
    cfg.ensure_dirs()

    try:
        source_sha = file_sha256(cfg.source_db) if cfg.source_db.exists() else None
    except OSError:
        source_sha = None

    storage = DeepStorage(cfg.db_path)
    run_id = _new_run_id()
    storage.start_run(
        run_id,
        source_db=str(cfg.source_db),
        source_sha256=source_sha,
        phases=cfg.phases,
        version=DEEP_VERSION,
        config=json.loads(cfg.model_dump_json()),
    )
    command = args.command
    started = time.time()

    try:
        if command == "stats":
            print(json.dumps(storage.stats(), ensure_ascii=False, indent=2))
            return 0

        if command == "plan":
            print(json.dumps(cmd_plan(cfg, storage), ensure_ascii=False, indent=2))
            return 0

        if command == "seed":
            result = seed_from_source(
                storage, cfg.source_db, only_completed=getattr(args, "only_completed", False)
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if command == "ads":
            if cfg.dry_run:
                print(json.dumps(cmd_plan(cfg, storage), ensure_ascii=False, indent=2))
                return 0
            stats = asyncio.run(_run_ads(cfg, storage, run_id, args.limit))
            print_summary("ads", stats)
            storage.finish_run(
                run_id,
                reason=stats.get("stop_reason") or "completed",
                requests_issued=stats.get("requests_issued", 0),
                summary=stats,
            )
            return 1 if stats.get("blocked") else 0

        if command in ("specs", "prices", "dealers", "run", "audit", "export", "reindex"):
            from .commands import dispatch

            return dispatch(command, cfg, storage, run_id, args, started)

        print(f"unknown command: {command}", file=sys.stderr)
        return 2

    except SeedError as exc:
        print(f"seed error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted.progress_persisted", **storage.stats())
        return 130
    finally:
        storage.close()


if __name__ == "__main__":
    sys.exit(main())
