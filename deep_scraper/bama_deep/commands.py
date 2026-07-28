"""Dispatch for the phases that build on phase A."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .config import DeepConfig
from .net import DeepFetcher
from .phases import print_summary
from .storage import DeepStorage


async def _with_fetcher(cfg: DeepConfig, coro_factory: Any) -> Any:
    async with DeepFetcher(cfg) as fetcher:
        return await coro_factory(fetcher)


def dispatch(
    command: str,
    cfg: DeepConfig,
    storage: DeepStorage,
    run_id: str,
    args: argparse.Namespace,
    started: float,
) -> int:
    limit = getattr(args, "limit", None)

    if command == "specs":
        from .specs import run_phase_specs

        storage.derive_review_keys()
        stats = asyncio.run(
            _with_fetcher(cfg, lambda f: run_phase_specs(cfg, storage, f, run_id, limit=limit))
        )
        print_summary("specs", stats)
        return 1 if stats.get("blocked") else 0

    if command == "prices":
        from .prices import run_phase_prices

        storage.derive_price_keys()
        stats = asyncio.run(
            _with_fetcher(cfg, lambda f: run_phase_prices(cfg, storage, f, run_id, limit=limit))
        )
        print_summary("prices", stats)
        return 1 if stats.get("blocked") else 0

    if command == "dealers":
        from .dealers import run_phase_dealers

        storage.derive_dealer_ids()
        stats = asyncio.run(
            _with_fetcher(cfg, lambda f: run_phase_dealers(cfg, storage, f, run_id, limit=limit))
        )
        print_summary("dealers", stats)
        return 1 if stats.get("blocked") else 0

    if command == "reindex":
        from .specs import rebuild_spec_catalog

        result = rebuild_spec_catalog(storage)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if command == "audit":
        from .audit import audit_dataset, write_report

        report = audit_dataset(cfg, storage)
        write_report(report, cfg.output_dir / "deep_audit.json")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["checks_passed"] else 1

    if command == "export":
        from .export import export_all

        manifest = export_all(cfg, storage)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        return 0

    if command == "run":
        return _run_all(cfg, storage, run_id, args, started)

    return 2


def _run_all(
    cfg: DeepConfig,
    storage: DeepStorage,
    run_id: str,
    args: argparse.Namespace,
    started: float,
) -> int:
    """Full pipeline: seed, then every enabled phase, then audit and export."""
    from .audit import audit_dataset, write_report
    from .dealers import run_phase_dealers
    from .export import export_all
    from .phases import run_phase_ads
    from .prices import run_phase_prices
    from .seed import seed_from_source
    from .specs import rebuild_spec_catalog, run_phase_specs

    enabled = cfg.enabled_phases()
    summary: dict[str, Any] = {"run_id": run_id, "phases": sorted(enabled)}
    summary["seed"] = seed_from_source(storage, cfg.source_db)

    async def sequence() -> dict[str, Any]:
        out: dict[str, Any] = {}
        async with DeepFetcher(cfg) as fetcher:
            if "a" in enabled:
                out["ads"] = await run_phase_ads(
                    cfg, storage, fetcher, run_id, limit=getattr(args, "limit", None)
                )
            if "b" in enabled:
                storage.derive_review_keys()
                out["specs"] = await run_phase_specs(cfg, storage, fetcher, run_id)
            if "c" in enabled:
                storage.derive_price_keys()
                out["prices"] = await run_phase_prices(cfg, storage, fetcher, run_id)
            if "d" in enabled:
                storage.derive_dealer_ids()
                out["dealers"] = await run_phase_dealers(cfg, storage, fetcher, run_id)
            out["requests_issued"] = fetcher.requests_issued
        return out

    summary.update(asyncio.run(sequence()))
    summary["reindex"] = rebuild_spec_catalog(storage)
    report = audit_dataset(cfg, storage)
    summary["audit"] = report
    summary["export"] = export_all(cfg, storage)
    write_report(summary, cfg.output_dir / "deep_run_summary.json")
    storage.finish_run(
        run_id,
        reason="completed",
        requests_issued=int(summary.get("requests_issued", 0)),
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if report["checks_passed"] else 1
