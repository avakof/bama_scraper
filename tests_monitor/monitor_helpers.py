"""Shared helpers for the monitoring tests.

Deliberately NOT in ``conftest.py``: the repository has more than one test
directory, each with its own ``conftest.py``, and all of them are importable as
the top-level module ``conftest``. Importing shared helpers from ``conftest``
therefore resolves to whichever directory pytest happened to put on ``sys.path``
first, which breaks as soon as a second suite is added.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bama_monitor.config import MonitorConfig
from bama_monitor.db import Database
from bama_monitor.models import DiscoveredCard, DiscoveryResult, RunContext
from bama_monitor.repository import Repository

BASE_TIME = datetime(2026, 7, 1, 11, 0, tzinfo=UTC)


def integrity_error(db: Database) -> type[Exception]:
    """The DB-API ``IntegrityError`` class for this backend.

    Naming the exception is what makes a constraint test mean anything: a typo in
    the INSERT would also raise, and a bare ``Exception`` assertion would call
    that a passing constraint.
    """
    if db.dialect == "sqlite":
        import sqlite3

        return sqlite3.IntegrityError
    import psycopg

    return psycopg.IntegrityError


def make_card(
    key: str,
    *,
    price: int | None = 1_000_000_000,
    title: str | None = None,
    position: int = 0,
    **kwargs: Any,
) -> DiscoveredCard:
    return DiscoveredCard(
        platform_ad_id=key,
        canonical_url=f"https://bama.ir/car/detail-{key}-car-1400",
        card_title=title if title is not None else f"car {key}",
        card_price_raw=f"{price:,}" if price else None,
        card_price_normalized=price,
        card_year="1400",
        card_mileage_raw="50,000 km",
        card_mileage_normalized=50_000,
        card_location="تهران",
        position=position,
        page_number=0,
        **kwargs,
    )


def make_discovery(
    keys: list[str],
    *,
    healthy: bool = True,
    prices: dict[str, int | None] | None = None,
    titles: dict[str, str] | None = None,
    **overrides: Any,
) -> DiscoveryResult:
    """Build a discovery result; ``healthy=False`` yields an unusable run."""
    prices = prices or {}
    titles = titles or {}
    cards = [
        make_card(
            key,
            price=prices.get(key, 1_000_000_000),
            title=titles.get(key),
            position=index,
        )
        for index, key in enumerate(keys)
    ]
    defaults: dict[str, Any] = {
        "search_url": "https://bama.ir/car?test=1",
        "cards": cards,
        "termination_reason": "api_exhausted_after_2_empty_pages"
        if healthy
        else "max_pages_reached",
        "pages_fetched": 3,
        "pages_failed": 0,
        "initial_page_ok": True,
        "reached_verified_end": healthy,
        "stabilization_completed": healthy,
        "blocked": False,
    }
    defaults.update(overrides)
    return DiscoveryResult(**defaults)


def make_context(
    repo: Repository,
    cfg: MonitorConfig,
    *,
    day: int,
    previous_valid_run_id: int | None = None,
    trigger_type: str = "scheduled",
    production_schedule_name: str = "bama-daily-1300-paris",
) -> RunContext:
    """Create (or resume) the run row for simulated day ``day`` and return its context.

    Defaults to ``scheduled``: these tests simulate the daily production run, and
    slot idempotency applies only to production triggers. A helper that created
    ``manual`` runs would silently stop exercising the resume path.
    """
    slot = BASE_TIME + timedelta(days=day - 1)
    run_id, _ = repo.create_or_get_run(
        search_url=cfg.search_url,
        scheduled_for=slot,
        timezone=cfg.timezone,
        configuration_hash=cfg.configuration_hash(),
        scraper_version="test",
        monitor_version="test",
        previous_valid_run_id=previous_valid_run_id,
        trigger_type=trigger_type,
        production_schedule_name=production_schedule_name,
    )
    return RunContext(
        run_id=run_id,
        scheduled_for=slot,
        started_at=slot,
        search_url=cfg.search_url,
        timezone=cfg.timezone,
        scraper_version="test",
        configuration_hash=cfg.configuration_hash(),
        previous_valid_run_id=previous_valid_run_id,
    )


def run_day(
    db: Database,
    cfg: MonitorConfig,
    *,
    day: int,
    keys: list[str],
    healthy: bool = True,
    prices: dict[str, int | None] | None = None,
    **overrides: Any,
) -> tuple[RunContext, Any, Any]:
    """Execute one simulated day. Returns ``(context, health, comparison)``."""
    from bama_monitor.inventory_comparison import InventoryComparator, finalize_run

    repo = Repository(db)
    previous = repo.last_valid_run(cfg.configuration_hash())
    context = make_context(
        repo, cfg, day=day, previous_valid_run_id=int(previous["id"]) if previous else None
    )
    result = make_discovery(keys, healthy=healthy, prices=prices, **overrides)
    comparator = InventoryComparator(db, cfg)
    persist = comparator.persist_inventory(result, context)
    health = comparator.validate(result, context, persisted=persist.persisted)
    comparison = comparator.compare(result, context, health, persist)
    finalize_run(db, context, health, comparison)
    return context, health, comparison


def status_of(db: Database, key: str) -> str:
    row = db.fetchone("SELECT current_status FROM advertisements WHERE platform_ad_id=?", [key])
    return str(row["current_status"]) if row else "absent"


def misses_of(db: Database, key: str) -> int:
    row = db.fetchone("SELECT consecutive_misses FROM advertisements WHERE platform_ad_id=?", [key])
    return int(row["consecutive_misses"]) if row else -1
