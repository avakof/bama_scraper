"""Scenario builders for the EDA tests.

Deliberately NOT in ``conftest.py``: several test directories in this repository
each have one, all importable as the top-level module ``conftest``, so importing
shared symbols from ``conftest`` resolves to whichever directory pytest put on
``sys.path`` first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bama_monitor.db import Database

PG_URL_ENV = "BAMA_MONITOR_TEST_PG_URL"

T0 = datetime(2026, 6, 1, 11, 0, tzinfo=UTC)

TABLES_IN_FK_ORDER = (
    "sale_validation_samples",
    "advertisement_events",
    "daily_ad_observations",
    "price_changes",
    "detail_verifications",
    "repost_links",
    "advertisement_snapshots",
    "scrape_errors",
    "alerts",
    "advertisements",
    "vehicle_entities",
    "run_locks",
    "monitoring_runs",
)


def _run(
    db: Database,
    run_id_slot: int,
    *,
    status: str,
    finished: bool,
    applied: bool = True,
    discovered: int = 6,
) -> int:
    slot = T0 + timedelta(days=run_id_slot - 1)
    return db.insert_returning_id(
        "INSERT INTO monitoring_runs (search_url, started_at, scheduled_for, timezone,"
        " status, configuration_hash, scraper_version, monitor_version, finished_at,"
        " comparison_applied, discovered_count, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            "https://bama.ir/car?year=1397-2018,&price=1000000000",
            slot,
            slot,
            "Europe/Paris",
            status,
            "cfg-hash-1",
            "test-scraper",
            "test-monitor",
            slot + timedelta(minutes=5) if finished else None,
            applied,
            discovered,
            slot,
        ],
    )


def _ad(db: Database, key: str, **fields: Any) -> int:
    payload: dict[str, Any] = {
        "platform": "bama",
        "platform_ad_id": key,
        "canonical_url": f"https://bama.ir/car/detail-{key}-car",
        "current_status": "active",
        "search_configuration_hash": "cfg-hash-1",
        "created_at": T0,
        "updated_at": T0,
        "first_seen_at": T0,
        "last_seen_at": T0 + timedelta(days=1),
        "sale_confidence": 0.0,
        "sale_label": "unknown",
        "left_truncated": True,
        "eligible_for_duration_ranking": False,
    }
    payload.update(fields)
    columns = ",".join(payload)
    placeholders = ",".join("?" * len(payload))
    return db.insert_returning_id(
        f"INSERT INTO advertisements ({columns}) VALUES ({placeholders})",  # noqa: S608
        list(payload.values()),
    )


def _observe(
    db: Database,
    run_id: int,
    ad_id: int,
    *,
    seen: bool,
    price: int | None = 1_500_000_000,
    at: datetime | None = None,
) -> None:
    db.execute(
        "INSERT INTO daily_ad_observations (run_id, advertisement_id, was_seen,"
        " card_title, card_price_normalized, card_year, card_mileage_normalized,"
        " card_location, observed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        [run_id, ad_id, seen, "دنا پلاس", price, "1402", 40_000, "تهران", at or T0],
    )


def _snapshot(
    db: Database,
    run_id: int,
    ad_id: int,
    *,
    scraped_at: datetime,
    price: int = 1_500_000_000,
    description: str | None = None,
    brand: str = "dena",
) -> int:
    return db.insert_returning_id(
        "INSERT INTO advertisement_snapshots (advertisement_id, run_id, scraped_at,"
        " title, brand, model, year, price_normalized, mileage_normalized, description,"
        " seller_type, city, province, content_hash, parser_version, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ad_id,
            run_id,
            scraped_at,
            "دنا پلاس",
            brand,
            "plus",
            "1402",
            price,
            40_000,
            description,
            "personal",
            "تهران",
            "تهران",
            f"hash-{ad_id}-{scraped_at}",
            "test",
            T0,
        ],
    )


def _event(
    db: Database,
    run_id: int,
    ad_id: int,
    event_type: str,
    previous: str | None = None,
    new: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO advertisement_events (advertisement_id, run_id, event_type,"
        " previous_status, new_status, event_at, created_at) VALUES (?,?,?,?,?,?,?)",
        [ad_id, run_id, event_type, previous, new, T0, T0],
    )


def seed(db: Database) -> dict[str, Any]:
    """Populate every scenario. Returns the ids for tests that need them."""
    run1 = _run(db, 1, status="valid", finished=True)
    run2 = _run(db, 2, status="valid", finished=True)
    run3 = _run(db, 3, status="partial", finished=True, applied=False)  # invalid
    run4 = _run(db, 4, status="valid", finished=False, applied=False)  # unfinished
    run5 = _run(db, 5, status="valid", finished=True)  # latest valid

    ads: dict[str, int] = {}

    # A: plain active listing, left-truncated (no publication time).
    ads["A"] = _ad(db, "A")
    # B: eligible for duration ranking — published shortly before first sighting.
    ads["B"] = _ad(
        db,
        "B",
        published_at=T0 - timedelta(hours=10),
        published_at_source="detail_relative",
        published_at_reliable=True,
        left_truncated=False,
        eligible_for_duration_ranking=True,
        entry_delay_seconds=36000.0,
    )
    # C: disappeared — the only observed event.
    ads["C"] = _ad(
        db,
        "C",
        current_status="likely_removed",
        first_missing_at=T0 + timedelta(days=2),
        last_seen_at=T0 + timedelta(days=1),
        consecutive_misses=2,
        sale_confidence=0.5,
        sale_label="possibly_sold",
        published_at=T0 - timedelta(hours=5),
        published_at_source="detail_absolute",
        published_at_reliable=True,
        left_truncated=False,
        eligible_for_duration_ranking=True,
        entry_delay_seconds=18000.0,
    )
    # D: reappeared after an absence — must never be counted as gone.
    ads["D"] = _ad(
        db,
        "D",
        current_status="reappeared",
        first_missing_at=T0 + timedelta(days=1),
        reappeared_at=T0 + timedelta(days=2),
        last_seen_at=T0 + timedelta(days=2),
    )
    # E: left the search filter but is still for sale.
    ads["E"] = _ad(
        db,
        "E",
        current_status="active_outside_filter",
        filter_exit_reason="price_below_filter",
        filter_exit_at=T0 + timedelta(days=1),
        detail_availability="outside_filter",
        first_missing_at=T0 + timedelta(days=1),
    )
    # F/G: one physical vehicle listed twice (repost chain).
    ads["F"] = _ad(
        db,
        "F",
        current_status="reposted",
        vehicle_entity_id="veh-1",
        first_missing_at=T0 + timedelta(days=1),
        last_seen_at=T0,
    )
    ads["G"] = _ad(
        db,
        "G",
        vehicle_entity_id="veh-1",
        repost_parent_ad_id=ads["F"],
        first_seen_at=T0 + timedelta(days=1),
        last_seen_at=T0 + timedelta(days=2),
    )
    # H: publication time known but too coarse to use.
    ads["H"] = _ad(
        db,
        "H",
        published_at=T0 - timedelta(minutes=5),
        published_at_source="detail_coarse",
        published_at_reliable=False,
        left_truncated=True,
        eligible_for_duration_ranking=False,
    )

    for key, ad_id in ads.items():
        for run_id in (run1, run2, run5):
            _observe(db, run_id, ad_id, seen=key not in ("C",))
        # The invalid run "sees" nothing: its absences must not be trusted.
        _observe(db, run3, ad_id, seen=False)

    # Snapshots.
    #  - two for A, so latest-selection must be deterministic;
    #  - one from a LATER run, which must never be joined;
    #  - none for H, so a missing snapshot is exercised.
    _snapshot(db, run1, ads["A"], scraped_at=T0, price=1_400_000_000)
    _snapshot(db, run2, ads["A"], scraped_at=T0 + timedelta(days=1), price=1_500_000_000)
    _snapshot(db, run1, ads["B"], scraped_at=T0, price=1_800_000_000, brand="peugeot")
    # Planted contact number: the privacy gate must catch it if it ever leaks.
    _snapshot(db, run1, ads["C"], scraped_at=T0, description="تماس 09123456789 فوری")
    # Dated AFTER every run's reference instant: the no-future-snapshot rule must
    # exclude it from an analysis of run 5.
    _snapshot(db, run5, ads["D"], scraped_at=T0 + timedelta(days=40))

    _event(db, run1, ads["A"], "discovered", None, "new")
    _event(db, run2, ads["A"], "seen", "new", "active")
    _event(db, run2, ads["C"], "missing_first_time", "active", "missing_once")
    _event(db, run5, ads["C"], "removal_confirmed", "missing_once", "likely_removed")
    _event(db, run5, ads["D"], "reappeared", "missing_once", "reappeared")
    _event(db, run5, ads["E"], "filter_exit", "missing_once", "active_outside_filter")
    _event(db, run5, ads["G"], "possible_repost", None, None)

    db.execute(
        "INSERT INTO repost_links (parent_ad_id, child_ad_id, run_id, score,"
        " vehicle_entity_id, created_at) VALUES (?,?,?,?,?,?)",
        [ads["F"], ads["G"], run5, 0.88, "veh-1", T0],
    )
    return {
        "runs": {"valid_first": run1, "invalid": run3, "unfinished": run4, "latest_valid": run5},
        "ads": ads,
    }
