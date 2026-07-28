"""End-to-end multi-day simulation through the real DailyRunner.

Runs the complete pipeline — lock, discover, persist, health-gate, compare,
detail scrape, verify, repost detection, sale scoring, finalize, report — using
scripted discovery and detail responses instead of the network. Nothing about the
workflow is stubbed; only the two scraper adapters are substituted, which is what
their interfaces exist for.

Usage::

    python tools/simulate_days.py --backend sqlite  [--invalid-day 3] [--reports]
    python tools/simulate_days.py --backend postgres --pg-url "postgresql://..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bama_monitor.config import MonitorConfig  # noqa: E402
from bama_monitor.db import connect  # noqa: E402
from bama_monitor.models import (  # noqa: E402
    DetailResult,
    DetailVerdict,
    DiscoveredCard,
    DiscoveryResult,
)
from bama_monitor.reports import generate_report  # noqa: E402
from bama_monitor.repository import Repository  # noqa: E402
from bama_monitor.runner import DailyRunner  # noqa: E402
from bama_monitor.scrapers import StaticDetailScraper, StaticDiscoveryScraper  # noqa: E402

BASE = datetime(2026, 7, 1, 11, 0, tzinfo=UTC)

#: The real reference search, so the filter bounds the monitor re-checks against
#: are the ones actually being simulated.
SEARCH_URL = (
    "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian"
)

#: The scenario specified for this system.
DAYS: dict[int, list[str]] = {
    1: ["A", "B", "C", "D", "G"],
    2: ["A", "B", "D", "E", "G"],
    3: ["A", "D", "E", "F"],
    4: ["A", "B", "D", "E", "F"],
}

EXPECTED = {
    "A": "active",
    "B": "reappeared",
    "C": "likely_removed",
    "D": "active",
    "E": "active",
    "F": "active",
    # G vanished from the search on day 3, but its page is live and its asking
    # price has dropped under the monitored minimum: it left the filter, not the
    # market, and must never reach likely_removed.
    "G": "active_outside_filter",
}

TABLES = (
    "sale_validation_samples",
    "vehicle_entities",
    "advertisement_events",
    "daily_ad_observations",
    "price_changes",
    "detail_verifications",
    "repost_links",
    "advertisement_snapshots",
    "scrape_errors",
    "alerts",
    "advertisements",
    "run_locks",
    "monitoring_runs",
)


def url_for(key: str) -> str:
    return f"https://bama.ir/car/detail-{key.lower()}00000-car-1400"


def card(key: str, position: int, price: int) -> DiscoveredCard:
    return DiscoveredCard(
        platform_ad_id=key,
        canonical_url=url_for(key),
        card_title=f"دنا پلاس {key}",
        card_price_raw=f"{price:,}",
        card_price_normalized=price,
        card_year="1403",
        card_mileage_raw="50,000 km",
        card_mileage_normalized=50_000,
        card_location="تهران",
        position=position,
        page_number=0,
    )


def discovery(keys: list[str], *, healthy: bool, prices: dict[str, int]) -> DiscoveryResult:
    return DiscoveryResult(
        search_url=SEARCH_URL,
        cards=[card(k, i, prices.get(k, 2_000_000_000)) for i, k in enumerate(keys)],
        termination_reason=(
            "api_exhausted_after_2_empty_pages" if healthy else "max_pages_reached"
        ),
        pages_fetched=3,
        initial_page_ok=True,
        reached_verified_end=healthy,
        stabilization_completed=healthy,
    )


def detail_responses(*, strong_evidence: bool) -> dict[str, DetailResult]:
    """Scripted detail pages.

    By default C's page returns a temporary error, i.e. no evidence either way,
    which reproduces the specified scenario exactly: C ends at `likely_removed`.
    With ``strong_evidence`` its page returns 410, which is genuine evidence of
    permanent removal and demonstrates the
    ``likely_removed + strong evidence -> likely_sold`` transition.

    B's page always stays live while it is absent from search -- the case that must
    *lower* sale confidence rather than raise it.
    """
    responses: dict[str, DetailResult] = {}
    for key in "ABDEF":
        responses[url_for(key)] = DetailResult(
            platform_ad_id=key,
            url=url_for(key),
            ok=True,
            verdict=DetailVerdict.STILL_ACTIVE,
            http_status=200,
            fields={
                "title": f"دنا پلاس {key}",
                "brand_fa": "دنا",
                "model_fa": "پلاس",
                "trim_fa": "EF7P",
                "year_text": "1403",
                "price_toman": 2_000_000_000,
                "price_text": "2,000,000,000",
                "mileage_km": 50_000,
                "mileage_text": "50,000 km",
                "city": "تهران",
                "province": "تهران",
                "seller_type": "personal",
                "fuel_type": "بنزینی",
                "body_status": "بدون رنگ",
            },
            media=[{"original_url": f"https://cdn-sth1.bama.ir/img/{key}/1.jpg"}],
            content_hash=f"hash-{key}",
            parser_version="sim",
        )
    # G is alive and for sale at 900,000,000 -- under the 1,000,000,000 filter
    # minimum, so its absence from the results is fully explained.
    responses[url_for("G")] = DetailResult(
        platform_ad_id="G",
        url=url_for("G"),
        ok=True,
        verdict=DetailVerdict.STILL_ACTIVE,
        http_status=200,
        fields={
            "title": "دنا پلاس G",
            "brand_fa": "دنا",
            "model_fa": "پلاس",
            "year_text": "1403",
            "price_toman": 900_000_000,
            "price_text": "900,000,000",
            "mileage_km": 50_000,
            "city": "تهران",
            "seller_type": "personal",
        },
        content_hash="hash-G",
        parser_version="sim",
    )
    responses[url_for("C")] = DetailResult(
        platform_ad_id="C",
        url=url_for("C"),
        ok=False,
        verdict=DetailVerdict.HTTP_410 if strong_evidence else DetailVerdict.TEMPORARY_ERROR,
        http_status=410 if strong_evidence else 503,
        error="simulated C detail outcome",
    )
    return responses


async def simulate(
    cfg: MonitorConfig,
    *,
    invalid_day: int | None,
    make_reports: bool,
    strong_evidence: bool = False,
) -> dict:
    db = connect(cfg.database_url)
    if db.dialect == "postgres":
        for table in TABLES:
            db.execute(f"TRUNCATE {table} CASCADE")
    repo = Repository(db)
    results: list[dict] = []

    try:
        for day, keys in DAYS.items():
            healthy = invalid_day != day
            prices = {"A": 1_800_000_000} if day >= 2 else {}
            plan = StaticDiscoveryScraper([discovery(keys, healthy=healthy, prices=prices)])
            scripted = detail_responses(strong_evidence=strong_evidence)

            runner = DailyRunner(
                cfg,
                db=db,
                discovery_scraper=plan,
                # Bound now, not on call: the loop rebinds `scripted` each day.
                detail_scraper_factory=lambda responses=scripted: StaticDetailScraper(responses),
            )
            outcome = await runner.run_daily(scheduled_for=BASE + timedelta(days=day - 1))
            row = {
                "day": day,
                "seen": keys,
                "run_status": str(outcome.status),
                "comparison_applied": outcome.comparison.applied if outcome.comparison else False,
                **(outcome.comparison.counters() if outcome.comparison else {}),
                "detail_success": outcome.detail_success,
                "detail_failure": outcome.detail_failure,
                "verifications": outcome.verifications,
                "filter_exits": outcome.filter_exits,
                "unexplained_absences": outcome.unexplained_absences,
                "promoted_likely_sold": outcome.promoted_sold,
                "statuses": {
                    str(r["platform_ad_id"]): str(r["current_status"])
                    for r in db.fetchall(
                        "SELECT platform_ad_id, current_status FROM advertisements"
                        " ORDER BY platform_ad_id"
                    )
                },
            }
            results.append(row)
            if make_reports:
                generate_report(db, cfg, run_id=outcome.run_id)

        final = {
            str(r["platform_ad_id"]): {
                "status": str(r["current_status"]),
                "consecutive_misses": int(r["consecutive_misses"]),
                "sale_confidence": float(r["sale_confidence"] or 0),
                "sale_label": str(r["sale_label"]),
                "first_seen_at": str(r["first_seen_at"])[:19],
                "last_seen_at": str(r["last_seen_at"])[:19],
                "first_missing_at": str(r["first_missing_at"])[:19]
                if r["first_missing_at"]
                else None,
            }
            for r in db.fetchall(
                "SELECT platform_ad_id, current_status, consecutive_misses, sale_confidence,"
                " sale_label, first_seen_at, last_seen_at, first_missing_at"
                " FROM advertisements ORDER BY platform_ad_id"
            )
        }
        from bama_monitor.analytics import survival_dataset, time_to_removal

        return {
            "backend": db.dialect,
            "invalid_day": invalid_day,
            "strong_evidence": strong_evidence,
            "days": results,
            "final": final,
            "durations": time_to_removal(db),
            "survival": survival_dataset(db),
            "status_totals": repo.count_by_status(),
        }
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--pg-url", default=None)
    parser.add_argument("--invalid-day", type=int, default=None)
    parser.add_argument("--reports", action="store_true")
    parser.add_argument(
        "--strong-evidence",
        action="store_true",
        help="return HTTP 410 for the removed listing, enabling the likely_sold promotion",
    )
    parser.add_argument("--out", default=None, help="write the JSON result here")
    args = parser.parse_args()

    suffix = args.invalid_day or ("strong" if args.strong_evidence else "all_valid")
    workdir = REPO / "monitor_data" / f"sim_{args.backend}_{suffix}"
    if workdir.exists():
        shutil.rmtree(workdir)
    url = args.pg_url if args.backend == "postgres" else f"sqlite:///{workdir / 'monitor.sqlite'}"
    if args.backend == "postgres" and not url:
        print("--pg-url required for the postgres backend", file=sys.stderr)
        return 2

    cfg = MonitorConfig(
        search_url=SEARCH_URL,
        database_url=url,
        output_dir=workdir,
        reports_dir=REPO / "reports",
        lock_dir=workdir / "locks",
        health={"min_absolute_count": 1},
    )
    cfg.ensure_dirs()

    result = asyncio.run(
        simulate(
            cfg,
            invalid_day=args.invalid_day,
            make_reports=args.reports,
            strong_evidence=args.strong_evidence,
        )
    )

    label = f"invalid day {args.invalid_day}" if args.invalid_day else "all days valid"
    print(f"=== multi-day simulation ({result['backend']}, {label}) ===")
    for row in result["days"]:
        statuses = "  ".join(f"{k}={v}" for k, v in row["statuses"].items())
        print(
            f"Day {row['day']} seen={''.join(row['seen'])}"
            f" health={row['run_status']:8} applied={row['comparison_applied']!s:5}"
            f" new={row.get('new_count', 0)} missing={row.get('missing_count', 0)}"
            f" removed={row.get('removed_count', 0)} reappeared={row.get('reappeared_count', 0)}"
            f" filter_exits={row.get('filter_exits', 0)}"
        )
        print(f"        {statuses}")

    print("\n--- final state ---")
    for key, info in result["final"].items():
        print(
            f"  {key}: {info['status']:15} misses={info['consecutive_misses']}"
            f" confidence={info['sale_confidence']:.2f} ({info['sale_label']})"
        )

    if args.invalid_day is None and not args.strong_evidence:
        print("\n--- expected vs actual ---")
        ok = True
        for key, want in EXPECTED.items():
            got = result["final"][key]["status"]
            match = "OK" if got == want else "MISMATCH"
            ok &= got == want
            print(f"  {key}: expected {want:15} got {got:15} {match}")
        b_ok = result["final"]["B"]["status"] not in ("likely_sold", "likely_removed")
        print(f"\n  B is never sold or removed: {b_ok}")
        ok &= b_ok
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    if args.strong_evidence:
        c = result["final"]["C"]
        print(
            f"\n  strong-evidence variant: C -> {c['status']}"
            f" (confidence {c['sale_confidence']:.2f}, {c['sale_label']})"
        )
        print("  demonstrates: likely_removed + strong evidence -> likely_sold")

    print("\n--- interval-censored durations ---")
    for row in result["durations"]:
        if row["minimum_active_days"] is None:
            continue
        print(
            f"  {row['advertisement_id']}: min={row['minimum_active_days']}d"
            f" est={row['duration_estimate_days'] if 'duration_estimate_days' in row else row['estimated_active_days']}d"
            f" max={row['maximum_active_days']}d"
            f" interval={row['observation_interval_hours']}h"
            f" censored={row['right_censored']}"
        )

    print("\n--- survival export (event_observed = disappearance, NOT a sale) ---")
    for row in result["survival"]:
        print(
            f"  {row['advertisement_id']}: event_observed={row['event_observed']}"
            f" type={row['event_type']:32} confidence={row['sale_confidence']}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
