"""Adversarial completeness audit.

The audit exists to *disprove* the claim "we got everything". It never trusts
the fact that discovery stopped; it re-derives counts from the database and
cross-checks them against an independent second discovery pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config
from .discovery import discover_via_api, extract_ad_id, merge_inventories, parse_filters
from .logging_config import get_logger
from .normalization import normalize_price, parse_year
from .storage import Storage

log = get_logger(__name__)


def filter_bounds(search_url: str) -> dict[str, int | None]:
    """Derive numeric lower bounds from the search URL for audit assertions.

    Returns ``None`` for a bound the URL does not specify, so the corresponding
    check is skipped rather than asserted against a made-up threshold.
    """
    params = parse_filters(search_url)["api_params"]
    price_from = normalize_price(params.get("priceFrom"))
    year_from_jalali, _ = parse_year(params.get("yearFrom"))
    return {"price_from": price_from, "year_from_jalali": year_from_jalali}


def audit(
    storage: Storage, cfg: Config, run_id: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compute the full completeness report from persisted data only."""
    conn = storage.conn
    scalar = lambda q, *p: conn.execute(q, p).fetchone()[0]  # noqa: E731

    discovered = scalar("SELECT COUNT(*) FROM discovered_ads")
    distinct_ids = scalar("SELECT COUNT(DISTINCT ad_id) FROM discovered_ads")
    dup_id_groups = scalar(
        "SELECT COUNT(*) FROM (SELECT ad_id FROM discovered_ads GROUP BY ad_id HAVING COUNT(*) > 1)"
    )
    repeat_sightings = scalar("SELECT COALESCE(SUM(times_seen - 1), 0) FROM discovered_ads")

    details = scalar("SELECT COUNT(*) FROM ad_details")
    completed = scalar("SELECT COUNT(*) FROM discovered_ads WHERE status='completed'")
    permanent = scalar("SELECT COUNT(*) FROM discovered_ads WHERE status='permanent_error'")
    retryable = scalar("SELECT COUNT(*) FROM discovered_ads WHERE status='retryable_error'")
    pending = scalar(
        "SELECT COUNT(*) FROM discovered_ads WHERE status IN ('discovered','pending','scraping')"
    )

    with_price = scalar(
        "SELECT COUNT(*) FROM ad_details WHERE json_extract(payload,'$.price_toman') IS NOT NULL"
    )
    with_images = scalar(
        "SELECT COUNT(*) FROM ad_details "
        "WHERE json_array_length(json_extract(payload,'$.image_urls')) > 0"
    )
    with_description = scalar(
        "SELECT COUNT(*) FROM ad_details WHERE json_extract(payload,'$.description') IS NOT NULL"
    )
    warnings = scalar(
        "SELECT COALESCE(SUM(json_array_length(json_extract(payload,'$.parse_warnings'))), 0) "
        "FROM ad_details"
    )
    partial = scalar("SELECT COUNT(*) FROM ad_details WHERE parse_status != 'ok'")

    first_row = conn.execute(
        "SELECT url FROM discovered_ads ORDER BY discovery_cycle, discovery_position LIMIT 1"
    ).fetchone()
    last_row = conn.execute(
        "SELECT url FROM discovered_ads ORDER BY discovery_cycle DESC, "
        "discovery_position DESC LIMIT 1"
    ).fetchone()

    # Integrity checks -- each must hold for the dataset to be internally consistent.
    orphan_details = scalar(
        "SELECT COUNT(*) FROM ad_details d "
        "LEFT JOIN discovered_ads a ON a.url = d.url WHERE a.url IS NULL"
    )
    completed_without_detail = scalar(
        "SELECT COUNT(*) FROM discovered_ads a "
        "LEFT JOIN ad_details d ON d.ad_id = a.ad_id "
        "WHERE a.status='completed' AND d.ad_id IS NULL"
    )
    bad_urls = [
        r["url"]
        for r in conn.execute("SELECT url FROM discovered_ads")
        if extract_ad_id(r["url"]) is None
    ]
    id_url_mismatch = scalar(
        "SELECT COUNT(*) FROM ad_details d JOIN discovered_ads a ON a.url=d.url "
        "WHERE a.ad_id != d.ad_id"
    )
    # Price sanity: the filter is priceFrom=1e9, so any stated price below that
    # would mean the filter was not actually applied. Thresholds come from the
    # URL that was actually scraped -- never hardcoded, so the audit stays valid
    # for any search URL.
    bounds = filter_bounds(cfg.url)
    below_filter = (
        scalar(
            "SELECT COUNT(*) FROM ad_details "
            "WHERE json_extract(payload,'$.price_toman') IS NOT NULL "
            "AND json_extract(payload,'$.price_toman') < ?",
            bounds["price_from"],
        )
        if bounds["price_from"] is not None
        else 0
    )
    year_below_filter = (
        scalar(
            "SELECT COUNT(*) FROM ad_details "
            "WHERE json_extract(payload,'$.year_jalali') IS NOT NULL "
            "AND json_extract(payload,'$.year_jalali') < ?",
            bounds["year_from_jalali"],
        )
        if bounds["year_from_jalali"] is not None
        else 0
    )
    leaked_phone = scalar("SELECT COUNT(*) FROM ad_details WHERE payload LIKE '%\"phone\"%'")

    report: dict[str, Any] = {
        "total_unique_advertisements_discovered": discovered,
        "distinct_advertisement_ids": distinct_ids,
        "duplicate_urls_encountered": repeat_sightings,
        "duplicate_advertisement_ids": dup_id_groups,
        "detail_pages_completed": completed,
        "detail_records_stored": details,
        "detail_pages_failed": permanent + retryable,
        "detail_pages_permanent_error": permanent,
        "detail_pages_retryable_error": retryable,
        "detail_pages_pending": pending,
        "with_price": with_price,
        "without_price": details - with_price,
        "with_images": with_images,
        "without_images": details - with_images,
        "with_description": with_description,
        "without_description": details - with_description,
        "first_discovered_url": first_row["url"] if first_row else None,
        "last_discovered_url": last_row["url"] if last_row else None,
        "parse_warnings": warnings,
        "partial_parses": partial,
        "errors_logged": scalar("SELECT COUNT(*) FROM scrape_errors"),
        "media_rows": scalar("SELECT COUNT(*) FROM ad_media"),
        "integrity": {
            "orphan_detail_rows": orphan_details,
            "completed_without_detail_record": completed_without_detail,
            "unparseable_urls": len(bad_urls),
            "ad_id_url_mismatch": id_url_mismatch,
            "priced_below_price_filter": below_filter,
            "year_below_year_filter": year_below_filter,
            "records_containing_phone_key": leaked_phone,
        },
    }
    if extra:
        report.update(extra)

    checks = report["integrity"]
    report["checks_passed"] = (
        checks["orphan_detail_rows"] == 0
        and checks["completed_without_detail_record"] == 0
        and checks["unparseable_urls"] == 0
        and checks["ad_id_url_mismatch"] == 0
        and checks["priced_below_price_filter"] == 0
        and checks["year_below_year_filter"] == 0
        and checks["records_containing_phone_key"] == 0
        and pending == 0
        and discovered > 0
        and completed == discovered - permanent
    )
    return report


def verification_pass(cfg: Config, storage: Storage, run_id: str) -> dict[str, Any]:
    """Run an independent second discovery pass and diff the inventories.

    New URLs are merged (live classifieds change during a run), and the caller
    is told whether the inventory is stable.
    """
    before = storage.all_discovered_urls()
    log.info("verification.starting", known=len(before))
    result = discover_via_api(cfg, storage, f"{run_id}-verify")
    after = storage.all_discovered_urls()
    diff = merge_inventories(before, after)
    diff.update(
        {
            "second_pass_termination": result["termination_reason"],
            "second_pass_pages": result["pages_fetched"],
            "second_pass_unique_seen": result["unique_urls"],
            "newly_added": len(after - before),
        }
    )
    return diff


def write_summary(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
