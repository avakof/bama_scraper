"""Adversarial audit of the deep dataset.

Written to *disprove* "we captured everything correctly". It re-derives every
count from the database and asserts a set of invariants that must hold; the
privacy gate in particular scans what actually reached disk rather than trusting
that the parsers stripped it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import DeepConfig
from .scrub import assert_no_contact
from .seed import file_sha256
from .storage import DeepStorage

#: Columns that hold free text or serialized payloads and so could carry contact
#: data if a scrub were missed.
_TEXT_COLUMNS = (
    "description",
    "meta_description",
    "meta_title",
    "api_raw_json",
    "html_payload_json",
    "json_ld_json",
    "dealer_name",
)


def audit_dataset(cfg: DeepConfig, storage: DeepStorage) -> dict[str, Any]:
    """Compute the full report. Returns a dict with ``checks_passed``."""
    conn = storage.conn
    scalar = lambda q, *p: conn.execute(q, p).fetchone()[0]  # noqa: E731

    queue = storage.status_breakdown("deep_queue")
    queued = sum(queue.values())
    terminal = sum(queue.get(s, 0) for s in ("completed", "partial", "delisted", "permanent_error"))

    ads = scalar("SELECT COUNT(*) FROM ad_deep")
    delisted = scalar("SELECT COUNT(*) FROM ad_deep WHERE is_delisted = 1")
    live_ads = ads - delisted

    def pct(numerator: int) -> str:
        return f"{100 * numerator / live_ads:.1f}%" if live_ads else "n/a"

    coverage_fields = (
        "fuel_type",
        "province",
        "city",
        "condition_new_used",
        "is_promoted",
        "stats_model_ad_count",
        "price_toman",
        "mileage_km",
        "acceleration_s",
        "engine_volume_l",
        "review_key",
        "price_key",
        "primary_image_url",
        "description",
        "publish_networks",
        "ad_class_id",
    )
    coverage = {
        name: {
            "count": scalar(
                f"SELECT COUNT(*) FROM ad_deep WHERE {name} IS NOT NULL AND is_delisted = 0"
            ),
            "share": pct(
                scalar(f"SELECT COUNT(*) FROM ad_deep WHERE {name} IS NOT NULL AND is_delisted = 0")
            ),
        }
        for name in coverage_fields
    }

    # -- privacy gate: scan what actually reached disk ---------------------
    leaks: list[dict[str, str]] = []
    for column in _TEXT_COLUMNS:
        for row in conn.execute(f"SELECT ad_id, {column} FROM ad_deep WHERE {column} IS NOT NULL"):
            found = assert_no_contact(str(row[1]))
            if found:
                leaks.append({"ad_id": row[0], "column": column, "sample": found[0][:24]})
                if len(leaks) >= 20:
                    break
        if len(leaks) >= 20:
            break
    phone_keys = scalar(
        "SELECT COUNT(*) FROM ad_deep WHERE api_raw_json LIKE '%\"phone\"%' "
        "OR html_payload_json LIKE '%\"phone\"%' OR json_ld_json LIKE '%\"phone\"%'"
    )
    raw_phone_keys = scalar("SELECT COUNT(*) FROM ad_raw_attributes WHERE key LIKE '%phone%'")

    # -- numeric reconciliation -------------------------------------------
    agreements: dict[str, int] = {}
    for row in conn.execute(
        "SELECT numeric_agreement_json FROM ad_deep WHERE numeric_agreement_json IS NOT NULL"
    ):
        for value in json.loads(row[0]).values():
            agreements[value] = agreements.get(value, 0) + 1

    conflicts_by_severity = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT severity, COUNT(*) FROM ad_field_conflicts GROUP BY severity"
        )
    }
    critical = conflicts_by_severity.get("critical", 0)
    critical_rate = critical / live_ads if live_ads else 0.0

    # -- join integrity ----------------------------------------------------
    ads_with_review = scalar(
        "SELECT COUNT(*) FROM ad_deep WHERE review_key IS NOT NULL AND is_delisted = 0"
    )
    ads_joined_to_specs = scalar(
        "SELECT COUNT(DISTINCT a.ad_id) FROM ad_deep a JOIN trim_specs t ON t.review_key = a.review_key"
    )
    orphan_specs = scalar(
        "SELECT COUNT(DISTINCT review_key) FROM trim_specs WHERE review_key NOT IN "
        "(SELECT review_key FROM ad_deep WHERE review_key IS NOT NULL)"
    )
    unknown_slugs = scalar("SELECT COUNT(*) FROM trim_specs WHERE item_slug LIKE 'unk_%'")
    series_out_of_range = scalar(
        "SELECT COUNT(*) FROM price_series WHERE point_count < 1 OR point_count > 400"
    )
    points_without_ts = scalar("SELECT COUNT(*) FROM price_points WHERE date_ts IS NULL")
    duplicate_point_dates = scalar(
        "SELECT COUNT(*) FROM (SELECT series_id, date_jalali FROM price_points "
        "WHERE date_jalali IS NOT NULL GROUP BY series_id, date_jalali HAVING COUNT(*) > 1)"
    )
    dealers_count_mismatch = scalar(
        "SELECT COUNT(*) FROM dealers WHERE ads_listed IS NOT NULL "
        "AND ads_total_count_reported IS NOT NULL AND ads_listed != ads_total_count_reported"
    )
    dealer_ads_new = scalar("SELECT COUNT(*) FROM dealer_ads WHERE in_seed_inventory = 0")

    orphan_ads = scalar(
        "SELECT COUNT(*) FROM ad_deep WHERE ad_id NOT IN (SELECT ad_id FROM deep_queue)"
    )
    completed_without_row = scalar(
        "SELECT COUNT(*) FROM deep_queue q WHERE q.status IN ('completed','partial') "
        "AND q.ad_id NOT IN (SELECT ad_id FROM ad_deep)"
    )

    # -- the source database must be untouched -----------------------------
    source_row = conn.execute(
        "SELECT source_db, source_db_sha256 FROM deep_runs "
        "WHERE source_db_sha256 IS NOT NULL ORDER BY started_at LIMIT 1"
    ).fetchone()
    source_unchanged: bool | None = None
    if source_row and Path(source_row[0]).exists():
        source_unchanged = file_sha256(source_row[0]) == source_row[1]

    integrity = {
        "orphan_ad_rows": orphan_ads,
        "completed_without_ad_row": completed_without_row,
        "queue_not_terminal": queued - terminal,
        "records_with_phone_key": phone_keys,
        "raw_attributes_with_phone_key": raw_phone_keys,
        "contact_leaks": len(leaks),
        "critical_conflicts": critical,
        "orphan_spec_review_keys": orphan_specs,
        "unknown_spec_slugs": unknown_slugs,
        "price_series_out_of_range": series_out_of_range,
        "price_points_without_timestamp": points_without_ts,
        "duplicate_point_dates": duplicate_point_dates,
        "dealer_reported_count_mismatch": dealers_count_mismatch,
        "source_db_unchanged": source_unchanged,
    }

    report: dict[str, Any] = {
        "queue": queue,
        "queue_total": queued,
        "ads_stored": ads,
        "ads_live": live_ads,
        "ads_delisted": delisted,
        "media_rows": scalar("SELECT COUNT(*) FROM ad_media_deep"),
        "raw_attribute_rows": scalar("SELECT COUNT(*) FROM ad_raw_attributes"),
        "conflicts_by_severity": conflicts_by_severity,
        "critical_conflict_rate": round(critical_rate, 6),
        "numeric_agreement": agreements,
        "field_coverage": coverage,
        "trim_reviews": storage.status_breakdown("trim_reviews"),
        "trim_spec_items": scalar("SELECT COUNT(*) FROM trim_specs"),
        "trim_spec_groups": scalar("SELECT COUNT(*) FROM trim_spec_groups"),
        "spec_catalog_items": scalar("SELECT COUNT(*) FROM spec_key_catalog"),
        "ads_with_review_key": ads_with_review,
        "ads_joined_to_specs": ads_joined_to_specs,
        "ads_without_specs": ads_with_review - ads_joined_to_specs,
        "price_keys": storage.status_breakdown("price_keys"),
        "price_series": scalar("SELECT COUNT(*) FROM price_series"),
        "price_points": scalar("SELECT COUNT(*) FROM price_points"),
        "dealers": storage.status_breakdown("dealers"),
        "dealer_ads": scalar("SELECT COUNT(*) FROM dealer_ads"),
        "dealer_ads_not_in_seed": dealer_ads_new,
        "errors": scalar("SELECT COUNT(*) FROM deep_errors"),
        "errors_by_type": {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT error_type, COUNT(*) FROM deep_errors GROUP BY error_type"
            )
        },
        "integrity": integrity,
        "contact_leak_samples": leaks[:5],
    }

    report["checks_passed"] = bool(
        integrity["orphan_ad_rows"] == 0
        and integrity["completed_without_ad_row"] == 0
        and integrity["queue_not_terminal"] == 0
        and integrity["records_with_phone_key"] == 0
        and integrity["raw_attributes_with_phone_key"] == 0
        and integrity["contact_leaks"] == 0
        and critical_rate < cfg.max_critical_conflict_rate
        and integrity["price_series_out_of_range"] == 0
        and integrity["source_db_unchanged"] is not False
        and ads > 0
    )
    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def snapshot(storage: DeepStorage, target: Path) -> None:
    """Consistent copy of the database, safe while WAL is active."""
    target.parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(target)
    with dest:
        storage.conn.backup(dest)
    dest.close()
