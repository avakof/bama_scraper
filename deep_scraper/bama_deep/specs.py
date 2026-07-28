"""Phase B: per-model-trim technical specifications -- the "all features" source.

Three public endpoints per model-trim:

* ``carreviewdetail``  -> research/model/generation ids, rating, pros and cons
* ``getspecification`` -> **13 groups / ~110 typed items**: weight, wheelbase,
  length/width/height, fuel-tank capacity, airbag count, ABS/ESC/TCS, sunroof,
  cameras, seats, lighting, multimedia
* ``carreviewseo``     -> a pre-typed numeric subset, useful as a cross-check

These are per *model-trim*, not per advertisement: 59 distinct review keys cover
2,760 of 2,846 ads, so the whole corpus is enriched for ~180 requests. Storage is
long-format (one row per item) because the item set varies by model -- an EV trim
carries battery items an ICE trim does not -- and because the keys are Persian
display strings the site can reword. ``deep reindex`` regenerates a wide view for
analysts.
"""

from __future__ import annotations

import json
import time
from typing import Any

from bama_scraper.normalization import normalize_price, normalize_text

from .coerce import as_dict, as_list, unwrap_data
from .config import DeepConfig
from .endpoints import (
    parse_review_key,
    review_detail_url,
    review_seo_url,
    review_specification_url,
)
from .net import (
    AbortedError,
    BlockedError,
    BudgetExhaustedError,
    DeepFetcher,
    PermanentFetchError,
)
from .numbers import parse_bool_string, parse_decimal, parse_measure, slugify_fa
from .spec_slugs import SPEC_GROUP_SLUGS
from .storage import DeepStorage


def parse_review_detail(payload: Any, review_key: str) -> dict[str, Any]:
    """Map ``carreviewdetail`` onto a ``trim_reviews`` row."""
    data = as_dict(unwrap_data(payload))
    if not data:
        return {}
    stats = as_dict(data.get("stats"))
    parts = parse_review_key(review_key)
    return {
        "brand_slug": data.get("brand_name") or parts["brand_slug"],
        "model_slug": data.get("model_name") or parts["model_slug"],
        "trim_slug": data.get("trim_name") or parts["trim_slug"],
        "extra_segment": parts["extra_segment"],
        "trim_name": data.get("trim_name") or parts["trim_slug"],
        "research_id": data.get("research_id"),
        "model_id": data.get("model_id"),
        "generation_id": data.get("generation_id"),
        "year": data.get("year"),
        "to_year": data.get("to_year"),
        "description": normalize_text(data.get("description")),
        "total_ad_count": data.get("total_ad_count"),
        "comments_count": data.get("comments_count"),
        "price_url": data.get("price_url"),
        "comparison_code": data.get("comparison_code"),
        "rating": stats.get("rating"),
        "final_score": stats.get("final_score"),
        "after_sales_percent": stats.get("after_sales_percent"),
        "body_design_percent": stats.get("body_design_percent"),
        "efficiency_percent": stats.get("efficiency_percent"),
        "capabilities_percent": stats.get("capabilities_percent"),
        "interior_design_percent": stats.get("interior_design_percent"),
        "pros_json": json.dumps(stats.get("pros_list"), ensure_ascii=False)
        if stats.get("pros_list")
        else None,
        "cons_json": json.dumps(stats.get("cons_list"), ensure_ascii=False)
        if stats.get("cons_list")
        else None,
        "images_json": json.dumps(data.get("images"), ensure_ascii=False)
        if data.get("images")
        else None,
        "trims_json": json.dumps(data.get("trims"), ensure_ascii=False)
        if data.get("trims")
        else None,
        "years_json": json.dumps(data.get("years"), ensure_ascii=False)
        if data.get("years")
        else None,
        "raw_detail_json": json.dumps(data, ensure_ascii=False),
    }


def parse_specification(payload: Any, review_key: str) -> tuple[list[dict], list[dict], list[str]]:
    """Map ``getspecification`` onto group rows plus one row per item.

    Returns ``(groups, items, unknown_keys)``. ``unknown_keys`` are Persian keys
    with no reviewed slug; they are still stored under a deterministic
    ``unk_<hash>`` and reported so the mapping can be completed deliberately.
    """
    data = unwrap_data(payload)
    groups: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    unknown: list[str] = []
    if not isinstance(data, list):
        return groups, items, unknown

    for index, group in enumerate(data):
        if not isinstance(group, dict):
            continue
        group_name = normalize_text(group.get("title")) or f"group_{index}"
        group_slug = slugify_fa(group_name, table=SPEC_GROUP_SLUGS)
        if group_slug.startswith("unk_"):
            unknown.append(f"group:{group_name}")
        raw_items = as_list(group.get("items"))
        groups.append(
            {
                "review_key": review_key,
                "group_index": index,
                "group_name": group_name,
                "group_slug": group_slug,
                "item_count": len(raw_items),
            }
        )
        seen_slugs: set[str] = set()
        for position, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            item_key = normalize_text(item.get("key")) or f"item_{position}"
            item_slug = slugify_fa(item_key)
            if item_slug.startswith("unk_"):
                unknown.append(f"{group_name}:{item_key}")
            # The primary key is (review_key, group_slug, item_slug); a duplicate
            # key inside one group would otherwise silently overwrite.
            if item_slug in seen_slugs:
                item_slug = f"{item_slug}_{position}"
            seen_slugs.add(item_slug)

            value_raw = item.get("value")
            value_type = item.get("type")
            value_text = normalize_text(str(value_raw)) if value_raw is not None else None
            value_bool = parse_bool_string(value_raw)
            number, unit = parse_measure(value_text)
            items.append(
                {
                    "review_key": review_key,
                    "group_slug": group_slug,
                    "item_slug": item_slug,
                    "item_key": item_key,
                    "group_name": group_name,
                    "position": position,
                    "value_type": value_type,
                    "value_raw": None if value_raw is None else str(value_raw),
                    "value_text": value_text,
                    "value_bool": None if value_bool is None else int(value_bool),
                    "value_num": number,
                    "value_unit": unit,
                }
            )
    return groups, items, unknown


def parse_review_seo(payload: Any, review_key: str) -> dict[str, Any]:
    """Map ``carreviewseo`` onto the pre-typed numeric row."""
    data = as_dict(unwrap_data(payload))
    if not data:
        return {}
    return {
        "review_key": review_key,
        "trim_name": normalize_text(data.get("vehicle_transmission")),
        "rating": data.get("rating"),
        "acceleration_time_s": parse_decimal(data.get("acceleration_time")),
        "weight_kg": parse_decimal(data.get("weight")),
        "fuel_capacity_l": parse_decimal(data.get("fuel_capacity")),
        "fuel_consumption_l100km": parse_decimal(data.get("fuel_consumption")),
        "drive_wheel_configuration": normalize_text(data.get("drive_wheel_configuration")),
        "number_of_airbags": int(parse_decimal(data.get("number_of_airbags")) or 0) or None,
        "vehicle_transmission": normalize_text(data.get("vehicle_transmission")),
        "speed_kmh": parse_decimal(data.get("speed")),
        "engine_displacement_l": parse_decimal(data.get("engine_displacement")),
        "engine_power_hp": parse_decimal(data.get("engine_power")),
        "torque_nm": parse_decimal(data.get("torque")),
        "price_toman": normalize_price(data.get("price")),
        "date_published": data.get("date_published"),
        "date_modified": data.get("date_modified"),
        "raw_json": json.dumps(data, ensure_ascii=False),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


async def run_phase_specs(
    cfg: DeepConfig,
    storage: DeepStorage,
    fetcher: DeepFetcher,
    run_id: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fetch specifications for every distinct model-trim seen on the ads."""
    stats: dict[str, Any] = {
        "phase": "specs",
        "reviews_attempted": 0,
        "reviews_completed": 0,
        "reviews_failed": 0,
        "spec_items": 0,
        "spec_groups": 0,
        "unknown_keys": [],
        "blocked": False,
        "interrupted": False,
    }
    started = time.time()
    stop = {"stop": False, "reason": ""}

    async def handle(row: Any) -> None:
        review_key = row["review_key"]
        review_url = row["review_url"]
        stats["reviews_attempted"] += 1
        try:
            detail_payload = await fetcher.fetch_json(review_detail_url(review_url))
            record = parse_review_detail(detail_payload, review_key)
            research_id = record.get("research_id")
            trim_name = record.get("trim_slug") or parse_review_key(review_key)["trim_slug"]

            groups: list[dict] = []
            items: list[dict] = []
            if research_id:
                spec_payload = await fetcher.fetch_json(
                    review_specification_url(research_id, trim_name)
                )
                groups, items, unknown = parse_specification(spec_payload, review_key)
                if unknown:
                    stats["unknown_keys"].extend(unknown[:20])

                try:
                    seo_payload = await fetcher.fetch_json(review_seo_url(research_id, trim_name))
                    typed = parse_review_seo(seo_payload, review_key)
                    if typed:
                        storage.upsert("trim_specs_typed", ["review_key"], typed)
                        record["raw_seo_json"] = typed.get("raw_json")
                except (PermanentFetchError, Exception) as exc:  # noqa: BLE001
                    if isinstance(exc, (BlockedError, AbortedError, BudgetExhaustedError)):
                        raise
                    storage.log_error(
                        run_id,
                        phase="specs",
                        entity_kind="review",
                        entity_key=review_key,
                        url=review_seo_url(research_id, trim_name),
                        stage="seo",
                        error_type="soft",
                        message=repr(exc),
                    )

            record["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            record["status"] = "completed"
            record = {k: v for k, v in record.items() if v is not None}
            record["review_key"] = review_key
            record["review_url"] = review_url
            storage.upsert("trim_reviews", ["review_key"], record)
            if groups:
                storage.upsert_many("trim_spec_groups", ["review_key", "group_index"], groups)
            if items:
                storage.upsert_many("trim_specs", ["review_key", "group_slug", "item_slug"], items)
            stats["spec_groups"] += len(groups)
            stats["spec_items"] += len(items)
            stats["reviews_completed"] += 1
        except AbortedError:
            return
        except (BlockedError, BudgetExhaustedError) as exc:
            if not stats["blocked"]:
                stats["blocked"] = True
                stop.update(stop=True, reason=str(exc))
                fetcher.abort(str(exc))
                storage.log_error(
                    run_id,
                    phase="specs",
                    entity_kind="review",
                    entity_key=review_key,
                    url=review_url,
                    stage="fetch",
                    error_type="blocked",
                    message=str(exc),
                )
        except Exception as exc:  # noqa: BLE001
            stats["reviews_failed"] += 1
            storage.set_generic_status(
                "trim_reviews", "review_key", review_key, "retryable_error", error=repr(exc)
            )
            storage.log_error(
                run_id,
                phase="specs",
                entity_kind="review",
                entity_key=review_key,
                url=review_url,
                stage="fetch",
                error_type="retryable",
                message=repr(exc),
            )

    processed = 0
    try:
        while True:
            size = 20 if limit is None else min(20, max(0, limit - processed))
            if size <= 0:
                break
            batch = storage.claim_generic(
                "trim_reviews", "review_key", size, max_attempts=cfg.max_retries
            )
            if not batch:
                break
            processed += len(batch)
            for row in batch:
                await handle(row)
                if stop["stop"]:
                    break
            print(
                f"specs reviews={stats['reviews_completed']}/{processed} "
                f"items={stats['spec_items']} requests={fetcher.requests_issued} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            if stop["stop"] or time.time() - started > cfg.max_runtime:
                break
    except KeyboardInterrupt:
        stats["interrupted"] = True

    stats["elapsed_seconds"] = round(time.time() - started, 1)
    stats["unknown_keys"] = sorted(set(stats["unknown_keys"]))
    stats["stop_reason"] = stop["reason"] or None
    return stats


def rebuild_spec_catalog(storage: DeepStorage) -> dict[str, Any]:
    """Regenerate ``spec_key_catalog`` and the wide analyst view.

    The wide schema is derived from the data rather than hard-coded, so a new
    model introducing new items becomes a ``reindex`` rather than a migration.
    """
    rows = list(
        storage.conn.execute(
            """SELECT item_slug, item_key, group_slug, group_name, MIN(position) position,
                      COUNT(DISTINCT value_raw) distinct_values,
                      COUNT(DISTINCT review_key) coverage,
                      MAX(value_type) value_type
               FROM trim_specs GROUP BY item_slug, group_slug"""
        )
    )
    group_order = {
        r["group_slug"]: r["group_index"]
        for r in storage.conn.execute(
            "SELECT group_slug, MIN(group_index) group_index FROM trim_spec_groups GROUP BY group_slug"
        )
    }
    storage.conn.execute("DELETE FROM spec_key_catalog")
    payload = [
        {
            "item_slug": r["item_slug"],
            "item_key": r["item_key"],
            "group_slug": r["group_slug"],
            "group_name": r["group_name"],
            "group_index": group_order.get(r["group_slug"], 99),
            "position": r["position"],
            "value_type": r["value_type"],
            "distinct_values": r["distinct_values"],
            "coverage_review_keys": r["coverage"],
            "first_seen": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for r in rows
    ]
    if payload:
        storage.upsert_many("spec_key_catalog", ["item_slug"], payload)

    ordered = sorted(payload, key=lambda r: (r["group_index"], r["position"], r["item_slug"]))

    # A slug can legitimately appear in several groups -- "سایر ویژگی‌ها" ("other
    # features") occurs in three. One column cannot represent all three, so a
    # collided slug is qualified with its group name; unique slugs keep the plain
    # name. Without this the view emitted duplicate column names.
    slug_counts: dict[str, int] = {}
    for row in ordered:
        slug_counts[row["item_slug"]] = slug_counts.get(row["item_slug"], 0) + 1

    storage.conn.execute("DROP VIEW IF EXISTS v_trim_specs_wide")
    seen: set[str] = set()
    selects: list[str] = []
    for row in ordered:
        slug, group = row["item_slug"], row["group_slug"]
        column = slug if slug_counts[slug] == 1 else f"{group}__{slug}"
        if column in seen:
            continue
        seen.add(column)
        selects.append(
            f"MAX(CASE WHEN item_slug='{slug}' AND group_slug='{group}' "
            "THEN COALESCE(CAST(value_bool AS TEXT), CAST(value_num AS TEXT), value_text) END) "
            f'AS "{column}"'
        )
    if selects:
        storage.conn.execute(
            "CREATE VIEW v_trim_specs_wide AS SELECT review_key,\n  "
            + ",\n  ".join(selects)
            + "\nFROM trim_specs GROUP BY review_key"
        )
    storage.conn.commit()

    return {
        # The catalog is keyed on item_slug alone, so it holds fewer rows than
        # there are (slug, group) pairs when a slug spans groups.
        "catalog_items": storage.count("spec_key_catalog"),
        "slug_group_pairs": len(payload),
        "wide_columns": len(selects),
        "collided_slugs": sorted(s for s, n in slug_counts.items() if n > 1),
        "unknown_slugs": sorted(
            {r["item_slug"] for r in payload if r["item_slug"].startswith("unk_")}
        ),
    }
