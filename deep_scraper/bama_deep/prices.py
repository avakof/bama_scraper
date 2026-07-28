"""Phase C: per-model-trim daily price history.

``/cad/api/price/detail`` returns roughly ten series per model -- market versus
factory price, split by model year and equipment class -- each with ~94 daily
points. It is what lets an individual listing be priced against its model's
market level rather than judged in isolation.

The history dates are **year-less** (``18 فروردین``), so the year has to be
inferred (see :func:`numbers.infer_series_years`) and every point records
``year_inferred`` so the inference is visible rather than implied.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from bama_scraper.normalization import normalize_price, normalize_text, parse_year

from .coerce import as_dict, as_list
from .config import DeepConfig
from .endpoints import price_brand_url, price_detail_url, price_hierarchy_url
from .net import AbortedError, BlockedError, BudgetExhaustedError, DeepFetcher
from .numbers import (
    infer_series_years,
    parse_jalali_day_month,
    parse_signed_toman,
)
from .storage import DeepStorage

#: ``price_provider`` values seen live, mapped to a stable machine label.
PROVIDER_KINDS: dict[str, str] = {
    "قیمت بازار": "market",
    "قیمت کارخانه": "factory",
    "قیمت نمایندگی": "dealership",
}


def provider_kind(provider: str | None) -> str:
    if not provider:
        return "unknown"
    return PROVIDER_KINDS.get(normalize_text(provider) or "", "unknown")


def series_id(price_key: str, detail: dict[str, Any]) -> str:
    """Stable identity for one series across runs."""
    parts = "|".join(
        str(detail.get(k) or "") for k in ("trim", "model_year", "class", "price_provider")
    )
    return hashlib.sha1(f"{price_key}|{parts}".encode()).hexdigest()[:16]


def _anchor_year(detail: dict[str, Any]) -> int:
    """Best available Jalali year to anchor a year-less history on."""
    model_year = detail.get("model_year")
    jalali, _ = parse_year(model_year)
    if jalali:
        return jalali
    try:
        import jdatetime

        return int(jdatetime.date.today().year)
    except Exception:
        return 1405


def parse_price_detail(payload: Any, price_key: str) -> dict[str, Any]:
    """Map the price response onto a key row, series rows and point rows."""
    data = as_dict(as_dict(payload).get("data"))
    if not data:
        return {"key": {}, "series": [], "points": []}

    items = as_list(data.get("items"))
    series_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        detail = as_dict(item.get("detail"))
        history = as_list(item.get("history"))
        sid = series_id(price_key, detail)
        model_year_jalali, _ = parse_year(detail.get("model_year"))
        date_label, date_ts = parse_jalali_day_month(
            normalize_text(str(detail.get("price_date") or "")), year=_anchor_year(detail)
        )

        date_texts = [
            normalize_text(str(p.get("date") or "")) or "" for p in history if isinstance(p, dict)
        ]
        years = infer_series_years(date_texts, anchor_jalali_year=_anchor_year(detail))
        timestamps: list[float] = []
        for position, point in enumerate(history):
            if not isinstance(point, dict):
                continue
            text = normalize_text(str(point.get("date") or "")) or ""
            label, ts = parse_jalali_day_month(text, year=years[position])
            if ts is not None:
                timestamps.append(ts)
            point_rows.append(
                {
                    "series_id": sid,
                    "point_index": position,
                    "date_text": text,
                    "date_jalali": label,
                    "date_ts": ts,
                    "year_inferred": years[position],
                    "price_text": normalize_text(str(point.get("price") or "")),
                    "price_toman": normalize_price(point.get("price")),
                }
            )

        series_rows.append(
            {
                "series_id": sid,
                "price_key": price_key,
                "trim": detail.get("trim"),
                "trim_fa": normalize_text(detail.get("trim_fa")),
                "model_year": str(detail.get("model_year")) if detail.get("model_year") else None,
                "model_year_jalali": model_year_jalali,
                "equipment_class": normalize_text(detail.get("class")),
                "price_provider": normalize_text(detail.get("price_provider")),
                "provider_kind": provider_kind(detail.get("price_provider")),
                "latest_price_text": normalize_text(str(detail.get("price") or "")),
                "latest_price_toman": normalize_price(detail.get("price")),
                "price_diff_text": normalize_text(str(detail.get("price_diff") or "")),
                # A diff may legitimately be negative, which normalize_price drops.
                "price_diff_toman": parse_signed_toman(detail.get("price_diff")),
                "price_date_text": normalize_text(str(detail.get("price_date") or "")),
                "price_date_jalali": date_label,
                "price_date_ts": date_ts,
                "point_count": len(history),
                "first_point_ts": min(timestamps) if timestamps else None,
                "last_point_ts": max(timestamps) if timestamps else None,
                "raw_detail_json": json.dumps(detail, ensure_ascii=False),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    key_row = {
        "brand_fa": normalize_text(data.get("brand_fa")),
        "model_fa": normalize_text(data.get("model_fa")),
        "trim_fa": normalize_text(data.get("trim_fa")),
        "last_update_text": normalize_text(data.get("last_update")),
        "series_count": len(series_rows),
        "point_count": len(point_rows),
        "raw_json": json.dumps(data, ensure_ascii=False)[:2_000_000],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "completed",
    }
    return {"key": key_row, "series": series_rows, "points": point_rows}


async def run_phase_prices(
    cfg: DeepConfig,
    storage: DeepStorage,
    fetcher: DeepFetcher,
    run_id: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fetch price history for every distinct model-trim seen on the ads."""
    stats: dict[str, Any] = {
        "phase": "prices",
        "keys_attempted": 0,
        "keys_completed": 0,
        "keys_failed": 0,
        "series": 0,
        "points": 0,
        "blocked": False,
        "interrupted": False,
    }
    started = time.time()
    stop = {"stop": False, "reason": ""}

    # Reference tables: cheap, and they describe the whole priced universe.
    for url, table, mapper in (
        (price_brand_url(), "price_brands", _map_brands),
        (price_hierarchy_url(), "price_hierarchy", _map_hierarchy),
    ):
        try:
            payload = await fetcher.fetch_json(url)
            rows = mapper(payload)
            if rows:
                storage.upsert_many(table, [next(iter(rows[0]))], rows)
                stats[table] = len(rows)
        except (BlockedError, BudgetExhaustedError, AbortedError):
            break
        except Exception as exc:  # noqa: BLE001
            storage.log_error(
                run_id,
                phase="prices",
                entity_kind="reference",
                entity_key=table,
                url=url,
                stage="fetch",
                error_type="soft",
                message=repr(exc),
            )

    processed = 0
    try:
        while True:
            size = 20 if limit is None else min(20, max(0, limit - processed))
            if size <= 0:
                break
            batch = storage.claim_generic(
                "price_keys", "price_key", size, max_attempts=cfg.max_retries
            )
            if not batch:
                break
            processed += len(batch)
            for row in batch:
                price_key = row["price_key"]
                stats["keys_attempted"] += 1
                try:
                    payload = await fetcher.fetch_json(
                        price_detail_url(row["brand"] or "", row["model"] or "", row["trim"] or "")
                    )
                    parsed = parse_price_detail(payload, price_key)
                    key_row = dict(parsed["key"])
                    key_row["price_key"] = price_key
                    # price_url is NOT NULL, and an upsert still has to satisfy the
                    # INSERT arm's constraints even when the row already exists.
                    key_row["price_url"] = row["price_url"]
                    key_row["brand"] = row["brand"]
                    key_row["model"] = row["model"]
                    key_row["trim"] = row["trim"]
                    storage.upsert("price_keys", ["price_key"], key_row)
                    if parsed["series"]:
                        storage.upsert_many("price_series", ["series_id"], parsed["series"])
                    if parsed["points"]:
                        storage.upsert_many(
                            "price_points", ["series_id", "point_index"], parsed["points"]
                        )
                    stats["series"] += len(parsed["series"])
                    stats["points"] += len(parsed["points"])
                    stats["keys_completed"] += 1
                except AbortedError:
                    break
                except (BlockedError, BudgetExhaustedError) as exc:
                    stats["blocked"] = True
                    stop.update(stop=True, reason=str(exc))
                    fetcher.abort(str(exc))
                    storage.log_error(
                        run_id,
                        phase="prices",
                        entity_kind="price_key",
                        entity_key=price_key,
                        url=row["price_url"],
                        stage="fetch",
                        error_type="blocked",
                        message=str(exc),
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    stats["keys_failed"] += 1
                    storage.set_generic_status(
                        "price_keys", "price_key", price_key, "retryable_error", error=repr(exc)
                    )
                    storage.log_error(
                        run_id,
                        phase="prices",
                        entity_kind="price_key",
                        entity_key=price_key,
                        url=row["price_url"],
                        stage="fetch",
                        error_type="retryable",
                        message=repr(exc),
                    )
            print(
                f"prices keys={stats['keys_completed']}/{processed} series={stats['series']} "
                f"points={stats['points']} requests={fetcher.requests_issued} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            if stop["stop"] or time.time() - started > cfg.max_runtime:
                break
    except KeyboardInterrupt:
        stats["interrupted"] = True

    stats["elapsed_seconds"] = round(time.time() - started, 1)
    stats["stop_reason"] = stop["reason"] or None
    return stats


def _map_brands(payload: Any) -> list[dict[str, Any]]:
    data = as_dict(payload).get("data", payload)
    items = data if isinstance(data, list) else as_list(as_dict(data).get("items"))
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = item.get("value") or item.get("brand") or item.get("slug")
        if not slug:
            continue
        rows.append(
            {
                "brand_slug": str(slug),
                "brand_fa": normalize_text(item.get("display_name") or item.get("title")),
                "model_count": item.get("items_count") or item.get("count"),
                "raw_json": json.dumps(item, ensure_ascii=False),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    return rows


def _map_hierarchy(payload: Any) -> list[dict[str, Any]]:
    data = as_dict(payload).get("data", payload)
    items = data if isinstance(data, list) else as_list(as_dict(data).get("items"))
    rows: list[dict[str, Any]] = []

    def walk(node: Any, brand: str | None = None, model: str | None = None) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, brand, model)
            return
        if not isinstance(node, dict):
            return
        slug = node.get("value") or node.get("slug")
        url = node.get("url") or node.get("price_url")
        children = as_list(node.get("items")) or None
        if url:
            rows.append(
                {
                    "path": str(url),
                    "brand_slug": brand,
                    "model_slug": model or (str(slug) if slug else None),
                    "trim_slug": str(slug) if model else None,
                    "title_fa": normalize_text(node.get("display_name") or node.get("title")),
                    "raw_json": json.dumps(node, ensure_ascii=False),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
        if children:
            walk(children, brand or (str(slug) if slug else None), model)

    walk(items)
    # De-duplicate on the primary key so executemany cannot collide with itself.
    unique: dict[str, dict[str, Any]] = {r["path"]: r for r in rows}
    return list(unique.values())
