"""Phase D: dealer profiles and inventories.

Two public endpoints per dealer:

* ``/cad/api/Corporation/<id>``     -> type, address, coordinates, active flag
* ``/cad/api/Corporation/ads/<id>`` -> the dealer's entire live inventory in one
  response, which doubles as a coverage cross-check on the original discovery

``metadata.total_count`` on the inventory endpoint is **known to be wrong** (it
reports 0 while returning a full array), so ``ads_listed`` counts the array and
the reported value is retained only as evidence.

``/cad/api/Corporation/phone/<id>`` is deliberately never called -- it is
authenticated and exists to reveal personal contact data.
"""

from __future__ import annotations

import json
import time
from typing import Any

from bama_scraper.normalization import normalize_mileage, normalize_price, normalize_text

from .coerce import as_dict, as_list
from .config import DeepConfig
from .endpoints import dealer_ads_url, dealer_profile_url
from .net import AbortedError, BlockedError, BudgetExhaustedError, DeepFetcher
from .numbers import descale_bounded
from .storage import DeepStorage


def parse_dealer_profile(payload: Any) -> dict[str, Any]:
    data = as_dict(as_dict(payload).get("data", payload))
    if not data:
        return {}
    score_raw = data.get("score")
    score, heuristic = (
        (float(score_raw), False)
        if isinstance(score_raw, float)
        else descale_bounded(score_raw, lo=0.0, hi=5.0)
    )
    return {
        "title": normalize_text(data.get("title")),
        "dealer_type": normalize_text(data.get("type")),
        "package_type": data.get("package_type"),
        "banner_url": data.get("banner"),
        "logo_url": data.get("logo"),
        "address": normalize_text(data.get("address")),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "is_active": int(bool(data.get("is_active")))
        if data.get("is_active") is not None
        else None,
        "vehicle_type": data.get("vehicle_type"),
        "score": score,
        "score_raw": int(score_raw) if isinstance(score_raw, (int, float)) else None,
        "raw_profile_json": json.dumps(data, ensure_ascii=False),
    }


def parse_dealer_ads(payload: Any, dealer_id: int, known_codes: set[str]) -> dict[str, Any]:
    """Extract the dealer's live inventory and flag anything not already known."""
    entries = as_dict(payload).get("data", payload)
    metadata = as_dict(payload).get("metadata")
    rows: list[dict[str, Any]] = []
    for entry in as_list(entries):
        if not isinstance(entry, dict):
            continue
        detail = as_dict(entry.get("detail"))
        price = as_dict(entry.get("price"))
        code = detail.get("code")
        if not code:
            continue
        rows.append(
            {
                "dealer_id": dealer_id,
                "ad_code": code,
                "url": detail.get("url"),
                "title": normalize_text(detail.get("title")),
                "subtitle": normalize_text(detail.get("subtitle")),
                "price_text": normalize_text(str(price.get("price") or "")),
                "price_toman": normalize_price(price.get("price"))
                if price.get("type") == "lumpsum"
                else None,
                "year_text": normalize_text(str(detail.get("year") or "")),
                "mileage_text": normalize_text(detail.get("mileage")),
                "mileage_km": normalize_mileage(detail.get("mileage")),
                "in_seed_inventory": int(code in known_codes),
                "raw_json": json.dumps(entry, ensure_ascii=False),
                "seen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    reported = as_dict(metadata).get("total_count")
    return {
        "rows": rows,
        # The array length is authoritative; the reported count is kept as evidence.
        "ads_listed": len(rows),
        "ads_total_count_reported": reported,
    }


async def run_phase_dealers(
    cfg: DeepConfig,
    storage: DeepStorage,
    fetcher: DeepFetcher,
    run_id: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "phase": "dealers",
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "dealer_ads": 0,
        "ads_not_in_seed": 0,
        "blocked": False,
        "interrupted": False,
    }
    started = time.time()
    stop = {"stop": False, "reason": ""}
    known = {r[0] for r in storage.conn.execute("SELECT ad_id FROM deep_queue")}

    processed = 0
    try:
        while True:
            size = 20 if limit is None else min(20, max(0, limit - processed))
            if size <= 0:
                break
            batch = storage.claim_generic(
                "dealers", "dealer_id", size, max_attempts=cfg.max_retries
            )
            if not batch:
                break
            processed += len(batch)
            for row in batch:
                dealer_id = row["dealer_id"]
                stats["attempted"] += 1
                record: dict[str, Any] = {"dealer_id": dealer_id}
                try:
                    profile = await fetcher.fetch_json(dealer_profile_url(dealer_id))
                    record.update(parse_dealer_profile(profile))

                    inventory = await fetcher.fetch_json(dealer_ads_url(dealer_id))
                    parsed = parse_dealer_ads(inventory, dealer_id, known)
                    record["ads_listed"] = parsed["ads_listed"]
                    record["ads_total_count_reported"] = parsed["ads_total_count_reported"]
                    record["raw_ads_json"] = json.dumps(
                        {"count": parsed["ads_listed"]}, ensure_ascii=False
                    )
                    if parsed["rows"]:
                        storage.upsert_many("dealer_ads", ["dealer_id", "ad_code"], parsed["rows"])
                        stats["dealer_ads"] += len(parsed["rows"])
                        stats["ads_not_in_seed"] += sum(
                            1 for r in parsed["rows"] if not r["in_seed_inventory"]
                        )

                    record["status"] = "completed"
                    record["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    storage.upsert(
                        "dealers", ["dealer_id"], {k: v for k, v in record.items() if v is not None}
                    )
                    stats["completed"] += 1
                except AbortedError:
                    break
                except (BlockedError, BudgetExhaustedError) as exc:
                    stats["blocked"] = True
                    stop.update(stop=True, reason=str(exc))
                    fetcher.abort(str(exc))
                    storage.log_error(
                        run_id,
                        phase="dealers",
                        entity_kind="dealer",
                        entity_key=str(dealer_id),
                        url=dealer_profile_url(dealer_id),
                        stage="fetch",
                        error_type="blocked",
                        message=str(exc),
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    stats["failed"] += 1
                    storage.set_generic_status(
                        "dealers", "dealer_id", dealer_id, "retryable_error", error=repr(exc)
                    )
                    storage.log_error(
                        run_id,
                        phase="dealers",
                        entity_kind="dealer",
                        entity_key=str(dealer_id),
                        url=dealer_profile_url(dealer_id),
                        stage="fetch",
                        error_type="retryable",
                        message=repr(exc),
                    )
            print(
                f"dealers completed={stats['completed']}/{processed} "
                f"inventory_ads={stats['dealer_ads']} new_to_us={stats['ads_not_in_seed']} "
                f"requests={fetcher.requests_issued} elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            if stop["stop"] or time.time() - started > cfg.max_runtime:
                break
    except KeyboardInterrupt:
        stats["interrupted"] = True

    stats["elapsed_seconds"] = round(time.time() - started, 1)
    stats["stop_reason"] = stop["reason"] or None
    return stats
