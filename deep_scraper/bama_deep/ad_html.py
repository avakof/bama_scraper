"""Parse the advertisement detail *page*.

The page adds three things the JSON API cannot supply:

* **typed numerics** -- ``vehicle.mileage.value``, ``vehicle.year.value``,
  ``price.fixed.value`` and the ``specs.details.*`` integer/text pairs that let
  :func:`numbers.reconcile_dotless` recover an exact decimal;
* **JSON-LD ``offers.itemCondition``** -- the only explicit new/used statement
  anywhere on the site (elsewhere it must be inferred from ``mileage == 0``);
* a handful of HTML-only fields: ``media.badge`` (the promoted flag),
  ``metadata.slogan``, ``stats.*`` and the breadcrumb slug trail.

Devalue resolution, JSON-LD extraction and private-key stripping are reused from
``bama_scraper.detail_parser`` rather than reimplemented.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bama_scraper.detail_parser import extract_json_ld, extract_nuxt_payload
from bama_scraper.normalization import (
    is_zero_km,
    normalize_mileage,
    normalize_price,
    normalize_text,
    parse_published_time,
    parse_year,
)

from .endpoints import dealer_id_from_url, normalize_price_key, normalize_review_key
from .numbers import descale_bounded, parse_decimal, reconcile_dotless
from .scrub import scrub_freetext, scrub_payload

#: ``specs.details`` key -> (text column, value column, raw column).
_NUMERIC_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("volume", "engine_volume_text", "engine_volume_l", "engine_volume_raw"),
    ("power", "power_text", "power_hp", "power_raw"),
    ("torque", "torque_text", "torque_nm", "torque_raw"),
    ("acceleration", "acceleration_text", "acceleration_s", "acceleration_raw"),
    (
        "fuelConsumption",
        "fuel_consumption_text",
        "fuel_consumption_l100km",
        "fuel_consumption_raw",
    ),
)


def _sub(node: Any, *path: str) -> dict[str, Any]:
    cur: Any = node
    for key in path:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key)
    return cur if isinstance(cur, dict) else {}


def _wrapped(node: Any, *path: str) -> tuple[Any, str | None]:
    """Read a Bama ``{value, text}`` wrapper, returning ``(value, text)``."""
    inner = _sub(node, *path)
    if not inner:
        return None, None
    return inner.get("value"), normalize_text(inner.get("text"))


def condition_from_json_ld(blocks: list[dict[str, Any]]) -> str | None:
    """Map schema.org ``itemCondition`` to ``new`` / ``used``.

    This is the only place Bama states condition explicitly.
    """
    for block in blocks:
        offers = block.get("offers")
        if not isinstance(offers, dict):
            continue
        condition = offers.get("itemCondition")
        if not isinstance(condition, str):
            continue
        tail = condition.rsplit("/", 1)[-1].lower()
        if "newcondition" in tail:
            return "new"
        if "usedcondition" in tail:
            return "used"
        if "refurbished" in tail:
            return "refurbished"
    return None


def parse_ad_html(html: str, *, now_ts: float) -> dict[str, Any]:
    """Extract a flat field map plus media, raw attributes and warnings."""
    warnings: list[str] = []
    digest = hashlib.sha256(html.encode("utf-8", "replace")).hexdigest()
    # JSON-LD Product.description embeds the seller's own description, so it is a
    # third place a typed-in phone number can hide (verified on live ad cuaxpwpu).
    json_ld_raw = extract_json_ld(html)
    json_ld, ld_redactions = scrub_payload(json_ld_raw)
    payload = extract_nuxt_payload(html)

    if payload is None:
        warnings.append("__NUXT_DATA__ payload missing")
        condition = condition_from_json_ld(json_ld)
        fields: dict[str, Any] = {"html_sha256": digest}
        if condition:
            fields["condition_new_used"] = condition
            fields["condition_source"] = "json_ld"
        return {
            "fields": fields,
            "media": [],
            "raw": [],
            "json_ld": json_ld,
            "payload": None,
            "warnings": warnings,
            "sha256": digest,
            "ok": False,
            "agreements": {},
        }

    safe, payload_redactions = scrub_payload(payload)
    content = _sub(safe, "content")
    vehicle = _sub(safe, "vehicle")
    specs = _sub(safe, "specs", "details")
    spec_urls = _sub(safe, "specs", "urls")
    media_node = _sub(safe, "media")
    seller = _sub(safe, "seller")
    meta = _sub(safe, "metadata")
    stats = _sub(safe, "stats")
    price = _sub(safe, "price")

    year_value, year_text = _wrapped(vehicle, "year")
    year_jalali, year_gregorian = parse_year(year_value or year_text)

    mileage_value, mileage_text = _wrapped(vehicle, "mileage")
    mileage_km = (
        int(mileage_value)
        if isinstance(mileage_value, (int, float))
        else normalize_mileage(mileage_text)
    )

    description, redactions = scrub_freetext(normalize_text(content.get("description")))
    location_value = _sub(content, "location", "value")
    published_text = normalize_text(_sub(content, "publishedDate").get("text"))

    fields = {
        "code": safe.get("code"),
        "title": normalize_text(content.get("title")),
        "vehicle_category": vehicle.get("type"),
        "brand": _sub(vehicle, "brand").get("value"),
        "brand_fa": normalize_text(_sub(vehicle, "brand").get("text")),
        "model": _sub(vehicle, "model").get("value"),
        "model_fa": normalize_text(_sub(vehicle, "model").get("text")),
        "trim": _sub(vehicle, "trim").get("value"),
        "trim_fa": normalize_text(_sub(vehicle, "trim").get("text")),
        "year_text": year_text,
        "year_jalali": year_jalali,
        "year_gregorian": year_gregorian,
        "body_type": _sub(vehicle, "bodyType").get("value") or _sub(safe, "body").get("value"),
        "body_display_name": normalize_text(_sub(safe, "body").get("display_name")),
        "body_type_fa": normalize_text(specs.get("bodyType", {}).get("value"))
        if isinstance(specs.get("bodyType"), dict)
        else None,
        # condition
        "mileage_text": mileage_text,
        "mileage_km": mileage_km,
        "is_zero_km": is_zero_km(mileage_text, mileage_km),
        "body_status": normalize_text(_sub(vehicle, "bodyStatus").get("value")),
        # powertrain
        "transmission": normalize_text(_sub(vehicle, "transmission").get("value")),
        "engine": normalize_text(specs.get("engine", {}).get("value"))
        if isinstance(specs.get("engine"), dict)
        else None,
        "drivetrain": normalize_text(specs.get("driveShaft", {}).get("value"))
        if isinstance(specs.get("driveShaft"), dict)
        else None,
        "body_color": normalize_text(_sub(vehicle, "color", "body").get("value")),
        "inside_color": normalize_text(_sub(vehicle, "color", "interior").get("value")),
        # place and time
        "location_text": normalize_text(_sub(content, "location").get("text")),
        "province": normalize_text(location_value.get("province")),
        "city": normalize_text(location_value.get("city")),
        "region": normalize_text(location_value.get("region")),
        "published_text": published_text,
        "published_ts": parse_published_time(published_text, now_ts=now_ts),
        # ad surface
        "description": description,
        "description_redactions": redactions + payload_redactions + ld_redactions,
        "badge": media_node.get("badge") if isinstance(media_node.get("badge"), bool) else None,
        "is_promoted": media_node.get("badge")
        if isinstance(media_node.get("badge"), bool)
        else None,
        "stats_model_ad_count": stats.get("model"),
        "stats_trim_ad_count": stats.get("trim"),
        "meta_slogan": normalize_text(meta.get("slogan")),
        "meta_description": scrub_freetext(normalize_text(meta.get("description")))[0],
        "meta_keywords": normalize_text(meta.get("keywords")),
        "meta_canonical": meta.get("canonicalURL"),
        "canonical_url": meta.get("canonicalURL"),
        "meta_noindex": meta.get("noIndex") if isinstance(meta.get("noIndex"), bool) else None,
        "breadcrumb_json": json.dumps(safe.get("breadcrumb"), ensure_ascii=False)
        if safe.get("breadcrumb")
        else None,
        "life_styles_json": json.dumps(safe.get("life_styles"), ensure_ascii=False)
        if safe.get("life_styles")
        else None,
        "review_url": spec_urls.get("review"),
        "review_key": normalize_review_key(spec_urls.get("review")),
        "price_url": spec_urls.get("price"),
        "html_sha256": digest,
    }

    price_key, _ = normalize_price_key(spec_urls.get("price"))
    fields["price_key"] = price_key

    # -- numerics: text authoritative, integer as checksum -----------------
    agreements: dict[str, str] = {}
    for key, text_col, value_col, raw_col in _NUMERIC_SPECS:
        node = specs.get(key)
        if not isinstance(node, dict):
            continue
        raw_int = node.get("value")
        text = normalize_text(node.get("text"))
        value, agreement = reconcile_dotless(raw_int, text)
        if text:
            fields[text_col] = text
        if value is not None:
            fields[value_col] = value
        if isinstance(raw_int, (int, float)):
            fields[raw_col] = int(raw_int)
        agreements[value_col] = agreement
        if agreement == "mismatch":
            warnings.append(f"{key}: int {raw_int!r} disagrees with text {text!r}")

    for key, text_col, value_col in (
        ("batteryCapacity", "battery_capacity_text", "battery_capacity_kwh"),
        ("allElectricRange", "all_electric_range_text", "all_electric_range_km"),
    ):
        node = specs.get(key)
        if isinstance(node, dict) and node:
            text = normalize_text(node.get("text"))
            if text:
                fields[text_col] = text
            number = parse_decimal(text) if text else None
            if number is None and isinstance(node.get("value"), (int, float)):
                number = float(node["value"])
            if number is not None:
                fields[value_col] = number

    # -- price -------------------------------------------------------------
    price_type = price.get("type")
    fields["price_type"] = price_type
    fields["is_negotiable"] = price_type == "negotiable"
    fields["price_hidden"] = price_type in ("hidden", "notShown")
    fixed = price.get("fixed")
    if isinstance(fixed, dict) and fixed.get("value") is not None:
        fields["price_toman"] = normalize_price(fixed.get("value"))
        fields["price_text"] = normalize_text(fixed.get("text"))
    installment = price.get("installment")
    if isinstance(installment, dict) and installment:
        fields["is_installment"] = True
        for node_key, column in (
            ("total", "installment_total_toman"),
            ("prepayment", "down_payment_toman"),
            ("secondPrepayment", "prepayment_secondary_toman"),
            ("payment", "installment_amount_toman"),
        ):
            node = installment.get(node_key)
            if isinstance(node, dict):
                fields[column] = normalize_price(node.get("value"))
        for node_key, column in (
            ("months", "installment_months"),
            ("installments", "installments_count"),
            ("deliveryDays", "delivery_days"),
        ):
            node = installment.get(node_key)
            if isinstance(node, dict) and isinstance(node.get("value"), (int, float)):
                fields[column] = int(node["value"]) or None
    else:
        fields["is_installment"] = price_type == "installment"

    presale = price.get("presale")
    if isinstance(presale, dict) and presale:
        fields["is_pre_sale"] = True
        fields["pre_sale_delivery"] = normalize_text(presale.get("text") or presale.get("value"))

    # -- seller ------------------------------------------------------------
    if seller:
        fields["seller_type"] = seller.get("type")
        if isinstance(seller.get("authenticated"), bool):
            fields["authenticated"] = seller["authenticated"]
        if seller.get("id") is not None or seller.get("url"):
            score_raw = _sub(seller, "score").get("value")
            score, heuristic = descale_bounded(score_raw, lo=0.0, hi=5.0)
            activity_value = _sub(seller, "activityDuration", "value")
            fields.update(
                {
                    "dealer_id": seller.get("id") or dealer_id_from_url(seller.get("url")),
                    "dealer_name": normalize_text(seller.get("name")),
                    "dealer_link": seller.get("url"),
                    "dealer_address": normalize_text(seller.get("address")),
                    "dealer_ad_count": _sub(seller, "adsCount").get("value"),
                    "dealer_score": score,
                    "dealer_score_raw": int(score_raw)
                    if isinstance(score_raw, (int, float))
                    else None,
                    "dealer_score_heuristic": heuristic,
                    "dealer_activity_years": activity_value.get("year"),
                    "dealer_activity_months": activity_value.get("month"),
                    "dealer_activity_text": normalize_text(
                        _sub(seller, "activityDuration").get("text")
                    ),
                }
            )

    # -- media -------------------------------------------------------------
    media: list[dict[str, Any]] = []
    images = media_node.get("images")
    if isinstance(images, list):
        for i, img in enumerate(images):
            if not isinstance(img, dict):
                continue
            media.append(
                {
                    "kind": "image",
                    "position": i,
                    "original_url": img.get("original"),
                    "large_url": img.get("src"),
                    "thumb_url": img.get("thumb"),
                    "alt": normalize_text(img.get("alt")),
                    "source": "html",
                }
            )
    videos = media_node.get("videos")
    if isinstance(videos, list):
        for i, vid in enumerate(videos):
            if not isinstance(vid, dict):
                continue
            media.append(
                {
                    "kind": "video",
                    "position": i,
                    "video_url": vid.get("src") or vid.get("url"),
                    # AdMedia.thumb_url existed but the old parser never set it here.
                    "video_thumb": vid.get("poster") or vid.get("thumbnail"),
                    "source": "html",
                }
            )
    if isinstance(media_node.get("count"), int):
        fields["media_count"] = media_node["count"]

    # -- JSON-LD -----------------------------------------------------------
    condition = condition_from_json_ld(json_ld)
    if condition:
        fields["condition_new_used"] = condition
        fields["condition_source"] = "json_ld"
        if condition == "new" and isinstance(mileage_km, int) and mileage_km > 0:
            warnings.append(
                f"condition_mileage_conflict: itemCondition=new but mileage_km={mileage_km}"
            )

    return {
        "fields": {k: v for k, v in fields.items() if v is not None},
        "media": media,
        "raw": _flatten(safe),
        "json_ld": json_ld,
        "payload": safe,
        "warnings": warnings,
        "sha256": digest,
        "ok": True,
        "agreements": agreements,
    }


def _flatten(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten to ``(path, value)`` pairs, collapsing ``{value,text}`` wrappers.

    Unlike the original scraper's flattener this keeps the machine ``value``
    alongside the display ``text``, so nothing is lost for unmapped fields.
    """
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        wrapper_keys = {"value", "text", "label", "textKey", "unit", "icon", "display_name"}
        if "value" in node and set(node) <= wrapper_keys:
            if node.get("text") is not None:
                out.append((f"{prefix}.text" if prefix else "text", node["text"]))
            if node.get("value") is not None:
                out.append((f"{prefix}.value" if prefix else "value", node["value"]))
            if node.get("unit") is not None:
                out.append((f"{prefix}.unit" if prefix else "unit", node["unit"]))
            return out
        for key, value in node.items():
            if key == "images" and prefix.endswith("media"):
                continue  # image blobs live in ad_media_deep
            out.extend(_flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(node, list):
        if node and all(not isinstance(v, (dict, list)) for v in node):
            out.append((prefix, node))
        else:
            for i, value in enumerate(node):
                out.extend(_flatten(value, f"{prefix}[{i}]"))
    elif node is not None and node != "":
        out.append((prefix, node))
    return out
