"""Parse ``GET /cad/api/detail/<code>`` -- the public per-advertisement JSON API.

This is the endpoint the detail page itself is server-rendered from. It is the
only source for several fields: ``fuel`` (absent from the HTML payload entirely),
``province``, ``is_pre_sale``, ``ad_class_i_d``, ``publish_networks``,
``inspection_station_count``, full-resolution ``images[].original`` and a real
float ``dealer.score``.

What it does *not* give is typed numerics -- ``specs.*`` and ``price.*`` arrive as
display strings -- so the HTML payload remains the authority for those.

``data.detail.phone`` carries a server-masked seller mobile and is removed before
anything is returned.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bama_scraper.normalization import (
    is_zero_km,
    normalize_mileage,
    normalize_price,
    normalize_text,
    parse_published_time,
    parse_year,
)

from .endpoints import dealer_id_from_url, normalize_price_key, normalize_review_key
from .numbers import descale_bounded, parse_decimal, parse_iso_ts
from .scrub import scrub_freetext, scrub_payload

#: Keys under ``data`` that are pure presentation and carry no car information.
_NOISE_KEYS = frozenset({"icon"})


def _text_or_none(value: Any) -> str | None:
    """Normalize only what is genuinely text.

    Bama returns some fields as a bare string and the same field as a
    ``{"value", "display_name"}`` object on other listings. Passing the object
    into a text normalizer raises, which is how this was found: four pre-sale
    listings failed their detail check with
    ``TypeError: normalize() argument 2 must be str, not dict``.
    """
    return normalize_text(value) if isinstance(value, str) else None


def _code_or_none(value: Any) -> str | None:
    """The machine code when the field arrived as a bare string."""
    return value if isinstance(value, str) and value else None


def _sub(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    """Bama uses 0 as an "inapplicable" sentinel for installment fields."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value) or None
    if isinstance(value, str):
        parsed = parse_decimal(value)
        return int(parsed) if parsed else None
    return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def parse_ad_api(payload: dict[str, Any], *, now_ts: float) -> dict[str, Any]:
    """Extract a flat field map plus media, raw attributes and warnings.

    Returns ``{"fields": {...}, "media": [...], "raw": [(key, value)],
    "warnings": [...], "sha256": ...}``. The caller merges ``fields`` with the
    HTML-derived map; nothing here writes to the database directly.
    """
    warnings: list[str] = []
    safe, payload_redactions = scrub_payload(payload)
    digest = hashlib.sha256(
        json.dumps(safe, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    data = safe.get("data")
    if not isinstance(data, dict):
        return {
            "fields": {},
            "media": [],
            "raw": [],
            "warnings": ["api payload has no data object"],
            "sha256": digest,
        }

    detail = _sub(data, "detail")
    specs = _sub(data, "specs")
    price = _sub(data, "price")
    dealer = _sub(data, "dealer")
    meta = _sub(data, "metadata")
    stats = _sub(data, "ads_statistics")
    body = _sub(detail, "body")
    pre_sale_delivery = _sub(detail, "pre_sale_delivery")

    year_text = detail.get("year")
    year_jalali, year_gregorian = parse_year(year_text)

    mileage_text = normalize_text(detail.get("mileage"))
    mileage_km = normalize_mileage(detail.get("mileage"))

    # Only the scrubbed text is ever persisted (see scrub.scrub_payload).
    description, redactions = scrub_freetext(normalize_text(detail.get("description")))

    # Location: "city" or "city / region"; province is a separate field.
    location_text = normalize_text(detail.get("location"))
    city = region = None
    if location_text:
        parts = [p.strip() for p in location_text.split("/")]
        city = parts[0] or None
        region = parts[1] if len(parts) > 1 else None

    price_type = price.get("type")
    fields: dict[str, Any] = {
        # identity
        "code": detail.get("code"),
        "ad_type": detail.get("type"),
        "title": normalize_text(detail.get("title")),
        "subtitle": normalize_text(detail.get("subtitle")),
        "brand": detail.get("brand"),
        "model": detail.get("model"),
        "trim_en": detail.get("trim_en"),
        "trim_fa": normalize_text(detail.get("trim")),
        "year_text": normalize_text(str(year_text)) if year_text else None,
        "year_jalali": year_jalali,
        "year_gregorian": year_gregorian,
        "body_type": detail.get("body_type"),
        "body_value": body.get("value"),
        "body_display_name": normalize_text(body.get("display_name")),
        "body_type_fa": normalize_text(specs.get("body_type"))
        or normalize_text(body.get("display_name")),
        "ad_class_id": detail.get("ad_class_i_d"),
        "canonical_url": meta.get("canonical"),
        # price and terms
        "price_text": normalize_text(price.get("price")),
        "price_toman": normalize_price(price.get("price")),
        "price_type": price_type,
        "is_negotiable": price_type == "negotiable",
        "is_installment": price_type == "installment",
        "price_hidden": price_type in ("hidden", "notShown"),
        "down_payment_toman": normalize_price(price.get("prepayment_primary")),
        "prepayment_secondary_toman": normalize_price(price.get("prepayment_secondary")),
        "installment_amount_toman": normalize_price(price.get("payment_primary")),
        "installment_months": _int_or_none(price.get("month_number")),
        "installments_count": _int_or_none(price.get("installments")),
        "delivery_days": _int_or_none(price.get("delivery_days")),
        "is_pre_sale": _bool_or_none(detail.get("is_pre_sale")),
        # A coded enum, not a string: {"value": "SixToNineMonths",
        # "display_name": "تحویل 6 تا 9 ماه"}. Treating it as text raised
        # TypeError on every pre-sale listing. Both halves are kept — the code is
        # stable across wording changes, the label is what a reader recognises.
        "pre_sale_delivery": normalize_text(pre_sale_delivery.get("display_name"))
        or _text_or_none(detail.get("pre_sale_delivery")),
        "pre_sale_delivery_value": pre_sale_delivery.get("value")
        or _code_or_none(detail.get("pre_sale_delivery")),
        "collaboration_price_toman": normalize_price(data.get("collaboration_price")),
        # condition
        "mileage_text": mileage_text,
        "mileage_km": mileage_km,
        "is_zero_km": is_zero_km(mileage_text, mileage_km),
        "body_status": normalize_text(detail.get("body_status")),
        "authenticated": _bool_or_none(detail.get("authenticated")),
        "authenticity_json": json.dumps(data.get("authenticity"), ensure_ascii=False)
        if data.get("authenticity") is not None
        else None,
        # powertrain -- strings only; the HTML supplies the typed values
        "transmission": normalize_text(detail.get("transmission")),
        "fuel_type": normalize_text(detail.get("fuel")),
        "engine": normalize_text(specs.get("engine")),
        "drivetrain": normalize_text(specs.get("drive_shaft")),
        "engine_volume_text": normalize_text(specs.get("volume")),
        "power_text": normalize_text(specs.get("power")),
        "torque_text": normalize_text(specs.get("torque")),
        "acceleration_text": normalize_text(specs.get("acceleration")),
        "fuel_consumption_text": normalize_text(specs.get("fuel")),
        "battery_capacity_text": normalize_text(specs.get("battery_capacity")),
        "all_electric_range_text": normalize_text(specs.get("all_electric_range")),
        "body_color": normalize_text(detail.get("body_color")),
        "inside_color": normalize_text(detail.get("inside_color")),
        "color_combined": normalize_text(detail.get("color")),
        # place and time
        "location_text": location_text,
        "province": normalize_text(detail.get("province")),
        "city": city,
        "region": region,
        "published_text": normalize_text(detail.get("time")),
        "published_ts": parse_published_time(normalize_text(detail.get("time")), now_ts=now_ts),
        "modified_date": detail.get("modified_date"),
        "modified_ts": parse_iso_ts(detail.get("modified_date")),
        # ad surface
        "description": description,
        "description_redactions": redactions + payload_redactions,
        "badge": _bool_or_none(detail.get("badge")),
        "is_promoted": _bool_or_none(detail.get("badge")),
        "has_related": _bool_or_none(detail.get("has_related")),
        "specialcase": normalize_text(detail.get("specialcase")),
        "publish_networks": data.get("publish_networks"),
        "inspection_station_count": data.get("inspection_station_count"),
        "life_styles_json": json.dumps(detail.get("life_styles"), ensure_ascii=False)
        if detail.get("life_styles")
        else None,
        "options_json": json.dumps(data.get("options"), ensure_ascii=False)
        if data.get("options") is not None
        else None,
        "stats_model_ad_count": stats.get("model"),
        "stats_trim_ad_count": stats.get("trim"),
        "breadcrumb_json": json.dumps(_sub(data, "breadcrump").get("links"), ensure_ascii=False)
        if _sub(data, "breadcrump").get("links")
        else None,
        "meta_title": scrub_freetext(normalize_text(meta.get("title_tag")))[0],
        "meta_description": scrub_freetext(normalize_text(meta.get("description")))[0],
        "meta_keywords": normalize_text(meta.get("keywords")),
        "meta_canonical": meta.get("canonical"),
        "meta_noindex": _bool_or_none(meta.get("noindex")),
        "image_count": detail.get("image_count"),
        # join keys
        "review_url": specs.get("url_review"),
        "review_key": normalize_review_key(specs.get("url_review")),
        "price_url": specs.get("url_price"),
        "api_json_sha256": digest,
    }

    price_key, _ = normalize_price_key(specs.get("url_price"))
    fields["price_key"] = price_key

    # -- seller -----------------------------------------------------------
    if dealer:
        score_raw = dealer.get("score")
        score, heuristic = (
            (float(score_raw), False)
            if isinstance(score_raw, float)
            else descale_bounded(score_raw, lo=0.0, hi=5.0)
        )
        activity = dealer.get("activity_duration")
        activity = activity if isinstance(activity, dict) else {}
        fields.update(
            {
                "seller_type": "nonpersonal",
                "dealer_id": dealer.get("id") or dealer_id_from_url(dealer.get("link")),
                "dealer_name": normalize_text(dealer.get("name")),
                "dealer_link": dealer.get("link"),
                "dealer_address": normalize_text(dealer.get("address")),
                "dealer_ad_count": dealer.get("ad_count"),
                "dealer_score": score,
                "dealer_score_raw": int(score_raw) if isinstance(score_raw, (int, float)) else None,
                "dealer_score_heuristic": heuristic,
                "dealer_activity_years": activity.get("year"),
                "dealer_activity_months": activity.get("month"),
            }
        )
    else:
        fields["seller_type"] = "personal"

    # -- media ------------------------------------------------------------
    media: list[dict[str, Any]] = []
    images = data.get("images")
    if isinstance(images, list):
        for i, img in enumerate(images):
            if not isinstance(img, dict):
                continue
            media.append(
                {
                    "kind": "image",
                    "position": i,
                    "original_url": img.get("original"),
                    "large_url": img.get("large"),
                    "small_url": img.get("small"),
                    "thumb_url": img.get("thumb"),
                    "pinkie_url": img.get("pinkie"),
                    "source": "api",
                }
            )
    videos = data.get("videos")
    if isinstance(videos, list):
        for i, vid in enumerate(videos):
            if not isinstance(vid, dict):
                continue
            media.append(
                {
                    "kind": "video",
                    "position": i,
                    "video_id": str(vid.get("id")) if vid.get("id") is not None else None,
                    "video_url": vid.get("url") or vid.get("src"),
                    "video_thumb": vid.get("thumbnail") or vid.get("poster"),
                    "source": "api",
                }
            )

    image_media = [m for m in media if m["kind"] == "image"]
    fields["media_count"] = len(image_media)
    fields["video_count"] = len(media) - len(image_media)
    fields["primary_image_url"] = (
        image_media[0].get("original") or image_media[0].get("large_url") if image_media else None
    )
    if not image_media:
        warnings.append("no images on advertisement")
    declared = detail.get("image_count")
    if isinstance(declared, int) and declared != len(image_media):
        warnings.append(f"image_count={declared} but {len(image_media)} image objects")

    return {
        "fields": {k: v for k, v in fields.items() if v is not None},
        "media": media,
        "raw": _flatten(data),
        "warnings": warnings,
        "sha256": digest,
    }


def _flatten(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten the payload to ``(path, value)`` pairs so nothing is ever lost.

    Icon URL blocks are dropped -- they are presentation assets, not car data.
    """
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _NOISE_KEYS:
                continue
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
