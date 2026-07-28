"""Parse a Bama advertisement detail page.

Three evidence sources, in order of reliability:

1. ``__NUXT_DATA__`` — the devalue-serialized SSR payload the page itself
   renders from. Richest and most stable; used as the primary source.
2. ``application/ld+json`` — schema.org blocks, used to fill gaps and as
   corroboration.
3. The rendered DOM — last-resort fallback if the payload is absent.

Privacy: the SSR payload contains a partially masked seller phone number.
It is deliberately dropped and never persisted (see :data:`_PRIVATE_KEYS`).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from selectolax.parser import HTMLParser

from .models import AdDetail, AdMedia
from .normalization import (
    is_zero_km,
    normalize_mileage,
    normalize_price,
    normalize_text,
    parse_published_time,
    parse_year,
)

#: Keys removed from every raw payload before persistence.
_PRIVATE_KEYS: frozenset[str] = frozenset(
    {"phone", "phoneNumber", "mobile", "tel", "telephone", "contact", "phones"}
)

_NUXT_RE = re.compile(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)

#: devalue tags whose payload is a single wrapped reference.
_WRAPPER_TAGS = frozenset(
    {"Reactive", "ShallowReactive", "Ref", "ShallowRef", "EmptyRef", "EmptyShallowRef", "NuxtError"}
)


def resolve_devalue(nodes: list[Any]) -> Any:
    """Resolve a Nuxt 3 ``__NUXT_DATA__`` flat array into plain Python objects.

    The format stores every value once in a flat list; containers reference
    their members by integer index. Negative indices are sentinels
    (``-1`` = undefined, ``-2`` = hole, ``-3`` = NaN, ...), all of which map to
    ``None`` here. Results are memoised so shared sub-trees are resolved once,
    and a depth cap protects against pathological nesting.
    """
    memo: dict[int, Any] = {}

    def walk(index: Any, depth: int = 0) -> Any:
        if not isinstance(index, int):
            return index
        if index < 0 or depth > 40:
            return None
        if index in memo:
            return memo[index]
        if index >= len(nodes):
            return None
        node = nodes[index]
        memo[index] = None  # cycle guard

        out: Any
        if isinstance(node, list):
            if node and isinstance(node[0], str):
                tag = node[0]
                if tag in _WRAPPER_TAGS:
                    out = walk(node[1], depth + 1) if len(node) > 1 else None
                elif tag in ("Date", "BigInt", "URL", "RegExp"):
                    out = node[1] if len(node) > 1 else None
                elif tag == "Set":
                    out = [walk(x, depth + 1) for x in node[1:]]
                elif tag == "Map":
                    items = [walk(x, depth + 1) for x in node[1:]]
                    out = dict(zip(items[::2], items[1::2], strict=False))
                else:
                    out = [walk(x, depth + 1) for x in node[1:]]
            else:
                out = [walk(x, depth + 1) for x in node]
        elif isinstance(node, dict):
            out = {k: walk(v, depth + 1) for k, v in node.items()}
        else:
            out = node
        memo[index] = out
        return out

    return walk(0)


def strip_private(obj: Any) -> Any:
    """Recursively remove contact-information keys from a payload."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if k not in _PRIVATE_KEYS}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def extract_nuxt_payload(html: str) -> dict[str, Any] | None:
    """Return the ``get-ad-pdp-*`` payload from the page, or ``None``."""
    match = _NUXT_RE.search(html)
    if not match:
        return None
    try:
        nodes = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(nodes, list):
        return None
    root = resolve_devalue(nodes)
    if not isinstance(root, dict):
        return None
    data = root.get("data")
    if not isinstance(data, dict):
        return None
    for key, value in data.items():
        if key.startswith("get-ad-pdp-") and isinstance(value, dict):
            inner = value.get("data")
            if isinstance(inner, dict):
                return inner
    return None


def extract_json_ld(html: str) -> list[dict[str, Any]]:
    """Return all parseable JSON-LD blocks."""
    out: list[dict[str, Any]] = []
    for node in HTMLParser(html).css('script[type="application/ld+json"]'):
        text = node.text()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            out.extend(x for x in parsed if isinstance(x, dict))
        elif isinstance(parsed, dict):
            out.append(parsed)
    return out


def _get(obj: Any, *path: str) -> Any:
    """Safe nested lookup that tolerates ``None`` and non-dict nodes."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _val(obj: Any, *path: str) -> Any:
    """Lookup that unwraps Bama's ``{value, text}`` wrapper objects."""
    node = _get(obj, *path)
    if isinstance(node, dict):
        return node.get("value", node.get("text"))
    return node


def _txt(obj: Any, *path: str) -> str | None:
    node = _get(obj, *path)
    if isinstance(node, dict):
        node = node.get("text", node.get("value"))
    return normalize_text(str(node)) if node not in (None, "") else None


def _parse_price(price: Any, warnings: list[str]) -> dict[str, Any]:
    """Map Bama's price object onto normalized fields.

    Bama encodes "no stated price" as ``type: negotiable`` with empty
    sub-objects. That must never become ``0``.
    """
    result: dict[str, Any] = {
        "price_text": None,
        "price_toman": None,
        "price_type": None,
        "is_negotiable": None,
        "is_installment": None,
        "price_hidden": None,
        "is_presale": None,
        "down_payment_text": None,
        "down_payment_toman": None,
        "installment_amount_text": None,
        "installment_amount_toman": None,
        "installment_months": None,
        "delivery_days": None,
    }
    if not isinstance(price, dict):
        return result

    ptype = price.get("type")
    result["price_type"] = ptype
    result["is_negotiable"] = ptype == "negotiable"
    result["is_presale"] = ptype == "presale" or bool(price.get("presale"))
    result["price_hidden"] = ptype in ("hidden", "notShown")

    fixed = price.get("fixed")
    if isinstance(fixed, dict) and fixed.get("value") is not None:
        result["price_text"] = normalize_text(str(fixed.get("text") or ""))
        result["price_toman"] = normalize_price(fixed.get("value"))

    inst = price.get("installment")
    if isinstance(inst, dict) and inst:
        result["is_installment"] = True
        prepay = inst.get("prepayment") if isinstance(inst.get("prepayment"), dict) else None
        payment = inst.get("payment") if isinstance(inst.get("payment"), dict) else None
        result["down_payment_text"] = normalize_text(str((prepay or {}).get("text") or "")) or None
        result["down_payment_toman"] = normalize_price((prepay or {}).get("value"))
        result["installment_amount_text"] = (
            normalize_text(str((payment or {}).get("text") or "")) or None
        )
        result["installment_amount_toman"] = normalize_price((payment or {}).get("value"))
        months = inst.get("month") or inst.get("monthNumber") or inst.get("months")
        if isinstance(months, dict):
            months = months.get("value")
        result["installment_months"] = int(months) if isinstance(months, (int, float)) else None
    else:
        result["is_installment"] = False

    presale = price.get("presale")
    if isinstance(presale, dict) and presale:
        days = presale.get("deliveryDays") or presale.get("delivery_days")
        if isinstance(days, dict):
            days = days.get("value")
        result["delivery_days"] = int(days) if isinstance(days, (int, float)) else None
        if result["price_toman"] is None:
            pv = presale.get("value") or _get(presale, "price", "value")
            result["price_toman"] = normalize_price(pv)

    if result["price_toman"] is None and ptype == "lumpsum":
        warnings.append("lumpsum price without a numeric value")
    return result


def parse_detail(
    html: str,
    url: str,
    ad_id: str,
    *,
    scraper_version: str = "1.0.0",
    now_ts: float | None = None,
) -> tuple[AdDetail, list[AdMedia]]:
    """Parse detail HTML into a validated :class:`AdDetail` plus media rows."""
    now = time.time() if now_ts is None else now_ts
    warnings: list[str] = []
    payload = extract_nuxt_payload(html)
    json_ld = extract_json_ld(html)
    source = "nuxt_data"

    if payload is None:
        source = "json_ld" if json_ld else "dom"
        warnings.append("__NUXT_DATA__ payload missing; degraded parse")
        payload = {}

    safe_payload = strip_private(payload)

    content = safe_payload.get("content") or {}
    vehicle = safe_payload.get("vehicle") or {}
    specs = _get(safe_payload, "specs", "details") or {}
    media = safe_payload.get("media") or {}
    seller = safe_payload.get("seller") or {}
    meta = safe_payload.get("metadata") or {}

    # --- identity ---------------------------------------------------------
    title = _txt(content, "title")
    year_text = _txt(vehicle, "year")
    year_jalali, year_gregorian = parse_year(_val(vehicle, "year") or year_text)

    # --- price ------------------------------------------------------------
    price_fields = _parse_price(safe_payload.get("price"), warnings)

    # --- mileage ----------------------------------------------------------
    mileage_text = _txt(vehicle, "mileage")
    mileage_value = _val(vehicle, "mileage")
    mileage_km = (
        int(mileage_value)
        if isinstance(mileage_value, (int, float))
        else normalize_mileage(mileage_text)
    )

    # --- media ------------------------------------------------------------
    raw_images = media.get("images")
    images: list[Any] = raw_images if isinstance(raw_images, list) else []
    raw_videos = media.get("videos")
    videos: list[Any] = raw_videos if isinstance(raw_videos, list) else []
    media_rows: list[AdMedia] = []
    image_urls: list[str] = []
    thumb_urls: list[str] = []
    for i, img in enumerate(images):
        if not isinstance(img, dict):
            continue
        full = img.get("original") or img.get("src")
        media_rows.append(
            AdMedia(
                ad_id=ad_id,
                position=i,
                kind="image",
                url=img.get("src"),
                original_url=img.get("original"),
                thumb_url=img.get("thumb"),
                alt=normalize_text(img.get("alt")),
            )
        )
        if full:
            image_urls.append(full)
        if img.get("thumb"):
            thumb_urls.append(img["thumb"])

    video_urls: list[str] = []
    for i, vid in enumerate(videos):
        vurl = vid.get("src") or vid.get("url") if isinstance(vid, dict) else vid
        if isinstance(vurl, str):
            video_urls.append(vurl)
            media_rows.append(AdMedia(ad_id=ad_id, position=i, kind="video", url=vurl))

    if not images:
        warnings.append("no images on advertisement")

    # --- description / location ------------------------------------------
    description = _txt(content, "description")
    if description is None:
        warnings.append("empty description")
    loc_value = _get(content, "location", "value") or {}

    published_text = _txt(content, "publishedDate")
    published_ts = parse_published_time(published_text, now_ts=now)

    # --- JSON-LD corroboration -------------------------------------------
    ld_product = next((b for b in json_ld if b.get("@type") in ("Product", "Car", "Vehicle")), {})
    canonical = meta.get("canonicalURL") or ld_product.get("url") or url

    detail = AdDetail(
        ad_id=ad_id,
        url=url,
        canonical_url=canonical,
        title=title or normalize_text(ld_product.get("name")),
        brand=_val(vehicle, "brand"),
        brand_fa=_txt(vehicle, "brand"),
        model=_val(vehicle, "model"),
        model_fa=_txt(vehicle, "model"),
        trim=_val(vehicle, "trim"),
        trim_fa=_txt(vehicle, "trim"),
        year_text=year_text,
        year_jalali=year_jalali,
        year_gregorian=year_gregorian,
        vehicle_category=vehicle.get("type"),
        body_type=_val(vehicle, "bodyType") or _val(safe_payload, "body"),
        body_type_fa=(_get(safe_payload, "body", "display_name") or _val(specs, "bodyType")),
        manufacturer_country=safe_payload.get("country"),
        price_currency="IRT",
        mileage_text=mileage_text,
        mileage_km=mileage_km,
        is_zero_km=is_zero_km(mileage_text, mileage_km),
        transmission=_val(vehicle, "transmission"),
        fuel_type=_val(vehicle, "fuel") or _val(specs, "fuel"),
        engine=_val(specs, "engine"),
        engine_volume_text=_txt(specs, "volume"),
        drivetrain=_val(specs, "driveShaft"),
        body_color=_val(vehicle, "color", "body"),
        interior_color=_val(vehicle, "color", "interior"),
        body_status=_val(vehicle, "bodyStatus"),
        insurance_text=_txt(vehicle, "insurance"),
        cylinders=_val(specs, "cylinder") or _val(vehicle, "cylinder"),
        power_text=_txt(specs, "power"),
        torque_text=_txt(specs, "torque"),
        acceleration_text=_txt(specs, "acceleration"),
        fuel_consumption_text=_txt(specs, "fuelConsumption"),
        description=description,
        seller_type=seller.get("type"),
        seller_name=normalize_text(seller.get("name")),
        seller_authenticated=(
            seller.get("authenticated") if isinstance(seller.get("authenticated"), bool) else None
        ),
        dealer_url=_get(seller, "url") or _get(safe_payload, "dealer", "url"),
        province=normalize_text(loc_value.get("province")),
        city=normalize_text(loc_value.get("city")),
        neighbourhood=normalize_text(loc_value.get("region")),
        location_text=_txt(content, "location"),
        published_text=published_text,
        published_ts=published_ts,
        modified_date=safe_payload.get("modifiedDate"),
        badges=_badges(safe_payload),
        primary_image_url=image_urls[0] if image_urls else None,
        image_urls=image_urls,
        thumbnail_urls=thumb_urls,
        video_urls=video_urls,
        media_count=media.get("count") if isinstance(media.get("count"), int) else len(images),
        raw_attributes=_flatten_attributes(safe_payload),
        json_ld=json_ld,
        html_sha256=hashlib.sha256(html.encode("utf-8", "replace")).hexdigest(),
        scraper_version=scraper_version,
        scraped_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        parse_status="ok" if source == "nuxt_data" else "partial",
        parse_warnings=warnings,
        source=source,
        **price_fields,
    )
    return detail, media_rows


def _badges(payload: dict[str, Any]) -> list[str]:
    """Collect lifestyle/badge labels shown on the advertisement."""
    raw = payload.get("life_styles")
    if not isinstance(raw, list):
        return []
    return [str(b["display_name"]) for b in raw if isinstance(b, dict) and b.get("display_name")]


def _flatten_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten the payload into ``label -> value`` pairs for the raw table.

    This preserves every visible attribute Bama exposes, including ones this
    schema has no dedicated column for.
    """
    out: dict[str, Any] = {}

    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            # Collapse Bama's {value,text,label,...} wrappers to a single entry.
            if "value" in node and set(node) <= {
                "value",
                "text",
                "label",
                "textKey",
                "unit",
                "icon",
                "display_name",
            }:
                out[prefix] = node.get("text") or node.get("value")
                return
            for key, value in node.items():
                if key in _PRIVATE_KEYS:
                    continue
                walk(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(node, list):
            if all(not isinstance(x, (dict, list)) for x in node):
                if node:
                    out[prefix] = node
            else:
                for i, value in enumerate(node):
                    walk(value, f"{prefix}[{i}]")
        elif node is not None and node != "":
            out[prefix] = node

    walk(payload)
    # Image blobs are stored in ad_media; keep the raw table readable.
    return {k: v for k, v in out.items() if not k.startswith("media.images")}
