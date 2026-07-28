"""Merge the JSON-API and HTML views of one advertisement.

Each source is authoritative for different things:

* the **API** is the only place ``fuel``, ``province``, ``is_pre_sale``,
  ``publish_networks`` and full-resolution image URLs exist;
* the **HTML** carries the typed integers (``mileage.value``, ``price.fixed.value``)
  and the dot-stripped numeric pairs, plus ``media.badge`` and the breadcrumb slugs;
* **JSON-LD** is the sole explicit statement of new-versus-used condition.

Rather than hand-coding precedence at every field, :data:`FIELD_RULES` declares it
once. Every merged field records where its value came from
(``provenance_json``), and a disagreement additionally writes a row to
``ad_field_conflicts`` -- the provenance map answers "where did this come from?"
for all fields cheaply, while the conflicts table answers "what disagreed?"
without scanning 170k rows.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from bama_scraper.normalization import normalize_text

from .scrub import scrub_payload

Source = Literal["api", "html", "ld"]
Severity = Literal["critical", "warn", "info"]

#: Provenance codes stored in ``provenance_json``.
CODES: dict[str, str] = {"api": "A", "html": "H", "ld": "L", "both": "B", "none": "-"}

#: Fields where a disagreement means the dataset is untrustworthy, not merely odd.
CRITICAL_FIELDS: frozenset[str] = frozenset(
    {"code", "price_toman", "mileage_km", "year_jalali", "trim", "review_key"}
)


CompareMode = Literal["text", "loose", "subset", "components", "breadcrumb", "temporal"]

#: Fields the two surfaces format differently by design. The API renders a title
#: as ``دنا، پلاس EF7`` while the page renders ``دنا پلاس EF7``, and the API's
#: ``location`` is city-only where the page gives ``city، province``. Comparing
#: these strictly reported a 100% conflict rate on real data and buried the
#: genuine signals, so they are compared with punctuation and granularity
#: tolerance instead.
_LOOSE_FIELDS: frozenset[str] = frozenset({"title", "subtitle", "meta_title", "color_combined"})
_SUBSET_FIELDS: frozenset[str] = frozenset({"meta_description", "description"})

#: Location is a component list rendered at different granularity: the API gives
#: ``کرج / فردیس`` (city / region) while the page gives ``کرج، البرز، فردیس``
#: (city، province، region). Comparing the component *sets* treats the coarser
#: rendering as consistent with the finer one instead of a 62% conflict rate.
_COMPONENT_FIELDS: frozenset[str] = frozenset({"location_text"})

#: The two surfaces serialize the breadcrumb with different key names -- the API
#: emits ``{"title","url"}`` while the page emits ``{"title","to","navigatable"}``.
#: The trail itself is identical, so compare the extracted (title, link) pairs
#: rather than the raw JSON, which would otherwise conflict on every single ad.
_BREADCRUMB_FIELDS: frozenset[str] = frozenset({"breadcrumb_json"})

#: Values that legitimately differ between two captures of the same ad, because
#: they are relative to when the capture happened. A difference here is not a
#: defect and must not be reported as a conflict.
_TEMPORAL_FIELDS: frozenset[str] = frozenset({"published_text", "published_ts"})


@dataclass(frozen=True)
class Rule:
    """Precedence for one output field."""

    field: str
    #: Source order; the first source with a non-None value wins.
    prefer: tuple[Source, ...] = ("api", "html")
    severity: Severity = "info"
    #: How to decide whether two source values mean the same thing.
    compare_mode: CompareMode = "text"


def _compare_mode_for(name: str) -> CompareMode:
    if name in _TEMPORAL_FIELDS:
        return "temporal"
    if name in _BREADCRUMB_FIELDS:
        return "breadcrumb"
    if name in _COMPONENT_FIELDS:
        return "components"
    if name in _SUBSET_FIELDS:
        return "subset"
    if name in _LOOSE_FIELDS:
        return "loose"
    return "text"


def _rule(name: str, *prefer: Source, severity: Severity = "info") -> Rule:
    return Rule(
        field=name,
        prefer=prefer or ("api", "html"),
        severity="critical" if name in CRITICAL_FIELDS else severity,
        compare_mode=_compare_mode_for(name),
    )


#: HTML wins for anything it exposes as a typed number.
_HTML_FIRST = (
    "mileage_km",
    "year_jalali",
    "year_gregorian",
    "price_toman",
    "installment_total_toman",
    "installments_count",
    "stats_model_ad_count",
    "stats_trim_ad_count",
    "engine_volume_l",
    "power_hp",
    "torque_nm",
    "acceleration_s",
    "fuel_consumption_l100km",
    "battery_capacity_kwh",
    "all_electric_range_km",
    "engine_volume_raw",
    "power_raw",
    "torque_raw",
    "acceleration_raw",
    "fuel_consumption_raw",
    "is_promoted",
    "badge",
    "meta_slogan",
    "breadcrumb_json",
    "trim",
    "model_fa",
    "brand_fa",
    "vehicle_category",
)

#: API wins for everything it uniquely knows, and for canonical display strings.
_API_FIRST = (
    "code",
    "ad_type",
    "title",
    "subtitle",
    "trim_en",
    "trim_fa",
    "brand",
    "model",
    "body_type",
    "body_type_fa",
    "body_value",
    "body_display_name",
    "ad_class_id",
    "canonical_url",
    "fuel_type",
    "province",
    "city",
    "region",
    "location_text",
    "is_pre_sale",
    "pre_sale_delivery",
    "pre_sale_delivery_value",
    "publish_networks",
    "inspection_station_count",
    "specialcase",
    "modified_date",
    "modified_ts",
    "image_count",
    "has_related",
    "authenticity_json",
    "options_json",
    "meta_title",
    "meta_keywords",
    "collaboration_price_toman",
    "dealer_score",
    "dealer_score_heuristic",
    "dealer_activity_years",
    "dealer_activity_months",
    "transmission",
    "engine",
    "drivetrain",
    "body_status",
    "body_color",
    "inside_color",
    "color_combined",
    "mileage_text",
    "price_text",
    "price_type",
    "year_text",
    "seller_type",
    "dealer_id",
    "dealer_name",
    "dealer_link",
    "dealer_address",
    "dealer_ad_count",
    "dealer_score_raw",
    "authenticated",
    "published_text",
    "published_ts",
    "description",
    "description_redactions",
    "meta_description",
    "meta_canonical",
    "meta_noindex",
    "review_url",
    "review_key",
    "price_url",
    "price_key",
    "engine_volume_text",
    "power_text",
    "torque_text",
    "acceleration_text",
    "fuel_consumption_text",
    "battery_capacity_text",
    "all_electric_range_text",
    "is_negotiable",
    "is_installment",
    "price_hidden",
    "down_payment_toman",
    "prepayment_secondary_toman",
    "installment_amount_toman",
    "installment_months",
    "delivery_days",
    "is_zero_km",
    "life_styles_json",
    "media_count",
    "video_count",
    "primary_image_url",
    "dealer_activity_text",
)

#: JSON-LD is the only source of an explicit condition statement.
_LD_ONLY = ("condition_new_used", "condition_source")

FIELD_RULES: tuple[Rule, ...] = (
    *(_rule(name, "html", "api") for name in _HTML_FIRST),
    *(_rule(name, "api", "html") for name in _API_FIRST),
    *(_rule(name, "html") for name in _LD_ONLY),
)


@dataclass
class MergeResult:
    record: dict[str, Any] = field(default_factory=dict)
    media: list[dict[str, Any]] = field(default_factory=list)
    raw: list[tuple[str, str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _loose(text: str) -> str:
    """Normalize away punctuation and spacing differences between surfaces."""
    return re.sub(r"[،,/\\\s‌|-]+", "", normalize_text(text) or "")


def _breadcrumb_trail(text: str) -> list[tuple[str, str]]:
    """Extract (title, link) pairs from either surface's breadcrumb encoding."""
    try:
        items = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    trail: list[tuple[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        link = item.get("url") if item.get("url") is not None else item.get("to")
        trail.append((str(item.get("title") or ""), str(link or "")))
    return trail


def _components(text: str) -> set[str]:
    """Split a location-style string into its component parts."""
    parts = re.split(r"[،,/\\|]+", normalize_text(text) or "")
    return {p.strip() for p in parts if p.strip()}


def _equivalent(a: Any, b: Any, *, mode: CompareMode) -> bool:
    """Whether two source values mean the same thing under the field's rule."""
    if a is None or b is None:
        return True  # nothing to disagree about
    if mode == "temporal":
        return True  # relative to capture time; a difference is not a defect
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, str) and isinstance(b, str):
        if mode == "loose":
            return _loose(a) == _loose(b)
        if mode == "breadcrumb":
            return _breadcrumb_trail(a) == _breadcrumb_trail(b)
        if mode == "components":
            left_set, right_set = _components(a), _components(b)
            return (
                bool(left_set)
                and bool(right_set)
                and (left_set <= right_set or right_set <= left_set)
            )
        if mode == "subset":
            left, right = _loose(a), _loose(b)
            return left == right or left in right or right in left
        return (normalize_text(a) or "") == (normalize_text(b) or "")
    return a == b


def merge_ad(
    api_result: dict[str, Any] | None,
    html_result: dict[str, Any] | None,
    *,
    ad_id: str,
    url: str,
    api_url: str | None = None,
    now_ts: float | None = None,
    deep_version: str = "1.0.0",
) -> MergeResult:
    """Reconcile both views into a single ``ad_deep`` row.

    Either side may be ``None`` (a 404 on that source). With both ``None`` the
    advertisement is recorded as delisted rather than dropped, so it stays
    accounted for.
    """
    now = time.time() if now_ts is None else now_ts
    out = MergeResult()

    api_fields: dict[str, Any] = (api_result or {}).get("fields", {}) or {}
    html_fields: dict[str, Any] = (html_result or {}).get("fields", {}) or {}
    values: dict[Source, dict[str, Any]] = {"api": api_fields, "html": html_fields, "ld": {}}

    record: dict[str, Any] = {
        "ad_id": ad_id,
        "url": url,
        "api_url": api_url,
        "api_ok": 1 if api_result else 0,
        "html_ok": 1 if (html_result and html_result.get("ok")) else 0,
        "deep_version": deep_version,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
    }

    for rule in FIELD_RULES:
        candidates = [(src, values[src].get(rule.field)) for src in ("api", "html", "ld")]
        present = [(src, val) for src, val in candidates if val is not None]
        if not present:
            out.provenance[rule.field] = CODES["none"]
            continue

        # Choose by declared preference.
        chosen_source, chosen = present[0]
        for src in rule.prefer:
            match = next(((s, v) for s, v in present if s == src), None)
            if match:
                chosen_source, chosen = match
                break

        record[rule.field] = chosen

        api_value, html_value = api_fields.get(rule.field), html_fields.get(rule.field)
        agree = _equivalent(api_value, html_value, mode=rule.compare_mode)
        if api_value is not None and html_value is not None:
            out.provenance[rule.field] = CODES["both"] if agree else CODES[chosen_source]
        else:
            out.provenance[rule.field] = CODES[chosen_source]

        if not agree:
            out.conflicts.append(
                {
                    "field": rule.field,
                    "api_value": api_value,
                    "html_value": html_value,
                    "chosen_value": chosen,
                    "chosen_source": chosen_source,
                    "severity": rule.severity,
                }
            )

    # -- evidence ---------------------------------------------------------
    if api_result:
        record["api_json_sha256"] = api_result.get("sha256")
        record["api_raw_json"] = json.dumps(
            dict(api_result.get("raw") or []), ensure_ascii=False, default=str
        )
    if html_result:
        record["html_sha256"] = html_result.get("sha256")
        if html_result.get("payload") is not None:
            record["html_payload_json"] = json.dumps(
                html_result["payload"], ensure_ascii=False, default=str
            )
        if html_result.get("json_ld"):
            # Re-scrubbed here as well as in the parser: JSON-LD Product.description
            # replicates the seller's text, and this invariant must hold for any
            # caller, including replays of legacy payloads.
            safe_ld, _ = scrub_payload(html_result["json_ld"])
            record["json_ld_json"] = json.dumps(safe_ld, ensure_ascii=False, default=str)
        if html_result.get("agreements"):
            record["numeric_agreement_json"] = json.dumps(html_result["agreements"])

    # -- media: API preferred (it has original/large/small), HTML fills gaps
    api_media = (api_result or {}).get("media") or []
    html_media = (html_result or {}).get("media") or []
    out.media = _merge_media(api_media, html_media)

    # -- raw attributes from both sources, tagged with provenance
    for source, result in (("api", api_result), ("html", html_result)):
        for key, value in (result or {}).get("raw") or []:
            out.raw.append((source, key, value))

    out.warnings = list((api_result or {}).get("warnings") or []) + list(
        (html_result or {}).get("warnings") or []
    )
    for conflict in out.conflicts:
        if conflict["severity"] == "critical":
            out.warnings.append(f"critical_conflict:{conflict['field']}")

    if not api_result and not html_result:
        record["is_delisted"] = 1
        record["delisted_at"] = record["scraped_at"]
        record["parse_status"] = "delisted"
    elif not api_result or not (html_result and html_result.get("ok")):
        record["parse_status"] = "partial"
    else:
        record["parse_status"] = "ok"

    record["provenance_json"] = json.dumps(out.provenance, ensure_ascii=False)
    record["conflict_count"] = len(out.conflicts)
    record["parse_warnings_json"] = (
        json.dumps(out.warnings, ensure_ascii=False) if out.warnings else None
    )
    out.record = {k: v for k, v in record.items() if v is not None}
    return out


def _merge_media(
    api_media: list[dict[str, Any]], html_media: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Union both media lists on ``(kind, position)``, preferring API URLs.

    The API supplies ``original``/``large``/``small``; the HTML supplies ``alt``
    text and the video ``poster`` the original scraper never stored.
    """
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for item in api_media:
        merged[(item["kind"], item["position"])] = dict(item)
    for item in html_media:
        key = (item["kind"], item["position"])
        if key in merged:
            target = merged[key]
            for column, value in item.items():
                if value is not None and target.get(column) is None:
                    target[column] = value
            target["source"] = "both"
        else:
            merged[key] = dict(item)
    return [merged[k] for k in sorted(merged, key=lambda k: (k[0], k[1]))]
