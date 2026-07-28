"""Offline merge validation. Issues ZERO network requests.

Joins the 8,536 offline search-API records against the 2,846 HTML-derived
payloads already stored by the original scraper, runs the real merge engine over
every pair, and reports the conflict distribution.

This is the highest-value free checkpoint in the build: it exercises the whole
precedence/provenance/conflict machinery on genuine data before a single live
request is made.

Usage::

    python deep_scraper/tools/merge_report.py
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bama_deep.ad_api import parse_ad_api  # noqa: E402
from bama_deep.merge import CRITICAL_FIELDS, merge_ad  # noqa: E402
from bama_deep.scrub import assert_no_contact  # noqa: E402

from tools.make_fixtures import envelope  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NOW = 1_800_000_000.0
CRITICAL_RATE_LIMIT = 0.01


def load_api_records() -> dict[str, dict]:
    """Unique advertisement records from the offline corpus, keyed by code."""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(REPO / "output" / "raw" / "*.json.gz"))):
        blob = json.load(gzip.open(path, "rt", encoding="utf-8"))
        for entry in (blob.get("data") or {}).get("ads") or []:
            if entry.get("type") != "ad" or not entry.get("detail"):
                continue
            code = entry["detail"].get("code")
            if code and code not in out:
                out[code] = envelope(entry)
    return out


def load_html_payloads() -> dict[str, dict]:
    """The stored HTML-derived payloads, reshaped as an ad_html parse result."""
    conn = sqlite3.connect(f"file:{REPO / 'output' / 'bama_ads.sqlite'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[str, dict] = {}
    for row in conn.execute("SELECT ad_id, payload FROM ad_details"):
        record = json.loads(row["payload"])
        # Rebuild the subset of ad_html's output that the merge engine consumes.
        fields = {
            k: record.get(k)
            for k in (
                "mileage_km",
                "year_jalali",
                "year_gregorian",
                "price_toman",
                "price_text",
                "price_type",
                "trim",
                "model_fa",
                "brand_fa",
                "vehicle_category",
                "title",
                "body_status",
                "body_color",
                "inside_color",
                "transmission",
                "engine",
                "drivetrain",
                "location_text",
                "province",
                "city",
                "published_text",
                "published_ts",
                "description",
                "review_url",
                "price_url",
                "mileage_text",
                "is_zero_km",
                "media_count",
                "engine_volume_text",
                "acceleration_text",
                "power_text",
                "torque_text",
                "fuel_consumption_text",
                "is_negotiable",
                "is_installment",
            )
            if record.get(k) is not None
        }
        from bama_deep.endpoints import normalize_review_key

        if fields.get("review_url"):
            fields["review_key"] = normalize_review_key(fields["review_url"])
        out[row["ad_id"]] = {
            "fields": fields,
            "media": [],
            "raw": [],
            "json_ld": record.get("json_ld") or [],
            "payload": None,
            "warnings": [],
            "sha256": record.get("html_sha256"),
            "ok": True,
            "agreements": {},
        }
    conn.close()
    return out


def main() -> int:
    started = time.time()
    api_records = load_api_records()
    html_records = load_html_payloads()
    codes = sorted(set(api_records) & set(html_records))
    print("=== offline merge validation (zero requests) ===")
    print(
        f"api records={len(api_records)}  html payloads={len(html_records)}  "
        f"joined pairs={len(codes)}"
    )

    conflicts: collections.Counter[str] = collections.Counter()
    critical: collections.Counter[str] = collections.Counter()
    provenance: collections.Counter[str] = collections.Counter()
    examples: dict[str, tuple] = {}
    leaks: list[str] = []
    failures: list[tuple[str, str]] = []
    merged = 0

    for code in codes:
        api_parsed = parse_ad_api(api_records[code], now_ts=NOW)
        try:
            result = merge_ad(
                api_parsed,
                html_records[code],
                ad_id=code,
                url=f"https://bama.ir/car/detail-{code}-x",
                now_ts=NOW,
            )
        except Exception as exc:
            failures.append((code, repr(exc)))
            continue
        merged += 1
        for conflict in result.conflicts:
            conflicts[conflict["field"]] += 1
            if conflict["severity"] == "critical":
                critical[conflict["field"]] += 1
            examples.setdefault(
                conflict["field"],
                (code, conflict["api_value"], conflict["html_value"], conflict["chosen_source"]),
            )
        for code_letter in result.provenance.values():
            provenance[code_letter] += 1
        text_blob = json.dumps(
            {k: v for k, v in result.record.items() if isinstance(v, str) and "sha256" not in k},
            ensure_ascii=False,
        )
        if assert_no_contact(text_blob):
            leaks.append(code)

    print(f"merged={merged}  failures={len(failures)}  contact leaks={len(leaks)}")
    for code, err in failures[:10]:
        print(f"   {code}: {err}")

    print("\n--- provenance distribution (per merged field) ---")
    labels = {"A": "api only", "H": "html only", "B": "both agreed", "L": "json-ld", "-": "absent"}
    total_fields = sum(provenance.values()) or 1
    for letter, count in provenance.most_common():
        print(
            f"   {letter} {labels.get(letter, '?'):12} {count:8}  {100 * count / total_fields:5.1f}%"
        )

    print("\n--- conflicts by field ---")
    if not conflicts:
        print("   (none)")
    for field_name, count in conflicts.most_common(25):
        rate = 100 * count / max(merged, 1)
        flag = "CRITICAL" if field_name in CRITICAL_FIELDS else ""
        code, api_value, html_value, chosen = examples[field_name]
        print(f"   {field_name:26} {count:6} ({rate:5.1f}%) {flag}")
        print(f"        e.g. {code}: api={api_value!r:28} html={html_value!r:28} -> {chosen}")

    critical_total = sum(critical.values())
    critical_rate = critical_total / max(merged, 1)
    print(f"\ncritical conflicts: {critical_total}  rate={critical_rate:.4%}")
    ok = not failures and not leaks and critical_rate < CRITICAL_RATE_LIMIT and merged > 0
    print(f"elapsed {time.time() - started:.1f}s   GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
