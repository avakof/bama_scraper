"""Offline validation sweep. Issues ZERO network requests.

Parses every advertisement record already on disk and reports field coverage, so
the parsers are proven against real data before any live request is made:

* ``output/raw/*.json.gz`` -- 8,536 search-API records whose shape closely matches
  the per-ad detail API (same ``detail``/``specs``/``price``/``images`` blocks).
* ``output/bama_ads.sqlite`` -- the 2,846 stored HTML-derived payloads, replayed
  through the numeric reconciliation to exercise the dot-stripped rule on real
  distributions.

Usage::

    python deep_scraper/tools/sweep_corpus.py [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bama_deep.ad_api import parse_ad_api  # noqa: E402
from bama_deep.numbers import reconcile_dotless  # noqa: E402
from bama_deep.scrub import assert_no_contact  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NOW = 1_800_000_000.0


def sweep_search_corpus(limit: int | None = None) -> dict:
    """Run every offline record through the API parser shape."""
    files = sorted(glob.glob(str(REPO / "output" / "raw" / "*.json.gz")))
    seen: set[str] = set()
    coverage: collections.Counter[str] = collections.Counter()
    warnings: collections.Counter[str] = collections.Counter()
    failures: list[tuple[str, str]] = []
    contact_leaks: list[str] = []
    records = 0

    for path in files:
        try:
            blob = json.load(gzip.open(path, "rt", encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            failures.append((path, f"unreadable: {exc!r}"))
            continue
        for entry in (blob.get("data") or {}).get("ads") or []:
            if entry.get("type") != "ad" or not entry.get("detail"):
                continue
            code = entry["detail"].get("code")
            if not code or code in seen:
                continue
            seen.add(code)
            records += 1
            if limit and records > limit:
                break
            # Wrap a search card in the detail-API envelope: the blocks line up.
            envelope = {
                "status": True,
                "errors": [],
                "data": {
                    "detail": entry.get("detail") or {},
                    "specs": _specs_from_card(entry.get("specs")),
                    "price": entry.get("price") or {},
                    "images": entry.get("images"),
                    "videos": entry.get("videos"),
                    "dealer": entry.get("dealer"),
                    "metadata": entry.get("metadata") or {},
                    "authenticity": entry.get("authenticity"),
                    "collaboration_price": entry.get("collaboration_price"),
                },
            }
            try:
                result = parse_ad_api(envelope, now_ts=NOW)
            except Exception as exc:  # a parser crash is a hard failure
                failures.append((code, repr(exc)))
                continue
            for key, value in result["fields"].items():
                if value is not None:
                    coverage[key] += 1
            for warning in result["warnings"]:
                warnings[warning.split(":")[0]] += 1
            leaks = assert_no_contact(json.dumps(result["fields"], ensure_ascii=False))
            if leaks:
                contact_leaks.append(code)

    return {
        "files": len(files),
        "records": len(seen),
        "parsed": records - len(failures),
        "failures": failures,
        "coverage": coverage,
        "warnings": warnings,
        "contact_leaks": contact_leaks,
    }


def _specs_from_card(specs: dict | None) -> dict:
    """Search cards name the spec keys slightly differently from the detail API."""
    specs = specs or {}
    return {
        "volume": specs.get("volume"),
        "engine": specs.get("engine"),
        "acceleration": specs.get("acceleration"),
        "fuel": specs.get("fuel"),
        "battery_capacity": specs.get("battery_capacity"),
        "all_electric_range": specs.get("all_electric_range"),
        "url_price": specs.get("url_price"),
        "url_review": specs.get("url_review"),
    }


def sweep_dotless_rule() -> dict:
    """Replay every stored spec text/int pair through the reconciliation.

    The stored database keeps only the display text, so this checks the parse of
    the text itself and the agreement classes the rule produces for each shape.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{REPO / 'output' / 'bama_ads.sqlite'}?mode=ro", uri=True)
    agreements: collections.Counter[str] = collections.Counter()
    samples: dict[str, tuple[str, float]] = {}
    for key in ("acceleration", "volume", "fuelConsumption", "power", "torque"):
        rows = conn.execute(
            "SELECT value, COUNT(*) FROM raw_attributes WHERE key=? GROUP BY value",
            (f"specs.details.{key}",),
        ).fetchall()
        for text, count in rows:
            from bama_deep.numbers import fractional_digits, parse_decimal

            number = parse_decimal(text)
            if number is None:
                agreements[f"{key}:unparseable"] += count
                continue
            digits = fractional_digits(text)
            implied_int = round(number * (10**digits))
            value, agreement = reconcile_dotless(implied_int, text)
            agreements[f"{key}:{agreement}"] += count
            if value is not None and agreement not in samples:
                samples[agreement] = (text, value)
    conn.close()
    return {"agreements": agreements, "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    print("=== offline search-corpus sweep (zero requests) ===")
    result = sweep_search_corpus(args.limit)
    print(f"files={result['files']}  unique_records={result['records']}  parsed={result['parsed']}")
    print(f"parser failures: {len(result['failures'])}")
    for code, err in result["failures"][:10]:
        print(f"   {code}: {err}")
    print(f"contact leaks:   {len(result['contact_leaks'])}")

    total = max(result["records"], 1)
    print("\n--- field coverage (share of records with a non-null value) ---")
    for field, count in sorted(result["coverage"].items(), key=lambda kv: -kv[1]):
        print(f"   {field:34} {count:6}  {100 * count / total:5.1f}%")

    print("\n--- warnings ---")
    for warning, count in result["warnings"].most_common():
        print(f"   {warning:34} {count}")

    print("\n=== dot-stripped reconciliation over stored spec texts ===")
    dot = sweep_dotless_rule()
    for label, count in sorted(dot["agreements"].items()):
        print(f"   {label:40} {count}")
    print("\n   samples:")
    for agreement, (text, value) in sorted(dot["samples"].items()):
        print(f"     {agreement:14} {text!r:26} -> {value}")

    ok = not result["failures"] and not result["contact_leaks"]
    print(f"\nelapsed {time.time() - started:.1f}s   GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
