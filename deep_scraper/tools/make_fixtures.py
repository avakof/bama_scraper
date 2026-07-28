"""Build test fixtures offline. Issues ZERO network requests.

Fixtures come from data already on disk (``output/raw/*.json.gz`` and the base
project's captured HTML), scrubbed of real contact data. Fake phone numbers are
then *deliberately re-inserted* into the designated privacy fixtures, so the
scrub tests exercise a realistic payload rather than a sanitised one.

Usage::

    python deep_scraper/tools/make_fixtures.py
"""

from __future__ import annotations

import copy
import glob
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bama_deep.scrub import scrub_payload  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

FAKE_PHONE = "۰۹۱۲۰۰۰۱۱XX"


def load_cards() -> list[dict]:
    """Every unique advertisement record from the offline corpus."""
    seen: set[str] = set()
    cards: list[dict] = []
    for path in sorted(glob.glob(str(REPO / "output" / "raw" / "*.json.gz"))):
        blob = json.load(gzip.open(path, "rt", encoding="utf-8"))
        for entry in (blob.get("data") or {}).get("ads") or []:
            if entry.get("type") != "ad" or not entry.get("detail"):
                continue
            code = entry["detail"].get("code")
            if not code or code in seen:
                continue
            seen.add(code)
            cards.append(entry)
    return cards


def envelope(card: dict) -> dict:
    """Wrap a search card in the detail-API response envelope.

    Deep-copied: the corpus dicts are shared between records, so mutating a
    derived fixture in place would silently contaminate the others.
    """
    card = copy.deepcopy(card)
    specs = card.get("specs") or {}
    return {
        "status": True,
        "errors": [],
        "metadata": None,
        "data": {
            "detail": card.get("detail") or {},
            "specs": {
                "volume": specs.get("volume"),
                "engine": specs.get("engine"),
                "acceleration": specs.get("acceleration"),
                "fuel": specs.get("fuel"),
                "power": specs.get("power"),
                "torque": specs.get("torque"),
                "drive_shaft": specs.get("drive_shaft"),
                "body_type": specs.get("body_type"),
                "battery_capacity": specs.get("battery_capacity"),
                "all_electric_range": specs.get("all_electric_range"),
                "url_price": specs.get("url_price"),
                "url_review": specs.get("url_review"),
            },
            "dealer": card.get("dealer"),
            "price": card.get("price") or {},
            "images": card.get("images"),
            "videos": card.get("videos"),
            "metadata": card.get("metadata") or {},
            "breadcrump": card.get("breadcrump") or {},
            "authenticity": card.get("authenticity"),
            "collaboration_price": card.get("collaboration_price"),
            "ads_statistics": {"model": 682, "trim": 103},
            "publish_networks": "B2C",
            "inspection_station_count": 0,
            "options": None,
        },
    }


FORCE = False


def write(name: str, payload: dict) -> None:
    """Write a fixture, preserving any file that is already there.

    Some fixtures are captured live from the detail API (which carries fields the
    offline search corpus lacks, e.g. `province`). Those must not be silently
    replaced by a corpus-derived approximation -- pass --force to overwrite.
    """
    target = FIXTURES / name
    if target.exists() and not FORCE:
        print(f"  {name} (kept existing)")
        return
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {name}")


def main() -> int:
    global FORCE
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing fixtures")
    FORCE = parser.parse_args().force

    FIXTURES.mkdir(parents=True, exist_ok=True)
    cards = load_cards()
    print(f"loaded {len(cards)} unique offline records")

    def pick(predicate) -> dict | None:
        return next((c for c in cards if predicate(c)), None)

    personal = pick(lambda c: not c.get("dealer") and (c.get("images") or []))
    dealer = pick(lambda c: c.get("dealer"))
    no_images = pick(lambda c: not (c.get("images") or []))
    with_video = pick(lambda c: c.get("videos"))
    no_review = pick(lambda c: not (c.get("specs") or {}).get("url_review"))
    two_seg_price = pick(lambda c: ((c.get("specs") or {}).get("url_price") or "").count("_") == 1)

    print("writing API fixtures:")
    for name, card in (
        ("api_detail_personal.json", personal),
        ("api_detail_dealer.json", dealer),
        ("api_detail_no_images.json", no_images),
        ("api_detail_video.json", with_video),
        ("api_detail_no_review_url.json", no_review),
        ("api_detail_two_segment_price.json", two_seg_price),
    ):
        if card is None:
            print(f"  ! no corpus record for {name}")
            continue
        cleaned, _ = scrub_payload(envelope(card))
        write(name, cleaned)

    # Privacy fixture: a realistic payload that DOES carry contact data, so the
    # scrub tests prove removal rather than asserting on already-clean input.
    if personal is not None:
        leaky = envelope(personal)
        leaky["data"]["detail"]["phone"] = FAKE_PHONE
        leaky["data"]["detail"]["description"] = (
            "خودرو سالم و بی رنگ، تماس 09120001122 یا تلگرام @fake_dealer_test"
        )
        leaky["data"]["metadata"]["description"] = (
            "خرید خودرو - خودرو سالم و بی رنگ، تماس 09120001122"
        )
        write("api_detail_phone_present.json", leaky)

    # Synthetic shapes absent from this filter set (installment / pre-sale / EV).
    # Prefer a real captured detail-API response as the base when one is present,
    # so the synthetics carry genuine detail-only fields such as `province`.
    real_base = FIXTURES / "api_detail_personal.json"
    base_payload = (
        json.loads(real_base.read_text(encoding="utf-8"))
        if real_base.exists()
        else (envelope(personal) if personal is not None else None)
    )
    if base_payload is not None:
        base = base_payload
        inst = json.loads(json.dumps(base, ensure_ascii=False))
        inst["data"]["price"] = {
            "type": "installment",
            "price": "0",
            "prepayment_primary": "1,010,000,000",
            "prepayment_secondary": "0",
            "payment_primary": "63,000,000",
            "month_number": 2,
            "installments": 24,
            "delivery_days": 1,
        }
        write("api_detail_installment.json", inst)

        ev = json.loads(json.dumps(base, ensure_ascii=False))
        ev["data"]["specs"].update(
            {
                "battery_capacity": "50 کیلووات ساعت",
                "all_electric_range": "400 کیلومتر",
                "fuel": None,
                "volume": None,
            }
        )
        ev["data"]["detail"]["fuel"] = "برقی"
        write("api_detail_ev.json", ev)

        presale = json.loads(json.dumps(base, ensure_ascii=False))
        presale["data"]["detail"]["is_pre_sale"] = True
        presale["data"]["detail"]["pre_sale_delivery"] = "کمتر از 3 ماه"
        presale["data"]["detail"]["mileage"] = "صفر کیلومتر"
        write("api_detail_presale.json", presale)

    # Error envelope shapes.
    write("api_detail_empty.json", {"status": False, "errors": ["not found"], "data": None})

    # HTML fixtures: reuse the base project's captured page, plus degenerate cases.
    print("writing HTML fixtures:")
    source_html = REPO / "tests" / "fixtures" / "detail_full.html"
    if source_html.exists():
        if FORCE or not (FIXTURES / "html_detail_full.html").exists():
            shutil.copy(source_html, FIXTURES / "html_detail_full.html")
        print("  html_detail_full.html")
        html = source_html.read_text(encoding="utf-8")
        # A NewCondition variant, to exercise the JSON-LD condition path.
        new_condition = html.replace("UsedCondition", "NewCondition")
        (FIXTURES / "html_detail_new_condition.html").write_text(new_condition, encoding="utf-8")
        print("  html_detail_new_condition.html")
        # No payload at all: JSON-LD only.
        stripped = re.sub(
            r'<script[^>]*id="__NUXT_DATA__"[^>]*>.*?</script>', "", html, flags=re.DOTALL
        )
        (FIXTURES / "html_detail_no_payload.html").write_text(stripped, encoding="utf-8")
        print("  html_detail_no_payload.html")
    else:
        print(f"  ! missing {source_html}")

    (FIXTURES / "html_detail_broken_payload.html").write_text(
        '<html><body><script type="application/json" id="__NUXT_DATA__">'
        "[[not valid json,,,</script></body></html>",
        encoding="utf-8",
    )
    print("  html_detail_broken_payload.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
