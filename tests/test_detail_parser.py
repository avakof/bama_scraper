"""Detail-page parsing against local sanitized fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bama_scraper.detail_parser import (
    extract_json_ld,
    extract_nuxt_payload,
    parse_detail,
    resolve_devalue,
    strip_private,
)

FIXTURES = Path(__file__).parent / "fixtures"
FULL_URL = "https://bama.ir/car/detail-lanch67j-dena-plusef7p-6mt-1403"


@pytest.fixture
def full_html() -> str:
    return (FIXTURES / "detail_full.html").read_text(encoding="utf-8")


class TestDevalueResolver:
    def test_resolves_index_references(self) -> None:
        nodes = [{"a": 1, "b": 2}, "hello", 42]
        assert resolve_devalue(nodes) == {"a": "hello", "b": 42}

    def test_negative_sentinels_become_none(self) -> None:
        # -1 is devalue's UNDEFINED marker, not an index from the end.
        assert resolve_devalue([{"a": -1}]) == {"a": None}

    def test_unwraps_reactive_wrappers(self) -> None:
        nodes = [["ShallowReactive", 1], {"x": 2}, "v"]
        assert resolve_devalue(nodes) == {"x": "v"}

    def test_shared_subtrees_resolve_consistently(self) -> None:
        nodes = [{"a": 1, "b": 1}, "shared"]
        assert resolve_devalue(nodes) == {"a": "shared", "b": "shared"}

    def test_self_reference_does_not_hang(self) -> None:
        assert resolve_devalue([{"self": 0}]) is not None

    def test_plain_arrays_of_references(self) -> None:
        assert resolve_devalue([[1, 2], "x", "y"]) == ["x", "y"]


class TestPrivacy:
    def test_phone_keys_removed_recursively(self) -> None:
        payload = {"content": {"phone": "۰۹۱۲۰۷۵۷۵XX", "title": "x"}, "list": [{"mobile": "0912"}]}
        cleaned = strip_private(payload)
        assert "phone" not in cleaned["content"]
        assert "mobile" not in cleaned["list"][0]
        assert cleaned["content"]["title"] == "x"

    def test_parsed_record_contains_no_contact_fields(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        blob = json.dumps(detail.model_dump(), ensure_ascii=False)
        for key in ("phone", "mobile", "telephone"):
            assert f'"{key}"' not in blob


class TestFullPayload:
    def test_nuxt_payload_found(self, full_html: str) -> None:
        payload = extract_nuxt_payload(full_html)
        assert payload is not None and payload.get("code") == "lanch67j"

    def test_json_ld_blocks(self, full_html: str) -> None:
        blocks = extract_json_ld(full_html)
        assert len(blocks) >= 2
        assert any(b.get("@type") == "Product" for b in blocks)

    def test_core_identity_fields(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.ad_id == "lanch67j"
        assert detail.parse_status == "ok"
        assert detail.brand == "dena"
        assert detail.brand_fa == "دنا"
        assert detail.year_jalali == 1403
        assert detail.year_gregorian == 2024
        assert detail.body_type == "passenger_car"
        assert detail.canonical_url == FULL_URL

    def test_vehicle_attributes(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.mileage_km == 40_000
        assert detail.is_zero_km is False
        assert detail.transmission == "دنده ای"
        assert detail.body_color == "سفید"
        assert detail.interior_color == "مارون"
        assert detail.body_status == "بدون رنگ"
        assert detail.engine == "4 سیلندر EF7P"
        assert detail.drivetrain == "دیفرانسیل جلو"

    def test_location_split(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.province == "تهران"
        assert detail.city == "تهران"
        assert detail.neighbourhood == "بلوار فردوس"

    def test_negotiable_price_is_null_not_zero(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.price_type == "negotiable"
        assert detail.price_toman is None
        assert detail.is_negotiable is True

    def test_media_extracted_in_order(self, full_html: str) -> None:
        detail, media = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.media_count == 3
        assert len(detail.image_urls) == 3
        assert len(media) == 3
        assert [m.position for m in media] == [0, 1, 2]
        # full-resolution URL preferred over the resized variant
        assert all("_thumb_" not in u for u in detail.image_urls)
        assert detail.primary_image_url == detail.image_urls[0]

    def test_persian_text_preserved_alongside_normalized_values(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.mileage_text == "40,000 km"  # raw kept
        assert detail.mileage_km == 40_000  # normalized kept
        assert detail.year_text == "1403"

    def test_raw_attributes_populated(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.raw_attributes
        assert "phone" not in json.dumps(detail.raw_attributes)

    def test_evidence_recorded(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        assert detail.html_sha256 and len(detail.html_sha256) == 64
        assert detail.scraper_version and detail.scraped_at
        assert detail.source == "nuxt_data"

    def test_parse_source_and_fetch_source_are_distinct(self, full_html: str) -> None:
        # Regression: the transport used to overwrite `source`, destroying the
        # record of which evidence source the fields actually came from.
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j")
        detail.fetch_source = "httpx"
        assert detail.source == "nuxt_data"
        assert detail.fetch_source == "httpx"

    def test_published_timestamp_normalized(self, full_html: str) -> None:
        detail, _ = parse_detail(full_html, FULL_URL, "lanch67j", now_ts=5000.0)
        assert detail.published_text is not None
        assert detail.published_ts == 5000.0

    def test_deterministic(self, full_html: str) -> None:
        a, _ = parse_detail(full_html, FULL_URL, "lanch67j", now_ts=1.0)
        b, _ = parse_detail(full_html, FULL_URL, "lanch67j", now_ts=1.0)
        assert a.model_dump() == b.model_dump()


class TestDegradedPages:
    def test_minimal_page_still_yields_a_record(self) -> None:
        html = (FIXTURES / "detail_minimal.html").read_text(encoding="utf-8")
        detail, media = parse_detail(
            html, "https://bama.ir/car/detail-zz00zz99-pride-1397", "zz00zz99"
        )
        assert detail.ad_id == "zz00zz99"
        assert detail.parse_status == "partial"
        assert detail.price_toman is None  # missing price
        assert detail.image_urls == []  # missing images
        assert detail.description is None  # missing description
        assert media == []
        assert any("payload missing" in w for w in detail.parse_warnings)

    def test_missing_media_warning(self) -> None:
        html = (FIXTURES / "detail_minimal.html").read_text(encoding="utf-8")
        detail, _ = parse_detail(html, "https://bama.ir/car/detail-zz00zz99-x", "zz00zz99")
        assert any("no images" in w for w in detail.parse_warnings)

    def test_broken_payload_does_not_raise(self) -> None:
        html = (FIXTURES / "detail_broken.html").read_text(encoding="utf-8")
        detail, _ = parse_detail(html, "https://bama.ir/car/detail-bad00001-x", "bad00001")
        assert detail.parse_status == "partial"
        assert detail.ad_id == "bad00001"

    def test_empty_html(self) -> None:
        detail, media = parse_detail("", "https://bama.ir/car/detail-none0001-x", "none0001")
        assert detail.parse_status == "partial"
        assert media == []

    def test_malformed_attribute_rows_are_tolerated(self) -> None:
        # value wrappers with unexpected shapes must not crash the flattener
        nodes = [
            {"data": 1},
            {"get-ad-pdp-car_x": 2},
            {"data": 3},
            {"vehicle": 4, "specs": 7, "content": 9},
            {"mileage": 5, "year": 6},
            ["unexpected", "list", "shape"],
            {"value": "not-a-year"},
            {"details": 8},
            {"engine": -1},
            {"title": 10},
            "عنوان",
        ]
        html = (
            '<html><body><script type="application/json" id="__NUXT_DATA__">'
            + json.dumps(nodes)
            + "</script></body></html>"
        )
        detail, _ = parse_detail(html, "https://bama.ir/car/detail-odd00001-x", "odd00001")
        assert detail.ad_id == "odd00001"
        assert detail.year_jalali is None  # unparseable year -> null, not a crash


class TestPriceVariants:
    def _html(self, price: dict) -> str:
        nodes = [{"data": 1}, {"get-ad-pdp-car_x": 2}, {"data": 3}, {"price": 4}, price]
        return (
            '<html><body><script type="application/json" id="__NUXT_DATA__">'
            + json.dumps(nodes, ensure_ascii=False)
            + "</script></body></html>"
        )

    def test_lumpsum(self) -> None:
        html = self._html(
            {"type": "lumpsum", "fixed": {"value": 1_585_000_000, "text": "1,585,000,000"}}
        )
        detail, _ = parse_detail(html, "https://bama.ir/car/detail-p1-x", "p1")
        assert detail.price_toman == 1_585_000_000
        assert detail.price_text == "1,585,000,000"
        assert detail.is_negotiable is False

    def test_negotiable(self) -> None:
        detail, _ = parse_detail(
            self._html({"type": "negotiable"}), "https://bama.ir/car/detail-p2-x", "p2"
        )
        assert detail.price_toman is None and detail.is_negotiable is True

    def test_installment(self) -> None:
        html = self._html(
            {
                "type": "installment",
                "installment": {
                    "prepayment": {"value": 500_000_000, "text": "500,000,000"},
                    "payment": {"value": 25_000_000, "text": "25,000,000"},
                    "month": 36,
                },
            }
        )
        detail, _ = parse_detail(html, "https://bama.ir/car/detail-p3-x", "p3")
        assert detail.is_installment is True
        assert detail.down_payment_toman == 500_000_000
        assert detail.installment_amount_toman == 25_000_000
        assert detail.installment_months == 36

    def test_currency_recorded(self) -> None:
        detail, _ = parse_detail(
            self._html({"type": "negotiable"}), "https://bama.ir/car/detail-p4-x", "p4"
        )
        assert detail.price_currency == "IRT"
