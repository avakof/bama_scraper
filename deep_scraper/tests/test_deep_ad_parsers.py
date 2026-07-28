"""Advertisement parsers against real, offline fixtures.

Fixtures are built by ``tools/make_fixtures.py`` from data already on disk; no
test in this file touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bama_deep.ad_api import parse_ad_api
from bama_deep.ad_html import condition_from_json_ld, parse_ad_html
from bama_deep.scrub import assert_no_contact

FIXTURES = Path(__file__).parent / "fixtures"
NOW = 1_800_000_000.0


def api(name: str) -> dict:
    return json.loads((FIXTURES / f"api_detail_{name}.json").read_text(encoding="utf-8"))


def html(name: str) -> str:
    return (FIXTURES / f"html_detail_{name}.html").read_text(encoding="utf-8")


def parse_api(name: str) -> dict:
    return parse_ad_api(api(name), now_ts=NOW)


class TestApiCoreFields:
    def test_identity(self) -> None:
        fields = parse_api("personal")["fields"]
        assert fields["code"]
        assert fields["ad_type"] == "car"
        assert fields["title"]
        assert fields["year_jalali"] and fields["year_gregorian"]

    def test_fuel_type_is_present(self) -> None:
        # The headline win: `fuel` exists ONLY in the JSON API, which is why the
        # original scraper's fuel_type column was 0% populated.
        assert parse_api("personal")["fields"]["fuel_type"] in (
            "بنزینی",
            "دوگانه سوز",
            "هیبریدی",
            "برقی",
        )

    def test_province_is_present(self) -> None:
        assert parse_api("personal")["fields"].get("province")

    def test_price_and_mileage_normalized(self) -> None:
        fields = parse_api("personal")["fields"]
        assert isinstance(fields["price_toman"], int) and fields["price_toman"] > 0
        assert isinstance(fields["mileage_km"], int)
        assert fields["mileage_text"]  # raw text kept beside the number

    def test_join_keys_derived(self) -> None:
        fields = parse_api("personal")["fields"]
        assert fields["review_key"].startswith("/car-reviews/")

    def test_seller_type_personal(self) -> None:
        fields = parse_api("personal")["fields"]
        assert fields["seller_type"] == "personal"
        assert "dealer_id" not in fields

    def test_media_extracted(self) -> None:
        result = parse_api("personal")
        images = [m for m in result["media"] if m["kind"] == "image"]
        assert images
        assert result["fields"]["media_count"] == len(images)
        assert result["fields"]["primary_image_url"]

    def test_raw_attributes_captured(self) -> None:
        assert parse_api("personal")["raw"]

    def test_deterministic(self) -> None:
        a = parse_ad_api(api("personal"), now_ts=NOW)
        b = parse_ad_api(api("personal"), now_ts=NOW)
        assert a["fields"] == b["fields"]
        assert a["sha256"] == b["sha256"]


class TestApiDealer:
    def test_dealer_fields(self) -> None:
        fields = parse_api("dealer")["fields"]
        assert fields["seller_type"] == "nonpersonal"
        assert isinstance(fields["dealer_id"], int)
        assert fields["dealer_name"]

    def test_dealer_score_is_within_range(self) -> None:
        fields = parse_api("dealer")["fields"]
        score = fields.get("dealer_score")
        if score is not None:
            assert 0.0 <= score <= 5.0


class TestApiZeroSentinels:
    """Bama writes 0 for inapplicable installment fields; 0 is not a value."""

    def test_lumpsum_has_no_installment_numbers(self) -> None:
        fields = parse_api("personal")["fields"]
        for key in (
            "down_payment_toman",
            "prepayment_secondary_toman",
            "installment_amount_toman",
            "installment_months",
            "installments_count",
            "delivery_days",
        ):
            assert fields.get(key) is None, key

    def test_installment_fixture_populates_them(self) -> None:
        fields = parse_api("installment")["fields"]
        assert fields["is_installment"] is True
        assert fields["down_payment_toman"] == 1_010_000_000
        assert fields["installment_amount_toman"] == 63_000_000
        assert fields["installment_months"] == 2
        assert fields["installments_count"] == 24
        assert fields["delivery_days"] == 1
        # A "0" secondary prepayment must stay absent, not become 0.
        assert fields.get("prepayment_secondary_toman") is None


class TestApiEdgeCases:
    def test_no_images(self) -> None:
        result = parse_api("no_images")
        assert [m for m in result["media"] if m["kind"] == "image"] == []
        assert result["fields"]["media_count"] == 0
        assert result["fields"].get("primary_image_url") is None
        assert any("no images" in w for w in result["warnings"])

    def test_ev_fields_and_null_specs(self) -> None:
        fields = parse_api("ev")["fields"]
        assert fields["fuel_type"] == "برقی"
        assert fields["battery_capacity_text"]
        assert fields["all_electric_range_text"]
        # A null spec must simply be absent rather than crashing the parse.
        assert fields.get("engine_volume_text") is None

    def test_presale(self) -> None:
        fields = parse_api("presale")["fields"]
        assert fields["is_pre_sale"] is True
        assert fields["pre_sale_delivery"] == "کمتر از 3 ماه"
        assert fields["is_zero_km"] is True

    def test_no_review_url(self) -> None:
        fields = parse_api("no_review_url")["fields"]
        assert fields.get("review_key") is None

    def test_two_segment_price_url_still_yields_a_key(self) -> None:
        fields = parse_api("two_segment_price")["fields"]
        assert fields.get("price_key"), "single-trim models must still join to prices"

    def test_video(self) -> None:
        videos = [m for m in parse_api("video")["media"] if m["kind"] == "video"]
        assert videos and videos[0]["video_url"]

    def test_empty_envelope_does_not_raise(self) -> None:
        result = parse_ad_api(api("empty"), now_ts=NOW)
        assert result["fields"] == {}
        assert result["warnings"]

    def test_garbage_payload_does_not_raise(self) -> None:
        assert parse_ad_api({}, now_ts=NOW)["fields"] == {}


class TestApiPrivacy:
    def test_phone_key_removed(self) -> None:
        result = parse_api("phone_present")
        blob = json.dumps(result, ensure_ascii=False, default=str)
        assert '"phone"' not in blob

    def test_description_phone_redacted(self) -> None:
        fields = parse_api("phone_present")["fields"]
        assert "[redacted]" in fields["description"]
        assert "09120001122" not in fields["description"]
        assert fields["description_redactions"] >= 1

    def test_telegram_handle_redacted(self) -> None:
        assert "@fake_dealer_test" not in parse_api("phone_present")["fields"]["description"]

    def test_meta_description_also_redacted(self) -> None:
        # Bama builds its SEO description from the seller's text, so the phone
        # number appears there too -- verified on live ad cuaxpwpu.
        meta = parse_api("phone_present")["fields"]["meta_description"]
        assert "09120001122" not in meta

    def test_whole_result_passes_the_contact_gate(self) -> None:
        fields = parse_api("phone_present")["fields"]
        text_only = {k: v for k, v in fields.items() if isinstance(v, str) and "sha256" not in k}
        assert assert_no_contact(json.dumps(text_only, ensure_ascii=False)) == []

    def test_raw_attributes_carry_no_phone(self) -> None:
        raw = parse_api("phone_present")["raw"]
        assert not [k for k, _ in raw if "phone" in k.lower()]


class TestHtmlParser:
    def test_typed_numerics(self) -> None:
        fields = parse_ad_html(html("full"), now_ts=NOW)["fields"]
        assert fields["mileage_km"] == 40_000
        assert fields["year_jalali"] == 1403

    def test_dot_stripped_numerics_resolved_from_text(self) -> None:
        result = parse_ad_html(html("full"), now_ts=NOW)
        fields, agreements = result["fields"], result["agreements"]
        assert fields["engine_volume_l"] == 1.7
        assert agreements["engine_volume_l"] == "scaled_x10"
        # 12 stays 12 seconds; a fixed /10 would have produced 1.2.
        assert fields["acceleration_s"] == 12.0
        assert agreements["acceleration_s"] == "exact"
        assert fields["power_hp"] == 113.0
        assert fields["acceleration_raw"] == 12  # checksum retained

    def test_promoted_flag_from_media_badge(self) -> None:
        fields = parse_ad_html(html("full"), now_ts=NOW)["fields"]
        assert fields["is_promoted"] is False
        assert fields["badge"] is False

    def test_stats_and_slogan_are_html_only(self) -> None:
        fields = parse_ad_html(html("full"), now_ts=NOW)["fields"]
        assert isinstance(fields["stats_model_ad_count"], int)
        assert fields["meta_slogan"]

    def test_condition_from_json_ld(self) -> None:
        assert parse_ad_html(html("full"), now_ts=NOW)["fields"]["condition_new_used"] == "used"

    def test_new_condition_variant(self) -> None:
        result = parse_ad_html(html("new_condition"), now_ts=NOW)
        assert result["fields"]["condition_new_used"] == "new"
        # New but with 40,000 km recorded -> flagged, not silently accepted.
        assert any("condition_mileage_conflict" in w for w in result["warnings"])

    def test_negotiable_price_is_null_not_zero(self) -> None:
        fields = parse_ad_html(html("full"), now_ts=NOW)["fields"]
        assert fields["price_type"] == "negotiable"
        assert fields.get("price_toman") is None
        assert fields["is_negotiable"] is True

    def test_missing_payload_degrades_to_json_ld(self) -> None:
        result = parse_ad_html(html("no_payload"), now_ts=NOW)
        assert result["ok"] is False
        assert result["fields"]["condition_new_used"] == "used"  # JSON-LD survives
        assert any("payload missing" in w for w in result["warnings"])

    def test_broken_payload_does_not_raise(self) -> None:
        result = parse_ad_html(html("broken_payload"), now_ts=NOW)
        assert result["ok"] is False

    def test_empty_html(self) -> None:
        assert parse_ad_html("", now_ts=NOW)["ok"] is False

    def test_html_carries_no_phone(self) -> None:
        result = parse_ad_html(html("full"), now_ts=NOW)
        assert '"phone"' not in json.dumps(result["payload"], ensure_ascii=False)

    def test_raw_keeps_both_value_and_text(self) -> None:
        # Unlike the original flattener, the machine value is preserved too.
        keys = {k for k, _ in parse_ad_html(html("full"), now_ts=NOW)["raw"]}
        assert any(k.endswith(".value") for k in keys)
        assert any(k.endswith(".text") for k in keys)


class TestJsonLdCondition:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("https://schema.org/UsedCondition", "used"),
            ("https://schema.org/NewCondition", "new"),
            ("https://schema.org/RefurbishedCondition", "refurbished"),
        ],
    )
    def test_mapping(self, value: str, expected: str) -> None:
        assert condition_from_json_ld([{"offers": {"itemCondition": value}}]) == expected

    def test_absent(self) -> None:
        assert condition_from_json_ld([]) is None
        assert condition_from_json_ld([{"offers": {}}]) is None
        assert condition_from_json_ld([{"@type": "WebPage"}]) is None


class TestCodedEnumFields:
    """Fields that arrive as ``{"value", "display_name"}`` rather than a string.

    Found in production: four pre-sale listings failed their daily detail check
    with ``TypeError: normalize() argument 2 must be str, not dict`` because
    ``pre_sale_delivery`` was handed straight to a text normalizer.
    """

    def _payload(self, delivery: object) -> dict:
        payload = api("personal")
        payload["data"]["detail"]["pre_sale_delivery"] = delivery
        return payload

    def test_object_form_yields_label_and_code(self) -> None:
        fields = parse_ad_api(
            self._payload({"value": "SixToNineMonths", "display_name": "تحویل 6 تا 9 ماه"}),
            now_ts=NOW,
        )["fields"]
        assert fields["pre_sale_delivery"] == "تحویل 6 تا 9 ماه"
        assert fields["pre_sale_delivery_value"] == "SixToNineMonths"

    def test_bare_string_form_still_parses(self) -> None:
        """The same field arrives as a plain string on some listings."""
        fields = parse_ad_api(self._payload("تحویل روز"), now_ts=NOW)["fields"]
        assert fields["pre_sale_delivery"] == "تحویل روز"
        assert fields["pre_sale_delivery_value"] == "تحویل روز"

    def test_absent_field_is_omitted_not_an_error(self) -> None:
        """The parser drops empty fields, so absence is absence — not a raise."""
        fields = parse_ad_api(self._payload(None), now_ts=NOW)["fields"]
        assert "pre_sale_delivery" not in fields
        assert "pre_sale_delivery_value" not in fields

    def test_delivery_is_recorded_even_when_not_a_pre_sale(self) -> None:
        """`is_pre_sale=False` with a same-day delivery code is a real observed shape."""
        payload = self._payload({"value": "SameDay", "display_name": "تحویل روز"})
        payload["data"]["detail"]["is_pre_sale"] = False
        fields = parse_ad_api(payload, now_ts=NOW)["fields"]
        assert fields["is_pre_sale"] is False
        assert fields["pre_sale_delivery_value"] == "SameDay"

    def test_body_uses_the_same_shape(self) -> None:
        fields = parse_api("personal")["fields"]
        assert fields["body_value"] and fields["body_display_name"]
