"""Merge precedence, provenance, conflicts and delisted handling."""

from __future__ import annotations

import json

from bama_deep.merge import CODES, FIELD_RULES, merge_ad

NOW = 1_800_000_000.0


def result(fields: dict, *, ok: bool = True, **extra) -> dict:
    base = {
        "fields": fields,
        "media": [],
        "raw": [],
        "json_ld": [],
        "payload": None,
        "warnings": [],
        "sha256": "d" * 64,
        "ok": ok,
        "agreements": {},
    }
    base.update(extra)
    return base


def merge(api: dict | None, html: dict | None, **kw):
    return merge_ad(api, html, ad_id="a1", url="https://bama.ir/car/detail-a1-x", now_ts=NOW, **kw)


class TestPrecedence:
    def test_api_wins_for_api_only_fields(self) -> None:
        out = merge(result({"fuel_type": "بنزینی"}), result({}))
        assert out.record["fuel_type"] == "بنزینی"
        assert out.provenance["fuel_type"] == CODES["api"]

    def test_html_wins_for_typed_integers(self) -> None:
        # Both present and disagreeing: the HTML integer is authoritative.
        out = merge(result({"mileage_km": 66000}), result({"mileage_km": 66500}))
        assert out.record["mileage_km"] == 66500
        assert out.provenance["mileage_km"] == CODES["html"]

    def test_html_wins_for_promoted_flag(self) -> None:
        out = merge(result({}), result({"is_promoted": True}))
        assert out.record["is_promoted"] is True

    def test_agreement_is_recorded_as_both(self) -> None:
        out = merge(result({"brand": "peugeot"}), result({"brand": "peugeot"}))
        assert out.provenance["brand"] == CODES["both"]
        assert out.conflicts == []

    def test_absent_everywhere(self) -> None:
        out = merge(result({}), result({}))
        assert out.provenance["fuel_type"] == CODES["none"]
        assert "fuel_type" not in out.record

    def test_every_rule_gets_exactly_one_provenance_entry(self) -> None:
        out = merge(result({}), result({}))
        assert len(out.provenance) == len({r.field for r in FIELD_RULES})


class TestConflicts:
    def test_price_disagreement_is_critical(self) -> None:
        out = merge(result({"price_toman": 1_520_000_000}), result({"price_toman": 1_550_000_000}))
        conflict = next(c for c in out.conflicts if c["field"] == "price_toman")
        assert conflict["severity"] == "critical"
        assert conflict["chosen_source"] == "html"
        assert out.record["conflict_count"] >= 1

    def test_character_variants_are_not_a_conflict(self) -> None:
        # Arabic yeh vs Persian yeh must not read as disagreement.
        out = merge(result({"body_status": "بدون رنگ"}), result({"body_status": "بدون رنگ"}))
        assert out.conflicts == []

    def test_title_punctuation_difference_is_not_a_conflict(self) -> None:
        # The API renders "دنا، پلاس EF7"; the page renders "دنا پلاس EF7".
        out = merge(result({"title": "دنا، پلاس EF7"}), result({"title": "دنا پلاس EF7"}))
        assert [c["field"] for c in out.conflicts] == []

    def test_location_granularity_is_not_a_conflict(self) -> None:
        # API: city / region. Page: city، province، region.
        out = merge(
            result({"location_text": "کرج / فردیس"}),
            result({"location_text": "کرج، البرز، فردیس"}),
        )
        assert [c["field"] for c in out.conflicts] == []

    def test_genuinely_different_location_is_a_conflict(self) -> None:
        out = merge(result({"location_text": "تهران"}), result({"location_text": "اصفهان"}))
        assert any(c["field"] == "location_text" for c in out.conflicts)

    def test_relative_timestamps_are_never_conflicts(self) -> None:
        # "5 ساعت پیش" vs "4 ساعت پیش" is elapsed time, not a defect.
        out = merge(
            result({"published_text": "5 ساعت پیش", "published_ts": 100.0}),
            result({"published_text": "4 ساعت پیش", "published_ts": 200.0}),
        )
        assert [c["field"] for c in out.conflicts] == []

    def test_one_sided_value_is_not_a_conflict(self) -> None:
        out = merge(result({"province": "کرمان"}), result({}))
        assert out.conflicts == []

    def test_critical_conflict_is_surfaced_as_a_warning(self) -> None:
        out = merge(result({"price_toman": 1}), result({"price_toman": 2}))
        assert any("critical_conflict:price_toman" in w for w in out.warnings)


class TestSingleSourceAndDelisted:
    def test_api_only(self) -> None:
        out = merge(result({"fuel_type": "بنزینی"}), None)
        assert out.record["api_ok"] == 1 and out.record["html_ok"] == 0
        assert out.record["parse_status"] == "partial"

    def test_html_only(self) -> None:
        out = merge(None, result({"mileage_km": 1000}))
        assert out.record["api_ok"] == 0 and out.record["html_ok"] == 1
        assert out.record["parse_status"] == "partial"
        assert out.record.get("condition_new_used") is None

    def test_html_present_but_payload_missing_is_partial(self) -> None:
        out = merge(result({"fuel_type": "بنزینی"}), result({}, ok=False))
        assert out.record["html_ok"] == 0
        assert out.record["parse_status"] == "partial"

    def test_both_missing_is_delisted_not_dropped(self) -> None:
        out = merge(None, None)
        assert out.record["is_delisted"] == 1
        assert out.record["delisted_at"]
        assert out.record["parse_status"] == "delisted"
        assert out.record["ad_id"] == "a1"  # still accounted for

    def test_both_ok_is_status_ok(self) -> None:
        out = merge(result({"fuel_type": "x"}), result({"mileage_km": 1}))
        assert out.record["parse_status"] == "ok"


class TestEvidence:
    def test_provenance_json_is_serialized(self) -> None:
        out = merge(result({"fuel_type": "x"}), result({}))
        assert json.loads(out.record["provenance_json"])["fuel_type"] == CODES["api"]

    def test_numeric_agreement_is_carried_through(self) -> None:
        out = merge(result({}), result({}, agreements={"acceleration_s": "exact"}))
        assert json.loads(out.record["numeric_agreement_json"]) == {"acceleration_s": "exact"}

    def test_json_ld_is_scrubbed_defensively(self) -> None:
        # Even if a caller hands over unscrubbed JSON-LD (e.g. replaying legacy
        # payloads), no contact data may reach the stored record.
        leaky = [{"@type": "Product", "description": "تماس 09215650627"}]
        out = merge(result({}), result({}, json_ld=leaky))
        assert "09215650627" not in out.record["json_ld_json"]
        assert "[redacted]" in out.record["json_ld_json"]

    def test_raw_attributes_are_tagged_by_source(self) -> None:
        out = merge(
            result({}, raw=[("detail.fuel", "بنزینی")]),
            result({}, raw=[("vehicle.mileage.value", 66000)]),
        )
        sources = {src for src, _, _ in out.raw}
        assert sources == {"api", "html"}


class TestMediaMerge:
    def test_api_urls_preferred_html_fills_gaps(self) -> None:
        out = merge(
            result({}, media=[{"kind": "image", "position": 0, "original_url": "orig.jpg"}]),
            result({}, media=[{"kind": "image", "position": 0, "alt": "پراید"}]),
        )
        assert len(out.media) == 1
        assert out.media[0]["original_url"] == "orig.jpg"
        assert out.media[0]["alt"] == "پراید"  # HTML-only field carried over
        assert out.media[0]["source"] == "both"

    def test_html_only_video_poster_is_kept(self) -> None:
        out = merge(
            result({}),
            result({}, media=[{"kind": "video", "position": 0, "video_thumb": "poster.jpg"}]),
        )
        assert out.media[0]["video_thumb"] == "poster.jpg"

    def test_positions_are_ordered(self) -> None:
        out = merge(
            result(
                {},
                media=[
                    {"kind": "image", "position": 2, "original_url": "c"},
                    {"kind": "image", "position": 0, "original_url": "a"},
                    {"kind": "image", "position": 1, "original_url": "b"},
                ],
            ),
            result({}),
        )
        assert [m["position"] for m in out.media] == [0, 1, 2]

    def test_no_media(self) -> None:
        assert merge(result({}), result({})).media == []
