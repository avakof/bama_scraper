"""Specs, prices and dealer parsers against real captured fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bama_deep.dealers import parse_dealer_ads, parse_dealer_profile
from bama_deep.prices import parse_price_detail, provider_kind, series_id
from bama_deep.specs import parse_review_detail, parse_review_seo, parse_specification

FIXTURES = Path(__file__).parent / "fixtures"
REVIEW_KEY = "/car-reviews/dena/plusef7p-specs-1481-6mt"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestReviewDetail:
    def test_identifiers_and_scores(self) -> None:
        record = parse_review_detail(load("review_detail.json"), REVIEW_KEY)
        assert record["research_id"] == 2620
        assert record["generation_id"] == 1481
        assert isinstance(record["rating"], (int, float))
        assert record["final_score"] is not None

    def test_pros_and_cons_are_kept(self) -> None:
        record = parse_review_detail(load("review_detail.json"), REVIEW_KEY)
        assert json.loads(record["pros_json"])
        assert json.loads(record["cons_json"])

    def test_price_url_is_carried_for_the_join(self) -> None:
        record = parse_review_detail(load("review_detail.json"), REVIEW_KEY)
        assert record["price_url"].startswith("/price/")

    def test_garbage_does_not_raise(self) -> None:
        assert parse_review_detail({}, REVIEW_KEY) == {}
        assert parse_review_detail(None, REVIEW_KEY) == {}


class TestSpecification:
    @pytest.fixture(scope="class")
    def parsed(self):
        return parse_specification(load("getspecification.json"), REVIEW_KEY)

    def test_thirteen_groups(self, parsed) -> None:
        groups, _, _ = parsed
        assert len(groups) == 13

    def test_roughly_one_hundred_items(self, parsed) -> None:
        _, items, _ = parsed
        assert 90 <= len(items) <= 130, len(items)

    def test_every_group_slug_is_known(self, parsed) -> None:
        # A double-ZWNJ group title ("سیستم‌‌ کمکی راننده") must still map.
        groups, _, _ = parsed
        unknown = [g["group_name"] for g in groups if g["group_slug"].startswith("unk_")]
        assert unknown == [], unknown

    def test_primary_key_is_unique(self, parsed) -> None:
        _, items, _ = parsed
        keys = [(i["review_key"], i["group_slug"], i["item_slug"]) for i in items]
        assert len(keys) == len(set(keys))

    def test_booleans_become_real_integers(self, parsed) -> None:
        _, items, _ = parsed
        booleans = [i for i in items if i["value_type"] == "Boolean"]
        assert booleans, "the fixture should contain Boolean items"
        # The API delivers the STRINGS "true"/"false"; value_bool must be 1/0.
        assert all(i["value_bool"] in (0, 1) for i in booleans)

    def test_measurements_are_split_into_number_and_unit(self, parsed) -> None:
        _, items, _ = parsed
        weight = next((i for i in items if i["item_slug"] == "weight"), None)
        assert weight is not None
        assert weight["value_num"] and weight["value_num"] > 500

    def test_raw_value_is_always_preserved(self, parsed) -> None:
        _, items, _ = parsed
        assert all(i["value_raw"] is not None or i["value_text"] is None for i in items)

    def test_known_feature_slugs_are_present(self, parsed) -> None:
        _, items, _ = parsed
        slugs = {i["item_slug"] for i in items}
        # The point of the whole phase: real feature columns.
        for expected in ("weight", "fuel_tank_capacity", "abs", "airbag_count"):
            assert expected in slugs, f"{expected} missing from {sorted(slugs)[:40]}"

    def test_non_list_payload(self) -> None:
        assert parse_specification({}, REVIEW_KEY) == ([], [], [])


class TestReviewSeo:
    def test_typed_numbers(self) -> None:
        row = parse_review_seo(load("carreviewseo.json"), REVIEW_KEY)
        assert row["weight_kg"] and row["weight_kg"] > 500
        assert row["fuel_capacity_l"] and row["fuel_capacity_l"] > 10
        assert row["engine_power_hp"] and row["engine_power_hp"] > 50

    def test_price_normalized(self) -> None:
        row = parse_review_seo(load("carreviewseo.json"), REVIEW_KEY)
        assert row["price_toman"] is None or row["price_toman"] > 0


class TestPriceDetail:
    @pytest.fixture(scope="class")
    def parsed(self):
        return parse_price_detail(load("price_detail.json"), "dena|plusef7p|6mt")

    def test_multiple_series(self, parsed) -> None:
        assert len(parsed["series"]) >= 2

    def test_series_have_many_daily_points(self, parsed) -> None:
        assert max(s["point_count"] for s in parsed["series"]) > 50

    def test_market_and_factory_providers_distinguished(self, parsed) -> None:
        kinds = {s["provider_kind"] for s in parsed["series"]}
        assert "market" in kinds

    def test_every_point_carries_an_inferred_year(self, parsed) -> None:
        assert all(p["year_inferred"] for p in parsed["points"])

    def test_most_points_resolve_to_a_timestamp(self, parsed) -> None:
        resolved = [p for p in parsed["points"] if p["date_ts"] is not None]
        assert len(resolved) / max(len(parsed["points"]), 1) > 0.9

    def test_series_id_is_stable(self, parsed) -> None:
        again = parse_price_detail(load("price_detail.json"), "dena|plusef7p|6mt")
        assert [s["series_id"] for s in parsed["series"]] == [
            s["series_id"] for s in again["series"]
        ]

    def test_point_primary_key_is_unique(self, parsed) -> None:
        keys = [(p["series_id"], p["point_index"]) for p in parsed["points"]]
        assert len(keys) == len(set(keys))

    def test_prices_are_normalized_integers(self, parsed) -> None:
        priced = [p for p in parsed["points"] if p["price_toman"] is not None]
        assert priced and all(p["price_toman"] > 0 for p in priced)

    def test_two_segment_model_also_parses(self) -> None:
        parsed = parse_price_detail(load("price_detail_two_segment.json"), "shahin|g|")
        assert parsed["series"], "single-trim models must still yield series"
        assert parsed["points"]

    def test_empty_payload(self) -> None:
        empty = parse_price_detail({}, "x|y|z")
        assert empty == {"key": {}, "series": [], "points": []}

    @pytest.mark.parametrize(
        "provider,kind",
        [
            ("قیمت بازار", "market"),
            ("قیمت کارخانه", "factory"),
            ("قیمت نمایندگی", "dealership"),
            ("چیز دیگر", "unknown"),
            (None, "unknown"),
        ],
    )
    def test_provider_kinds(self, provider: str | None, kind: str) -> None:
        assert provider_kind(provider) == kind

    def test_series_id_differs_per_series(self) -> None:
        a = series_id("k", {"trim": "6mt", "model_year": 1405, "class": "x", "price_provider": "p"})
        b = series_id("k", {"trim": "6mt", "model_year": 1404, "class": "x", "price_provider": "p"})
        assert a != b


class TestDealers:
    def test_profile(self) -> None:
        record = parse_dealer_profile(load("corporation.json"))
        assert record["title"]
        assert record["dealer_type"]
        assert record["is_active"] in (0, 1)

    def test_score_is_bounded(self) -> None:
        record = parse_dealer_profile(load("corporation.json"))
        if record.get("score") is not None:
            assert 0.0 <= record["score"] <= 5.0

    def test_inventory_uses_array_length_not_reported_count(self) -> None:
        # metadata.total_count is known-buggy: it reports 0 while returning rows.
        payload = load("corporation_ads.json")
        parsed = parse_dealer_ads(payload, 921, known_codes=set())
        assert parsed["ads_listed"] == len(parsed["rows"]) > 0
        assert parsed["ads_total_count_reported"] == 0
        assert parsed["ads_listed"] != parsed["ads_total_count_reported"]

    def test_seed_membership_flag(self) -> None:
        payload = load("corporation_ads.json")
        codes = {r["ad_code"] for r in parse_dealer_ads(payload, 921, set())["rows"]}
        one = next(iter(codes))
        parsed = parse_dealer_ads(payload, 921, known_codes={one})
        flags = {r["ad_code"]: r["in_seed_inventory"] for r in parsed["rows"]}
        assert flags[one] == 1
        assert any(v == 0 for k, v in flags.items() if k != one)

    def test_prices_and_mileage_normalized(self) -> None:
        rows = parse_dealer_ads(load("corporation_ads.json"), 921, set())["rows"]
        assert any(r["mileage_km"] is not None for r in rows)

    def test_no_phone_in_dealer_rows(self) -> None:
        rows = parse_dealer_ads(load("corporation_ads.json"), 921, set())["rows"]
        assert '"phone"' not in json.dumps(rows, ensure_ascii=False)

    def test_empty_payloads(self) -> None:
        assert parse_dealer_profile({}) == {}
        assert parse_dealer_ads({}, 1, set())["ads_listed"] == 0
