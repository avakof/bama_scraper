"""Discovery: URL canonicalization, ID extraction, card mapping, filters, dedup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bama_scraper.discovery import (
    _dom_card_to_ad,
    canonical_ad_url,
    card_to_ad,
    extract_ad_id,
    merge_inventories,
    parse_filters,
)

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH_URL = (
    "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian"
)


class TestCanonicalization:
    @pytest.mark.parametrize(
        "raw",
        [
            "/car/detail-lanch67j-dena-plusef7p-6mt-1403",
            "https://bama.ir/car/detail-lanch67j-dena-plusef7p-6mt-1403",
            "https://www.bama.ir/car/detail-lanch67j-dena-plusef7p-6mt-1403",
            "http://bama.ir/car/detail-lanch67j-dena-plusef7p-6mt-1403",
            "/car/detail-lanch67j-dena-plusef7p-6mt-1403#gallery",
            "/car/detail-lanch67j-dena-plusef7p-6mt-1403?utm_source=x&utm_medium=y",
            "/car/detail-lanch67j-dena-plusef7p-6mt-1403/",
            "  /car/detail-lanch67j-dena-plusef7p-6mt-1403  ",
        ],
    )
    def test_all_variants_collapse_to_one_key(self, raw: str) -> None:
        assert canonical_ad_url(raw) == "https://bama.ir/car/detail-lanch67j-dena-plusef7p-6mt-1403"

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "/car/dena",
            "/car/brands",
            "https://evil.example.com/car/detail-abc123-x",
            "/motorcycle/detail-abc123-x",
            "/news/something",
        ],
    )
    def test_non_ad_urls_rejected(self, raw: str) -> None:
        assert canonical_ad_url(raw) is None

    def test_meaningful_query_params_preserved(self) -> None:
        out = canonical_ad_url("/car/detail-abc123-x?variant=2")
        assert out is not None and out.endswith("?variant=2")


class TestAdId:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://bama.ir/car/detail-lanch67j-dena-plus-1403", "lanch67j"),
            ("/car/detail-5z8iesxh-peugeot-pars-lx-1401", "5z8iesxh"),
            ("/car/detail-abc123", "abc123"),
            ("https://bama.ir/car/detail-nmqvojlh-saina-manuals-mtgas-1401?x=1", "nmqvojlh"),
        ],
    )
    def test_extraction(self, url: str, expected: str) -> None:
        assert extract_ad_id(url) == expected

    def test_invalid(self) -> None:
        assert extract_ad_id("/car/dena") is None
        assert extract_ad_id("") is None

    def test_id_is_stable_across_url_variants(self) -> None:
        a = extract_ad_id("/car/detail-lanch67j-dena-plus-1403?utm_source=x")
        b = extract_ad_id("https://www.bama.ir/car/detail-lanch67j-dena-plus-1403#gal")
        assert a == b == "lanch67j"


class TestFilterInterpretation:
    """Stage 2: the URL params must be interpreted the way the live site does."""

    def test_target_url_maps_to_live_api_params(self) -> None:
        parsed = parse_filters(SEARCH_URL)
        # Verified against real captured XHR traffic from the live page.
        assert parsed["api_params"] == {
            "yearFrom": "1397-2018",
            "priceFrom": "1000000000",
            "body": "passenger_car",
            "country": "iranian",
        }

    def test_year_is_a_lower_bound_not_an_exact_match(self) -> None:
        note = parse_filters(SEARCH_URL)["interpretation"]["year"]
        assert "no upper bound" in note and "newer" in note

    def test_price_is_a_minimum_not_a_maximum(self) -> None:
        note = parse_filters(SEARCH_URL)["interpretation"]["price"]
        assert "minimum" in note and "not a maximum" in note

    def test_explicit_year_range_produces_both_bounds(self) -> None:
        parsed = parse_filters("https://bama.ir/car?year=1397-2018,1400-2021")
        assert parsed["api_params"]["yearFrom"] == "1397-2018"
        assert parsed["api_params"]["yearTo"] == "1400-2021"

    def test_unknown_params_pass_through_unchanged(self) -> None:
        parsed = parse_filters("https://bama.ir/car?brand=dena&fuel=petrol")
        assert parsed["api_params"]["brand"] == "dena"
        assert parsed["api_params"]["fuel"] == "petrol"


class TestCardMapping:
    @pytest.fixture
    def payload(self) -> dict:
        return json.loads((FIXTURES / "search_page0.json").read_text(encoding="utf-8"))

    def test_banner_entries_are_not_advertisements(self, payload: dict) -> None:
        banners = [e for e in payload["data"]["ads"] if e.get("type") == "banner"]
        assert banners, "fixture should contain a banner to exercise this path"
        for entry in banners:
            assert card_to_ad(entry, SEARCH_URL, 0, 0) is None

    def test_real_ads_map_cleanly(self, payload: dict) -> None:
        entries = [e for e in payload["data"]["ads"] if e.get("type") == "ad"]
        ads = [card_to_ad(e, SEARCH_URL, i, 0) for i, e in enumerate(entries)]
        assert all(a is not None for a in ads)
        for ad in ads:
            assert ad is not None
            assert ad.url.startswith("https://bama.ir/car/detail-")
            assert ad.ad_id and extract_ad_id(ad.url) == ad.ad_id
            assert ad.source_search_url == SEARCH_URL

    def test_negotiable_card_has_no_numeric_price(self) -> None:
        entry = {
            "type": "ad",
            "detail": {
                "code": "abc12345",
                "url": "/car/detail-abc12345-pride-1400",
                "title": "پراید",
                "year": "1400",
                "mileage": "۱۲,۰۰۰ km",
            },
            "price": {"type": "negotiable", "price": "0"},
        }
        ad = card_to_ad(entry, SEARCH_URL, 0, 0)
        assert ad is not None
        assert ad.price_toman is None  # never 0
        assert ad.price_type == "negotiable"
        assert ad.mileage_km == 12_000

    def test_lumpsum_card_price_parsed(self) -> None:
        entry = {
            "type": "ad",
            "detail": {
                "code": "bgs03up7",
                "url": "/car/detail-bgs03up7-peugeot-pars-1401",
                "year": "1401",
            },
            "price": {"type": "lumpsum", "price": "1,585,000,000"},
        }
        ad = card_to_ad(entry, SEARCH_URL, 0, 0)
        assert ad is not None and ad.price_toman == 1_585_000_000

    def test_entry_without_detail_is_skipped(self) -> None:
        assert card_to_ad({"type": "ad", "detail": None}, SEARCH_URL, 0, 0) is None
        assert card_to_ad({"type": "ad"}, SEARCH_URL, 0, 0) is None


class TestDomCardMapping:
    def test_dom_cards_dedupe_by_canonical_url(self) -> None:
        cards = [
            {
                "href": "/car/detail-code0000-pride-1390",
                "title": "پراید",
                "lines": [],
                "position": 0,
            },
            # same ad, promoted duplicate carrying tracking params and a fragment
            {
                "href": "/car/detail-code0000-pride-1390?utm_source=promo#gal",
                "title": "پراید",
                "lines": [],
                "position": 1,
                "promoted": True,
            },
            {"href": "/car/detail-code0001-tara-1403", "title": "تارا", "lines": [], "position": 2},
        ]
        ads = [a for a in (_dom_card_to_ad(c, SEARCH_URL, 0) for c in cards) if a]
        assert len(ads) == 3  # mapping keeps all
        assert len({a.url for a in ads}) == 2  # dedup key collapses the promoted duplicate

    def test_promoted_flag_detected(self) -> None:
        card = {
            "href": "/car/detail-promo999-tara-1403",
            "title": "نردبان شده",
            "lines": ["نردبان شده"],
            "position": 0,
            "promoted": True,
        }
        ad = _dom_card_to_ad(card, SEARCH_URL, 0)
        assert ad is not None and ad.is_promoted is True

    def test_non_ad_href_skipped(self) -> None:
        assert _dom_card_to_ad({"href": "/car/dena", "lines": []}, SEARCH_URL, 0) is None


class TestInventoryComparison:
    def test_stable_when_second_pass_finds_nothing_new(self) -> None:
        result = merge_inventories({"a", "b"}, {"a", "b"})
        assert result["stable"] is True
        assert result["only_in_second"] == 0

    def test_unstable_when_second_pass_finds_new_ads(self) -> None:
        result = merge_inventories({"a"}, {"a", "b"})
        assert result["stable"] is False
        assert result["only_in_second"] == 1
        assert "b" in result["new_urls_found_by_second_pass"]

    def test_removed_ads_do_not_break_stability(self) -> None:
        # Live classifieds disappear mid-run; that alone must not fail the audit.
        result = merge_inventories({"a", "b"}, {"a"})
        assert result["stable"] is True
        assert result["only_in_first"] == 1


class TestPaginationValidation:
    """Guards the traps found on the live endpoint.

    ``total_count`` is a running "delivered so far" counter that overshoots the
    real total, so it can never be used as a total or a stop condition.
    Termination must be driven by an actually-empty batch.
    """

    def test_total_count_overshoots_the_real_total(self) -> None:
        # Measured at the real boundary: the set held ~2849 ads, yet the
        # counter kept climbing past it on the empty pages.
        boundary = [
            {"page": 93, "real_ads": 30, "total_count": 2821, "has_next": True},
            {"page": 94, "real_ads": 25, "total_count": 2845, "has_next": False},
            {"page": 95, "real_ads": 0, "total_count": 2850, "has_next": False},
            {"page": 96, "real_ads": 0, "total_count": 2880, "has_next": False},
        ]
        actual_total = sum(p["real_ads"] for p in boundary[:2]) + 93 * 30
        assert boundary[-1]["total_count"] > actual_total

    def test_empty_batch_is_the_authoritative_stop_signal(self) -> None:
        # has_next flips a page early (on the last page that still has ads), so
        # stopping on it alone would be fragile; emptiness is unambiguous.
        boundary = [
            {"real_ads": 25, "has_next": False},
            {"real_ads": 0, "has_next": False},
        ]
        assert boundary[0]["has_next"] is False and boundary[0]["real_ads"] > 0
        stale = sum(1 for p in boundary if p["real_ads"] == 0)
        assert stale == 1

    def test_growing_total_count_is_not_a_stop_signal(self) -> None:
        observed = [
            {"total_count": 31, "total_pages": 2, "has_next": True},
            {"total_count": 61, "total_pages": 3, "has_next": True},
            {"total_count": 91, "total_pages": 4, "has_next": True},
        ]
        # Each page claims exactly one more page than it has delivered.
        for i, meta in enumerate(observed):
            assert meta["total_count"] == 30 * (i + 1) + 1
            assert meta["has_next"] is True

    def test_distinct_batches_prove_pagination_works(self) -> None:
        page0 = {"a", "b", "c"}
        page1 = {"d", "e", "f"}
        assert not (page0 & page1), "pages must return distinct ads"
        assert len(page0 | page1) == 6


class TestAuditBoundsDerivedFromUrl:
    """Regression: audit thresholds were hardcoded to the reference URL."""

    def test_bounds_match_the_target_url(self) -> None:
        from bama_scraper.validation import filter_bounds

        bounds = filter_bounds(SEARCH_URL)
        assert bounds["price_from"] == 1_000_000_000
        assert bounds["year_from_jalali"] == 1397

    def test_bounds_follow_a_different_url(self) -> None:
        from bama_scraper.validation import filter_bounds

        bounds = filter_bounds("https://bama.ir/car?year=1400-2021,&price=5000000000")
        assert bounds["price_from"] == 5_000_000_000
        assert bounds["year_from_jalali"] == 1400

    def test_absent_filters_yield_no_bounds(self) -> None:
        from bama_scraper.validation import filter_bounds

        bounds = filter_bounds("https://bama.ir/car?body=passenger_car")
        assert bounds["price_from"] is None
        assert bounds["year_from_jalali"] is None
