"""Persian/Arabic normalization tests."""

from __future__ import annotations

import pytest

from bama_scraper.normalization import (
    convert_digits,
    is_zero_km,
    normalize_chars,
    normalize_mileage,
    normalize_price,
    normalize_text,
    normalize_whitespace,
    parse_jalali_date,
    parse_published_time,
    parse_relative_time,
    parse_year,
)


class TestDigits:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),  # Persian
            ("٠١٢٣٤٥٦٧٨٩", "0123456789"),  # Arabic-Indic
            ("۱۴۰۳", "1403"),
            ("1,585,000,000", "1,585,000,000"),  # ASCII untouched
            ("قیمت ۱۲۳ تومان", "قیمت 123 تومان"),  # letters preserved
            ("", ""),
        ],
    )
    def test_convert_digits(self, raw: str, expected: str) -> None:
        assert convert_digits(raw) == expected

    def test_mixed_scripts(self) -> None:
        assert convert_digits("۱٢3") == "123"


class TestChars:
    def test_arabic_yeh_and_kaf_folded_to_persian(self) -> None:
        assert normalize_chars("كيف") == "کیف"

    def test_diacritics_stripped(self) -> None:
        assert normalize_chars("مَشهَد") == "مشهد"

    def test_nbsp_and_zero_width_collapse(self) -> None:
        assert normalize_whitespace("تهران ​شمال") == "تهران شمال"

    def test_zwnj_preserved_by_default(self) -> None:
        # ZWNJ is orthographically meaningful; it must survive normalization.
        assert "‌" in normalize_whitespace("می‌رود")

    def test_zwnj_removable_on_request(self) -> None:
        assert "‌" not in normalize_whitespace("می‌رود", keep_zwnj=False)

    def test_normalize_text_blank_becomes_none(self) -> None:
        assert normalize_text("   ​ ") is None
        assert normalize_text(None) is None

    def test_original_text_not_mutated(self) -> None:
        original = "كيف ۱۲۳"
        normalize_text(original)
        assert original == "كيف ۱۲۳"


class TestPrice:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("۱٬۲۵۰٬۰۰۰٬۰۰۰ تومان", 1_250_000_000),
            ("1,585,000,000", 1_585_000_000),
            ("۱۲۳۴۵", 12345),
            (1_585_000_000, 1_585_000_000),
        ],
    )
    def test_valid_prices(self, raw: object, expected: int) -> None:
        assert normalize_price(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["توافقی", "تماس بگیرید", "", None, "0", 0, "قیمت مخفی", "اقساطی", "بدون قیمت"],
    )
    def test_missing_price_is_none_never_zero(self, raw: object) -> None:
        # A missing price must never be confused with a price of zero.
        assert normalize_price(raw) is None

    def test_negotiable_with_digits_still_none(self) -> None:
        assert normalize_price("توافقی ۱۲۳") is None


class TestMileage:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("۴۰,۰۰۰ km", 40_000),
            ("40,000 km", 40_000),
            ("۱۲۳۴۵۶ کیلومتر", 123_456),
            ("صفر", 0),
            ("0 km", 0),
            (40_000, 40_000),
        ],
    )
    def test_mileage(self, raw: object, expected: int) -> None:
        assert normalize_mileage(raw) == expected

    def test_missing_mileage(self) -> None:
        assert normalize_mileage(None) is None
        assert normalize_mileage("نامشخص") is None

    def test_zero_km_flag(self) -> None:
        assert is_zero_km("صفر", None) is True
        assert is_zero_km("40,000 km", 40_000) is False
        assert is_zero_km(None, 0) is True
        assert is_zero_km(None, None) is None


class TestYear:
    @pytest.mark.parametrize(
        "raw,jalali,greg",
        [
            ("۱۴۰۳", 1403, 2024),
            ("1397", 1397, 2018),
            (1403, 1403, 2024),
            ("2018", 1397, 2018),
            ("2020", 1399, 2020),
        ],
    )
    def test_year(self, raw: object, jalali: int, greg: int) -> None:
        assert parse_year(raw) == (jalali, greg)

    def test_unparseable_year(self) -> None:
        assert parse_year("نامشخص") == (None, None)
        assert parse_year(None) == (None, None)

    def test_filter_boundary_years_are_equivalent(self) -> None:
        # The target URL uses "1397-2018"; both halves must denote the same year.
        assert parse_year("1397")[1] == parse_year("2018")[1] == 2018


class TestRelativeTime:
    def test_moments_ago(self) -> None:
        assert parse_relative_time("لحظاتی پیش", now_ts=1000.0) == 1000.0

    def test_days_ago(self) -> None:
        assert parse_relative_time("۲ روز پیش", now_ts=1_000_000.0) == 1_000_000.0 - 172_800

    def test_hours_ago(self) -> None:
        assert parse_relative_time("۳ ساعت پیش", now_ts=1000.0) == 1000.0 - 10_800

    @pytest.mark.parametrize(
        "phrase",
        ["لحظاتی پیش", "دقایقی پیش", "هم اکنون", "همین الان", "چند لحظه پیش"],
    )
    def test_all_just_now_phrasings(self, phrase: str) -> None:
        # Regression: "دقایقی پیش" is the most common value on live listings and
        # was previously unparsed, leaving every timestamp null.
        assert parse_relative_time(phrase, now_ts=1000.0) == 1000.0

    def test_vague_plural_assumes_one_unit(self) -> None:
        assert parse_relative_time("چند ساعت پیش", now_ts=10_000.0) == 10_000.0 - 3600

    def test_named_day_is_handled(self) -> None:
        # "دیروز" (yesterday) carries no number but is unambiguous.
        assert parse_relative_time("دیروز", now_ts=1000.0) == 1000.0 - 86400

    def test_unparseable(self) -> None:
        assert parse_relative_time("زمان نامشخص", now_ts=1000.0) is None
        assert parse_relative_time(None, now_ts=1000.0) is None


class TestPublishedTime:
    """Bama mixes relative phrases, named days and absolute Jalali dates.

    Absolute dates dominate any large result set (listings older than a week),
    so handling only relative phrasings leaves most timestamps null.
    """

    def test_absolute_jalali_date(self) -> None:
        from datetime import datetime

        ts = parse_jalali_date("1405/4/28")
        assert ts is not None
        assert datetime.fromtimestamp(ts).strftime("%Y-%m-%d") == "2026-07-19"

    def test_absolute_jalali_date_with_persian_digits(self) -> None:
        assert parse_jalali_date("۱۴۰۵/۴/۲۸") == parse_jalali_date("1405/4/28")

    def test_dashed_separator(self) -> None:
        assert parse_jalali_date("1405-4-28") == parse_jalali_date("1405/4/28")

    @pytest.mark.parametrize("bad", ["1405/13/45", "not a date", "", None, "1405/0/0"])
    def test_invalid_dates_are_none_not_guesses(self, bad: str | None) -> None:
        assert parse_jalali_date(bad) is None

    def test_named_days(self) -> None:
        assert parse_published_time("دیروز", now_ts=1_000_000.0) == 1_000_000.0 - 86400
        assert parse_published_time("پریروز", now_ts=1_000_000.0) == 1_000_000.0 - 172800
        assert parse_published_time("امروز", now_ts=1_000_000.0) == 1_000_000.0

    def test_relative_still_works(self) -> None:
        assert parse_published_time("2 روز پیش", now_ts=1_000_000.0) == 1_000_000.0 - 172800
        assert parse_published_time("دقایقی پیش", now_ts=1_000_000.0) == 1_000_000.0

    def test_relative_wins_over_a_stray_number(self) -> None:
        # "5 روز پیش" must be read as relative, never as a Jalali date.
        assert parse_published_time("5 روز پیش", now_ts=1_000_000.0) == 1_000_000.0 - 432000

    def test_unknown_format_is_none(self) -> None:
        assert parse_published_time("زمان نامشخص", now_ts=1_000_000.0) is None
