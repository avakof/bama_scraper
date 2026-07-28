"""Numeric helpers, above all the dot-stripped decimal encoding."""

from __future__ import annotations

from datetime import datetime

import pytest
from bama_deep.numbers import (
    descale_bounded,
    fractional_digits,
    infer_series_years,
    jalali_month_index,
    parse_bool_string,
    parse_decimal,
    parse_iso_ts,
    parse_jalali_day_month,
    parse_measure,
    parse_signed_toman,
    reconcile_dotless,
    slugify_fa,
)

from bama_scraper.normalization import normalize_price


class TestParseDecimal:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.7 لیتر", 1.7),
            ("۱٫۶ لیتر", 1.6),  # Persian decimal separator U+066B
            ("۱۳.۲ ثانیه", 13.2),
            ("113 اسب‌بخار", 113.0),
            ("1,180 کیلوگرم", 1180.0),
            ("-5,000,000", -5000000.0),
            (17, 17.0),
        ],
    )
    def test_values(self, raw: object, expected: float) -> None:
        assert parse_decimal(raw) == expected

    @pytest.mark.parametrize("raw", ["نامشخص", "", None, "دارد"])
    def test_no_number(self, raw: str | None) -> None:
        assert parse_decimal(raw) is None

    @pytest.mark.parametrize(
        "raw,expected", [("1.7 لیتر", 1), ("12 ثانیه", 0), ("6.95", 2), ("نامشخص", 0), (None, 0)]
    )
    def test_fractional_digits(self, raw: str | None, expected: int) -> None:
        assert fractional_digits(raw) == expected


class TestDotStrippedEncoding:
    """The central rule: scale is 10**(fractional digits), not a constant.

    Bama writes a decimal as its display string with the point deleted, so the
    integer alone is ambiguous -- 12 could mean 12 or 1.2. The text decides.
    """

    @pytest.mark.parametrize(
        "raw_int,text,value,agreement",
        [
            (17, "1.7 لیتر", 1.7, "scaled_x10"),
            (69, "6.9 لیتر در صد کیلومتر", 6.9, "scaled_x10"),
            (105, "10.5 ثانیه", 10.5, "scaled_x10"),
            (132, "13.2 ثانیه", 13.2, "scaled_x10"),
            # THE regression: a fixed /10 would report 1.2 seconds here.
            (12, "12 ثانیه", 12.0, "exact"),
            (13, "13 ثانیه", 13.0, "exact"),
            (113, "113 اسب‌بخار", 113.0, "exact"),
            (155, "155 نیوتن‌متر", 155.0, "exact"),
            (695, "6.95 لیتر", 6.95, "scaled_x100"),
        ],
    )
    def test_real_pairs(self, raw_int: int, text: str, value: float, agreement: str) -> None:
        assert reconcile_dotless(raw_int, text) == (value, agreement)

    def test_integer_acceleration_is_not_divided(self) -> None:
        # 1,296 live ads have integer acceleration; dividing would corrupt them all.
        value, agreement = reconcile_dotless(12, "12 ثانیه")
        assert value == 12.0 and agreement == "exact"
        assert value != 1.2

    def test_disagreement_prefers_text_and_is_reported(self) -> None:
        value, agreement = reconcile_dotless(999, "1.7 لیتر")
        assert value == 1.7  # text wins
        assert agreement == "mismatch"  # but the conflict is surfaced

    def test_missing_int(self) -> None:
        assert reconcile_dotless(None, "1.7 لیتر") == (1.7, "text_only")

    def test_missing_text_reports_int_verbatim(self) -> None:
        # With no text the scale is unknowable, so do not guess a divisor.
        assert reconcile_dotless(45, None) == (45.0, "int_only")

    def test_both_missing(self) -> None:
        assert reconcile_dotless(None, None) == (None, "text_only")

    def test_persian_digits_in_text(self) -> None:
        assert reconcile_dotless(17, "۱.۷ لیتر") == (1.7, "scaled_x10")


class TestDescaleBounded:
    """seller.score has no text sibling, so bounds are the only disambiguator."""

    def test_out_of_range_is_descaled_and_flagged(self) -> None:
        assert descale_bounded(45, lo=0, hi=5) == (4.5, True)

    def test_in_range_is_kept_and_not_flagged(self) -> None:
        assert descale_bounded(4, lo=0, hi=5) == (4.0, False)

    def test_two_orders_of_magnitude(self) -> None:
        assert descale_bounded(450, lo=0, hi=5) == (4.5, True)

    def test_unrecoverable(self) -> None:
        assert descale_bounded(99999, lo=0, hi=5) == (None, True)

    def test_none(self) -> None:
        assert descale_bounded(None, lo=0, hi=5) == (None, False)


class TestSignedToman:
    """price_diff is a difference, not a price, so it may be negative."""

    def test_negative_kept(self) -> None:
        assert parse_signed_toman("-5,000,000") == -5_000_000

    def test_normalize_price_silently_flips_the_sign(self) -> None:
        # normalize_price extracts the first digit run and ignores the minus, so a
        # negative input comes back POSITIVE. Harmless for prices (never negative)
        # but it would invert every price_diff -- hence a separate parser.
        assert normalize_price("-5,000,000") == 5_000_000
        assert parse_signed_toman("-5,000,000") == -5_000_000

    def test_normalize_price_still_rejects_zero(self) -> None:
        assert normalize_price("0") is None
        assert parse_signed_toman("0") == 0

    def test_positive_and_persian(self) -> None:
        assert parse_signed_toman("۱٬۵۰۰٬۰۰۰") == 1_500_000
        assert parse_signed_toman(-42) == -42

    def test_no_number(self) -> None:
        assert parse_signed_toman("نامشخص") is None
        assert parse_signed_toman(None) is None


class TestBooleans:
    @pytest.mark.parametrize("raw", ["true", "True", True, 1, "بله", "دارد"])
    def test_truthy(self, raw: object) -> None:
        assert parse_bool_string(raw) is True

    @pytest.mark.parametrize("raw", ["false", "False", False, 0, "خیر", "ندارد"])
    def test_falsy(self, raw: object) -> None:
        assert parse_bool_string(raw) is False

    @pytest.mark.parametrize("raw", ["-", "", None, "نامشخص", "6 اسپیکر"])
    def test_unknown_is_none_not_false(self, raw: object) -> None:
        # "unknown" must stay distinguishable from "absent".
        assert parse_bool_string(raw) is None


class TestMeasure:
    @pytest.mark.parametrize(
        "raw,number,unit",
        [
            ("1,180 کیلوگرم", 1180.0, "کیلوگرم"),
            ("۶۰ لیتر", 60.0, "لیتر"),
            ("4559 میلی متر", 4559.0, "میلی متر"),
            ("190 km/h", 190.0, "km/h"),
        ],
    )
    def test_split(self, raw: str, number: float, unit: str) -> None:
        got_number, got_unit = parse_measure(raw)
        assert got_number == number
        assert got_unit is not None and unit.replace(" ", "") in got_unit.replace(" ", "")

    def test_non_numeric(self) -> None:
        assert parse_measure("دارد") == (None, None)
        assert parse_measure(None) == (None, None)


class TestJalaliDayMonth:
    def test_valid(self) -> None:
        label, ts = parse_jalali_day_month("18 فروردین", year=1405)
        assert label == "1405-01-18"
        assert ts is not None
        assert datetime.fromtimestamp(ts).strftime("%Y-%m-%d") == "2026-04-07"

    def test_persian_digits(self) -> None:
        assert parse_jalali_day_month("۰۵ مرداد", year=1405)[0] == "1405-05-05"

    def test_invalid_calendar_day_is_rejected_not_shifted(self) -> None:
        # 30 Esfand only exists in a leap year; 1405 is not one.
        label, ts = parse_jalali_day_month("30 اسفند", year=1405)
        assert ts is None

    def test_unknown_month(self) -> None:
        assert parse_jalali_day_month("18 غیرماه", year=1405) == (None, None)

    def test_all_twelve_months_resolve(self) -> None:
        from bama_deep.numbers import JALALI_MONTHS

        assert [jalali_month_index(m) for m in JALALI_MONTHS] == list(range(1, 13))


class TestSeriesYearInference:
    """Price-history points carry no year, so it must be inferred."""

    def test_within_one_year(self) -> None:
        dates = ["18 فروردین", "20 اردیبهشت", "05 مرداد"]
        assert infer_series_years(dates, anchor_jalali_year=1405) == [1405, 1405, 1405]

    def test_wraps_across_new_year(self) -> None:
        # Going back from Farvardin 1405 into Esfand lands in 1404.
        dates = ["25 بهمن", "10 اسفند", "05 فروردین"]
        assert infer_series_years(dates, anchor_jalali_year=1405) == [1404, 1404, 1405]

    def test_unknown_month_inherits_the_running_year(self) -> None:
        dates = ["18 فروردین", "???", "05 اردیبهشت"]
        years = infer_series_years(dates, anchor_jalali_year=1405)
        assert len(years) == 3 and all(y == 1405 for y in years)

    def test_empty(self) -> None:
        assert infer_series_years([], anchor_jalali_year=1405) == []


class TestIsoTimestamp:
    def test_fractional_seconds(self) -> None:
        ts = parse_iso_ts("2026-07-27T20:35:47.08")
        assert ts is not None
        assert datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") == "2026-07-27 20:35"

    def test_variants(self) -> None:
        assert parse_iso_ts("2026-07-27T20:35:47") is not None
        assert parse_iso_ts("2026-07-27") is not None

    def test_garbage(self) -> None:
        assert parse_iso_ts("not a date") is None
        assert parse_iso_ts(None) is None


class TestSlugs:
    def test_known_keys_map(self) -> None:
        assert slugify_fa("سانروف") == "sunroof"
        assert slugify_fa("وزن") == "weight"
        assert slugify_fa("حجم باک") == "fuel_tank_capacity"

    def test_deterministic_for_unknown(self) -> None:
        a = slugify_fa("یک کلید ناشناخته")
        b = slugify_fa("یک کلید ناشناخته")
        assert a == b and a.startswith("unk_")

    def test_unknown_keys_do_not_collide(self) -> None:
        assert slugify_fa("کلید الف") != slugify_fa("کلید ب")

    def test_ascii_keys_pass_through(self) -> None:
        assert slugify_fa("ABS") == "abs"

    def test_character_variants_fold_to_one_slug(self) -> None:
        # Arabic yeh vs Persian yeh must not produce two different slugs.
        assert slugify_fa("ایمنی") == slugify_fa("ايمنی")

    def test_empty(self) -> None:
        assert slugify_fa("") == "unk_empty"
        assert slugify_fa(None) == "unk_empty"
