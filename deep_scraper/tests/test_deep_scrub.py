"""Privacy scrubbing: key stripping and free-text redaction."""

from __future__ import annotations

import json

import pytest
from bama_deep.scrub import DEEP_PRIVATE_KEYS, assert_no_contact, scrub_freetext, strip_keys


class TestStripKeys:
    def test_removes_phone_at_top_level(self) -> None:
        assert "phone" not in strip_keys({"phone": "۰۹۱۲۳۴۵۶۷۸۹", "title": "x"})

    def test_removes_phone_nested(self) -> None:
        cleaned = strip_keys({"data": {"detail": {"phone": "0912", "code": "abc"}}})
        assert "phone" not in cleaned["data"]["detail"]
        assert cleaned["data"]["detail"]["code"] == "abc"

    def test_removes_phone_inside_lists(self) -> None:
        cleaned = strip_keys({"items": [{"phone": "0912"}, {"mobile": "0913"}]})
        assert cleaned["items"] == [{}, {}]

    def test_removes_extended_keys(self) -> None:
        payload = dict.fromkeys(("whatsapp", "telegram", "email", "phoneNumber"), "x")
        payload["keep"] = "y"
        cleaned = strip_keys(payload)
        assert cleaned == {"keep": "y"}

    def test_keeps_everything_else(self) -> None:
        payload = {"code": "abc", "price": {"fixed": 1}, "images": ["a", "b"]}
        assert strip_keys(payload) == payload

    def test_base_scraper_keys_are_included(self) -> None:
        # Must be a superset of the original scraper's private-key set.
        from bama_scraper.detail_parser import _PRIVATE_KEYS

        assert set(_PRIVATE_KEYS) <= set(DEEP_PRIVATE_KEYS)

    def test_serialized_payload_has_no_phone_key(self) -> None:
        payload = {"content": {"phone": "۰۹۱۹۳۶۹۱۰XX", "title": "پراید"}}
        assert '"phone"' not in json.dumps(strip_keys(payload), ensure_ascii=False)


class TestScrubFreetext:
    @pytest.mark.parametrize(
        "raw",
        [
            "تماس 09123456789",
            "تماس ۰۹۱۲۳۴۵۶۷۸۹",
            "شماره 0912-345-6789",
            "شماره 0912 345 6789",
            "+989123456789",
            "تلفن ۰۹۱۹۳۶۹۱۰XX",
        ],
    )
    def test_phone_numbers_redacted(self, raw: str) -> None:
        scrubbed, count = scrub_freetext(raw)
        assert count >= 1
        assert "[redacted]" in scrubbed

    @pytest.mark.parametrize(
        "raw", ["@my_shop", "t.me/someseller", "https://t.me/x", "seller@example.com"]
    )
    def test_handles_redacted(self, raw: str) -> None:
        scrubbed, count = scrub_freetext(raw)
        assert count >= 1 and "[redacted]" in scrubbed

    @pytest.mark.parametrize(
        "raw",
        [
            "قیمت 1,000,000,000 تومان",
            "مدل 1403",
            "کارکرد 100,000 کیلومتر",
            "بیمه شخص ثالث تا اسفند 1405",
            "موتور EF7 با گیربکس 6 دنده",
            "قیمت ۱٬۵۸۵٬۰۰۰٬۰۰۰",
        ],
    )
    def test_prices_years_mileages_survive(self, raw: str) -> None:
        # The false-positive guard: a naive "long digit run" rule would eat these.
        scrubbed, count = scrub_freetext(raw)
        assert count == 0
        assert scrubbed == raw

    def test_persian_text_preserved_around_redaction(self) -> None:
        scrubbed, _ = scrub_freetext("خودرو سالم تماس 09123456789 فقط عصرها")
        assert "خودرو سالم" in scrubbed and "فقط عصرها" in scrubbed

    def test_idempotent(self) -> None:
        once, first = scrub_freetext("تماس 09123456789")
        twice, second = scrub_freetext(once)
        assert twice == once and second == 0

    def test_multiple_redactions_counted(self) -> None:
        _, count = scrub_freetext("09123456789 و 09351234567")
        assert count == 2

    def test_empty(self) -> None:
        assert scrub_freetext(None) == (None, 0)
        assert scrub_freetext("") == ("", 0)


class TestFinalGate:
    """assert_no_contact is the audit's last line of defence on stored data."""

    def test_clean_blob(self) -> None:
        assert assert_no_contact('{"code":"abc","price":1000000000,"year":1403}') == []

    def test_detects_phone_digits(self) -> None:
        assert assert_no_contact('{"note":"09123456789"}')

    def test_detects_persian_phone_digits(self) -> None:
        assert assert_no_contact('{"note":"۰۹۱۲۳۴۵۶۷۸۹"}')

    def test_detects_phone_key(self) -> None:
        assert assert_no_contact('{"phone":"hidden"}')

    def test_prices_do_not_trigger(self) -> None:
        assert assert_no_contact('{"price_toman":1585000000,"mileage_km":100000}') == []
