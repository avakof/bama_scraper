"""Persian/Arabic text and numeric normalization.

Design rule enforced throughout the project: normalization NEVER destroys the
original text. Every ``normalize_*`` helper returns a value derived from the
input, and callers are expected to persist the raw string alongside it.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Final

# ---------------------------------------------------------------------------
# Digit / character maps
# ---------------------------------------------------------------------------

#: Persian (Extended Arabic-Indic) digits U+06F0..U+06F9
PERSIAN_DIGITS: Final[str] = "۰۱۲۳۴۵۶۷۸۹"
#: Arabic-Indic digits U+0660..U+0669
ARABIC_DIGITS: Final[str] = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS: Final[str] = "0123456789"

_DIGIT_TABLE: Final[dict[int, str]] = {
    **{ord(p): a for p, a in zip(PERSIAN_DIGITS, ASCII_DIGITS, strict=True)},
    **{ord(p): a for p, a in zip(ARABIC_DIGITS, ASCII_DIGITS, strict=True)},
}

#: Arabic letters that must be folded to their Persian equivalents.
_CHAR_TABLE: Final[dict[int, str]] = {
    0x064A: "ی",  # ARABIC YEH -> FARSI YEH
    0x0649: "ی",  # ALEF MAKSURA -> FARSI YEH
    0x0643: "ک",  # ARABIC KAF -> KEHEH
    0x06AA: "ک",  # SWASH KAF -> KEHEH
    0x0629: "ه",  # TEH MARBUTA -> HEH
    0x0624: "و",  # WAW WITH HAMZA -> WAW
}

#: Whitespace-ish characters that should collapse to a normal space.
_SPACE_CHARS: Final[str] = (
    " "  # NO-BREAK SPACE
    "​"  # ZERO WIDTH SPACE
    "‌"  # ZERO WIDTH NON-JOINER (ZWNJ) -- see note below
    "‎‏"  # LRM / RLM
    "‪‫‬‭‮"  # bidi embedding controls
    "﻿"  # BOM
    "           "
    " 　\t\r\n\v\f"
)

#: Diacritics (harakat) that carry no lexical meaning for our purposes.
_DIACRITICS_RE: Final[re.Pattern[str]] = re.compile(r"[ً-ْٰـ]")


def convert_digits(text: str) -> str:
    """Convert Persian and Arabic-Indic digits to ASCII digits.

    Non-digit characters are left untouched.

    >>> convert_digits("۱۴۰۳ و ٥٦")
    '1403 و 56'
    """
    if not text:
        return text
    return text.translate(_DIGIT_TABLE)


def normalize_chars(text: str) -> str:
    """Fold Arabic character variants to Persian and strip diacritics."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_CHAR_TABLE)
    return _DIACRITICS_RE.sub("", text)


def normalize_whitespace(text: str, *, keep_zwnj: bool = True) -> str:
    """Collapse exotic whitespace to single ASCII spaces and trim.

    ZWNJ (U+200C) is meaningful in Persian orthography (e.g. ``می‌رود``), so it
    is preserved by default rather than being treated as whitespace.
    """
    if not text:
        return text
    chars = _SPACE_CHARS.replace("‌", "") if keep_zwnj else _SPACE_CHARS
    text = text.translate({ord(c): " " for c in chars})
    return re.sub(r" {2,}", " ", text).strip()


def normalize_text(text: str | None, *, keep_zwnj: bool = True) -> str | None:
    """Full text pipeline: char folding + digit conversion + whitespace cleanup.

    Returns ``None`` for ``None`` input and for strings that are empty once
    normalized, so that "absent" and "blank" collapse to a single SQL NULL.
    """
    if text is None:
        return None
    out = normalize_whitespace(convert_digits(normalize_chars(text)), keep_zwnj=keep_zwnj)
    return out or None


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

#: Separators used as thousands markers in Persian/Arabic content.
_THOUSANDS_RE: Final[re.Pattern[str]] = re.compile(r"[,٬،  ']")

#: Words that mean "price not stated" rather than "price is zero".
PRICE_UNAVAILABLE_MARKERS: Final[tuple[str, ...]] = (
    "توافقی",  # negotiable
    "تماس",  # contact
    "تماس بگیرید",
    "ناموجود",
    "اقساطی",  # installment-only
    "قیمت مخفی",
    "بدون قیمت",
    "توافقی است",
)


def _digits_only(text: str) -> str | None:
    """Return the first contiguous run of digits after separator removal."""
    cleaned = _THOUSANDS_RE.sub("", convert_digits(text))
    m = re.search(r"\d+", cleaned)
    return m.group(0) if m else None


def normalize_price(text: str | int | float | None) -> int | None:
    """Convert a displayed price string to an integer Toman value.

    Returns ``None`` — never ``0`` — when the price is absent, negotiable, or
    otherwise not a real number. A literal ``"0"`` from the API also maps to
    ``None`` because Bama uses ``price: "0"`` as a sentinel for
    negotiable/installment listings rather than a genuine free car.

    >>> normalize_price("۱٬۲۵۰٬۰۰۰٬۰۰۰ تومان")
    1250000000
    >>> normalize_price("توافقی") is None
    True
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        value = int(text)
        return value if value > 0 else None

    normalized = normalize_chars(text)
    if any(marker in normalized for marker in PRICE_UNAVAILABLE_MARKERS):
        return None

    digits = _digits_only(normalized)
    if digits is None:
        return None
    value = int(digits)
    return value if value > 0 else None


def normalize_mileage(text: str | int | float | None) -> int | None:
    """Convert a displayed mileage string to integer kilometres.

    ``"صفر"`` (zero) and ``"0 km"`` are genuine zero-kilometre values, so unlike
    price this function *does* return ``0``.

    >>> normalize_mileage("۴۰,۰۰۰ km")
    40000
    >>> normalize_mileage("صفر")
    0
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text)

    normalized = normalize_chars(text)
    if "صفر" in normalized or "کارکرده نشده" in normalized:
        return 0
    digits = _digits_only(normalized)
    return int(digits) if digits is not None else None


def is_zero_km(mileage_text: str | None, mileage_km: int | None) -> bool | None:
    """Best-effort zero-kilometre flag; ``None`` when undeterminable."""
    if mileage_text:
        if "صفر" in normalize_chars(mileage_text):
            return True
    if mileage_km is None:
        return None
    return mileage_km == 0


# ---------------------------------------------------------------------------
# Year handling
# ---------------------------------------------------------------------------

#: Jalali years plausibly used by Bama for production year.
_JALALI_RANGE: Final[tuple[int, int]] = (1300, 1450)
_GREGORIAN_RANGE: Final[tuple[int, int]] = (1900, 2100)


def parse_year(text: str | int | None) -> tuple[int | None, int | None]:
    """Return ``(jalali_year, gregorian_year)`` from a displayed year.

    Bama shows Jalali years for Iranian-market cars (e.g. ``1403``) and
    Gregorian years for some imports (e.g. ``2018``). Conversion is only
    applied when the value is unambiguous.

    >>> parse_year("۱۴۰۳")
    (1403, 2024)
    >>> parse_year("2018")
    (1397, 2018)
    """
    if text is None:
        return None, None
    raw = convert_digits(str(text))
    m = re.search(r"\d{3,4}", raw)
    if not m:
        return None, None
    value = int(m.group(0))
    if _JALALI_RANGE[0] <= value <= _JALALI_RANGE[1]:
        return value, value + 621
    if _GREGORIAN_RANGE[0] <= value <= _GREGORIAN_RANGE[1]:
        return value - 621, value
    return None, None


# ---------------------------------------------------------------------------
# Relative Persian dates
# ---------------------------------------------------------------------------

#: Phrases meaning "just now" that carry no numeric quantity.
#: ``دقایقی پیش`` ("some minutes ago") is by far the most common on Bama.
_JUST_NOW_MARKERS: Final[tuple[str, ...]] = (
    "لحظاتی پیش",
    "دقایقی پیش",
    "هم اکنون",
    "همین الان",
    "چند لحظه پیش",
)

_REL_UNITS: Final[dict[str, int]] = {
    "ثانیه": 1,
    "دقیقه": 60,
    "ساعت": 3600,
    "روز": 86400,
    "هفته": 604800,
    "ماه": 2592000,
    "سال": 31536000,
}


#: Named day offsets Bama uses instead of a count.
_NAMED_DAYS: Final[dict[str, int]] = {
    "دیروز": -86400,
    "پریروز": -172800,
    "امروز": 0,
}

#: Absolute Jalali date, e.g. ``1405/4/28``.
_JALALI_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(1[34]\d{2})[/\-](\d{1,2})[/\-](\d{1,2})\b"
)


def parse_jalali_date(text: str | None) -> float | None:
    """Convert an absolute Jalali date (``1405/4/28``) to a UNIX timestamp.

    Bama shows absolute Jalali dates once a listing is more than a week old,
    which is the majority of any large result set. Returns ``None`` when the
    string holds no such date or the date is not a real calendar day.
    """
    if not text:
        return None
    match = _JALALI_DATE_RE.search(convert_digits(text))
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        import jdatetime

        gregorian = jdatetime.date(year, month, day).togregorian()
        return datetime(gregorian.year, gregorian.month, gregorian.day).timestamp()
    except Exception:
        # Invalid calendar date, or jdatetime unavailable: report "unknown"
        # rather than inventing a timestamp.
        return None


def parse_published_time(text: str | None, *, now_ts: float) -> float | None:
    """Normalize any of Bama's publication-date formats to a UNIX timestamp.

    Handles, in order: relative phrases (``۲ روز پیش``), named days (``دیروز``)
    and absolute Jalali dates (``1405/4/28``).
    """
    if not text:
        return None
    relative = parse_relative_time(text, now_ts=now_ts)
    if relative is not None:
        return relative
    return parse_jalali_date(text)


def parse_relative_time(text: str | None, *, now_ts: float) -> float | None:
    """Convert Persian relative time (``"۲ روز پیش"``) to a UNIX timestamp.

    ``now_ts`` is injected rather than read from the clock so the behaviour is
    deterministic and unit-testable.
    """
    if not text:
        return None
    normalized = normalize_text(text) or ""
    # Bama uses several count-less "just now" phrasings; treat them all as now.
    if any(marker in normalized for marker in _JUST_NOW_MARKERS):
        return now_ts
    for phrase, offset in _NAMED_DAYS.items():
        if phrase in normalized:
            return now_ts + offset
    # Vague plurals without a number ("a few hours ago") -- assume one unit.
    m = re.search(r"(?:چند|چندین)\s*(" + "|".join(_REL_UNITS) + r")", normalized)
    if m:
        return now_ts - _REL_UNITS[m.group(1)]
    m = re.search(r"(\d+)\s*(" + "|".join(_REL_UNITS) + r")", normalized)
    if not m:
        return None
    amount, unit = int(m.group(1)), m.group(2)
    return now_ts - amount * _REL_UNITS[unit]
