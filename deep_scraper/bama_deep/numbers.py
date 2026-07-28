"""Numeric and date helpers specific to Bama's deep payloads.

These fill genuine gaps in ``bama_scraper.normalization``, which is imported and
reused for everything it already covers.

The central concern here is Bama's **dot-stripped decimal encoding**. A decimal
is serialized as its display string with the decimal point deleted, so the
implied scale is ``10 ** (fractional digits in the text)`` -- it is *not* a
constant:

    volume          value=17   text="1.7 لیتر"                 -> /10
    fuelConsumption value=69   text="6.9 لیتر در صد کیلومتر"    -> /10
    acceleration    value=12   text="12 ثانیه"                 -> /1   (12 s, not 1.2 s)
    power           value=113  text="113 اسب‌بخار"              -> /1

Measured on the live corpus, acceleration is an integer on 1,296 advertisements
and one-decimal on 1,019. A fixed ``/10`` would therefore silently turn 12
seconds into 1.2 on ~1,300 cars. Consequently **the text is authoritative** and
the integer is kept only as a checksum whose agreement class is recorded.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final, Literal

from bama_scraper.normalization import convert_digits, normalize_chars, normalize_text

from .spec_slugs import SPEC_KEY_SLUGS

# ---------------------------------------------------------------------------
# Decimal parsing
# ---------------------------------------------------------------------------

#: Persian decimal separator U+066B, plus the ASCII point.
_DECIMAL_SEPARATORS: Final[str] = ".٫"

#: Thousands separators that must vanish before a number is read.
_GROUPING_RE: Final[re.Pattern[str]] = re.compile(r"[,٬، ‏\s']")

_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"-?\d+(?:[.]\d+)?")


def _to_ascii_number_text(text: str) -> str:
    """ASCII-ify digits and separators so a plain regex can read the number."""
    out = convert_digits(text)
    for sep in _DECIMAL_SEPARATORS[1:]:
        out = out.replace(sep, ".")
    return _GROUPING_RE.sub("", out)


def parse_decimal(text: str | int | float | None) -> float | None:
    """Return the first decimal number in a Persian or ASCII string.

    >>> parse_decimal("1.7 لیتر")
    1.7
    >>> parse_decimal("۱٫۶ لیتر")
    1.6
    >>> parse_decimal("نامشخص") is None
    True
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = _NUMBER_RE.search(_to_ascii_number_text(text))
    return float(match.group(0)) if match else None


def fractional_digits(text: str | None) -> int:
    """Count digits after the decimal separator in a displayed number.

    This is what determines the dot-stripped scale.

    >>> fractional_digits("1.7 لیتر")
    1
    >>> fractional_digits("12 ثانیه")
    0
    """
    if not text:
        return 0
    match = _NUMBER_RE.search(_to_ascii_number_text(text))
    if not match:
        return 0
    _, _, frac = match.group(0).partition(".")
    return len(frac)


# ---------------------------------------------------------------------------
# Dot-stripped reconciliation
# ---------------------------------------------------------------------------

Agreement = Literal[
    "exact",  # int equals the text value (no fractional digits)
    "scaled_x10",  # int is the text value with one decimal point removed
    "scaled_x100",  # ... two
    "mismatch",  # int cannot be reconciled with the text -- trust the text
    "text_only",  # no integer supplied
    "int_only",  # no text supplied; value taken from the integer as-is
]


def reconcile_dotless(
    raw_int: int | float | None, text: str | None
) -> tuple[float | None, Agreement]:
    """Resolve a Bama numeric pair into ``(value, agreement)``.

    The text wins because it is unambiguous; ``raw_int`` is a checksum. When the
    two disagree the text is still returned and the disagreement is reported via
    the agreement class, so a downstream audit can surface it rather than having
    a wrong number silently persisted.

    >>> reconcile_dotless(17, "1.7 لیتر")
    (1.7, 'scaled_x10')
    >>> reconcile_dotless(12, "12 ثانیه")
    (12.0, 'exact')
    >>> reconcile_dotless(999, "1.7 لیتر")
    (1.7, 'mismatch')
    """
    from_text = parse_decimal(text)

    if from_text is None:
        if raw_int is None:
            return None, "text_only"
        # No text to disambiguate the scale: report the integer verbatim rather
        # than guessing a divisor.
        return float(raw_int), "int_only"

    if raw_int is None:
        return from_text, "text_only"

    digits = fractional_digits(text)
    expected = round(from_text * (10**digits))
    if round(float(raw_int)) == expected:
        return from_text, ("exact" if digits == 0 else f"scaled_x{10**digits}")  # type: ignore[return-value]
    return from_text, "mismatch"


def descale_bounded(
    raw_int: int | float | None, *, lo: float, hi: float
) -> tuple[float | None, bool]:
    """Recover a decimal from a dot-stripped integer that has no text sibling.

    Used for ``seller.score``, where the HTML gives ``45`` and the JSON API gives
    ``4.5``. Tries the value and successive divisions by ten, returning the first
    that falls inside ``[lo, hi]``. The second element flags that a heuristic was
    applied, so a genuine in-range value is not misreported as guessed.

    >>> descale_bounded(45, lo=0, hi=5)
    (4.5, True)
    >>> descale_bounded(4, lo=0, hi=5)
    (4.0, False)
    """
    if raw_int is None:
        return None, False
    value = float(raw_int)
    if lo <= value <= hi:
        return value, False
    for power in (10.0, 100.0):
        candidate = value / power
        if lo <= candidate <= hi:
            return candidate, True
    return None, True


def parse_signed_toman(text: str | int | float | None) -> int | None:
    """Parse a Toman amount that may legitimately be negative.

    ``normalize_price`` deliberately maps values ``<= 0`` to ``None`` because a
    zero price means "not stated". Price *differences* are a different quantity:
    ``-5,000,000`` is real information, so it needs its own parser. Never use
    this for a price.

    >>> parse_signed_toman("-5,000,000")
    -5000000
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text)
    cleaned = _to_ascii_number_text(normalize_chars(text))
    match = re.search(r"-?\d+", cleaned)
    return int(match.group(0)) if match else None


# ---------------------------------------------------------------------------
# Booleans and measurements
# ---------------------------------------------------------------------------

_TRUE_TOKENS: Final[frozenset[str]] = frozenset({"true", "1", "yes", "بله", "دارد", "دارای"})
_FALSE_TOKENS: Final[frozenset[str]] = frozenset({"false", "0", "no", "خیر", "ندارد", "فاقد"})


def parse_bool_string(value: Any) -> bool | None:
    """Interpret Bama's boolean spec items.

    The CarReview API reports ``type: "Boolean"`` but delivers the *strings*
    ``"true"`` / ``"false"``. Persian yes/no wordings appear on some String-typed
    items too. Anything unrecognised returns ``None`` rather than defaulting to
    ``False``, so "unknown" and "absent" stay distinguishable.

    >>> parse_bool_string("true"), parse_bool_string("ندارد"), parse_bool_string("-")
    (True, False, None)
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    token = (normalize_text(str(value)) or "").strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def parse_measure(text: str | None) -> tuple[float | None, str | None]:
    """Split a measurement into ``(number, unit)``.

    >>> parse_measure("1,180 کیلوگرم")
    (1180.0, 'کیلوگرم')
    >>> parse_measure("دارد")
    (None, None)
    """
    if not text:
        return None, None
    normalized = normalize_text(text) or ""
    number = parse_decimal(normalized)
    if number is None:
        return None, None
    match = _NUMBER_RE.search(_to_ascii_number_text(normalized))
    unit = None
    if match:
        # Take the trailing words after the number, from the original string so
        # the Persian unit is preserved verbatim.
        tail = _NUMBER_RE.sub(" ", _to_ascii_number_text(normalized), count=1)
        unit_text = normalize_text(tail)
        # Recover the unit from the normalized (non-ASCII-ified) text where possible.
        persian_tail = re.sub(r"[\d.,٬،]+", " ", normalized).strip()
        unit = normalize_text(persian_tail) or unit_text
    return number, unit or None


# ---------------------------------------------------------------------------
# Jalali day-month dates (price history)
# ---------------------------------------------------------------------------

JALALI_MONTHS: Final[tuple[str, ...]] = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)

_MONTH_INDEX: Final[dict[str, int]] = {
    (normalize_text(name) or name): i + 1 for i, name in enumerate(JALALI_MONTHS)
}


def jalali_month_index(text: str | None) -> int | None:
    """Return 1-12 for a Persian month name, else ``None``."""
    if not text:
        return None
    normalized = normalize_text(text) or ""
    for name, index in _MONTH_INDEX.items():
        if name in normalized:
            return index
    return None


def parse_jalali_day_month(text: str | None, *, year: int) -> tuple[str | None, float | None]:
    """Convert ``"18 فروردین"`` plus a supplied year to ``(iso_jalali, timestamp)``.

    Price-history points carry no year, so the caller must supply one (see
    :func:`infer_series_years`). Invalid calendar days -- ``30 اسفند`` in a common
    year -- return ``(None, None)`` rather than being silently shifted.

    >>> parse_jalali_day_month("18 فروردین", year=1405)[0]
    '1405-01-18'
    """
    if not text:
        return None, None
    ascii_text = convert_digits(normalize_text(text) or "")
    day_match = re.search(r"\d{1,2}", ascii_text)
    month = jalali_month_index(ascii_text)
    if not day_match or month is None:
        return None, None
    day = int(day_match.group(0))
    label = f"{year:04d}-{month:02d}-{day:02d}"
    try:
        import jdatetime

        gregorian = jdatetime.date(year, month, day).togregorian()
        ts = datetime(gregorian.year, gregorian.month, gregorian.day).timestamp()
    except Exception:
        return label, None
    return label, ts


def infer_series_years(
    date_texts: Sequence[str], *, anchor_jalali_year: int, newest_last: bool = True
) -> list[int]:
    """Assign a Jalali year to each year-less day-month point in a series.

    A ~94-point daily series spans roughly three months, so it crosses at most
    one new-year boundary. Walking from the newest point backwards, the year is
    decremented whenever the month index *increases* going back in time (i.e. the
    sequence has wrapped past Farvardin).

    The result is inherently inferential, which is why callers record it per row
    via ``price_points.year_inferred``.
    """
    months = [jalali_month_index(t) for t in date_texts]
    order = range(len(months) - 1, -1, -1) if newest_last else range(len(months))
    years: dict[int, int] = {}
    year = anchor_jalali_year
    previous: int | None = None
    for position in order:
        month = months[position]
        if month is not None:
            if previous is not None and month > previous:
                year -= 1
            previous = month
        years[position] = year
    return [years[i] for i in range(len(months))]


def parse_iso_ts(text: str | None) -> float | None:
    """Parse Bama's ``modified_date`` (``2026-07-27T20:35:47.08``) to a timestamp.

    Interpreted in local time, which is the site's own clock (Asia/Tehran); the
    assumption is documented in the README rather than hidden here.
    """
    if not text:
        return None
    cleaned = convert_digits(text.strip()).replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).timestamp()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------

_SLUG_SAFE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def _slug_lookup_forms(text: str) -> tuple[str, ...]:
    """Candidate spellings for a slug-map lookup.

    Bama's own group titles are inconsistent about ZWNJ -- ``سیستم‌‌ کمکی راننده``
    ships with *two* consecutive ZWNJ characters where the item keys use one --
    so matching must be insensitive to ZWNJ runs and to spacing.
    """
    collapsed = re.sub("‌+", "‌", text)
    no_zwnj = text.replace("‌", "")
    squeezed = re.sub(r"\s+", " ", no_zwnj).strip()
    return (text, collapsed, no_zwnj, squeezed)


def _lookup_slug(text: str, table: dict[str, str]) -> str | None:
    """Find ``text`` in a slug table, tolerating ZWNJ and spacing variance."""
    normalized_table = {re.sub(r"\s+", " ", k.replace("‌", "")).strip(): v for k, v in table.items()}
    for form in _slug_lookup_forms(text):
        if form in table:
            return table[form]
    for form in _slug_lookup_forms(text):
        key = re.sub(r"\s+", " ", form.replace("‌", "")).strip()
        if key in normalized_table:
            return normalized_table[key]
    return None


def slugify_fa(text: str | None, *, table: dict[str, str] | None = None) -> str:
    """Map a Persian spec key to a stable ASCII slug.

    ``trim_specs.item_slug`` is part of a primary key, so stability across runs
    is mandatory. Reviewed keys come from :data:`SPEC_KEY_SLUGS` (or the supplied
    ``table``); anything else gets a deterministic ``unk_<hash>`` so it is never
    lost, and the audit reports it so a human can add the mapping.
    """
    normalized = normalize_text(text or "") or ""
    if not normalized:
        return "unk_empty"
    mapped = _lookup_slug(normalized, table if table is not None else SPEC_KEY_SLUGS)
    if mapped:
        return mapped
    # The ASCII fallback is only valid for a genuinely ASCII key. Applying it to
    # a Persian key strips the letters and leaves whatever digits happen to be in
    # it -- "خروجی 12 ولت" would become the slug "12", which is both unreadable
    # and liable to collide. Anything containing Persian falls through to a hash.
    if normalized.isascii():
        ascii_attempt = _SLUG_SAFE_RE.sub("_", normalized.lower()).strip("_")
        if ascii_attempt:
            return ascii_attempt
    # Hash the ZWNJ-free form so the same key never yields two different slugs.
    stable = re.sub(r"\s+", " ", normalized.replace("‌", "")).strip()
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:10]
    return f"unk_{digest}"
