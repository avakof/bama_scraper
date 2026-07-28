"""Privacy scrubbing.

Two distinct jobs:

1. :func:`strip_keys` removes contact-information *keys* from a payload before it
   is persisted. Bama's detail payload carries a server-masked seller mobile
   (``content.phone`` in the HTML, ``data.detail.phone`` in the JSON API). Even
   masked it is a personal identifier, so it never reaches disk.

2. :func:`scrub_freetext` redacts contact details that sellers type *into* the
   free-text description, where no key-based filter can reach them.

The false-positive risk is real and is guarded by tests: prices
(``1,000,000,000``), years (``1403``) and mileages (``100,000``) must survive
untouched, so the phone patterns are deliberately anchored on Iranian mobile
prefixes rather than "any long digit run".
"""

from __future__ import annotations

import re
from typing import Any, Final

from bama_scraper.detail_parser import _PRIVATE_KEYS
from bama_scraper.normalization import convert_digits

#: Contact keys removed at any depth. Extends the base scraper's set.
DEEP_PRIVATE_KEYS: Final[frozenset[str]] = frozenset(_PRIVATE_KEYS) | frozenset(
    {
        "seller_phone",
        "sellerPhone",
        "phone_number",
        "phoneNumber",
        "mobile_number",
        "mobileNumber",
        "whatsapp",
        "telegram",
        "instagram_id",
        "email",
        "contact_info",
        "contactInfo",
    }
)

REDACTION: Final[str] = "[redacted]"

#: Iranian mobile numbers: 09xxxxxxxxx / +989xxxxxxxxx / 00989..., tolerating
#: spaces, hyphens and dots as separators, and a trailing masked "XX".
#:
#: Both ends are guarded against hex characters. Without that guard a SHA-256
#: digest or an image UUID -- e.g. ``...eee7ed59112429361`` -- matches a mobile
#: pattern purely by chance; measured on the real corpus that produced 5 false
#: positives against 2 genuine hits.
_HEX_BEFORE: Final[str] = r"(?<![0-9A-Fa-f])"
_HEX_AFTER: Final[str] = r"(?![0-9A-Fa-f])"

_PHONE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        _HEX_BEFORE
        + r"(?:\+98|0098)?[\s.\-]?0?9\d{2}[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}"
        + _HEX_AFTER
    ),
    re.compile(_HEX_BEFORE + r"0?9\d{2}[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?[Xx]{2}\b"),
    # Landlines written with an explicit area code, e.g. 021-12345678
    re.compile(_HEX_BEFORE + r"0\d{2}[\s.\-]\d{7,8}" + _HEX_AFTER),
)

#: Payload keys whose values are (or embed) seller free text and therefore need
#: redaction, not just key removal. ``meta_description`` matters because Bama
#: builds its SEO description by concatenating the seller's own description --
#: verified on ad ``cuaxpwpu``, where the phone number appeared in both.
FREETEXT_KEYS: Final[frozenset[str]] = frozenset(
    {"description", "title_tag", "pageTitle", "slogan", "keywords", "title", "subtitle"}
)

_HANDLE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"@[A-Za-z0-9_]{3,}"),
    re.compile(r"(?:https?://)?(?:t\.me|telegram\.me|wa\.me|instagram\.com)/[A-Za-z0-9_.\-/]+"),
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
)


def strip_keys(obj: Any, *, keys: frozenset[str] = DEEP_PRIVATE_KEYS) -> Any:
    """Recursively remove contact keys from dicts and lists."""
    if isinstance(obj, dict):
        return {k: strip_keys(v, keys=keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [strip_keys(v, keys=keys) for v in obj]
    return obj


def scrub_freetext(text: str | None) -> tuple[str | None, int]:
    """Redact phone numbers and messaging handles from seller free text.

    Returns ``(scrubbed_text, redaction_count)``. Idempotent: re-running over an
    already-scrubbed string makes no further changes.

    Digits are compared in ASCII form so Persian-numeral phone numbers are caught,
    but the *original* text is what gets edited, so untouched Persian content
    keeps its original numerals.

    >>> scrub_freetext("تماس ۰۹۱۲۳۴۵۶۷۸۹")[1]
    1
    >>> scrub_freetext("قیمت 1,000,000,000 تومان")[1]
    0
    """
    if not text:
        return text, 0

    count = 0
    result = text

    # Phone detection needs ASCII digits, but edits must apply to the original
    # string. Work on a converted copy to locate spans, then splice by offset --
    # convert_digits is a 1:1 character mapping, so offsets are preserved.
    ascii_view = convert_digits(result)
    spans: list[tuple[int, int]] = []
    for pattern in _PHONE_PATTERNS:
        for match in pattern.finditer(ascii_view):
            # Ignore matches that are part of a grouped number like 1,000,000,000
            if "," in ascii_view[max(0, match.start() - 1) : match.end() + 1]:
                continue
            spans.append((match.start(), match.end()))

    for start, end in sorted(spans, reverse=True):
        if REDACTION in result[start:end]:
            continue
        result = result[:start] + REDACTION + result[end:]
        count += 1

    for pattern in _HANDLE_PATTERNS:
        result, hits = pattern.subn(REDACTION, result)
        count += hits

    return result, count


def scrub_payload(obj: Any, *, keys: frozenset[str] = DEEP_PRIVATE_KEYS) -> tuple[Any, int]:
    """Strip contact keys *and* redact contact details inside free-text values.

    Applied before any raw payload is persisted, so the stored evidence blobs
    cannot carry a phone number that a seller typed into a description.
    Returns ``(cleaned, redaction_count)``.
    """
    total = 0

    def walk(node: Any, key_hint: str | None = None) -> Any:
        nonlocal total
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items() if k not in keys}
        if isinstance(node, list):
            return [walk(v, key_hint) for v in node]
        if isinstance(node, str) and key_hint in FREETEXT_KEYS:
            cleaned, hits = scrub_freetext(node)
            total += hits
            return cleaned
        return node

    return walk(obj), total


def assert_no_contact(blob: str) -> list[str]:
    """Return the contact-shaped fragments found in a serialized blob.

    Used by the audit as a final gate on what actually reached disk. An empty
    list means clean.
    """
    ascii_view = convert_digits(blob)
    findings: list[str] = []
    for pattern in _PHONE_PATTERNS[:2]:
        findings.extend(match.group(0) for match in pattern.finditer(ascii_view))
    for key in ('"phone"', "'phone'", '"mobile"'):
        if key in blob:
            findings.append(key)
    return findings
