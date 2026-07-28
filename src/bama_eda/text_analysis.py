"""Description text analysis, on scrubbed text only.

Two rules:

1. **Never analyse unscrubbed descriptions.** Sellers type phone numbers and
   messaging handles into the free-text field. The deep scraper produces a
   scrubbed variant; this module prefers it and re-scrubs defensively before
   counting anything, because a token frequency table is still a disclosure.
2. **No sentiment or quality claims.** Counting words does not measure how good an
   advertisement is, and this module makes no such claim.

Persian normalization follows the scraper's own rules, so ZWNJ survives and Arabic
character variants fold to Persian — otherwise ``كتاب`` and ``کتاب`` count as two
different words.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from .models import is_texty

#: Contact-shaped patterns removed before any counting.
_SCRUB_PATTERNS = (
    re.compile(r"(?<![0-9a-fA-F])0?9\d{9}(?![0-9a-fA-F])"),  # mobile numbers
    re.compile(r"\b0\d{2,3}[-\s]?\d{7,8}\b"),  # landlines
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"@[A-Za-z0-9_]{3,}"),  # handles
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),  # e-mail
    re.compile(r"(?:t\.me|telegram|whatsapp|واتساپ|تلگرام)\S*", re.IGNORECASE),
)

#: Persian stop words: too frequent to be informative in a term list.
_STOP_WORDS = frozenset(
    """
    و در به از با این که را برای های ها یک می بی تا هم یا بر است شده بدون
    فقط خیلی کاملا بسیار دارای مورد نظر لطفا تماس عدد سال کارکرد رنگ
    """.split()
)

#: Boilerplate phrases that appear across many unrelated listings.
_BOILERPLATE = (
    "بدون خط و خش",
    "دارای ضمانت",
    "قیمت کارشناسی",
    "معاوضه",
    "تماس بگیرید",
)


def scrub(text: Any) -> str | None:
    """Remove contact-shaped content. Defensive: the input may already be scrubbed."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text
    for pattern in _SCRUB_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip() or None


def normalize(text: str) -> str:
    """Apply the scraper's Persian normalization so tokens are comparable."""
    try:
        from bama_scraper.normalization import normalize_text

        return normalize_text(text) or ""
    except Exception:  # noqa: BLE001 - normalization is an enhancement, not a gate
        return text


def tokenize(text: str) -> list[str]:
    """Whitespace/punctuation tokenizer that keeps ZWNJ-joined Persian words whole."""
    normalized = normalize(text)
    # ‌ (ZWNJ) is part of a word in Persian orthography, so it is not a split.
    tokens = re.split(r"[^\w‌]+", normalized)
    return [t for t in tokens if len(t) > 1 and not t.isdigit() and t not in _STOP_WORDS]


def _series(cross: pd.DataFrame) -> pd.Series:
    for column in ("deep_description_scrubbed", "snap_description", "deep_description"):
        if column in cross.columns:
            return cross[column]
    return pd.Series(dtype=object)


def describe_descriptions(cross: pd.DataFrame) -> dict[str, Any]:
    """Availability and size of the description field."""
    raw = _series(cross)
    if raw.empty:
        return {"available": False, "note": "no description column in the dataset"}

    cleaned = raw.map(scrub)
    present = cleaned.dropna()
    if present.empty:
        return {
            "available": True,
            "with_description": 0,
            "note": "no non-empty description survived scrubbing",
        }

    chars = present.map(len)
    tokens = present.map(lambda t: len(tokenize(t)))
    sentences = present.map(lambda t: len([s for s in re.split(r"[.!?\n۔؟]+", t) if s.strip()]))
    return {
        "available": True,
        "advertisements": int(len(cross)),
        "with_description": int(len(present)),
        "coverage_pct": round(len(present) / len(cross) * 100, 3),
        "character_count": {
            "median": float(chars.median()),
            "p25": float(chars.quantile(0.25)),
            "p75": float(chars.quantile(0.75)),
            "max": int(chars.max()),
        },
        "token_count": {
            "median": float(tokens.median()),
            "p25": float(tokens.quantile(0.25)),
            "p75": float(tokens.quantile(0.75)),
            "max": int(tokens.max()),
        },
        "sentence_count_median": float(sentences.median()),
        "privacy_note": (
            "analysed on scrubbed text; contact-shaped sequences are removed before "
            "any counting, because a term-frequency table is still a disclosure"
        ),
    }


def frequent_terms(cross: pd.DataFrame, *, top: int = 40, ngram: int = 1) -> pd.DataFrame:
    """Most frequent normalized terms, bigrams or trigrams."""
    raw = _series(cross)
    if raw.empty:
        return pd.DataFrame(columns=["term", "occurrences", "advertisements", "ngram"])

    counter: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    documents = 0
    for text in raw.dropna():
        cleaned = scrub(text)
        if not cleaned:
            continue
        documents += 1
        tokens = tokenize(cleaned)
        grams = (
            tokens
            if ngram == 1
            else [" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
        )
        counter.update(grams)
        document_frequency.update(set(grams))

    if not counter:
        return pd.DataFrame(columns=["term", "occurrences", "advertisements", "ngram"])
    rows = [
        {
            "term": term,
            "occurrences": int(count),
            "advertisements": int(document_frequency[term]),
            "advertisement_share_pct": round(document_frequency[term] / documents * 100, 3)
            if documents
            else None,
            "ngram": ngram,
        }
        for term, count in counter.most_common(top)
    ]
    return pd.DataFrame(rows)


def duplicate_descriptions(cross: pd.DataFrame, *, min_length: int = 40) -> dict[str, Any]:
    """Exactly repeated descriptions, which usually mean one seller or a template."""
    raw = _series(cross)
    if raw.empty:
        return {"available": False}
    cleaned = raw.map(scrub).dropna()
    cleaned = cleaned[cleaned.map(len) >= min_length]
    if cleaned.empty:
        return {"available": True, "duplicated_texts": 0}

    counts = cleaned.map(normalize).value_counts()
    duplicated = counts[counts > 1]
    return {
        "available": True,
        "descriptions_considered": int(len(cleaned)),
        "distinct_texts": int(len(counts)),
        "duplicated_texts": int(len(duplicated)),
        "advertisements_sharing_a_duplicate": int(duplicated.sum()),
        "largest_duplicate_group": int(duplicated.max()) if len(duplicated) else 0,
        "note": (
            "identical descriptions usually indicate one seller listing several cars, "
            "or a dealer template; they are not on their own evidence of a repost"
        ),
    }


def boilerplate_frequency(cross: pd.DataFrame) -> pd.DataFrame:
    """How often each known boilerplate phrase appears."""
    raw = _series(cross)
    if raw.empty:
        return pd.DataFrame(columns=["phrase", "advertisements", "share_pct"])
    cleaned = raw.map(scrub).dropna().map(normalize)
    total = len(cleaned)
    rows = [
        {
            "phrase": phrase,
            "advertisements": int(cleaned.str.contains(normalize(phrase), regex=False).sum()),
            "share_pct": round(cleaned.str.contains(normalize(phrase), regex=False).mean() * 100, 3)
            if total
            else None,
        }
        for phrase in _BOILERPLATE
    ]
    return pd.DataFrame(rows).sort_values("advertisements", ascending=False)


def description_length_by_group(
    cross: pd.DataFrame, group_column: str, *, min_group_size: int = 10
) -> pd.DataFrame:
    """Median description length per group, suppressed below the threshold."""
    raw = _series(cross)
    if raw.empty or group_column not in cross.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "group": cross[group_column].astype(str),
            "length": raw.map(scrub).map(lambda t: len(t) if t else None),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("group")["length"].agg(
        advertisements="count", median_length="median", p75_length=lambda s: s.quantile(0.75)
    )
    out = grouped.reset_index()
    out["sufficient_sample"] = out["advertisements"] >= min_group_size
    out["group_column"] = group_column
    return out.sort_values("advertisements", ascending=False).reset_index(drop=True)


def assert_no_contact_leak(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Final privacy sweep over text outputs."""
    pattern = re.compile(r"(?<![0-9a-fA-F])09\d{9}(?![0-9a-fA-F])")
    leaks: list[str] = []
    for name, frame in frames.items():
        if frame.empty:
            continue
        for column in frame.columns:
            if not is_texty(frame[column]):
                continue
            series = frame[column].dropna().astype(str)
            if series.empty:
                continue
            hits = int(series.str.contains(pattern, regex=True).sum())
            if hits:
                leaks.append(f"{name}.{column}: {hits}")
    return leaks
