"""Seed the deep queue from the existing scraper's inventory.

The source database is opened **read-only** via a URI connection, and its SHA-256
is recorded in ``deep_runs`` so a test (and the audit) can prove the original
dataset was not modified by any deep run.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from bama_scraper.discovery import extract_ad_id

from .endpoints import ad_api_url, ad_html_url
from .storage import DeepStorage


class SeedError(RuntimeError):
    """The source inventory is missing or unusable."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_source_readonly(path: str | Path) -> sqlite3.Connection:
    """Open the existing scraper database read-only.

    ``mode=ro`` makes any write attempt raise, which is the point: the deep
    scraper must never touch the dataset it reads from.
    """
    source = Path(path)
    if not source.exists():
        raise SeedError(f"source database not found: {source}")
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def seed_from_source(
    storage: DeepStorage, source_db: str | Path, *, only_completed: bool = False
) -> dict[str, int]:
    """Copy the discovered-ad inventory into ``deep_queue``.

    Idempotent: re-seeding adds nothing. By default every discovered ad is seeded
    including the ones the original run could not fetch (HTTP 410), because the
    deep scraper should re-check them and record them as ``delisted`` rather than
    inherit a stale verdict.
    """
    conn = open_source_readonly(source_db)
    try:
        where = "WHERE status='completed'" if only_completed else ""
        rows = list(conn.execute(f"SELECT url, ad_id, status FROM discovered_ads {where}"))
    except sqlite3.Error as exc:  # pragma: no cover - malformed source
        raise SeedError(f"cannot read discovered_ads: {exc}") from exc
    finally:
        conn.close()

    payload: list[tuple[str, str, str, str]] = []
    skipped = 0
    for row in rows:
        url = ad_html_url(row["url"])
        code = row["ad_id"] or extract_ad_id(url)
        if not code:
            skipped += 1
            continue
        payload.append((code, url, ad_api_url(code), row["status"] or ""))

    inserted = storage.seed_queue(payload)
    return {
        "source_rows": len(rows),
        "seeded": inserted,
        "already_present": len(payload) - inserted,
        "skipped_no_code": skipped,
        "queue_total": storage.count("deep_queue"),
    }
