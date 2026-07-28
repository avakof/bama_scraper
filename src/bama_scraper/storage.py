"""Durable SQLite storage: the single source of truth for a run.

The database is designed so that any process crash leaves a consistent,
resumable state: discovery and detail scraping both write through idempotent
upserts, and per-ad status transitions are committed immediately.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .models import AdDetail, AdMedia, DiscoveredAd, NetworkCandidate

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    target_url       TEXT NOT NULL,
    applied_filters  TEXT,
    mode             TEXT,
    scraper_version  TEXT,
    termination_reason TEXT,
    cycles           INTEGER DEFAULT 0,
    stale_cycles_at_end INTEGER DEFAULT 0,
    summary_json     TEXT
);

CREATE TABLE IF NOT EXISTS discovered_ads (
    url                TEXT PRIMARY KEY,
    ad_id              TEXT NOT NULL,
    source_search_url  TEXT,
    card_title         TEXT,
    card_subtitle      TEXT,
    price_text         TEXT,
    price_toman        INTEGER,
    price_type         TEXT,
    year_text          TEXT,
    year_jalali        INTEGER,
    year_gregorian     INTEGER,
    mileage_text       TEXT,
    mileage_km         INTEGER,
    location_text      TEXT,
    thumbnail_url      TEXT,
    is_promoted        INTEGER,
    discovery_position INTEGER,
    discovery_cycle    INTEGER,
    discovered_at      TEXT,
    first_seen_run     TEXT,
    times_seen         INTEGER DEFAULT 1,
    status             TEXT NOT NULL DEFAULT 'discovered',
    attempts           INTEGER DEFAULT 0,
    last_error         TEXT,
    updated_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_disc_status ON discovered_ads(status);
CREATE INDEX IF NOT EXISTS idx_disc_adid   ON discovered_ads(ad_id);

CREATE TABLE IF NOT EXISTS ad_details (
    ad_id        TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    payload      TEXT NOT NULL,
    parse_status TEXT,
    scraped_at   TEXT,
    html_sha256  TEXT
);

CREATE TABLE IF NOT EXISTS ad_media (
    ad_id        TEXT NOT NULL,
    position     INTEGER NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'image',
    url          TEXT,
    original_url TEXT,
    thumb_url    TEXT,
    alt          TEXT,
    PRIMARY KEY (ad_id, position, kind)
);

CREATE TABLE IF NOT EXISTS raw_attributes (
    ad_id      TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT,
    PRIMARY KEY (ad_id, key)
);

CREATE TABLE IF NOT EXISTS scrape_errors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    ad_id      TEXT,
    url        TEXT,
    stage      TEXT,
    error_type TEXT,
    message    TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS network_candidates (
    url                 TEXT PRIMARY KEY,
    method              TEXT,
    status              INTEGER,
    resource_type       TEXT,
    is_listing_endpoint INTEGER,
    top_level_keys      TEXT,
    ad_count            INTEGER,
    notes               TEXT,
    seen_at             TEXT
);

CREATE TABLE IF NOT EXISTS discovery_checkpoint (
    run_id      TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT,
    updated_at  TEXT,
    PRIMARY KEY (run_id, key)
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Storage:
    """Thin, explicit SQLite wrapper. No ORM, no hidden magic."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- runs ---------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        target_url: str,
        applied_filters: dict[str, Any],
        mode: str,
        version: str,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, started_at, target_url, applied_filters, mode, scraper_version)
               VALUES (?,?,?,?,?,?)""",
            (
                run_id,
                _now(),
                target_url,
                json.dumps(applied_filters, ensure_ascii=False),
                mode,
                version,
            ),
        )
        self.conn.commit()

    def finish_run(
        self,
        run_id: str,
        termination_reason: str,
        cycles: int,
        stale: int,
        summary: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """UPDATE runs SET finished_at=?, termination_reason=?, cycles=?,
               stale_cycles_at_end=?, summary_json=? WHERE run_id=?""",
            (
                _now(),
                termination_reason,
                cycles,
                stale,
                json.dumps(summary, ensure_ascii=False, default=str),
                run_id,
            ),
        )
        self.conn.commit()

    # -- discovery ----------------------------------------------------------

    def upsert_discovered(self, ads: Sequence[DiscoveredAd], run_id: str) -> int:
        """Insert new ads, refresh card fields for known ones.

        Returns the number of URLs that were not already present, which is the
        "new this cycle" figure used by the termination algorithm.
        """
        if not ads:
            return 0
        urls = [a.url for a in ads]
        placeholders = ",".join("?" * len(urls))
        existing = {
            r["url"]
            for r in self.conn.execute(
                f"SELECT url FROM discovered_ads WHERE url IN ({placeholders})", urls
            )
        }
        rows = [
            (
                a.url,
                a.ad_id,
                a.source_search_url,
                a.card_title,
                a.card_subtitle,
                a.price_text,
                a.price_toman,
                a.price_type,
                a.year_text,
                a.year_jalali,
                a.year_gregorian,
                a.mileage_text,
                a.mileage_km,
                a.location_text,
                a.thumbnail_url,
                int(a.is_promoted) if a.is_promoted is not None else None,
                a.discovery_position,
                a.discovery_cycle,
                a.discovered_at,
                run_id,
                _now(),
            )
            for a in ads
        ]
        self.conn.executemany(
            """INSERT INTO discovered_ads
               (url, ad_id, source_search_url, card_title, card_subtitle, price_text,
                price_toman, price_type, year_text, year_jalali, year_gregorian,
                mileage_text, mileage_km, location_text, thumbnail_url, is_promoted,
                discovery_position, discovery_cycle, discovered_at, first_seen_run, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 times_seen    = times_seen + 1,
                 card_title    = COALESCE(excluded.card_title, card_title),
                 price_text    = COALESCE(excluded.price_text, price_text),
                 price_toman   = COALESCE(excluded.price_toman, price_toman),
                 is_promoted   = COALESCE(excluded.is_promoted, is_promoted),
                 updated_at    = excluded.updated_at""",
            rows,
        )
        self.conn.commit()
        return len(set(urls) - existing)

    def count_discovered(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM discovered_ads").fetchone()[0])

    def all_discovered_urls(self) -> set[str]:
        return {r["url"] for r in self.conn.execute("SELECT url FROM discovered_ads")}

    def set_checkpoint(self, run_id: str, key: str, value: Any) -> None:
        self.conn.execute(
            """INSERT INTO discovery_checkpoint (run_id, key, value, updated_at)
               VALUES (?,?,?,?) ON CONFLICT(run_id, key)
               DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (run_id, key, json.dumps(value, ensure_ascii=False, default=str), _now()),
        )
        self.conn.commit()

    def latest_discovery_result(self) -> dict[str, Any] | None:
        """Most recent stored discovery outcome, across all runs.

        Lets ``validate`` report termination evidence even when discovery ran in
        an earlier, separate invocation.
        """
        row = self.conn.execute(
            "SELECT value FROM discovery_checkpoint WHERE key='discovery_result' "
            "ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if not row or row["value"] is None:
            return None
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def get_checkpoint(self, run_id: str, key: str) -> Any:
        row = self.conn.execute(
            "SELECT value FROM discovery_checkpoint WHERE run_id=? AND key=?", (run_id, key)
        ).fetchone()
        return json.loads(row["value"]) if row and row["value"] is not None else None

    # -- work queue ---------------------------------------------------------

    def claim_pending(
        self, limit: int, *, refresh: bool = False, max_attempts: int = 4
    ) -> list[sqlite3.Row]:
        """Return ads that still need a detail fetch.

        ``retryable_error`` rows are re-queued until ``max_attempts``; rows left
        in ``scraping`` by a crashed process are recovered here as well.
        """
        if refresh:
            where = "1=1"
            params: tuple[Any, ...] = ()
        else:
            where = (
                "(status IN ('discovered','pending','scraping') "
                " OR (status='retryable_error' AND attempts < ?))"
            )
            params = (max_attempts,)
        return list(
            self.conn.execute(
                f"SELECT url, ad_id, attempts FROM discovered_ads WHERE {where} "
                "ORDER BY discovery_cycle, discovery_position LIMIT ?",
                (*params, limit),
            )
        )

    def all_ads(self) -> list[sqlite3.Row]:
        """Every discovered advertisement in discovery order.

        Used by ``--refresh``, which must walk a fixed snapshot: a status-driven
        claim loop cannot terminate when status is ignored.
        """
        return list(
            self.conn.execute(
                "SELECT url, ad_id, attempts FROM discovered_ads "
                "ORDER BY discovery_cycle, discovery_position"
            )
        )

    def recover_stuck(self) -> int:
        """Reset rows abandoned mid-flight by an interrupted process."""
        cur = self.conn.execute(
            "UPDATE discovered_ads SET status='pending' WHERE status='scraping'"
        )
        self.conn.commit()
        return cur.rowcount

    def mark_status(
        self, url: str, status: str, error: str | None = None, bump_attempt: bool = False
    ) -> None:
        self.conn.execute(
            f"""UPDATE discovered_ads
                SET status=?, last_error=?, updated_at=?
                    {", attempts = attempts + 1" if bump_attempt else ""}
                WHERE url=?""",
            (status, error, _now(), url),
        )
        self.conn.commit()

    def mark_many_scraping(self, urls: Sequence[str]) -> None:
        if not urls:
            return
        self.conn.executemany(
            "UPDATE discovered_ads SET status='scraping', updated_at=? WHERE url=?",
            [(_now(), u) for u in urls],
        )
        self.conn.commit()

    # -- details ------------------------------------------------------------

    def save_detail(self, detail: AdDetail, media: Iterable[AdMedia]) -> None:
        payload = detail.model_dump(mode="json")
        self.conn.execute(
            """INSERT INTO ad_details (ad_id, url, payload, parse_status, scraped_at, html_sha256)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(ad_id) DO UPDATE SET
                 url=excluded.url, payload=excluded.payload,
                 parse_status=excluded.parse_status, scraped_at=excluded.scraped_at,
                 html_sha256=excluded.html_sha256""",
            (
                detail.ad_id,
                detail.url,
                json.dumps(payload, ensure_ascii=False),
                detail.parse_status,
                detail.scraped_at,
                detail.html_sha256,
            ),
        )
        media_list = list(media)
        if media_list:
            self.conn.execute("DELETE FROM ad_media WHERE ad_id=?", (detail.ad_id,))
            self.conn.executemany(
                """INSERT OR REPLACE INTO ad_media
                   (ad_id, position, kind, url, original_url, thumb_url, alt)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (m.ad_id, m.position, m.kind, m.url, m.original_url, m.thumb_url, m.alt)
                    for m in media_list
                ],
            )
        if detail.raw_attributes:
            self.conn.executemany(
                "INSERT OR REPLACE INTO raw_attributes (ad_id, key, value) VALUES (?,?,?)",
                [
                    (
                        detail.ad_id,
                        k,
                        json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
                    )
                    for k, v in detail.raw_attributes.items()
                ],
            )
        self.conn.commit()

    def iter_details(self) -> Iterable[dict[str, Any]]:
        for row in self.conn.execute("SELECT payload FROM ad_details ORDER BY ad_id"):
            yield json.loads(row["payload"])

    # -- errors / network ---------------------------------------------------

    def log_error(
        self,
        run_id: str,
        ad_id: str | None,
        url: str | None,
        stage: str,
        error_type: str,
        message: str,
    ) -> None:
        self.conn.execute(
            """INSERT INTO scrape_errors (run_id, ad_id, url, stage, error_type, message, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, ad_id, url, stage, error_type, message[:2000], _now()),
        )
        self.conn.commit()

    def save_network_candidates(self, candidates: Iterable[NetworkCandidate]) -> None:
        rows = [
            (
                c.url,
                c.method,
                c.status,
                c.resource_type,
                int(c.is_listing_endpoint),
                json.dumps(c.top_level_keys, ensure_ascii=False),
                c.ad_count,
                c.notes,
                _now(),
            )
            for c in candidates
        ]
        if not rows:
            return
        self.conn.executemany(
            """INSERT OR REPLACE INTO network_candidates
               (url, method, status, resource_type, is_listing_endpoint,
                top_level_keys, ad_count, notes, seen_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        c = self.conn
        one = lambda q, *p: c.execute(q, p).fetchone()[0]  # noqa: E731
        return {
            "discovered": one("SELECT COUNT(*) FROM discovered_ads"),
            "distinct_ad_ids": one("SELECT COUNT(DISTINCT ad_id) FROM discovered_ads"),
            "completed": one("SELECT COUNT(*) FROM discovered_ads WHERE status='completed'"),
            "pending": one(
                "SELECT COUNT(*) FROM discovered_ads WHERE status IN "
                "('discovered','pending','scraping')"
            ),
            "retryable": one("SELECT COUNT(*) FROM discovered_ads WHERE status='retryable_error'"),
            "permanent": one("SELECT COUNT(*) FROM discovered_ads WHERE status='permanent_error'"),
            "details": one("SELECT COUNT(*) FROM ad_details"),
            "media": one("SELECT COUNT(*) FROM ad_media"),
            "errors": one("SELECT COUNT(*) FROM scrape_errors"),
        }
