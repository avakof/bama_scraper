"""Durable SQLite storage for the deep scraper.

Conventions deliberately mirror ``bama_scraper.storage``: WAL, one schema string
executed via ``executescript``, ``CREATE TABLE IF NOT EXISTS``, explicit upserts,
no ORM. Every phase has its own queue with its own status column so a run is
restartable at any point.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- ------------------------------------------------------------------ bookkeeping
CREATE TABLE IF NOT EXISTS deep_runs (
    run_id            TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    phases            TEXT,
    source_db         TEXT NOT NULL,
    source_db_sha256  TEXT,
    deep_version      TEXT,
    config_json       TEXT,
    requests_issued   INTEGER DEFAULT 0,
    termination_reason TEXT,
    summary_json      TEXT
);

CREATE TABLE IF NOT EXISTS deep_checkpoint (
    run_id     TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT,
    updated_at TEXT,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS deep_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    phase       TEXT,
    entity_kind TEXT,
    entity_key  TEXT,
    url         TEXT,
    stage       TEXT,
    error_type  TEXT,
    http_status INTEGER,
    message     TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_deep_err_entity ON deep_errors(entity_kind, entity_key);

-- -------------------------------------------------------------- phase A: ads
CREATE TABLE IF NOT EXISTS deep_queue (
    ad_id         TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    api_url       TEXT NOT NULL,
    source_status TEXT,
    seeded_at     TEXT,
    -- pending|fetching|completed|partial|delisted|retryable_error|permanent_error
    status        TEXT NOT NULL DEFAULT 'pending',
    api_status    TEXT,
    html_status   TEXT,
    attempts      INTEGER DEFAULT 0,
    last_error    TEXT,
    completed_at  TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_deep_queue_status ON deep_queue(status);

CREATE TABLE IF NOT EXISTS ad_deep (
    ad_id TEXT PRIMARY KEY,
    url TEXT NOT NULL, canonical_url TEXT, api_url TEXT,
    -- identity
    code TEXT, ad_type TEXT, title TEXT, subtitle TEXT,
    brand TEXT, brand_fa TEXT, model TEXT, model_fa TEXT,
    trim TEXT, trim_en TEXT, trim_fa TEXT,
    year_text TEXT, year_jalali INTEGER, year_gregorian INTEGER,
    body_type TEXT, body_type_fa TEXT, body_value TEXT, body_display_name TEXT,
    vehicle_category TEXT, ad_class_id TEXT,
    -- price and sale terms
    price_text TEXT, price_toman INTEGER, price_type TEXT,
    is_negotiable INTEGER, price_hidden INTEGER, is_installment INTEGER,
    down_payment_toman INTEGER, prepayment_secondary_toman INTEGER,
    installment_amount_toman INTEGER, installment_months INTEGER,
    installments_count INTEGER, installment_total_toman INTEGER,
    is_pre_sale INTEGER, pre_sale_delivery TEXT, pre_sale_delivery_value TEXT,
    delivery_days INTEGER,
    collaboration_price_toman INTEGER,
    -- condition
    mileage_text TEXT, mileage_km INTEGER, is_zero_km INTEGER,
    condition_new_used TEXT, condition_source TEXT,
    body_status TEXT, authenticated INTEGER, authenticity_json TEXT,
    -- powertrain: text is authoritative, REAL parsed, raw int kept as checksum
    transmission TEXT, fuel_type TEXT, engine TEXT, cylinder_fa TEXT, drivetrain TEXT,
    engine_volume_text TEXT,    engine_volume_l REAL,         engine_volume_raw INTEGER,
    power_text TEXT,            power_hp REAL,                power_raw INTEGER,
    torque_text TEXT,           torque_nm REAL,               torque_raw INTEGER,
    acceleration_text TEXT,     acceleration_s REAL,          acceleration_raw INTEGER,
    fuel_consumption_text TEXT, fuel_consumption_l100km REAL, fuel_consumption_raw INTEGER,
    battery_capacity_text TEXT, battery_capacity_kwh REAL,
    all_electric_range_text TEXT, all_electric_range_km REAL,
    numeric_agreement_json TEXT,
    body_color TEXT, inside_color TEXT, color_combined TEXT,
    -- place and time
    location_text TEXT, province TEXT, city TEXT, region TEXT,
    published_text TEXT, published_ts REAL, modified_date TEXT, modified_ts REAL,
    -- seller
    seller_type TEXT, dealer_id INTEGER, dealer_name TEXT, dealer_link TEXT,
    dealer_address TEXT, dealer_ad_count INTEGER,
    dealer_score REAL, dealer_score_raw INTEGER, dealer_score_heuristic INTEGER,
    dealer_activity_years INTEGER, dealer_activity_months INTEGER, dealer_activity_text TEXT,
    -- ad surface
    -- description holds the SCRUBBED text; contact fragments are never stored
    description TEXT, description_redactions INTEGER,
    badge INTEGER, is_promoted INTEGER, has_related INTEGER, rank INTEGER, pin INTEGER,
    specialcase TEXT, publish_networks TEXT, inspection_station_count INTEGER,
    life_styles_json TEXT, options_json TEXT, icon_json TEXT,
    stats_model_ad_count INTEGER, stats_trim_ad_count INTEGER, breadcrumb_json TEXT,
    meta_title TEXT, meta_description TEXT, meta_keywords TEXT,
    meta_canonical TEXT, meta_noindex INTEGER, meta_slogan TEXT,
    image_count INTEGER, media_count INTEGER, video_count INTEGER, primary_image_url TEXT,
    -- join keys
    review_url TEXT, review_key TEXT, price_url TEXT, price_key TEXT,
    -- evidence and provenance
    api_ok INTEGER, html_ok INTEGER, api_json_sha256 TEXT, html_sha256 TEXT,
    provenance_json TEXT, conflict_count INTEGER DEFAULT 0,
    api_raw_json TEXT, html_payload_json TEXT, json_ld_json TEXT,
    parse_status TEXT, parse_warnings_json TEXT,
    is_delisted INTEGER DEFAULT 0, delisted_at TEXT,
    deep_version TEXT, scraped_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ad_deep_review ON ad_deep(review_key);
CREATE INDEX IF NOT EXISTS idx_ad_deep_price  ON ad_deep(price_key);
CREATE INDEX IF NOT EXISTS idx_ad_deep_dealer ON ad_deep(dealer_id);

CREATE TABLE IF NOT EXISTS ad_media_deep (
    ad_id TEXT NOT NULL, kind TEXT NOT NULL, position INTEGER NOT NULL,
    original_url TEXT, large_url TEXT, small_url TEXT, thumb_url TEXT, pinkie_url TEXT,
    video_id TEXT, video_url TEXT, video_thumb TEXT, alt TEXT,
    source TEXT,
    PRIMARY KEY (ad_id, kind, position)
);

CREATE TABLE IF NOT EXISTS ad_raw_attributes (
    ad_id TEXT NOT NULL, source TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
    PRIMARY KEY (ad_id, source, key)
);

CREATE TABLE IF NOT EXISTS ad_field_conflicts (
    ad_id TEXT NOT NULL, field TEXT NOT NULL,
    api_value TEXT, html_value TEXT, ld_value TEXT,
    chosen_value TEXT, chosen_source TEXT,
    severity TEXT, detected_at TEXT,
    PRIMARY KEY (ad_id, field)
);
CREATE INDEX IF NOT EXISTS idx_conflict_field ON ad_field_conflicts(field, severity);

-- ------------------------------------------------------- phase B: trim specs
CREATE TABLE IF NOT EXISTS trim_reviews (
    review_key TEXT PRIMARY KEY,
    review_url TEXT NOT NULL,
    brand_slug TEXT, model_slug TEXT, trim_slug TEXT, extra_segment TEXT, trim_name TEXT,
    research_id INTEGER, model_id INTEGER, generation_id INTEGER,
    year INTEGER, to_year INTEGER, description TEXT,
    total_ad_count INTEGER, comments_count INTEGER, price_url TEXT, comparison_code TEXT,
    rating REAL, final_score REAL, after_sales_percent REAL, body_design_percent REAL,
    efficiency_percent REAL, capabilities_percent REAL, interior_design_percent REAL,
    pros_json TEXT, cons_json TEXT, images_json TEXT, trims_json TEXT, years_json TEXT,
    ad_count_in_dataset INTEGER,
    status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, last_error TEXT,
    raw_detail_json TEXT, raw_seo_json TEXT, fetched_at TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_trim_reviews_status ON trim_reviews(status);

CREATE TABLE IF NOT EXISTS trim_spec_groups (
    review_key TEXT NOT NULL, group_index INTEGER NOT NULL,
    group_name TEXT NOT NULL, group_slug TEXT, item_count INTEGER,
    PRIMARY KEY (review_key, group_index)
);

CREATE TABLE IF NOT EXISTS trim_specs (
    review_key TEXT NOT NULL, group_slug TEXT NOT NULL, item_slug TEXT NOT NULL,
    item_key TEXT NOT NULL, group_name TEXT NOT NULL, position INTEGER,
    value_type TEXT, value_raw TEXT, value_text TEXT,
    value_bool INTEGER, value_num REAL, value_unit TEXT,
    PRIMARY KEY (review_key, group_slug, item_slug)
);
CREATE INDEX IF NOT EXISTS idx_trim_specs_item ON trim_specs(item_slug, value_bool);

CREATE TABLE IF NOT EXISTS trim_specs_typed (
    review_key TEXT PRIMARY KEY, trim_name TEXT, rating REAL,
    acceleration_time_s REAL, weight_kg REAL, fuel_capacity_l REAL,
    fuel_consumption_l100km REAL, drive_wheel_configuration TEXT,
    number_of_airbags INTEGER, vehicle_transmission TEXT, speed_kmh REAL,
    engine_displacement_l REAL, engine_power_hp REAL, torque_nm REAL,
    price_toman INTEGER, date_published TEXT, date_modified TEXT,
    raw_json TEXT, fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS spec_key_catalog (
    item_slug TEXT PRIMARY KEY, item_key TEXT, group_slug TEXT, group_name TEXT,
    group_index INTEGER, position INTEGER, value_type TEXT,
    distinct_values INTEGER, coverage_review_keys INTEGER, first_seen TEXT
);

-- ------------------------------------------------------ phase C: price history
CREATE TABLE IF NOT EXISTS price_keys (
    price_key TEXT PRIMARY KEY,
    price_url TEXT NOT NULL, brand TEXT, model TEXT, trim TEXT,
    ad_count_in_dataset INTEGER, series_count INTEGER, point_count INTEGER,
    brand_fa TEXT, model_fa TEXT, trim_fa TEXT, last_update_text TEXT,
    status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, last_error TEXT,
    raw_json TEXT, fetched_at TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_price_keys_status ON price_keys(status);

CREATE TABLE IF NOT EXISTS price_series (
    series_id TEXT PRIMARY KEY,
    price_key TEXT NOT NULL, trim TEXT, trim_fa TEXT,
    model_year TEXT, model_year_jalali INTEGER, equipment_class TEXT,
    price_provider TEXT, provider_kind TEXT,
    latest_price_text TEXT, latest_price_toman INTEGER,
    price_diff_text TEXT, price_diff_toman INTEGER,
    price_date_text TEXT, price_date_jalali TEXT, price_date_ts REAL,
    point_count INTEGER, first_point_ts REAL, last_point_ts REAL,
    raw_detail_json TEXT, fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_price_series_key ON price_series(price_key);

CREATE TABLE IF NOT EXISTS price_points (
    series_id TEXT NOT NULL, point_index INTEGER NOT NULL,
    date_text TEXT NOT NULL, date_jalali TEXT, date_ts REAL, year_inferred INTEGER,
    price_text TEXT, price_toman INTEGER,
    PRIMARY KEY (series_id, point_index)
);
CREATE INDEX IF NOT EXISTS idx_price_points_date ON price_points(date_ts);

CREATE TABLE IF NOT EXISTS price_brands (
    brand_slug TEXT PRIMARY KEY, brand_fa TEXT, model_count INTEGER,
    raw_json TEXT, fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS price_hierarchy (
    path TEXT PRIMARY KEY, brand_slug TEXT, model_slug TEXT, trim_slug TEXT,
    title_fa TEXT, raw_json TEXT, fetched_at TEXT
);

-- ---------------------------------------------------------- phase D: dealers
CREATE TABLE IF NOT EXISTS dealers (
    dealer_id INTEGER PRIMARY KEY, dealer_url TEXT,
    title TEXT, dealer_type TEXT, package_type TEXT, banner_url TEXT, logo_url TEXT,
    address TEXT, latitude REAL, longitude REAL, is_active INTEGER, vehicle_type TEXT,
    score REAL, score_raw INTEGER,
    ad_count_reported INTEGER, activity_years INTEGER, activity_months INTEGER,
    activity_text TEXT,
    ads_listed INTEGER,
    ads_total_count_reported INTEGER,
    ad_count_in_dataset INTEGER,
    status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, last_error TEXT,
    raw_profile_json TEXT, raw_ads_json TEXT, fetched_at TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_dealers_status ON dealers(status);

CREATE TABLE IF NOT EXISTS dealer_ads (
    dealer_id INTEGER NOT NULL, ad_code TEXT NOT NULL, url TEXT,
    title TEXT, subtitle TEXT, price_text TEXT, price_toman INTEGER,
    year_text TEXT, mileage_text TEXT, mileage_km INTEGER,
    in_seed_inventory INTEGER, raw_json TEXT, seen_at TEXT,
    PRIMARY KEY (dealer_id, ad_code)
);
"""

#: Statuses that a resumed run must not re-fetch.
TERMINAL_AD_STATUSES = ("completed", "partial", "delisted", "permanent_error")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


class DeepStorage:
    """Explicit SQLite wrapper for the deep dataset."""

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

    def __enter__(self) -> DeepStorage:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- runs / checkpoints -------------------------------------------------

    def start_run(
        self,
        run_id: str,
        *,
        source_db: str,
        source_sha256: str | None,
        phases: str,
        version: str,
        config: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO deep_runs
               (run_id, started_at, phases, source_db, source_db_sha256,
                deep_version, config_json)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, _now(), phases, source_db, source_sha256, version, _dumps(config)),
        )
        self.conn.commit()

    def finish_run(
        self, run_id: str, *, reason: str, requests_issued: int, summary: dict[str, Any]
    ) -> None:
        self.conn.execute(
            """UPDATE deep_runs SET finished_at=?, termination_reason=?,
               requests_issued=?, summary_json=? WHERE run_id=?""",
            (_now(), reason, requests_issued, _dumps(summary), run_id),
        )
        self.conn.commit()

    def set_checkpoint(self, run_id: str, key: str, value: Any) -> None:
        self.conn.execute(
            """INSERT INTO deep_checkpoint (run_id, key, value, updated_at)
               VALUES (?,?,?,?) ON CONFLICT(run_id, key)
               DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (run_id, key, _dumps(value), _now()),
        )
        self.conn.commit()

    def get_checkpoint(self, run_id: str, key: str) -> Any:
        row = self.conn.execute(
            "SELECT value FROM deep_checkpoint WHERE run_id=? AND key=?", (run_id, key)
        ).fetchone()
        if not row or row["value"] is None:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def log_error(
        self,
        run_id: str,
        *,
        phase: str,
        entity_kind: str,
        entity_key: str | None,
        url: str | None,
        stage: str,
        error_type: str,
        message: str,
        http_status: int | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO deep_errors (run_id, phase, entity_kind, entity_key, url,
               stage, error_type, http_status, message, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                phase,
                entity_kind,
                entity_key,
                url,
                stage,
                error_type,
                http_status,
                message[:2000],
                _now(),
            ),
        )
        self.conn.commit()

    # -- phase A queue -----------------------------------------------------

    def seed_queue(self, rows: Sequence[tuple[str, str, str, str]]) -> int:
        """Insert ``(ad_id, url, api_url, source_status)``; existing rows untouched."""
        if not rows:
            return 0
        before = self.count("deep_queue")
        self.conn.executemany(
            """INSERT OR IGNORE INTO deep_queue
               (ad_id, url, api_url, source_status, seeded_at, updated_at)
               VALUES (?,?,?,?,?,?)""",
            [(a, u, api, st, _now(), _now()) for a, u, api, st in rows],
        )
        self.conn.commit()
        return self.count("deep_queue") - before

    def claim_ads(self, limit: int, *, max_attempts: int = 4) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(TERMINAL_AD_STATUSES))
        return list(
            self.conn.execute(
                f"""SELECT ad_id, url, api_url, attempts FROM deep_queue
                    WHERE status NOT IN ({placeholders})
                      AND (status != 'retryable_error' OR attempts < ?)
                    ORDER BY ad_id LIMIT ?""",
                (*TERMINAL_AD_STATUSES, max_attempts, limit),
            )
        )

    def all_queue(self) -> list[sqlite3.Row]:
        """Fixed snapshot for ``--refresh`` (a status-driven loop cannot end)."""
        return list(
            self.conn.execute("SELECT ad_id, url, api_url, attempts FROM deep_queue ORDER BY ad_id")
        )

    def recover_stuck(self) -> int:
        cur = self.conn.execute("UPDATE deep_queue SET status='pending' WHERE status='fetching'")
        self.conn.commit()
        return cur.rowcount

    def requeue_delisted(self) -> int:
        cur = self.conn.execute(
            "UPDATE deep_queue SET status='pending', attempts=0 WHERE status='delisted'"
        )
        self.conn.commit()
        return cur.rowcount

    def mark_ads_fetching(self, ad_ids: Sequence[str]) -> None:
        if not ad_ids:
            return
        self.conn.executemany(
            "UPDATE deep_queue SET status='fetching', updated_at=? WHERE ad_id=?",
            [(_now(), a) for a in ad_ids],
        )
        self.conn.commit()

    def set_ad_status(
        self,
        ad_id: str,
        status: str,
        *,
        api_status: str | None = None,
        html_status: str | None = None,
        error: str | None = None,
        bump_attempt: bool = False,
    ) -> None:
        self.conn.execute(
            f"""UPDATE deep_queue SET status=?, api_status=COALESCE(?, api_status),
                html_status=COALESCE(?, html_status), last_error=?, updated_at=?,
                completed_at=CASE WHEN ? IN ('completed','partial','delisted')
                                  THEN ? ELSE completed_at END
                {", attempts = attempts + 1" if bump_attempt else ""}
                WHERE ad_id=?""",
            (status, api_status, html_status, error, _now(), status, _now(), ad_id),
        )
        self.conn.commit()

    # -- phase A records ---------------------------------------------------

    def save_ad(
        self,
        record: dict[str, Any],
        media: Iterable[dict[str, Any]],
        raw_attrs: Iterable[tuple[str, str, Any]],
        conflicts: Iterable[dict[str, Any]],
    ) -> None:
        """Upsert one advertisement plus its media, raw attributes and conflicts."""
        columns = [c for c in record if c is not None]
        placeholders = ",".join("?" * len(columns))
        updates = ",".join(f"{c}=excluded.{c}" for c in columns if c != "ad_id")
        self.conn.execute(
            f"""INSERT INTO ad_deep ({",".join(columns)}) VALUES ({placeholders})
                ON CONFLICT(ad_id) DO UPDATE SET {updates}""",
            [record[c] for c in columns],
        )

        ad_id = record["ad_id"]
        media_rows = list(media)
        self.conn.execute("DELETE FROM ad_media_deep WHERE ad_id=?", (ad_id,))
        if media_rows:
            self.conn.executemany(
                """INSERT OR REPLACE INTO ad_media_deep
                   (ad_id, kind, position, original_url, large_url, small_url, thumb_url,
                    pinkie_url, video_id, video_url, video_thumb, alt, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        ad_id,
                        m.get("kind", "image"),
                        m.get("position", 0),
                        m.get("original_url"),
                        m.get("large_url"),
                        m.get("small_url"),
                        m.get("thumb_url"),
                        m.get("pinkie_url"),
                        m.get("video_id"),
                        m.get("video_url"),
                        m.get("video_thumb"),
                        m.get("alt"),
                        m.get("source"),
                    )
                    for m in media_rows
                ],
            )

        attrs = list(raw_attrs)
        if attrs:
            self.conn.executemany(
                "INSERT OR REPLACE INTO ad_raw_attributes (ad_id, source, key, value) VALUES (?,?,?,?)",
                [(ad_id, src, key, _dumps(val)) for src, key, val in attrs],
            )

        conflict_rows = list(conflicts)
        self.conn.execute("DELETE FROM ad_field_conflicts WHERE ad_id=?", (ad_id,))
        if conflict_rows:
            self.conn.executemany(
                """INSERT OR REPLACE INTO ad_field_conflicts
                   (ad_id, field, api_value, html_value, ld_value, chosen_value,
                    chosen_source, severity, detected_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        ad_id,
                        c["field"],
                        _dumps(c.get("api_value")),
                        _dumps(c.get("html_value")),
                        _dumps(c.get("ld_value")),
                        _dumps(c.get("chosen_value")),
                        c.get("chosen_source"),
                        c.get("severity", "info"),
                        _now(),
                    )
                    for c in conflict_rows
                ],
            )
        self.conn.commit()

    # -- derived work queues ----------------------------------------------

    def derive_review_keys(self) -> int:
        """Populate ``trim_reviews`` from the review URLs seen on ads."""
        rows = list(
            self.conn.execute(
                """SELECT review_key, review_url, COUNT(*) n FROM ad_deep
                   WHERE review_key IS NOT NULL AND review_key != ''
                   GROUP BY review_key, review_url"""
            )
        )
        for row in rows:
            self.conn.execute(
                """INSERT INTO trim_reviews (review_key, review_url, ad_count_in_dataset, updated_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(review_key) DO UPDATE SET
                     ad_count_in_dataset=excluded.ad_count_in_dataset,
                     updated_at=excluded.updated_at""",
                (row["review_key"], row["review_url"], row["n"], _now()),
            )
        self.conn.commit()
        return len(rows)

    def derive_price_keys(self) -> int:
        rows = list(
            self.conn.execute(
                """SELECT price_key, price_url, COUNT(*) n FROM ad_deep
                   WHERE price_key IS NOT NULL AND price_key != ''
                   GROUP BY price_key, price_url"""
            )
        )
        for row in rows:
            brand, _, rest = str(row["price_key"]).partition("|")
            model, _, trim = rest.partition("|")
            self.conn.execute(
                """INSERT INTO price_keys
                   (price_key, price_url, brand, model, trim, ad_count_in_dataset, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(price_key) DO UPDATE SET
                     ad_count_in_dataset=excluded.ad_count_in_dataset,
                     updated_at=excluded.updated_at""",
                (row["price_key"], row["price_url"], brand, model, trim, row["n"], _now()),
            )
        self.conn.commit()
        return len(rows)

    def derive_dealer_ids(self) -> int:
        rows = list(
            self.conn.execute(
                """SELECT dealer_id, dealer_link, COUNT(*) n FROM ad_deep
                   WHERE dealer_id IS NOT NULL GROUP BY dealer_id, dealer_link"""
            )
        )
        for row in rows:
            self.conn.execute(
                """INSERT INTO dealers (dealer_id, dealer_url, ad_count_in_dataset, updated_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(dealer_id) DO UPDATE SET
                     ad_count_in_dataset=excluded.ad_count_in_dataset,
                     updated_at=excluded.updated_at""",
                (row["dealer_id"], row["dealer_link"], row["n"], _now()),
            )
        self.conn.commit()
        return len(rows)

    def claim_generic(
        self, table: str, key_column: str, limit: int, *, max_attempts: int = 4
    ) -> list[sqlite3.Row]:
        """Claim pending rows from any phase-B/C/D queue table."""
        return list(
            self.conn.execute(
                f"""SELECT * FROM {table}
                    WHERE status NOT IN ('completed','permanent_error')
                      AND (status != 'retryable_error' OR attempts < ?)
                    ORDER BY {key_column} LIMIT ?""",
                (max_attempts, limit),
            )
        )

    def set_generic_status(
        self,
        table: str,
        key_column: str,
        key: Any,
        status: str,
        *,
        error: str | None = None,
        bump_attempt: bool = True,
    ) -> None:
        self.conn.execute(
            f"""UPDATE {table} SET status=?, last_error=?, updated_at=?
                {", attempts = attempts + 1" if bump_attempt else ""}
                WHERE {key_column}=?""",
            (status, error, _now(), key),
        )
        self.conn.commit()

    # -- generic upsert helper --------------------------------------------

    def upsert(self, table: str, key_columns: Sequence[str], record: dict[str, Any]) -> None:
        columns = list(record)
        placeholders = ",".join("?" * len(columns))
        updates = ",".join(f"{c}=excluded.{c}" for c in columns if c not in key_columns)
        conflict = ",".join(key_columns)
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        sql += (
            f" ON CONFLICT({conflict}) DO UPDATE SET {updates}"
            if updates
            else f" ON CONFLICT({conflict}) DO NOTHING"
        )
        self.conn.execute(sql, [record[c] for c in columns])
        self.conn.commit()

    def upsert_many(
        self, table: str, key_columns: Sequence[str], records: Sequence[dict[str, Any]]
    ) -> None:
        if not records:
            return
        columns = list(records[0])
        placeholders = ",".join("?" * len(columns))
        updates = ",".join(f"{c}=excluded.{c}" for c in columns if c not in key_columns)
        conflict = ",".join(key_columns)
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        sql += (
            f" ON CONFLICT({conflict}) DO UPDATE SET {updates}"
            if updates
            else f" ON CONFLICT({conflict}) DO NOTHING"
        )
        self.conn.executemany(sql, [[r[c] for c in columns] for r in records])
        self.conn.commit()

    # -- stats -------------------------------------------------------------

    def count(self, table: str, where: str = "") -> int:
        clause = f" WHERE {where}" if where else ""
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}{clause}").fetchone()[0])

    def status_breakdown(self, table: str) -> dict[str, int]:
        return {
            row[0]: row[1]
            for row in self.conn.execute(f"SELECT status, COUNT(*) FROM {table} GROUP BY status")
        }

    def stats(self) -> dict[str, Any]:
        return {
            "queue": self.status_breakdown("deep_queue"),
            "ads": self.count("ad_deep"),
            "media": self.count("ad_media_deep"),
            "conflicts": self.count("ad_field_conflicts"),
            "raw_attributes": self.count("ad_raw_attributes"),
            "trim_reviews": self.status_breakdown("trim_reviews"),
            "trim_specs": self.count("trim_specs"),
            "price_keys": self.status_breakdown("price_keys"),
            "price_series": self.count("price_series"),
            "price_points": self.count("price_points"),
            "dealers": self.status_breakdown("dealers"),
            "dealer_ads": self.count("dealer_ads"),
            "errors": self.count("deep_errors"),
        }
