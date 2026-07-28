"""Export the SQLite source of truth to CSV / JSONL / Parquet."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .logging_config import get_logger
from .storage import Storage

log = get_logger(__name__)

#: List/dict columns are JSON-encoded so one row stays one advertisement.
_JSON_COLUMNS = (
    "badges",
    "image_urls",
    "thumbnail_urls",
    "video_urls",
    "raw_attributes",
    "json_ld",
    "parse_warnings",
)


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in _JSON_COLUMNS:
        if key in out and not isinstance(out[key], (str, type(None))):
            out[key] = json.dumps(out[key], ensure_ascii=False)
    return out


def export_all(storage: Storage, output_dir: Path) -> dict[str, Any]:
    """Write every export artefact and return a manifest of paths and counts."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flatten(r) for r in storage.iter_details()]
    manifest: dict[str, Any] = {"rows": len(rows), "files": {}}

    jsonl_path = output_dir / "bama_ads.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["files"]["jsonl"] = str(jsonl_path)

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "bama_ads.csv"
    # utf-8-sig gives Excel the BOM it needs to render Persian correctly.
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest["files"]["csv"] = str(csv_path)

    parquet_path = output_dir / "bama_ads.parquet"
    try:
        frame.astype({c: "string" for c in frame.columns if frame[c].dtype == "object"}).to_parquet(
            parquet_path, index=False
        )
        manifest["files"]["parquet"] = str(parquet_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("export.parquet_failed", error=repr(exc))
        manifest["parquet_error"] = repr(exc)

    media_path = output_dir / "bama_media.csv"
    media = pd.read_sql_query(
        "SELECT ad_id, position, kind, url, original_url, thumb_url, alt FROM ad_media "
        "ORDER BY ad_id, kind, position",
        storage.conn,
    )
    media.to_csv(media_path, index=False, encoding="utf-8-sig")
    manifest["files"]["media_csv"] = str(media_path)
    manifest["media_rows"] = int(len(media))

    errors_path = output_dir / "bama_errors.csv"
    errors = pd.read_sql_query(
        "SELECT run_id, ad_id, url, stage, error_type, message, created_at "
        "FROM scrape_errors ORDER BY id",
        storage.conn,
    )
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")
    manifest["files"]["errors_csv"] = str(errors_path)
    manifest["error_rows"] = int(len(errors))

    candidates_path = output_dir / "network_candidates.json"
    candidates = [
        dict(r)
        for r in storage.conn.execute(
            "SELECT * FROM network_candidates ORDER BY is_listing_endpoint DESC, url"
        )
    ]
    candidates_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["files"]["network_candidates"] = str(candidates_path)

    discovered_path = output_dir / "bama_discovered.csv"
    pd.read_sql_query(
        "SELECT * FROM discovered_ads ORDER BY discovery_cycle, discovery_position", storage.conn
    ).to_csv(discovered_path, index=False, encoding="utf-8-sig")
    manifest["files"]["discovered_csv"] = str(discovered_path)

    manifest["files"]["sqlite"] = str(storage.db_path)
    return manifest


def export_sqlite_copy(storage: Storage, target: Path) -> None:
    """Write a consistent snapshot of the database (safe while WAL is active)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(target)
    with dest:
        storage.conn.backup(dest)
    dest.close()
