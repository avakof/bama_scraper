"""Export the deep dataset to CSV / JSONL / Parquet plus a nested JSONL.

Conventions follow ``bama_scraper.export``: ``utf-8-sig`` CSV so Excel renders
Persian correctly, one row per entity, and Parquet failures warn rather than
abort the run.

The wide specification pivot is generated from ``spec_key_catalog`` rather than
hard-coded, so a new model introducing new items needs a ``reindex``, not a code
change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .audit import snapshot
from .config import DEEP_VERSION, DeepConfig
from .storage import DeepStorage

#: One row per entity for these tables, exported verbatim.
_TABLE_EXPORTS: tuple[tuple[str, str, str], ...] = (
    ("ads_media", "ad_media_deep", "ad_id, kind, position"),
    ("ads_raw_attributes", "ad_raw_attributes", "ad_id, source, key"),
    ("ads_conflicts", "ad_field_conflicts", "severity DESC, field, ad_id"),
    ("trim_reviews", "trim_reviews", "review_key"),
    ("trim_specs_long", "trim_specs", "review_key, group_slug, position"),
    ("trim_specs_typed", "trim_specs_typed", "review_key"),
    ("spec_key_catalog", "spec_key_catalog", "group_index, position"),
    ("price_series", "price_series", "price_key, series_id"),
    ("price_points", "price_points", "series_id, point_index"),
    ("price_brands", "price_brands", "brand_slug"),
    ("price_hierarchy", "price_hierarchy", "path"),
    ("dealers", "dealers", "dealer_id"),
    ("dealer_ads", "dealer_ads", "dealer_id, ad_code"),
    ("deep_errors", "deep_errors", "id"),
)

#: Parquet is worth having for the big tables only.
_PARQUET_TABLES = frozenset({"ads_deep", "trim_specs_long", "price_points", "trim_specs_wide"})

#: Columns dropped from the primary export unless explicitly requested; they are
#: large evidence blobs, not analysis fields.
_HEAVY_COLUMNS = ("api_raw_json", "html_payload_json", "json_ld_json", "raw_attributes")


def export_all(cfg: DeepConfig, storage: DeepStorage) -> dict[str, Any]:
    import pandas as pd

    out = cfg.exports_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"deep_version": DEEP_VERSION, "files": {}, "rows": {}}

    def write_frame(name: str, frame: Any) -> None:
        csv_path = out / f"{name}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        manifest["files"][f"{name}.csv"] = _fingerprint(csv_path)
        manifest["rows"][name] = int(len(frame))
        if name in _PARQUET_TABLES:
            try:
                typed = frame.astype(
                    {c: "string" for c in frame.columns if frame[c].dtype == "object"}
                )
                parquet_path = out / f"{name}.parquet"
                typed.to_parquet(parquet_path, index=False)
                manifest["files"][f"{name}.parquet"] = _fingerprint(parquet_path)
            except Exception as exc:  # noqa: BLE001
                manifest.setdefault("warnings", []).append(f"{name} parquet: {exc!r}")

    # -- primary advertisement table ---------------------------------------
    ads = pd.read_sql_query("SELECT * FROM ad_deep ORDER BY ad_id", storage.conn)
    drop = [c for c in _HEAVY_COLUMNS if c in ads.columns]
    write_frame("ads_deep", ads.drop(columns=drop))

    jsonl_path = out / "ads_deep.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in ads.drop(columns=drop).to_dict(orient="records"):
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    manifest["files"]["ads_deep.jsonl"] = _fingerprint(jsonl_path)

    # -- straight table exports -------------------------------------------
    for name, table, order in _TABLE_EXPORTS:
        if name == "dealers" and cfg.no_dealer_address:
            columns = [
                r[1]
                for r in storage.conn.execute(f"PRAGMA table_info({table})")
                if r[1] != "address"
            ]
            query = f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"
        else:
            query = f"SELECT * FROM {table} ORDER BY {order}"
        write_frame(name, pd.read_sql_query(query, storage.conn))

    # -- generated wide specification pivot -------------------------------
    catalog = pd.read_sql_query(
        "SELECT item_slug FROM spec_key_catalog ORDER BY group_index, position, item_slug",
        storage.conn,
    )
    if len(catalog):
        try:
            wide = pd.read_sql_query("SELECT * FROM v_trim_specs_wide", storage.conn)
            write_frame("trim_specs_wide", wide)
        except Exception as exc:  # noqa: BLE001
            manifest.setdefault("warnings", []).append(f"wide pivot: {exc!r}")

    # -- nested per-advertisement JSONL ------------------------------------
    manifest["files"]["ads_full.jsonl"] = _fingerprint(_write_nested(cfg, storage, out))

    # -- data dictionary ---------------------------------------------------
    dictionary = out / "data_dictionary.md"
    dictionary.write_text(_data_dictionary(storage), encoding="utf-8")
    manifest["files"]["data_dictionary.md"] = _fingerprint(dictionary)

    # -- snapshot ----------------------------------------------------------
    snapshot(storage, cfg.snapshot_path)
    manifest["files"][cfg.snapshot_path.name] = _fingerprint(cfg.snapshot_path)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_nested(cfg: DeepConfig, storage: DeepStorage, out: Path) -> Path:
    """One JSON object per advertisement with its media, specs, prices and dealer."""
    conn = storage.conn
    media: dict[str, list[dict[str, Any]]] = {}
    for row in conn.execute("SELECT * FROM ad_media_deep ORDER BY ad_id, kind, position"):
        media.setdefault(row["ad_id"], []).append(dict(row))

    specs: dict[str, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT review_key, group_slug, item_slug, item_key, value_text, value_bool, "
        "value_num, value_unit FROM trim_specs ORDER BY review_key, group_slug, position"
    ):
        specs.setdefault(row["review_key"], []).append(dict(row))

    series: dict[str, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT price_key, series_id, provider_kind, model_year, equipment_class, "
        "latest_price_toman, point_count FROM price_series ORDER BY price_key"
    ):
        series.setdefault(row["price_key"], []).append(dict(row))

    dealers = {row["dealer_id"]: dict(row) for row in conn.execute("SELECT * FROM dealers")}
    if cfg.no_dealer_address:
        for record in dealers.values():
            record.pop("address", None)

    path = out / "ads_full.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in conn.execute("SELECT * FROM ad_deep ORDER BY ad_id"):
            ad = {k: row[k] for k in row.keys() if k not in _HEAVY_COLUMNS}
            ad["media"] = media.get(row["ad_id"], [])
            ad["trim_specs"] = specs.get(row["review_key"], []) if row["review_key"] else []
            ad["price_series"] = series.get(row["price_key"], []) if row["price_key"] else []
            ad["dealer"] = dealers.get(row["dealer_id"]) if row["dealer_id"] else None
            fh.write(json.dumps(ad, ensure_ascii=False, default=str) + "\n")
    return path


def _data_dictionary(storage: DeepStorage) -> str:
    """Generate the spec data dictionary from the catalog."""
    lines = [
        "# Deep dataset data dictionary",
        "",
        "Generated by `deep export`. The specification items below are discovered",
        "from the live data, not hard-coded -- `deep reindex` refreshes them.",
        "",
        "## Specification items (`trim_specs`)",
        "",
        "| group | item_slug | Persian key | type | distinct values | model-trims covered |",
        "|---|---|---|---|---|---|",
    ]
    for row in storage.conn.execute(
        "SELECT group_name, item_slug, item_key, value_type, distinct_values, "
        "coverage_review_keys FROM spec_key_catalog ORDER BY group_index, position, item_slug"
    ):
        lines.append(
            f"| {row['group_name']} | `{row['item_slug']}` | {row['item_key']} | "
            f"{row['value_type']} | {row['distinct_values']} | {row['coverage_review_keys']} |"
        )
    lines += [
        "",
        "## Numeric convention",
        "",
        "Bama serializes a decimal as its display string with the decimal point",
        "removed, so the implied scale is `10 ** (fractional digits in the text)`",
        "and varies per value. Every such field therefore carries three columns:",
        "`*_text` (authoritative), a parsed numeric, and `*_raw` (the site's",
        "integer, kept as a checksum). `numeric_agreement_json` records whether",
        "the two reconciled.",
        "",
        "## Privacy",
        "",
        "Seller telephone numbers are never stored. `description` holds the",
        "scrubbed text and `description_redactions` counts what was removed.",
    ]
    return "\n".join(lines) + "\n"
