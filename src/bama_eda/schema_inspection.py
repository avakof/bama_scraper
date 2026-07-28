"""Inspect the real database, then decide what may be analysed.

The rule this module enforces: **nothing is assumed to exist because the
documentation mentions it.** Every table and column used downstream is checked
here first, and each column is assigned an analytical role so that, for example, a
heuristic score can never be silently treated as a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .database import ReadOnlyDatabase
from .models import EXCLUDED_BLOB_COLUMNS, PRIVATE_COLUMNS, AnalyticalRole

#: Tables the EDA cannot work without.
REQUIRED_TABLES: tuple[str, ...] = (
    "monitoring_runs",
    "advertisements",
    "daily_ad_observations",
)

#: Tables that enable a specific analysis; absence degrades, not fails.
OPTIONAL_TABLES: tuple[str, ...] = (
    "advertisement_snapshots",
    "advertisement_events",
    "price_changes",
    "detail_verifications",
    "repost_links",
    "vehicle_entities",
    "sale_validation_samples",
    "scrape_errors",
    "alerts",
)

#: Columns whose meaning must not be inferred from the name alone. Assigning these
#: explicitly is what stops `sale_confidence` being charted as a probability.
EXPLICIT_ROLES: dict[str, AnalyticalRole] = {
    "sale_confidence": AnalyticalRole.HEURISTIC_INFERENCE,
    "sale_evidence_score": AnalyticalRole.HEURISTIC_INFERENCE,
    "sale_label": AnalyticalRole.HEURISTIC_INFERENCE,
    "predicted_score": AnalyticalRole.HEURISTIC_INFERENCE,
    "predicted_label": AnalyticalRole.HEURISTIC_INFERENCE,
    "current_status": AnalyticalRole.OUTCOME_OBSERVATION,
    "was_seen": AnalyticalRole.OUTCOME_OBSERVATION,
    "detail_availability": AnalyticalRole.OUTCOME_OBSERVATION,
    "last_detail_verdict": AnalyticalRole.OUTCOME_OBSERVATION,
    "filter_exit_reason": AnalyticalRole.OUTCOME_OBSERVATION,
    "observed_outcome": AnalyticalRole.OUTCOME_OBSERVATION,
    "left_truncated": AnalyticalRole.PROVENANCE,
    "eligible_for_duration_ranking": AnalyticalRole.PROVENANCE,
    "published_at_source": AnalyticalRole.PROVENANCE,
    "published_at_reliable": AnalyticalRole.PROVENANCE,
    "parser_version": AnalyticalRole.PROVENANCE,
    "scraper_version": AnalyticalRole.PROVENANCE,
    "monitor_version": AnalyticalRole.PROVENANCE,
    "configuration_hash": AnalyticalRole.PROVENANCE,
    "content_hash": AnalyticalRole.PROVENANCE,
    "description": AnalyticalRole.RAW_TEXT,
    "description_scrubbed": AnalyticalRole.RAW_TEXT,
    "card_title": AnalyticalRole.RAW_TEXT,
    "title": AnalyticalRole.RAW_TEXT,
}

_NUMERIC_HINTS = ("_normalized", "_toman", "_km", "_count", "_seconds", "_hours", "_days")
_CATEGORY_HINTS = (
    "brand",
    "model",
    "trim",
    "city",
    "province",
    "seller_type",
    "transmission",
    "fuel_type",
    "body",
    "status",
    "colour",
    "color",
    "reason",
    "verdict",
    "label",
    "type",
)


def classify_column(name: str, database_type: str, primary_key: bool) -> AnalyticalRole:
    """Assign an analytical role.

    Order matters: privacy first, then explicit overrides, then heuristics. A name
    that looks numeric must not outrank an explicit "this is an inference".
    """
    lowered = name.lower()
    if lowered in PRIVATE_COLUMNS or lowered in EXCLUDED_BLOB_COLUMNS:
        return AnalyticalRole.PRIVATE_OR_EXCLUDED
    if lowered in EXPLICIT_ROLES:
        return EXPLICIT_ROLES[lowered]
    if primary_key or lowered == "id" or lowered.endswith("_id") or lowered.endswith("_url"):
        return AnalyticalRole.IDENTIFIER
    if lowered.endswith(("_at", "_ts")) or "timestamp" in lowered or "TIMESTAMP" in database_type:
        return AnalyticalRole.TIMESTAMP
    if lowered.endswith("_json") or database_type in ("JSONB", "JSON"):
        return AnalyticalRole.PRIVATE_OR_EXCLUDED
    if lowered.endswith(("_raw", "_text")):
        return AnalyticalRole.RAW_TEXT
    if any(lowered.endswith(h) or h in lowered for h in _NUMERIC_HINTS):
        return AnalyticalRole.NORMALIZED_NUMERIC
    if database_type.startswith(("INT", "BIGINT", "REAL", "DOUBLE", "NUMERIC", "DECIMAL")):
        return AnalyticalRole.NORMALIZED_NUMERIC
    if any(hint in lowered for hint in _CATEGORY_HINTS):
        return AnalyticalRole.CATEGORY
    return AnalyticalRole.RAW_TEXT


def infer_semantic_type(values: list[Any]) -> str:
    """Cheap semantic guess from a handful of example values."""
    sample = [v for v in values if v is not None][:20]
    if not sample:
        return "unknown_all_null"
    if all(isinstance(v, bool) for v in sample):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in sample):
        return "integer"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample):
        return "numeric"
    texts = [str(v) for v in sample]
    if all(len(t) >= 10 and t[4:5] == "-" and t[7:8] == "-" for t in texts):
        return "timestamp_text"
    if any(any("؀" <= ch <= "ۿ" for ch in t) for t in texts):
        return "persian_text"
    if all(t.startswith("http") for t in texts):
        return "url"
    if all(len(t) == 64 and all(c in "0123456789abcdef" for c in t.lower()) for t in texts):
        return "sha256_hex"
    if len(set(texts)) <= max(2, len(texts) // 3):
        return "low_cardinality_text"
    return "text"


def inspect_schema(db: ReadOnlyDatabase, *, sample_rows: int = 200) -> pd.DataFrame:
    """One row per column, with counts and an analytical role.

    Distinct and non-null counts are computed with SQL aggregates rather than by
    loading the table, so this stays usable as the database grows.
    """
    records: list[dict[str, Any]] = []
    for table in db.table_names():
        row_count = db.row_count(table)
        for column in db.columns(table):
            name = column["column_name"]
            role = classify_column(name, column["database_type"], column["primary_key"])

            non_null: int | None = None
            distinct: int | None = None
            examples: list[Any] = []
            # Blobs and private columns are never read, not even for examples.
            if role is not AnalyticalRole.PRIVATE_OR_EXCLUDED and row_count:
                non_null = int(
                    db.scalar(f'SELECT COUNT("{name}") FROM {table}') or 0  # noqa: S608
                )
                distinct = int(
                    db.scalar(f'SELECT COUNT(DISTINCT "{name}") FROM {table}') or 0  # noqa: S608
                )
                rows = db.fetchall(
                    f'SELECT "{name}" AS v FROM {table} WHERE "{name}" IS NOT NULL'  # noqa: S608
                    f" LIMIT {int(sample_rows)}"
                )
                examples = [r["v"] for r in rows]

            records.append(
                {
                    "table_name": table,
                    "column_name": name,
                    "database_type": column["database_type"],
                    "nullable": column["nullable"],
                    "primary_key": column["primary_key"],
                    "foreign_key": None,
                    "unique_constraint": False,
                    "row_count": row_count,
                    "non_null_count": non_null,
                    "distinct_count": distinct,
                    "example_values": _format_examples(examples, role),
                    "inferred_semantic_type": (
                        "excluded"
                        if role is AnalyticalRole.PRIVATE_OR_EXCLUDED
                        else infer_semantic_type(examples)
                    ),
                    "analytical_role": str(role),
                }
            )

        for fk in db.foreign_keys(table):
            for record in records:
                if record["table_name"] == table and record["column_name"] == fk["column"]:
                    record["foreign_key"] = f"{fk['references_table']}.{fk['references_column']}"
        unique_columns = set()
        for group in db.unique_constraints(table):
            unique_columns.update(group)
        for record in records:
            if record["table_name"] == table and record["column_name"] in unique_columns:
                record["unique_constraint"] = True

    return pd.DataFrame(records)


def _format_examples(values: list[Any], role: AnalyticalRole) -> str:
    """Short, privacy-safe example string."""
    if role is AnalyticalRole.PRIVATE_OR_EXCLUDED:
        return "<excluded>"
    seen: list[str] = []
    for value in values:
        text = str(value)
        if len(text) > 40:
            text = text[:37] + "..."
        if text not in seen:
            seen.append(text)
        if len(seen) == 3:
            break
    return " | ".join(seen)


def table_relationships(db: ReadOnlyDatabase) -> dict[str, Any]:
    """Foreign-key graph, for the report and for orphan checks."""
    graph: dict[str, Any] = {"tables": {}, "edges": []}
    for table in db.table_names():
        graph["tables"][table] = {"row_count": db.row_count(table)}
        for fk in db.foreign_keys(table):
            graph["edges"].append(
                {
                    "from_table": table,
                    "from_column": fk["column"],
                    "to_table": fk["references_table"],
                    "to_column": fk["references_column"],
                }
            )
    return graph


def build_data_dictionary(schema: pd.DataFrame) -> pd.DataFrame:
    """The analyst-facing dictionary: what each column means and may be used for."""
    notes = {
        AnalyticalRole.HEURISTIC_INFERENCE: (
            "INFERENCE, not measurement. Never present as a probability or a rate."
        ),
        AnalyticalRole.OUTCOME_OBSERVATION: "Recorded observation of what was seen.",
        AnalyticalRole.PROVENANCE: "Describes how the value was obtained; gates other analyses.",
        AnalyticalRole.PRIVATE_OR_EXCLUDED: "Excluded from every analytical output.",
        AnalyticalRole.IDENTIFIER: "Identity only; never a quantity. Ad id is not a vehicle id.",
        AnalyticalRole.TIMESTAMP: "Instant. Check which clock it belongs to before differencing.",
        AnalyticalRole.NORMALIZED_NUMERIC: "Parsed number; the raw text sits beside it.",
        AnalyticalRole.CATEGORY: "Categorical; apply the minimum group size before reporting.",
        AnalyticalRole.RAW_TEXT: "Original text, preserved. Persian normalization applies.",
    }
    frame = schema.copy()
    frame["usage_note"] = frame["analytical_role"].map(lambda r: notes.get(AnalyticalRole(r), ""))
    # Excluded columns are never counted, so non_null_count is NA for them; the
    # ratio must stay float-typed or NA propagates into an unroundable object.
    non_null = pd.to_numeric(frame["non_null_count"], errors="coerce")
    row_count = pd.to_numeric(frame["row_count"], errors="coerce").replace(0, np.nan)
    frame["null_fraction"] = (1.0 - (non_null / row_count)).astype("Float64").round(4)
    frame["is_constant"] = frame["distinct_count"].le(1) & frame["row_count"].gt(0)
    frame["modelling_suitable"] = (
        frame["analytical_role"].isin(
            [str(AnalyticalRole.NORMALIZED_NUMERIC), str(AnalyticalRole.CATEGORY)]
        )
        & ~frame["is_constant"]
        & frame["null_fraction"].fillna(1.0).lt(0.5)
    )
    return frame[
        [
            "table_name",
            "column_name",
            "database_type",
            "nullable",
            "primary_key",
            "foreign_key",
            "unique_constraint",
            "row_count",
            "non_null_count",
            "distinct_count",
            "null_fraction",
            "is_constant",
            "example_values",
            "inferred_semantic_type",
            "analytical_role",
            "usage_note",
            "modelling_suitable",
        ]
    ]


class SchemaError(RuntimeError):
    """A required table or column is missing."""


def assert_analysable(db: ReadOnlyDatabase) -> list[str]:
    """Fail fast and clearly, listing everything that is missing at once."""
    missing = [t for t in REQUIRED_TABLES if not db.has_table(t)]
    if missing:
        raise SchemaError(
            "cannot analyse this database; required table(s) absent: " + ", ".join(missing)
        )
    absent_optional = [t for t in OPTIONAL_TABLES if not db.has_table(t)]
    return absent_optional


def write_schema_outputs(db: ReadOnlyDatabase, out_dir: Path) -> dict[str, Path]:
    """Write the three schema artefacts and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = inspect_schema(db)
    dictionary = build_data_dictionary(schema)

    schema_path = out_dir / "schema_inventory.csv"
    dict_path = out_dir / "data_dictionary.csv"
    rel_path = out_dir / "table_relationships.json"

    schema.to_csv(schema_path, index=False, encoding="utf-8-sig")
    dictionary.to_csv(dict_path, index=False, encoding="utf-8-sig")
    rel_path.write_text(
        json.dumps(table_relationships(db), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "schema_inventory": schema_path,
        "data_dictionary": dict_path,
        "table_relationships": rel_path,
    }
