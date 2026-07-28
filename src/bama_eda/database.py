"""Read-only database access for analysis.

Deliberately thin and deliberately read-only. The EDA has no business writing to
the mother database, so there is no write path here at all — not as a discipline
but as an absence of capability.

Reuses the monitoring package's dialect layer for connection handling and type
adaptation, so a query written once runs on both PostgreSQL and SQLite exactly as
the monitor's own queries do.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pandas as pd

from bama_monitor.db import Database, parse_ts, render_sql


class ReadOnlyDatabase:
    """A connection that can only be read from.

    ``query_frame`` is the single entry point for analysis, so every SQL statement
    the EDA issues passes one place that can hash it, count its rows and record it
    in the manifest.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        # `run_migrations=False`: analysis must never alter the schema of the
        # database it is analysing, and a migration would also change the very
        # row counts being reported.
        self._db = Database(url)
        self.dialect = self._db.dialect
        self.query_log: list[dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ReadOnlyDatabase:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- introspection -----------------------------------------------------

    def table_names(self) -> list[str]:
        if self.dialect == "sqlite":
            rows = self._db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table'"
                " AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            return [str(r["name"]) for r in rows]
        rows = self._db.fetchall(
            "SELECT table_name AS name FROM information_schema.tables"
            " WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
        )
        return [str(r["name"]) for r in rows]

    def columns(self, table: str) -> list[dict[str, Any]]:
        """Column metadata, normalized across backends."""
        if self.dialect == "sqlite":
            rows = self._db.fetchall(f"PRAGMA table_info({table})")
            return [
                {
                    "column_name": str(r["name"]),
                    "database_type": str(r["type"] or "").upper() or "UNKNOWN",
                    "nullable": not bool(r["notnull"]),
                    "primary_key": bool(r["pk"]),
                    "ordinal": int(r["cid"]),
                }
                for r in rows
            ]
        rows = self._db.fetchall(
            "SELECT column_name, data_type, is_nullable, ordinal_position"
            " FROM information_schema.columns WHERE table_schema='public'"
            " AND table_name=? ORDER BY ordinal_position",
            [table],
        )
        pk = self._primary_keys(table)
        return [
            {
                "column_name": str(r["column_name"]),
                "database_type": str(r["data_type"]).upper(),
                "nullable": str(r["is_nullable"]).upper() == "YES",
                "primary_key": str(r["column_name"]) in pk,
                "ordinal": int(r["ordinal_position"]),
            }
            for r in rows
        ]

    def _primary_keys(self, table: str) -> set[str]:
        rows = self._db.fetchall(
            "SELECT kcu.column_name FROM information_schema.table_constraints tc"
            " JOIN information_schema.key_column_usage kcu"
            "   ON kcu.constraint_name = tc.constraint_name"
            " WHERE tc.table_name=? AND tc.constraint_type='PRIMARY KEY'",
            [table],
        )
        return {str(r["column_name"]) for r in rows}

    def foreign_keys(self, table: str) -> list[dict[str, Any]]:
        if self.dialect == "sqlite":
            rows = self._db.fetchall(f"PRAGMA foreign_key_list({table})")
            return [
                {
                    "column": str(r["from"]),
                    "references_table": str(r["table"]),
                    "references_column": str(r["to"]),
                }
                for r in rows
            ]
        rows = self._db.fetchall(
            "SELECT kcu.column_name, ccu.table_name AS ref_table,"
            " ccu.column_name AS ref_column"
            " FROM information_schema.table_constraints tc"
            " JOIN information_schema.key_column_usage kcu"
            "   ON kcu.constraint_name = tc.constraint_name"
            " JOIN information_schema.constraint_column_usage ccu"
            "   ON ccu.constraint_name = tc.constraint_name"
            " WHERE tc.table_name=? AND tc.constraint_type='FOREIGN KEY'",
            [table],
        )
        return [
            {
                "column": str(r["column_name"]),
                "references_table": str(r["ref_table"]),
                "references_column": str(r["ref_column"]),
            }
            for r in rows
        ]

    def unique_constraints(self, table: str) -> list[list[str]]:
        if self.dialect == "sqlite":
            groups: list[list[str]] = []
            for index in self._db.fetchall(f"PRAGMA index_list({table})"):
                if not bool(index["unique"]):
                    continue
                cols = self._db.fetchall(f"PRAGMA index_info({index['name']})")
                groups.append([str(c["name"]) for c in cols])
            return groups
        rows = self._db.fetchall(
            "SELECT tc.constraint_name, kcu.column_name"
            " FROM information_schema.table_constraints tc"
            " JOIN information_schema.key_column_usage kcu"
            "   ON kcu.constraint_name = tc.constraint_name"
            " WHERE tc.table_name=? AND tc.constraint_type='UNIQUE'"
            " ORDER BY tc.constraint_name, kcu.ordinal_position",
            [table],
        )
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(str(row["constraint_name"]), []).append(str(row["column_name"]))
        return list(grouped.values())

    def has_table(self, table: str) -> bool:
        return table in set(self.table_names())

    def has_columns(self, table: str, names: Sequence[str]) -> set[str]:
        """Return the subset of ``names`` that actually exists.

        Used everywhere instead of assuming a column exists because the
        documentation mentions it. A missing column degrades one analysis; a
        crash halfway through wastes the whole run.
        """
        if not self.has_table(table):
            return set()
        present = {c["column_name"] for c in self.columns(table)}
        return set(names) & present

    # -- reads -------------------------------------------------------------

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        return self._db.scalar(sql, params)

    def row_count(self, table: str) -> int:
        return int(self._db.scalar(f"SELECT COUNT(*) FROM {table}") or 0)

    def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        return self._db.fetchall(sql, params)

    def fetchone(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        return self._db.fetchone(sql, params)

    def query_frame(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        label: str,
        parse_dates: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run one analytical query and return it as a DataFrame.

        Every call is logged with a hash of the rendered SQL, so the manifest can
        state exactly which queries produced the numbers in the report.
        """
        rendered = render_sql(sql, self.dialect) if "{{" in sql else sql
        rows = self._db.fetchall(rendered, params)
        frame = pd.DataFrame(rows)
        self.query_log.append(
            {
                "label": label,
                "sql_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "sql": " ".join(rendered.split()),
                "params": [str(p) for p in (params or [])],
                "rows": int(len(frame)),
            }
        )
        for column in parse_dates or ():
            if column in frame.columns:
                frame[column] = frame[column].map(parse_ts)
        return frame

    def iter_frames(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        label: str,
        chunk_size: int = 50_000,
    ) -> Iterator[pd.DataFrame]:
        """Chunked read for tables that may outgrow memory.

        Uses keyset pagination on a monotonically increasing ``id`` rather than
        OFFSET, which degrades linearly and can also skip or repeat rows if the
        table changes under a long read.
        """
        last_id = 0
        while True:
            page = self.query_frame(
                sql,
                [*(params or []), last_id, chunk_size],
                label=f"{label}[after={last_id}]",
            )
            if page.empty:
                return
            yield page
            last_id = int(page["id"].max())


@contextmanager
def connect(url: str) -> Iterator[ReadOnlyDatabase]:
    db = ReadOnlyDatabase(url)
    try:
        yield db
    finally:
        db.close()


def utcnow() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)
