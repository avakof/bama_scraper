"""Database access supporting PostgreSQL (production) and SQLite (development).

Rather than an ORM, this is a thin dialect layer: queries are written once with
``?`` placeholders and portable type tokens, and the dialect rewrites them. That
keeps the SQL readable and reviewable while letting the same migrations and the
same comparison logic run on both backends.

Timestamps are always timezone-aware UTC ``datetime`` objects at the Python
boundary. PostgreSQL stores them as ``TIMESTAMPTZ``; SQLite stores ISO-8601 text
and this module converts on the way in and out, so callers never see the
difference.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

Dialect = Literal["postgres", "sqlite"]

#: Portable type tokens used in migrations, resolved per dialect.
TYPE_TOKENS: dict[Dialect, dict[str, str]] = {
    "postgres": {
        "PK": "BIGSERIAL PRIMARY KEY",
        "FK": "BIGINT",
        "TS": "TIMESTAMPTZ",
        "JSON": "JSONB",
        "BOOL": "BOOLEAN",
        "INT": "BIGINT",
        "REAL": "DOUBLE PRECISION",
        "TEXT": "TEXT",
        "NOW": "NOW()",
        # Null-safe equality. SQLite spells it `IS`; PostgreSQL rejects `IS ?`
        # outright and requires the SQL-standard form. Written as a token so a
        # query cannot accidentally be portable on one backend only.
        "EQ": "IS NOT DISTINCT FROM",
        # Boolean literals: PostgreSQL rejects `DEFAULT 0` on a BOOLEAN column.
        "TRUE": "TRUE",
        "FALSE": "FALSE",
        # Dropping a table constraint is one statement on PostgreSQL and a full
        # table rebuild on SQLite, which cannot DROP CONSTRAINT at all.
        "DROP_LEGACY_RUN_SLOT_CONSTRAINT": "ALTER TABLE monitoring_runs DROP CONSTRAINT IF EXISTS uq_runs_slot",
    },
    "sqlite": {
        "PK": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "FK": "INTEGER",
        "TS": "TEXT",
        "JSON": "TEXT",
        "BOOL": "INTEGER",
        "INT": "INTEGER",
        "REAL": "REAL",
        "TEXT": "TEXT",
        "NOW": "CURRENT_TIMESTAMP",
        "EQ": "IS",
        "TRUE": "1",
        "FALSE": "0",
        "DROP_LEGACY_RUN_SLOT_CONSTRAINT": "CREATE TABLE monitoring_runs_rebuilt (\n    id                    INTEGER PRIMARY KEY AUTOINCREMENT,\n    search_url            TEXT NOT NULL,\n    started_at            TEXT,\n    finished_at           TEXT,\n    scheduled_for         TEXT NOT NULL,\n    timezone              TEXT NOT NULL,\n    status                TEXT NOT NULL,\n    termination_reason    TEXT,\n    health_reason         TEXT,\n    discovered_count      INTEGER DEFAULT 0,\n    new_count             INTEGER DEFAULT 0,\n    active_count          INTEGER DEFAULT 0,\n    missing_count         INTEGER DEFAULT 0,\n    removed_count         INTEGER DEFAULT 0,\n    reappeared_count      INTEGER DEFAULT 0,\n    reposted_count        INTEGER DEFAULT 0,\n    likely_sold_count     INTEGER DEFAULT 0,\n    detail_success_count  INTEGER DEFAULT 0,\n    detail_failure_count  INTEGER DEFAULT 0,\n    duplicate_count       INTEGER DEFAULT 0,\n    error_count           INTEGER DEFAULT 0,\n    scraper_version       TEXT,\n    monitor_version       TEXT,\n    configuration_hash    TEXT NOT NULL,\n    previous_valid_run_id INTEGER,\n    comparison_applied    INTEGER DEFAULT FALSE,\n    evidence_path         TEXT,\n    created_at            TEXT NOT NULL, trigger_type TEXT NOT NULL DEFAULT 'manual', is_synthetic INTEGER NOT NULL DEFAULT 0, scheduled_date_paris TEXT, scheduled_hour_paris TEXT, host_name TEXT, process_id INTEGER, scheduler_instance_id TEXT, production_schedule_name TEXT, discovery_started_at TEXT, discovery_finished_at TEXT, detail_started_at TEXT, detail_finished_at TEXT, detail_accounted_count INTEGER NOT NULL DEFAULT 0, detail_gone_count INTEGER NOT NULL DEFAULT 0, detail_permanent_failure_count INTEGER NOT NULL DEFAULT 0, detail_retryable_failure_count INTEGER NOT NULL DEFAULT 0, detail_coverage_rate REAL, snapshot_export_path TEXT\n);\nINSERT INTO monitoring_runs_rebuilt (id, search_url, started_at, finished_at, scheduled_for, timezone, status, termination_reason, health_reason, discovered_count, new_count, active_count, missing_count, removed_count, reappeared_count, reposted_count, likely_sold_count, detail_success_count, detail_failure_count, duplicate_count, error_count, scraper_version, monitor_version, configuration_hash, previous_valid_run_id, comparison_applied, evidence_path, created_at, trigger_type, is_synthetic, scheduled_date_paris, scheduled_hour_paris, host_name, process_id, scheduler_instance_id, production_schedule_name, discovery_started_at, discovery_finished_at, detail_started_at, detail_finished_at, detail_accounted_count, detail_gone_count, detail_permanent_failure_count, detail_retryable_failure_count, detail_coverage_rate, snapshot_export_path)\n    SELECT id, search_url, started_at, finished_at, scheduled_for, timezone, status, termination_reason, health_reason, discovered_count, new_count, active_count, missing_count, removed_count, reappeared_count, reposted_count, likely_sold_count, detail_success_count, detail_failure_count, duplicate_count, error_count, scraper_version, monitor_version, configuration_hash, previous_valid_run_id, comparison_applied, evidence_path, created_at, trigger_type, is_synthetic, scheduled_date_paris, scheduled_hour_paris, host_name, process_id, scheduler_instance_id, production_schedule_name, discovery_started_at, discovery_finished_at, detail_started_at, detail_finished_at, detail_accounted_count, detail_gone_count, detail_permanent_failure_count, detail_retryable_failure_count, detail_coverage_rate, snapshot_export_path FROM monitoring_runs;\nDROP TABLE monitoring_runs;\nALTER TABLE monitoring_runs_rebuilt RENAME TO monitoring_runs;\nCREATE INDEX IF NOT EXISTS idx_runs_status ON monitoring_runs(status);\nCREATE INDEX IF NOT EXISTS idx_runs_scheduled ON monitoring_runs(scheduled_for);\nCREATE INDEX IF NOT EXISTS idx_runs_trigger ON monitoring_runs(trigger_type);\nCREATE INDEX IF NOT EXISTS idx_runs_genuine ON monitoring_runs(is_synthetic, trigger_type);\nCREATE UNIQUE INDEX IF NOT EXISTS uq_production_slot\n    ON monitoring_runs (production_schedule_name, configuration_hash, scheduled_for)\n    WHERE trigger_type IN ('scheduled', 'catch_up') AND is_synthetic = 0",
    },
}

_TOKEN_RE = re.compile(r"\{\{(\w+)\}\}")


def render_sql(template: str, dialect: Dialect) -> str:
    """Resolve ``{{TS}}``-style type tokens for one dialect."""
    tokens = TYPE_TOKENS[dialect]

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in tokens:
            raise KeyError(f"unknown SQL type token: {{{{{name}}}}}")
        return tokens[name]

    return _TOKEN_RE.sub(replace, template)


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_utc(value: datetime) -> datetime:
    """Normalize any datetime to timezone-aware UTC.

    A naive datetime is *assumed* UTC rather than silently localised, so a
    daylight-saving boundary can never shift a stored observation time.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_ts(value: Any) -> datetime | None:
    """Read a timestamp back from either backend."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_utc(value)
    if isinstance(value, str):
        text = value.strip().replace(" ", "T", 1) if " " in value else value.strip()
        try:
            return to_utc(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


class Connection(Protocol):  # pragma: no cover - structural typing only
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


class Database:
    """Dialect-aware connection wrapper.

    Queries use ``?`` placeholders and are rewritten to ``%s`` for PostgreSQL.
    Rows come back as plain dicts so downstream code is backend-agnostic.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.dialect: Dialect = (
            "postgres" if url.startswith(("postgres://", "postgresql://")) else "sqlite"
        )
        self._conn: Any = None
        self._connect()

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> None:
        if self.dialect == "sqlite":
            path = Path(self.url[len("sqlite:///") :])
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, timeout=60.0, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # Referential integrity is not on by default in SQLite, and this
            # schema relies on it.
            self._conn.execute("PRAGMA foreign_keys=ON")
        else:
            import psycopg

            self._conn = psycopg.connect(self._dsn(), autocommit=True)

    def _dsn(self) -> str:
        """psycopg accepts libpq URLs directly."""
        return self.url

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- statement helpers -------------------------------------------------

    def _rewrite(self, sql: str) -> str:
        # Operator tokens such as ``{{EQ}}`` are resolved for both dialects; only
        # the placeholder style differs.
        if "{{" in sql:
            sql = render_sql(sql, self.dialect)
        if self.dialect == "sqlite":
            return sql
        # Placeholders only; ``?`` does not otherwise occur in this codebase's SQL.
        return sql.replace("?", "%s")

    def _adapt(self, params: Sequence[Any] | None) -> tuple[Any, ...]:
        if not params:
            return ()
        out: list[Any] = []
        for value in params:
            if isinstance(value, datetime):
                value = to_utc(value)
                out.append(value.isoformat() if self.dialect == "sqlite" else value)
            elif isinstance(value, bool) and self.dialect == "sqlite":
                out.append(int(value))
            else:
                out.append(value)
        return tuple(out)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        cur = self._conn.cursor()
        cur.execute(self._rewrite(sql), self._adapt(params))
        return cur

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        payload = [self._adapt(r) for r in rows]
        if not payload:
            return
        cur = self._conn.cursor()
        cur.executemany(self._rewrite(sql), payload)

    def executescript(self, script: str) -> None:
        """Run a multi-statement script (migrations only)."""
        if self.dialect == "sqlite":
            self._conn.executescript(script)
        else:
            cur = self._conn.cursor()
            cur.execute(script)

    def _row_to_dict(self, cur: Any, row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if self.dialect == "sqlite":
            return dict(row)
        columns = [d[0] for d in cur.description]
        return dict(zip(columns, row, strict=False))

    def fetchone(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return self._row_to_dict(cur, row) if row is not None else None

    def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        cur = self.execute(sql, params)
        rows = cur.fetchall()
        return [self._row_to_dict(cur, r) for r in rows]

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return None if row is None else row[0]

    def insert_returning_id(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """Insert one row and return its generated id.

        PostgreSQL uses ``RETURNING``; SQLite uses ``lastrowid``. The caller
        writes the statement without a RETURNING clause and this adds it.
        """
        if self.dialect == "postgres":
            cur = self.execute(sql + " RETURNING id", params)
            row = cur.fetchone()
            return int(row[0])
        cur = self.execute(sql, params)
        return int(cur.lastrowid)

    # -- transactions ------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Database]:
        """Explicit transaction.

        The comparison workflow must be all-or-nothing: a partial application
        would leave some advertisements with incremented miss counters and others
        untouched, which is unrecoverable without manual repair.
        """
        if self.dialect == "sqlite":
            self._conn.execute("BEGIN IMMEDIATE")
        else:
            self._conn.autocommit = False
        try:
            yield self
        except BaseException:
            try:
                self._conn.rollback()
            finally:
                if self.dialect == "postgres":
                    self._conn.autocommit = True
            raise
        else:
            self._conn.commit()
            if self.dialect == "postgres":
                self._conn.autocommit = True

    # -- upsert helper -----------------------------------------------------

    def upsert(
        self,
        table: str,
        conflict_columns: Sequence[str],
        record: dict[str, Any],
        *,
        update_columns: Sequence[str] | None = None,
    ) -> None:
        """Portable ``INSERT ... ON CONFLICT DO UPDATE``.

        Both backends support the same syntax for this form, which is why the
        schema uses explicit unique constraints rather than relying on
        backend-specific merge statements.
        """
        columns = list(record)
        placeholders = ",".join("?" * len(columns))
        updates = [c for c in (update_columns or columns) if c not in conflict_columns]
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        if updates:
            assignments = ",".join(f"{c}=excluded.{c}" for c in updates)
            sql += f" ON CONFLICT ({','.join(conflict_columns)}) DO UPDATE SET {assignments}"
        else:
            sql += f" ON CONFLICT ({','.join(conflict_columns)}) DO NOTHING"
        self.execute(sql, [record[c] for c in columns])

    def json_dump(self, value: Any) -> Any:
        """Serialize a JSON column value for the active backend."""
        import json

        if value is None:
            return None
        if self.dialect == "postgres":
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def json_load(value: Any) -> Any:
        """Read a JSON column back, whatever the backend returned."""
        import json

        if value is None or isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _migration_files(dialect: Dialect) -> list[Path]:
    """Migrations in filename order, filtered to this dialect.

    The dialect is required rather than defaulting to "all files". A default of
    None is a footgun a long-running process will eventually find: this daemon had
    loaded an older version of this module and, on reconnecting, walked every file
    including the PostgreSQL-only one. It happened to be harmless — the statement
    is legal on SQLite 3.53 and the column was already nullable — but the next
    PostgreSQL-only migration would not have been.

    A file may be named ``NNN_name.sql`` (both backends) or
    ``NNN_name.<dialect>.sql`` (that backend only). The suffixed form exists
    because some changes have no portable spelling: PostgreSQL drops a constraint
    or a NOT NULL in one statement, while SQLite supports neither and needs a full
    table rebuild. Expressing that as one string with substitution tokens produced
    a multi-kilobyte unreadable literal; two small files say the same thing
    legibly.
    """
    if not MIGRATIONS_DIR.exists():
        return []
    files = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.is_file())
    known = set(TYPE_TOKENS)
    selected: list[Path] = []
    for path in files:
        parts = path.name.split(".")
        if len(parts) >= 3 and parts[-2] in known:
            if parts[-2] == dialect:
                selected.append(path)
        else:
            selected.append(path)
    return selected


def applied_migrations(db: Database) -> set[str]:
    db.executescript(
        render_sql(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name        {{TEXT}} PRIMARY KEY,
                applied_at  {{TS}} NOT NULL
            );
            """,
            db.dialect,
        )
    )
    return {r["name"] for r in db.fetchall("SELECT name FROM schema_migrations")}


def migrate(db: Database, *, verbose: bool = False) -> list[str]:
    """Apply pending migrations in filename order. Idempotent."""
    done = applied_migrations(db)
    applied: list[str] = []
    for path in _migration_files(db.dialect):
        if path.name in done:
            continue
        script = render_sql(path.read_text(encoding="utf-8"), db.dialect)
        # SQLite cannot DROP CONSTRAINT, so a migration that removes one has to
        # rebuild the table: create, copy, drop, rename. Foreign keys must be off
        # for that, or the DROP fails against every table referencing this one.
        # The pragma is a no-op inside a transaction, so it is issued around the
        # script rather than inside it.
        rebuilding = db.dialect == "sqlite" and "RENAME TO" in script
        if rebuilding:
            db.execute("PRAGMA foreign_keys=OFF")
        try:
            db.executescript(script)
        finally:
            if rebuilding:
                db.execute("PRAGMA foreign_keys=ON")
                violations = db.fetchall("PRAGMA foreign_key_check")
                if violations:
                    raise RuntimeError(
                        f"{path.name} left {len(violations)} foreign-key violation(s): "
                        f"{violations[:5]}"
                    )
        db.execute(
            "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            [path.name, utcnow()],
        )
        applied.append(path.name)
        if verbose:
            print(f"applied {path.name}", flush=True)
    return applied


def connect(url: str, *, run_migrations: bool = True) -> Database:
    db = Database(url)
    if run_migrations:
        migrate(db)
    return db
