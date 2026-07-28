"""Shared fixtures for the monitoring test suite.

Every test runs against a real database. Where a backend is parameterised, the
same test executes on SQLite and — when a server is reachable — on PostgreSQL,
because the comparison logic must behave identically on both.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from bama_monitor.config import MonitorConfig
from bama_monitor.db import Database, connect
from bama_monitor.repository import Repository

#: Set by the harness when a throwaway PostgreSQL instance is available.
PG_URL_ENV = "BAMA_MONITOR_TEST_PG_URL"

TABLES_IN_FK_ORDER = (
    "advertisement_events",
    "daily_ad_observations",
    "price_changes",
    "detail_verifications",
    "repost_links",
    "advertisement_snapshots",
    "scrape_errors",
    "alerts",
    "advertisements",
    "run_locks",
    "monitoring_runs",
)


def _backends() -> list[str]:
    backends = ["sqlite"]
    if os.environ.get(PG_URL_ENV):
        backends.append("postgres")
    return backends


@pytest.fixture(params=_backends())
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def db(backend: str, tmp_path: Path) -> Iterator[Database]:
    """A migrated, empty database on the parameterised backend."""
    if backend == "sqlite":
        database = connect(f"sqlite:///{tmp_path / 'monitor.sqlite'}")
    else:
        database = connect(os.environ[PG_URL_ENV])
        for table in TABLES_IN_FK_ORDER:
            database.execute(f"TRUNCATE {table} CASCADE")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Iterator[Database]:
    """SQLite-only database, for tests that do not need backend parity."""
    database = connect(f"sqlite:///{tmp_path / 'monitor.sqlite'}")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def cfg(tmp_path: Path, db: Database) -> MonitorConfig:
    """Config wired to the active database, with thresholds relaxed for small fixtures."""
    return MonitorConfig(
        database_url=db.url,
        output_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        lock_dir=tmp_path / "locks",
        health={"min_absolute_count": 1, "median_window_runs": 7},
    )


@pytest.fixture
def repo(db: Database) -> Repository:
    return Repository(db)


from monitor_helpers import (  # noqa: E402
    BASE_TIME,
    make_context,
    make_discovery,
    misses_of,
    run_day,
    status_of,
)

__all__ = [
    "BASE_TIME",
    "make_context",
    "make_discovery",
    "misses_of",
    "run_day",
    "status_of",
]
