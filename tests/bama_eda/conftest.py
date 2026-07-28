"""Synthetic fixtures for the EDA tests.

Every scenario the analysis must get right is constructed explicitly here rather
than sampled from real data, so a test failure names a behaviour instead of a row.

The fixture database contains, by construction:

* a valid run, an invalid run and an unfinished run (run selection);
* a snapshot from a later run (must never be joined);
* two snapshots for one advertisement (deterministic latest-selection);
* an advertisement with no snapshot at all;
* a filter exit and a re-entry;
* a disappearance, a reappearance and a repost chain;
* a left-truncated listing and an eligible one;
* a listing with an unreliable publication timestamp;
* a contact number planted in a description (privacy gate).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

#: Set when a throwaway PostgreSQL instance is available.
from eda_scenarios import PG_URL_ENV, TABLES_IN_FK_ORDER, seed

from bama_eda.config import EdaConfig
from bama_eda.database import ReadOnlyDatabase
from bama_monitor.db import Database, connect


def _backends() -> list[str]:
    backends = ["sqlite"]
    if os.environ.get(PG_URL_ENV):
        backends.append("postgres")
    return backends


@pytest.fixture(params=_backends())
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def seeded_db(backend: str, tmp_path: Path) -> Iterator[Database]:
    """A migrated database populated with the scenarios above."""
    if backend == "sqlite":
        db = connect(f"sqlite:///{tmp_path / 'eda.sqlite'}")
    else:
        db = connect(os.environ[PG_URL_ENV])
        for table in TABLES_IN_FK_ORDER:
            db.execute(f"TRUNCATE {table} CASCADE")
    try:
        seed(db)
        yield db
    finally:
        db.close()


@pytest.fixture
def eda_db(seeded_db: Database) -> Iterator[ReadOnlyDatabase]:
    db = ReadOnlyDatabase(seeded_db.url)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def cfg(tmp_path: Path, seeded_db: Database) -> EdaConfig:
    return EdaConfig(
        database_url=seeded_db.url,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        html=False,
        export_parquet=False,
    )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_ids(seeded_db: Database) -> dict[str, Any]:
    """Ids for the seeded scenarios (the database is already populated)."""
    runs = {
        int(r["id"]): r for r in seeded_db.fetchall("SELECT * FROM monitoring_runs ORDER BY id")
    }
    ads = {
        str(r["platform_ad_id"]): int(r["id"])
        for r in seeded_db.fetchall("SELECT id, platform_ad_id FROM advertisements")
    }
    return {"runs": runs, "ads": ads}
