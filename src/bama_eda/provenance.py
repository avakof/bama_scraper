"""Run selection, provenance stamping and the manifest.

Two responsibilities, both about making a number traceable:

1. choose the analysis run **reproducibly**, from conditions checked against the
   real schema rather than assumed from documentation;
2. record everything needed to reproduce the analysis — versions, query hashes,
   row counts, exclusions — in ``eda_manifest.json``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bama_monitor.db import parse_ts

from .config import EdaConfig
from .database import ReadOnlyDatabase, utcnow
from .models import ANALYSIS_VERSION, AnalysisContext, AnalysisWarning

#: Run-selection conditions, expressed as (column, predicate) so they can be
#: applied only when the column actually exists.
RUN_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("status", "status = 'valid'"),
    ("finished_at", "finished_at IS NOT NULL"),
    ("comparison_applied", "comparison_applied = {{TRUE}}"),
    ("discovered_count", "discovered_count > 0"),
)


class RunSelectionError(RuntimeError):
    """No run satisfies the required conditions."""


def select_run(
    db: ReadOnlyDatabase, run_id: str | int = "latest-valid"
) -> tuple[dict[str, Any], list[AnalysisWarning]]:
    """Choose the analysis run.

    ``latest-valid`` picks the newest run that is ``valid``, finished, had its
    comparison applied and persisted a non-empty inventory. Each condition is
    applied only if the column exists, and any condition that had to be skipped is
    returned as a warning rather than silently dropped — a run chosen under fewer
    conditions than advertised is a different run.
    """
    if not db.has_table("monitoring_runs"):
        raise RunSelectionError(
            "monitoring_runs table is absent; this is not a monitoring database"
        )

    warnings: list[AnalysisWarning] = []
    available = {c["column_name"] for c in db.columns("monitoring_runs")}

    if str(run_id) not in ("latest-valid", "latest"):
        row = db.fetchone("SELECT * FROM monitoring_runs WHERE id = ?", [int(run_id)])
        if row is None:
            raise RunSelectionError(f"run {run_id} does not exist")
        for column, _ in RUN_CONDITIONS:
            if column in available and column == "status" and str(row.get("status")) != "valid":
                warnings.append(
                    AnalysisWarning(
                        "run_selection",
                        f"run {run_id} was explicitly requested but its status is "
                        f"{row.get('status')!r}; absence observations from a non-valid run "
                        "are not trustworthy",
                    )
                )
        return row, warnings

    clauses: list[str] = []
    for column, predicate in RUN_CONDITIONS:
        if column in available:
            clauses.append(predicate)
        else:
            warnings.append(
                AnalysisWarning(
                    "run_selection",
                    f"column monitoring_runs.{column} is absent, so the condition "
                    f"{predicate!r} could not be applied",
                )
            )
    where = " AND ".join(clauses) if clauses else "1=1"
    order = "scheduled_for DESC, id DESC" if "scheduled_for" in available else "id DESC"
    row = db.fetchone(f"SELECT * FROM monitoring_runs WHERE {where} ORDER BY {order} LIMIT 1")
    if row is None:
        raise RunSelectionError(
            "no run satisfies status='valid' AND finished_at IS NOT NULL AND "
            "comparison_applied AND discovered_count > 0"
        )
    return row, warnings


def build_context(
    db: ReadOnlyDatabase, run: dict[str, Any], *, analysis_started_at: datetime | None = None
) -> AnalysisContext:
    """Freeze the identity of this analysis."""
    return AnalysisContext(
        run_id=int(run["id"]),
        scheduled_for=parse_ts(run.get("scheduled_for")),
        run_started_at=parse_ts(run.get("started_at")),
        run_finished_at=parse_ts(run.get("finished_at")),
        search_url=run.get("search_url"),
        search_configuration_hash=run.get("configuration_hash"),
        scraper_version=run.get("scraper_version"),
        monitor_version=run.get("monitor_version"),
        database_backend=db.dialect,
        analysis_started_at=analysis_started_at or utcnow(),
    )


def stamp(frame: pd.DataFrame, context: AnalysisContext) -> pd.DataFrame:
    """Prefix a dataset with its provenance columns.

    Prefix rather than append so the identifiers are the first thing a reader sees
    when the file is opened in a spreadsheet.

    A source column of the same name (an advertisement carries its own
    ``search_configuration_hash``, for instance) is **renamed**, not overwritten.
    The two can legitimately differ — an advertisement first discovered under an
    older search definition keeps that hash — and losing the difference would hide
    exactly the population change the hash exists to reveal.
    """
    provenance = context.provenance_columns()
    stamped = frame.copy()
    collisions = [name for name in provenance if name in stamped.columns]
    if collisions:
        stamped = stamped.rename(columns={name: f"source_{name}" for name in collisions})
    for name, value in reversed(list(provenance.items())):
        if stamped.empty:
            stamped.insert(0, name, pd.Series(dtype="object"))
        else:
            stamped.insert(0, name, value)
    return stamped


def environment() -> dict[str, Any]:
    """Everything needed to reproduce the numbers, or explain why they differ."""
    versions: dict[str, str] = {}
    for module in ("pandas", "numpy", "pyarrow", "matplotlib", "scipy", "yaml", "pydantic"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - a missing optional dep is informative
            versions[module] = "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": versions,
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


class Manifest:
    """Accumulates everything that makes the outputs auditable."""

    def __init__(self, context: AnalysisContext, cfg: EdaConfig) -> None:
        self.context = context
        self.cfg = cfg
        self.source_tables: dict[str, int] = {}
        self.outputs: dict[str, Any] = {}
        self.exclusions: list[dict[str, Any]] = []
        self.warnings: list[AnalysisWarning] = []
        self.queries: list[dict[str, Any]] = []
        self.enrichment: dict[str, Any] = {"enabled": False}
        self.duplicates: dict[str, int] = {}

    def record_output(self, name: str, path: Path, rows: int | None = None) -> None:
        self.outputs[name] = {
            "path": str(path),
            "rows": rows,
            "bytes": path.stat().st_size if path.exists() else None,
        }

    def record_exclusion(self, dataset: str, reason: str, count: int) -> None:
        """Every row dropped is accounted for, with a reason.

        Silent row removal is how an analysis ends up describing a population that
        does not exist.
        """
        self.exclusions.append({"dataset": dataset, "reason": reason, "count": int(count)})

    def warn(self, area: str, message: str) -> None:
        self.warnings.append(AnalysisWarning(area, message))

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.context.as_dict(),
            "database": {
                "backend": self.cfg.backend,
                # The URL itself may carry credentials, so only the shape is kept.
                "url_shape": (
                    "postgresql://<redacted>" if self.cfg.is_postgres else "sqlite:///<path>"
                ),
                "source_tables": self.source_tables,
            },
            "configuration": {
                "run_id_requested": self.cfg.run_id,
                "thresholds": self.cfg.thresholds.model_dump(),
                "include_left_truncated": self.cfg.include_left_truncated,
                "random_seed": self.cfg.random_seed,
                "fingerprint": self.cfg.fingerprint(),
            },
            "attribute_enrichment": self.enrichment,
            "outputs": self.outputs,
            "duplicates": self.duplicates,
            "exclusions": self.exclusions,
            "warnings": [w.as_dict() for w in self.warnings],
            "queries": self.queries,
            "environment": environment(),
            "package_version": ANALYSIS_VERSION,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
