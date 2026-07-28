"""Provenance, terminology discipline and the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bama_eda.config import EdaConfig
from bama_eda.database import ReadOnlyDatabase
from bama_eda.models import APPROVED_TERMS, FORBIDDEN_PHRASES, AnalysisContext
from bama_eda.provenance import Manifest, build_context, environment, select_run, stamp


class TestAnalysisContext:
    def test_carries_every_required_identifier(self, eda_db: ReadOnlyDatabase) -> None:
        run, _ = select_run(eda_db, "latest-valid")
        context = build_context(eda_db, run)
        payload = context.as_dict()
        for key in (
            "run_id",
            "scheduled_for",
            "run_started_at",
            "run_finished_at",
            "search_url",
            "search_configuration_hash",
            "scraper_version",
            "monitor_version",
            "database_backend",
            "analysis_started_at",
            "analysis_version",
        ):
            assert key in payload, key

    def test_provenance_columns_are_a_subset_of_the_context(self, eda_db: ReadOnlyDatabase) -> None:
        run, _ = select_run(eda_db, "latest-valid")
        context = build_context(eda_db, run)
        assert set(context.provenance_columns()) <= set(context.as_dict())


class TestStamping:
    def _context(self, eda_db: ReadOnlyDatabase) -> AnalysisContext:
        run, _ = select_run(eda_db, "latest-valid")
        return build_context(eda_db, run)

    def test_columns_are_prefixed_so_they_are_seen_first(self, eda_db: ReadOnlyDatabase) -> None:
        frame = pd.DataFrame({"x": [1, 2]})
        out = stamp(frame, self._context(eda_db))
        assert list(out.columns)[0] == "run_id"
        assert list(out.columns)[-1] == "x"

    def test_a_colliding_source_column_is_renamed_not_lost(self, eda_db: ReadOnlyDatabase) -> None:
        frame = pd.DataFrame({"search_configuration_hash": ["older-hash"], "x": [1]})
        out = stamp(frame, self._context(eda_db))
        assert out.loc[0, "source_search_configuration_hash"] == "older-hash"
        assert out.loc[0, "search_configuration_hash"] != "older-hash"

    def test_an_empty_frame_still_gets_the_columns(self, eda_db: ReadOnlyDatabase) -> None:
        out = stamp(pd.DataFrame(), self._context(eda_db))
        assert "run_id" in out.columns


class TestManifest:
    def test_records_everything_needed_to_reproduce(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig, tmp_path: Path
    ) -> None:
        run, _ = select_run(eda_db, "latest-valid")
        manifest = Manifest(build_context(eda_db, run), cfg)
        manifest.source_tables["advertisements"] = 8
        manifest.record_exclusion("d", "reason", 3)
        manifest.warn("area", "message")
        path = manifest.write(tmp_path / "m.json")

        payload = json.loads(path.read_text())
        assert payload["analysis"]["run_id"] == int(run["id"])
        assert payload["database"]["source_tables"]["advertisements"] == 8
        assert payload["exclusions"][0]["count"] == 3
        assert payload["warnings"][0]["area"] == "area"
        assert payload["environment"]["python"]
        assert payload["configuration"]["random_seed"] == cfg.random_seed
        assert payload["configuration"]["fingerprint"]

    def test_credentials_never_reach_the_manifest(
        self, eda_db: ReadOnlyDatabase, tmp_path: Path
    ) -> None:
        cfg = EdaConfig(
            database_url="postgresql://user:sup3rsecret@host:5432/db",
            output_dir=tmp_path,
        )
        run, _ = select_run(eda_db, "latest-valid")
        payload = Manifest(build_context(eda_db, run), cfg).as_dict()
        blob = json.dumps(payload)
        assert "sup3rsecret" not in blob
        assert payload["database"]["url_shape"] == "postgresql://<redacted>"

    def test_environment_records_versions(self) -> None:
        env = environment()
        assert env["python"]
        assert "pandas" in env["dependencies"]


class TestTerminology:
    def test_forbidden_phrases_are_declared(self) -> None:
        assert "time_to_sale" in FORBIDDEN_PHRASES
        assert "sale_probability" in FORBIDDEN_PHRASES

    def test_approved_terms_are_declared(self) -> None:
        for term in (
            "time_to_disappearance",
            "sale_evidence_score",
            "left_truncated",
            "right_censored",
            "vehicle_entity_id",
        ):
            assert term in APPROVED_TERMS

    def test_no_module_uses_a_forbidden_phrase_as_an_identifier(self) -> None:
        """A column or function named `time_to_sale` would leak into every export."""
        offenders = []
        for path in Path("src/bama_eda").rglob("*.py"):
            if path.name == "models.py":
                continue  # where the forbidden list itself lives
            text = path.read_text(encoding="utf-8")
            for phrase in ("time_to_sale", "sale_probability", "probability_of_sale"):
                for line in text.splitlines():
                    stripped = line.strip()
                    if phrase in line and not stripped.startswith(("#", '"', "'")):
                        offenders.append(f"{path.name}: {stripped[:70]}")
        assert not offenders, offenders
