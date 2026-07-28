"""CLI behaviour, exit codes and end-to-end determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bama_eda.cli import main
from bama_eda.config import EdaConfig
from bama_eda.report_builder import check_forbidden_phrases, run_analysis


class TestCommands:
    def test_inspect_schema(self, seeded_db, tmp_path: Path, capsys) -> None:
        code = main(
            ["inspect-schema", "--database-url", seeded_db.url, "--output-dir", str(tmp_path)]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["tables"] > 5
        assert payload["columns"] > 50
        assert Path(payload["outputs"]["schema_inventory"]).exists()
        assert Path(payload["outputs"]["data_dictionary"]).exists()
        assert Path(payload["outputs"]["table_relationships"]).exists()

    def test_build_datasets(self, seeded_db, tmp_path: Path, capsys) -> None:
        code = main(
            [
                "build-datasets",
                "--database-url",
                seeded_db.url,
                "--output-dir",
                str(tmp_path),
                "--no-export-parquet",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        counts = payload["row_counts"]
        assert counts["advertisement_cross_section"] == 8
        assert counts["vehicle_entity_dataset"] == 7  # F and G share one vehicle
        assert counts["longitudinal_panel"] > counts["advertisement_cross_section"]

    def test_validate_returns_zero_on_clean_data(self, seeded_db, tmp_path: Path, capsys) -> None:
        code = main(["validate", "--database-url", seeded_db.url, "--output-dir", str(tmp_path)])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["passed"]

    def test_compare_runs(self, seeded_db, tmp_path: Path, capsys) -> None:
        runs = seeded_db.fetchall("SELECT id FROM monitoring_runs ORDER BY id")
        code = main(
            [
                "compare-runs",
                "--database-url",
                seeded_db.url,
                "--output-dir",
                str(tmp_path),
                "--run-id-a",
                str(runs[0]["id"]),
                "--run-id-b",
                str(runs[-1]["id"]),
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "in_both" in payload
        assert "not a sale" in payload["interpretation"]

    def test_unknown_run_id_exits_four(self, seeded_db, tmp_path: Path) -> None:
        code = main(
            [
                "validate",
                "--database-url",
                seeded_db.url,
                "--output-dir",
                str(tmp_path),
                "--run-id",
                "999999",
            ]
        )
        assert code == 4

    def test_a_non_monitoring_database_exits_three(self, tmp_path: Path) -> None:
        import sqlite3

        path = tmp_path / "empty.sqlite"
        sqlite3.connect(path).execute("CREATE TABLE unrelated (x INTEGER)")
        code = main(
            ["inspect-schema", "--database-url", f"sqlite:///{path}", "--output-dir", str(tmp_path)]
        )
        assert code == 3


class TestFullRun:
    def test_run_produces_every_required_artefact(self, seeded_db, tmp_path: Path) -> None:
        cfg = EdaConfig(
            database_url=seeded_db.url,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            html=True,
            export_parquet=False,
        )
        outcome = run_analysis(cfg, make_charts=True)
        out = outcome["output_dir"]
        for name in (
            "README.txt",
            "eda_manifest.json",
            "data_quality_report.json",
            "data_quality_issues.csv",
            "schema_inventory.csv",
            "data_dictionary.csv",
            "advertisement_cross_section.csv",
            "field_completeness.csv",
            "missingness_patterns.csv",
            "descriptive_statistics.csv",
            "categorical_distributions.csv",
            "price_by_segment.csv",
            "price_changes.csv",
            "status_transitions.csv",
            "duration_summary.csv",
            "duration_ranking_eligible.csv",
            "left_truncated_excluded.csv",
            "filter_exits.csv",
            "repost_analysis.csv",
            "sale_evidence_summary.csv",
            "analysis_warnings.csv",
            "bama_eda_report.html",
        ):
            assert (out / name).exists(), name
        assert (out / "charts").is_dir()
        assert list((out / "charts").glob("*.png"))

    def test_no_forbidden_phrasing_in_any_artefact(self, seeded_db, tmp_path: Path) -> None:
        cfg = EdaConfig(
            database_url=seeded_db.url,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "c",
            html=True,
            export_parquet=False,
        )
        outcome = run_analysis(cfg, make_charts=False)
        assert check_forbidden_phrases(outcome["output_dir"]) == []

    def test_report_is_deterministic(self, seeded_db, tmp_path: Path) -> None:
        """Two runs over the same database must produce the same numbers."""

        def analyse(tag: str) -> pd.DataFrame:
            cfg = EdaConfig(
                database_url=seeded_db.url,
                output_dir=tmp_path / tag,
                cache_dir=tmp_path / f"c{tag}",
                html=False,
                export_parquet=False,
            )
            outcome = run_analysis(cfg, make_charts=False)
            return pd.read_csv(outcome["output_dir"] / "descriptive_statistics.csv")

        pd.testing.assert_frame_equal(analyse("a"), analyse("b"))

    def test_left_truncated_rows_are_excluded_but_published(
        self, seeded_db, tmp_path: Path
    ) -> None:
        cfg = EdaConfig(
            database_url=seeded_db.url,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "c",
            html=False,
            export_parquet=False,
        )
        outcome = run_analysis(cfg, make_charts=False)
        excluded = pd.read_csv(outcome["output_dir"] / "left_truncated_excluded.csv")
        assert len(excluded) > 0
        ranking = pd.read_csv(outcome["output_dir"] / "duration_ranking_eligible.csv")
        assert "insufficient_data_for_duration_ranking" in ranking["verdict"].tolist()

    def test_private_columns_never_reach_an_export(self, seeded_db, tmp_path: Path) -> None:
        cfg = EdaConfig(
            database_url=seeded_db.url,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "c",
            html=False,
            export_parquet=False,
        )
        outcome = run_analysis(cfg, make_charts=False)
        cross = pd.read_csv(outcome["output_dir"] / "advertisement_cross_section.csv")
        assert not {"phone", "dealer_address", "address"} & set(cross.columns)
        # the fixture plants a contact number in a description
        text = (outcome["output_dir"] / "advertisement_cross_section.csv").read_text(
            encoding="utf-8-sig"
        )
        assert "09123456789" not in text

    def test_manifest_records_row_counts_and_queries(self, seeded_db, tmp_path: Path) -> None:
        cfg = EdaConfig(
            database_url=seeded_db.url,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "c",
            html=False,
            export_parquet=False,
        )
        outcome = run_analysis(cfg, make_charts=False)
        manifest = json.loads((outcome["output_dir"] / "eda_manifest.json").read_text())
        assert manifest["database"]["source_tables"]["advertisements"] == 8
        assert manifest["queries"], "every analytical query must be logged"
        assert all("sql_sha256" in q for q in manifest["queries"])
