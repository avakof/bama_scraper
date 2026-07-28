"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_DEEP_SQLITE, EdaConfig, load_config
from .database import ReadOnlyDatabase
from .dataset_builder import build_cross_section, build_panel, build_vehicle_dataset
from .provenance import Manifest, RunSelectionError, build_context, select_run
from .schema_inspection import SchemaError, assert_analysable, write_schema_outputs
from .validation import CriticalIntegrityError, audit, summarise, write_reports

COMMANDS = (
    ("inspect-schema", "inspect the database and write the schema inventory"),
    ("build-datasets", "build the three analytical datasets and write them"),
    ("validate", "run the integrity audit without producing charts"),
    ("run", "full analysis: datasets, audit, statistics, charts, report"),
    ("report", "regenerate the HTML report from a fresh analysis"),
    ("compare-runs", "compare the inventory of two monitoring runs"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bama_eda",
        description="Exploratory data analysis over the Bama monitoring database.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS:
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--database-url",
            default=None,
            help="postgresql://... or sqlite:///... (never hard-code credentials)",
        )
        p.add_argument("--config", default=None, help="YAML config file")
        p.add_argument("--output-dir", default=None)
        p.add_argument("--log-level", default=None)
        if name != "compare-runs":
            p.add_argument("--run-id", default=None, help="numeric id, or 'latest-valid' (default)")
        if name in ("run", "report", "build-datasets", "validate"):
            p.add_argument("--min-group-size", type=int, default=None)
            p.add_argument("--min-survival-group-size", type=int, default=None)
            p.add_argument("--min-survival-events", type=int, default=None)
            p.add_argument("--snapshot-stale-hours", type=float, default=None)
            p.add_argument(
                "--include-left-truncated",
                action="store_true",
                default=None,
                help="include left-truncated rows in duration rankings (off by "
                "default: their earlier market life was never observed)",
            )
            p.add_argument(
                "--enrich-attributes",
                action="store_true",
                default=None,
                help="join vehicle attributes from the deep scraper database",
            )
            p.add_argument("--enrichment-db", default=None)
            p.add_argument("--export-csv", dest="export_csv", action="store_true", default=None)
            p.add_argument("--no-export-csv", dest="export_csv", action="store_false")
            p.add_argument(
                "--export-parquet", dest="export_parquet", action="store_true", default=None
            )
            p.add_argument("--no-export-parquet", dest="export_parquet", action="store_false")
        if name in ("run", "report"):
            p.add_argument("--html", dest="html", action="store_true", default=None)
            p.add_argument("--no-html", dest="html", action="store_false")
            p.add_argument("--no-charts", dest="charts", action="store_false", default=True)
        if name == "compare-runs":
            p.add_argument("--run-id-a", type=int, required=True)
            p.add_argument("--run-id-b", type=int, required=True)
    return parser


def _config(args: argparse.Namespace) -> EdaConfig:
    overrides: dict[str, Any] = {
        "database_url": getattr(args, "database_url", None),
        "run_id": getattr(args, "run_id", None),
        "output_dir": Path(args.output_dir) if getattr(args, "output_dir", None) else None,
        "log_level": getattr(args, "log_level", None),
        "include_left_truncated": getattr(args, "include_left_truncated", None),
        "html": getattr(args, "html", None),
        "export_csv": getattr(args, "export_csv", None),
        "export_parquet": getattr(args, "export_parquet", None),
        "min_group_size": getattr(args, "min_group_size", None),
        "min_survival_group_size": getattr(args, "min_survival_group_size", None),
        "min_survival_events": getattr(args, "min_survival_events", None),
        "snapshot_stale_hours": getattr(args, "snapshot_stale_hours", None),
    }
    cfg = load_config(getattr(args, "config", None), **overrides)
    if getattr(args, "enrich_attributes", None):
        cfg.enrichment.enabled = True
        cfg.enrichment.database_path = (
            Path(args.enrichment_db)
            if getattr(args, "enrichment_db", None)
            else DEFAULT_DEEP_SQLITE
        )
    cfg.ensure_dirs()
    return cfg


def _dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)

    try:
        if args.command == "inspect-schema":
            return _inspect_schema(cfg)
        if args.command == "build-datasets":
            return _build_datasets(cfg)
        if args.command == "validate":
            return _validate(cfg)
        if args.command in ("run", "report"):
            return _run(cfg, make_charts=getattr(args, "charts", True))
        if args.command == "compare-runs":
            return _compare_runs(cfg, args.run_id_a, args.run_id_b)
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 3
    except RunSelectionError as exc:
        print(f"run selection failed: {exc}", file=sys.stderr)
        return 4
    except CriticalIntegrityError as exc:
        print(f"CRITICAL: {exc}", file=sys.stderr)
        print(
            "the data-quality report was still written; inspect it before rerunning",
            file=sys.stderr,
        )
        return 5

    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


def _inspect_schema(cfg: EdaConfig) -> int:
    with ReadOnlyDatabase(cfg.database_url) as db:
        absent = assert_analysable(db)
        out_dir = cfg.output_dir / "schema"
        paths = write_schema_outputs(db, out_dir)
        schema = pd.read_csv(paths["schema_inventory"])
        _dump(
            {
                "backend": db.dialect,
                "tables": len(db.table_names()),
                "columns": int(len(schema)),
                "absent_optional_tables": absent,
                "roles": schema["analytical_role"].value_counts().to_dict(),
                "outputs": {k: str(v) for k, v in paths.items()},
            }
        )
    return 0


def _build_datasets(cfg: EdaConfig) -> int:
    with ReadOnlyDatabase(cfg.database_url) as db:
        assert_analysable(db)
        run, warnings = select_run(db, cfg.run_id)
        context = build_context(db, run)
        manifest = Manifest(context, cfg)
        manifest.warnings.extend(warnings)

        out_dir = cfg.output_dir / f"datasets_run_{context.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cross = build_cross_section(db, context, cfg, manifest)
        panel = build_panel(db, context, cfg, manifest)
        vehicles = build_vehicle_dataset(db, cross, context, cfg, manifest)

        counts = {}
        for name, frame in (
            ("advertisement_cross_section", cross),
            ("longitudinal_panel", panel),
            ("vehicle_entity_dataset", vehicles),
        ):
            if cfg.export_parquet:
                from .report_builder import _parquet_safe

                _parquet_safe(frame).to_parquet(out_dir / f"{name}.parquet", index=False)
            if cfg.export_csv:
                frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            counts[name] = int(len(frame))

        manifest.write(out_dir / "eda_manifest.json")
        _dump(
            {
                "run_id": context.run_id,
                "output_dir": str(out_dir),
                "row_counts": counts,
                "exclusions": manifest.exclusions,
                "warnings": [w.as_dict() for w in manifest.warnings],
            }
        )
    return 0


def _validate(cfg: EdaConfig) -> int:
    with ReadOnlyDatabase(cfg.database_url) as db:
        assert_analysable(db)
        run, warnings = select_run(db, cfg.run_id)
        context = build_context(db, run)
        manifest = Manifest(context, cfg)
        manifest.warnings.extend(warnings)

        cross = build_cross_section(db, context, cfg, manifest)
        panel = build_panel(db, context, cfg, manifest)
        vehicles = build_vehicle_dataset(db, cross, context, cfg, manifest)
        datasets = {
            "advertisement_cross_section": cross,
            "longitudinal_panel": panel,
            "vehicle_entity_dataset": vehicles,
        }
        findings = audit(db, datasets, context, cfg)
        out_dir = cfg.output_dir / f"validation_run_{context.run_id}"
        write_reports(findings, out_dir)
        summary = summarise(findings)
        _dump(
            {
                "run_id": context.run_id,
                "output_dir": str(out_dir),
                "summary": summary,
                "findings_with_issues": [f.as_dict() for f in findings if f.count],
            }
        )
        # Non-zero when the data cannot be trusted, so this is usable in CI.
        return 0 if summary["passed"] else 1


def _run(cfg: EdaConfig, *, make_charts: bool) -> int:
    from .report_builder import check_forbidden_phrases, run_analysis

    outcome = run_analysis(cfg, make_charts=make_charts)
    offenders = check_forbidden_phrases(outcome["output_dir"])
    _dump(
        {
            "run_id": outcome["context"].run_id,
            "output_dir": str(outcome["output_dir"]),
            "html_report": str(outcome["html"]) if outcome["html"] else None,
            "row_counts": outcome["row_counts"],
            "data_quality": outcome["quality"],
            "charts_generated": len(outcome["charts"].get("charts", [])),
            "charts_skipped": len(outcome["charts"].get("skipped", [])),
            "eligible_for_duration_ranking": outcome["results"]["publication"].get(
                "eligible_for_duration_ranking"
            ),
            "duration_ranking": outcome["results"]["duration_feasibility"].get("verdict"),
            "sale_evidence_calibrated": outcome["results"]["sale_evidence"]
            .get("calibration", {})
            .get("calibrated"),
            "forbidden_phrase_offenders": offenders,
        }
    )
    return 0 if not offenders else 1


def _compare_runs(cfg: EdaConfig, run_a: int, run_b: int) -> int:
    """Inventory difference between two runs, with both runs' provenance."""
    with ReadOnlyDatabase(cfg.database_url) as db:
        assert_analysable(db)
        rows = db.fetchall("SELECT * FROM monitoring_runs WHERE id IN (?, ?)", [run_a, run_b])
        runs = {int(r["id"]): r for r in rows}
        missing = [r for r in (run_a, run_b) if r not in runs]
        if missing:
            print(f"run(s) not found: {missing}", file=sys.stderr)
            return 4

        def ids(run_id: int) -> set[str]:
            found = db.fetchall(
                "SELECT a.platform_ad_id FROM daily_ad_observations o"
                " JOIN advertisements a ON a.id = o.advertisement_id"
                " WHERE o.run_id = ? AND o.was_seen = ?",
                [run_id, True],
            )
            return {str(r["platform_ad_id"]) for r in found}

        seen_a, seen_b = ids(run_a), ids(run_b)
        _dump(
            {
                "run_a": {
                    "id": run_a,
                    "scheduled_for": str(runs[run_a].get("scheduled_for")),
                    "status": runs[run_a].get("status"),
                    "seen": len(seen_a),
                },
                "run_b": {
                    "id": run_b,
                    "scheduled_for": str(runs[run_b].get("scheduled_for")),
                    "status": runs[run_b].get("status"),
                    "seen": len(seen_b),
                },
                "in_both": len(seen_a & seen_b),
                "only_in_a": len(seen_a - seen_b),
                "only_in_b": len(seen_b - seen_a),
                "sample_only_in_a": sorted(seen_a - seen_b)[:10],
                "sample_only_in_b": sorted(seen_b - seen_a)[:10],
                "interpretation": (
                    "'only_in_a' means absent from run B's results. That is an observed "
                    "absence, not a sale, and it is only meaningful if both runs are valid."
                ),
                "both_runs_valid": all(
                    str(runs[r].get("status")) == "valid" for r in (run_a, run_b)
                ),
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
