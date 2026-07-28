"""Audit gates, the generated wide pivot, exports and the CLI surface."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from bama_deep.audit import audit_dataset
from bama_deep.cli import build_parser, main
from bama_deep.config import DeepConfig
from bama_deep.export import export_all
from bama_deep.specs import rebuild_spec_catalog
from bama_deep.storage import DeepStorage


def build(
    tmp_path: Path, *, leak: bool = False, critical: int = 0
) -> tuple[DeepConfig, DeepStorage]:
    """A small but structurally complete dataset."""
    cfg = DeepConfig(output_dir=tmp_path, source_db=tmp_path / "missing.sqlite")
    cfg.ensure_dirs()
    store = DeepStorage(cfg.db_path)
    store.start_run("r1", source_db="s", source_sha256=None, phases="a", version="t", config={})
    store.seed_queue([(f"a{i}", f"https://bama.ir/car/detail-a{i}-x", "api", "") for i in range(3)])

    for i in range(3):
        record = {
            "ad_id": f"a{i}",
            "url": f"https://bama.ir/car/detail-a{i}-x",
            "fuel_type": "بنزینی",
            "province": "تهران",
            "city": "تهران",
            "price_toman": 1_000_000_000 + i,
            "mileage_km": 1000 * i,
            "is_promoted": 0,
            "condition_new_used": "used",
            "review_key": "/car-reviews/dena/plusef7p-specs-1481-6mt",
            "description": "تماس 09123456789" if (leak and i == 0) else "توضیحات سالم",
            "numeric_agreement_json": json.dumps({"acceleration_s": "exact"}),
            "is_delisted": 0,
            "parse_status": "ok",
        }
        store.save_ad(record, [], [], [])
        store.set_ad_status(f"a{i}", "completed")

    for i in range(critical):
        store.conn.execute(
            "INSERT OR REPLACE INTO ad_field_conflicts (ad_id, field, severity) VALUES (?,?,?)",
            (f"a{i}", "price_toman", "critical"),
        )
    store.conn.commit()

    store.upsert(
        "trim_reviews",
        ["review_key"],
        {
            "review_key": "/car-reviews/dena/plusef7p-specs-1481-6mt",
            "review_url": "/car-reviews/dena/plusef7p-specs-1481-6mt",
            "status": "completed",
        },
    )
    store.upsert_many(
        "trim_spec_groups",
        ["review_key", "group_index"],
        [
            {
                "review_key": "/car-reviews/dena/plusef7p-specs-1481-6mt",
                "group_index": i,
                "group_name": name,
                "group_slug": slug,
                "item_count": 1,
            }
            for i, (name, slug) in enumerate(
                [("ابعاد و اندازه ها", "dimensions"), ("سیستم‌های ترمز", "brakes")]
            )
        ],
    )
    store.upsert_many(
        "trim_specs",
        ["review_key", "group_slug", "item_slug"],
        [
            {
                "review_key": "/car-reviews/dena/plusef7p-specs-1481-6mt",
                "group_slug": group,
                "item_slug": slug,
                "item_key": key,
                "group_name": group,
                "position": 0,
                "value_type": vtype,
                "value_raw": raw,
                "value_text": raw,
                "value_bool": vbool,
                "value_num": vnum,
                "value_unit": unit,
            }
            for group, slug, key, vtype, raw, vbool, vnum, unit in [
                ("dimensions", "weight", "وزن", "String", "1258 کیلوگرم", None, 1258.0, "کیلوگرم"),
                ("brakes", "abs", "ترمز ضدقفل (ABS)", "Boolean", "true", 1, None, None),
                # The same slug in two groups -- must not collapse into one column.
                ("dimensions", "other_features", "سایر ویژگی‌ها", "String", "x", None, None, None),
                ("brakes", "other_features", "سایر ویژگی‌ها", "String", "y", None, None, None),
            ]
        ],
    )
    return cfg, store


class TestAuditGates:
    def test_clean_dataset_passes(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        report = audit_dataset(cfg, store)
        assert report["checks_passed"] is True
        assert report["integrity"]["contact_leaks"] == 0
        store.close()

    def test_contact_leak_fails_the_audit(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path, leak=True)
        report = audit_dataset(cfg, store)
        assert report["checks_passed"] is False
        assert report["integrity"]["contact_leaks"] >= 1
        assert report["contact_leak_samples"]
        store.close()

    def test_critical_conflict_rate_over_threshold_fails(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path, critical=3)
        report = audit_dataset(cfg, store)
        assert report["critical_conflict_rate"] > cfg.max_critical_conflict_rate
        assert report["checks_passed"] is False
        store.close()

    def test_unprocessed_queue_row_fails(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        store.seed_queue([("zz", "https://bama.ir/car/detail-zz-x", "api", "")])
        report = audit_dataset(cfg, store)
        assert report["integrity"]["queue_not_terminal"] == 1
        assert report["checks_passed"] is False
        store.close()

    def test_completed_without_a_row_fails(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        store.seed_queue([("zz", "u", "api", "")])
        store.set_ad_status("zz", "completed")
        report = audit_dataset(cfg, store)
        assert report["integrity"]["completed_without_ad_row"] == 1
        assert report["checks_passed"] is False
        store.close()

    def test_delisted_counted_separately_from_failures(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        store.seed_queue([("gone", "u", "api", "")])
        store.save_ad({"ad_id": "gone", "url": "u", "is_delisted": 1}, [], [], [])
        store.set_ad_status("gone", "delisted")
        report = audit_dataset(cfg, store)
        assert report["ads_delisted"] == 1
        assert report["checks_passed"] is True  # churn is not a failure
        store.close()

    def test_coverage_is_reported(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        report = audit_dataset(cfg, store)
        assert report["field_coverage"]["fuel_type"]["count"] == 3
        store.close()


class TestReindex:
    def test_collided_slug_is_group_qualified(self, tmp_path: Path) -> None:
        # "other features" exists in several groups; one column cannot hold all.
        cfg, store = build(tmp_path)
        result = rebuild_spec_catalog(store)
        assert result["collided_slugs"] == ["other_features"]
        columns = [r[1] for r in store.conn.execute("PRAGMA table_info(v_trim_specs_wide)")]
        assert "dimensions__other_features" in columns
        assert "brakes__other_features" in columns
        assert len(columns) == len(set(columns)), "duplicate column names in the wide view"
        store.close()

    def test_unique_slugs_keep_plain_names(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        rebuild_spec_catalog(store)
        columns = [r[1] for r in store.conn.execute("PRAGMA table_info(v_trim_specs_wide)")]
        assert "weight" in columns and "abs" in columns
        store.close()

    def test_catalog_count_is_the_stored_count(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        result = rebuild_spec_catalog(store)
        assert result["catalog_items"] == store.count("spec_key_catalog")
        # The catalog is keyed on slug alone, so it is smaller than the pair count.
        assert result["slug_group_pairs"] > result["catalog_items"]
        store.close()

    def test_reindex_is_idempotent(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        first = rebuild_spec_catalog(store)
        second = rebuild_spec_catalog(store)
        assert first == second
        store.close()


class TestExports:
    @pytest.fixture
    def exported(self, tmp_path: Path):
        cfg, store = build(tmp_path)
        rebuild_spec_catalog(store)
        manifest = export_all(cfg, store)
        yield cfg, store, manifest
        store.close()

    def test_core_files_written(self, exported) -> None:
        cfg, _, manifest = exported
        for name in (
            "ads_deep.csv",
            "ads_deep.jsonl",
            "trim_specs_long.csv",
            "trim_specs_wide.csv",
            "price_points.csv",
            "dealers.csv",
            "deep_errors.csv",
            "data_dictionary.md",
        ):
            assert (cfg.exports_dir / name).exists(), name
            assert name in manifest["files"]

    def test_one_row_per_advertisement(self, exported) -> None:
        cfg, _, manifest = exported
        assert manifest["rows"]["ads_deep"] == 3
        lines = (cfg.exports_dir / "ads_deep.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_csv_has_bom_and_persian_survives(self, exported) -> None:
        cfg, _, _ = exported
        raw = (cfg.exports_dir / "ads_deep.csv").read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        assert "بنزینی" in raw.decode("utf-8-sig")

    def test_heavy_evidence_columns_are_excluded(self, exported) -> None:
        cfg, _, _ = exported
        header = (cfg.exports_dir / "ads_deep.csv").read_text(encoding="utf-8-sig").split("\n")[0]
        for column in ("api_raw_json", "html_payload_json", "json_ld_json"):
            assert column not in header

    def test_nested_export_carries_specs(self, exported) -> None:
        cfg, _, _ = exported
        record = json.loads(
            (cfg.exports_dir / "ads_full.jsonl").read_text(encoding="utf-8").split("\n")[0]
        )
        assert {"media", "trim_specs", "price_series", "dealer"} <= set(record)
        assert record["trim_specs"], "specs should join through review_key"

    def test_manifest_fingerprints_every_file(self, exported) -> None:
        _, _, manifest = exported
        for name, info in manifest["files"].items():
            assert len(info["sha256"]) == 64, name
            assert info["bytes"] > 0, name

    def test_snapshot_is_queryable(self, exported) -> None:
        cfg, _, _ = exported
        conn = sqlite3.connect(cfg.snapshot_path)
        assert conn.execute("SELECT COUNT(*) FROM ad_deep").fetchone()[0] == 3
        conn.close()

    def test_dealer_address_can_be_omitted(self, tmp_path: Path) -> None:
        cfg, store = build(tmp_path)
        cfg = cfg.model_copy(update={"no_dealer_address": True})
        store.upsert("dealers", ["dealer_id"], {"dealer_id": 1, "address": "خیابان تست"})
        export_all(cfg, store)
        header = (cfg.exports_dir / "dealers.csv").read_text(encoding="utf-8-sig").split("\n")[0]
        assert "address" not in header
        store.close()

    def test_data_dictionary_lists_the_features(self, exported) -> None:
        cfg, _, _ = exported
        text = (cfg.exports_dir / "data_dictionary.md").read_text(encoding="utf-8")
        assert "weight" in text and "abs" in text


class TestCli:
    @pytest.mark.parametrize(
        "command",
        [
            "seed",
            "ads",
            "specs",
            "prices",
            "dealers",
            "run",
            "audit",
            "export",
            "reindex",
            "stats",
            "plan",
        ],
    )
    def test_every_subcommand_parses(self, command: str) -> None:
        args = build_parser().parse_args([command])
        assert args.command == command

    def test_unknown_subcommand_exits_2(self) -> None:
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["nonsense"])
        assert exc.value.code == 2

    def test_stats_on_a_fresh_database(self, tmp_path: Path, capsys) -> None:
        code = main(["stats", "--output-dir", str(tmp_path)])
        assert code == 0
        assert json.loads(capsys.readouterr().out)["ads"] == 0

    def test_plan_issues_no_requests(self, tmp_path: Path, capsys) -> None:
        code = main(["plan", "--output-dir", str(tmp_path)])
        assert code == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["requests_per_ad"] == 2
        assert plan["total_estimated_requests"] >= 0

    def test_seed_with_missing_source_exits_2(self, tmp_path: Path) -> None:
        assert (
            main(["seed", "--output-dir", str(tmp_path), "--source-db", str(tmp_path / "no.db")])
            == 2
        )

    def test_output_dir_is_respected(self, tmp_path: Path) -> None:
        main(["stats", "--output-dir", str(tmp_path)])
        assert (tmp_path / "bama_deep.sqlite").exists()
