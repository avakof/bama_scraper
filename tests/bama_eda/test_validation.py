"""The integrity audit must actually catch the things it claims to catch."""

from __future__ import annotations

import pandas as pd
import pytest
from eda_scenarios import T0

from bama_eda.config import EdaConfig
from bama_eda.database import ReadOnlyDatabase
from bama_eda.dataset_builder import build_cross_section, build_panel, build_vehicle_dataset
from bama_eda.models import Severity
from bama_eda.provenance import Manifest, build_context, select_run
from bama_eda.validation import CriticalIntegrityError, audit, enforce, summarise


def _datasets(eda_db: ReadOnlyDatabase, cfg: EdaConfig):
    run, _ = select_run(eda_db, "latest-valid")
    context = build_context(eda_db, run)
    manifest = Manifest(context, cfg)
    cross = build_cross_section(eda_db, context, cfg, manifest)
    panel = build_panel(eda_db, context, cfg, manifest)
    vehicles = build_vehicle_dataset(eda_db, cross, context, cfg, manifest)
    return {
        "advertisement_cross_section": cross,
        "longitudinal_panel": panel,
        "vehicle_entity_dataset": vehicles,
    }, context


class TestCleanFixturePasses:
    def test_no_critical_findings(self, eda_db: ReadOnlyDatabase, cfg: EdaConfig) -> None:
        datasets, context = _datasets(eda_db, cfg)
        findings = audit(eda_db, datasets, context, cfg)
        summary = summarise(findings)
        assert summary["passed"], summary["critical_findings"]

    def test_every_check_is_reported_including_the_ones_that_passed(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        """A report listing only failures cannot be told from one where nothing ran."""
        datasets, context = _datasets(eda_db, cfg)
        findings = audit(eda_db, datasets, context, cfg)
        assert len(findings) > 20
        assert any(f.count == 0 for f in findings)


class TestDetectsInjectedFaults:
    def test_the_database_itself_forbids_a_duplicate_advertisement_id(self, seeded_db) -> None:
        """The first line of defence is the constraint, not the audit."""
        import sqlite3

        with pytest.raises((sqlite3.IntegrityError, Exception)):
            seeded_db.execute(
                "INSERT INTO advertisements (platform, platform_ad_id, canonical_url,"
                " current_status, created_at, updated_at)"
                " VALUES ('bama','A','https://bama.ir/car/detail-A-dup','active',?,?)",
                [T0, T0],
            )

    def test_a_duplicated_cross_section_row_is_critical(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        """The audit's own job: catch a join that stopped being one-to-one."""
        datasets, context = _datasets(eda_db, cfg)
        cross = datasets["advertisement_cross_section"]
        datasets["advertisement_cross_section"] = pd.concat(
            [cross, cross.head(1)], ignore_index=True
        )
        findings = audit(eda_db, datasets, context, cfg)
        finding = next(f for f in findings if f.check == "cross_section_duplicate_rows")
        assert finding.count == 2  # both copies are reported
        assert finding.severity is Severity.CRITICAL
        with pytest.raises(CriticalIntegrityError):
            enforce(findings)

    def test_private_column_in_a_dataset_is_critical(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        datasets, context = _datasets(eda_db, cfg)
        datasets["advertisement_cross_section"]["phone"] = "09123456789"
        findings = audit(eda_db, datasets, context, cfg)
        leak = next(f for f in findings if f.check == "private_column_in_output")
        assert leak.count == 1 and leak.severity is Severity.CRITICAL

    def test_contact_number_in_free_text_is_critical(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        datasets, context = _datasets(eda_db, cfg)
        datasets["advertisement_cross_section"]["notes"] = "call 09123456789 now"
        findings = audit(eda_db, datasets, context, cfg)
        leak = next(f for f in findings if f.check == "contact_number_in_output")
        assert leak.count > 0 and leak.severity is Severity.CRITICAL

    def test_a_price_is_not_mistaken_for_a_phone_number(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        """2,000,000,000 and a sha256 digest must not trip the contact detector."""
        datasets, context = _datasets(eda_db, cfg)
        datasets["advertisement_cross_section"]["notes"] = [
            "قیمت 2000000000 تومان" if i % 2 else "a" * 40 + "0912345678901234"
            for i in range(len(datasets["advertisement_cross_section"]))
        ]
        findings = audit(eda_db, datasets, context, cfg)
        leak = next(f for f in findings if f.check == "contact_number_in_output")
        # A ten-digit price and a digit run embedded in a hex-like string must not
        # match; the fixture's planted number is scrubbed at build time.
        assert leak.count == 0

    def test_missing_provenance_is_critical(self, eda_db: ReadOnlyDatabase, cfg: EdaConfig) -> None:
        datasets, context = _datasets(eda_db, cfg)
        datasets["advertisement_cross_section"] = datasets["advertisement_cross_section"].drop(
            columns=["run_id"]
        )
        findings = audit(eda_db, datasets, context, cfg)
        missing = next(f for f in findings if f.check == "missing_provenance_columns")
        assert missing.count == 1

    def test_sale_label_on_a_live_listing_is_an_error(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        datasets, context = _datasets(eda_db, cfg)
        cross = datasets["advertisement_cross_section"]
        cross.loc[cross["current_status"] == "active", "sale_label"] = "likely_sold"
        findings = audit(eda_db, datasets, context, cfg)
        finding = next(f for f in findings if f.check == "sale_label_inconsistent_with_status")
        assert finding.count > 0 and finding.severity is Severity.ERROR

    def test_confirmed_sold_label_is_critical(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        """Nothing in this system produces one, so its presence means contamination."""
        datasets, context = _datasets(eda_db, cfg)
        cross = datasets["advertisement_cross_section"]
        cross.loc[cross.index[0], "sale_label"] = "confirmed_sold"
        findings = audit(eda_db, datasets, context, cfg)
        finding = next(f for f in findings if f.check == "confirmed_sale_label_present")
        assert finding.count == 1 and finding.severity is Severity.CRITICAL

    def test_filter_exit_marked_as_removed_is_an_error(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        datasets, context = _datasets(eda_db, cfg)
        cross = datasets["advertisement_cross_section"]
        mask = cross["filter_exit_reason"].notna()
        cross.loc[mask, "current_status"] = "likely_removed"
        findings = audit(eda_db, datasets, context, cfg)
        finding = next(f for f in findings if f.check == "filter_exit_counted_as_disappearance")
        assert finding.count > 0

    def test_absence_from_an_invalid_run_is_critical(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        datasets, context = _datasets(eda_db, cfg)
        panel = datasets["longitudinal_panel"]
        panel["was_seen_trusted"] = panel["was_seen_trusted"].astype("boolean")
        panel.loc[~panel["run_is_valid"].astype(bool), "was_seen_trusted"] = False
        findings = audit(eda_db, datasets, context, cfg)
        finding = next(f for f in findings if f.check == "invalid_run_contributed_absence")
        assert finding.count > 0 and finding.severity is Severity.CRITICAL

    def test_implausible_year_is_flagged(self, eda_db: ReadOnlyDatabase, cfg: EdaConfig) -> None:
        datasets, context = _datasets(eda_db, cfg)
        datasets["advertisement_cross_section"].loc[:, "year_jalali"] = 3000
        findings = audit(eda_db, datasets, context, cfg)
        finding = next(f for f in findings if f.check == "implausible_production_year")
        assert finding.count > 0

    def test_negative_price_and_mileage_are_flagged(
        self, eda_db: ReadOnlyDatabase, cfg: EdaConfig
    ) -> None:
        datasets, context = _datasets(eda_db, cfg)
        cross = datasets["advertisement_cross_section"]
        cross.loc[cross.index[0], "price_toman"] = -1
        cross.loc[cross.index[0], "mileage_km"] = -5
        findings = audit(eda_db, datasets, context, cfg)
        assert next(f for f in findings if f.check == "negative_price").count == 1
        assert next(f for f in findings if f.check == "negative_mileage").count == 1
