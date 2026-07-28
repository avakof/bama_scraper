"""Dataset construction: grain, temporal rules and determinism."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from eda_scenarios import T0

from bama_eda.config import EdaConfig
from bama_eda.database import ReadOnlyDatabase
from bama_eda.dataset_builder import (
    build_cross_section,
    build_panel,
    build_vehicle_dataset,
)
from bama_eda.provenance import Manifest, build_context, select_run


@pytest.fixture
def built(eda_db: ReadOnlyDatabase, cfg: EdaConfig):
    run, _ = select_run(eda_db, "latest-valid")
    context = build_context(eda_db, run)
    manifest = Manifest(context, cfg)
    cross = build_cross_section(eda_db, context, cfg, manifest)
    panel = build_panel(eda_db, context, cfg, manifest)
    vehicles = build_vehicle_dataset(eda_db, cross, context, cfg, manifest)
    return {
        "cross": cross,
        "panel": panel,
        "vehicles": vehicles,
        "context": context,
        "manifest": manifest,
    }


class TestRunSelection:
    def test_latest_valid_run_is_chosen(self, eda_db: ReadOnlyDatabase) -> None:
        run, warnings = select_run(eda_db, "latest-valid")
        assert str(run["status"]) == "valid"
        assert run["finished_at"] is not None
        assert bool(run["comparison_applied"])
        assert warnings == []

    def test_invalid_and_unfinished_runs_are_not_chosen(self, eda_db: ReadOnlyDatabase) -> None:
        run, _ = select_run(eda_db, "latest-valid")
        rejected = eda_db.fetchall(
            "SELECT id, status, finished_at FROM monitoring_runs"
            " WHERE status <> 'valid' OR finished_at IS NULL"
        )
        assert rejected, "the fixture must contain rejectable runs"
        assert int(run["id"]) not in {int(r["id"]) for r in rejected}

    def test_selection_is_reproducible(self, eda_db: ReadOnlyDatabase) -> None:
        first, _ = select_run(eda_db, "latest-valid")
        second, _ = select_run(eda_db, "latest-valid")
        assert int(first["id"]) == int(second["id"])

    def test_explicit_run_id_is_honoured_with_a_warning_when_invalid(
        self, eda_db: ReadOnlyDatabase
    ) -> None:
        invalid = eda_db.fetchone("SELECT id FROM monitoring_runs WHERE status <> 'valid'")
        run, warnings = select_run(eda_db, int(invalid["id"]))
        assert int(run["id"]) == int(invalid["id"])
        assert any("not trustworthy" in w.message for w in warnings)


class TestCrossSectionGrain:
    def test_one_row_per_advertisement(self, built) -> None:
        cross = built["cross"]
        assert len(cross) == cross["platform_ad_id"].nunique()

    def test_provenance_is_stamped_on_every_row(self, built) -> None:
        cross = built["cross"]
        for column in (
            "run_id",
            "scheduled_for",
            "search_configuration_hash",
            "scraper_version",
            "analysis_version",
        ):
            assert column in cross.columns
            assert cross[column].notna().all()
        assert cross["run_id"].nunique() == 1

    def test_a_colliding_source_column_is_renamed_not_overwritten(self, built) -> None:
        # advertisements carries its own search_configuration_hash.
        cross = built["cross"]
        assert "source_search_configuration_hash" in cross.columns


class TestNoFutureSnapshots:
    def test_a_snapshot_after_the_run_is_excluded(self, built) -> None:
        """The fixture plants a snapshot 40 days after every run."""
        cross = built["cross"]
        context = built["context"]
        reference = max(
            t for t in (context.run_finished_at, context.scheduled_for) if t is not None
        )
        scraped = pd.to_datetime(cross["snapshot_scraped_at"], errors="coerce", utc=True)
        assert not (scraped > reference).any()

    def test_the_excluded_snapshot_is_recorded_not_silently_dropped(self, built) -> None:
        reasons = [e["reason"] for e in built["manifest"].exclusions]
        assert any("reference instant" in r for r in reasons)
        counts = {
            e["reason"]: e["count"]
            for e in built["manifest"].exclusions
            if "reference instant" in e["reason"]
        }
        assert sum(counts.values()) >= 1

    def test_latest_qualifying_snapshot_wins_deterministically(self, built) -> None:
        """Advertisement A has two snapshots; the later one must be chosen."""
        cross = built["cross"]
        row = cross[cross["platform_ad_id"] == "A"].iloc[0]
        assert pd.to_datetime(row["snapshot_scraped_at"], utc=True) == pd.Timestamp(
            T0 + timedelta(days=1)
        )
        assert row["snap_price_normalized"] == 1_500_000_000

    def test_an_advertisement_without_a_snapshot_still_appears(self, built) -> None:
        cross = built["cross"]
        row = cross[cross["platform_ad_id"] == "H"]
        assert len(row) == 1
        assert pd.isna(row.iloc[0]["snapshot_scraped_at"])

    def test_price_source_records_which_source_supplied_the_value(self, built) -> None:
        cross = built["cross"]
        sources = set(cross["price_source"].dropna())
        assert sources <= {"detail_snapshot", "search_card"}
        with_snapshot = cross[cross["snapshot_scraped_at"].notna()]
        assert (with_snapshot["price_source"] == "detail_snapshot").all()


class TestPanel:
    def test_grain_is_run_times_advertisement(self, built) -> None:
        panel = built["panel"]
        assert not panel.duplicated(subset=["observation_run_id", "advertisement_id"]).any()

    def test_analysis_run_id_does_not_shadow_the_observation_run(self, built) -> None:
        panel = built["panel"]
        assert panel["observation_run_id"].nunique() > 1
        assert panel["run_id"].nunique() == 1  # the provenance stamp

    def test_absences_from_invalid_runs_are_not_trusted(self, built) -> None:
        panel = built["panel"]
        invalid = panel[~panel["run_is_valid"].astype(bool)]
        assert len(invalid) > 0, "the fixture must contain an invalid run"
        # The raw observation is kept as evidence...
        assert invalid["obs_was_seen"].notna().any()
        # ...but the trusted signal is blank.
        assert invalid["was_seen_trusted"].isna().all()

    def test_only_runs_up_to_the_analysis_run_are_included(self, built) -> None:
        panel = built["panel"]
        assert panel["observation_run_id"].max() <= built["context"].run_id


class TestVehicleDataset:
    def test_a_repost_chain_collapses_to_one_vehicle(self, built) -> None:
        vehicles = built["vehicles"]
        chain = vehicles[vehicles["vehicle_entity_id"] == "veh-1"]
        assert len(chain) == 1
        assert int(chain.iloc[0]["advertisement_count"]) == 2
        assert chain.iloc[0]["was_reposted"]

    def test_vehicle_count_is_below_advertisement_count(self, built) -> None:
        assert len(built["vehicles"]) < len(built["cross"])

    def test_advertisement_ids_are_listed_for_traceability(self, built) -> None:
        vehicles = built["vehicles"]
        chain = vehicles[vehicles["vehicle_entity_id"] == "veh-1"].iloc[0]
        assert set(str(chain["advertisement_ids"]).split(",")) == {"F", "G"}

    def test_vehicle_span_covers_the_whole_chain(self, built) -> None:
        vehicles = built["vehicles"]
        chain = vehicles[vehicles["vehicle_entity_id"] == "veh-1"].iloc[0]
        first = pd.to_datetime(chain["first_seen_at"], utc=True)
        last = pd.to_datetime(chain["last_seen_at"], utc=True)
        # F ran from T0; G was last seen at T0+2d. One life, not two.
        assert (last - first) == pd.Timedelta(days=2)
