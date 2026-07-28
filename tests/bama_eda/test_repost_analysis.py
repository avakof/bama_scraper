"""Vehicle grain: a relisted car is one vehicle, not two sales."""

from __future__ import annotations

import pandas as pd

from bama_eda.repost_analysis import (
    mileage_consistency_check,
    repost_transitions,
    vehicle_grain_summary,
)
from bama_eda.status_analysis import filter_exit_analysis, filter_exit_table


def _vehicles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vehicle_entity_id": "veh-1",
                "advertisement_count": 2,
                "advertisement_ids": "F,G",
                "first_advertisement_id": "F",
                "latest_advertisement_id": "G",
                "latest_status": "active",
            },
            {
                "vehicle_entity_id": "veh-2",
                "advertisement_count": 1,
                "advertisement_ids": "H",
                "first_advertisement_id": "H",
                "latest_advertisement_id": "H",
                "latest_status": "active",
            },
        ]
    )


def _cross() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "platform_ad_id": "F",
                "price_toman": 2_000_000_000,
                "mileage_km": 50_000,
                "deep_seller_type": "personal",
                "deep_description_scrubbed": "دنا پلاس تمیز",
                "first_missing_at": "2026-06-02T11:00:00+00:00",
                "last_seen_at": "2026-06-01T11:00:00+00:00",
                "first_seen_at": "2026-06-01T11:00:00+00:00",
            },
            {
                "platform_ad_id": "G",
                "price_toman": 1_900_000_000,
                "mileage_km": 52_000,
                "deep_seller_type": "personal",
                "deep_description_scrubbed": "دنا پلاس تمیز",
                "first_missing_at": None,
                "last_seen_at": "2026-06-05T11:00:00+00:00",
                "first_seen_at": "2026-06-04T11:00:00+00:00",
            },
            {
                "platform_ad_id": "H",
                "price_toman": 1_500_000_000,
                "mileage_km": 10_000,
                "deep_seller_type": "dealer",
                "deep_description_scrubbed": None,
                "first_missing_at": None,
                "last_seen_at": "2026-06-05T11:00:00+00:00",
                "first_seen_at": "2026-06-01T11:00:00+00:00",
            },
        ]
    )


class TestVehicleGrain:
    def test_chain_counted_once(self) -> None:
        out = vehicle_grain_summary(_vehicles())
        assert out["vehicle_entities"] == 2
        assert out["advertisements_covered"] == 3
        assert out["entities_with_multiple_advertisements"] == 1

    def test_repost_is_explained_as_a_covariate_not_an_outcome(self) -> None:
        out = vehicle_grain_summary(_vehicles())
        assert "not an outcome" in out["interpretation"]

    def test_chain_size_distribution(self) -> None:
        out = vehicle_grain_summary(_vehicles())
        assert out["chain_size_distribution"] == {"1": 1, "2": 1}
        assert out["max_chain_size"] == 2


class TestRepostTransitions:
    def test_price_and_mileage_movement_between_listings(self) -> None:
        out = repost_transitions(_cross(), _vehicles())
        assert len(out) == 1
        row = out.iloc[0]
        assert row["price_delta"] == -100_000_000
        assert row["mileage_delta"] == 2_000
        assert bool(row["seller_consistent"]) is True

    def test_description_similarity_is_measured(self) -> None:
        out = repost_transitions(_cross(), _vehicles())
        assert out.iloc[0]["description_similarity"] == 1.0

    def test_gap_between_listings(self) -> None:
        out = repost_transitions(_cross(), _vehicles())
        assert out.iloc[0]["days_between_listings"] == 2.0

    def test_note_forbids_reading_a_repost_as_a_sale(self) -> None:
        out = repost_transitions(_cross(), _vehicles())
        assert "NOT a sale" in out.iloc[0]["note"]

    def test_no_chains_yields_an_empty_typed_frame(self) -> None:
        single = _vehicles().head(0)
        out = repost_transitions(_cross(), single)
        assert out.empty
        assert "vehicle_entity_id" in out.columns


class TestMileageConsistency:
    def test_a_decrease_signals_a_false_match(self) -> None:
        chains = pd.DataFrame([{"mileage_delta": -50_000}])
        check = mileage_consistency_check(chains)
        assert not check["ok"] and check["decreases"] == 1

    def test_an_increase_is_fine(self) -> None:
        chains = pd.DataFrame([{"mileage_delta": 2_000}])
        assert mileage_consistency_check(chains)["ok"]


class TestFilterExits:
    def _with_exit(self) -> pd.DataFrame:
        frame = _cross()
        frame["filter_exit_reason"] = [None, None, "price_below_filter"]
        frame["current_status"] = ["active", "active", "active_outside_filter"]
        frame["canonical_url"] = ["u1", "u2", "u3"]
        return frame

    def test_counted_and_explained(self) -> None:
        out = filter_exit_analysis(self._with_exit())
        assert out["filter_exits"] == 1
        assert out["reason_counts"] == {"price_below_filter": 1}
        assert out["excluded_from_disappearance"] is True
        assert "CENSORING event" in out["interpretation"]

    def test_no_exits_is_reported_as_a_window_limit(self) -> None:
        frame = _cross()
        frame["filter_exit_reason"] = None
        out = filter_exit_analysis(frame)
        assert out["filter_exits"] == 0
        assert "not evidence that filter exits do not occur" in out["note"]

    def test_table_labels_every_row(self) -> None:
        table = filter_exit_table(self._with_exit())
        assert len(table) == 1
        assert "NOT a disappearance" in table.iloc[0]["interpretation"]
