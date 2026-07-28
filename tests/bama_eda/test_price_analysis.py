"""Price segmentation, suppression thresholds and exploratory tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bama_eda.correlations import adjust_pvalues, correlation_matrix, price_association_summary
from bama_eda.price_analysis import (
    CURRENCY,
    comparable_segments,
    group_difference_tests,
    price_by_segment,
    price_change_analysis,
    price_change_table,
)


def _frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    brands = ["dena"] * (n - 5) + ["rare"] * 5
    return pd.DataFrame(
        {
            "platform_ad_id": [f"ad{i}" for i in range(n)],
            "deep_brand": brands,
            "deep_model": ["plus"] * n,
            "deep_seller_type": ["personal" if i % 2 else "dealer" for i in range(n)],
            "price_toman": rng.integers(1_000_000_000, 3_000_000_000, n).astype(float),
            "mileage_km": rng.integers(0, 200_000, n).astype(float),
            "year_jalali": rng.integers(1397, 1405, n).astype(float),
            "current_status": ["active"] * n,
            "listing_age_days_observed": rng.integers(1, 30, n).astype(float),
        }
    )


class TestPriceBySegment:
    def test_currency_is_toman_and_labelled_as_asking_price(self) -> None:
        out = price_by_segment(_frame())
        assert (out["currency"] == CURRENCY).all()
        assert (out["price_basis"] == "asking_price_not_transaction_price").all()

    def test_small_groups_are_kept_but_labelled(self) -> None:
        out = price_by_segment(_frame(), min_group_size=10)
        rare = out[(out["dimension"] == "brand") & (out["segment_value"] == "rare")]
        assert len(rare) == 1
        assert not bool(rare.iloc[0]["sufficient_sample"])
        assert "insufficient_sample" in rare.iloc[0]["sample_note"]

    def test_robust_statistics_are_present(self) -> None:
        out = price_by_segment(_frame())
        for column in (
            "median_price",
            "p25_price",
            "p75_price",
            "iqr_price",
            "trimmed_mean_price",
            "ci95_median_low",
            "ci95_median_high",
        ):
            assert column in out.columns

    def test_confidence_interval_is_suppressed_for_tiny_groups(self) -> None:
        out = price_by_segment(_frame(), min_group_size=10)
        rare = out[(out["dimension"] == "brand") & (out["segment_value"] == "rare")].iloc[0]
        assert pd.isna(rare["ci95_median_low"])

    def test_grouping_by_a_value_column_does_not_crash(self) -> None:
        """production_year groups BY year_jalali, which is also a value column."""
        out = price_by_segment(_frame())
        assert "production_year" in set(out["dimension"])

    def test_zero_prices_are_excluded_from_price_statistics(self) -> None:
        """Bama encodes 'negotiable' as 0; averaging it invents free cars."""
        frame = _frame()
        frame.loc[frame.index[:10], "price_toman"] = 0
        out = price_by_segment(frame)
        dena = out[(out["dimension"] == "brand") & (out["segment_value"] == "dena")].iloc[0]
        assert dena["minimum_price"] > 0


class TestComparableSegments:
    def test_groups_and_thresholds(self) -> None:
        out = comparable_segments(_frame(), min_group_size=10)
        assert not out.empty
        assert "disappearance_rate" in out.columns
        assert "sufficient_sample" in out.columns


class TestGroupTests:
    def test_reports_effect_size_and_adjusted_p(self) -> None:
        out = group_difference_tests(_frame(), dimensions=("deep_seller_type",))
        row = out.iloc[0]
        assert row["test"] == "mann_whitney_u"
        assert row["effect_size_name"] == "rank_biserial"
        assert "p_holm_adjusted" in out.columns
        assert "EXPLORATORY" in row["note"]

    def test_refuses_when_groups_are_too_small(self) -> None:
        frame = _frame(12)
        out = group_difference_tests(frame, dimensions=("deep_brand",), min_group_size=10)
        assert out.iloc[0]["test"] == "none"
        assert out.iloc[0]["p_value"] is None


class TestMultipleComparisons:
    def test_holm_adjustment_is_monotone_and_conservative(self) -> None:
        raw = [0.01, 0.02, 0.03, 0.04]
        adjusted = adjust_pvalues(raw)
        assert all(a >= r for a, r in zip(adjusted, raw, strict=False))
        assert adjusted == sorted(adjusted)

    def test_none_values_survive(self) -> None:
        assert adjust_pvalues([0.01, None])[1] is None


class TestCorrelations:
    def test_both_coefficients_reported_and_spearman_preferred(self) -> None:
        out = correlation_matrix(_frame(), min_pairs=10)
        assert not out.empty
        row = out.iloc[0]
        assert row["preferred"] == "spearman"
        assert "not causation" in row["note"]

    def test_insufficient_pairs_are_not_computed(self) -> None:
        out = correlation_matrix(_frame(5), min_pairs=30)
        assert (out["strength"] == "not_computed").all()

    def test_price_summary_ranks_by_absolute_rho(self) -> None:
        summary = price_association_summary(correlation_matrix(_frame(), min_pairs=10))
        assert summary["available"]
        rhos = [abs(r["spearman_rho"]) for r in summary["ranked"]]
        assert rhos == sorted(rhos, reverse=True)
        assert "not identify what determines price" in summary["caveat"]


class TestPriceChanges:
    def _panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observation_run_id": [1, 2, 1, 2],
                "advertisement_id": [1, 1, 2, 2],
                "obs_card_price_normalized": [
                    2_000_000_000,
                    1_800_000_000,
                    1_000_000_000,
                    1_000_000_000,
                ],
                "run_is_valid": [True, True, True, True],
            }
        )

    def test_detects_a_reduction(self) -> None:
        result = price_change_analysis(self._panel(), pd.DataFrame())
        assert result["advertisements_with_price_change"] == 1
        assert result["decreases"] == 1
        assert result["median_percentage_decrease"] == 10.0

    def test_does_not_claim_causation(self) -> None:
        result = price_change_analysis(self._panel(), pd.DataFrame())
        assert "does not establish" in result["caveat"]

    def test_invalid_runs_are_excluded(self) -> None:
        panel = self._panel()
        panel.loc[panel["observation_run_id"] == 2, "run_is_valid"] = False
        result = price_change_analysis(panel, pd.DataFrame())
        assert result["change_events"] == 0

    def test_table_shape(self) -> None:
        table = price_change_table(self._panel())
        assert list(table["direction"]) == ["price_decrease"]
        assert table.iloc[0]["currency"] == CURRENCY
