"""Feature discovery, classification, grain tracking and cross-dimension comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bama_eda.feature_analysis import (
    BOOLEAN,
    CATEGORICAL,
    NUMERIC,
    TEXT,
    classify_feature,
    compare_across,
    differentiating_features,
    discover_features,
    feature_price_associations,
    is_price_alias,
    is_redundant_with,
    level_distribution,
    profile_all,
    summarise_catalogue,
    to_boolean,
    to_numeric,
    usable_features,
)


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    cities = ["tehran"] * (n - 40) + ["isfahan"] * 30 + ["rare"] * 10
    return pd.DataFrame(
        {
            "platform_ad_id": [f"ad{i}" for i in range(n)],
            "price_toman": rng.integers(1_000_000_000, 3_000_000_000, n).astype(float),
            "log10_price": rng.normal(9.2, 0.1, n),
            "mileage_km": rng.integers(0, 200_000, n).astype(float),
            "deep_city": cities,
            "deep_brand": ["dena" if i % 2 else "peugeot" for i in range(n)],
            # trim-level: identical within a trim
            "spec_sunroof": ["present" if i % 3 else "absent" for i in range(n)],
            "spec_weight": [f"{1100 + (i % 5) * 50} کیلوگرم" for i in range(n)],
            "spec_transmission": ["5 دنده دستی" if i % 2 else "6 دنده اتوماتیک" for i in range(n)],
            "spec_esc": ["present" if i % 4 else "(از پاییز 1402)" for i in range(n)],
            "deep_is_promoted": [bool(i % 5 == 0) for i in range(n)],
            "consecutive_misses": [0] * n,  # a count that happens to be constant-ish
            "always_empty": [None] * n,
            "free_text": [f"unique description number {i}" for i in range(n)],
        }
    )


class TestClassification:
    def test_measure_with_a_unit_is_numeric(self) -> None:
        series = pd.Series(["1165 کیلوگرم", "1258 کیلوگرم", "1400 کیلوگرم"] * 5)
        assert classify_feature(series, "spec_weight") == NUMERIC
        assert to_numeric(series).iloc[0] == 1165

    def test_a_description_starting_with_a_number_is_not_numeric(self) -> None:
        """'5 دنده دستی' is 5-speed MANUAL; reading it as 5 loses the transmission."""
        series = pd.Series(["5 دنده دستی", "6 دنده اتوماتیک"] * 10)
        assert classify_feature(series, "spec_transmission") == CATEGORICAL

    def test_present_absent_vocabulary_is_boolean(self) -> None:
        series = pd.Series(["present", "absent", "present"] * 5)
        assert classify_feature(series, "spec_sunroof") == BOOLEAN
        assert to_boolean(series).tolist()[:3] == [True, False, True]

    def test_persian_presence_vocabulary_is_boolean(self) -> None:
        series = pd.Series(["دارد", "ندارد", "دارد"] * 5)
        assert classify_feature(series, "spec_abs") == BOOLEAN

    def test_a_count_column_stays_numeric_even_when_only_zeros_and_ones(self) -> None:
        """Otherwise the feature's kind flips as the history grows.

        A column holding only 0 and 1 today may hold 5 tomorrow; classifying by the
        observed range would silently change a median into a share between runs.
        """
        series = pd.Series([0, 1, 0, 1] * 10)
        assert classify_feature(series, "consecutive_misses") == NUMERIC

    def test_a_named_flag_with_zero_one_is_boolean(self) -> None:
        series = pd.Series([0, 1, 0, 1] * 10)
        assert classify_feature(series, "deep_is_promoted") == BOOLEAN
        assert classify_feature(series, "deep_api_ok") == BOOLEAN

    def test_high_cardinality_text_is_text(self) -> None:
        series = pd.Series([f"unique sentence {i}" for i in range(200)])
        assert classify_feature(series, "description") == TEXT


class TestDiscovery:
    def test_catalogue_covers_every_analysable_column(self) -> None:
        catalogue = discover_features(_frame())
        assert "spec_sunroof" in set(catalogue["feature"])
        assert "price_toman" in set(catalogue["feature"])

    def test_unusable_columns_are_kept_with_a_reason(self) -> None:
        """Dropping them makes 'not analysed' indistinguishable from 'not present'."""
        catalogue = discover_features(_frame()).set_index("feature")
        assert not catalogue.loc["always_empty", "usable"]
        assert "entirely empty" in catalogue.loc["always_empty", "excluded_because"]
        assert not catalogue.loc["free_text", "usable"]
        assert "free text" in catalogue.loc["free_text", "excluded_because"]

    def test_identifiers_are_not_features(self) -> None:
        catalogue = discover_features(_frame())
        assert "platform_ad_id" not in set(catalogue["feature"])

    def test_grain_is_tracked(self) -> None:
        catalogue = discover_features(_frame()).set_index("feature")
        assert catalogue.loc["spec_sunroof", "grain"] == "trim"
        assert catalogue.loc["deep_brand", "grain"] == "vehicle"
        assert catalogue.loc["price_toman", "grain"] == "listing"

    def test_summary_states_the_grain_caveat(self) -> None:
        summary = summarise_catalogue(discover_features(_frame()))
        assert "not per-vehicle measurements" in summary["grain_note"]
        assert summary["features_usable"] > 0

    def test_usable_features_can_be_filtered_by_kind(self) -> None:
        catalogue = discover_features(_frame())
        numeric = usable_features(catalogue, (NUMERIC,))
        assert "price_toman" in numeric
        assert "deep_city" not in numeric


class TestProfiles:
    def test_each_kind_gets_the_right_summary(self) -> None:
        frame = _frame()
        profiles = profile_all(frame, discover_features(frame)).set_index("feature")
        assert profiles.loc["price_toman", "summary"] == "median [IQR]"
        assert profiles.loc["spec_sunroof", "summary"] == "share present"
        assert profiles.loc["deep_city", "summary"] == "top levels"

    def test_level_distribution_marks_small_levels(self) -> None:
        frame = _frame()
        levels = level_distribution(frame, discover_features(frame), min_group_size=20)
        rare = levels[(levels["feature"] == "deep_city") & (levels["level"] == "rare")]
        assert len(rare) == 1
        assert not bool(rare.iloc[0]["sufficient_sample"])


class TestComparisonAcrossDimensions:
    def test_groups_below_the_threshold_are_excluded_and_counted(self) -> None:
        frame = _frame()
        out = compare_across(frame, discover_features(frame), "deep_city", min_group_size=20)
        assert "rare" not in set(out["group"])
        assert (out["groups_below_min_excluded"] >= 1).all()

    def test_trim_features_carry_the_grain_note(self) -> None:
        frame = _frame()
        out = compare_across(frame, discover_features(frame), "deep_city", min_group_size=20)
        trim_rows = out[out["grain"] == "trim"]
        assert not trim_rows.empty
        assert trim_rows["grain_note"].str.contains("TRIM-LEVEL").all()

    def test_a_dimension_is_not_compared_against_itself(self) -> None:
        frame = _frame()
        out = compare_across(frame, discover_features(frame), "deep_city", min_group_size=20)
        assert "deep_city" not in set(out["feature"])

    def test_ranking_reports_effect_sizes_and_adjusted_p(self) -> None:
        frame = _frame()
        ranked = differentiating_features(
            frame, discover_features(frame), "deep_brand", min_group_size=20
        )
        assert not ranked.empty
        assert "effect_size" in ranked.columns
        assert "p_holm_adjusted" in ranked.columns
        assert ranked["note"].str.contains("EXPLORATORY").any()

    def test_near_tautological_results_are_flagged_not_hidden(self) -> None:
        frame = _frame()
        frame["deep_city_copy"] = frame["deep_city"]
        ranked = differentiating_features(
            frame, discover_features(frame), "deep_city", min_group_size=20
        )
        copy_row = ranked[ranked["feature"] == "deep_city_copy"]
        assert len(copy_row) == 1
        assert bool(copy_row.iloc[0]["near_tautological"])
        assert "restatement" in copy_row.iloc[0]["note"]


class TestRedundancyGuards:
    def test_price_aliases_are_recognised(self) -> None:
        assert is_price_alias("price_toman")
        assert is_price_alias("obs_card_price_normalized")
        assert not is_price_alias("mileage_km")

    def test_a_variable_against_its_own_band_is_redundant(self) -> None:
        assert is_redundant_with("price_toman", "price_band")
        assert is_redundant_with("mileage_km", "mileage_band")
        assert not is_redundant_with("mileage_km", "price_band")

    def test_price_is_not_compared_against_price_bands(self) -> None:
        frame = _frame()
        frame["price_band"] = pd.cut(
            frame["price_toman"],
            bins=[0, 1.5e9, 2e9, 1e12],
            labels=["low", "mid", "high"],
        )
        ranked = differentiating_features(
            frame, discover_features(frame), "price_band", min_group_size=20
        )
        assert "price_toman" not in set(ranked["feature"])
        assert "log10_price" not in set(ranked["feature"])


class TestPriceAssociations:
    def test_restatements_of_price_are_excluded(self) -> None:
        """Correlating price with itself measures nothing."""
        frame = _frame()
        out = feature_price_associations(frame, discover_features(frame), min_n=20)
        assert "log10_price" not in set(out["feature"])
        assert "price_toman" not in set(out["feature"])

    def test_the_right_test_is_used_per_kind(self) -> None:
        frame = _frame()
        out = feature_price_associations(frame, discover_features(frame), min_n=20)
        tests = dict(zip(out["feature"], out["test"], strict=False))
        assert tests.get("mileage_km") == "spearman"
        assert tests.get("spec_sunroof") == "mann_whitney_u"
        assert tests.get("deep_city") == "kruskal_wallis"

    def test_trim_features_carry_the_grain_caveat(self) -> None:
        frame = _frame()
        out = feature_price_associations(frame, discover_features(frame), min_n=20)
        trim_rows = out[out["grain"] == "trim"]
        assert not trim_rows.empty
        assert trim_rows["caveat"].str.contains("TRIM-LEVEL").all()

    def test_effect_sizes_accompany_every_p_value(self) -> None:
        frame = _frame()
        out = feature_price_associations(frame, discover_features(frame), min_n=20)
        assert out["effect_size"].notna().all()
        assert out["effect_size_name"].notna().all()
        assert "p_holm_adjusted" in out.columns

    def test_zero_prices_are_not_treated_as_prices(self) -> None:
        """0 is Bama's 'negotiable'."""
        frame = _frame()
        frame.loc[frame.index[:50], "price_toman"] = 0
        out = feature_price_associations(frame, discover_features(frame), min_n=20)
        assert not out.empty
        assert (out["n"] <= len(frame) - 50).all()
