"""Linear, monotone and general dependence — and telling them apart."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bama_eda.correlations import (
    NONLINEARITY_GAP,
    association_matrix,
    benjamini_hochberg,
    classify_relationship,
    correlation_ratio,
    distance_correlation,
    nonlinearity_report,
    pairwise_associations,
    summarise_pairs,
    trimmed_pearson,
)
from bama_eda.feature_analysis import discover_features

SEED = 20260728


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


class TestDistanceCorrelation:
    def test_detects_a_parabola_that_pearson_and_spearman_both_miss(self) -> None:
        """The case that motivates the measure at all."""
        rng = _rng()
        x = rng.normal(size=800)
        y = x**2 + rng.normal(size=800) * 0.2

        pearson = abs(np.corrcoef(x, y)[0, 1])
        from scipy import stats as sp

        spearman = abs(sp.spearmanr(x, y).statistic)
        dcor = distance_correlation(x, y, seed=SEED)

        assert pearson < 0.2, "a parabola has no linear component"
        assert spearman < 0.2, "and no monotone component either"
        assert dcor is not None and dcor > 0.4, "but the variables are plainly dependent"

    def test_is_near_zero_for_independent_variables(self) -> None:
        rng = _rng()
        value = distance_correlation(rng.normal(size=600), rng.normal(size=600), seed=SEED)
        assert value is not None and value < 0.2

    def test_is_near_one_for_an_exact_relationship(self) -> None:
        x = np.linspace(0, 10, 400)
        assert distance_correlation(x, 3 * x + 1, seed=SEED) > 0.95

    def test_subsampling_is_deterministic(self) -> None:
        rng = _rng()
        x, y = rng.normal(size=5000), rng.normal(size=5000)
        first = distance_correlation(x, y, max_n=500, seed=SEED)
        second = distance_correlation(x, y, max_n=500, seed=SEED)
        assert first == second

    def test_too_few_points_returns_none(self) -> None:
        assert distance_correlation(np.arange(5.0), np.arange(5.0)) is None


class TestRelationshipClassification:
    def test_linear(self) -> None:
        assert classify_relationship(0.9, 0.9, 0.9) == "approximately_linear"

    def test_monotone_but_curved(self) -> None:
        assert classify_relationship(0.4, 0.9, 0.9) == "monotone_nonlinear"

    def test_non_monotone(self) -> None:
        """Both correlations near zero, dependence high."""
        assert classify_relationship(0.05, 0.05, 0.6) == "non_monotone_dependence"

    def test_recovery_under_trimming_gets_its_own_label(self) -> None:
        """Named for what was measured, not for a cause it cannot establish.

        Both a contaminated minority and tail curvature make Pearson recover once
        the outer deciles are clipped, so the label says only that.
        """
        assert classify_relationship(0.02, 0.95, 0.3, robust=0.94) == "linear_within_trimmed_range"
        assert classify_relationship(0.02, 0.95, 0.3, robust=0.03) == "monotone_nonlinear"

    def test_nothing(self) -> None:
        assert classify_relationship(0.01, 0.02, 0.03) == "no_detected_association"

    def test_missing_inputs(self) -> None:
        assert classify_relationship(None, 0.5, 0.5) == "not_computed"


class TestTrimmedPearson:
    def test_recovers_a_linear_relationship_broken_by_a_minority(self) -> None:
        """Mirrors the live data: ~1.6% of years published on a different calendar."""
        rng = _rng()
        n = 1000
        x = rng.uniform(1397, 1405, n)
        y = x + 621
        contaminated = rng.choice(n, size=16, replace=False)
        x[contaminated] += 621  # same year, other calendar

        raw = float(np.corrcoef(x, y)[0, 1])
        robust = trimmed_pearson(x, y, lower=10.0, upper=90.0)
        assert abs(raw) < 0.5, "the contaminated minority destroys the estimate"
        assert robust is not None and abs(robust) > abs(raw) + NONLINEARITY_GAP

    def test_a_gentle_trim_does_not_catch_a_subpopulation(self) -> None:
        """1/99 is the wrong tool for a 1.6% contaminated group; 10/90 is right."""
        rng = _rng()
        n = 1000
        x = rng.uniform(1397, 1405, n)
        y = x + 621
        x[rng.choice(n, size=16, replace=False)] += 621
        gentle = trimmed_pearson(x, y, lower=1.0, upper=99.0)
        strong = trimmed_pearson(x, y, lower=10.0, upper=90.0)
        assert abs(gentle) < abs(strong)


class TestCorrelationRatio:
    def test_zero_when_groups_do_not_differ(self) -> None:
        rng = _rng()
        frame = pd.DataFrame({"v": rng.normal(size=300), "g": ["a", "b", "c"] * 100})
        eta = correlation_ratio(frame["v"], frame["g"])
        assert eta is not None and eta < 0.2

    def test_high_when_groups_separate_the_values(self) -> None:
        values = pd.Series([1.0] * 100 + [10.0] * 100)
        groups = pd.Series(["low"] * 100 + ["high"] * 100)
        eta = correlation_ratio(values, groups)
        assert eta is not None and eta > 0.95

    def test_needs_two_groups(self) -> None:
        assert correlation_ratio(pd.Series([1.0, 2.0, 3.0]), pd.Series(["a"] * 3)) is None


class TestMultipleComparisons:
    def test_fdr_is_monotone_and_no_smaller_than_the_raw_value(self) -> None:
        raw = [0.001, 0.01, 0.02, 0.5]
        adjusted = benjamini_hochberg(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw, strict=False))
        assert adjusted == sorted(adjusted)

    def test_fdr_is_less_conservative_than_holm(self) -> None:
        """Which is why it is used for the pairwise family."""
        from bama_eda.correlations import adjust_pvalues

        raw = [0.001 * i for i in range(1, 200)]
        assert sum(p < 0.05 for p in benjamini_hochberg(raw)) >= sum(
            p < 0.05 for p in adjust_pvalues(raw)
        )
        # Holm controls the chance of ANY false positive; BH controls their share,
        # so BH never rejects fewer.
        assert all(
            bh <= holm + 1e-12
            for bh, holm in zip(benjamini_hochberg(raw), adjust_pvalues(raw), strict=False)
        )

    def test_none_values_survive(self) -> None:
        assert benjamini_hochberg([0.01, None])[1] is None


class TestPairwiseAssociations:
    def _frame(self, n: int = 400) -> pd.DataFrame:
        rng = _rng()
        x = rng.normal(size=n)
        return pd.DataFrame(
            {
                "platform_ad_id": [f"a{i}" for i in range(n)],
                "linear_x": x,
                "linear_y": 2 * x + rng.normal(size=n) * 0.1,
                "curved_y": np.exp(x),  # monotone, not linear
                "parabola_y": x**2 + rng.normal(size=n) * 0.1,  # not monotone
                "noise": rng.normal(size=n),
                "deep_group": rng.choice(["a", "b", "c"], size=n),
                "deep_other": rng.choice(["x", "y"], size=n),
            }
        )

    def test_each_pair_type_gets_its_own_measure(self) -> None:
        frame = self._frame()
        pairs = pairwise_associations(frame, discover_features(frame), seed=SEED)
        measures = dict(zip(pairs["pair_type"], pairs["general_measure"], strict=False))
        assert measures["numeric_numeric"] == "distance_correlation"
        assert measures["numeric_categorical"] == "correlation_ratio_eta"
        assert measures["categorical_categorical"] == "cramers_v"

    def test_linear_and_monotone_are_blank_where_they_do_not_apply(self) -> None:
        """A grouping has no order, so 'linear in it' is not a question."""
        frame = self._frame()
        pairs = pairwise_associations(frame, discover_features(frame), seed=SEED)
        mixed = pairs[pairs["pair_type"] != "numeric_numeric"]
        assert mixed["linear_pearson_r"].isna().all()
        assert mixed["monotone_spearman_rho"].isna().all()

    def test_the_three_shapes_are_identified(self) -> None:
        frame = self._frame()
        pairs = pairwise_associations(frame, discover_features(frame), seed=SEED)
        shapes = {
            frozenset((row.feature_a, row.feature_b)): row.relationship
            for row in pairs.itertuples()
        }
        assert shapes[frozenset(("linear_x", "linear_y"))] == "approximately_linear"
        # exp(x) is curved, but the curvature lives in the tail: clipping the outer
        # deciles makes it near-linear, which is exactly what the label reports.
        assert shapes[frozenset(("linear_x", "curved_y"))] in (
            "monotone_nonlinear",
            "linear_within_trimmed_range",
        )
        assert shapes[frozenset(("linear_x", "parabola_y"))] == "non_monotone_dependence"
        assert shapes[frozenset(("linear_x", "noise"))] == "no_detected_association"

    def test_pearson_alone_would_call_the_parabola_unrelated(self) -> None:
        """The headline reason this analysis reports three measures, not one."""
        frame = self._frame()
        pairs = pairwise_associations(frame, discover_features(frame), seed=SEED)
        row = pairs[(pairs["feature_a"] == "linear_x") & (pairs["feature_b"] == "parabola_y")].iloc[
            0
        ]
        assert abs(row["linear_pearson_r"]) < 0.2
        assert row["general_value"] > 0.4

    def test_a_variable_is_not_paired_with_its_own_band(self) -> None:
        frame = self._frame()
        frame["price_toman"] = np.abs(frame["linear_x"]) * 1e9 + 1e9
        frame["price_band"] = pd.cut(frame["price_toman"], bins=3, labels=["l", "m", "h"])
        pairs = pairwise_associations(frame, discover_features(frame), seed=SEED)
        combined = {
            frozenset((a, b)) for a, b in zip(pairs["feature_a"], pairs["feature_b"], strict=False)
        }
        assert frozenset(("price_toman", "price_band")) not in combined

    def test_summary_explains_the_measures(self) -> None:
        frame = self._frame()
        summary = summarise_pairs(pairwise_associations(frame, discover_features(frame), seed=SEED))
        assert "LINEAR association only" in summary["interpretation"]
        assert "Benjamini-Hochberg" in summary["multiple_comparisons"]
        assert summary["pairs_tested"] > 0


class TestNonlinearityReport:
    def _pairs(self) -> pd.DataFrame:
        rng = _rng()
        n = 400
        x = rng.normal(size=n)
        frame = pd.DataFrame(
            {
                "platform_ad_id": [f"a{i}" for i in range(n)],
                "linear_x": x,
                "curved_y": np.exp(x),
                "parabola_y": x**2 + rng.normal(size=n) * 0.1,
                "linear_y": 2 * x,
            }
        )
        return pairwise_associations(frame, discover_features(frame), seed=SEED)

    def test_lists_only_relationships_pearson_would_misdescribe(self) -> None:
        report = nonlinearity_report(self._pairs())
        assert not report.empty
        assert set(report["relationship"]) <= {
            "monotone_nonlinear",
            "non_monotone_dependence",
            "linear_within_trimmed_range",
        }

    def test_each_row_says_what_to_do_about_it(self) -> None:
        report = nonlinearity_report(self._pairs())
        assert report["interpretation"].str.len().gt(20).all()

    def test_alias_duplicates_are_collapsed(self) -> None:
        """power / power_hp / power_text are one quantity; three identical rows are noise."""
        pairs = self._pairs()
        duplicated = pd.concat([pairs, pairs], ignore_index=True)
        report = nonlinearity_report(duplicated)
        assert int(report["alias_duplicates_collapsed"].iloc[0]) > 0
        signature = report[["linear_pearson_r", "monotone_spearman_rho", "general_value", "n"]]
        assert not signature.duplicated().any()


class TestAssociationMatrix:
    def test_is_symmetric_with_a_unit_diagonal(self) -> None:
        rng = _rng()
        n = 300
        x = rng.normal(size=n)
        frame = pd.DataFrame(
            {
                "platform_ad_id": [f"a{i}" for i in range(n)],
                "a": x,
                "b": 2 * x + rng.normal(size=n) * 0.1,
                "c": rng.normal(size=n),
            }
        )
        pairs = pairwise_associations(frame, discover_features(frame), seed=SEED)
        matrix = association_matrix(pairs, measure="monotone_spearman_rho")
        assert not matrix.empty
        assert (np.diag(matrix.to_numpy()) == 1.0).all()
        np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)

    def test_empty_input_yields_an_empty_matrix(self) -> None:
        assert association_matrix(pd.DataFrame()).empty
