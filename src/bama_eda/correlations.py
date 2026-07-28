"""Association measures, with the caveats attached to the numbers themselves.

Pearson and Spearman are both computed, and Spearman is the one narrated: asking
prices are strongly right-skewed and several relationships here (price against
mileage, price against age) are monotone but not linear, which is precisely the
case where Pearson understates the association and Spearman does not.

Nothing in this module establishes causation, and the returned records say so.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Numeric pairs worth testing, in report order.
DEFAULT_NUMERIC = (
    "price_toman",
    "log10_price",
    "mileage_km",
    "year_jalali",
    "description_length",
    "media_count",
    "listing_age_days_observed",
    "sale_confidence",
    "deep_engine_volume_l",
    "deep_power_hp",
)


def correlation_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = DEFAULT_NUMERIC,
    *,
    min_pairs: int = 30,
    weak_threshold: float = 0.2,
) -> pd.DataFrame:
    """Pairwise Pearson and Spearman with sample sizes and p-values."""
    present = [c for c in columns if c in frame.columns]
    records: list[dict[str, Any]] = []

    for left, right in combinations(present, 2):
        a = pd.to_numeric(frame[left], errors="coerce")
        b = pd.to_numeric(frame[right], errors="coerce")
        mask = a.notna() & b.notna()
        n = int(mask.sum())
        if n < min_pairs or a[mask].nunique() < 2 or b[mask].nunique() < 2:
            records.append(
                {
                    "variable_a": left,
                    "variable_b": right,
                    "n": n,
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_rho": None,
                    "spearman_p": None,
                    "preferred": "spearman",
                    "strength": "not_computed",
                    "note": f"fewer than {min_pairs} complete pairs, or no variation",
                }
            )
            continue

        pearson = stats.pearsonr(a[mask], b[mask])
        spearman = stats.spearmanr(a[mask], b[mask])
        rho = float(spearman.statistic)
        records.append(
            {
                "variable_a": left,
                "variable_b": right,
                "n": n,
                "pearson_r": round(float(pearson.statistic), 4),
                "pearson_p": _p(pearson.pvalue),
                "spearman_rho": round(rho, 4),
                "spearman_p": _p(spearman.pvalue),
                "preferred": "spearman",
                "strength": _strength(rho, weak_threshold),
                "note": (
                    "Spearman is preferred: prices are strongly skewed and several of "
                    "these relationships are monotone but not linear. Association, "
                    "not causation."
                ),
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        # to_numeric first: when every pair was too small to compute, the column is
        # all-None and object-typed, and .abs() raises rather than sorting.
        magnitude = pd.to_numeric(out["spearman_rho"], errors="coerce").abs()
        out = out.iloc[magnitude.sort_values(ascending=False, na_position="last").index]
        out = out.reset_index(drop=True)
    return out


def _p(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if number >= 1e-6 else 0.0


def _strength(rho: float, weak: float) -> str:
    magnitude = abs(rho)
    if magnitude < weak:
        return "negligible"
    if magnitude < 0.4:
        return "weak"
    if magnitude < 0.6:
        return "moderate"
    if magnitude < 0.8:
        return "strong"
    return "very_strong"


def cramers_v(frame: pd.DataFrame, left: str, right: str, *, min_n: int = 30) -> dict[str, Any]:
    """Association between two categorical variables, with a bias correction.

    The bias-corrected form matters here because several categoricals (city,
    model) have many levels and small cells, where the uncorrected statistic is
    inflated towards 1.
    """
    if left not in frame.columns or right not in frame.columns:
        return {
            "variable_a": left,
            "variable_b": right,
            "n": 0,
            "cramers_v": None,
            "note": "column absent",
        }
    block = frame[[left, right]].dropna()
    if len(block) < min_n:
        return {
            "variable_a": left,
            "variable_b": right,
            "n": int(len(block)),
            "cramers_v": None,
            "note": f"fewer than {min_n} complete pairs",
        }

    table = pd.crosstab(block[left].astype(str), block[right].astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "variable_a": left,
            "variable_b": right,
            "n": int(len(block)),
            "cramers_v": None,
            "note": "a variable has a single level",
        }

    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    value = float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else None

    small_cells = int((expected < 5).sum())
    return {
        "variable_a": left,
        "variable_b": right,
        "n": int(n),
        "levels_a": int(r),
        "levels_b": int(k),
        "chi2": round(float(chi2), 4),
        "p_value": _p(p_value),
        "dof": int(dof),
        "cramers_v": round(value, 4) if value is not None else None,
        "cells_with_expected_below_5": small_cells,
        "note": (
            "bias-corrected Cramer's V. "
            + (
                f"{small_cells} cell(s) have an expected count below 5, so the chi-square "
                "p-value is unreliable; treat as exploratory."
                if small_cells
                else "Exploratory: association, not causation."
            )
        ),
    }


DEFAULT_CATEGORICAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("deep_brand", "deep_seller_type"),
    ("deep_brand", "deep_fuel_type"),
    ("deep_brand", "deep_transmission"),
    ("deep_seller_type", "current_status"),
    ("deep_province", "deep_seller_type"),
    ("deep_condition_new_used", "deep_seller_type"),
)


def categorical_associations(
    frame: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_CATEGORICAL_PAIRS,
    *,
    min_n: int = 30,
) -> pd.DataFrame:
    records = [cramers_v(frame, a, b, min_n=min_n) for a, b in pairs]
    return pd.DataFrame(records)


def adjust_pvalues(p_values: list[float | None], *, method: str = "holm") -> list[float | None]:
    """Holm-Bonferroni correction.

    Applied because this module runs dozens of tests over one dataset; without an
    adjustment, roughly one in twenty "significant" results is significant by
    construction rather than by evidence.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return list(p_values)
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    adjusted: list[float | None] = list(p_values)
    running = 0.0
    for rank, (index, p) in enumerate(indexed):
        value = (m - rank) * p if method == "holm" else min(1.0, p * m)
        running = max(running, min(1.0, value))
        adjusted[index] = round(running, 6)
    return adjusted


def price_association_summary(
    correlations: pd.DataFrame, *, weak_threshold: float = 0.2
) -> dict[str, Any]:
    """Which variables move with price, ranked by |Spearman rho|."""
    if correlations.empty:
        return {"available": False, "note": "no correlations computed"}
    price_rows = correlations[
        (correlations["variable_a"] == "price_toman")
        | (correlations["variable_b"] == "price_toman")
    ].copy()
    if price_rows.empty:
        return {"available": False, "note": "price was not among the correlated variables"}

    price_rows["other"] = np.where(
        price_rows["variable_a"] == "price_toman",
        price_rows["variable_b"],
        price_rows["variable_a"],
    )
    price_rows = price_rows.dropna(subset=["spearman_rho"])
    price_rows["adjusted_p"] = adjust_pvalues(price_rows["spearman_p"].tolist())
    ranked = price_rows.reindex(price_rows["spearman_rho"].abs().sort_values(ascending=False).index)
    return {
        "available": True,
        "ranked": [
            {
                "variable": row["other"],
                "spearman_rho": row["spearman_rho"],
                "pearson_r": row["pearson_r"],
                "n": row["n"],
                "p_raw": row["spearman_p"],
                "p_holm_adjusted": row["adjusted_p"],
                "strength": row["strength"],
            }
            for _, row in ranked.iterrows()
        ],
        "strongest": ranked.iloc[0]["other"] if len(ranked) else None,
        "caveat": (
            "These are associations in one cross-section of asking prices. They do not "
            "identify what determines price, and they say nothing about transaction "
            "prices, which this dataset does not contain."
        ),
    }


# ---------------------------------------------------------------------------
# Pairwise association across every feature, of every type
# ---------------------------------------------------------------------------
#
# Three different questions get three different measures, and conflating them is
# how "these variables are unrelated" gets said about a perfect parabola:
#
#   Pearson r            linear association only
#   Spearman rho         monotone association, linear or not
#   distance correlation ANY dependence, including non-monotone; 0 iff independent
#
# For mixed pairs there is no linear/monotone distinction to draw, so a single
# general measure is used: the correlation ratio (eta) for numeric-versus-
# categorical, bias-corrected Cramer's V for categorical-versus-categorical.


def correlation_ratio(values: pd.Series, groups: pd.Series) -> float | None:
    """Correlation ratio (eta) between a numeric variable and a grouping.

    The share of the numeric variable's variance explained by the grouping,
    square-rooted so it sits on the same 0-1 scale as a correlation. Unlike
    Pearson it needs no ordering of the groups, which is the point: "city" has no
    natural order.
    """
    frame = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": groups}).dropna()
    if len(frame) < 3 or frame["g"].nunique() < 2:
        return None
    grand_mean = frame["v"].mean()
    total = float(((frame["v"] - grand_mean) ** 2).sum())
    if total <= 0:
        return None
    between = float(
        sum(
            len(block) * (block["v"].mean() - grand_mean) ** 2
            for _, block in frame.groupby("g", observed=True)
        )
    )
    return float(np.sqrt(max(0.0, min(1.0, between / total))))


def distance_correlation(
    x: np.ndarray, y: np.ndarray, *, max_n: int = 1000, seed: int = 0
) -> float | None:
    """Distance correlation: zero if and only if the variables are independent.

    This is the measure that catches what Pearson and Spearman both miss. A
    parabola has Pearson near zero and Spearman near zero, and distance
    correlation around 0.5 -- because the variables are plainly dependent, just
    not monotonically.

    O(n^2) in memory and time, so it runs on a deterministic subsample; the
    subsample is drawn with the configured seed so a rerun reproduces the value.
    """
    if len(x) != len(y) or len(x) < 10:
        return None
    if len(x) > max_n:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(x), size=max_n, replace=False)
        x, y = x[index], y[index]

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_centred = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    b_centred = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    covariance = float((a_centred * b_centred).mean())
    variance_x = float((a_centred * a_centred).mean())
    variance_y = float((b_centred * b_centred).mean())
    if variance_x <= 0 or variance_y <= 0:
        return None
    return float(np.sqrt(max(covariance, 0.0)) / np.sqrt(np.sqrt(variance_x * variance_y)))


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR.

    Holm is used elsewhere in this package, but over fifteen thousand pairs it is
    so conservative that nothing survives and the correction stops being
    informative. FDR is the appropriate control at this scale: it bounds the
    expected share of false discoveries rather than the chance of any.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return list(p_values)
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    adjusted: list[float | None] = list(p_values)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index, p = indexed[rank]
        running = min(running, p * m / (rank + 1))
        adjusted[index] = round(min(1.0, running), 8)
    return adjusted


#: |Spearman| minus |Pearson| above this means the relationship is monotone but
#: meaningfully non-linear.
NONLINEARITY_GAP = 0.1
#: Distance correlation this far above |Spearman| means dependence that is not
#: monotone at all -- the parabola case.
NON_MONOTONE_GAP = 0.15


def trimmed_pearson(
    x: np.ndarray, y: np.ndarray, *, lower: float = 1.0, upper: float = 99.0
) -> float | None:
    """Pearson after clipping both variables to the given percentiles.

    A gap between Pearson and Spearman has causes that demand opposite responses:

    * **curvature** -- the relationship really is not linear, so a linear model is
      the wrong model;
    * **a contaminated minority** -- the relationship IS linear within the bulk of
      the data, and a small subpopulation on a different scale is destroying the
      estimate. Here the data needs cleaning, not a different model.

    Clipping distinguishes them: if Pearson recovers once the extremes are pulled
    in, the gap was leverage, not shape.

    Two trims are used by the caller, because they catch different things. A 1/99
    clip catches a handful of stray points. It does **not** catch a contaminated
    *subpopulation*: the live data contains ~1.6% of listings whose year is
    published on the Gregorian calendar while the rest are Jalali, which puts two
    parallel clusters 621 apart in the scatter and leaves 1/99 untouched. A 10/90
    clip catches that.
    """
    if len(x) < 20:
        return None
    x_clipped = np.clip(x, *np.percentile(x, [lower, upper]))
    y_clipped = np.clip(y, *np.percentile(y, [lower, upper]))
    if np.std(x_clipped) == 0 or np.std(y_clipped) == 0:
        return None
    return float(np.corrcoef(x_clipped, y_clipped)[0, 1])


def classify_relationship(
    pearson: float | None,
    spearman: float | None,
    dcor: float | None,
    *,
    trimmed: float | None = None,
    robust: float | None = None,
    weak: float = 0.2,
) -> str:
    """Name the shape of a numeric-numeric relationship.

    ``linear_within_trimmed_range`` is separated from ``monotone_nonlinear``
    because the two suggest different next steps. It is named for what was
    *measured* -- Pearson recovers once the outer deciles are clipped -- and not
    for a cause, because two quite different situations produce it:

    * a contaminated minority on a different scale (the live data holds ~1.6% of
      years published on the Gregorian calendar among Jalali ones), where the fix
      is to clean the data;
    * genuine curvature concentrated in the tails, as with an exponential, where
      the relationship really is close to linear across the bulk.

    Clipping cannot tell those apart, so the label does not pretend to. Look at
    the scatter panel.
    """
    if pearson is None or spearman is None:
        return "not_computed"
    linear, monotone = abs(pearson), abs(spearman)
    general = dcor if dcor is not None else monotone

    if general is not None and general - monotone > NON_MONOTONE_GAP:
        return "non_monotone_dependence"
    if monotone - linear > NONLINEARITY_GAP:
        # Pearson recovering once the tails are clipped means the relationship was
        # linear all along and a minority of points was breaking the estimate.
        recovered = max((abs(v) for v in (trimmed, robust) if v is not None), default=linear)
        if recovered - linear > NONLINEARITY_GAP:
            return "linear_within_trimmed_range"
        return "monotone_nonlinear"
    if monotone < weak and (general is None or general < weak):
        return "no_detected_association"
    if linear >= weak:
        return "approximately_linear"
    return "weak_or_unclear"


def pairwise_associations(
    frame: pd.DataFrame,
    catalogue: pd.DataFrame,
    *,
    min_pairs: int = 30,
    max_levels: int = 30,
    compute_distance_correlation: bool = True,
    distance_correlation_sample: int = 1000,
    seed: int = 0,
    weak_threshold: float = 0.2,
) -> pd.DataFrame:
    """Every feature against every other, with the right measure per pair type.

    Returns one row per pair carrying the linear measure, the monotone measure,
    a general dependence measure, and a named relationship shape. Pairs whose
    types admit no linear/monotone distinction leave those columns empty rather
    than filling them with a number that would be read as a correlation.
    """
    from .feature_analysis import (
        BOOLEAN,
        CATEGORICAL,
        NUMERIC,
        is_redundant_with,
        to_boolean,
        to_numeric,
    )

    usable = catalogue[catalogue["usable"]]
    numeric = [f for f in usable[usable["kind"] == NUMERIC]["feature"] if f in frame.columns]
    grouped = [
        f
        for f in usable[usable["kind"].isin([CATEGORICAL, BOOLEAN])]["feature"]
        if f in frame.columns
    ]
    meta = usable.set_index("feature")

    # Materialise once: coercing inside the pair loop would repeat the same work
    # thousands of times.
    numeric_frame = pd.DataFrame({c: to_numeric(frame[c]) for c in numeric})
    group_frame = pd.DataFrame(
        {
            c: (
                to_boolean(frame[c]).map({True: "present", False: "absent"})
                if meta.loc[c, "kind"] == BOOLEAN
                else frame[c].astype("string")
            )
            for c in grouped
        }
    )
    # A categorical with hundreds of levels produces a near-1 Cramer's V by
    # construction; those pairs are skipped and counted rather than reported.
    level_counts = {c: int(group_frame[c].nunique(dropna=True)) for c in grouped}
    wide = [c for c, n in level_counts.items() if n > max_levels]
    group_frame = group_frame.drop(columns=wide)

    records: list[dict[str, Any]] = []

    # -- numeric x numeric: linear, monotone and general all defined ------
    if len(numeric_frame.columns) >= 2:
        pearson = numeric_frame.corr(method="pearson", min_periods=min_pairs)
        spearman = numeric_frame.corr(method="spearman", min_periods=min_pairs)
        columns = list(numeric_frame.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                if is_redundant_with(left, right) or is_redundant_with(right, left):
                    continue
                mask = numeric_frame[left].notna() & numeric_frame[right].notna()
                n = int(mask.sum())
                if n < min_pairs:
                    continue
                r = pearson.loc[left, right]
                rho = spearman.loc[left, right]
                r = None if pd.isna(r) else float(r)
                rho = None if pd.isna(rho) else float(rho)
                dcor = None
                if compute_distance_correlation and r is not None:
                    dcor = distance_correlation(
                        numeric_frame.loc[mask, left].to_numpy(dtype=float),
                        numeric_frame.loc[mask, right].to_numpy(dtype=float),
                        max_n=distance_correlation_sample,
                        seed=seed,
                    )
                trimmed = robust = None
                if r is not None and n >= 20:
                    left_values = numeric_frame.loc[mask, left].to_numpy(dtype=float)
                    right_values = numeric_frame.loc[mask, right].to_numpy(dtype=float)
                    trimmed = trimmed_pearson(left_values, right_values)
                    robust = trimmed_pearson(left_values, right_values, lower=10.0, upper=90.0)
                p_value = None
                if rho is not None and n > 2:
                    p_value = float(
                        stats.spearmanr(
                            numeric_frame.loc[mask, left], numeric_frame.loc[mask, right]
                        ).pvalue
                    )
                records.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "label_a": meta.loc[left, "label"],
                        "label_b": meta.loc[right, "label"],
                        "grain_a": meta.loc[left, "grain"],
                        "grain_b": meta.loc[right, "grain"],
                        "pair_type": "numeric_numeric",
                        "n": n,
                        "linear_pearson_r": None if r is None else round(r, 4),
                        "linear_pearson_trimmed": (None if trimmed is None else round(trimmed, 4)),
                        "linear_pearson_robust_10_90": (
                            None if robust is None else round(robust, 4)
                        ),
                        "monotone_spearman_rho": None if rho is None else round(rho, 4),
                        "general_measure": "distance_correlation",
                        "general_value": None if dcor is None else round(dcor, 4),
                        "nonlinearity_gap": (
                            round(abs(rho) - abs(r), 4)
                            if r is not None and rho is not None
                            else None
                        ),
                        "dependence_beyond_monotone": (
                            round(dcor - abs(rho), 4)
                            if dcor is not None and rho is not None
                            else None
                        ),
                        "relationship": classify_relationship(
                            r, rho, dcor, trimmed=trimmed, robust=robust, weak=weak_threshold
                        ),
                        "p_value": p_value,
                    }
                )

    # -- numeric x categorical: correlation ratio -------------------------
    for column in numeric_frame.columns:
        for group in group_frame.columns:
            if is_redundant_with(column, group) or is_redundant_with(group, column):
                continue
            mask = numeric_frame[column].notna() & group_frame[group].notna()
            n = int(mask.sum())
            if n < min_pairs:
                continue
            eta = correlation_ratio(numeric_frame.loc[mask, column], group_frame.loc[mask, group])
            if eta is None:
                continue
            blocks = [
                block.to_numpy()
                for _, block in numeric_frame.loc[mask, column].groupby(
                    group_frame.loc[mask, group], observed=True
                )
                if len(block) >= 3
            ]
            p_value = None
            if len(blocks) >= 2:
                try:
                    p_value = float(stats.kruskal(*blocks).pvalue)
                except ValueError:
                    p_value = None
            records.append(
                {
                    "feature_a": column,
                    "feature_b": group,
                    "label_a": meta.loc[column, "label"],
                    "label_b": meta.loc[group, "label"],
                    "grain_a": meta.loc[column, "grain"],
                    "grain_b": meta.loc[group, "grain"],
                    "pair_type": "numeric_categorical",
                    "n": n,
                    # Neither is defined: the grouping has no order to be linear in.
                    "linear_pearson_r": None,
                    "monotone_spearman_rho": None,
                    "general_measure": "correlation_ratio_eta",
                    "general_value": round(eta, 4),
                    "nonlinearity_gap": None,
                    "dependence_beyond_monotone": None,
                    "relationship": (
                        "group_differences" if eta >= weak_threshold else "no_detected_association"
                    ),
                    "p_value": p_value,
                }
            )

    # -- categorical x categorical: Cramer's V ----------------------------
    group_columns = list(group_frame.columns)
    for i, left in enumerate(group_columns):
        for right in group_columns[i + 1 :]:
            if is_redundant_with(left, right) or is_redundant_with(right, left):
                continue
            block = group_frame[[left, right]].dropna()
            if len(block) < min_pairs:
                continue
            result = cramers_v(block, left, right, min_n=min_pairs)
            if result.get("cramers_v") is None:
                continue
            records.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "label_a": meta.loc[left, "label"],
                    "label_b": meta.loc[right, "label"],
                    "grain_a": meta.loc[left, "grain"],
                    "grain_b": meta.loc[right, "grain"],
                    "pair_type": "categorical_categorical",
                    "n": result["n"],
                    "linear_pearson_r": None,
                    "monotone_spearman_rho": None,
                    "general_measure": "cramers_v",
                    "general_value": result["cramers_v"],
                    "nonlinearity_gap": None,
                    "dependence_beyond_monotone": None,
                    "relationship": (
                        "group_differences"
                        if result["cramers_v"] >= weak_threshold
                        else "no_detected_association"
                    ),
                    "p_value": result.get("p_value"),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["p_fdr_adjusted"] = benjamini_hochberg(
        [None if pd.isna(p) else float(p) for p in out["p_value"]]
    )
    out["p_value"] = out["p_value"].map(lambda p: None if pd.isna(p) else round(float(p), 8))
    out["strength"] = out.apply(
        lambda row: _strength(
            row["general_value"]
            if pd.notna(row["general_value"])
            else (row["monotone_spearman_rho"] or 0.0),
            weak_threshold,
        ),
        axis=1,
    )
    # The strongest of whichever measures apply. A pair can be a restatement on
    # the rank scale (Spearman ~ 1) while distance correlation is unremarkable.
    out["max_association"] = (
        out[["linear_pearson_r", "monotone_spearman_rho", "general_value"]].abs().max(axis=1)
    )
    out["near_tautological"] = out["max_association"] > 0.95
    out["measure_note"] = out["pair_type"].map(
        {
            "numeric_numeric": (
                "Pearson = linear only; Spearman = monotone; distance correlation = any "
                "dependence (0 iff independent)"
            ),
            "numeric_categorical": (
                "correlation ratio (eta): share of the numeric variable's variance "
                "explained by the grouping; no linear/monotone distinction applies"
            ),
            "categorical_categorical": (
                "bias-corrected Cramer's V; no linear/monotone distinction applies"
            ),
        }
    )
    out["skipped_high_cardinality"] = ", ".join(sorted(wide)) if wide else ""
    return out.sort_values("general_value", ascending=False, na_position="last").reset_index(
        drop=True
    )


def nonlinearity_report(pairs: pd.DataFrame, *, top: int = 40) -> pd.DataFrame:
    """Pairs whose relationship is not linear, ranked by how badly Pearson misses.

    This is the table that answers "where would a linear model be wrong". Every
    row here is a pair where reporting Pearson alone would understate, or entirely
    miss, a real relationship.
    """
    if pairs.empty:
        return pairs
    block = pairs[pairs["pair_type"] == "numeric_numeric"].copy()
    if block.empty:
        return block
    block = block[block["relationship"].isin(["monotone_nonlinear", "non_monotone_dependence"])]
    if block.empty:
        return block
    block["missed_by_pearson"] = block[["nonlinearity_gap", "dependence_beyond_monotone"]].max(
        axis=1
    )
    block["interpretation"] = np.select(
        [
            block["relationship"] == "non_monotone_dependence",
            block["relationship"] == "linear_within_trimmed_range",
        ],
        [
            "dependence that is not monotone: Pearson AND Spearman both understate it",
            "linear across the central 80%: Pearson recovers once the outer deciles are "
            "clipped, so the deviation lives in the tails -- either a contaminated "
            "minority or tail curvature. Clipping cannot tell which; read the scatter",
        ],
        default="monotone but curved: Spearman captures it, Pearson understates it",
    )
    # Alias columns (power / power_hp / power_text / power_raw are one quantity
    # under four names) produce byte-identical statistics. Keeping them all fills
    # the report and the scatter panel with the same relationship repeated.
    signature = (
        block["linear_pearson_r"].round(4).astype(str)
        + "|"
        + block["monotone_spearman_rho"].round(4).astype(str)
        + "|"
        + block["general_value"].round(4).astype(str)
        + "|"
        + block["n"].astype(str)
    )
    duplicates = int(signature.duplicated().sum())
    block = block[~signature.duplicated()].copy()
    block["alias_duplicates_collapsed"] = duplicates
    return block.sort_values("missed_by_pearson", ascending=False).head(top).reset_index(drop=True)


def association_matrix(
    pairs: pd.DataFrame,
    *,
    measure: str = "monotone_spearman_rho",
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Square matrix of one measure, for a heatmap.

    Symmetric by construction, with 1.0 on the diagonal.
    """
    if pairs.empty or measure not in pairs.columns:
        return pd.DataFrame()
    block = pairs.dropna(subset=[measure])
    if features:
        block = block[block["feature_a"].isin(features) & block["feature_b"].isin(features)]
    if block.empty:
        return pd.DataFrame()
    names = sorted(set(block["feature_a"]) | set(block["feature_b"]))
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in block.itertuples():
        value = float(getattr(row, measure))
        matrix.loc[row.feature_a, row.feature_b] = value
        matrix.loc[row.feature_b, row.feature_a] = value
    return matrix


def summarise_pairs(pairs: pd.DataFrame) -> dict[str, Any]:
    """Headline counts for the report."""
    if pairs.empty:
        return {"pairs": 0}
    numeric = pairs[pairs["pair_type"] == "numeric_numeric"]
    shapes = pairs["relationship"].value_counts().to_dict()
    return {
        "pairs_tested": int(len(pairs)),
        "by_pair_type": {str(k): int(v) for k, v in pairs["pair_type"].value_counts().items()},
        "relationship_shapes": {str(k): int(v) for k, v in shapes.items()},
        "numeric_pairs": int(len(numeric)),
        "monotone_nonlinear": int((numeric["relationship"] == "monotone_nonlinear").sum()),
        "linear_within_trimmed_range": int(
            (numeric["relationship"] == "linear_within_trimmed_range").sum()
        ),
        "non_monotone_dependence": int(
            (numeric["relationship"] == "non_monotone_dependence").sum()
        ),
        "near_tautological": int(pairs["near_tautological"].sum()),
        "significant_after_fdr": int((pairs["p_fdr_adjusted"].fillna(1.0) < 0.05).sum()),
        "multiple_comparisons": (
            "Benjamini-Hochberg FDR, not Holm: over this many pairs Holm is so "
            "conservative that nothing survives and the correction stops informing."
        ),
        "interpretation": (
            "Pearson measures LINEAR association only. Spearman measures MONOTONE "
            "association. Distance correlation detects ANY dependence and is zero only "
            "under independence. A pair where Spearman greatly exceeds Pearson is curved; "
            "a pair where distance correlation exceeds Spearman is not monotone at all."
        ),
    }
