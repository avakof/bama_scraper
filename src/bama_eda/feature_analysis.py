"""Every scraped feature, profiled and compared — not a curated shortlist.

The rest of the package analyses a fixed set of variables chosen in advance. That
is fine for the headline questions and wrong for the question "what is actually in
this data", because a hand-picked list silently decides what the analysis is
allowed to notice.

This module discovers the analysable features from the data itself, profiles all
of them, and compares their distributions across any dimension (city, province,
brand, seller type, …).

**Grain matters and is tracked.** Three kinds of column end up side by side in the
cross-section, and they do not mean the same thing:

``listing``
    observed for this individual advertisement (price, mileage, location).
``vehicle``
    scraped from this advertisement's detail page (brand, colour, engine).
``trim``
    a property of the model-trim, identical for every listing of that trim
    (sunroof, ABS, kerb weight, boot capacity).

A breakdown such as "sunroof by city" over a ``trim`` feature describes the **trim
mix** in that city. It is not a count of individually verified cars, and every
output from this module carries that label.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .models import PRIVATE_COLUMNS, is_texty

#: Column prefixes and what grain they carry.
GRAIN_BY_PREFIX: dict[str, str] = {
    "spec_": "trim",
    "deep_": "vehicle",
    "snap_": "vehicle",
    "obs_": "listing",
    "events_": "listing",
    "source_": "provenance",
}

#: Columns that are provenance or identity, never a feature to analyse.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "run_id",
        "scheduled_for",
        "run_started_at",
        "run_finished_at",
        "search_configuration_hash",
        "scraper_version",
        "monitor_version",
        "analysis_version",
        "database_backend",
        "ad_id",
        "platform",
        "platform_ad_id",
        "canonical_url",
        "vehicle_fingerprint",
        "latest_snapshot_id",
        "snapshot_id",
        "repost_parent_ad_id",
        "card_hash",
        "snap_content_hash",
        "deep_api_json_sha256",
        "deep_html_sha256",
        "age_reference_instant",
        "age_reference_basis",
        "snapshot_scraped_at",
        "obs_card_image_url",
        "deep_primary_image_url",
        "deep_dealer_link",
        "deep_meta_canonical",
        "deep_url",
        "deep_canonical_url",
        "deep_api_url",
        "deep_review_url",
        "deep_price_url",
        # Keys and codes: they identify a trim or a dealer, they do not describe
        # one. Left in, they top every ranking purely by having many levels.
        "deep_review_key",
        "deep_price_key",
        "deep_code",
        "deep_dealer_id",
        "deep_ad_class_id",
        "deep_generation_code",
        "spec_generation_code",
        "vehicle_entity_id",
    }
)

#: Columns that ARE the price, restated. Correlating them with price measures
#: nothing except that two copies of a number are equal, and they crowd every
#: real finding out of the ranking.
PRICE_ALIASES: frozenset[str] = frozenset(
    {
        "price_toman",
        "log10_price",
        "price_text",
        "price_raw",
        "price_normalized",
        "card_price_raw",
        "card_price_normalized",
        "collaboration_price_toman",
        "installment_total_toman",
        "down_payment_toman",
        "prepayment_secondary_toman",
        "installment_amount_toman",
        "price_band",
    }
)


def is_price_alias(column: str) -> bool:
    """Whether a column is the asking price under another name."""
    return _strip_prefix(column) in PRICE_ALIASES


def is_redundant_with(feature: str, dimension: str) -> bool:
    """Whether comparing ``feature`` across ``dimension`` is circular.

    Charting the price across price bands, or mileage across mileage bands,
    produces a perfect-looking result that says only that a variable agrees with a
    bucketing of itself.
    """
    left, right = _strip_prefix(feature), _strip_prefix(dimension)
    if left == right:
        return True
    if is_price_alias(feature) and is_price_alias(dimension):
        return True
    # `mileage_km` against `mileage_band`, `year_jalali` against `year_band`, ...
    stem = right.removesuffix("_band")
    return stem != right and (left == stem or left.startswith(stem))


#: Feature kinds.
NUMERIC, BOOLEAN, CATEGORICAL, TEXT = "numeric", "boolean", "categorical", "text"

#: Persian/English tokens the deep scraper uses for present/absent.
_TRUE_TOKENS = {"true", "1", "yes", "present", "دارد", "بله", "دارای"}
_FALSE_TOKENS = {"false", "0", "no", "absent", "ندارد", "خیر", "-", "—", "ندارد."}


def _grain(column: str) -> str:
    for prefix, grain in GRAIN_BY_PREFIX.items():
        if column.startswith(prefix):
            return grain
    return "listing"


def _strip_prefix(column: str) -> str:
    for prefix in GRAIN_BY_PREFIX:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def is_flag_name(column: str) -> bool:
    """Whether a column *name* declares a flag.

    Used instead of "the values happen to be 0 and 1", because a count column
    holding only 0 and 1 today may hold 5 tomorrow, and a feature whose kind flips
    between runs breaks reproducibility: the same code over a longer history would
    silently switch a variable from a share to a median.
    """
    label = _strip_prefix(column).lower()
    return (
        label.startswith(("is_", "has_", "can_"))
        or label.endswith(("_ok", "_flag", "_enabled", "_reliable", "_truncated"))
        or label in {"authenticated", "pin", "specialcase", "usable", "eligible"}
    )


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce, stripping the unit suffixes the spec values carry."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True).replace("", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _looks_like_measure(values: pd.Series, *, max_unit_tokens: int = 1) -> bool:
    """Whether text values are numbers with a unit rather than descriptions.

    "1165 کیلوگرم" leaves one token after the digits are removed -> a measure.
    "5 دنده دستی" leaves two -> a description that happens to start with a number.
    """
    import re

    residues = []
    for value in values.astype(str).head(200):
        residue = re.sub(r"[\d.,\-]+", " ", value).strip()
        residues.append(len(residue.split()))
    if not residues:
        return False
    return sum(r <= max_unit_tokens for r in residues) / len(residues) > 0.9


def classify_feature(series: pd.Series, column: str | None = None) -> str:
    """Decide how a column should be analysed.

    Booleans come from the dtype, from a present/absent **value vocabulary** (the
    spec pipeline stores these as ``دارد``/``ندارد``, which a dtype check would
    read as free text), or from a flag-shaped **name**. Never from the observed
    range, so the classification is stable as the data grows.
    """
    present = series.dropna()
    if present.empty:
        return TEXT
    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN
    if pd.api.types.is_numeric_dtype(series):
        if column and is_flag_name(column):
            unique = set(pd.to_numeric(present.unique(), errors="coerce"))
            if unique <= {0.0, 1.0}:
                return BOOLEAN
        return NUMERIC
    if is_texty(series):
        lowered = {str(v).strip().lower() for v in present.unique()[:50]}
        if lowered and lowered <= (_TRUE_TOKENS | _FALSE_TOKENS):
            return BOOLEAN
        # A measure that arrived as text with its unit attached -- the spec values
        # look like "1165 کیلوگرم". Plain pd.to_numeric fails on those, so a
        # unit-stripping coercion is needed. But it must not swallow *descriptions*
        # that merely begin with a number: "5 دنده دستی" ("5-speed manual") would
        # become the number 5 and lose manual-versus-automatic entirely. So the
        # residue after removing the digits has to look like a unit, not a phrase.
        if _looks_like_measure(present):
            coerced = to_numeric(present)
            if coerced.notna().mean() > 0.9 and coerced.nunique() > 2:
                return NUMERIC
        if present.nunique() <= max(30, len(present) // 20):
            return CATEGORICAL
        return TEXT
    return CATEGORICAL


def to_boolean(series: pd.Series) -> pd.Series:
    """Map a presence/absence column to nullable booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda v: pd.NA if pd.isna(v) else bool(v)).astype("boolean")

    def convert(value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        text = str(value).strip().lower()
        if text in _TRUE_TOKENS:
            return True
        if text in _FALSE_TOKENS:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def discover_features(
    frame: pd.DataFrame, *, min_coverage: float = 0.01, min_distinct: int = 2
) -> pd.DataFrame:
    """Catalogue every column, with its kind, grain, coverage and usability.

    Unusable columns are **kept in the catalogue** with a reason. Dropping them
    would make "we did not analyse this" indistinguishable from "this does not
    exist".
    """
    total = len(frame)
    records: list[dict[str, Any]] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS or column.lower() in PRIVATE_COLUMNS:
            continue
        series = frame[column]
        non_null = int(series.notna().sum())
        coverage = non_null / total if total else 0.0
        distinct = int(series.nunique(dropna=True))
        kind = classify_feature(series, column)

        reasons: list[str] = []
        if non_null == 0:
            reasons.append("entirely empty")
        elif coverage < min_coverage:
            reasons.append(f"coverage {coverage:.1%} below {min_coverage:.0%}")
        if distinct < min_distinct:
            reasons.append("no variation")
        if kind == TEXT:
            reasons.append("free text; analysed by text_analysis, not as a feature")

        records.append(
            {
                "feature": column,
                "label": _strip_prefix(column),
                "kind": kind,
                "grain": _grain(column),
                "non_null": non_null,
                "coverage_pct": round(coverage * 100, 3),
                "distinct": distinct,
                "usable": not reasons,
                "excluded_because": "; ".join(reasons),
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(
        ["usable", "grain", "kind", "coverage_pct"], ascending=[False, True, True, False]
    ).reset_index(drop=True)


def usable_features(catalogue: pd.DataFrame, kinds: tuple[str, ...] | None = None) -> list[str]:
    if catalogue.empty:
        return []
    block = catalogue[catalogue["usable"]]
    if kinds:
        block = block[block["kind"].isin(kinds)]
    return block["feature"].tolist()


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def profile_all(
    frame: pd.DataFrame, catalogue: pd.DataFrame, *, min_group_size: int = 10, top: int = 12
) -> pd.DataFrame:
    """One distribution summary per usable feature, whatever its kind."""
    records: list[dict[str, Any]] = []
    for _, meta in catalogue[catalogue["usable"]].iterrows():
        column, kind = meta["feature"], meta["kind"]
        series = frame[column]
        base = {
            "feature": column,
            "label": meta["label"],
            "kind": kind,
            "grain": meta["grain"],
            "non_null": meta["non_null"],
            "coverage_pct": meta["coverage_pct"],
        }
        if kind == NUMERIC:
            values = to_numeric(series).dropna()
            if values.empty:
                continue
            quantiles = values.quantile([0.25, 0.5, 0.75])
            records.append(
                {
                    **base,
                    "summary": "median [IQR]",
                    "value": f"{quantiles.loc[0.5]:,.4g}",
                    "median": round(float(quantiles.loc[0.5]), 4),
                    "p25": round(float(quantiles.loc[0.25]), 4),
                    "p75": round(float(quantiles.loc[0.75]), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                    "skewness": round(float(values.skew()), 4) if len(values) > 2 else None,
                }
            )
        elif kind == BOOLEAN:
            flags = to_boolean(series).dropna()
            if flags.empty:
                continue
            share = float(flags.mean())
            records.append(
                {
                    **base,
                    "summary": "share present",
                    "value": f"{share:.1%}",
                    "present": int(flags.sum()),
                    "absent": int((~flags.astype(bool)).sum()),
                    "share_present_pct": round(share * 100, 3),
                }
            )
        else:
            counts = series.dropna().astype(str).value_counts()
            if counts.empty:
                continue
            top_values = counts.head(top)
            records.append(
                {
                    **base,
                    "summary": "top levels",
                    "value": "; ".join(f"{v} ({c:,})" for v, c in top_values.items())[:200],
                    "levels": int(len(counts)),
                    "levels_above_min_group": int((counts >= min_group_size).sum()),
                    "modal_value": str(counts.index[0]),
                    "modal_share_pct": round(counts.iloc[0] / counts.sum() * 100, 3),
                }
            )
    return pd.DataFrame(records)


def level_distribution(
    frame: pd.DataFrame, catalogue: pd.DataFrame, *, min_group_size: int = 10, top: int = 25
) -> pd.DataFrame:
    """Long-format level counts for every categorical and boolean feature."""
    blocks: list[dict[str, Any]] = []
    selected = catalogue[catalogue["usable"] & catalogue["kind"].isin([CATEGORICAL, BOOLEAN])]
    for _, meta in selected.iterrows():
        column = meta["feature"]
        series = (
            to_boolean(frame[column]).map({True: "present", False: "absent"})
            if meta["kind"] == BOOLEAN
            else frame[column]
        )
        counts = series.dropna().astype(str).value_counts().head(top)
        known = int(series.notna().sum())
        for value, count in counts.items():
            blocks.append(
                {
                    "feature": column,
                    "label": meta["label"],
                    "kind": meta["kind"],
                    "grain": meta["grain"],
                    "level": str(value),
                    "count": int(count),
                    "share_of_known_pct": round(count / known * 100, 3) if known else None,
                    "known_values": known,
                    "sufficient_sample": bool(count >= min_group_size),
                }
            )
    return pd.DataFrame(blocks)


# ---------------------------------------------------------------------------
# Comparison across a dimension
# ---------------------------------------------------------------------------


def compare_across(
    frame: pd.DataFrame,
    catalogue: pd.DataFrame,
    dimension: str,
    *,
    min_group_size: int = 10,
    max_groups: int = 12,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """How every feature's distribution differs across the groups of ``dimension``.

    Groups below ``min_group_size`` are excluded from the comparison entirely
    rather than shown as a thin bar, and the count of excluded groups is carried
    on every row so a reader can see what the table is not covering.
    """
    if dimension not in frame.columns:
        return pd.DataFrame()

    groups = frame[dimension].dropna().astype(str)
    counts = groups.value_counts()
    keep = counts[counts >= min_group_size].head(max_groups)
    if keep.empty:
        return pd.DataFrame()
    dropped = int((counts < min_group_size).sum())

    subset = frame[frame[dimension].astype(str).isin(keep.index)]
    chosen = features or usable_features(catalogue, (NUMERIC, BOOLEAN, CATEGORICAL))
    meta_by_feature = catalogue.set_index("feature")

    records: list[dict[str, Any]] = []
    for column in chosen:
        if column == dimension or column not in subset.columns:
            continue
        if is_redundant_with(column, dimension):
            continue
        meta = meta_by_feature.loc[column] if column in meta_by_feature.index else None
        if meta is None:
            continue
        kind, grain = meta["kind"], meta["grain"]
        for group, block in subset.groupby(subset[dimension].astype(str)):
            record = {
                "dimension": dimension,
                "group": group,
                "group_n": int(len(block)),
                "feature": column,
                "label": meta["label"],
                "kind": kind,
                "grain": grain,
                "groups_below_min_excluded": dropped,
            }
            if kind == NUMERIC:
                values = to_numeric(block[column]).dropna()
                if values.empty:
                    continue
                record.update(
                    {
                        "n": int(len(values)),
                        "statistic": "median",
                        "value": round(float(values.median()), 4),
                        "p25": round(float(values.quantile(0.25)), 4),
                        "p75": round(float(values.quantile(0.75)), 4),
                    }
                )
            elif kind == BOOLEAN:
                flags = to_boolean(block[column]).dropna()
                if flags.empty:
                    continue
                record.update(
                    {
                        "n": int(len(flags)),
                        "statistic": "share_present_pct",
                        "value": round(float(flags.mean()) * 100, 3),
                    }
                )
            else:
                values = block[column].dropna().astype(str)
                if values.empty:
                    continue
                counts_in_group = values.value_counts()
                record.update(
                    {
                        "n": int(len(values)),
                        "statistic": "modal_share_pct",
                        "value": round(counts_in_group.iloc[0] / len(values) * 100, 3),
                        "modal_value": str(counts_in_group.index[0]),
                    }
                )
            records.append(record)

    out = pd.DataFrame(records)
    if not out.empty and "grain" in out.columns:
        out["grain_note"] = np.where(
            out["grain"] == "trim",
            "TRIM-LEVEL: describes the trim mix in this group, not per-car measurement",
            "",
        )
    return out


def differentiating_features(
    frame: pd.DataFrame,
    catalogue: pd.DataFrame,
    dimension: str,
    *,
    min_group_size: int = 10,
    max_groups: int = 12,
    top: int = 25,
) -> pd.DataFrame:
    """Rank features by how much they differ across the groups of ``dimension``.

    Rank-based and non-parametric throughout — Kruskal-Wallis with
    epsilon-squared for numeric features, chi-square with bias-corrected
    Cramér's V for categorical and boolean ones. Holm-adjusted across the family,
    and labelled exploratory: these comparisons were chosen after seeing the data.
    """
    from .correlations import adjust_pvalues, cramers_v

    if dimension not in frame.columns:
        return pd.DataFrame()

    counts = frame[dimension].dropna().astype(str).value_counts()
    keep = counts[counts >= min_group_size].head(max_groups)
    if len(keep) < 2:
        return pd.DataFrame()
    subset = frame[frame[dimension].astype(str).isin(keep.index)].copy()
    subset["_group"] = subset[dimension].astype(str)

    records: list[dict[str, Any]] = []
    for _, meta in catalogue[catalogue["usable"]].iterrows():
        column, kind = meta["feature"], meta["kind"]
        if column == dimension or column not in subset.columns:
            continue
        if is_redundant_with(column, dimension):
            continue

        if kind == NUMERIC:
            values = to_numeric(subset[column])
            block = pd.DataFrame({"v": values, "g": subset["_group"]}).dropna()
            groups = [g["v"].to_numpy() for _, g in block.groupby("g") if len(g) >= min_group_size]
            if len(groups) < 2 or all(len(np.unique(g)) == 1 for g in groups):
                continue
            try:
                result = stats.kruskal(*groups)
            except ValueError:
                continue
            n, k = sum(len(g) for g in groups), len(groups)
            epsilon: float | None = float((result.statistic - k + 1) / (n - k)) if n > k else None
            records.append(
                {
                    "dimension": dimension,
                    "feature": column,
                    "label": meta["label"],
                    "kind": kind,
                    "grain": meta["grain"],
                    "groups": k,
                    "n": n,
                    "test": "kruskal_wallis",
                    "statistic": round(float(result.statistic), 4),
                    "p_value": float(result.pvalue),
                    "effect_size": round(epsilon, 4) if epsilon is not None else None,
                    "effect_size_name": "epsilon_squared",
                }
            )
        elif kind in (BOOLEAN, CATEGORICAL):
            values = (
                to_boolean(subset[column]).map({True: "present", False: "absent"})
                if kind == BOOLEAN
                else subset[column].astype(str)
            )
            block = pd.DataFrame({"v": values, "g": subset["_group"]}).dropna()
            if block["v"].nunique() < 2 or len(block) < min_group_size * 2:
                continue
            result = cramers_v(block, "v", "g", min_n=min_group_size * 2)
            if result.get("cramers_v") is None:
                continue
            records.append(
                {
                    "dimension": dimension,
                    "feature": column,
                    "label": meta["label"],
                    "kind": kind,
                    "grain": meta["grain"],
                    "groups": result.get("levels_b"),
                    "n": result["n"],
                    "test": "chi_square",
                    "statistic": result.get("chi2"),
                    "p_value": result.get("p_value"),
                    "effect_size": result["cramers_v"],
                    "effect_size_name": "cramers_v",
                    "cells_with_expected_below_5": result.get("cells_with_expected_below_5"),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["p_holm_adjusted"] = adjust_pvalues(
        [None if pd.isna(p) else float(p) for p in out["p_value"]]
    )
    out["p_value"] = out["p_value"].map(lambda p: None if pd.isna(p) else round(float(p), 8))
    out["near_tautological"] = out["effect_size"].abs() > 0.95
    out["note"] = np.where(
        out["near_tautological"],
        "|effect| > 0.95: the feature nearly determines the grouping (or vice versa); "
        "a restatement, not a finding",
        np.where(
            out["grain"] == "trim",
            "EXPLORATORY; TRIM-LEVEL feature, so this reflects the trim mix per group",
            "EXPLORATORY; rank-based/non-parametric; association, not causation",
        ),
    )
    return (
        out.sort_values("effect_size", ascending=False, na_position="last")
        .head(top)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Association with price
# ---------------------------------------------------------------------------


def feature_price_associations(
    frame: pd.DataFrame,
    catalogue: pd.DataFrame,
    *,
    price_column: str = "price_toman",
    min_n: int = 30,
    top: int = 40,
) -> pd.DataFrame:
    """Every usable feature against asking price, with the right test per kind.

    Spearman for numeric, Mann-Whitney with rank-biserial for boolean,
    Kruskal-Wallis with epsilon-squared for categorical. Holm-adjusted across the
    whole family, because running one test per feature over ~200 features
    guarantees false positives otherwise.
    """
    from .correlations import adjust_pvalues

    if price_column not in frame.columns:
        return pd.DataFrame()
    price = pd.to_numeric(frame[price_column], errors="coerce")
    # 0 is Bama's "negotiable"; it is not a price.
    price = price.where(price > 0)

    records: list[dict[str, Any]] = []
    skipped_aliases: list[str] = []
    for _, meta in catalogue[catalogue["usable"]].iterrows():
        column, kind = meta["feature"], meta["kind"]
        if column == price_column or column not in frame.columns:
            continue
        # A restatement of the price cannot inform an analysis of the price.
        if is_price_alias(column):
            skipped_aliases.append(column)
            continue

        if kind == NUMERIC:
            values = to_numeric(frame[column])
            mask = values.notna() & price.notna()
            if int(mask.sum()) < min_n or values[mask].nunique() < 2:
                continue
            result = stats.spearmanr(values[mask], price[mask])
            records.append(
                {
                    "feature": column,
                    "label": meta["label"],
                    "kind": kind,
                    "grain": meta["grain"],
                    "n": int(mask.sum()),
                    "test": "spearman",
                    "effect_size": round(float(result.statistic), 4),
                    "effect_size_name": "spearman_rho",
                    "p_value": float(result.pvalue),
                    "direction": "higher price" if result.statistic > 0 else "lower price",
                }
            )
        elif kind == BOOLEAN:
            flags = to_boolean(frame[column])
            mask = flags.notna() & price.notna()
            present = price[mask & (flags == True)]  # noqa: E712 - nullable boolean
            absent = price[mask & (flags == False)]  # noqa: E712
            if len(present) < min_n or len(absent) < min_n:
                continue
            result = stats.mannwhitneyu(present, absent, alternative="two-sided")
            effect = float(2 * result.statistic / (len(present) * len(absent)) - 1)
            records.append(
                {
                    "feature": column,
                    "label": meta["label"],
                    "kind": kind,
                    "grain": meta["grain"],
                    "n": int(len(present) + len(absent)),
                    "test": "mann_whitney_u",
                    "effect_size": round(effect, 4),
                    "effect_size_name": "rank_biserial",
                    "p_value": float(result.pvalue),
                    "median_present": round(float(present.median()), 2),
                    "median_absent": round(float(absent.median()), 2),
                    "direction": "higher price when present"
                    if present.median() > absent.median()
                    else "lower price when present",
                }
            )
        elif kind == CATEGORICAL:
            values = frame[column].astype(str)
            block = pd.DataFrame({"v": values, "p": price}).dropna()
            groups = [g["p"].to_numpy() for _, g in block.groupby("v") if len(g) >= min_n]
            if len(groups) < 2:
                continue
            try:
                result = stats.kruskal(*groups)
            except ValueError:
                continue
            n, k = sum(len(g) for g in groups), len(groups)
            epsilon: float | None = float((result.statistic - k + 1) / (n - k)) if n > k else None
            records.append(
                {
                    "feature": column,
                    "label": meta["label"],
                    "kind": kind,
                    "grain": meta["grain"],
                    "n": n,
                    "test": "kruskal_wallis",
                    "effect_size": round(epsilon, 4) if epsilon is not None else None,
                    "effect_size_name": "epsilon_squared",
                    "p_value": float(result.pvalue),
                    "direction": f"{k} levels compared",
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["p_holm_adjusted"] = adjust_pvalues(
        [None if pd.isna(p) else float(p) for p in out["p_value"]]
    )
    out["p_value"] = out["p_value"].map(lambda p: None if pd.isna(p) else round(float(p), 8))
    out["abs_effect"] = out["effect_size"].abs()
    out["near_tautological"] = out["effect_size"].abs() > 0.95
    out["caveat"] = np.where(
        out["near_tautological"],
        "|effect| > 0.95: almost certainly a restatement of the same quantity, not a finding",
        np.where(
            out["grain"] == "trim",
            "TRIM-LEVEL feature: it varies by model-trim, so this association is largely "
            "a statement about which trims are expensive, not about the feature itself",
            "association in one cross-section of ASKING prices; not causal",
        ),
    )
    return (
        out.sort_values("abs_effect", ascending=False, na_position="last")
        .head(top)
        .drop(columns=["abs_effect"])
        .reset_index(drop=True)
    )


def summarise_catalogue(catalogue: pd.DataFrame) -> dict[str, Any]:
    """Headline counts for the report."""
    if catalogue.empty:
        return {"features_total": 0}
    usable = catalogue[catalogue["usable"]]
    return {
        "features_total": int(len(catalogue)),
        "features_usable": int(len(usable)),
        "by_kind": {str(k): int(v) for k, v in usable["kind"].value_counts().items()},
        "by_grain": {str(k): int(v) for k, v in usable["grain"].value_counts().items()},
        "excluded": {
            str(k): int(v)
            for k, v in catalogue[~catalogue["usable"]]["excluded_because"]
            .str.split(";")
            .str[0]
            .value_counts()
            .items()
        },
        "grain_note": (
            "trim-grain features are properties of a model-trim, identical across every "
            "listing of that trim. Valid as covariates; not per-vehicle measurements."
        ),
    }
