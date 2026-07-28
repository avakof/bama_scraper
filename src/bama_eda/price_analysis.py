"""Asking-price analysis.

The unit is **Iranian toman** throughout and is never converted. These are
**asking** prices from listings, not transaction prices: this dataset contains no
record of what anything sold for, and no statistic here should be described as a
market price realised.

Medians and IQRs are the primary summaries; a trimmed mean is offered alongside
for groups large enough to support it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .profiling import add_bands

CURRENCY = "toman"

#: Dimensions price is summarised across, in report order.
PRICE_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("brand", "deep_brand"),
    ("model", "deep_model"),
    ("trim", "deep_trim"),
    ("production_year", "year_jalali"),
    ("mileage_band", "mileage_band"),
    ("body_condition", "deep_body_status"),
    ("condition_new_used", "deep_condition_new_used"),
    ("transmission", "deep_transmission"),
    ("fuel_type", "deep_fuel_type"),
    ("seller_type", "deep_seller_type"),
    ("city", "deep_city"),
    ("province", "deep_province"),
    ("price_band", "price_band"),
    ("status", "current_status"),
    ("was_reposted", "was_reposted_flag"),
    ("media_count_band", "media_count_band"),
    ("description_completeness", "description_band"),
)


def _prepare(cross: pd.DataFrame) -> pd.DataFrame:
    """Add the derived grouping columns price is segmented by."""
    frame = add_bands(cross)
    if "media_count" in frame.columns:
        frame["media_count_band"] = pd.cut(
            pd.to_numeric(frame["media_count"], errors="coerce"),
            bins=[-0.1, 0, 3, 6, 10, 1000],
            labels=["none", "1-3", "4-6", "7-10", "11+"],
        )
    if "description_length" in frame.columns:
        frame["description_band"] = pd.cut(
            pd.to_numeric(frame["description_length"], errors="coerce"),
            bins=[-0.1, 0, 100, 300, 800, 100000],
            labels=["absent", "short", "medium", "long", "very_long"],
        )
    if "repost_parent_ad_id" in frame.columns or "current_status" in frame.columns:
        parent = frame.get("repost_parent_ad_id")
        status = frame.get("current_status", pd.Series([""] * len(frame), index=frame.index))
        frame["was_reposted_flag"] = np.where(
            (parent.notna() if parent is not None else False) | status.astype(str).eq("reposted"),
            "reposted",
            "not_reposted",
        )
    return frame


def price_by_segment(
    cross: pd.DataFrame, *, min_group_size: int = 10, top_per_dimension: int = 25
) -> pd.DataFrame:
    """Robust price summary for every segment of every dimension."""
    frame = _prepare(cross)
    if "price_toman" not in frame.columns:
        return pd.DataFrame()

    blocks: list[pd.DataFrame] = []
    for label, column in PRICE_SEGMENTS:
        if column not in frame.columns:
            continue
        # dict.fromkeys de-duplicates while preserving order: several dimensions
        # group BY one of the value columns (production_year groups by
        # year_jalali), and a repeated name makes frame[column] a DataFrame.
        wanted = list(dict.fromkeys([column, "price_toman", "mileage_km", "year_jalali"]))
        block = frame[[c for c in wanted if c in frame.columns]].copy()
        block = block.dropna(subset=[column, "price_toman"])
        block = block[pd.to_numeric(block["price_toman"], errors="coerce") > 0]
        if block.empty:
            continue

        grouped = block.groupby(block[column].astype(str), observed=True)
        summary = grouped["price_toman"].agg(
            listing_count="count",
            median_price="median",
            mean_price="mean",
            p25_price=lambda s: s.quantile(0.25),
            p75_price=lambda s: s.quantile(0.75),
            minimum_price="min",
            maximum_price="max",
            std_price="std",
        )
        summary["trimmed_mean_price"] = grouped["price_toman"].apply(_trimmed_mean)
        summary["median_mileage"] = (
            grouped["mileage_km"].median() if "mileage_km" in block.columns else None
        )
        summary["median_year"] = (
            grouped["year_jalali"].median() if "year_jalali" in block.columns else None
        )
        summary["iqr_price"] = summary["p75_price"] - summary["p25_price"]
        summary["ci95_median_low"], summary["ci95_median_high"] = zip(
            *grouped["price_toman"].apply(_median_ci), strict=False
        )
        summary = summary.reset_index().rename(columns={column: "segment_value"})
        summary.insert(0, "dimension", label)
        summary.insert(1, "source_column", column)
        summary["sufficient_sample"] = summary["listing_count"] >= min_group_size
        summary["sample_note"] = np.where(
            summary["sufficient_sample"], "", f"insufficient_sample (< {min_group_size})"
        )
        summary["currency"] = CURRENCY
        summary["price_basis"] = "asking_price_not_transaction_price"
        blocks.append(summary.sort_values("listing_count", ascending=False).head(top_per_dimension))

    if not blocks:
        return pd.DataFrame()
    out = pd.concat(blocks, ignore_index=True)
    numeric = [c for c in out.columns if out[c].dtype.kind in "fc"]
    out[numeric] = out[numeric].round(2)
    return out


def _trimmed_mean(series: pd.Series, proportion: float = 0.1) -> float | None:
    """10% trimmed mean, only where trimming leaves something to average."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 10:
        return None
    return float(stats.trim_mean(values, proportiontocut=proportion))


def _median_ci(series: pd.Series, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Distribution-free confidence interval for the median (order statistics).

    Reported only for groups of at least 10; for smaller groups the interval spans
    essentially the whole range and would imply a precision that is not there.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().sort_values().to_numpy()
    n = len(values)
    if n < 10:
        return (None, None)
    # Normal approximation to the binomial order statistic bounds.
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower_rank = int(np.floor((n - z * np.sqrt(n)) / 2))
    upper_rank = int(np.ceil(1 + (n + z * np.sqrt(n)) / 2))
    lower_rank = max(0, min(lower_rank, n - 1))
    upper_rank = max(0, min(upper_rank - 1, n - 1))
    return (float(values[lower_rank]), float(values[upper_rank]))


def comparable_segments(
    cross: pd.DataFrame, *, min_group_size: int = 10, top: int = 40
) -> pd.DataFrame:
    """Comparable-vehicle groups: brand x model x year x mileage band.

    This is the closest this dataset gets to "what is a car like this asking",
    and it is still an asking-price distribution, not a valuation.
    """
    frame = _prepare(cross)
    keys = [
        c for c in ("deep_brand", "deep_model", "year_jalali", "mileage_band") if c in frame.columns
    ]
    if len(keys) < 2 or "price_toman" not in frame.columns:
        return pd.DataFrame()

    block = frame.dropna(subset=[*keys, "price_toman"]).copy()
    block = block[pd.to_numeric(block["price_toman"], errors="coerce") > 0]
    if block.empty:
        return pd.DataFrame()

    for key in keys:
        block[key] = block[key].astype(str)

    grouped = block.groupby(keys, observed=True)
    summary = grouped.agg(
        listing_count=("price_toman", "count"),
        median_price=("price_toman", "median"),
        p25_price=("price_toman", lambda s: s.quantile(0.25)),
        p75_price=("price_toman", lambda s: s.quantile(0.75)),
        minimum_price=("price_toman", "min"),
        maximum_price=("price_toman", "max"),
        median_mileage=("mileage_km", "median"),
        median_listing_age_days=("listing_age_days_observed", "median"),
    ).reset_index()

    disappeared = (
        block.assign(
            gone=block["current_status"].astype(str).isin(["likely_removed", "likely_sold"])
        )
        .groupby(keys, observed=True)["gone"]
        .mean()
        .reset_index()
        .rename(columns={"gone": "disappearance_rate"})
    )
    summary = summary.merge(disappeared, on=keys, how="left")

    if "was_reposted_flag" in block.columns:
        reposts = (
            block.assign(rp=block["was_reposted_flag"].eq("reposted"))
            .groupby(keys, observed=True)["rp"]
            .mean()
            .reset_index()
            .rename(columns={"rp": "repost_rate"})
        )
        summary = summary.merge(reposts, on=keys, how="left")

    summary["sufficient_sample"] = summary["listing_count"] >= min_group_size
    summary["sample_note"] = np.where(
        summary["sufficient_sample"], "", f"insufficient_sample (< {min_group_size})"
    )
    summary["currency"] = CURRENCY
    numeric = [c for c in summary.columns if summary[c].dtype.kind in "fc"]
    summary[numeric] = summary[numeric].round(4)
    return summary.sort_values("listing_count", ascending=False).head(top).reset_index(drop=True)


def group_difference_tests(
    cross: pd.DataFrame,
    dimensions: tuple[str, ...] = ("deep_seller_type", "deep_transmission", "deep_fuel_type"),
    *,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Exploratory rank-based tests for price differences between groups.

    Kruskal-Wallis (3+ groups) and Mann-Whitney (2 groups), both rank-based
    because price is skewed. Effect sizes accompany every p-value, and the whole
    family is Holm-adjusted. Labelled exploratory: these tests were chosen after
    seeing the data, which is not a confirmatory design.
    """
    from .correlations import adjust_pvalues

    frame = _prepare(cross)
    records: list[dict[str, Any]] = []
    for column in dimensions:
        if column not in frame.columns or "price_toman" not in frame.columns:
            continue
        block = frame.dropna(subset=[column, "price_toman"])
        block = block[pd.to_numeric(block["price_toman"], errors="coerce") > 0]
        groups = [
            g["price_toman"].to_numpy()
            for _, g in block.groupby(block[column].astype(str), observed=True)
            if len(g) >= min_group_size
        ]
        names = [
            str(name)
            for name, g in block.groupby(block[column].astype(str), observed=True)
            if len(g) >= min_group_size
        ]
        if len(groups) < 2:
            records.append(
                {
                    "dimension": column,
                    "test": "none",
                    "groups": len(groups),
                    "n": int(sum(len(g) for g in groups)),
                    "statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "effect_size_name": None,
                    "note": f"fewer than 2 groups reach {min_group_size} listings",
                }
            )
            continue

        if len(groups) == 2:
            result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            n1, n2 = len(groups[0]), len(groups[1])
            # Rank-biserial correlation: interpretable as P(a>b) - P(b>a).
            effect = round(float(2 * result.statistic / (n1 * n2) - 1), 4)
            record = {
                "dimension": column,
                "test": "mann_whitney_u",
                "groups": 2,
                "n": n1 + n2,
                "statistic": round(float(result.statistic), 2),
                "p_value": float(result.pvalue),
                "effect_size": effect,
                "effect_size_name": "rank_biserial",
            }
        else:
            result = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            k = len(groups)
            # epsilon-squared: the share of rank variance explained.
            epsilon = round(float((result.statistic - k + 1) / (n - k)), 4) if n > k else None
            record = {
                "dimension": column,
                "test": "kruskal_wallis",
                "groups": k,
                "n": n,
                "statistic": round(float(result.statistic), 4),
                "p_value": float(result.pvalue),
                "effect_size": epsilon,
                "effect_size_name": "epsilon_squared",
            }
        record["group_names"] = ", ".join(names[:8])
        record["note"] = "EXPLORATORY; rank-based because price is skewed; not causal"
        records.append(record)

    out = pd.DataFrame(records)
    if not out.empty and "p_value" in out.columns:
        out["p_holm_adjusted"] = adjust_pvalues(
            [None if pd.isna(p) else float(p) for p in out["p_value"]]
        )
        out["p_value"] = out["p_value"].map(lambda p: None if pd.isna(p) else round(float(p), 6))
    return out


def price_change_analysis(panel: pd.DataFrame, cross: pd.DataFrame) -> dict[str, Any]:
    """Price movement over the observed runs.

    Uses the card price from consecutive observations of the same advertisement,
    so it works even when detail coverage is thin. A change here is an observed
    change in the advertised price; nothing about why it changed is inferred.
    """
    if panel.empty or "obs_card_price_normalized" not in panel.columns:
        return {"available": False, "note": "no card prices in the panel"}

    frame = panel[
        ["observation_run_id", "advertisement_id", "obs_card_price_normalized", "run_is_valid"]
    ].copy()
    frame = frame[frame["run_is_valid"].astype(bool)]
    frame["price"] = pd.to_numeric(frame["obs_card_price_normalized"], errors="coerce")
    frame = frame.sort_values(["advertisement_id", "observation_run_id"])
    frame["previous_price"] = frame.groupby("advertisement_id")["price"].shift(1)

    changed = frame.dropna(subset=["price", "previous_price"])
    changed = changed[changed["price"] != changed["previous_price"]].copy()
    if changed.empty:
        return {
            "available": True,
            "advertisements_with_price_change": 0,
            "change_events": 0,
            "note": (
                "no price change was observed across the runs in this database; with "
                "few runs this is expected and is not evidence that sellers do not "
                "adjust prices"
            ),
        }

    changed["absolute_change"] = changed["price"] - changed["previous_price"]
    changed["percentage_change"] = (
        changed["absolute_change"] / changed["previous_price"] * 100
    ).round(3)
    changed["direction"] = np.where(
        changed["absolute_change"] < 0, "price_decrease", "price_increase"
    )

    decreases = changed[changed["direction"] == "price_decrease"]
    per_ad = changed.groupby("advertisement_id").size()
    return {
        "available": True,
        "advertisements_with_price_change": int(changed["advertisement_id"].nunique()),
        "advertisements_total": int(frame["advertisement_id"].nunique()),
        "share_with_change_pct": round(
            changed["advertisement_id"].nunique()
            / max(1, frame["advertisement_id"].nunique())
            * 100,
            3,
        ),
        "change_events": int(len(changed)),
        "decreases": int(len(decreases)),
        "increases": int(len(changed) - len(decreases)),
        "median_absolute_decrease_toman": (
            float(decreases["absolute_change"].abs().median()) if len(decreases) else None
        ),
        "median_percentage_decrease": (
            float(decreases["percentage_change"].abs().median()) if len(decreases) else None
        ),
        "max_changes_per_advertisement": int(per_ad.max()),
        "median_changes_per_changing_advertisement": float(per_ad.median()),
        "currency": CURRENCY,
        "caveat": (
            "A price change is an observed edit to the advertisement. This analysis "
            "does not establish that a reduction caused a later disappearance, and the "
            "observation cadence bounds how quickly a change can be detected."
        ),
    }


def price_change_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-advertisement price-change rows for export."""
    if panel.empty or "obs_card_price_normalized" not in panel.columns:
        return pd.DataFrame(
            columns=[
                "advertisement_id",
                "from_run",
                "to_run",
                "old_price",
                "new_price",
                "absolute_change",
                "percentage_change",
                "direction",
                "currency",
            ]
        )
    frame = panel[panel["run_is_valid"].astype(bool)][
        ["observation_run_id", "advertisement_id", "obs_card_price_normalized"]
    ].copy()
    frame["price"] = pd.to_numeric(frame["obs_card_price_normalized"], errors="coerce")
    frame = frame.sort_values(["advertisement_id", "observation_run_id"])
    frame["previous_price"] = frame.groupby("advertisement_id")["price"].shift(1)
    frame["previous_run"] = frame.groupby("advertisement_id")["observation_run_id"].shift(1)

    changed = frame.dropna(subset=["price", "previous_price"])
    changed = changed[changed["price"] != changed["previous_price"]]
    if changed.empty:
        return pd.DataFrame(
            columns=[
                "advertisement_id",
                "from_run",
                "to_run",
                "old_price",
                "new_price",
                "absolute_change",
                "percentage_change",
                "direction",
                "currency",
            ]
        )
    out = pd.DataFrame(
        {
            "advertisement_id": changed["advertisement_id"],
            "from_run": changed["previous_run"].astype("Int64"),
            "to_run": changed["observation_run_id"],
            "old_price": changed["previous_price"],
            "new_price": changed["price"],
            "absolute_change": changed["price"] - changed["previous_price"],
        }
    )
    out["percentage_change"] = (out["absolute_change"] / out["old_price"] * 100).round(3)
    out["direction"] = np.where(out["absolute_change"] < 0, "price_decrease", "price_increase")
    out["currency"] = CURRENCY
    return out.reset_index(drop=True)
