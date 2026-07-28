"""Numeric profiling.

Robust summaries throughout: asking prices in a classifieds market are strongly
right-skewed, so a mean and a standard deviation describe a distribution that does
not exist. Percentiles and the IQR are reported alongside, and the skew is stated
so a reader can see when the mean is the wrong summary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Percentiles reported for every numeric variable.
PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def describe_numeric(series: pd.Series, name: str | None = None) -> dict[str, Any]:
    """Full robust summary of one numeric variable."""
    values = pd.to_numeric(series, errors="coerce")
    present = values.dropna()
    record: dict[str, Any] = {
        "variable": name or series.name,
        "count": int(len(present)),
        "missing": int(values.isna().sum()),
        "missing_pct": round(float(values.isna().mean()) * 100, 2) if len(values) else None,
    }
    if present.empty:
        record.update(dict.fromkeys(_EMPTY_KEYS))
        record["note"] = "no non-null values"
        return record

    quantiles = present.quantile(list(PERCENTILES))
    record.update(
        {
            "mean": _round(present.mean()),
            "std": _round(present.std(ddof=1)) if len(present) > 1 else None,
            "min": _round(present.min()),
            "p01": _round(quantiles.loc[0.01]),
            "p05": _round(quantiles.loc[0.05]),
            "p10": _round(quantiles.loc[0.10]),
            "p25": _round(quantiles.loc[0.25]),
            "median": _round(quantiles.loc[0.50]),
            "p75": _round(quantiles.loc[0.75]),
            "p90": _round(quantiles.loc[0.90]),
            "p95": _round(quantiles.loc[0.95]),
            "p99": _round(quantiles.loc[0.99]),
            "max": _round(present.max()),
            "iqr": _round(quantiles.loc[0.75] - quantiles.loc[0.25]),
            "skewness": _round(present.skew()) if len(present) > 2 else None,
            "zeros": int((present == 0).sum()),
            "negatives": int((present < 0).sum()),
            "distinct": int(present.nunique()),
        }
    )
    skew = record["skewness"]
    if skew is not None and abs(skew) > 1:
        record["note"] = (
            "strongly skewed; report the median, not the mean"
            if abs(skew) > 2
            else "moderately skewed; prefer the median"
        )
    else:
        record["note"] = ""
    return record


_EMPTY_KEYS = (
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
    "p95",
    "p99",
    "max",
    "iqr",
    "skewness",
    "zeros",
    "negatives",
    "distinct",
)


def _round(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # Large tomans stay integral; small ratios keep four places.
    return round(number, 4) if abs(number) < 1000 else round(number, 2)


#: The numeric variables profiled by default, in report order.
DEFAULT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "price_toman",
    "log10_price",
    "mileage_km",
    "year_jalali",
    "description_length",
    "media_count",
    "snapshot_age_hours",
    "listing_age_days_observed",
    "sale_confidence",
    "consecutive_misses",
    "total_seen_runs",
    "entry_delay_seconds",
    "deep_engine_volume_l",
    "deep_power_hp",
    "deep_dealer_ad_count",
)


def profile_frame(
    frame: pd.DataFrame, columns: tuple[str, ...] | list[str] = DEFAULT_NUMERIC_COLUMNS
) -> pd.DataFrame:
    """Profile every requested numeric column that exists."""
    records = [describe_numeric(frame[c], c) for c in columns if c in frame.columns]
    return pd.DataFrame(records)


def categorical_distribution(
    frame: pd.DataFrame, column: str, *, min_group_size: int = 10, top: int = 50
) -> pd.DataFrame:
    """Counts and shares for one categorical column.

    Groups below ``min_group_size`` are kept but labelled: dropping them would
    understate the tail, and presenting them unlabelled would invite a reader to
    compare a segment of 3 with one of 300.
    """
    if column not in frame.columns:
        return pd.DataFrame(
            columns=["value", "count", "share_pct", "sufficient_sample", "sample_note"]
        )
    series = frame[column]
    counts = series.fillna("(missing)").astype(str).value_counts(dropna=False)
    total = int(len(series))
    out = pd.DataFrame(
        {
            "value": counts.index.astype(str),
            "count": counts.to_numpy(),
        }
    )
    out["share_pct"] = (out["count"] / total * 100).round(2) if total else 0.0
    out["sufficient_sample"] = out["count"] >= min_group_size
    out["sample_note"] = np.where(
        out["sufficient_sample"], "", f"insufficient_sample (< {min_group_size})"
    )
    out.insert(0, "variable", column)
    return out.head(top)


#: Categorical columns profiled by default.
DEFAULT_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "current_status",
    "detail_availability",
    "sale_label",
    "filter_exit_reason",
    "price_source",
    "year_calendar_source",
    "published_at_source",
    "deep_brand",
    "deep_model",
    "deep_trim",
    "deep_body_type_fa",
    "deep_transmission",
    "deep_fuel_type",
    "deep_condition_new_used",
    "deep_body_status",
    "deep_seller_type",
    "deep_province",
    "deep_city",
    "deep_body_color",
    "obs_card_location",
)


def profile_categoricals(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = DEFAULT_CATEGORICAL_COLUMNS,
    *,
    min_group_size: int = 10,
    top: int = 30,
) -> pd.DataFrame:
    blocks = [
        categorical_distribution(frame, c, min_group_size=min_group_size, top=top)
        for c in columns
        if c in frame.columns
    ]
    blocks = [b for b in blocks if not b.empty]
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def bucket(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """Bucket a numeric series, keeping NA as NA rather than as a bucket."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False)


#: Toman price bands. Chosen on round numbers rather than quantiles so the bands
#: mean the same thing between runs and can be compared over time.
PRICE_BANDS = (
    [0, 1e9, 1.5e9, 2e9, 3e9, 5e9, 10e9, float("inf")],
    ["<1B", "1-1.5B", "1.5-2B", "2-3B", "3-5B", "5-10B", "10B+"],
)

MILEAGE_BANDS = (
    [0, 1, 20_000, 50_000, 100_000, 200_000, 300_000, float("inf")],
    ["0 (zero-km)", "1-20k", "20-50k", "50-100k", "100-200k", "200-300k", "300k+"],
)


def add_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard price and mileage bands."""
    out = frame.copy()
    if "price_toman" in out.columns:
        out["price_band"] = bucket(out["price_toman"], *PRICE_BANDS)
    if "mileage_km" in out.columns:
        out["mileage_band"] = bucket(out["mileage_km"], *MILEAGE_BANDS)
    return out
