"""Inventory structure: what the monitored population actually contains."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .profiling import add_bands


def inventory_summary(
    cross: pd.DataFrame, panel: pd.DataFrame, vehicles: pd.DataFrame
) -> dict[str, Any]:
    """Headline counts. Every one of these is an observed fact."""

    def nunique(column: str) -> int | None:
        return int(cross[column].nunique(dropna=True)) if column in cross.columns else None

    def rng(column: str) -> dict[str, Any]:
        if column not in cross.columns:
            return {"min": None, "max": None}
        values = pd.to_numeric(cross[column], errors="coerce").dropna()
        if values.empty:
            return {"min": None, "max": None}
        return {"min": float(values.min()), "max": float(values.max())}

    status_counts = (
        cross["current_status"].astype(str).value_counts().to_dict()
        if "current_status" in cross.columns
        else {}
    )

    published = (
        pd.to_datetime(cross["published_at"], errors="coerce", utc=True)
        if "published_at" in cross.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )

    return {
        "advertisements": int(len(cross)),
        "vehicle_entities": int(len(vehicles)),
        "observations_all_runs": int(len(panel)),
        "runs_covered": int(panel["observation_run_id"].nunique())
        if "observation_run_id" in panel.columns
        else None,
        "unique_brands": nunique("deep_brand"),
        "unique_models": nunique("deep_model"),
        "unique_trims": nunique("deep_trim"),
        "unique_provinces": nunique("deep_province"),
        "unique_cities": nunique("deep_city"),
        "unique_seller_types": nunique("deep_seller_type"),
        "price_range_toman": rng("price_toman"),
        "mileage_range_km": rng("mileage_km"),
        "year_range_jalali": rng("year_jalali"),
        "publication_range": {
            "min": published.min().isoformat() if published.notna().any() else None,
            "max": published.max().isoformat() if published.notna().any() else None,
            "known_for": int(published.notna().sum()),
        },
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "active": int(status_counts.get("active", 0) + status_counts.get("new", 0)),
        "missing_once": int(status_counts.get("missing_once", 0)),
        "likely_removed": int(status_counts.get("likely_removed", 0)),
        "likely_sold": int(status_counts.get("likely_sold", 0)),
        "reappeared": int(status_counts.get("reappeared", 0)),
        "reposted": int(status_counts.get("reposted", 0)),
        "active_outside_filter": int(status_counts.get("active_outside_filter", 0)),
        "zero_km_listings": int(pd.to_numeric(cross.get("mileage_km"), errors="coerce").eq(0).sum())
        if "mileage_km" in cross.columns
        else None,
    }


#: (label, column) pairs broken down by default.
COMPOSITION_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("brand", "deep_brand"),
    ("model", "deep_model"),
    ("trim", "deep_trim"),
    ("production_year", "year_jalali"),
    ("body_type", "deep_body_type_fa"),
    ("body_condition", "deep_body_status"),
    ("condition_new_used", "deep_condition_new_used"),
    ("transmission", "deep_transmission"),
    ("fuel_type", "deep_fuel_type"),
    ("seller_type", "deep_seller_type"),
    ("province", "deep_province"),
    ("city", "deep_city"),
    ("price_band", "price_band"),
    ("mileage_band", "mileage_band"),
    ("body_color", "deep_body_color"),
    ("status", "current_status"),
)


def composition(cross: pd.DataFrame, *, min_group_size: int = 10, top: int = 30) -> pd.DataFrame:
    """Counts and shares across every composition dimension present."""
    frame = add_bands(cross)
    blocks: list[pd.DataFrame] = []
    total = len(frame)
    for label, column in COMPOSITION_DIMENSIONS:
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.notna().sum() == 0:
            continue
        counts = series.dropna().astype(str).value_counts().head(top)
        block = pd.DataFrame(
            {
                "dimension": label,
                "source_column": column,
                "value": counts.index.astype(str),
                "count": counts.to_numpy(),
            }
        )
        block["share_of_known_pct"] = (block["count"] / series.notna().sum() * 100).round(2)
        block["share_of_total_pct"] = (block["count"] / total * 100).round(2) if total else None
        block["known_values"] = int(series.notna().sum())
        block["coverage_pct"] = round(series.notna().mean() * 100, 2)
        block["sufficient_sample"] = block["count"] >= min_group_size
        block["sample_note"] = np.where(
            block["sufficient_sample"], "", f"insufficient_sample (< {min_group_size})"
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def brand_model_combinations(
    cross: pd.DataFrame, *, min_group_size: int = 10, top: int = 30
) -> pd.DataFrame:
    """The brand x model cross-tab, which is how the market is actually shaped."""
    if not {"deep_brand", "deep_model"} <= set(cross.columns):
        return pd.DataFrame()
    block = cross.dropna(subset=["deep_brand", "deep_model"])
    if block.empty:
        return pd.DataFrame()
    grouped = (
        block.groupby(["deep_brand", "deep_model"])
        .agg(
            listing_count=("platform_ad_id", "count"),
            median_price=("price_toman", "median"),
            median_mileage=("mileage_km", "median"),
            median_year=("year_jalali", "median"),
        )
        .reset_index()
        .sort_values("listing_count", ascending=False)
    )
    grouped["sufficient_sample"] = grouped["listing_count"] >= min_group_size
    grouped["sample_note"] = np.where(
        grouped["sufficient_sample"], "", f"insufficient_sample (< {min_group_size})"
    )
    return grouped.head(top).reset_index(drop=True)
