"""Missingness, and — where the data permits — *why* a value is missing.

A single "null" count conflates causes that call for different responses:

``not_displayed``
    the platform never showed the field. Nothing to fix; model around it.
``parser_failure``
    the page had it and the parser did not get it. A bug.
``snapshot_unavailable``
    no detail page was scraped for this listing at all. A coverage limit.
``not_applicable``
    the field cannot apply (battery capacity on a petrol car).
``disappeared_before_deep_scrape``
    the listing left before a detail scrape was due. Structurally unfixable, and
    it biases exactly the listings that disappear fastest.

The last one matters most: those are the short-lived listings, so treating their
missingness as random would bias every duration analysis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: Fields that only apply to some vehicles, so a null is not a gap.
CONDITIONAL_FIELDS: dict[str, str] = {
    "deep_battery_capacity_kwh": "electric vehicles only",
    "deep_all_electric_range_km": "electric vehicles only",
    "deep_installment_amount_toman": "instalment listings only",
    "deep_down_payment_toman": "instalment listings only",
    "deep_dealer_name": "dealer listings only",
    "deep_dealer_score": "dealer listings only",
    "deep_dealer_ad_count": "dealer listings only",
    "filter_exit_reason": "listings that left the search only",
    "first_missing_at": "listings that have been absent at least once",
    "confirmed_removed_at": "listings whose removal was confirmed",
    "reappeared_at": "listings that returned after an absence",
}


def field_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column non-null counts, with a cause where one can be attributed."""
    total = len(frame)
    records: list[dict[str, Any]] = []
    has_snapshot = (
        frame["snapshot_scraped_at"].notna()
        if "snapshot_scraped_at" in frame.columns
        else pd.Series([False] * total, index=frame.index)
    )
    disappeared = (
        frame["current_status"].astype(str).isin(["likely_removed", "likely_sold"])
        if "current_status" in frame.columns
        else pd.Series([False] * total, index=frame.index)
    )

    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        missing = total - non_null
        record = {
            "column": column,
            "non_null_count": non_null,
            "missing_count": missing,
            "missing_pct": round(missing / total * 100, 2) if total else None,
            "distinct_count": int(series.nunique(dropna=True)),
            "always_missing": non_null == 0,
            "near_zero_variance": int(series.nunique(dropna=True)) <= 1 and non_null > 0,
            "conditional_note": CONDITIONAL_FIELDS.get(column, ""),
        }
        record["missing_cause"] = _attribute_cause(
            column, series, has_snapshot, disappeared, record
        )
        record["modelling_suitable"] = bool(
            not record["always_missing"]
            and not record["near_zero_variance"]
            and (record["missing_pct"] or 0) < 50
        )
        records.append(record)

    out = pd.DataFrame(records).sort_values("missing_pct", ascending=False)
    return out.reset_index(drop=True)


def _attribute_cause(
    column: str,
    series: pd.Series,
    has_snapshot: pd.Series,
    disappeared: pd.Series,
    record: dict[str, Any],
) -> str:
    """Attribute the dominant cause of missingness, or say it is unattributed."""
    if record["missing_count"] == 0:
        return "complete"
    if column in CONDITIONAL_FIELDS:
        return "not_applicable"

    missing_mask = series.isna()
    # A monitoring-snapshot column missing exactly where no snapshot exists is a
    # coverage limit, not a parser bug. `deep_` columns come from a different
    # database entirely, so snapshot coverage says nothing about them.
    if column.startswith("snap_") and has_snapshot.any():
        without = int((missing_mask & ~has_snapshot).sum())
        if without and without / max(1, int(missing_mask.sum())) > 0.9:
            return "snapshot_unavailable"
    if column.startswith("deep_"):
        gone_and_missing = int((missing_mask & disappeared).sum())
        if gone_and_missing and gone_and_missing / max(1, int(missing_mask.sum())) > 0.5:
            return "disappeared_before_deep_scrape"
        return "not_scraped_or_not_displayed"
    if column.startswith("snap_"):
        return "snapshot_unavailable"
    if column.startswith("events_"):
        return "not_applicable"
    return "unattributed"


def missingness_patterns(
    frame: pd.DataFrame, columns: list[str] | None = None, *, top: int = 25
) -> pd.DataFrame:
    """Joint missingness: which combinations of fields go missing together.

    Co-missingness usually means one upstream cause, so this is the cheapest way
    to find that twelve "independent" gaps are really one failed request.
    """
    if frame.empty:
        return pd.DataFrame(columns=["pattern", "count", "share_pct", "missing_fields"])
    candidates = columns or [
        c
        for c in frame.columns
        if frame[c].isna().any() and not frame[c].isna().all() and not c.startswith("events_")
    ]
    candidates = candidates[:40]
    if not candidates:
        return pd.DataFrame(columns=["pattern", "count", "share_pct", "missing_fields"])

    mask = frame[candidates].isna()
    signature = mask.apply(lambda row: "".join("1" if v else "0" for v in row), axis=1)
    counts = signature.value_counts().head(top)
    records = []
    for pattern, count in counts.items():
        missing_fields = [c for c, flag in zip(candidates, pattern, strict=False) if flag == "1"]
        records.append(
            {
                "pattern": pattern,
                "count": int(count),
                "share_pct": round(count / len(frame) * 100, 2),
                "missing_field_count": len(missing_fields),
                "missing_fields": ", ".join(missing_fields[:12])
                + ("..." if len(missing_fields) > 12 else ""),
            }
        )
    return pd.DataFrame(records)


def missingness_by_group(
    frame: pd.DataFrame, group_column: str, value_columns: list[str], *, min_group_size: int = 10
) -> pd.DataFrame:
    """Missing share of each value column within each group.

    Used to detect *systematic* gaps: a field missing for one brand or one parser
    version is a different problem from one missing at random.
    """
    if group_column not in frame.columns:
        return pd.DataFrame()
    present = [c for c in value_columns if c in frame.columns]
    if not present:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for group, block in frame.groupby(frame[group_column].fillna("(missing)").astype(str)):
        record: dict[str, Any] = {
            "group_column": group_column,
            "group_value": group,
            "n": int(len(block)),
            "sufficient_sample": len(block) >= min_group_size,
        }
        for column in present:
            record[f"missing_pct__{column}"] = round(float(block[column].isna().mean()) * 100, 2)
        records.append(record)
    out = pd.DataFrame(records).sort_values("n", ascending=False)
    return out.reset_index(drop=True)


def unusable_fields(completeness: pd.DataFrame) -> dict[str, list[str]]:
    """Fields that cannot support analysis, split by why."""
    if completeness.empty:
        return {"always_missing": [], "constant": [], "mostly_missing": []}
    return {
        "always_missing": completeness.loc[completeness["always_missing"], "column"].tolist(),
        "constant": completeness.loc[completeness["near_zero_variance"], "column"].tolist(),
        "mostly_missing": completeness.loc[
            completeness["missing_pct"].fillna(0) >= 50, "column"
        ].tolist(),
    }


def summarise(frame: pd.DataFrame, completeness: pd.DataFrame) -> dict[str, Any]:
    unusable = unusable_fields(completeness)
    by_cause = (
        completeness.groupby("missing_cause").size().sort_values(ascending=False).to_dict()
        if not completeness.empty
        else {}
    )
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "columns_complete": int((completeness["missing_count"] == 0).sum())
        if not completeness.empty
        else 0,
        "columns_always_missing": len(unusable["always_missing"]),
        "columns_constant": len(unusable["constant"]),
        "columns_mostly_missing": len(unusable["mostly_missing"]),
        "missing_cause_counts": {str(k): int(v) for k, v in by_cause.items()},
        "unusable_fields": unusable,
        "interpretation": (
            "missing_cause distinguishes a coverage limit from a parser bug. "
            "'disappeared_before_deep_scrape' is not missing at random: those are the "
            "shortest-lived listings, so any duration analysis restricted to complete "
            "rows is biased towards listings that lasted."
        ),
    }
