"""Status transitions, filter exits and sale-evidence analysis.

Three vocabularies are reported side by side and never merged:

``current_status``       what the search showed        (observation)
``detail_availability``  what the detail page said     (observation)
``sale_label``           what the monitor inferred     (inference)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .database import ReadOnlyDatabase

#: Reasons a live listing can leave the monitored search.
FILTER_EXIT_REASONS = (
    "price_below_filter",
    "price_above_filter",
    "year_outside_filter",
    "category_changed",
    "country_classification_changed",
    "search_index_inconsistency",
)


def status_transitions(db: ReadOnlyDatabase, run_id: int) -> pd.DataFrame:
    """Observed transition counts from the immutable event log."""
    if not db.has_table("advertisement_events"):
        return pd.DataFrame(columns=["event_type", "previous_status", "new_status", "count"])
    return db.query_frame(
        "SELECT event_type, previous_status, new_status, COUNT(*) AS count"
        " FROM advertisement_events WHERE run_id <= ?"
        " GROUP BY event_type, previous_status, new_status"
        " ORDER BY COUNT(*) DESC",
        [run_id],
        label="status.transitions",
    )


def status_distribution(cross: pd.DataFrame) -> pd.DataFrame:
    """Current status counts with an explicit interpretation per status."""
    if "current_status" not in cross.columns:
        return pd.DataFrame()
    meaning = {
        "new": "first counted discovery (observation)",
        "active": "present in the latest valid run (observation)",
        "missing_once": "absent from one valid run; NOT a removal (observation)",
        "likely_removed": "absent for the confirmation threshold (observation-driven)",
        "likely_sold": "INFERENCE from removal plus supporting evidence",
        "reappeared": "returned after an absence; supersedes any prior inference",
        "reposted": "matched to a newer listing for the same vehicle",
        "active_outside_filter": "still for sale, outside the monitored search; NOT a disappearance",
        "unknown": "could not be classified",
    }
    counts = cross["current_status"].astype(str).value_counts()
    out = pd.DataFrame(
        {
            "status": counts.index.astype(str),
            "advertisements": counts.to_numpy().astype(int),
        }
    )
    out["share_pct"] = (out["advertisements"] / len(cross) * 100).round(3)
    out["meaning"] = out["status"].map(meaning).fillna("")
    out["is_disappearance"] = out["status"].isin(["likely_removed", "likely_sold"])
    return out


def filter_exit_analysis(cross: pd.DataFrame, *, min_group_size: int = 10) -> dict[str, Any]:
    """Listings that left the search while remaining for sale.

    These are the near-misses of the whole system: every one of them would have
    become a false removal, and then fed a false sale inference.
    """
    if "filter_exit_reason" not in cross.columns:
        return {"available": False, "note": "filter_exit_reason column absent"}

    exits = cross[cross["filter_exit_reason"].notna()].copy()
    total = int(len(cross))
    result: dict[str, Any] = {
        "available": True,
        "filter_exits": int(len(exits)),
        "share_of_inventory_pct": round(len(exits) / total * 100, 4) if total else 0.0,
        "reason_counts": {
            str(k): int(v) for k, v in exits["filter_exit_reason"].value_counts().items()
        },
        "reasons_possible": list(FILTER_EXIT_REASONS),
        "excluded_from_disappearance": True,
        "interpretation": (
            "A filter exit is a CENSORING event: the listing is still for sale, it "
            "simply no longer matches the monitored search. It is never counted as a "
            "disappearance or as evidence of a sale."
        ),
    }
    if exits.empty:
        result["note"] = (
            "no filter exit was recorded in this database; with few runs and a stable "
            "price floor this is expected, and is not evidence that filter exits do "
            "not occur"
        )
        return result

    if "deep_brand" in exits.columns:
        result["brands_affected"] = {
            str(k): int(v) for k, v in exits["deep_brand"].value_counts().head(10).items()
        }
    if "detail_availability" in exits.columns:
        result["detail_availability"] = {
            str(k): int(v) for k, v in exits["detail_availability"].value_counts().items()
        }
    if "price_toman" in exits.columns:
        prices = pd.to_numeric(exits["price_toman"], errors="coerce").dropna()
        result["price_at_exit"] = {
            "n": int(len(prices)),
            "median": float(prices.median()) if len(prices) else None,
            "min": float(prices.min()) if len(prices) else None,
            "max": float(prices.max()) if len(prices) else None,
        }
    return result


def filter_exit_table(cross: pd.DataFrame) -> pd.DataFrame:
    """Row-level filter-exit ledger for export."""
    columns = [
        "platform_ad_id",
        "canonical_url",
        "current_status",
        "filter_exit_reason",
        "filter_exit_at",
        "detail_availability",
        "last_detail_verdict",
        "price_toman",
        "year_jalali",
        "deep_brand",
        "deep_model",
        "first_seen_at",
        "last_seen_at",
        "first_missing_at",
        "sale_confidence",
    ]
    present = [c for c in columns if c in cross.columns]
    if "filter_exit_reason" not in cross.columns:
        return pd.DataFrame(columns=present + ["interpretation"])
    exits = cross[cross["filter_exit_reason"].notna()][present].copy()
    exits["interpretation"] = "still for sale, outside the monitored filter - NOT a disappearance"
    return exits.reset_index(drop=True)


def sale_evidence_analysis(
    cross: pd.DataFrame, db: ReadOnlyDatabase, *, min_calibration_labels: int = 40
) -> dict[str, Any]:
    """Distribution of the heuristic score, and whether it may be read as a rate.

    The score is an **ordering**, not a probability, until enough human-labelled
    outcomes exist to measure a rate per band. This function reads the calibration
    table and reports the answer rather than estimating one.
    """
    score_column = "sale_confidence" if "sale_confidence" in cross.columns else None
    result: dict[str, Any] = {
        "terminology": (
            "sale_evidence_score is a heuristic additive score in [0,1]. It is NOT a "
            "probability: 0.65 does not mean '65% of these sold'."
        )
    }
    if score_column is None:
        return {**result, "available": False, "note": "no sale evidence column"}

    scores = pd.to_numeric(cross[score_column], errors="coerce")
    bands = pd.cut(
        scores,
        bins=[-0.001, 0.4, 0.65, 0.85, 1.0],
        labels=[
            "0.00-0.39 (unknown)",
            "0.40-0.64 (possibly)",
            "0.65-0.84 (likely)",
            "0.85-1.00 (highly likely)",
        ],
    )
    result.update(
        {
            "available": True,
            "advertisements": int(len(cross)),
            "score_distribution": {
                "n": int(scores.notna().sum()),
                "zero": int((scores == 0).sum()),
                "median": round(float(scores.median()), 4) if scores.notna().any() else None,
                "max": round(float(scores.max()), 4) if scores.notna().any() else None,
                "above_0_40": int((scores >= 0.40).sum()),
                "above_0_65": int((scores >= 0.65).sum()),
            },
            "band_counts": {str(k): int(v) for k, v in bands.value_counts().sort_index().items()},
        }
    )

    if "sale_label" in cross.columns:
        result["label_counts"] = {
            str(k): int(v) for k, v in cross["sale_label"].astype(str).value_counts().items()
        }
        result["confirmed_sold_labels"] = int(
            cross["sale_label"].astype(str).eq("confirmed_sold").sum()
        )

    for dimension in ("current_status", "detail_availability"):
        if dimension in cross.columns:
            grouped = cross.groupby(cross[dimension].astype(str))[score_column]
            result[f"score_by_{dimension}"] = {
                str(name): {
                    "n": int(block.notna().sum()),
                    "median": round(float(block.median()), 4) if block.notna().any() else None,
                }
                for name, block in grouped
            }
    if "consecutive_misses" in cross.columns:
        grouped = cross.groupby(pd.to_numeric(cross["consecutive_misses"], errors="coerce"))[
            score_column
        ]
        result["score_by_consecutive_misses"] = {
            str(int(name)): {
                "n": int(block.notna().sum()),
                "median": round(float(block.median()), 4) if block.notna().any() else None,
            }
            for name, block in grouped
            if pd.notna(name)
        }

    result["calibration"] = _calibration(db, min_labels=min_calibration_labels)
    return result


def _calibration(db: ReadOnlyDatabase, *, min_labels: int) -> dict[str, Any]:
    """Read the monitor's calibration table; never interpolate a missing rate."""
    if not db.has_table("sale_validation_samples"):
        return {
            "calibrated": False,
            "empirical_sale_rate": "unavailable",
            "labelled": 0,
            "note": "no validation table in this database",
        }
    total = int(db.scalar("SELECT COUNT(*) FROM sale_validation_samples") or 0)
    decided = int(
        db.scalar(
            "SELECT COUNT(*) FROM sale_validation_samples"
            " WHERE observed_outcome IS NOT NULL AND observed_outcome <> 'unknown'"
        )
        or 0
    )
    calibrated = decided >= min_labels
    result: dict[str, Any] = {
        "sampled": total,
        "labelled_decided": decided,
        "minimum_required": min_labels,
        "calibrated": calibrated,
        "empirical_sale_rate": "unavailable" if not calibrated else None,
        "note": (
            f"{decided} decided human label(s) against a required {min_labels}. "
            "No empirical sale rate is reported, and no rate is estimated or "
            "interpolated from the score itself."
        )
        if not calibrated
        else "sufficient labels; per-band rates may be computed",
    }
    if calibrated:
        rows = db.fetchall(
            "SELECT predicted_score, observed_outcome FROM sale_validation_samples"
            " WHERE observed_outcome IS NOT NULL"
        )
        frame = pd.DataFrame(rows)
        frame["band"] = pd.cut(
            pd.to_numeric(frame["predicted_score"], errors="coerce"),
            bins=[-0.001, 0.4, 0.65, 0.85, 1.0],
            labels=["0.00-0.39", "0.40-0.64", "0.65-0.84", "0.85-1.00"],
        )
        per_band = []
        for band, block in frame.groupby("band", observed=True):
            decided_block = block[block["observed_outcome"] != "unknown"]
            if len(decided_block) < 10:
                per_band.append(
                    {
                        "band": str(band),
                        "decided": int(len(decided_block)),
                        "sale_rate": None,
                        "note": "too few decided labels for a rate",
                    }
                )
                continue
            per_band.append(
                {
                    "band": str(band),
                    "decided": int(len(decided_block)),
                    "sale_rate": round(
                        float((decided_block["observed_outcome"] == "sold").mean()), 4
                    ),
                }
            )
        result["bands"] = per_band
    return result


def sale_evidence_table(cross: pd.DataFrame) -> pd.DataFrame:
    """Export-shaped summary keeping observation and inference in separate columns."""
    if "sale_confidence" not in cross.columns:
        return pd.DataFrame()
    frame = cross.copy()
    frame["sale_evidence_band"] = pd.cut(
        pd.to_numeric(frame["sale_confidence"], errors="coerce"),
        bins=[-0.001, 0.4, 0.65, 0.85, 1.0],
        labels=["0.00-0.39", "0.40-0.64", "0.65-0.84", "0.85-1.00"],
    )
    grouped = frame.groupby(
        [frame["current_status"].astype(str), frame["sale_evidence_band"]], observed=True
    )
    out = grouped.size().reset_index(name="advertisements")
    out.columns = ["current_status_observed", "sale_evidence_band_inferred", "advertisements"]
    out["reminder"] = "score is an ordering of evidence, not a probability of sale"
    return out
