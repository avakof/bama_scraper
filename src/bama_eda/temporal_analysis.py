"""Publication timing, entry delay and left truncation.

Five instants are kept strictly apart, because collapsing any two of them is how a
time-on-market number becomes wrong:

``published_at``        when the seller published it
``first_seen_at``       when monitoring first observed it
``last_seen_at``        the final positive observation
``first_missing_at``    the first valid run in which it was absent
``confirmed_removed_at``when absence reached the confirmation threshold

``first_seen_at`` is **not** a publication date. For any listing that predates
monitoring the gap between the two is unknown, which is what left truncation means
and why most of a young installation cannot support a duration claim at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bama_monitor.db import parse_ts


def publication_coverage(cross: pd.DataFrame) -> dict[str, Any]:
    """How much of the population can support a time-on-market statement."""
    total = int(len(cross))
    if total == 0:
        return {"advertisements_total": 0, "note": "empty cross-section"}

    published = cross.get("published_at")
    reliable = cross.get("published_at_reliable")
    truncated = cross.get("left_truncated")
    eligible = cross.get("eligible_for_duration_ranking")
    delay = pd.to_numeric(cross.get("entry_delay_seconds"), errors="coerce")

    def count_true(series: Any) -> int:
        if series is None:
            return 0
        return int(pd.Series(series).fillna(False).astype(bool).sum())

    known = int(published.notna().sum()) if published is not None else 0
    eligible_n = count_true(eligible)
    result = {
        "advertisements_total": total,
        "publication_timestamp_known": known,
        "publication_timestamp_known_pct": round(known / total * 100, 3),
        "publication_timestamp_reliable": count_true(reliable),
        "left_truncated": count_true(truncated),
        "left_truncated_pct": round(count_true(truncated) / total * 100, 3),
        "eligible_for_duration_ranking": eligible_n,
        "eligible_share_pct": round(eligible_n / total * 100, 3),
        "entry_delay_days": {
            "median": round(float(delay.median() / 86400), 4) if delay.notna().any() else None,
            "p90": round(float(delay.quantile(0.9) / 86400), 4) if delay.notna().any() else None,
            "max": round(float(delay.max() / 86400), 4) if delay.notna().any() else None,
        },
    }
    if "published_at_source" in cross.columns:
        result["publication_source_counts"] = {
            str(k): int(v)
            for k, v in cross["published_at_source"].fillna("(none)").value_counts().items()
        }
    result["interpretation"] = (
        f"{eligible_n} of {total} advertisements ({result['eligible_share_pct']}%) were first "
        "observed close enough to publication for their full market life to have been seen. "
        "Only that subset can support an estimated market duration; the rest are "
        "left-truncated and their observed durations are lower bounds."
    )
    return result


def duration_ranking_feasibility(cross: pd.DataFrame, *, min_eligible: int = 30) -> dict[str, Any]:
    """Decide, before ranking anything, whether a ranking would mean anything.

    Returns a refusal rather than a small, confident-looking table. A top-10 drawn
    from six eligible listings is a list of six listings.
    """
    eligible = (
        pd.Series(cross.get("eligible_for_duration_ranking")).fillna(False).astype(bool)
        if "eligible_for_duration_ranking" in cross.columns
        else pd.Series([], dtype=bool)
    )
    n = int(eligible.sum())
    disappeared = (
        cross.loc[eligible, "current_status"].astype(str).isin(["likely_removed", "likely_sold"])
        if n and "current_status" in cross.columns
        else pd.Series([], dtype=bool)
    )
    events = int(disappeared.sum())

    feasible = n >= min_eligible and events >= 1
    return {
        "eligible_advertisements": n,
        "eligible_with_observed_disappearance": events,
        "minimum_required": min_eligible,
        "feasible": feasible,
        "verdict": "duration_ranking_available"
        if feasible
        else "insufficient_data_for_duration_ranking",
        "note": (
            f"only {n} advertisement(s) are eligible and {events} of them have an observed "
            f"disappearance, against a minimum of {min_eligible}. No ranking is produced: "
            "a ranking of this many rows would be a list, not a ranking."
        )
        if not feasible
        else f"{n} eligible advertisements with {events} observed disappearances",
    }


def publication_volume(cross: pd.DataFrame, *, freq: str = "D") -> pd.DataFrame:
    """Listings published per calendar period, among those with a known date."""
    if "published_at" not in cross.columns:
        return pd.DataFrame(columns=["period", "advertisements"])
    stamps = pd.to_datetime(cross["published_at"], errors="coerce", utc=True).dropna()
    if stamps.empty:
        return pd.DataFrame(columns=["period", "advertisements"])
    counts = stamps.dt.floor(freq).value_counts().sort_index()
    return pd.DataFrame(
        {
            "period": [t.isoformat() for t in counts.index],
            "advertisements": counts.to_numpy().astype(int),
        }
    )


def run_timeline(panel: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run inventory counts, so churn is visible run by run."""
    if panel.empty or "observation_run_id" not in panel.columns:
        return pd.DataFrame()
    grouped = panel.groupby("observation_run_id")
    out = pd.DataFrame(
        {
            "run_id": grouped.size().index,
            "observations": grouped.size().to_numpy(),
            "seen": grouped["obs_was_seen"]
            .apply(lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()))
            .to_numpy()
            if "obs_was_seen" in panel.columns
            else 0,
        }
    )
    if not runs.empty and "id" in runs.columns:
        # Filter against the RENAMED frame: `run_id` does not exist before the
        # rename, so checking `runs.columns` silently drops the merge key.
        renamed = runs.rename(columns={"id": "run_id"})
        wanted = [
            c
            for c in (
                "run_id",
                "scheduled_for",
                "status",
                "discovered_count",
                "new_count",
                "missing_count",
                "removed_count",
                "reappeared_count",
            )
            if c in renamed.columns
        ]
        out = out.merge(renamed[wanted], on="run_id", how="left")
    out["absent"] = out["observations"] - out["seen"]
    return out


def observation_intervals(panel: pd.DataFrame) -> dict[str, Any]:
    """Spacing between consecutive observations of the same advertisement.

    This interval is the resolution limit on every duration in the dataset: a
    disappearance can only be located to within one of these gaps.
    """
    if panel.empty or "obs_observed_at" not in panel.columns:
        return {"available": False}
    frame = panel[["advertisement_id", "obs_observed_at"]].copy()
    frame["observed_at"] = frame["obs_observed_at"].map(parse_ts)
    frame = frame.dropna(subset=["observed_at"]).sort_values(["advertisement_id", "observed_at"])
    frame["gap_hours"] = (
        frame.groupby("advertisement_id")["observed_at"].diff().dt.total_seconds() / 3600
    )
    gaps = frame["gap_hours"].dropna()
    if gaps.empty:
        return {"available": False, "note": "fewer than two observations per advertisement"}
    return {
        "available": True,
        "median_interval_hours": round(float(gaps.median()), 3),
        "min_interval_hours": round(float(gaps.min()), 3),
        "max_interval_hours": round(float(gaps.max()), 3),
        "p90_interval_hours": round(float(gaps.quantile(0.9)), 3),
        "intervals_measured": int(len(gaps)),
        "interpretation": (
            "every disappearance time in this dataset is known only to within one "
            "observation interval; the midpoint of that interval is an estimate, never "
            "a time of sale"
        ),
    }


def entry_delay_distribution(cross: pd.DataFrame) -> pd.DataFrame:
    """How long after publication each listing entered observation."""
    if "entry_delay_seconds" not in cross.columns:
        return pd.DataFrame(columns=["band", "advertisements"])
    days = pd.to_numeric(cross["entry_delay_seconds"], errors="coerce") / 86400
    if days.notna().sum() == 0:
        return pd.DataFrame(columns=["band", "advertisements"])
    bands = pd.cut(
        days,
        bins=[-np.inf, 0.5, 1.5, 3, 7, 14, 30, np.inf],
        labels=["<12h", "12-36h", "1.5-3d", "3-7d", "7-14d", "14-30d", "30d+"],
    )
    counts = bands.value_counts().sort_index()
    return pd.DataFrame(
        {
            "band": counts.index.astype(str),
            "advertisements": counts.to_numpy().astype(int),
            "eligible_band": [b in ("<12h", "12-36h") for b in counts.index.astype(str)],
        }
    )
