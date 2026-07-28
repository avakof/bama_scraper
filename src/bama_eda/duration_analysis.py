"""Time to **disappearance**, with censoring handled explicitly.

The event modelled here is *the listing left the observed inventory*. It is not a
sale. A listing can vanish because it sold, expired, was withdrawn, was moderated,
was edited outside the search filter, or was relisted under a new id — and this
dataset cannot distinguish those.

Every observation is placed in exactly one censoring class, because they need
different handling:

``fully_observed``          entry and disappearance both observed
``interval_censored``       disappeared somewhere inside a known interval
``right_censored``          still present when observation ended
``left_truncated``          already present when observation began (delayed entry)
``active_outside_filter``   left the search, not the market: a censoring event
``reappeared`` / ``reposted`` / ``unknown``

The Kaplan-Meier estimator here supports **delayed entry** through a risk set that
counts a subject only from its entry time, which is the correct handling for
left-truncated data. Where the eligible sample is too small, it refuses to
estimate rather than drawing a confident-looking step function over six points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import CensoringClass

#: Statuses that mean the listing left the observed inventory.
DISAPPEARED_STATUSES = ("likely_removed", "likely_sold")
#: Statuses that mean it is still there, or came back.
PRESENT_STATUSES = ("active", "new", "reappeared", "missing_once")


def classify_censoring(row: pd.Series) -> str:
    """Assign one censoring class per advertisement.

    Order matters. A listing that left the filter is classified as such even if it
    also carries an absence, because the filter exit *explains* the absence and
    counting it as a disappearance is the false-removal error.
    """
    status = str(row.get("current_status") or "unknown")
    if status == "active_outside_filter" or pd.notna(row.get("filter_exit_reason")):
        return str(CensoringClass.ACTIVE_OUTSIDE_FILTER)
    if status == "reposted":
        return str(CensoringClass.REPOSTED)
    if status == "reappeared":
        return str(CensoringClass.REAPPEARED)

    truncated = bool(row.get("left_truncated", True))
    if status in DISAPPEARED_STATUSES:
        if truncated:
            # Disappearance observed, but the start of life was not: still an event,
            # and still requiring delayed-entry handling.
            return str(CensoringClass.LEFT_TRUNCATED)
        return str(CensoringClass.INTERVAL_CENSORED)
    if status in PRESENT_STATUSES:
        return str(CensoringClass.RIGHT_CENSORED)
    return str(CensoringClass.UNKNOWN)


def build_duration_table(cross: pd.DataFrame, *, now: Any = None) -> pd.DataFrame:
    """One row per advertisement with both duration bases and a censoring class."""
    if cross.empty:
        return pd.DataFrame()

    frame = cross.copy()
    frame["censoring_class"] = frame.apply(classify_censoring, axis=1)

    first_seen = pd.to_datetime(frame.get("first_seen_at"), errors="coerce", utc=True)
    last_seen = pd.to_datetime(frame.get("last_seen_at"), errors="coerce", utc=True)
    first_missing = pd.to_datetime(frame.get("first_missing_at"), errors="coerce", utc=True)
    published = pd.to_datetime(frame.get("published_at"), errors="coerce", utc=True)

    day = np.timedelta64(1, "D")
    frame["observed_monitoring_duration"] = ((last_seen - first_seen) / day).round(4)
    # Upper bound only where the listing really left: an absence that later
    # reversed is history, not an end.
    ended = frame["current_status"].astype(str).isin(DISAPPEARED_STATUSES)
    frame["minimum_disappearance_duration"] = frame["observed_monitoring_duration"]
    frame["maximum_disappearance_duration"] = np.where(
        ended, ((first_missing - first_seen) / day).round(4), np.nan
    )
    frame["estimated_disappearance_duration"] = np.where(
        ended,
        (frame["minimum_disappearance_duration"] + frame["maximum_disappearance_duration"]) / 2,
        np.nan,
    )
    frame["observation_interval_hours"] = np.where(
        ended, ((first_missing - last_seen) / np.timedelta64(1, "h")).round(3), np.nan
    )

    eligible = (
        pd.Series(frame.get("eligible_for_duration_ranking")).fillna(False).astype(bool)
        if "eligible_for_duration_ranking" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    # Market duration is measured from publication and only where the whole life
    # was observed; everywhere else it stays NaN by design.
    frame["estimated_market_duration"] = np.where(
        eligible & published.notna(), ((last_seen - published) / day).round(4), np.nan
    )
    frame["event_observed"] = ended.astype(int)
    frame["event_type"] = np.where(
        ended,
        "disappearance_observed",
        np.where(
            frame["censoring_class"] == str(CensoringClass.ACTIVE_OUTSIDE_FILTER),
            "left_search_filter",
            np.where(
                frame["censoring_class"] == str(CensoringClass.REPOSTED),
                "reposted",
                "still_present_or_returned",
            ),
        ),
    )
    frame["right_censored"] = (~ended).astype(int)

    keep = [
        "platform_ad_id",
        "vehicle_entity_id",
        "current_status",
        "censoring_class",
        "event_observed",
        "event_type",
        "right_censored",
        "left_truncated",
        "eligible_for_duration_ranking",
        "published_at",
        "first_seen_at",
        "last_seen_at",
        "first_missing_at",
        "observed_monitoring_duration",
        "estimated_market_duration",
        "minimum_disappearance_duration",
        "maximum_disappearance_duration",
        "estimated_disappearance_duration",
        "observation_interval_hours",
        "entry_delay_seconds",
        "sale_confidence",
        "sale_label",
        "detail_availability",
        "filter_exit_reason",
        "deep_brand",
        "deep_model",
        "price_toman",
        "mileage_km",
        "year_jalali",
    ]
    return frame[[c for c in keep if c in frame.columns]].copy()


def duration_summary(durations: pd.DataFrame) -> dict[str, Any]:
    """Counts per censoring class plus distributions, kept separate."""
    if durations.empty:
        return {"available": False}

    classes = durations["censoring_class"].value_counts().to_dict()
    observed = durations[durations["event_observed"] == 1]
    eligible = durations[
        pd.Series(durations.get("eligible_for_duration_ranking")).fillna(False).astype(bool)
    ]

    def describe(series: pd.Series) -> dict[str, Any]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return {"n": 0}
        return {
            "n": int(len(values)),
            "median_days": round(float(values.median()), 4),
            "p25_days": round(float(values.quantile(0.25)), 4),
            "p75_days": round(float(values.quantile(0.75)), 4),
            "min_days": round(float(values.min()), 4),
            "max_days": round(float(values.max()), 4),
        }

    return {
        "available": True,
        "advertisements": int(len(durations)),
        "censoring_classes": {str(k): int(v) for k, v in classes.items()},
        "observed_disappearances": int(len(observed)),
        "right_censored": int(durations["right_censored"].sum()),
        "left_truncated": int(
            pd.Series(durations.get("left_truncated")).fillna(False).astype(bool).sum()
        ),
        "filter_exits": int(
            (durations["censoring_class"] == str(CensoringClass.ACTIVE_OUTSIDE_FILTER)).sum()
        ),
        "reappeared": int((durations["censoring_class"] == str(CensoringClass.REAPPEARED)).sum()),
        "reposted": int((durations["censoring_class"] == str(CensoringClass.REPOSTED)).sum()),
        "observed_monitoring_duration": describe(durations["observed_monitoring_duration"]),
        "estimated_market_duration_eligible_only": describe(
            eligible["estimated_market_duration"] if not eligible.empty else pd.Series(dtype=float)
        ),
        "disappearance_interval_hours": describe(
            observed["observation_interval_hours"]
            if "observation_interval_hours" in observed
            else pd.Series(dtype=float)
        ),
        "interpretation": (
            "The event is DISAPPEARANCE FROM THE OBSERVED INVENTORY, not a confirmed "
            "sale. Filter exits are censoring events, not disappearances. "
            "estimated_market_duration is populated only for advertisements whose full "
            "market life was observed."
        ),
    }


def check_bound_ordering(durations: pd.DataFrame) -> dict[str, Any]:
    """min <= estimate <= max, verified rather than assumed."""
    if durations.empty:
        return {"checked": 0, "violations": 0, "ok": True}
    low = pd.to_numeric(durations.get("minimum_disappearance_duration"), errors="coerce")
    est = pd.to_numeric(durations.get("estimated_disappearance_duration"), errors="coerce")
    high = pd.to_numeric(durations.get("maximum_disappearance_duration"), errors="coerce")
    complete = low.notna() & est.notna() & high.notna()
    violations = complete & ~((low <= est) & (est <= high))
    return {
        "checked": int(complete.sum()),
        "violations": int(violations.sum()),
        "ok": bool(violations.sum() == 0),
        "rule": "minimum <= estimated <= maximum",
    }


# ---------------------------------------------------------------------------
# Kaplan-Meier with delayed entry
# ---------------------------------------------------------------------------


def kaplan_meier(
    durations: pd.DataFrame,
    *,
    entry_column: str | None = "entry_time_days",
    duration_column: str = "observed_monitoring_duration",
    event_column: str = "event_observed",
    min_subjects: int = 30,
    min_events: int = 10,
) -> dict[str, Any]:
    """Kaplan-Meier survival for time to disappearance, with delayed entry.

    The risk set at time ``t`` counts subjects with ``entry < t <= exit``, which is
    what makes left-truncated observations admissible: a listing already three
    weeks old when monitoring began contributes to the risk set only from week
    three onward, instead of pretending it was born when we first saw it.

    Refuses to estimate below the configured thresholds. Greenwood's formula gives
    the variance; the confidence band is on the log-log scale so it stays inside
    [0, 1].
    """
    if durations.empty:
        return {"available": False, "reason": "no rows"}

    frame = durations.copy()
    time = pd.to_numeric(frame.get(duration_column), errors="coerce")
    event = pd.to_numeric(frame.get(event_column), errors="coerce").fillna(0).astype(int)
    entry = (
        pd.to_numeric(frame.get(entry_column), errors="coerce").fillna(0.0)
        if entry_column and entry_column in frame.columns
        else pd.Series(0.0, index=frame.index)
    )

    usable = time.notna() & (time >= 0)
    time, event, entry = time[usable], event[usable], entry[usable]
    n_subjects, n_events = int(len(time)), int(event.sum())

    if n_subjects < min_subjects or n_events < min_events:
        return {
            "available": False,
            "reason": "insufficient_sample",
            "subjects": n_subjects,
            "events": n_events,
            "minimum_subjects": min_subjects,
            "minimum_events": min_events,
            "note": (
                f"{n_subjects} subject(s) and {n_events} event(s) against a required "
                f"{min_subjects}/{min_events}. No curve is estimated: a Kaplan-Meier "
                "curve over this many events is a step function with confidence bands "
                "spanning almost the whole probability range."
            ),
        }

    event_times = np.sort(np.unique(time[event == 1].to_numpy()))
    survival: float = 1.0
    times: list[float] = []
    at_risk_out: list[int] = []
    events_out: list[int] = []
    survival_out: list[float] = []
    variance_sum = 0.0
    lower: list[float] = []
    upper: list[float] = []

    for t in event_times:
        # Delayed entry: at risk means entered strictly before t and not yet out.
        at_risk = int(((entry < t) & (time >= t)).sum())
        deaths = int(((time == t) & (event == 1)).sum())
        if at_risk <= 0:
            continue
        survival *= 1.0 - deaths / at_risk
        if at_risk > deaths:
            variance_sum += deaths / (at_risk * (at_risk - deaths))
        times.append(float(t))
        at_risk_out.append(at_risk)
        events_out.append(deaths)
        # Recorded here, from the same accumulation the risk set drives. Computing
        # it a second time downstream would be two sources of truth for one curve.
        survival_out.append(float(survival))

        # Greenwood variance, log-log transformed so the band cannot leave [0,1].
        if 0.0 < survival < 1.0 and variance_sum > 0:
            se_loglog = np.sqrt(variance_sum) / abs(np.log(survival))
            factor = np.exp(1.96 * se_loglog)
            lower.append(float(np.clip(survival**factor, 0.0, 1.0)))
            upper.append(float(np.clip(survival ** (1 / factor), 0.0, 1.0)))
        else:
            lower.append(float(survival))
            upper.append(float(survival))

    curve = pd.DataFrame(
        {
            "time_days": times,
            "at_risk": at_risk_out,
            "events": events_out,
            "survival": survival_out,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    )
    median = None
    below = curve[curve["survival"] <= 0.5]
    if not below.empty:
        median = float(below.iloc[0]["time_days"])

    return {
        "available": True,
        "subjects": n_subjects,
        "events": n_events,
        "curve": curve,
        "median_time_to_disappearance_days": median,
        "delayed_entry_applied": bool((entry > 0).any()),
        "event_definition": "disappearance from the observed inventory, NOT a confirmed sale",
        "caveat": (
            "Filter exits and reposts are censored, not counted as events. "
            "Left-truncated subjects enter the risk set at their entry time."
        ),
    }


def survival_by_group(
    durations: pd.DataFrame,
    group_column: str,
    *,
    min_subjects: int = 30,
    min_events: int = 10,
) -> dict[str, Any]:
    """Per-group curves, only for groups that clear both thresholds.

    Groups that do not clear them are listed with their counts rather than being
    silently dropped, so the reader can see what was not comparable.
    """
    if durations.empty or group_column not in durations.columns:
        return {"available": False, "reason": "column absent"}

    curves: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    for name, block in durations.groupby(durations[group_column].astype(str)):
        result = kaplan_meier(block, min_subjects=min_subjects, min_events=min_events)
        if result.get("available"):
            curves[str(name)] = result
        else:
            skipped.append(
                {
                    "group": str(name),
                    "subjects": result.get("subjects", int(len(block))),
                    "events": result.get("events", 0),
                    "reason": result.get("reason", "insufficient_sample"),
                }
            )
    return {
        "available": bool(curves),
        "group_column": group_column,
        "curves": curves,
        "groups_compared": len(curves),
        "groups_skipped": skipped,
        "note": (
            f"{len(curves)} group(s) met the {min_subjects}-subject / {min_events}-event "
            f"threshold; {len(skipped)} did not and are not compared."
        ),
    }
