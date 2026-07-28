"""Aggregation from listings to physical vehicles.

An advertisement id identifies a **listing**. When a seller relists the same car
under a new id, listing-level analysis sees one disappearance plus one new
arrival; vehicle-level analysis sees one unsold car that is still looking for a
buyer. Only the second reading is right, and only the second supports a question
like "which cars sell faster?".

So durations are also computed at the vehicle grain, spanning every linked
listing:

    vehicle_first_seen_at = min(first_seen_at) across linked listings
    vehicle_last_seen_at  = max(last_seen_at)  across linked listings

A repost is therefore an **explanatory variable**, not an endpoint: the chain
ends only when the last listing in it disappears without a successor.
"""

from __future__ import annotations

from typing import Any

from .db import Database, parse_ts, utcnow
from .duration_estimation import estimate_duration
from .models import DISAPPEARED_STATUSES, AdStatus


def _entity_key(row: dict[str, Any]) -> str:
    """Group key: the shared vehicle id if one was assigned, else the listing."""
    return str(row.get("vehicle_entity_id") or row.get("platform_ad_id"))


def rebuild_vehicle_entities(db: Database) -> int:
    """Recompute the ``vehicle_entities`` table from current listing state.

    Idempotent and cheap: it is derived data, so it is rebuilt rather than
    incrementally maintained, which removes a whole class of drift bugs.
    """
    rows = db.fetchall(
        "SELECT a.id, a.platform_ad_id, a.vehicle_entity_id, a.current_status,"
        " a.first_seen_at, a.last_seen_at, a.first_missing_at, a.published_at,"
        " a.sale_confidence, a.sale_label, a.left_truncated, a.repost_parent_ad_id"
        " FROM advertisements a"
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_entity_key(row), []).append(row)

    now = utcnow()
    written = 0
    for entity_id, members in groups.items():
        summary = summarize_vehicle(members)
        db.upsert(
            "vehicle_entities",
            ["vehicle_entity_id"],
            {
                "vehicle_entity_id": entity_id,
                "first_ad_id": summary["first_ad_id"],
                "listing_count": summary["listing_count"],
                "repost_count": summary["repost_count"],
                "vehicle_first_seen_at": summary["vehicle_first_seen_at"],
                "vehicle_last_seen_at": summary["vehicle_last_seen_at"],
                "vehicle_first_missing_at": summary["vehicle_first_missing_at"],
                "vehicle_published_at": summary["vehicle_published_at"],
                "current_status": summary["current_status"],
                "sale_evidence_score": summary["sale_evidence_score"],
                "sale_label": summary["sale_label"],
                "left_truncated": summary["left_truncated"],
                "created_at": now,
                "updated_at": now,
            },
            # `created_at` stays at its original value on conflict.
            update_columns=[
                "first_ad_id",
                "listing_count",
                "repost_count",
                "vehicle_first_seen_at",
                "vehicle_last_seen_at",
                "vehicle_first_missing_at",
                "vehicle_published_at",
                "current_status",
                "sale_evidence_score",
                "sale_label",
                "left_truncated",
                "updated_at",
            ],
        )
        written += 1
    return written


def summarize_vehicle(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse the listings of one vehicle into a single record.

    The vehicle's status is the status of its **latest** listing: an earlier
    listing marked ``reposted`` describes a chain link, not the car's fate. And a
    vehicle counts as left-truncated if *its earliest* listing was, because that
    is the one that determines whether the beginning of its life was observed.
    """

    def key(row: dict[str, Any]) -> Any:
        return parse_ts(row.get("first_seen_at")) or parse_ts(row.get("last_seen_at"))

    ordered = sorted(members, key=lambda r: (key(r) is None, key(r)))
    first = ordered[0]
    last = ordered[-1]

    first_seens = [parse_ts(r.get("first_seen_at")) for r in members]
    last_seens = [parse_ts(r.get("last_seen_at")) for r in members]
    firsts = [d for d in first_seens if d]
    lasts = [d for d in last_seens if d]

    # Only the final listing's absence ends the vehicle's observed life; an
    # earlier listing's first_missing_at is just where the chain handed over.
    final_missing = parse_ts(last.get("first_missing_at"))
    status = AdStatus(str(last.get("current_status") or AdStatus.UNKNOWN))
    if status is AdStatus.REPOSTED:
        # The chain continued past this listing but no successor is recorded --
        # report the honest unknown rather than implying the car went away.
        status = AdStatus.UNKNOWN

    published = [parse_ts(r.get("published_at")) for r in members]
    published_first = min((d for d in published if d), default=None)

    return {
        "vehicle_entity_id": _entity_key(first),
        "first_ad_id": first.get("id"),
        "listing_count": len(members),
        "repost_count": max(0, len(members) - 1),
        "vehicle_first_seen_at": min(firsts) if firsts else None,
        "vehicle_last_seen_at": max(lasts) if lasts else None,
        "vehicle_first_missing_at": final_missing if status in DISAPPEARED_STATUSES else None,
        "vehicle_published_at": published_first,
        "current_status": str(status),
        # Inference carried at the vehicle grain so a chain of three listings is
        # one candidate sale, not three.
        "sale_evidence_score": float(last.get("sale_confidence") or 0.0),
        "sale_label": str(last.get("sale_label") or "unknown"),
        "left_truncated": bool(first.get("left_truncated", 1)),
    }


def vehicle_durations(db: Database, *, now: Any = None) -> list[dict[str, Any]]:
    """Duration bounds at the vehicle grain, spanning linked reposts.

    Compare with :func:`bama_monitor.analytics.time_to_disappearance`, which
    measures listings. For a vehicle relisted twice the listing view reports two
    short lives; this reports one long one, which is what actually happened.
    """
    rows = db.fetchall(
        "SELECT v.*, (SELECT COUNT(*) FROM advertisements a"
        "  WHERE a.vehicle_entity_id = v.vehicle_entity_id) AS linked_listings"
        " FROM vehicle_entities v"
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        estimate = estimate_duration(
            first_seen_at=parse_ts(row.get("vehicle_first_seen_at")),
            last_seen_at=parse_ts(row.get("vehicle_last_seen_at")),
            first_missing_at=parse_ts(row.get("vehicle_first_missing_at")),
            status=str(row.get("current_status") or AdStatus.UNKNOWN),
            now=now,
        )
        out.append(
            {
                "vehicle_entity_id": row["vehicle_entity_id"],
                "listing_count": row.get("listing_count"),
                "repost_count": row.get("repost_count"),
                "vehicle_published_at": row.get("vehicle_published_at"),
                "vehicle_first_seen_at": row.get("vehicle_first_seen_at"),
                "vehicle_last_seen_at": row.get("vehicle_last_seen_at"),
                "vehicle_first_missing_at": row.get("vehicle_first_missing_at"),
                **estimate.as_days(),
                "observation_interval_hours": estimate.observation_interval_hours,
                "right_censored": int(estimate.censored),
                "status": row.get("current_status"),
                "sale_evidence_score": row.get("sale_evidence_score"),
                "sale_label": row.get("sale_label"),
                "left_truncated": int(bool(row.get("left_truncated", 1))),
                # A repost is a covariate, not an outcome: the vehicle stayed on
                # the market, which is why the duration spans the whole chain.
                "was_reposted": int((row.get("repost_count") or 0) > 0),
            }
        )
    return out
