"""Analytics queries.

Implemented as query functions rather than materialized views so the same code
runs on PostgreSQL and SQLite. ``create_views`` additionally installs plain SQL
views for ad-hoc exploration where the dialects agree.

Terminology is enforced in the column names: ``disappeared`` and ``removed`` are
observations, ``likely_sold`` is an inference, and no query produces a column that
merges them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .db import Database, parse_ts, utcnow
from .duration_estimation import estimate_duration, observed_vs_market, survival_row
from .models import AdStatus, RunHealth

# Views that are valid on both dialects.
_VIEWS: dict[str, str] = {
    "v_daily_inventory": """
        SELECT
            r.id                AS run_id,
            r.scheduled_for     AS run_date,
            r.status            AS run_status,
            r.discovered_count  AS total_discovered,
            r.new_count         AS new_today,
            r.active_count      AS active_today,
            r.missing_count     AS missing_today,
            r.removed_count     AS likely_removed_today,
            r.reappeared_count  AS reappeared_today,
            r.reposted_count    AS reposted_today,
            r.likely_sold_count AS likely_sold_today
        FROM monitoring_runs r
        WHERE r.status = 'valid'
    """,
    "v_advertisement_current": """
        SELECT
            a.id,
            a.platform_ad_id,
            a.canonical_url,
            a.current_status,
            a.consecutive_misses,
            a.total_seen_runs,
            a.total_missing_runs,
            a.first_seen_at,
            a.last_seen_at,
            a.first_missing_at,
            a.confirmed_removed_at,
            a.reappeared_at,
            a.sale_confidence,
            a.sale_label,
            a.vehicle_entity_id,
            a.repost_parent_ad_id,
            s.brand, s.model, s.trim, s.year,
            s.price_normalized, s.mileage_normalized,
            s.seller_type, s.seller_name, s.city, s.province, s.body_condition
        FROM advertisements a
        LEFT JOIN advertisement_snapshots s ON s.id = a.latest_snapshot_id
    """,
}


def create_views(db: Database) -> list[str]:
    """(Re)create the exploration views. Safe to call repeatedly."""
    created: list[str] = []
    for name, body in _VIEWS.items():
        db.execute(f"DROP VIEW IF EXISTS {name}")
        db.execute(f"CREATE VIEW {name} AS {body}")
        created.append(name)
    return created


# ---------------------------------------------------------------------------
# Daily inventory
# ---------------------------------------------------------------------------


def daily_inventory(db: Database, *, limit: int = 90) -> list[dict[str, Any]]:
    """One row per valid run: what the inventory did that day."""
    return db.fetchall(
        "SELECT scheduled_for AS date, discovered_count AS total_active,"
        " new_count AS new_today, missing_count AS missing_today,"
        " removed_count AS likely_removed_today, reappeared_count AS reappeared_today,"
        " reposted_count AS reposted_today, likely_sold_count AS likely_sold_today,"
        " status AS run_status, id AS run_id"
        " FROM monitoring_runs WHERE status=? ORDER BY scheduled_for DESC LIMIT ?",
        [str(RunHealth.VALID), limit],
    )


# ---------------------------------------------------------------------------
# Time to removal
# ---------------------------------------------------------------------------


def time_to_disappearance(
    db: Database, *, min_confidence: float | None = None, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Per-advertisement duration bounds, measured two ways.

    The metric is deliberately named *disappearance*, not *sale*: what is measured
    is the interval in which the listing left the observed inventory. Why it left
    is a separate, inferred question.

    Each row carries both duration concepts, never merged:

    ``observed_monitoring_*``
        measured from ``first_seen_at``. Defined for every listing, but for one
        that predates monitoring it is only a lower bound on time on market.
    ``estimated_market_*``
        measured from ``published_at``. Populated only when publication time is
        reliable *and* the listing entered observation soon enough after it that
        its whole life was seen; ``None`` otherwise, so a partial observation can
        never be mistaken for a complete one.
    """
    rows = db.fetchall(
        "SELECT a.id, a.platform_ad_id, a.canonical_url, a.current_status,"
        " a.first_seen_at, a.last_seen_at, a.first_missing_at, a.confirmed_removed_at,"
        " a.sale_confidence, a.sale_label, a.vehicle_entity_id, a.published_at,"
        " a.published_at_source, a.published_at_reliable, a.left_truncated,"
        " a.eligible_for_duration_ranking, a.entry_delay_seconds, a.detail_availability,"
        " a.filter_exit_reason,"
        " s.brand, s.model, s.year, s.price_normalized, s.mileage_normalized, s.seller_type"
        " FROM advertisements a"
        " LEFT JOIN advertisement_snapshots s ON s.id = a.latest_snapshot_id"
    )
    reference = now or utcnow()
    out: list[dict[str, Any]] = []
    for row in rows:
        score = float(row.get("sale_confidence") or 0.0)
        if min_confidence is not None and score < min_confidence:
            continue
        estimate = estimate_duration(
            first_seen_at=parse_ts(row.get("first_seen_at")),
            last_seen_at=parse_ts(row.get("last_seen_at")),
            first_missing_at=parse_ts(row.get("first_missing_at")),
            status=str(row["current_status"]),
            now=reference,
        )
        out.append(
            {
                "advertisement_id": row["platform_ad_id"],
                "vehicle_entity_id": row.get("vehicle_entity_id"),
                "canonical_url": row["canonical_url"],
                "brand": row.get("brand"),
                "model": row.get("model"),
                "year": row.get("year"),
                "price": row.get("price_normalized"),
                "mileage": row.get("mileage_normalized"),
                "published_at": row.get("published_at"),
                "first_seen_at": row.get("first_seen_at"),
                "last_seen_at": row.get("last_seen_at"),
                "first_missing_at": row.get("first_missing_at"),
                **estimate.as_days(),
                "observation_interval_hours": estimate.observation_interval_hours,
                "right_censored": estimate.censored,
                **observed_vs_market(row, now=reference),
                # Heuristic evidence score in [0,1]. NOT a calibrated probability.
                "sale_evidence_score": score,
                "sale_label": row.get("sale_label"),
                "status": row["current_status"],
                "detail_availability": row.get("detail_availability"),
                "filter_exit_reason": row.get("filter_exit_reason"),
            }
        )
    return out


#: Retained under the previous name so existing callers keep working; the metric
#: it computes is a time to *disappearance*.
time_to_removal = time_to_disappearance


def fastest_disappearing(
    db: Database,
    *,
    min_confidence: float = 0.65,
    limit: int = 50,
    require_full_observation: bool = True,
) -> list[dict[str, Any]]:
    """Shortest-lived listings, ranked only among comparably observed ones.

    Two filters, for two different reasons:

    * the evidence floor keeps out listings that merely dropped out of the search
      for a day, which is the opposite of the intent;
    * ``require_full_observation`` keeps out **left-truncated** listings. A listing
      that was already on the market when monitoring began has an unknown earlier
      life, so its observed duration is a lower bound. Ranking it against a listing
      whose whole life was watched would make the long-standing one look fast.

    Listings excluded by the second filter are not lost -- see
    :func:`left_truncated_disappearances`, which reports them separately.
    """
    rows = [
        r
        for r in time_to_disappearance(db, min_confidence=min_confidence)
        if r["estimated_active_days"] is not None
        and (not require_full_observation or r["eligible_for_duration_ranking"])
    ]
    rows.sort(key=lambda r: (r["estimated_active_days"], -r["sale_evidence_score"]))
    return rows[:limit]


def left_truncated_disappearances(
    db: Database, *, min_confidence: float = 0.65, limit: int = 200
) -> list[dict[str, Any]]:
    """Disappeared listings whose earlier life was never observed.

    Reported separately rather than dropped, because "excluded from the ranking"
    and "did not happen" are different facts and a silent exclusion reads as the
    second one.
    """
    rows = [
        r
        for r in time_to_disappearance(db, min_confidence=min_confidence)
        if r["estimated_active_days"] is not None and not r["eligible_for_duration_ranking"]
    ]
    for row in rows:
        row["excluded_from_ranking_because"] = (
            "publication time unknown or unreliable"
            if not row["published_at_reliable"]
            else f"already active {row['entry_delay_days']} days before first observation"
        )
    rows.sort(key=lambda r: r["estimated_active_days"])
    return rows[:limit]


def filter_exit_report(db: Database) -> list[dict[str, Any]]:
    """Listings that left the monitored search while remaining for sale.

    This is the false-removal ledger. Every row here is an absence that would
    otherwise have marched to ``likely_removed`` and then fed a sale inference,
    on evidence that says the car is still available.
    """
    rows = db.fetchall(
        "SELECT a.platform_ad_id, a.canonical_url, a.current_status, a.filter_exit_reason,"
        " a.filter_exit_at, a.detail_availability, a.last_detail_verdict,"
        " a.first_seen_at, a.last_seen_at, a.first_missing_at, a.sale_confidence,"
        " s.brand, s.model, s.year, s.price_normalized"
        " FROM advertisements a"
        " LEFT JOIN advertisement_snapshots s ON s.id = a.latest_snapshot_id"
        " WHERE a.filter_exit_reason IS NOT NULL"
        " ORDER BY a.filter_exit_at DESC"
    )
    return [
        {
            "advertisement_id": r["platform_ad_id"],
            "canonical_url": r["canonical_url"],
            "status": r["current_status"],
            "filter_exit_reason": r["filter_exit_reason"],
            "filter_exit_at": r["filter_exit_at"],
            "detail_availability": r["detail_availability"],
            "detail_verdict": r["last_detail_verdict"],
            "brand": r.get("brand"),
            "model": r.get("model"),
            "year": r.get("year"),
            "price_at_exit": r.get("price_normalized"),
            "first_seen_at": r["first_seen_at"],
            "last_seen_at": r["last_seen_at"],
            "first_missing_at": r["first_missing_at"],
            "sale_evidence_score": r.get("sale_confidence"),
            "interpretation": (
                "still for sale, outside the monitored filter -- NOT a disappearance"
            ),
        }
        for r in rows
    ]


def truncation_summary(db: Database) -> dict[str, Any]:
    """How much of the corpus can support a time-on-market claim at all."""
    total = db.scalar("SELECT COUNT(*) FROM advertisements") or 0
    truncated = db.scalar("SELECT COUNT(*) FROM advertisements WHERE left_truncated = ?", [True])
    eligible = db.scalar(
        "SELECT COUNT(*) FROM advertisements WHERE eligible_for_duration_ranking = ?", [True]
    )
    reliable = db.scalar(
        "SELECT COUNT(*) FROM advertisements WHERE published_at_reliable = ?", [True]
    )
    return {
        "advertisements_total": int(total),
        "left_truncated": int(truncated or 0),
        "publication_time_reliable": int(reliable or 0),
        "eligible_for_duration_ranking": int(eligible or 0),
        "eligible_share": round(float(eligible or 0) / total, 4) if total else None,
        "note": (
            "only the eligible subset supports a time-on-market claim; the rest were "
            "already active when monitoring began, so their earlier life is unobserved"
        ),
    }


# ---------------------------------------------------------------------------
# Market statistics
# ---------------------------------------------------------------------------

_DIMENSIONS = {
    "brand": "s.brand",
    "model": "s.model",
    "year": "s.year",
    "city": "s.city",
    "seller_type": "s.seller_type",
    "body_condition": "s.body_condition",
}


def market_statistics(
    db: Database, dimension: str = "brand", *, min_group: int = 1
) -> list[dict[str, Any]]:
    """Aggregate market behaviour by one dimension."""
    if dimension not in _DIMENSIONS and dimension not in ("price_range", "mileage_range"):
        raise ValueError(f"unsupported dimension: {dimension}")

    rows = db.fetchall(
        "SELECT a.id, a.current_status, a.sale_confidence, a.first_seen_at, a.last_seen_at,"
        " a.first_missing_at, a.repost_parent_ad_id, a.reappeared_at,"
        " s.brand, s.model, s.year, s.city, s.seller_type, s.body_condition,"
        " s.price_normalized, s.mileage_normalized"
        " FROM advertisements a"
        " LEFT JOIN advertisement_snapshots s ON s.id = a.latest_snapshot_id"
    )

    def key_of(row: dict[str, Any]) -> str:
        if dimension == "price_range":
            return _price_bucket(row.get("price_normalized"))
        if dimension == "mileage_range":
            return _mileage_bucket(row.get("mileage_normalized"))
        return str(row.get(dimension) or "unknown")

    price_reduced = {
        int(r["advertisement_id"])
        for r in db.fetchall(
            "SELECT DISTINCT advertisement_id FROM price_changes WHERE change_type='price_decrease'"
        )
    }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key_of(row), []).append(row)

    out: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        if len(members) < min_group:
            continue
        removed = [
            m
            for m in members
            if str(m["current_status"]) in (str(AdStatus.LIKELY_REMOVED), str(AdStatus.LIKELY_SOLD))
        ]
        reposted = [
            m
            for m in members
            if m.get("repost_parent_ad_id") or str(m["current_status"]) == str(AdStatus.REPOSTED)
        ]
        reappeared = [m for m in members if m.get("reappeared_at")]
        durations = []
        for m in members:
            estimate = estimate_duration(
                first_seen_at=parse_ts(m.get("first_seen_at")),
                last_seen_at=parse_ts(m.get("last_seen_at")),
                first_missing_at=parse_ts(m.get("first_missing_at")),
                status=str(m["current_status"]),
            )
            if estimate.estimated_active_seconds is not None:
                durations.append(estimate.estimated_active_seconds / 86400.0)
        prices = [m["price_normalized"] for m in members if m.get("price_normalized")]
        out.append(
            {
                dimension: key,
                "advertisements": len(members),
                "new_advertisements": sum(
                    1 for m in members if str(m["current_status"]) == str(AdStatus.NEW)
                ),
                "disappeared_count": len(removed),
                "estimated_removal_rate": round(len(removed) / len(members), 4),
                "median_observed_active_days": _median(durations),
                "min_observed_active_days": round(min(durations), 3) if durations else None,
                "max_observed_active_days": round(max(durations), 3) if durations else None,
                "median_price": _median([float(p) for p in prices]),
                "price_reduction_frequency": round(
                    sum(1 for m in members if int(m["id"]) in price_reduced) / len(members), 4
                ),
                "repost_rate": round(len(reposted) / len(members), 4),
                "reappearance_rate": round(len(reappeared) / len(members), 4),
            }
        )
    return out


def _price_bucket(price: Any) -> str:
    if not price:
        return "unknown"
    billions = float(price) / 1_000_000_000
    lower = int(billions)
    return f"{lower}-{lower + 1}B toman"


def _mileage_bucket(mileage: Any) -> str:
    if mileage is None:
        return "unknown"
    step = 25_000
    lower = (int(mileage) // step) * step
    return f"{lower}-{lower + step} km"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    import statistics

    return round(float(statistics.median(values)), 3)


# ---------------------------------------------------------------------------
# Cohorts and survival
# ---------------------------------------------------------------------------


def cohort_analysis(db: Database, *, granularity: str = "week") -> list[dict[str, Any]]:
    """Share of each first-seen cohort still active after N days.

    "Still active" means still observed present — a reappeared listing counts as
    active, which is the point of tracking reappearance at all.
    """
    if granularity not in ("week", "month"):
        raise ValueError("granularity must be 'week' or 'month'")
    rows = db.fetchall(
        "SELECT platform_ad_id, current_status, first_seen_at, last_seen_at, first_missing_at"
        " FROM advertisements WHERE first_seen_at IS NOT NULL"
    )
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        first_seen = parse_ts(row.get("first_seen_at"))
        if first_seen is None:
            continue
        if granularity == "week":
            iso = first_seen.isocalendar()
            label = f"{iso.year}-W{iso.week:02d}"
        else:
            label = first_seen.strftime("%Y-%m")
        cohorts.setdefault(label, []).append(row)

    horizons = (1, 3, 7, 14, 30)
    out: list[dict[str, Any]] = []
    for label, members in sorted(cohorts.items()):
        record: dict[str, Any] = {"cohort": label, "size": len(members)}
        for horizon in horizons:
            survivors = 0
            for member in members:
                first_seen = parse_ts(member.get("first_seen_at"))
                last_seen = parse_ts(member.get("last_seen_at"))
                first_missing = parse_ts(member.get("first_missing_at"))
                if first_seen is None:
                    continue
                # Present at the horizon if it was still being seen then, or if
                # its first miss came after the horizon.
                boundary = first_seen.timestamp() + horizon * 86400
                seen_until = (last_seen or first_seen).timestamp()
                missing_at = first_missing.timestamp() if first_missing else None
                if missing_at is None:
                    survivors += int(seen_until >= boundary)
                else:
                    survivors += int(missing_at > boundary)
            record[f"active_after_{horizon}d_pct"] = round(100.0 * survivors / len(members), 2)
        out.append(record)
    return out


def survival_dataset(db: Database, *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Kaplan-Meier / interval-censored ready rows.

    ``event_observed = 1`` means a disappearance was observed. It does **not** mean
    a sale was confirmed; ``sale_confidence`` and ``sale_label`` carry the inference
    separately so a model cannot conflate the two.
    """
    rows = db.fetchall(
        "SELECT a.platform_ad_id, a.current_status, a.first_seen_at, a.last_seen_at,"
        " a.first_missing_at, a.sale_confidence, a.sale_label, a.vehicle_entity_id,"
        " a.repost_parent_ad_id,"
        " s.brand, s.model, s.year, s.price_normalized, s.mileage_normalized, s.seller_type"
        " FROM advertisements a"
        " LEFT JOIN advertisement_snapshots s ON s.id = a.latest_snapshot_id"
    )
    return [survival_row(row, now=now) for row in rows]


def price_change_history(db: Database, *, limit: int = 1000) -> list[dict[str, Any]]:
    return db.fetchall(
        "SELECT p.*, a.platform_ad_id, a.canonical_url FROM price_changes p"
        " JOIN advertisements a ON a.id = p.advertisement_id"
        " ORDER BY p.changed_at DESC LIMIT ?",
        [limit],
    )
