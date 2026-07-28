"""Repost chains at listing and vehicle grain.

An advertisement id identifies a **listing**. A relisted car produces a second id,
and at listing grain that reads as one disappearance plus one arrival — which
would count a single unsold vehicle as a sale and a new entrant at once.

Everything that answers "how long was this car on the market" therefore belongs at
vehicle grain, which is what this module computes.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .database import ReadOnlyDatabase


def repost_link_summary(db: ReadOnlyDatabase, run_id: int) -> dict[str, Any]:
    """What the monitor's own repost detection found."""
    if not db.has_table("repost_links"):
        return {"available": False, "note": "repost_links table absent"}
    total = int(db.scalar("SELECT COUNT(*) FROM repost_links") or 0)
    if total == 0:
        return {
            "available": True,
            "links": 0,
            "note": (
                "no repost link was detected in this database. With three runs over a "
                "few minutes, a relisting would have had almost no opportunity to "
                "occur, so this is a limit of the observation window rather than "
                "evidence that reposting is rare on this platform."
            ),
        }
    rows = db.query_frame(
        "SELECT parent_ad_id, child_ad_id, score, vehicle_entity_id, matched_on_json"
        " FROM repost_links WHERE run_id <= ?",
        [run_id],
        label="repost.links",
    )
    return {
        "available": True,
        "links": total,
        "distinct_vehicles": int(rows["vehicle_entity_id"].nunique())
        if "vehicle_entity_id" in rows
        else None,
        "score": {
            "median": round(float(pd.to_numeric(rows["score"], errors="coerce").median()), 4),
            "min": round(float(pd.to_numeric(rows["score"], errors="coerce").min()), 4),
        }
        if "score" in rows and not rows.empty
        else None,
    }


def vehicle_grain_summary(vehicles: pd.DataFrame) -> dict[str, Any]:
    """Chain sizes and what they mean for counting."""
    if vehicles.empty:
        return {"available": False}
    counts = pd.to_numeric(vehicles.get("advertisement_count"), errors="coerce").fillna(1)
    multi = vehicles[counts > 1]
    return {
        "available": True,
        "vehicle_entities": int(len(vehicles)),
        "advertisements_covered": int(counts.sum()),
        "entities_with_multiple_advertisements": int(len(multi)),
        "share_multi_listing_pct": round(len(multi) / len(vehicles) * 100, 4),
        "chain_size_distribution": {
            str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()
        },
        "max_chain_size": int(counts.max()),
        "interpretation": (
            "Each vehicle entity is counted once, whatever its chain length. A repost "
            "is an explanatory variable, not an outcome: the chain ends only when its "
            "last listing disappears without a successor."
        ),
    }


def repost_transitions(cross: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    """Per-chain detail: price and mileage movement between linked listings.

    Only chains of length 2+ produce rows. On a database without detected reposts
    this returns an empty frame with the right columns, which is the honest output.
    """
    columns = [
        "vehicle_entity_id",
        "advertisement_count",
        "advertisement_ids",
        "first_advertisement_id",
        "latest_advertisement_id",
        "latest_status",
        "price_first",
        "price_latest",
        "price_delta",
        "price_delta_pct",
        "mileage_first",
        "mileage_latest",
        "mileage_delta",
        "seller_consistent",
        "description_similarity",
        "days_between_listings",
        "note",
    ]
    if vehicles.empty or "advertisement_count" not in vehicles.columns:
        return pd.DataFrame(columns=columns)

    chains = vehicles[pd.to_numeric(vehicles["advertisement_count"], errors="coerce").fillna(1) > 1]
    if chains.empty:
        return pd.DataFrame(columns=columns)

    indexed = cross.set_index("platform_ad_id") if "platform_ad_id" in cross.columns else cross
    records: list[dict[str, Any]] = []
    for _, chain in chains.iterrows():
        ids = str(chain.get("advertisement_ids", "")).split(",")
        members = [indexed.loc[i] for i in ids if i in indexed.index]
        if len(members) < 2:
            continue
        first, latest = members[0], members[-1]

        price_first = pd.to_numeric(pd.Series([first.get("price_toman")]), errors="coerce").iloc[0]
        price_latest = pd.to_numeric(pd.Series([latest.get("price_toman")]), errors="coerce").iloc[
            0
        ]
        mileage_first = pd.to_numeric(pd.Series([first.get("mileage_km")]), errors="coerce").iloc[0]
        mileage_latest = pd.to_numeric(pd.Series([latest.get("mileage_km")]), errors="coerce").iloc[
            0
        ]

        records.append(
            {
                "vehicle_entity_id": chain["vehicle_entity_id"],
                "advertisement_count": int(chain["advertisement_count"]),
                "advertisement_ids": chain.get("advertisement_ids"),
                "first_advertisement_id": chain.get("first_advertisement_id"),
                "latest_advertisement_id": chain.get("latest_advertisement_id"),
                "latest_status": chain.get("latest_status"),
                "price_first": price_first,
                "price_latest": price_latest,
                "price_delta": (
                    price_latest - price_first
                    if pd.notna(price_first) and pd.notna(price_latest)
                    else None
                ),
                "price_delta_pct": (
                    round((price_latest - price_first) / price_first * 100, 3)
                    if pd.notna(price_first) and price_first and pd.notna(price_latest)
                    else None
                ),
                "mileage_first": mileage_first,
                "mileage_latest": mileage_latest,
                "mileage_delta": (
                    mileage_latest - mileage_first
                    if pd.notna(mileage_first) and pd.notna(mileage_latest)
                    else None
                ),
                "seller_consistent": _same(
                    first.get("deep_seller_type"), latest.get("deep_seller_type")
                ),
                "description_similarity": _similarity(
                    first.get("deep_description_scrubbed"),
                    latest.get("deep_description_scrubbed"),
                ),
                "days_between_listings": _gap_days(first, latest),
                "note": "a repost means the vehicle stayed on the market; NOT a sale",
            }
        )
    return pd.DataFrame(records, columns=columns)


def _same(left: Any, right: Any) -> Any:
    if pd.isna(left) or pd.isna(right):
        return None
    return bool(str(left) == str(right))


def _similarity(left: Any, right: Any) -> float | None:
    """Description similarity, the same measure the monitor's matcher uses."""
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return None
    return round(SequenceMatcher(None, left, right).ratio(), 4)


def _gap_days(first: Any, latest: Any) -> float | None:
    from bama_monitor.db import parse_ts

    gone = parse_ts(first.get("first_missing_at")) or parse_ts(first.get("last_seen_at"))
    started = parse_ts(latest.get("first_seen_at"))
    if gone is None or started is None:
        return None
    return round((started - gone).total_seconds() / 86400.0, 4)


def mileage_consistency_check(chains: pd.DataFrame) -> dict[str, Any]:
    """Mileage should not fall between relistings of the same car.

    A decrease is evidence the match is wrong, and a wrong match merges two
    different vehicles into one entity — which corrupts every vehicle-grain
    duration downstream.
    """
    if chains.empty or "mileage_delta" in chains.columns is False:
        return {"checked": 0, "decreases": 0, "ok": True}
    deltas = pd.to_numeric(chains.get("mileage_delta"), errors="coerce").dropna()
    decreases = int((deltas < -1000).sum())
    return {
        "checked": int(len(deltas)),
        "decreases": decreases,
        "ok": decreases == 0,
        "note": (
            "a mileage decrease across a repost chain suggests a false match; the two "
            "listings may describe different vehicles"
        ),
    }
