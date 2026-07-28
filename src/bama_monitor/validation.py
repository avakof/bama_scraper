"""Ground-truth validation of the sale inference.

``sale_evidence_score`` is an additive heuristic. Until it has been compared with
real outcomes it is an ordering, not a rate: a score of 0.65 means "more evidence
than 0.40", **not** "65% of these sold". Nothing in this package will tell you
otherwise, and this module is how that claim eventually gets earned — or refuted.

The workflow is deliberately manual, because there is no automatic oracle:

1. ``sample`` draws a stratified sample of disappeared listings and writes a
   worksheet with one row per listing and an empty ``observed_outcome`` column;
2. a human checks each listing (seller contact, the platform, a repost search)
   and fills the column in;
3. ``ingest`` loads the completed worksheet back;
4. ``calibration`` reports, per score band, how often the outcome was actually a
   sale — the number that would justify probability language, if it holds.

Until step 4 has been run on a decent sample, every report in this system keeps
calling the score heuristic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .db import Database, utcnow
from .models import DISAPPEARED_STATUSES

#: Outcomes a human may record. ``unknown`` is a first-class answer: guessing
#: would poison the very measurement this exists to make.
VALID_OUTCOMES = (
    "sold",
    "not_sold",
    "withdrawn",
    "expired",
    "reposted",
    "outside_filter",
    "unknown",
)

WORKSHEET_COLUMNS = [
    "advertisement_id",
    "canonical_url",
    "status",
    "sale_evidence_score",
    "sale_label",
    "score_band",
    "detail_availability",
    "first_seen_at",
    "last_seen_at",
    "first_missing_at",
    "brand",
    "model",
    "year",
    "price",
    # Filled in by a human.
    "observed_outcome",
    "outcome_source",
    "notes",
]

#: Score bands the calibration report groups by.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("0.00-0.39", 0.0, 0.40),
    ("0.40-0.64", 0.40, 0.65),
    ("0.65-0.84", 0.65, 0.85),
    ("0.85-1.00", 0.85, 1.01),
)


def band_of(score: float) -> str:
    for name, low, high in BANDS:
        if low <= score < high:
            return name
    return BANDS[-1][0]


def draw_sample(
    db: Database, *, per_band: int = 25, run_id: int | None = None
) -> list[dict[str, Any]]:
    """Stratified sample of disappeared listings, ``per_band`` from each band.

    Stratifying matters: an unstratified sample of a corpus where most
    disappearances score low would produce almost no high-score rows, and the high
    band is exactly where a probability claim would be made.
    """
    placeholders = ",".join("?" * len(DISAPPEARED_STATUSES))
    rows = db.fetchall(
        "SELECT a.id, a.platform_ad_id, a.canonical_url, a.current_status,"
        " a.sale_confidence, a.sale_label, a.detail_availability,"
        " a.first_seen_at, a.last_seen_at, a.first_missing_at,"
        " s.brand, s.model, s.year, s.price_normalized"
        " FROM advertisements a"
        " LEFT JOIN advertisement_snapshots s ON s.id = a.latest_snapshot_id"
        f" WHERE a.current_status IN ({placeholders})"
        " ORDER BY a.id",
        [str(s) for s in sorted(DISAPPEARED_STATUSES)],
    )

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in BANDS}
    for row in rows:
        score = float(row.get("sale_confidence") or 0.0)
        bucket = buckets[band_of(score)]
        if len(bucket) < per_band:
            bucket.append(
                {
                    "advertisement_id": row["platform_ad_id"],
                    "_row_id": row["id"],
                    "canonical_url": row["canonical_url"],
                    "status": row["current_status"],
                    "sale_evidence_score": score,
                    "sale_label": row.get("sale_label"),
                    "score_band": band_of(score),
                    "detail_availability": row.get("detail_availability"),
                    "first_seen_at": row.get("first_seen_at"),
                    "last_seen_at": row.get("last_seen_at"),
                    "first_missing_at": row.get("first_missing_at"),
                    "brand": row.get("brand"),
                    "model": row.get("model"),
                    "year": row.get("year"),
                    "price": row.get("price_normalized"),
                    "observed_outcome": "",
                    "outcome_source": "",
                    "notes": "",
                }
            )

    sample = [row for bucket in buckets.values() for row in bucket]
    now = utcnow()
    for row in sample:
        db.upsert(
            "sale_validation_samples",
            ["advertisement_id", "sampled_run_id"],
            {
                "advertisement_id": row["_row_id"],
                "sampled_run_id": run_id,
                "sampled_at": now,
                "predicted_score": row["sale_evidence_score"],
                "predicted_label": row["sale_label"],
            },
            update_columns=["predicted_score", "predicted_label"],
        )
    return sample


def write_worksheet(sample: list[dict[str, Any]], path: Path) -> int:
    """Write the human-facing worksheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sample:
            writer.writerow(row)
    return len(sample)


def ingest_worksheet(db: Database, path: Path, *, labelled_by: str = "manual") -> dict[str, Any]:
    """Load completed labels back, rejecting anything outside the vocabulary."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    loaded = 0
    skipped: list[str] = []
    rejected: list[str] = []
    now = utcnow()

    for row in reader:
        outcome = (row.get("observed_outcome") or "").strip().lower()
        ad_id = (row.get("advertisement_id") or "").strip()
        if not outcome:
            skipped.append(ad_id)
            continue
        if outcome not in VALID_OUTCOMES:
            rejected.append(f"{ad_id}: {outcome!r}")
            continue
        record = db.fetchone("SELECT id FROM advertisements WHERE platform_ad_id=?", [ad_id])
        if record is None:
            rejected.append(f"{ad_id}: unknown advertisement")
            continue
        db.execute(
            "UPDATE sale_validation_samples SET observed_outcome=?, outcome_source=?,"
            " labelled_at=?, labelled_by=?, notes=? WHERE advertisement_id=?",
            [
                outcome,
                (row.get("outcome_source") or "").strip() or None,
                now,
                labelled_by,
                (row.get("notes") or "").strip() or None,
                record["id"],
            ],
        )
        loaded += 1

    return {
        "labelled": loaded,
        "left_blank": len(skipped),
        "rejected": rejected,
        "valid_outcomes": list(VALID_OUTCOMES),
    }


def calibration(db: Database) -> dict[str, Any]:
    """Per-band sale rate among labelled samples.

    ``sale_rate`` is the only number in this system that may legitimately be read
    as a probability, and only for bands whose ``labelled`` count is large enough
    to mean anything — which is why the count is reported beside it and no
    confidence-interval-free rate is emitted for tiny samples.
    """
    rows = db.fetchall(
        "SELECT predicted_score, predicted_label, observed_outcome"
        " FROM sale_validation_samples WHERE observed_outcome IS NOT NULL"
    )
    bands: dict[str, dict[str, Any]] = {
        name: {"band": name, "labelled": 0, "sold": 0, "not_sold": 0, "unknown": 0}
        for name, _, _ in BANDS
    }
    for row in rows:
        band = bands[band_of(float(row.get("predicted_score") or 0.0))]
        outcome = str(row.get("observed_outcome"))
        band["labelled"] += 1
        if outcome == "sold":
            band["sold"] += 1
        elif outcome == "unknown":
            band["unknown"] += 1
        else:
            band["not_sold"] += 1

    for band in bands.values():
        decided = band["labelled"] - band["unknown"]
        # A rate from a handful of labels is noise dressed as a measurement.
        band["sale_rate"] = round(band["sold"] / decided, 4) if decided >= 10 else None
        band["decided"] = decided
        if band["sale_rate"] is None:
            band["note"] = f"only {decided} decided label(s); too few for a rate"

    total_labelled = sum(b["labelled"] for b in bands.values())
    return {
        "bands": list(bands.values()),
        "total_labelled": total_labelled,
        "calibrated": total_labelled >= 40
        and any(b["sale_rate"] is not None for b in bands.values()),
        "interpretation": (
            "sale_evidence_score is an ordering, not a probability, until these "
            "bands are populated. Until `calibrated` is true, do not describe a "
            "score as a percentage chance of sale."
        ),
    }
