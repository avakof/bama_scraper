"""Build the three analytical datasets.

Three, not one, because they have different grains and conflating them is how
analyses come to double-count:

* **A — advertisement cross-section**: one row per advertisement in the selected
  run. The "what does the inventory look like right now" table.
* **B — longitudinal panel**: one row per ``(run_id, advertisement_id)``. The "what
  happened over time" table.
* **C — vehicle entity**: one row per physical vehicle, spanning repost chains. The
  only grain at which "how long did this car take to sell" is even a coherent
  question.

Two rules are enforced in SQL rather than trusted to the caller:

1. **No future snapshots.** A detail snapshot may only describe an analysis run if
   ``snapshot.scraped_at <= run.finished_at``. Joining the newest snapshot
   regardless of time would let tomorrow's price explain yesterday's inventory.
2. **Deterministic selection.** When several snapshots qualify, the latest is
   chosen with a window function ordered by ``(scraped_at DESC, id DESC)``, so the
   same database always yields the same row.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bama_monitor.db import parse_ts

from .config import EdaConfig
from .database import ReadOnlyDatabase
from .models import (
    EXCLUDED_BLOB_COLUMNS,
    PRIVATE_COLUMNS,
    AnalysisContext,
    is_texty,
)
from .provenance import Manifest, stamp

#: Run statuses whose absence observations may be trusted. Anything else means the
#: scrape was incomplete, so "not seen" carries no information.
VALID_RUN_STATUSES: tuple[str, ...] = ("valid",)


#: Free-text columns that carry whatever the seller typed, including contact
#: details they were asked not to type. Scrubbed on the way into every dataset.
FREETEXT_COLUMNS: tuple[str, ...] = (
    "snap_description",
    "snap_title",
    "deep_description",
    "deep_description_scrubbed",
    "obs_card_title",
    "card_title",
    "title",
    "description",
)


def _scrub_freetext(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip contact-shaped content from free-text before it enters a dataset.

    The monitoring snapshot stores the seller's description verbatim -- it has to,
    it is evidence -- but an analytical export is a different artefact with a
    different audience. Scrubbing here means the audit's contact detector is a
    check on the unexpected, not the only thing standing between a phone number
    and a CSV.
    """
    from .text_analysis import scrub

    out = frame.copy()
    scrubbed: list[str] = []
    for column in FREETEXT_COLUMNS:
        if column in out.columns and is_texty(out[column]):
            out[column] = out[column].map(lambda v: scrub(v) if isinstance(v, str) else v)
            scrubbed.append(column)
    return out, scrubbed


def _drop_forbidden(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove private and blob columns wherever they came from.

    An independent second gate: the scraper already strips contact fields, but a
    privacy guarantee enforced in exactly one place is a privacy hope.
    """
    dropped = [
        c
        for c in frame.columns
        if c.lower() in PRIVATE_COLUMNS or c.lower() in EXCLUDED_BLOB_COLUMNS
    ]
    return (frame.drop(columns=dropped) if dropped else frame), dropped


# ---------------------------------------------------------------------------
# Dataset A - advertisement cross-section
# ---------------------------------------------------------------------------


def build_cross_section(
    db: ReadOnlyDatabase, context: AnalysisContext, cfg: EdaConfig, manifest: Manifest
) -> pd.DataFrame:
    """One row per advertisement observed in (or known to) the selected run."""
    ad_columns = _existing(
        db,
        "advertisements",
        [
            "id",
            "platform",
            "platform_ad_id",
            "canonical_url",
            "current_status",
            "first_seen_at",
            "last_seen_at",
            "first_missing_at",
            "confirmed_removed_at",
            "reappeared_at",
            "consecutive_misses",
            "total_seen_runs",
            "total_missing_runs",
            "repost_parent_ad_id",
            "vehicle_entity_id",
            "vehicle_fingerprint",
            "sale_confidence",
            "sale_label",
            "last_detail_verdict",
            "last_detail_checked_at",
            "detail_availability",
            "filter_exit_reason",
            "filter_exit_at",
            "published_at",
            "published_at_source",
            "published_at_reliable",
            "left_truncated",
            "eligible_for_duration_ranking",
            "entry_delay_seconds",
            "detail_failure_count",
            "search_configuration_hash",
            "latest_snapshot_id",
        ],
    )
    select_ads = ", ".join(f"a.{c} AS ad_{c}" if c == "id" else f"a.{c}" for c in ad_columns)

    obs_columns = _existing(
        db,
        "daily_ad_observations",
        [
            "was_seen",
            "position",
            "page_number",
            "card_title",
            "card_price_raw",
            "card_price_normalized",
            "card_year",
            "card_mileage_raw",
            "card_mileage_normalized",
            "card_location",
            "card_image_url",
            "card_hash",
            "observed_at",
        ],
    )
    select_obs = ", ".join(f"o.{c} AS obs_{c}" for c in obs_columns)

    sql = (
        f"SELECT {select_ads}, {select_obs}"
        " FROM advertisements a"
        " JOIN daily_ad_observations o"
        "   ON o.advertisement_id = a.id AND o.run_id = ?"
        " ORDER BY a.id"
    )
    frame = db.query_frame(sql, [context.run_id], label="cross_section.base")
    manifest.source_tables.setdefault("advertisements", db.row_count("advertisements"))
    manifest.source_tables.setdefault(
        "daily_ad_observations", db.row_count("daily_ad_observations")
    )

    if frame.empty:
        manifest.warn(
            "cross_section",
            f"run {context.run_id} has no observation rows; the cross-section is empty",
        )
        return stamp(frame, context)

    frame = _attach_snapshot(db, frame, context, cfg, manifest)
    frame = _attach_media_and_description(frame, cfg)
    frame = _attach_event_counts(db, frame, context, manifest)
    frame = _derive_cross_section_fields(frame, context, cfg)

    if cfg.enrichment.enabled:
        frame = _attach_enrichment(frame, context, cfg, manifest)

    frame, dropped = _drop_forbidden(frame)
    if dropped:
        manifest.warn(
            "privacy",
            f"dropped {len(dropped)} private/blob column(s) from the cross-section: "
            + ", ".join(sorted(dropped)),
        )
    frame, scrubbed = _scrub_freetext(frame)
    if scrubbed:
        manifest.warn(
            "privacy",
            "scrubbed contact-shaped content from free-text column(s): "
            + ", ".join(sorted(scrubbed)),
        )
    return stamp(frame, context)


def _reference_instant(context: AnalysisContext) -> Any:
    """The latest instant that still belongs to the analysis run.

    Takes the later of the wall-clock finish and the scheduled slot, because the
    monitor writes on both clocks and either one alone would exclude legitimate
    same-run rows.
    """
    candidates = [t for t in (context.run_finished_at, context.scheduled_for) if t is not None]
    return max(candidates) if candidates else None


def _reference_basis(context: AnalysisContext) -> str:
    finished, slot = context.run_finished_at, context.scheduled_for
    if finished and slot:
        return "scheduled_for" if slot >= finished else "finished_at"
    if finished:
        return "finished_at"
    return "scheduled_for" if slot else "none"


def _existing(db: ReadOnlyDatabase, table: str, wanted: list[str]) -> list[str]:
    """Intersect a wish list with reality, preserving order."""
    present = db.has_columns(table, wanted)
    return [c for c in wanted if c in present]


def _attach_snapshot(
    db: ReadOnlyDatabase,
    frame: pd.DataFrame,
    context: AnalysisContext,
    cfg: EdaConfig,
    manifest: Manifest,
) -> pd.DataFrame:
    """Join the latest snapshot that existed **at or before** the run finished."""
    if not db.has_table("advertisement_snapshots"):
        manifest.warn("cross_section", "advertisement_snapshots is absent; no detail attributes")
        return frame

    snap_columns = _existing(
        db,
        "advertisement_snapshots",
        [
            "id",
            "advertisement_id",
            "scraped_at",
            "title",
            "brand",
            "model",
            "trim",
            "year",
            "price_raw",
            "price_normalized",
            "mileage_raw",
            "mileage_normalized",
            "description",
            "seller_type",
            "seller_name",
            "city",
            "province",
            "body_color",
            "interior_color",
            "transmission",
            "fuel_type",
            "engine",
            "body_condition",
            "chassis_condition",
            "insurance",
            "published_at_raw",
            "published_at_normalized",
            "media_json",
            "content_hash",
            "detail_page_status",
            "parser_version",
        ],
    )
    # -- the no-future-snapshot rule, applied across two clocks ---------
    #
    # The monitor deliberately stamps observations and snapshots with the run's
    # *scheduled slot* (so duration bounds sit on a clean 24-hour grid), while
    # `finished_at` records the real wall-clock finish. For a run attributed to a
    # slot ahead of when it actually executed -- a catch-up run, or a back-dated
    # test run -- the slot is LATER than `finished_at`, and a literal
    # `scraped_at <= finished_at` would discard every snapshot the run itself
    # wrote. That is not the rule doing its job; it is two clocks being compared.
    #
    # So the guarantee is expressed twice:
    #   1. `run_id <= analysis_run_id`  -- clock-free, and the real intent: a later
    #      run's snapshot can never describe an earlier run;
    #   2. `scraped_at <= reference`    -- the timestamp rule, where `reference` is
    #      the run's own latest instant on either clock.
    #
    # Both counts are reported, so the difference between the literal rule and the
    # applied one is visible rather than quietly resolved.
    reference = _reference_instant(context)
    if reference is None:
        manifest.warn(
            "cross_section",
            "the selected run has neither finished_at nor scheduled_for; the "
            "no-future-snapshot rule falls back to run_id ordering alone",
        )

    selected = ", ".join(f"s.{c}" for c in snap_columns)
    conditions = ["s.run_id <= ?"]
    params: list[Any] = [context.run_id]
    if reference is not None:
        conditions.append("s.scraped_at <= ?")
        params.append(reference)
    where = " AND ".join(conditions)

    # ROW_NUMBER with an explicit tiebreak on id: two snapshots sharing a
    # scraped_at would otherwise make the result depend on physical row order.
    sql = (
        f"SELECT {selected} FROM ("
        "  SELECT s.*, ROW_NUMBER() OVER ("
        "     PARTITION BY s.advertisement_id"
        "     ORDER BY s.scraped_at DESC, s.id DESC) AS rn"
        f"  FROM advertisement_snapshots s WHERE {where}"
        ") s WHERE s.rn = 1"
    )
    snaps = db.query_frame(sql, params, label="cross_section.latest_snapshot")

    manifest.source_tables.setdefault(
        "advertisement_snapshots", db.row_count("advertisement_snapshots")
    )
    total_snapshots = db.row_count("advertisement_snapshots")
    excluded_by_run = int(
        db.scalar("SELECT COUNT(*) FROM advertisement_snapshots WHERE run_id > ?", [context.run_id])
        or 0
    )
    excluded_by_time = (
        int(
            db.scalar(
                "SELECT COUNT(*) FROM advertisement_snapshots WHERE scraped_at > ?", [reference]
            )
            or 0
        )
        if reference is not None
        else 0
    )
    literal_rule_excluded = (
        int(
            db.scalar(
                "SELECT COUNT(*) FROM advertisement_snapshots WHERE scraped_at > ?",
                [context.run_finished_at],
            )
            or 0
        )
        if context.run_finished_at is not None
        else 0
    )
    manifest.record_exclusion(
        "cross_section.snapshots",
        f"snapshot from a later run (run_id > {context.run_id})",
        excluded_by_run,
    )
    manifest.record_exclusion(
        "cross_section.snapshots",
        f"snapshot scraped_at > analysis reference instant ({reference})",
        excluded_by_time,
    )
    manifest.enrichment["snapshot_coverage"] = {
        "snapshot_rows_total": total_snapshots,
        "analysis_reference_instant": reference.isoformat() if reference else None,
        "reference_basis": _reference_basis(context),
        "excluded_by_run_id": excluded_by_run,
        "excluded_by_reference_instant": excluded_by_time,
        "would_be_excluded_by_finished_at_alone": literal_rule_excluded,
        "advertisements_with_snapshot": int(len(snaps)),
    }
    if literal_rule_excluded and literal_rule_excluded != excluded_by_time:
        manifest.warn(
            "cross_section",
            f"{literal_rule_excluded} snapshot(s) postdate the run's wall-clock "
            f"finished_at but not its scheduled slot; snapshots are slot-stamped by the "
            "monitor, so the reference instant is used instead of finished_at alone",
        )

    if snaps.empty:
        manifest.warn(
            "cross_section",
            "no snapshot qualifies at or before the run finish time; vehicle attributes "
            "(brand, model, condition, seller type) are unavailable from the monitoring "
            "database for this run",
        )
        return frame

    snaps = snaps.rename(columns={"id": "snapshot_id", "scraped_at": "snapshot_scraped_at"})
    snaps = snaps.add_prefix("snap_").rename(
        columns={
            "snap_advertisement_id": "advertisement_id",
            "snap_snapshot_scraped_at": "snapshot_scraped_at",
            "snap_snapshot_id": "snapshot_id",
        }
    )
    merged = frame.merge(
        snaps, left_on="ad_id", right_on="advertisement_id", how="left", validate="one_to_one"
    ).drop(columns=["advertisement_id"], errors="ignore")
    return merged


def _attach_media_and_description(frame: pd.DataFrame, cfg: EdaConfig) -> pd.DataFrame:
    """Derive counts from the media/description fields without keeping the blobs."""
    out = frame.copy()
    if "snap_media_json" in out.columns:
        out["media_count"] = out["snap_media_json"].map(_count_media)
        out = out.drop(columns=["snap_media_json"])
    else:
        out["media_count"] = pd.NA

    description = None
    for candidate in ("snap_description_scrubbed", "snap_description"):
        if candidate in out.columns:
            description = out[candidate]
            break
    if description is not None:
        out["description_length"] = description.map(
            lambda v: len(str(v)) if isinstance(v, str) and v else (0 if v == "" else pd.NA)
        )
        out["has_description"] = description.map(lambda v: bool(isinstance(v, str) and v.strip()))
    else:
        out["description_length"] = pd.NA
        out["has_description"] = pd.NA
    return out


def _count_media(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return pd.NA
        return len(parsed) if isinstance(parsed, list) else pd.NA
    return pd.NA


def _attach_event_counts(
    db: ReadOnlyDatabase, frame: pd.DataFrame, context: AnalysisContext, manifest: Manifest
) -> pd.DataFrame:
    """Per-advertisement event tallies up to the analysis run."""
    if not db.has_table("advertisement_events"):
        return frame
    manifest.source_tables.setdefault("advertisement_events", db.row_count("advertisement_events"))
    counts = db.query_frame(
        "SELECT advertisement_id, event_type, COUNT(*) AS n"
        " FROM advertisement_events WHERE run_id <= ?"
        " GROUP BY advertisement_id, event_type",
        [context.run_id],
        label="cross_section.event_counts",
    )
    if counts.empty:
        return frame
    wide = counts.pivot(index="advertisement_id", columns="event_type", values="n").fillna(0)
    wide.columns = [f"events_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return frame.merge(wide, left_on="ad_id", right_on="advertisement_id", how="left").drop(
        columns=["advertisement_id"], errors="ignore"
    )


def _derive_cross_section_fields(
    frame: pd.DataFrame, context: AnalysisContext, cfg: EdaConfig
) -> pd.DataFrame:
    """Snapshot age, staleness, coalesced price/mileage/year, and their sources.

    Price is taken from the detail snapshot when one exists and from the search
    card otherwise, and ``price_source`` records which. Without that column a
    reader cannot tell a detail-page price from a card price, and the two are
    collected at different times by different code paths.
    """
    out = frame.copy()
    # The SAME reference instant used for the snapshot cutoff. Using `finished_at`
    # here while cutting off on the reference instant produced negative ages: the
    # monitor writes observations on the scheduled-slot clock and `finished_at` on
    # the wall clock, so differencing across the two is meaningless.
    finished = _reference_instant(context)

    if "snapshot_scraped_at" in out.columns:
        scraped = out["snapshot_scraped_at"].map(parse_ts)
        # `pd.notna`, not truthiness: pandas coerces a datetime column containing
        # None to NaT, and NaT is TRUTHY. `d.isoformat()` on NaT returns the string
        # "NaT", which every downstream completeness check would read as a present
        # value -- turning "no snapshot" into "snapshot known".
        out["snapshot_scraped_at"] = scraped.map(
            lambda d: d.isoformat() if pd.notna(d) and d is not None else None
        )
        if finished is not None:
            out["snapshot_age_hours"] = scraped.map(
                lambda d: (
                    round((finished - d).total_seconds() / 3600.0, 4)
                    if pd.notna(d) and d is not None
                    else pd.NA
                )
            )
        else:
            out["snapshot_age_hours"] = pd.NA
        out["snapshot_is_stale"] = out["snapshot_age_hours"].map(
            lambda h: bool(h > cfg.thresholds.snapshot_stale_hours) if pd.notna(h) else pd.NA
        )
    else:
        out["snapshot_scraped_at"] = None
        out["snapshot_age_hours"] = pd.NA
        out["snapshot_is_stale"] = pd.NA

    out["price_toman"], out["price_source"] = _coalesce(
        out,
        [
            ("snap_price_normalized", "detail_snapshot"),
            ("obs_card_price_normalized", "search_card"),
        ],
    )
    out["mileage_km"], out["mileage_source"] = _coalesce(
        out,
        [
            ("snap_mileage_normalized", "detail_snapshot"),
            ("obs_card_mileage_normalized", "search_card"),
        ],
    )
    out["year_jalali"], out["year_source"] = _coalesce(
        out, [("snap_year", "detail_snapshot"), ("obs_card_year", "search_card")]
    )
    raw_year = pd.to_numeric(out["year_jalali"], errors="coerce")
    # Bama shows a Gregorian year on some cards (2023) and a Jalali one on others
    # (1402). Both land in the same field. Converting rather than discarding keeps
    # ~1.5% of the inventory in the year analysis, and `year_calendar_source`
    # records which rows were converted so the choice stays auditable.
    is_gregorian = raw_year.notna() & (raw_year >= 1500)
    out["year_jalali"] = raw_year.where(~is_gregorian, raw_year - 621)
    out["year_calendar_source"] = np.where(
        raw_year.isna(), None, np.where(is_gregorian, "gregorian_converted", "jalali_as_published")
    )
    out["year_gregorian"] = out["year_jalali"] + 621

    # log price for the skew-heavy analyses; guarded so a zero cannot produce -inf
    price = pd.to_numeric(out["price_toman"], errors="coerce")
    out["price_toman"] = price
    out["log10_price"] = np.where(price > 0, np.log10(price.astype("float64")), np.nan)

    out["listing_age_days_observed"] = _age_days(out, "first_seen_at", finished)
    out["age_reference_instant"] = finished.isoformat() if finished else None
    out["age_reference_basis"] = _reference_basis(context)
    out["is_zero_km"] = pd.to_numeric(out["mileage_km"], errors="coerce").eq(0)
    return out


def _coalesce(frame: pd.DataFrame, sources: list[tuple[str, str]]) -> tuple[pd.Series, pd.Series]:
    """First non-null value across ``sources``, plus the name of the source used."""
    value = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    origin = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    for column, name in sources:
        if column not in frame.columns:
            continue
        candidate = frame[column]
        mask = value.isna() & candidate.notna()
        value = value.where(~mask, candidate)
        origin = origin.where(~mask, name)
    return value, origin


def _age_days(frame: pd.DataFrame, column: str, reference: Any) -> pd.Series:
    if column not in frame.columns or reference is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    return frame[column].map(
        lambda v: (
            round((reference - parse_ts(v)).total_seconds() / 86400.0, 4)
            if parse_ts(v) is not None
            else pd.NA
        )
    )


def _attach_enrichment(
    frame: pd.DataFrame, context: AnalysisContext, cfg: EdaConfig, manifest: Manifest
) -> pd.DataFrame:
    """Optional attribute join from the deep scraper database.

    Explicit and recorded, never silent. Two joins happen here:

    1. **per-advertisement attributes** from ``ad_deep`` -- every analysable
       column, not a curated shortlist, because a shortlist decides in advance
       what the analysis is allowed to notice;
    2. **trim-level technical specifications** from the wide spec view, joined
       through the trim key.

    The temporal rule applies to the first: a deep record scraped after the
    analysis run finished may not describe that run. It does **not** apply to the
    second, and the reason is worth stating -- a trim's dimensions and safety
    equipment are properties of the model, not observations with a timestamp.
    """
    import sqlite3

    path = cfg.enrichment.database_path
    if path is None or not Path(path).exists():
        manifest.warn("enrichment", f"attribute database not found at {path}; skipped")
        manifest.enrichment["enabled"] = False
        return frame

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        merged = _join_deep_attributes(frame, con, context, cfg, manifest)
        if cfg.enrichment.include_specs:
            merged = _join_trim_specs(merged, con, cfg, manifest)
    finally:
        con.close()
    return merged


def _analysable_columns(
    con: Any, table: str, *, min_coverage: float, manifest: Manifest
) -> tuple[list[str], dict[str, Any]]:
    """Every column worth analysing, with the rejections counted.

    Rejected for three reasons, each recorded separately so "we did not analyse
    it" never gets confused with "there was nothing there":
    private/blob, entirely empty, or constant.
    """
    present = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0

    keep: list[str] = []
    rejected: dict[str, list[str]] = {
        "private_or_blob": [],
        "empty": [],
        "constant": [],
        "sparse": [],
    }
    for column in present:
        lowered = column.lower()
        if (
            lowered in PRIVATE_COLUMNS
            or lowered in EXCLUDED_BLOB_COLUMNS
            or lowered.endswith("_json")
        ):
            rejected["private_or_blob"].append(column)
            continue
        non_null = con.execute(f'SELECT COUNT("{column}") FROM {table}').fetchone()[0]
        if non_null == 0:
            rejected["empty"].append(column)
            continue
        if total and non_null / total < min_coverage:
            rejected["sparse"].append(column)
            continue
        distinct = con.execute(f'SELECT COUNT(DISTINCT "{column}") FROM {table}').fetchone()[0]
        if distinct <= 1:
            rejected["constant"].append(column)
            continue
        keep.append(column)

    summary = {
        "columns_present": len(present),
        "columns_kept": len(keep),
        "rejected_private_or_blob": len(rejected["private_or_blob"]),
        "rejected_entirely_empty": len(rejected["empty"]),
        "rejected_constant": len(rejected["constant"]),
        "rejected_too_sparse": len(rejected["sparse"]),
        "rejected_columns": {k: sorted(v) for k, v in rejected.items() if v},
    }
    return keep, summary


def _join_deep_attributes(
    frame: pd.DataFrame,
    con: Any,
    context: AnalysisContext,
    cfg: EdaConfig,
    manifest: Manifest,
) -> pd.DataFrame:
    table = cfg.enrichment.table
    join_right = cfg.enrichment.join_right
    timestamp = cfg.enrichment.timestamp_column

    if cfg.enrichment.include_all_columns:
        columns, rejection = _analysable_columns(
            con, table, min_coverage=cfg.enrichment.min_column_coverage, manifest=manifest
        )
    else:
        columns, rejection = _CURATED_DEEP_COLUMNS, {"note": "curated column list"}

    for required in (join_right, timestamp):
        if required not in columns:
            columns = [*columns, required]
    present = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    columns = [c for c in columns if c in present]
    if join_right not in columns:
        manifest.warn(
            "enrichment",
            f"join column {join_right!r} absent from {table}; enrichment skipped",
        )
        manifest.enrichment["enabled"] = False
        return frame

    selected = ", ".join(f'"{c}"' for c in columns)
    cutoff = _reference_instant(context)
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if timestamp in columns and cutoff is not None:
        rows = con.execute(
            f'SELECT {selected} FROM {table} WHERE "{timestamp}" <= ?',  # noqa: S608
            [cutoff.replace(tzinfo=None).isoformat()],
        ).fetchall()
        excluded = total - len(rows)
    else:
        rows = con.execute(f"SELECT {selected} FROM {table}").fetchall()  # noqa: S608
        excluded = 0

    deep = pd.DataFrame(rows, columns=columns)
    deep, dropped_private = _drop_forbidden(deep)
    deep = deep.add_prefix("deep_").rename(columns={f"deep_{join_right}": "_join_key"})
    merged = frame.merge(
        deep, left_on=cfg.enrichment.join_left, right_on="_join_key", how="left"
    ).drop(columns=["_join_key"], errors="ignore")

    matched = int(merged["deep_brand"].notna().sum()) if "deep_brand" in merged.columns else 0
    manifest.record_exclusion(
        "enrichment.deep_attributes",
        f"deep record {timestamp} > analysis reference instant ({cutoff})",
        int(excluded),
    )
    manifest.enrichment.update(
        {
            "enabled": True,
            "source_database": str(cfg.enrichment.database_path),
            "source_table": table,
            "source_rows_total": int(total),
            "source_rows_within_cutoff": int(len(deep)),
            "join": f"{cfg.enrichment.join_left} = {join_right}",
            "temporal_rule": f"{timestamp} <= run reference instant ({cutoff})",
            "advertisements_matched": matched,
            "coverage": round(matched / len(merged), 4) if len(merged) else None,
            "column_selection": rejection,
            "columns_added": len([c for c in merged.columns if c.startswith("deep_")]),
            "private_columns_dropped": sorted(dropped_private),
            "caveat": (
                "these attributes come from a SEPARATE scrape with its own timestamps; "
                "they describe the vehicle, not the state of the search on this run"
            ),
        }
    )
    manifest.warn(
        "enrichment",
        f"vehicle attributes for {matched}/{len(merged)} advertisements come from the deep "
        f"scraper database ({table}), not from monitoring snapshots",
    )
    return merged


def _join_trim_specs(
    frame: pd.DataFrame, con: Any, cfg: EdaConfig, manifest: Manifest
) -> pd.DataFrame:
    """Join trim-level technical specifications, pivoted here rather than read
    from the deep scraper's wide view.

    The view exists, but it projects each value as
    ``COALESCE(value_bool, value_num, value_text)``, and ``value_num`` is
    populated opportunistically by the measure parser. So a *string* value like
    ``(از پاییز 1402)`` -- "electronic stability control, available from autumn
    1402" -- surfaces as the bare number ``1402.0``, which in an analysis reads as
    a nonsensical ESC value rather than an availability note.

    This projection is type-aware instead:

    1. ``value_bool`` present            -> ``present`` / ``absent``
    2. ``value_num`` present *and the row is not a String* -> the number
    3. otherwise                         -> ``value_text``

    Every column added is a **trim-level** attribute: all listings of the same
    model-trim carry identical values, which makes them legitimate covariates and
    illegitimate per-car measurements. They are prefixed ``spec_`` and flagged in
    the feature catalogue.
    """
    key = cfg.enrichment.spec_join_column
    exists = con.execute(
        "SELECT name FROM sqlite_master WHERE name = 'trim_specs' AND type IN ('view','table')"
    ).fetchone()
    if not exists:
        manifest.warn("enrichment", "trim_specs absent; technical features not joined")
        return frame

    left_key = f"deep_{key}"
    if left_key not in frame.columns:
        manifest.warn(
            "enrichment",
            f"{left_key} not present after the attribute join; specifications skipped",
        )
        return frame

    rows = con.execute(
        "SELECT review_key, group_slug, item_slug, value_type, value_raw, value_text,"
        " value_bool, value_num FROM trim_specs"
    ).fetchall()
    if not rows:
        return frame
    long = pd.DataFrame(
        rows,
        columns=[
            "review_key",
            "group_slug",
            "item_slug",
            "value_type",
            "value_raw",
            "value_text",
            "value_bool",
            "value_num",
        ],
    )

    def project(row: pd.Series) -> Any:
        if pd.notna(row["value_bool"]):
            return "present" if int(row["value_bool"]) else "absent"
        if pd.notna(row["value_num"]) and str(row["value_type"]) != "String":
            return row["value_num"]
        text = row["value_text"] if pd.notna(row["value_text"]) else row["value_raw"]
        return text if pd.notna(text) else None

    long["projected"] = long.apply(project, axis=1)
    # Group-qualify the shared "other_features" slug so several groups' free-text
    # buckets do not collapse into one column.
    long["column"] = np.where(
        long["item_slug"].astype(str).str.contains("other_features"),
        long["group_slug"].astype(str) + "__" + long["item_slug"].astype(str),
        long["item_slug"].astype(str),
    )
    duplicated = int(long.duplicated(subset=["review_key", "column"]).sum())
    long = long.drop_duplicates(subset=["review_key", "column"], keep="first")

    specs = long.pivot(index="review_key", columns="column", values="projected").reset_index()
    specs.columns.name = None
    specs = specs.add_prefix("spec_").rename(columns={f"spec_{key}": "_spec_key"})

    merged = frame.merge(specs, left_on=left_key, right_on="_spec_key", how="left").drop(
        columns=["_spec_key"], errors="ignore"
    )

    spec_columns = [c for c in merged.columns if c.startswith("spec_")]
    matched = int(merged[spec_columns[0]].notna().sum()) if spec_columns else 0
    manifest.enrichment["specifications"] = {
        "source_table": "trim_specs",
        "projection": (
            "value_bool -> present/absent; value_num when the row is not a String; "
            "otherwise value_text"
        ),
        "why_not_the_wide_view": (
            "v_trim_specs_wide coalesces value_num ahead of value_text, so a string "
            "value such as '(from autumn 1402)' is projected as the number 1402.0"
        ),
        "join": f"{left_key} = {key}",
        "distinct_trims": int(specs["_spec_key"].nunique())
        if "_spec_key" in specs
        else int(len(specs)),
        "feature_columns_added": len(spec_columns),
        "duplicate_item_rows_dropped": duplicated,
        "advertisements_matched": matched,
        "coverage": round(matched / len(merged), 4) if len(merged) else None,
        "grain_caveat": (
            "TRIM-LEVEL attributes: every listing of the same model-trim carries "
            "identical values. Valid as covariates; NOT per-vehicle measurements. "
            "A breakdown such as 'sunroof by city' describes the trim mix in that "
            "city, not a count of individually verified cars."
        ),
    }
    manifest.warn(
        "enrichment",
        f"{len(spec_columns)} trim-level specification features joined for "
        f"{matched}/{len(merged)} advertisements; they vary by trim, not by listing",
    )
    return merged


#: Fallback list when `include_all_columns` is off.
_CURATED_DEEP_COLUMNS: list[str] = [
    "ad_id",
    "brand",
    "model",
    "trim",
    "year_jalali",
    "body_type_fa",
    "price_toman",
    "mileage_km",
    "condition_new_used",
    "body_status",
    "transmission",
    "fuel_type",
    "province",
    "city",
    "seller_type",
    "description_scrubbed",
    "image_count",
    "scraped_at",
]


# ---------------------------------------------------------------------------
# Dataset B - longitudinal panel
# ---------------------------------------------------------------------------


def build_panel(
    db: ReadOnlyDatabase, context: AnalysisContext, cfg: EdaConfig, manifest: Manifest
) -> pd.DataFrame:
    """One row per ``(run_id, advertisement_id)`` up to and including the run.

    ``run_is_valid`` is carried explicitly, and ``was_seen`` is blanked for runs
    that were not valid: a scrape that failed halfway did not observe an absence,
    it merely failed, and letting those rows look like absences is precisely the
    corruption the monitor's health gate exists to prevent.
    """
    obs_columns = _existing(
        db,
        "daily_ad_observations",
        [
            "id",
            "run_id",
            "advertisement_id",
            "was_seen",
            "position",
            "page_number",
            "card_price_normalized",
            "card_price_raw",
            "card_mileage_normalized",
            "card_year",
            "card_location",
            "card_hash",
            "observed_at",
        ],
    )
    run_columns = _existing(
        db,
        "monitoring_runs",
        [
            "status",
            "scheduled_for",
            "started_at",
            "finished_at",
            "comparison_applied",
            "configuration_hash",
            "discovered_count",
        ],
    )
    ad_columns = _existing(
        db,
        "advertisements",
        [
            "platform_ad_id",
            "current_status",
            "consecutive_misses",
            "vehicle_entity_id",
            "sale_confidence",
            "sale_label",
            "detail_availability",
            "filter_exit_reason",
            "left_truncated",
            "eligible_for_duration_ranking",
        ],
    )

    # `observation_run_id` rather than `run_id`: the provenance stamp adds the
    # ANALYSIS run id, and in a panel whose grain is the observation run, letting
    # one shadow the other would silently collapse every row onto one run.
    def _obs_alias(column: str) -> str:
        if column == "run_id":
            return "o.run_id AS observation_run_id"
        if column == "advertisement_id":
            return "o.advertisement_id"
        return f"o.{column} AS obs_{column}"

    select = ", ".join(
        [
            *(_obs_alias(c) for c in obs_columns),
            *(f"r.{c} AS run_{c}" for c in run_columns),
            *(f"a.{c} AS ad_{c}" for c in ad_columns),
        ]
    )
    frame = db.query_frame(
        f"SELECT {select}"
        " FROM daily_ad_observations o"
        " JOIN monitoring_runs r ON r.id = o.run_id"
        " JOIN advertisements a ON a.id = o.advertisement_id"
        " WHERE o.run_id <= ?"
        " ORDER BY o.run_id, o.advertisement_id",
        [context.run_id],
        label="panel.base",
    )
    if frame.empty:
        manifest.warn("panel", "no observations exist up to the selected run")
        return stamp(frame, context)

    frame["run_is_valid"] = (
        frame.get("run_status", pd.Series(["unknown"] * len(frame)))
        .astype(str)
        .isin(VALID_RUN_STATUSES)
    )

    invalid = int((~frame["run_is_valid"]).sum())
    if invalid:
        # Keep the rows (they are evidence that a run happened) but strip the
        # absence signal, which is the only part that would mislead.
        # Nullable boolean, not float-with-NaN: the value is "seen", "not seen" or
        # "we cannot say", and a float column invites arithmetic on a truth value.
        frame["was_seen_trusted"] = (
            frame["obs_was_seen"].astype("boolean").where(frame["run_is_valid"].astype(bool))
        )
        manifest.record_exclusion(
            "panel.absence_signal",
            "observation belongs to a run whose status is not 'valid'; was_seen_trusted "
            "set to NA so it cannot be counted as an absence",
            invalid,
        )
        manifest.warn(
            "panel",
            f"{invalid} observation row(s) come from non-valid runs; their was_seen is "
            "retained as obs_was_seen but excluded from was_seen_trusted",
        )
    else:
        frame["was_seen_trusted"] = frame["obs_was_seen"].astype("boolean")

    frame = _attach_panel_events(db, frame, context)
    frame = _attach_panel_price_changes(db, frame, context)

    # Status before/after each run, from the event log rather than recomputed.
    if "event_previous_status" in frame.columns:
        frame["status_before_run"] = frame["event_previous_status"]
        frame["status_after_run"] = frame["event_new_status"]

    frame, _dropped = _drop_forbidden(frame)
    frame, _scrubbed = _scrub_freetext(frame)
    manifest.source_tables.setdefault("monitoring_runs", db.row_count("monitoring_runs"))
    return stamp(frame, context)


def _attach_panel_events(
    db: ReadOnlyDatabase, frame: pd.DataFrame, context: AnalysisContext
) -> pd.DataFrame:
    if not db.has_table("advertisement_events"):
        return frame
    # GROUP_CONCAT is SQLite-only. PostgreSQL needs string_agg(x, ','), and a
    # query that works on the development backend but not the production one is a
    # bug that only shows up in production.
    concat = "GROUP_CONCAT(event_type)" if db.dialect == "sqlite" else "string_agg(event_type, ',')"
    events = db.query_frame(
        "SELECT run_id AS observation_run_id, advertisement_id,"
        f" {concat} AS event_types,"
        " COUNT(*) AS event_count,"
        " MIN(previous_status) AS event_previous_status,"
        " MAX(new_status) AS event_new_status"
        " FROM advertisement_events WHERE run_id <= ?"
        " GROUP BY run_id, advertisement_id",
        [context.run_id],
        label="panel.events",
    )
    if events.empty:
        return frame
    return frame.merge(events, on=["observation_run_id", "advertisement_id"], how="left")


def _attach_panel_price_changes(
    db: ReadOnlyDatabase, frame: pd.DataFrame, context: AnalysisContext
) -> pd.DataFrame:
    if not db.has_table("price_changes"):
        return frame
    changes = db.query_frame(
        "SELECT run_id AS observation_run_id, advertisement_id,"
        " change_type AS price_change_type,"
        " old_price AS price_change_old, new_price AS price_change_new"
        " FROM price_changes WHERE run_id <= ?",
        [context.run_id],
        label="panel.price_changes",
    )
    if changes.empty:
        frame["price_change_type"] = None
        return frame
    changes = changes.drop_duplicates(
        subset=["observation_run_id", "advertisement_id"], keep="last"
    )
    return frame.merge(changes, on=["observation_run_id", "advertisement_id"], how="left")


# ---------------------------------------------------------------------------
# Dataset C - vehicle entity
# ---------------------------------------------------------------------------


def build_vehicle_dataset(
    db: ReadOnlyDatabase,
    cross_section: pd.DataFrame,
    context: AnalysisContext,
    cfg: EdaConfig,
    manifest: Manifest,
) -> pd.DataFrame:
    """One row per physical vehicle, aggregating its listings.

    Built from the cross-section rather than from ``vehicle_entities`` so that the
    aggregation is reproducible from the analysed population, and so a repost chain
    contributes **one** row — never one per listing.
    """
    if cross_section.empty:
        return cross_section

    frame = cross_section.copy()
    key = "vehicle_entity_id"
    if key not in frame.columns or frame[key].isna().all():
        # Fall back to the listing id, and say so: without linkage every listing is
        # its own vehicle, which understates market presence for reposted cars.
        frame[key] = frame["platform_ad_id"]
        manifest.warn(
            "vehicle_dataset",
            "no vehicle_entity_id present; each advertisement is treated as its own "
            "vehicle, so repost chains are not collapsed",
        )
    frame[key] = frame[key].fillna(frame["platform_ad_id"])

    groups: list[dict[str, Any]] = []
    for entity_id, block in frame.groupby(key, sort=True):
        groups.append(_summarise_entity(str(entity_id), block, cfg))
    vehicles = pd.DataFrame(groups)

    manifest.source_tables.setdefault(
        "vehicle_entities",
        db.row_count("vehicle_entities") if db.has_table("vehicle_entities") else 0,
    )
    return stamp(vehicles, context)


def _numeric_sum(block: pd.DataFrame, column: str) -> int:
    """Sum a possibly-absent numeric column. Absent means zero occurrences."""
    if column not in block.columns:
        return 0
    return int(pd.to_numeric(block[column], errors="coerce").fillna(0).sum())


def _summarise_entity(entity_id: str, block: pd.DataFrame, cfg: EdaConfig) -> dict[str, Any]:
    """Collapse one vehicle's listings into a single record."""

    def timestamps(column: str) -> list[datetime]:
        """Every parseable instant in one column, Nones dropped."""
        if column not in block.columns:
            return []
        parsed = [parse_ts(v) for v in block[column]]
        return [t for t in parsed if t is not None]

    ordered = block.copy()
    if "first_seen_at" in ordered.columns:
        ordered["_order"] = ordered["first_seen_at"].map(parse_ts)
        ordered = ordered.sort_values("_order", na_position="first")
    latest = ordered.iloc[-1]
    earliest = ordered.iloc[0]

    prices = pd.to_numeric(block.get("price_toman"), errors="coerce").dropna()
    statuses = [str(s) for s in block.get("current_status", pd.Series(dtype=str))]

    first_seen = parse_ts(earliest.get("first_seen_at"))
    seen_times = timestamps("last_seen_at")
    last_seen = max(seen_times) if seen_times else None
    # Only the final listing's absence ends the vehicle's observed presence; an
    # earlier listing's first_missing_at is where the chain handed over.
    final_missing = parse_ts(latest.get("first_missing_at"))
    latest_status = str(latest.get("current_status") or "unknown")
    reposted = latest_status == "reposted" or bool(latest.get("repost_parent_ad_id"))

    published = timestamps("published_at")
    publication_lower = min(published).isoformat() if published else None
    # Without a reliable publication instant the only defensible upper bound on the
    # publication time is the first sighting.
    publication_upper = (
        min(published).isoformat()
        if published
        else (first_seen.isoformat() if first_seen else None)
    )

    observed_duration = (
        round((last_seen - first_seen).total_seconds() / 86400.0, 4)
        if first_seen and last_seen
        else None
    )
    eligible = (
        bool(pd.Series([latest.get("eligible_for_duration_ranking")]).astype("boolean").iloc[0])
        if "eligible_for_duration_ranking" in block.columns
        else False
    )

    market_duration = None
    if eligible and published and last_seen:
        market_duration = round((last_seen - min(published)).total_seconds() / 86400.0, 4)

    minimum = observed_duration
    maximum = (
        round((final_missing - first_seen).total_seconds() / 86400.0, 4)
        if first_seen and final_missing and latest_status in ("likely_removed", "likely_sold")
        else None
    )
    estimated = (
        round((minimum + maximum) / 2, 4) if minimum is not None and maximum is not None else None
    )

    return {
        "vehicle_entity_id": entity_id,
        "advertisement_count": int(len(block)),
        "advertisement_ids": ",".join(sorted(str(v) for v in block["platform_ad_id"])),
        "first_seen_at": first_seen.isoformat() if first_seen else None,
        "last_seen_at": last_seen.isoformat() if last_seen else None,
        "publication_lower_bound": publication_lower,
        "publication_upper_bound": publication_upper,
        "observed_monitoring_duration": observed_duration,
        "estimated_market_duration": market_duration,
        "minimum_disappearance_duration": minimum,
        "maximum_disappearance_duration": maximum,
        "estimated_disappearance_duration": estimated,
        "was_reposted": bool(reposted or len(block) > 1),
        "repost_count": max(0, int(len(block)) - 1),
        "latest_status": latest_status,
        "latest_price": float(prices.iloc[-1]) if len(prices) else None,
        "minimum_price": float(prices.min()) if len(prices) else None,
        "maximum_price": float(prices.max()) if len(prices) else None,
        "price_change_count": _numeric_sum(block, "events_price_changed"),
        # Inference carried at vehicle grain so a chain is one candidate, not three.
        "sale_evidence_score": float(latest.get("sale_confidence") or 0.0),
        "sale_label": str(latest.get("sale_label") or "unknown"),
        "left_truncated": bool(earliest.get("left_truncated", True)),
        "right_censored": latest_status not in ("likely_removed", "likely_sold"),
        "eligible_for_duration_ranking": eligible,
        "statuses_in_chain": ",".join(sorted(set(statuses))),
        "first_advertisement_id": str(earliest.get("platform_ad_id")),
        "latest_advertisement_id": str(latest.get("platform_ad_id")),
    }


def dataset_row_counts(datasets: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: int(len(frame)) for name, frame in datasets.items()}
