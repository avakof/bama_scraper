"""Daily report generation.

The report's job is to keep four things visibly separate:

    observed disappearance -> the listing was absent from a valid run
    likely removal         -> absent for the confirmation threshold
    likely sale            -> an inference with a confidence and its components
    confirmed sale         -> never produced by this system

They get their own files and their own columns. The dashboard states the
distinction in prose as well, because a CSV column name is easy to misread once
it reaches a spreadsheet.
"""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .analytics import (
    cohort_analysis,
    daily_inventory,
    fastest_disappearing,
    filter_exit_report,
    left_truncated_disappearances,
    market_statistics,
    survival_dataset,
    time_to_disappearance,
)
from .config import MonitorConfig
from .db import Database, parse_ts
from .models import EventType, RunHealth
from .repository import Repository
from .vehicle_grain import rebuild_vehicle_entities, vehicle_durations

DISCLAIMER = (
    "An advertisement disappearing from the observed inventory is a FACT. "
    "A sale is an INFERENCE. This report never reports a confirmed sale: "
    "'likely_sold' means the observed evidence is consistent with a sale, and "
    "every score is decomposed into auditable components. Causes of "
    "disappearance also include expiry, seller removal, moderation, a price or "
    "filter change, a repost under a new id, or a scraping failure."
)


#: Column order for the per-event listings, declared so that a day with no such
#: event still writes a file with the real header. A consumer reading
#: `missing_ads.csv` every day must not have its columns disappear on a quiet day.
EVENT_COLUMNS = [
    "platform_ad_id",
    "canonical_url",
    "previous_status",
    "new_status",
    "event_at",
    "confidence",
    "evidence",
]

PRICE_CHANGE_COLUMNS = [
    "platform_ad_id",
    "canonical_url",
    "change_type",
    "old_price",
    "new_price",
    "absolute_change",
    "percentage_change",
    "changed_at",
]

ERROR_COLUMNS = [
    "stage",
    "url",
    "error_type",
    "error_message",
    "retryable",
    "attempt",
    "created_at",
]


#: Immutable identifiers stamped onto every exported row. Counts from a live
#: marketplace are meaningless without them: "the full pass found 2,816" invites
#: the reader to treat a changing inventory as a scraper inconsistency, whereas
#: "run 7, 13:00-13:04 on 2026-07-28, observed 2,816" is a fact about a moment.
PROVENANCE_COLUMNS = [
    "run_id",
    "scheduled_for",
    "run_started_at",
    "run_finished_at",
    "search_configuration_hash",
    "scraper_version",
    "monitor_version",
]


def run_provenance(run: dict[str, Any]) -> dict[str, Any]:
    """The identifiers that make a count interpretable."""
    return {
        "run_id": run.get("id"),
        "scheduled_for": run.get("scheduled_for"),
        "run_started_at": run.get("started_at"),
        "run_finished_at": run.get("finished_at"),
        "search_configuration_hash": run.get("configuration_hash"),
        "scraper_version": run.get("scraper_version"),
        "monitor_version": run.get("monitor_version"),
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    *,
    provenance: dict[str, Any] | None = None,
) -> int:
    """Write one CSV, header always present.

    ``columns`` pins the schema for files that are frequently empty; without it an
    empty result has no keys to read a header from. ``provenance`` prefixes every
    row with the run identifiers, so a file cannot be separated from the run that
    produced it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if provenance:
        rows = [{**provenance, **row} for row in rows]
        columns = PROVENANCE_COLUMNS + list(columns) if columns else columns
    fieldnames = columns or (list(rows[0]) if rows else ["note"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def generate_report(
    db: Database, cfg: MonitorConfig, *, run_id: int | None = None, date: str | None = None
) -> dict[str, Any]:
    """Write the report tree for one run and return a manifest."""
    repo = Repository(db)

    run = None
    if run_id is not None:
        run = repo.get_run(run_id)
    elif date is not None:
        run = db.fetchone(
            "SELECT * FROM monitoring_runs WHERE substr(CAST(scheduled_for AS TEXT),1,10)=?"
            " ORDER BY id DESC LIMIT 1",
            [date],
        )
    if run is None:
        run = db.fetchone("SELECT * FROM monitoring_runs ORDER BY id DESC LIMIT 1")
    if run is None:
        raise ValueError("no monitoring runs exist yet")

    run_id = int(run["id"])
    scheduled = parse_ts(run.get("scheduled_for"))
    label = (scheduled or datetime.now()).strftime("%Y-%m-%d")
    out = cfg.reports_dir / label
    out.mkdir(parents=True, exist_ok=True)

    provenance = run_provenance(run)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "date": label,
        # Repeated in the manifest so a bundle handed on by itself still says
        # which run it describes.
        "provenance": provenance,
        "files": {},
        "rows": {},
    }

    # -- run summary -------------------------------------------------------
    summary = {
        "run_id": run_id,
        "date": label,
        "scheduled_for": str(run.get("scheduled_for")),
        "started_at": str(run.get("started_at")),
        "finished_at": str(run.get("finished_at")),
        "timezone": run.get("timezone"),
        "run_status": run.get("status"),
        "comparison_applied": bool(run.get("comparison_applied")),
        "health_reason": run.get("health_reason"),
        "termination_reason": run.get("termination_reason"),
        "counters": {
            key: run.get(key)
            for key in (
                "discovered_count",
                "new_count",
                "active_count",
                "missing_count",
                "removed_count",
                "reappeared_count",
                "reposted_count",
                "likely_sold_count",
                "detail_success_count",
                "detail_failure_count",
                "duplicate_count",
                "error_count",
            )
        },
        "current_status_totals": repo.count_by_status(),
        "interpretation": {
            "observed_disappearance": "missing_count — absent from this valid run",
            "likely_removal": "removed_count — absent for the confirmation threshold",
            "likely_sale": "likely_sold_count — INFERRED, not confirmed",
            "confirmed_sale": "not produced by this system",
        },
        "disclaimer": DISCLAIMER,
        "removal_confirmation_misses": cfg.removal_confirmation_misses,
        "configuration_hash": run.get("configuration_hash"),
        "provenance": run_provenance(run),
        "measurement_notes": {
            "counts": (
                "every count in this bundle describes ONE run of a live marketplace. "
                "Quote it with its run id and timestamps; inventory differences "
                "between runs are market churn, not scraper inconsistency."
            ),
            "sale_evidence_score": (
                "heuristic additive score in [0,1], NOT a calibrated probability. "
                "0.65 does not mean '65% of these sold'. Validate against labelled "
                "outcomes before reading it as a rate."
            ),
            "duration_basis": (
                "duration_* columns are measured from first observation. "
                "estimated_market_* columns are measured from publication and are "
                "populated only for listings whose whole life was observed."
            ),
            "left_truncation": (
                "listings already active when monitoring began are left-truncated: "
                "their earlier life was never seen, so they are excluded from "
                "duration rankings and reported in left_truncated_excluded.csv."
            ),
            "filter_exit": (
                "active_outside_filter means the listing is still for sale but no "
                "longer matches the monitored search. It is not a disappearance."
            ),
        },
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest["files"]["run_summary.json"] = str(out / "run_summary.json")

    # -- daily inventory ---------------------------------------------------
    inventory = daily_inventory(db)
    manifest["rows"]["daily_inventory"] = _write_csv(
        out / "daily_inventory.csv", inventory, provenance=provenance
    )

    # -- per-category listings --------------------------------------------
    def events_csv(name: str, event_type: EventType) -> None:
        rows = [
            {
                "platform_ad_id": r.get("platform_ad_id"),
                "canonical_url": r.get("canonical_url"),
                "previous_status": r.get("previous_status"),
                "new_status": r.get("new_status"),
                "event_at": r.get("event_at"),
                "confidence": r.get("confidence"),
                "evidence": json.dumps(
                    Database.json_load(r.get("evidence_json")), ensure_ascii=False, default=str
                ),
            }
            for r in repo.events_for_run(run_id, event_type)
        ]
        manifest["rows"][name] = _write_csv(
            out / f"{name}.csv", rows, EVENT_COLUMNS, provenance=provenance
        )

    events_csv("new_ads", EventType.DISCOVERED)
    events_csv("missing_ads", EventType.MISSING_FIRST_TIME)
    events_csv("likely_removed_ads", EventType.REMOVAL_CONFIRMED)
    events_csv("reappeared_ads", EventType.REAPPEARED)
    events_csv("possible_reposts", EventType.POSSIBLE_REPOST)
    events_csv("likely_sold_ads", EventType.LIKELY_SOLD)

    # -- price changes -----------------------------------------------------
    price_rows = [
        {
            "platform_ad_id": r.get("platform_ad_id"),
            "canonical_url": r.get("canonical_url"),
            "change_type": r.get("change_type"),
            "old_price": r.get("old_price"),
            "new_price": r.get("new_price"),
            "absolute_change": r.get("absolute_change"),
            "percentage_change": r.get("percentage_change"),
            "changed_at": r.get("changed_at"),
        }
        for r in repo.price_changes_for_run(run_id)
    ]
    manifest["rows"]["price_changes"] = _write_csv(
        out / "price_changes.csv", price_rows, PRICE_CHANGE_COLUMNS, provenance=provenance
    )

    # -- errors ------------------------------------------------------------
    error_rows = [{k: r.get(k) for k in ERROR_COLUMNS} for r in repo.errors_for_run(run_id)]
    manifest["rows"]["scrape_errors"] = _write_csv(
        out / "scrape_errors.csv", error_rows, ERROR_COLUMNS, provenance=provenance
    )

    # -- analytics extras --------------------------------------------------
    # Named for what is measured. A listing leaving the inventory is a
    # disappearance; calling the file time_to_sale would assert the cause.
    manifest["rows"]["time_to_disappearance"] = _write_csv(
        out / "time_to_disappearance.csv", time_to_disappearance(db), provenance=provenance
    )
    # Ranking is restricted to listings whose whole life was observed. Everything
    # excluded is written to its own file with the reason, so an exclusion is
    # never mistaken for an absence of data.
    manifest["rows"]["fastest_disappearing"] = _write_csv(
        out / "fastest_disappearing.csv",
        fastest_disappearing(db, min_confidence=cfg.scoring.ranking_min_confidence),
        provenance=provenance,
    )
    manifest["rows"]["left_truncated_excluded"] = _write_csv(
        out / "left_truncated_excluded.csv",
        left_truncated_disappearances(db, min_confidence=cfg.scoring.ranking_min_confidence),
        provenance=provenance,
    )
    manifest["rows"]["survival_dataset"] = _write_csv(
        out / "survival_dataset.csv", survival_dataset(db), provenance=provenance
    )
    # Vehicle grain: durations span linked reposts, so a relisted car is one long
    # life rather than several short ones.
    rebuild_vehicle_entities(db)
    manifest["rows"]["vehicle_durations"] = _write_csv(
        out / "vehicle_durations.csv", vehicle_durations(db), provenance=provenance
    )
    manifest["rows"]["filter_exits"] = _write_csv(
        out / "filter_exits.csv", filter_exit_report(db), provenance=provenance
    )
    manifest["rows"]["cohorts"] = _write_csv(
        out / "cohorts.csv", cohort_analysis(db), provenance=provenance
    )
    for dimension in (
        "brand",
        "model",
        "year",
        "city",
        "seller_type",
        "price_range",
        "mileage_range",
    ):
        manifest["rows"][f"market_{dimension}"] = _write_csv(
            out / f"market_by_{dimension}.csv",
            market_statistics(db, dimension),
            provenance=provenance,
        )

    # -- dashboard ---------------------------------------------------------
    dashboard = _dashboard_html(summary, inventory, db, cfg, run_id)
    (out / "dashboard.html").write_text(dashboard, encoding="utf-8")
    manifest["files"]["dashboard.html"] = str(out / "dashboard.html")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _dashboard_html(
    summary: dict[str, Any],
    inventory: list[dict[str, Any]],
    db: Database,
    cfg: MonitorConfig,
    run_id: int,
) -> str:
    """Single-file dashboard. No external assets, so it works offline."""
    counters = summary["counters"]
    status_totals = summary["current_status_totals"]
    valid = summary["run_status"] == str(RunHealth.VALID)

    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else "—"))

    rows = "".join(
        f"<tr><td>{esc(r.get('date'))}</td><td>{esc(r.get('total_active'))}</td>"
        f"<td>{esc(r.get('new_today'))}</td><td>{esc(r.get('missing_today'))}</td>"
        f"<td>{esc(r.get('likely_removed_today'))}</td>"
        f"<td>{esc(r.get('reappeared_today'))}</td>"
        f"<td>{esc(r.get('likely_sold_today'))}</td></tr>"
        for r in inventory[:30]
    )
    status_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in sorted(status_totals.items())
    )
    fastest = fastest_disappearing(db, min_confidence=cfg.scoring.ranking_min_confidence, limit=15)
    fastest_rows = "".join(
        f"<tr><td>{esc(r['advertisement_id'])}</td><td>{esc(r.get('brand'))}</td>"
        f"<td>{esc(r.get('model'))}</td><td>{esc(r.get('year'))}</td>"
        f"<td>{esc(r.get('minimum_active_days'))}</td>"
        f"<td>{esc(r.get('estimated_active_days'))}</td>"
        f"<td>{esc(r.get('maximum_active_days'))}</td>"
        f"<td>{esc(r.get('sale_confidence'))}</td><td>{esc(r.get('sale_label'))}</td></tr>"
        for r in fastest
    )

    banner_class = "ok" if valid else "bad"
    banner_text = (
        "Run valid — status transitions were applied."
        if valid
        else f"Run {esc(summary['run_status'])} — NO status transitions were applied. "
        f"{esc(summary.get('health_reason'))}"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Bama monitor — {esc(summary["date"])}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;
       background:#fbfbfd;color:#1c1c1e}}
 h1{{margin:0 0 4px}} h2{{margin:28px 0 8px;font-size:16px}}
 .muted{{color:#6b6b70}}
 .banner{{padding:10px 14px;border-radius:8px;margin:16px 0;font-weight:600}}
 .banner.ok{{background:#e7f7ec;color:#12602f}} .banner.bad{{background:#fdeceb;color:#8c1d18}}
 .note{{background:#fff8e6;border-left:4px solid #e0a800;padding:12px 14px;border-radius:6px;
        margin:16px 0}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
 .card{{background:#fff;border:1px solid #e5e5ea;border-radius:8px;padding:12px}}
 .card .n{{font-size:22px;font-weight:700}} .card .l{{font-size:12px;color:#6b6b70}}
 table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;
        border:1px solid #e5e5ea;border-radius:8px;overflow:hidden}}
 th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #f0f0f2}}
 th{{background:#f5f5f7;font-weight:600}}
 .scroll{{overflow-x:auto}}
 code{{background:#f0f0f2;padding:1px 5px;border-radius:4px}}
</style></head><body>
<h1>Bama listing monitor</h1>
<div class="muted">{esc(summary["date"])} · run #{run_id} · scheduled
 {esc(summary["scheduled_for"])} ({esc(summary["timezone"])})</div>

<div class="banner {banner_class}">{banner_text}</div>

<div class="note"><strong>How to read this report.</strong> {esc(DISCLAIMER)}</div>

<h2>This run</h2>
<div class="grid">
 <div class="card"><div class="n">{esc(counters.get("discovered_count"))}</div>
   <div class="l">discovered</div></div>
 <div class="card"><div class="n">{esc(counters.get("new_count"))}</div>
   <div class="l">new</div></div>
 <div class="card"><div class="n">{esc(counters.get("missing_count"))}</div>
   <div class="l">observed disappearance</div></div>
 <div class="card"><div class="n">{esc(counters.get("removed_count"))}</div>
   <div class="l">likely removal (inferred)</div></div>
 <div class="card"><div class="n">{esc(counters.get("reappeared_count"))}</div>
   <div class="l">reappeared</div></div>
 <div class="card"><div class="n">{esc(counters.get("reposted_count"))}</div>
   <div class="l">reposted (not a sale)</div></div>
 <div class="card"><div class="n">{esc(counters.get("likely_sold_count"))}</div>
   <div class="l">likely sale (inferred)</div></div>
 <div class="card"><div class="n">0</div>
   <div class="l">confirmed sale (never inferred)</div></div>
</div>

<h2>Current status totals</h2>
<div class="scroll"><table><tr><th>status</th><th>advertisements</th></tr>
{status_rows}</table></div>

<h2>Daily inventory (valid runs only)</h2>
<div class="scroll"><table>
<tr><th>date</th><th>total</th><th>new</th><th>disappeared</th><th>likely removed</th>
<th>reappeared</th><th>likely sold</th></tr>
{rows or '<tr><td colspan="7" class="muted">no valid runs yet</td></tr>'}</table></div>

<h2>Fastest disappearing (confidence &ge; {esc(cfg.scoring.ranking_min_confidence)})</h2>
<p class="muted">Durations are interval-censored: the true removal time lies between
the last sighting and the first miss, so a minimum, an estimate and a maximum are
all reported. The estimate is a midpoint, not an observed sale time.</p>
<div class="scroll"><table>
<tr><th>ad id</th><th>brand</th><th>model</th><th>year</th><th>min days</th>
<th>est days</th><th>max days</th><th>confidence</th><th>label</th></tr>
{fastest_rows or '<tr><td colspan="9" class="muted">nothing meets the confidence floor yet</td></tr>'}
</table></div>

<h2>Method</h2>
<ul>
 <li>Removal requires <code>{esc(cfg.removal_confirmation_misses)}</code> consecutive
   misses in <em>valid</em> runs. One miss is only <code>missing_once</code>.</li>
 <li>An invalid, partial or blocked run never changes any advertisement's status.</li>
 <li>A reappearance always supersedes an earlier removal or sale inference.</li>
 <li>A repost links the new advertisement to its parent and <em>reduces</em>
   sale confidence — the vehicle is still on the market.</li>
</ul>
</body></html>
"""
