"""Adversarial integrity audit.

Written to *fail*. Every check here is a way the datasets could be wrong while
still producing a plausible-looking chart, which is the dangerous failure mode:
a crash gets fixed, a quietly wrong median gets quoted.

Severity drives behaviour:

``critical``
    the dataset cannot be analysed. Substantive EDA stops.
``error``
    a specific analysis is invalid and is suppressed.
``warning``
    the result stands but needs a caveat in the report.
``informational``
    worth recording, changes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bama_monitor.db import parse_ts

from .config import EdaConfig
from .database import ReadOnlyDatabase
from .models import PRIVATE_COLUMNS, AnalysisContext, Finding, Severity, is_texty

#: A Jalali production year outside this range is a parsing failure, not a car.
PLAUSIBLE_YEAR_RANGE = (1300, 1450)

#: Contact-shaped digit runs. Iranian mobile numbers start 09 and run 11 digits;
#: the boundary guards stop a 2,000,000,000 price or a hex digest matching.
_PHONE_PATTERN = r"(?<![0-9a-fA-F])09\d{9}(?![0-9a-fA-F])"


class CriticalIntegrityError(RuntimeError):
    """A critical finding makes the analysis untrustworthy."""

    def __init__(self, findings: list[Finding]) -> None:
        detail = "; ".join(f"{f.check} ({f.count})" for f in findings)
        super().__init__(f"critical integrity findings, analysis halted: {detail}")
        self.findings = findings


def audit(
    db: ReadOnlyDatabase,
    datasets: dict[str, pd.DataFrame],
    context: AnalysisContext,
    cfg: EdaConfig,
) -> list[Finding]:
    """Run every check that the present schema supports."""
    findings: list[Finding] = []
    cross = datasets.get("advertisement_cross_section", pd.DataFrame())
    panel = datasets.get("longitudinal_panel", pd.DataFrame())
    vehicles = datasets.get("vehicle_entity_dataset", pd.DataFrame())

    findings += _identity_checks(db, cross)
    findings += _referential_checks(db)
    findings += _temporal_checks(db, cross, context)
    findings += _value_range_checks(cross, cfg)
    findings += _duration_bound_checks(cross, vehicles)
    findings += _semantic_consistency_checks(cross, panel)
    findings += _privacy_checks(datasets)
    findings += _provenance_checks(datasets)
    # Every check is returned, including the ones that found nothing. A quality
    # report that lists only failures cannot be distinguished from one where the
    # checks never ran.
    return findings


# -- identity ---------------------------------------------------------------


def _identity_checks(db: ReadOnlyDatabase, cross: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []

    duplicates = db.fetchall(
        "SELECT platform_ad_id, COUNT(*) AS n FROM advertisements"
        " GROUP BY platform_ad_id HAVING COUNT(*) > 1"
    )
    out.append(
        Finding(
            "duplicate_platform_ad_id",
            Severity.CRITICAL,
            len(duplicates),
            "the same platform advertisement id appears on several rows; every "
            "per-advertisement count would be inflated",
            [r["platform_ad_id"] for r in duplicates],
        )
    )

    dup_urls = db.fetchall(
        "SELECT canonical_url, COUNT(*) AS n FROM advertisements"
        " GROUP BY canonical_url HAVING COUNT(*) > 1"
    )
    out.append(
        Finding(
            "duplicate_canonical_url",
            Severity.CRITICAL,
            len(dup_urls),
            "one URL maps to several advertisement rows",
            [r["canonical_url"] for r in dup_urls],
        )
    )

    if not cross.empty and {"platform_ad_id", "canonical_url"} <= set(cross.columns):
        mismatched = cross[
            ~cross.apply(
                lambda r: str(r["platform_ad_id"]).lower() in str(r["canonical_url"]).lower(),
                axis=1,
            )
        ]
        out.append(
            Finding(
                "ad_id_url_mismatch",
                Severity.ERROR,
                len(mismatched),
                "the advertisement id does not appear in its own canonical URL, so one "
                "of the two identifiers is wrong",
                mismatched["platform_ad_id"].head().tolist(),
            )
        )

        dup_rows = cross[cross.duplicated(subset=["platform_ad_id"], keep=False)]
        out.append(
            Finding(
                "cross_section_duplicate_rows",
                Severity.CRITICAL,
                len(dup_rows),
                "an advertisement appears more than once in the cross-section; the "
                "snapshot join is not one-to-one",
                dup_rows["platform_ad_id"].head().tolist(),
            )
        )

    dup_obs = db.fetchall(
        "SELECT run_id, advertisement_id, COUNT(*) AS n FROM daily_ad_observations"
        " GROUP BY run_id, advertisement_id HAVING COUNT(*) > 1"
    )
    out.append(
        Finding(
            "duplicate_run_observation",
            Severity.CRITICAL,
            len(dup_obs),
            "the same advertisement was observed twice in one run",
            [f"run={r['run_id']} ad={r['advertisement_id']}" for r in dup_obs],
        )
    )
    return out


# -- referential integrity --------------------------------------------------


def _referential_checks(db: ReadOnlyDatabase) -> list[Finding]:
    out: list[Finding] = []
    orphan_specs = [
        ("orphan_observations", "daily_ad_observations", "advertisement_id", Severity.CRITICAL),
        ("orphan_events", "advertisement_events", "advertisement_id", Severity.ERROR),
        ("orphan_snapshots", "advertisement_snapshots", "advertisement_id", Severity.ERROR),
        ("orphan_price_changes", "price_changes", "advertisement_id", Severity.WARNING),
    ]
    for name, table, column, severity in orphan_specs:
        if not db.has_table(table):
            continue
        count = int(
            db.scalar(
                f"SELECT COUNT(*) FROM {table} t"  # noqa: S608
                f" LEFT JOIN advertisements a ON a.id = t.{column}"
                " WHERE a.id IS NULL"
            )
            or 0
        )
        out.append(
            Finding(
                name,
                severity,
                count,
                f"{table} rows reference an advertisement that does not exist",
            )
        )

    if db.has_table("daily_ad_observations"):
        count = int(
            db.scalar(
                "SELECT COUNT(*) FROM daily_ad_observations o"
                " LEFT JOIN monitoring_runs r ON r.id = o.run_id WHERE r.id IS NULL"
            )
            or 0
        )
        out.append(
            Finding(
                "orphan_observation_run",
                Severity.CRITICAL,
                count,
                "observations reference a run that does not exist",
            )
        )
    return out


# -- temporal ---------------------------------------------------------------


def _temporal_checks(
    db: ReadOnlyDatabase, cross: pd.DataFrame, context: AnalysisContext
) -> list[Finding]:
    out: list[Finding] = []

    if not cross.empty and {"first_seen_at", "last_seen_at"} <= set(cross.columns):
        first = cross["first_seen_at"].map(parse_ts)
        last = cross["last_seen_at"].map(parse_ts)
        both = first.notna() & last.notna()
        bad = cross[both & (last < first)]
        out.append(
            Finding(
                "last_seen_before_first_seen",
                Severity.ERROR,
                len(bad),
                "an advertisement was last seen before it was first seen",
                bad["platform_ad_id"].head().tolist() if "platform_ad_id" in bad else [],
            )
        )

    if not cross.empty and {"last_seen_at", "first_missing_at"} <= set(cross.columns):
        last = cross["last_seen_at"].map(parse_ts)
        missing = cross["first_missing_at"].map(parse_ts)
        both = last.notna() & missing.notna()
        # A listing seen again after its first miss is legitimate (it reappeared),
        # so only flag rows whose status says they never came back.
        status = cross.get("current_status", pd.Series([""] * len(cross)))
        gone = status.astype(str).isin(["likely_removed", "likely_sold"])
        bad = cross[both & gone & (missing < last)]
        out.append(
            Finding(
                "disappearance_before_last_seen",
                Severity.ERROR,
                len(bad),
                "a listing that never returned is recorded as missing before it was last seen",
                bad["platform_ad_id"].head().tolist() if "platform_ad_id" in bad else [],
            )
        )

    if db.has_table("advertisement_events"):
        count = int(
            db.scalar(
                "SELECT COUNT(*) FROM advertisement_events e"
                " JOIN advertisements a ON a.id = e.advertisement_id"
                " WHERE e.event_at < a.created_at"
            )
            or 0
        )
        out.append(
            Finding(
                "event_before_advertisement_created",
                Severity.WARNING,
                count,
                "an event predates the advertisement row it belongs to",
            )
        )

    if db.has_table("advertisement_snapshots"):
        later_run = int(
            db.scalar(
                "SELECT COUNT(*) FROM advertisement_snapshots WHERE run_id > ?",
                [context.run_id],
            )
            or 0
        )
        out.append(
            Finding(
                "snapshot_from_later_run_present_in_database",
                Severity.INFORMATIONAL,
                later_run,
                "snapshots exist from runs after the analysis run; they were excluded "
                "from the cross-section by the no-future-snapshot rule",
            )
        )

    if not cross.empty and "snapshot_scraped_at" in cross.columns:
        reference = context.run_finished_at
        scraped = cross["snapshot_scraped_at"].map(parse_ts)
        if reference is not None:
            slot = context.scheduled_for
            bound = max([t for t in (reference, slot) if t is not None])
            leaked = int((scraped.notna() & (scraped > bound)).sum())
            out.append(
                Finding(
                    "future_snapshot_leaked_into_cross_section",
                    Severity.CRITICAL,
                    leaked,
                    "a snapshot dated after the analysis run reached the cross-section",
                )
            )
    return out


# -- value ranges -----------------------------------------------------------


def _value_range_checks(cross: pd.DataFrame, cfg: EdaConfig) -> list[Finding]:
    out: list[Finding] = []
    if cross.empty:
        return out

    price = pd.to_numeric(cross.get("price_toman"), errors="coerce")
    out.append(
        Finding(
            "negative_price",
            Severity.ERROR,
            int((price < 0).sum()),
            "a negative asking price is a parsing failure",
        )
    )
    out.append(
        Finding(
            "zero_price",
            Severity.WARNING,
            int((price == 0).sum()),
            "price 0 is Bama's encoding for 'negotiable'; it must not be averaged as a price",
        )
    )

    mileage = pd.to_numeric(cross.get("mileage_km"), errors="coerce")
    out.append(
        Finding(
            "negative_mileage",
            Severity.ERROR,
            int((mileage < 0).sum()),
            "negative mileage is a parsing failure",
        )
    )

    year = pd.to_numeric(cross.get("year_jalali"), errors="coerce")
    implausible = year.notna() & (
        (year < PLAUSIBLE_YEAR_RANGE[0]) | (year > PLAUSIBLE_YEAR_RANGE[1])
    )
    out.append(
        Finding(
            "implausible_production_year",
            Severity.ERROR,
            int(implausible.sum()),
            f"production year outside {PLAUSIBLE_YEAR_RANGE}; likely a Gregorian year "
            "stored in a Jalali column",
        )
    )

    # Price below the monitored lower bound: either the filter changed or the
    # listing left it. Either way the population is not what the search defines.
    bound = _search_lower_bound(cfg)
    if bound is not None:
        below = price.notna() & (price > 0) & (price < bound)
        out.append(
            Finding(
                "price_below_search_lower_bound",
                Severity.WARNING,
                int(below.sum()),
                f"asking price below the configured search minimum ({bound:,}); expected "
                "only for listings that left the filter",
            )
        )
    return out


def _search_lower_bound(cfg: EdaConfig) -> int | None:
    try:
        from bama_monitor.config import MonitorConfig
        from bama_monitor.search_filter import parse_search_filter

        return parse_search_filter(MonitorConfig().search_url).price_from
    except Exception:  # noqa: BLE001
        return None


# -- duration bounds --------------------------------------------------------


def _duration_bound_checks(cross: pd.DataFrame, vehicles: pd.DataFrame) -> list[Finding]:
    """min <= estimate <= max, or a report can contradict itself."""
    out: list[Finding] = []
    if vehicles.empty:
        return out
    low = pd.to_numeric(vehicles.get("minimum_disappearance_duration"), errors="coerce")
    est = pd.to_numeric(vehicles.get("estimated_disappearance_duration"), errors="coerce")
    high = pd.to_numeric(vehicles.get("maximum_disappearance_duration"), errors="coerce")

    inverted = (
        (low.notna() & est.notna() & (low > est))
        | (est.notna() & high.notna() & (est > high))
        | (low.notna() & high.notna() & (low > high))
    )
    out.append(
        Finding(
            "inverted_duration_bounds",
            Severity.ERROR,
            int(inverted.sum()),
            "duration bounds are out of order (min <= estimate <= max violated)",
            vehicles.loc[inverted, "vehicle_entity_id"].head().tolist()
            if "vehicle_entity_id" in vehicles
            else [],
        )
    )

    negative = (low.notna() & (low < 0)) | (high.notna() & (high < 0))
    out.append(
        Finding(
            "negative_duration", Severity.ERROR, int(negative.sum()), "a negative time on market"
        )
    )
    return out


# -- semantic consistency ---------------------------------------------------


def _semantic_consistency_checks(cross: pd.DataFrame, panel: pd.DataFrame) -> list[Finding]:
    """Checks that the vocabulary is being used consistently."""
    out: list[Finding] = []
    if not cross.empty and {"current_status", "sale_label"} <= set(cross.columns):
        status = cross["current_status"].astype(str)
        label = cross["sale_label"].astype(str)
        # A sale inference attached to a listing that is demonstrably still live.
        inconsistent = cross[
            status.isin(["active", "new", "reappeared", "active_outside_filter"])
            & label.isin(["likely_sold", "highly_likely_sold", "confirmed_sold"])
        ]
        out.append(
            Finding(
                "sale_label_inconsistent_with_status",
                Severity.ERROR,
                len(inconsistent),
                "a sale label is attached to an advertisement whose observed status says "
                "it is still present",
                inconsistent["platform_ad_id"].head().tolist()
                if "platform_ad_id" in inconsistent
                else [],
            )
        )

        confirmed = int(label.eq("confirmed_sold").sum())
        out.append(
            Finding(
                "confirmed_sale_label_present",
                Severity.CRITICAL,
                confirmed,
                "a 'confirmed_sold' label exists; this system never produces one from "
                "absence, so its presence means an outside source was merged in",
            )
        )

    if not cross.empty and {"current_status", "filter_exit_reason"} <= set(cross.columns):
        # A filter exit counted as a disappearance is the false-removal failure.
        bad = cross[
            cross["filter_exit_reason"].notna()
            & cross["current_status"].astype(str).isin(["likely_removed", "likely_sold"])
        ]
        out.append(
            Finding(
                "filter_exit_counted_as_disappearance",
                Severity.ERROR,
                len(bad),
                "a listing with a filter-exit reason is also marked removed or sold; a "
                "listing that left the search has not left the market",
                bad["platform_ad_id"].head().tolist() if "platform_ad_id" in bad else [],
            )
        )

    if not panel.empty and {"run_is_valid", "was_seen_trusted"} <= set(panel.columns):
        leaked = panel[(~panel["run_is_valid"]) & panel["was_seen_trusted"].notna()]
        out.append(
            Finding(
                "invalid_run_contributed_absence",
                Severity.CRITICAL,
                len(leaked),
                "an observation from a non-valid run carries a trusted seen/absent "
                "signal; absences from unhealthy runs are not observations",
            )
        )

    if not cross.empty and "vehicle_entity_id" in cross.columns:
        grouped = cross.dropna(subset=["vehicle_entity_id"]).groupby("vehicle_entity_id")
        conflicting = 0
        for _, block in grouped:
            if "deep_brand" in block.columns:
                brands = {str(b) for b in block["deep_brand"].dropna().unique()}
                if len(brands) > 1:
                    conflicting += 1
        out.append(
            Finding(
                "vehicle_entity_with_conflicting_attributes",
                Severity.WARNING,
                conflicting,
                "advertisements linked to one vehicle entity disagree on the brand; the "
                "repost match may be a false positive",
            )
        )
    return out


# -- privacy ----------------------------------------------------------------


def _privacy_checks(datasets: dict[str, pd.DataFrame]) -> list[Finding]:
    """The last gate before anything is written to disk."""
    out: list[Finding] = []
    leaked_columns: list[str] = []
    phone_hits = 0

    for name, frame in datasets.items():
        if frame.empty:
            continue
        for column in frame.columns:
            if column.lower() in PRIVATE_COLUMNS:
                leaked_columns.append(f"{name}.{column}")
        text_columns = [c for c in frame.columns if is_texty(frame[c]) and not c.endswith("_url")]
        for column in text_columns:
            series = frame[column].dropna().astype(str)
            if series.empty:
                continue
            phone_hits += int(series.str.contains(_PHONE_PATTERN, regex=True, na=False).sum())

    out.append(
        Finding(
            "private_column_in_output",
            Severity.CRITICAL,
            len(leaked_columns),
            "a private column reached an analytical dataset",
            leaked_columns,
        )
    )
    out.append(
        Finding(
            "contact_number_in_output",
            Severity.CRITICAL,
            phone_hits,
            "a value matching an Iranian mobile number reached an analytical dataset",
        )
    )
    return out


# -- provenance -------------------------------------------------------------


def _provenance_checks(datasets: dict[str, pd.DataFrame]) -> list[Finding]:
    required = {"run_id", "analysis_version", "search_configuration_hash"}
    missing: list[str] = []
    for name, frame in datasets.items():
        if frame.empty:
            continue
        absent = required - set(frame.columns)
        if absent:
            missing.append(f"{name}: {', '.join(sorted(absent))}")
    return [
        Finding(
            "missing_provenance_columns",
            Severity.CRITICAL,
            len(missing),
            "a dataset lacks the provenance columns that make its numbers quotable",
            missing,
        )
    ]


# -- reporting --------------------------------------------------------------


def summarise(findings: list[Finding]) -> dict[str, Any]:
    by_severity: dict[str, int] = {str(s): 0 for s in Severity}
    for finding in findings:
        if finding.count:
            by_severity[str(finding.severity)] += 1
    criticals = [f for f in findings if f.severity is Severity.CRITICAL and f.count]
    return {
        "checks_run": len(findings),
        "checks_with_findings": sum(1 for f in findings if f.count),
        "by_severity": by_severity,
        "critical_findings": [f.as_dict() for f in criticals],
        "passed": not criticals,
        "interpretation": (
            "passed = no critical finding. Errors suppress the specific analysis they "
            "affect; warnings are carried into the report as caveats."
        ),
    }


def write_reports(findings: list[Finding], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_quality_report.json"
    csv_path = out_dir / "data_quality_issues.csv"

    json_path.write_text(
        json.dumps(
            {"summary": summarise(findings), "findings": [f.as_dict() for f in findings]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame([f.as_dict() for f in findings])
    if frame.empty:
        frame = pd.DataFrame(columns=["check", "severity", "count", "detail", "sample"])
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {"data_quality_report": json_path, "data_quality_issues": csv_path}


def enforce(findings: list[Finding]) -> None:
    """Stop the analysis when a critical finding makes it meaningless."""
    criticals = [f for f in findings if f.severity is Severity.CRITICAL and f.count]
    if criticals:
        raise CriticalIntegrityError(criticals)
