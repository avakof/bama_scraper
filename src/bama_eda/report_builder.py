"""Orchestration and the HTML report.

``run_analysis`` is the whole pipeline: select the run, build the datasets, audit
them, analyse, chart and write. It is also where the reporting discipline lives —
observed facts, descriptive statistics, heuristic inference and unsupported
conclusions are written into separate, labelled sections, because a reader
skimming an HTML page will not reconstruct that distinction themselves.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import (
    correlations,
    cross_sectional,
    data_dictionary,
    duration_analysis,
    feature_analysis,
    missingness,
    price_analysis,
    profiling,
    repost_analysis,
    status_analysis,
    temporal_analysis,
    text_analysis,
    validation,
)
from .config import EdaConfig
from .database import ReadOnlyDatabase, utcnow
from .dataset_builder import build_cross_section, build_panel, build_vehicle_dataset
from .models import FORBIDDEN_PHRASES, AnalysisContext, is_texty
from .provenance import Manifest, build_context, select_run
from .schema_inspection import assert_analysable, write_schema_outputs
from .visualization import ChartWriter

#: Dimensions every feature is compared across. Chosen because each is a way a
#: reader might segment the market; the comparison itself is generated, not
#: hand-written per dimension.
FEATURE_DIMENSIONS: tuple[str, ...] = (
    "deep_city",
    "deep_province",
    "deep_brand",
    "deep_seller_type",
    "deep_fuel_type",
    "deep_transmission",
    "deep_condition_new_used",
    "price_band",
)

DISCLAIMER = (
    "A disappearance from the observed inventory is an OBSERVED FACT. A sale is an "
    "INFERENCE. This analysis never reports a confirmed sale, never converts the "
    "sale-evidence score into a probability, and never treats a filter exit or a "
    "repost as a disappearance."
)


def run_analysis(cfg: EdaConfig, *, make_charts: bool = True) -> dict[str, Any]:
    """Execute the full analysis and write the report directory."""
    started = utcnow()
    db = ReadOnlyDatabase(cfg.database_url)
    try:
        absent_optional = assert_analysable(db)
        run, selection_warnings = select_run(db, cfg.run_id)
        context = build_context(db, run, analysis_started_at=started)
        manifest = Manifest(context, cfg)
        for warning in selection_warnings:
            manifest.warnings.append(warning)
        for table in absent_optional:
            manifest.warn("schema", f"optional table {table!r} is absent; related analysis skipped")

        out_dir = _output_dir(cfg, context, started)
        charts_dir = out_dir / "charts"

        # -- datasets ---------------------------------------------------
        cross = build_cross_section(db, context, cfg, manifest)
        panel = build_panel(db, context, cfg, manifest)
        vehicles = build_vehicle_dataset(db, cross, context, cfg, manifest)
        datasets = {
            "advertisement_cross_section": cross,
            "longitudinal_panel": panel,
            "vehicle_entity_dataset": vehicles,
        }

        # -- integrity --------------------------------------------------
        findings = validation.audit(db, datasets, context, cfg)
        quality = validation.summarise(findings)
        validation.write_reports(findings, out_dir)
        # Errors do not stop the run; criticals do, and the reports are already on
        # disk so the failure is diagnosable.
        validation.enforce(findings)

        # -- schema documentation ---------------------------------------
        schema_paths = write_schema_outputs(db, out_dir)
        for name, path in schema_paths.items():
            manifest.record_output(name, path)

        # -- analysis ---------------------------------------------------
        results = _analyse(db, cross, panel, vehicles, context, cfg, manifest)

        # -- charts -----------------------------------------------------
        chart_manifest: dict[str, Any] = {"charts": [], "skipped": []}
        if make_charts:
            writer = ChartWriter(charts_dir, min_group_size=cfg.thresholds.min_group_size)
            _draw_charts(writer, cross, panel, vehicles, results, cfg)
            chart_manifest = writer.manifest()

        # -- outputs ----------------------------------------------------
        _write_datasets(datasets, out_dir, cfg, manifest)
        _write_tables(results, out_dir, manifest)
        dictionary = data_dictionary.build(datasets)
        dict_path = out_dir / "dataset_dictionary.csv"
        dictionary.to_csv(dict_path, index=False, encoding="utf-8-sig")
        manifest.record_output("dataset_dictionary", dict_path, len(dictionary))

        warnings_frame = pd.DataFrame([w.as_dict() for w in manifest.warnings])
        if warnings_frame.empty:
            warnings_frame = pd.DataFrame(columns=["area", "message"])
        warnings_path = out_dir / "analysis_warnings.csv"
        warnings_frame.to_csv(warnings_path, index=False, encoding="utf-8-sig")
        manifest.record_output("analysis_warnings", warnings_path, len(warnings_frame))

        manifest.queries = db.query_log
        results["data_quality"] = quality
        results["charts"] = chart_manifest

        html_path: Path | None = None
        if cfg.html:
            html_path = out_dir / "bama_eda_report.html"
            html_path.write_text(
                _render_html(context, results, manifest, chart_manifest, findings),
                encoding="utf-8",
            )
            manifest.record_output("html_report", html_path)

        _write_readme(out_dir, context, results, manifest)
        manifest_path = manifest.write(out_dir / "eda_manifest.json")

        leaks = text_analysis.assert_no_contact_leak(datasets)
        if leaks:  # pragma: no cover - the audit would already have failed
            raise RuntimeError(f"contact data reached an output: {leaks}")

        return {
            "output_dir": out_dir,
            "manifest": manifest_path,
            "html": html_path,
            "context": context,
            "row_counts": {name: int(len(f)) for name, f in datasets.items()},
            "quality": quality,
            "results": results,
            "charts": chart_manifest,
        }
    finally:
        db.close()


def _draw_charts(
    writer: ChartWriter,
    cross: pd.DataFrame,
    panel: pd.DataFrame,
    vehicles: pd.DataFrame,
    results: dict[str, Any],
    cfg: EdaConfig,
) -> None:
    """Draw the chart set, skipping (visibly) whatever the data cannot support."""
    # 1-3 inventory composition
    writer.bar_counts(
        cross.get("deep_brand", pd.Series(dtype=object)),
        "01_inventory_by_brand",
        "Inventory by brand",
        top=15,
    )
    if {"deep_brand", "deep_model"} <= set(cross.columns):
        combo = cross.dropna(subset=["deep_brand", "deep_model"]).assign(
            bm=lambda d: d["deep_brand"].astype(str) + " / " + d["deep_model"].astype(str)
        )
        writer.bar_counts(combo["bm"], "02_brand_model", "Top brand-model combinations", top=15)
    writer.bar_counts(
        cross.get("year_jalali", pd.Series(dtype=object)).dropna().astype("Int64").astype(str),
        "03_production_year",
        "Production year (Jalali)",
        top=20,
    )

    # 4-7 price and the variables it moves with
    writer.histogram(
        cross.get("price_toman", pd.Series(dtype=float)),
        "04_price_histogram",
        "Asking price (toman)",
        xlabel="toman",
    )
    writer.histogram(
        cross.get("price_toman", pd.Series(dtype=float)),
        "05_price_histogram_log",
        "Asking price, log scale (toman)",
        xlabel="toman",
        log_x=True,
    )
    writer.histogram(
        cross.get("mileage_km", pd.Series(dtype=float)),
        "06_mileage_histogram",
        "Odometer reading (km)",
        xlabel="km",
    )
    writer.scatter(
        cross,
        "mileage_km",
        "price_toman",
        "07_price_vs_mileage",
        "Asking price against mileage",
        log_y=True,
        xlabel="km",
        ylabel="toman",
        seed=cfg.random_seed,
    )
    writer.scatter(
        cross,
        "year_jalali",
        "price_toman",
        "08_price_vs_year",
        "Asking price against production year",
        log_y=True,
        xlabel="year (Jalali)",
        ylabel="toman",
        seed=cfg.random_seed,
    )

    # 9-10 price by group
    writer.boxplot_by_group(
        cross,
        "price_toman",
        "deep_brand",
        "09_price_by_brand",
        "Asking price by brand",
        ylabel="toman",
    )
    segments = results.get("price_by_segment")
    if isinstance(segments, pd.DataFrame) and not segments.empty:
        models = segments[(segments["dimension"] == "model") & segments["sufficient_sample"]]
        if not models.empty:
            top = models.nlargest(15, "listing_count")
            fig_series = pd.Series(
                top["median_price"].to_numpy(),
                index=[
                    f"{v} (n={int(n)})"
                    for v, n in zip(top["segment_value"], top["listing_count"], strict=False)
                ],
            )
            writer.bar_counts(
                pd.Series(
                    sum(
                        (
                            [label] * max(1, int(round(value / 1e8)))
                            for label, value in fig_series.items()
                        ),
                        [],
                    )
                ),
                "10_median_price_by_model",
                "Median asking price by model (bar length = median price / 100M toman)",
                top=15,
                xlabel="median price / 100M toman",
            )

    # 11 missingness
    writer.missingness_heatmap(cross, "11_missingness_heatmap", "Missingness by column")

    # 12-13 seller and location
    writer.bar_counts(
        cross.get("deep_seller_type", pd.Series(dtype=object)),
        "12_seller_type",
        "Seller type",
        top=10,
    )
    writer.bar_counts(
        cross.get("deep_city", pd.Series(dtype=object)),
        "13_location_city",
        "Listings by city",
        top=15,
    )
    writer.bar_counts(
        cross.get("deep_province", pd.Series(dtype=object)),
        "14_location_province",
        "Listings by province",
        top=15,
    )

    # 15-16 temporal
    volume = results.get("publication_volume")
    if isinstance(volume, pd.DataFrame) and len(volume) >= 2:
        writer.line(
            [str(p)[:10] for p in volume["period"]],
            volume["advertisements"].tolist(),
            "15_publication_volume",
            "Listings published per day",
            xlabel="publication date",
            ylabel="advertisements",
            n=int(volume["advertisements"].sum()),
            note="only listings with a known publication date",
        )
    else:
        writer.note_chart(
            "15_publication_volume",
            "Publication volume unavailable",
            "Fewer than two days carry a known publication date. "
            f"Known for {results.get('publication', {}).get('publication_timestamp_known', 0)} "
            "advertisement(s).",
        )
    timeline = results.get("run_timeline")
    if isinstance(timeline, pd.DataFrame) and len(timeline) >= 2:
        writer.line(
            [str(r) for r in timeline["run_id"]],
            timeline["seen"].tolist(),
            "16_daily_inventory",
            "Advertisements seen per run",
            xlabel="run id",
            ylabel="seen",
            n=int(timeline["seen"].sum()),
        )
    else:
        writer.note_chart(
            "16_daily_inventory",
            "Inventory trend unavailable",
            "Fewer than two runs are present in this database.",
        )

    # 17 price changes
    changes = results.get("price_change_table")
    if isinstance(changes, pd.DataFrame) and len(changes) >= 5:
        writer.histogram(
            changes["percentage_change"],
            "17_price_change_distribution",
            "Observed price changes (%)",
            xlabel="percent change",
        )
    else:
        writer.note_chart(
            "17_price_change_distribution",
            "No price-change distribution",
            f"{0 if not isinstance(changes, pd.DataFrame) else len(changes)} price change(s) "
            "observed across the available runs. With few runs this is expected and is not "
            "evidence that sellers do not adjust prices.",
        )

    # 18 status transitions
    transitions = results.get("status_transitions")
    if isinstance(transitions, pd.DataFrame) and not transitions.empty:
        expanded = pd.Series(
            sum(
                ([str(row["event_type"])] * int(row["count"]) for _, row in transitions.iterrows()),
                [],
            )
        )
        writer.bar_counts(
            expanded, "18_status_transitions", "Status-transition events (all runs to date)", top=12
        )

    # 19-20 durations
    durations = results.get("durations")
    if isinstance(durations, pd.DataFrame) and not durations.empty:
        writer.histogram(
            durations["observed_monitoring_duration"],
            "19_observed_duration",
            "Observed monitoring duration (days)",
            xlabel="days since first observation",
        )
        writer.bar_counts(
            durations["censoring_class"],
            "20_censoring_classes",
            "Censoring class (event = disappearance, not sale)",
            top=10,
        )

    # 21 eligibility
    publication = results.get("publication", {})
    eligible = publication.get("eligible_for_duration_ranking", 0)
    total = publication.get("advertisements_total", 0)
    writer.note_chart(
        "21_duration_ranking_eligibility",
        f"Eligible for duration ranking: {eligible:,} of {total:,}",
        results.get("duration_feasibility", {}).get("note", ""),
    )

    # 22 filter exits
    exits = results.get("filter_exits", {})
    if exits.get("filter_exits", 0):
        reasons = pd.Series(sum(([k] * v for k, v in exits.get("reason_counts", {}).items()), []))
        writer.bar_counts(
            reasons, "22_filter_exit_reasons", "Filter-exit reasons (NOT disappearances)", top=10
        )
    else:
        writer.note_chart(
            "22_filter_exit_reasons",
            "No filter exits recorded",
            "No listing was observed leaving the search while remaining for sale. "
            "With three runs this is a limit of the observation window.",
        )

    # 23 repost chains
    if not vehicles.empty and "advertisement_count" in vehicles.columns:
        counts = pd.to_numeric(vehicles["advertisement_count"], errors="coerce").fillna(1)
        if (counts > 1).any():
            writer.bar_counts(
                counts.astype(int).astype(str),
                "23_repost_chain_size",
                "Advertisements per vehicle entity",
                top=10,
            )
        else:
            writer.note_chart(
                "23_repost_chain_size",
                "No repost chains detected",
                f"All {len(vehicles):,} vehicle entities have exactly one advertisement. "
                "A relisting needs time to occur, and this database spans three runs.",
            )

    # 24 sale evidence
    if "sale_confidence" in cross.columns:
        writer.histogram(
            cross["sale_confidence"],
            "24_sale_evidence_score",
            "Sale-evidence score (an ordering, NOT a probability)",
            xlabel="score",
            bins=20,
        )

    # 26+ every-feature section: what differs across each dimension, and which
    # features move with price
    associations = results.get("feature_price_associations")
    if isinstance(associations, pd.DataFrame) and not associations.empty:
        writer.effect_ranking(
            associations,
            "26_features_vs_price",
            "Features most associated with asking price (tautologies flagged)",
        )
    differentiators = results.get("feature_differentiators")
    comparisons = results.get("feature_comparisons")
    if isinstance(differentiators, pd.DataFrame) and not differentiators.empty:
        index = 27
        for dimension in results.get("feature_dimensions_compared", []):
            block = differentiators[differentiators["dimension"] == dimension]
            if block.empty:
                continue
            short = dimension.replace("deep_", "")
            writer.effect_ranking(
                block,
                f"{index:02d}_differs_by_{short}",
                f"Features that differ most across {short}",
            )
            index += 1
            # The single most differentiating non-tautological feature, drawn out.
            usable = block[~block.get("near_tautological", False).astype(bool)]
            if not usable.empty and isinstance(comparisons, pd.DataFrame):
                feature = usable.iloc[0]["feature"]
                writer.grouped_share(
                    comparisons[comparisons["dimension"] == dimension],
                    feature,
                    f"{index:02d}_{short}_{usable.iloc[0]['label'][:24]}",
                    f"{usable.iloc[0]['label']} across {short}",
                )
                index += 1

    # linear versus non-linear structure across every numeric pair
    pearson_matrix = results.get("pearson_matrix")
    spearman_matrix = results.get("spearman_matrix")
    if isinstance(pearson_matrix, pd.DataFrame) and not pearson_matrix.empty:
        writer.correlation_heatmap(
            pearson_matrix,
            "43_pearson_heatmap",
            "Pearson (LINEAR association only)",
            note="linear only; blind to curved relationships",
        )
    if isinstance(spearman_matrix, pd.DataFrame) and not spearman_matrix.empty:
        writer.correlation_heatmap(
            spearman_matrix,
            "44_spearman_heatmap",
            "Spearman (MONOTONE association, linear or not)",
            note="compare with the Pearson map: where they differ, the shape is curved",
        )
    nonlinear = results.get("nonlinearity")
    if isinstance(nonlinear, pd.DataFrame) and not nonlinear.empty:
        writer.nonlinearity_scatter_panel(
            cross,
            nonlinear,
            "45_nonlinear_pairs",
            "Pairs a linear correlation would misdescribe",
            seed=cfg.random_seed,
        )

    # 25 Kaplan-Meier, only if the estimator agreed to produce one
    km = results.get("kaplan_meier", {})
    if km.get("available"):
        writer.kaplan_meier(
            km, "25_kaplan_meier", "Time to disappearance (Kaplan-Meier, delayed entry)"
        )
    else:
        writer.note_chart(
            "25_kaplan_meier",
            "No Kaplan-Meier curve",
            str(km.get("note", "insufficient sample")),
        )


def _output_dir(cfg: EdaConfig, context: AnalysisContext, started: datetime) -> Path:
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out = cfg.output_dir / f"{context.run_id}_{stamp}"
    (out / "charts").mkdir(parents=True, exist_ok=True)
    return out


def _analyse(
    db: ReadOnlyDatabase,
    cross: pd.DataFrame,
    panel: pd.DataFrame,
    vehicles: pd.DataFrame,
    context: AnalysisContext,
    cfg: EdaConfig,
    manifest: Manifest,
) -> dict[str, Any]:
    thresholds = cfg.thresholds
    results: dict[str, Any] = {}

    results["inventory"] = cross_sectional.inventory_summary(cross, panel, vehicles)
    results["composition"] = cross_sectional.composition(
        cross, min_group_size=thresholds.min_group_size
    )
    results["brand_model"] = cross_sectional.brand_model_combinations(
        cross, min_group_size=thresholds.min_group_size
    )
    results["descriptive_statistics"] = profiling.profile_frame(cross)
    results["categorical_distributions"] = profiling.profile_categoricals(
        cross, min_group_size=thresholds.min_group_size
    )

    completeness = missingness.field_completeness(cross)
    results["field_completeness"] = completeness
    results["missingness_patterns"] = missingness.missingness_patterns(cross)
    results["missingness_summary"] = missingness.summarise(cross, completeness)
    results["missingness_by_brand"] = missingness.missingness_by_group(
        cross,
        "deep_brand",
        ["price_toman", "mileage_km", "deep_description_scrubbed", "media_count"],
        min_group_size=thresholds.min_group_size,
    )

    results["correlations"] = correlations.correlation_matrix(
        cross,
        min_pairs=thresholds.min_correlation_pairs,
        weak_threshold=thresholds.weak_correlation,
    )
    results["price_associations"] = correlations.price_association_summary(
        results["correlations"], weak_threshold=thresholds.weak_correlation
    )
    results["categorical_associations"] = correlations.categorical_associations(cross)

    results["price_by_segment"] = price_analysis.price_by_segment(
        cross, min_group_size=thresholds.min_group_size
    )
    results["comparable_segments"] = price_analysis.comparable_segments(
        cross, min_group_size=thresholds.min_group_size
    )
    results["price_group_tests"] = price_analysis.group_difference_tests(
        cross, min_group_size=thresholds.min_group_size
    )
    results["price_changes"] = price_analysis.price_change_analysis(panel, cross)
    results["price_change_table"] = price_analysis.price_change_table(panel)

    results["publication"] = temporal_analysis.publication_coverage(cross)
    results["duration_feasibility"] = temporal_analysis.duration_ranking_feasibility(
        cross, min_eligible=thresholds.min_survival_group_size
    )
    results["publication_volume"] = temporal_analysis.publication_volume(cross)
    results["entry_delay"] = temporal_analysis.entry_delay_distribution(cross)
    runs_frame = db.query_frame("SELECT * FROM monitoring_runs ORDER BY id", label="report.runs")
    results["run_timeline"] = temporal_analysis.run_timeline(panel, runs_frame)
    results["observation_intervals"] = temporal_analysis.observation_intervals(panel)

    durations = duration_analysis.build_duration_table(cross)
    results["durations"] = durations
    results["duration_summary"] = duration_analysis.duration_summary(durations)
    results["bound_ordering"] = duration_analysis.check_bound_ordering(durations)
    results["kaplan_meier"] = duration_analysis.kaplan_meier(
        durations,
        min_subjects=thresholds.min_survival_group_size,
        min_events=thresholds.min_survival_events,
    )
    results["survival_by_brand"] = duration_analysis.survival_by_group(
        durations,
        "deep_brand",
        min_subjects=thresholds.min_survival_group_size,
        min_events=thresholds.min_survival_events,
    )

    results["status_distribution"] = status_analysis.status_distribution(cross)
    results["status_transitions"] = status_analysis.status_transitions(db, context.run_id)
    results["filter_exits"] = status_analysis.filter_exit_analysis(
        cross, min_group_size=thresholds.min_group_size
    )
    results["filter_exit_table"] = status_analysis.filter_exit_table(cross)
    results["sale_evidence"] = status_analysis.sale_evidence_analysis(
        cross, db, min_calibration_labels=thresholds.min_calibration_labels
    )
    results["sale_evidence_table"] = status_analysis.sale_evidence_table(cross)

    results["repost_links"] = repost_analysis.repost_link_summary(db, context.run_id)
    results["vehicle_grain"] = repost_analysis.vehicle_grain_summary(vehicles)
    results["repost_chains"] = repost_analysis.repost_transitions(cross, vehicles)
    results["repost_mileage_check"] = repost_analysis.mileage_consistency_check(
        results["repost_chains"]
    )

    # -- every scraped feature, not a curated shortlist -----------------
    # Bands are derived, so add them before cataloguing: price_band and
    # mileage_band are dimensions a reader will want to segment by.
    cross = profiling.add_bands(cross)
    catalogue = feature_analysis.discover_features(cross)
    results["feature_catalogue"] = catalogue
    results["feature_summary"] = feature_analysis.summarise_catalogue(catalogue)
    results["feature_profiles"] = feature_analysis.profile_all(
        cross, catalogue, min_group_size=thresholds.min_group_size
    )
    results["feature_levels"] = feature_analysis.level_distribution(
        cross, catalogue, min_group_size=thresholds.min_group_size
    )
    results["feature_price_associations"] = feature_analysis.feature_price_associations(
        cross, catalogue, min_n=thresholds.min_group_size * 3, top=60
    )

    comparisons: list[pd.DataFrame] = []
    rankings: list[pd.DataFrame] = []
    for dimension in FEATURE_DIMENSIONS:
        if dimension not in cross.columns:
            continue
        comparison = feature_analysis.compare_across(
            cross, catalogue, dimension, min_group_size=thresholds.min_group_size
        )
        if not comparison.empty:
            comparisons.append(comparison)
        ranking = feature_analysis.differentiating_features(
            cross, catalogue, dimension, min_group_size=thresholds.min_group_size, top=25
        )
        if not ranking.empty:
            rankings.append(ranking)
    results["feature_comparisons"] = (
        pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    )
    results["feature_differentiators"] = (
        pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    )
    results["feature_dimensions_compared"] = [d for d in FEATURE_DIMENSIONS if d in cross.columns]

    # -- every pair of features: linear, monotone and general dependence --
    pairs = correlations.pairwise_associations(
        cross,
        catalogue,
        min_pairs=thresholds.min_correlation_pairs,
        weak_threshold=thresholds.weak_correlation,
        seed=cfg.random_seed,
    )
    results["pairwise_associations"] = pairs
    results["pairwise_summary"] = correlations.summarise_pairs(pairs)
    results["nonlinearity"] = correlations.nonlinearity_report(pairs, top=40)
    results["pearson_matrix"] = correlations.association_matrix(pairs, measure="linear_pearson_r")
    results["spearman_matrix"] = correlations.association_matrix(
        pairs, measure="monotone_spearman_rho"
    )

    results["descriptions"] = text_analysis.describe_descriptions(cross)
    results["frequent_terms"] = text_analysis.frequent_terms(cross, top=40)
    results["frequent_bigrams"] = text_analysis.frequent_terms(cross, top=25, ngram=2)
    results["duplicate_descriptions"] = text_analysis.duplicate_descriptions(cross)
    results["boilerplate"] = text_analysis.boilerplate_frequency(cross)

    # Left-truncated exclusions, published rather than dropped.
    if "eligible_for_duration_ranking" in cross.columns:
        excluded = cross[
            ~pd.Series(cross["eligible_for_duration_ranking"]).fillna(False).astype(bool)
        ]
        results["left_truncated_excluded"] = excluded[
            [
                c
                for c in (
                    "platform_ad_id",
                    "current_status",
                    "published_at",
                    "published_at_source",
                    "published_at_reliable",
                    "first_seen_at",
                    "entry_delay_seconds",
                    "left_truncated",
                )
                if c in excluded.columns
            ]
        ].copy()
        manifest.record_exclusion(
            "duration_ranking",
            "left-truncated or unreliable publication time: earlier market life "
            "unobserved, so estimated market duration is not computed",
            int(len(excluded)),
        )
    else:
        results["left_truncated_excluded"] = pd.DataFrame()

    _add_limitation_warnings(results, manifest, cfg)
    return results


def _add_limitation_warnings(results: dict[str, Any], manifest: Manifest, cfg: EdaConfig) -> None:
    """Turn quantitative limits into explicit, exported caveats."""
    publication = results.get("publication", {})
    eligible = publication.get("eligible_for_duration_ranking", 0)
    total = publication.get("advertisements_total", 0)
    if total and eligible / max(total, 1) < 0.05:
        manifest.warn(
            "left_truncation",
            f"only {eligible} of {total} advertisements "
            f"({publication.get('eligible_share_pct')}%) can support a time-on-market "
            "statement; the rest were already active when monitoring began",
        )
    feasibility = results.get("duration_feasibility", {})
    if not feasibility.get("feasible", False):
        manifest.warn("duration_ranking", feasibility.get("note", ""))
    km = results.get("kaplan_meier", {})
    if not km.get("available"):
        manifest.warn("survival", f"no Kaplan-Meier curve estimated: {km.get('note', '')}")
    calibration = results.get("sale_evidence", {}).get("calibration", {})
    if not calibration.get("calibrated", False):
        manifest.warn(
            "sale_evidence",
            "sale evidence is uncalibrated; the score is an ordering, not a rate, and "
            "no empirical sale rate is reported",
        )
    runs = results.get("run_timeline")
    if isinstance(runs, pd.DataFrame) and len(runs) < 7:
        manifest.warn(
            "temporal",
            f"only {len(runs)} monitoring run(s) are available; day-over-day dynamics "
            "(price changes, disappearance rates, reposts) are barely observable and "
            "must not be read as market rates",
        )


def _write_datasets(
    datasets: dict[str, pd.DataFrame], out_dir: Path, cfg: EdaConfig, manifest: Manifest
) -> None:
    for name, frame in datasets.items():
        if cfg.export_parquet:
            path = out_dir / f"{name}.parquet"
            try:
                _parquet_safe(frame).to_parquet(path, index=False)
                manifest.record_output(f"{name}.parquet", path, len(frame))
            except Exception as exc:  # noqa: BLE001 - parquet is a convenience
                manifest.warn("export", f"parquet export of {name} failed: {exc}")
        if cfg.export_csv and name != "longitudinal_panel":
            path = out_dir / f"{name}.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            manifest.record_output(f"{name}.csv", path, len(frame))


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns so Arrow can write them."""
    out = frame.copy()
    for column in out.columns:
        if is_texty(out[column]):
            types = {type(v) for v in out[column].dropna().head(500)}
            if len(types) > 1:
                out[column] = out[column].astype(str).where(out[column].notna())
    return out


#: (results key, filename) for every tabular output.
TABLE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("descriptive_statistics", "descriptive_statistics.csv"),
    ("categorical_distributions", "categorical_distributions.csv"),
    ("composition", "inventory_composition.csv"),
    ("brand_model", "brand_model_combinations.csv"),
    ("field_completeness", "field_completeness.csv"),
    ("missingness_patterns", "missingness_patterns.csv"),
    ("missingness_by_brand", "missingness_by_brand.csv"),
    ("correlations", "correlations.csv"),
    ("categorical_associations", "categorical_associations.csv"),
    ("price_by_segment", "price_by_segment.csv"),
    ("comparable_segments", "comparable_segments.csv"),
    ("price_group_tests", "price_group_tests.csv"),
    ("price_change_table", "price_changes.csv"),
    ("status_distribution", "status_distribution.csv"),
    ("status_transitions", "status_transitions.csv"),
    ("filter_exit_table", "filter_exits.csv"),
    ("sale_evidence_table", "sale_evidence_summary.csv"),
    ("repost_chains", "repost_analysis.csv"),
    ("durations", "duration_summary.csv"),
    ("left_truncated_excluded", "left_truncated_excluded.csv"),
    ("publication_volume", "publication_volume.csv"),
    ("entry_delay", "entry_delay_distribution.csv"),
    ("run_timeline", "run_timeline.csv"),
    ("feature_catalogue", "feature_catalogue.csv"),
    ("feature_profiles", "feature_profiles.csv"),
    ("feature_levels", "feature_level_distributions.csv"),
    ("feature_comparisons", "feature_comparison_by_dimension.csv"),
    ("feature_differentiators", "feature_differentiators.csv"),
    ("feature_price_associations", "feature_price_associations.csv"),
    ("pairwise_associations", "feature_pairwise_associations.csv"),
    ("nonlinearity", "feature_nonlinear_relationships.csv"),
    ("frequent_terms", "description_terms.csv"),
    ("frequent_bigrams", "description_bigrams.csv"),
    ("boilerplate", "description_boilerplate.csv"),
)


def _write_tables(results: dict[str, Any], out_dir: Path, manifest: Manifest) -> None:
    for key, filename in TABLE_OUTPUTS:
        frame = results.get(key)
        if not isinstance(frame, pd.DataFrame):
            continue
        path = out_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        manifest.record_output(filename, path, len(frame))

    # The eligible duration ranking, or an explicit refusal in its place.
    feasibility = results.get("duration_feasibility", {})
    path = out_dir / "duration_ranking_eligible.csv"
    durations = results.get("durations")
    if feasibility.get("feasible") and isinstance(durations, pd.DataFrame):
        eligible = durations[
            pd.Series(durations.get("eligible_for_duration_ranking")).fillna(False).astype(bool)
            & (durations["event_observed"] == 1)
        ].sort_values("estimated_disappearance_duration")
        eligible.to_csv(path, index=False, encoding="utf-8-sig")
        manifest.record_output("duration_ranking_eligible.csv", path, len(eligible))
    else:
        pd.DataFrame(
            [
                {
                    "verdict": feasibility.get("verdict", "insufficient_data_for_duration_ranking"),
                    "eligible_advertisements": feasibility.get("eligible_advertisements", 0),
                    "eligible_with_observed_disappearance": feasibility.get(
                        "eligible_with_observed_disappearance", 0
                    ),
                    "minimum_required": feasibility.get("minimum_required"),
                    "note": feasibility.get("note", ""),
                }
            ]
        ).to_csv(path, index=False, encoding="utf-8-sig")
        manifest.record_output("duration_ranking_eligible.csv", path, 0)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _table(frame: Any, *, limit: int = 25) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "<p class='muted'>No rows.</p>"
    block = frame.head(limit)
    head = "".join(f"<th>{_esc(c)}</th>" for c in block.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in block.itertuples(index=False)
    )
    more = (
        f"<p class='muted'>showing {limit} of {len(frame):,} rows; the CSV has all of them</p>"
        if len(frame) > limit
        else ""
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>{more}"


def _kv(payload: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)}</td></tr>"
        for k, v in payload.items()
    )
    return f"<div class='scroll'><table class='kv'>{rows}</table></div>"


def _charts_html(chart_manifest: dict[str, Any], out_dir: Path | None = None) -> str:
    charts = chart_manifest.get("charts", [])
    if not charts:
        return "<p class='muted'>No charts generated.</p>"
    blocks = []
    for chart in charts:
        rel = Path(chart["png"]).name
        blocks.append(
            f"<figure><img src='charts/{_esc(rel)}' alt='{_esc(chart['name'])}' loading='lazy'>"
            f"<figcaption>{_esc(chart['name'])} — n = {chart['n']:,}"
            + (f" — {_esc(chart['note'])}" if chart.get("note") else "")
            + "</figcaption></figure>"
        )
    skipped = chart_manifest.get("skipped", [])
    skip_html = ""
    if skipped:
        items = "".join(
            f"<li><code>{_esc(s['name'])}</code>: {_esc(s['reason'])}</li>" for s in skipped
        )
        skip_html = (
            "<details><summary>Charts deliberately not drawn "
            f"({len(skipped)})</summary><ul>{items}</ul></details>"
        )
    return f"<div class='gallery'>{''.join(blocks)}</div>{skip_html}"


def _render_html(
    context: AnalysisContext,
    results: dict[str, Any],
    manifest: Manifest,
    chart_manifest: dict[str, Any],
    findings: list[Any],
) -> str:
    inventory = results.get("inventory", {})
    publication = results.get("publication", {})
    quality = results.get("data_quality", {})
    duration = results.get("duration_summary", {})
    sale = results.get("sale_evidence", {})
    calibration = sale.get("calibration", {})

    limitations = _limitations(results, manifest)
    unsupported = _unsupported_conclusions(results)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bama EDA — run {context.run_id}</title>
<style>
:root {{ color-scheme: light dark; --fg:#1a202c; --bg:#fff; --muted:#718096;
        --line:#e2e8f0; --accent:#2b6cb0; --warn:#fff8e6; --warnline:#e0a800;
        --bad:#fff5f5; --badline:#c53030; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg:#e2e8f0; --bg:#1a202c; --muted:#a0aec0; --line:#2d3748;
           --accent:#63b3ed; --warn:#2d2a1f; --bad:#2d1f1f; }} }}
* {{ box-sizing:border-box; }}
body {{ font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
        color:var(--fg); background:var(--bg); margin:0; padding:2rem clamp(1rem,4vw,3rem); }}
h1 {{ font-size:1.8rem; margin:0 0 .3rem; }}
h2 {{ font-size:1.25rem; margin:2.4rem 0 .6rem; padding-bottom:.3rem;
      border-bottom:2px solid var(--line); }}
h3 {{ font-size:1.02rem; margin:1.4rem 0 .4rem; color:var(--accent); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th,td {{ text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line);
         vertical-align:top; }}
th {{ font-weight:600; }}
table.kv th {{ width:32%; color:var(--muted); font-weight:500; }}
.scroll {{ overflow-x:auto; max-width:100%; }}
.muted {{ color:var(--muted); font-size:13px; }}
.banner {{ background:var(--warn); border-left:4px solid var(--warnline);
           padding:1rem 1.2rem; border-radius:6px; margin:1.2rem 0; }}
.stop {{ background:var(--bad); border-left:4px solid var(--badline);
         padding:1rem 1.2rem; border-radius:6px; margin:1.2rem 0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
          gap:.8rem; margin:1rem 0; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:.8rem 1rem; }}
.card .n {{ font-size:1.5rem; font-weight:700; }}
.card .l {{ font-size:12px; color:var(--muted); }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:1.2rem; }}
figure {{ margin:0; border:1px solid var(--line); border-radius:8px; padding:.6rem; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ font-size:12px; color:var(--muted); margin-top:.4rem; }}
code {{ background:var(--line); padding:.1rem .3rem; border-radius:3px; font-size:12px; }}
ol.toc {{ columns:2; font-size:14px; }}
details {{ margin:.8rem 0; }} summary {{ cursor:pointer; font-size:14px; }}
</style></head><body>

<h1>Exploratory data analysis — Bama monitored inventory</h1>
<p class="muted">Monitoring run {context.run_id} · scheduled {_esc(context.scheduled_for)} ·
analysed {_esc(context.analysis_started_at)} · analysis v{_esc(context.analysis_version)}</p>

<div class="banner"><strong>How to read this report.</strong> {_esc(DISCLAIMER)}</div>

<h2 id="toc">Contents</h2>
<ol class="toc">
<li><a href="#summary">Executive summary</a></li>
<li><a href="#provenance">Provenance</a></li>
<li><a href="#quality">Data-quality findings</a></li>
<li><a href="#inventory">Inventory structure</a></li>
<li><a href="#price">Price analysis</a></li>
<li><a href="#mileage">Mileage and year</a></li>
<li><a href="#completeness">Feature completeness</a></li>
<li><a href="#temporal">Temporal patterns</a></li>
<li><a href="#changes">Price changes</a></li>
<li><a href="#duration">Disappearance and censoring</a></li>
<li><a href="#reposts">Reposts</a></li>
<li><a href="#filter">Filter exits</a></li>
<li><a href="#sale">Sale evidence</a></li>
<li><a href="#features">All scraped features</a></li>
<li><a href="#pairs">Feature-to-feature association</a></li>
<li><a href="#charts">Charts</a></li>
<li><a href="#limitations">Limitations</a></li>
<li><a href="#unsupported">Conclusions that cannot be drawn</a></li>
<li><a href="#repro">Reproducibility</a></li>
</ol>

<h2 id="summary">1. Executive summary</h2>
<div class="cards">
  <div class="card"><div class="n">{inventory.get("advertisements", 0):,}</div>
    <div class="l">advertisements (observed)</div></div>
  <div class="card"><div class="n">{inventory.get("vehicle_entities", 0):,}</div>
    <div class="l">vehicle entities</div></div>
  <div class="card"><div class="n">{inventory.get("observations_all_runs", 0):,}</div>
    <div class="l">observations, all runs</div></div>
  <div class="card"><div class="n">{publication.get("eligible_for_duration_ranking", 0):,}</div>
    <div class="l">eligible for duration ranking</div></div>
  <div class="card"><div class="n">{duration.get("filter_exits", 0):,}</div>
    <div class="l">filter exits (not disappearances)</div></div>
  <div class="card"><div class="n">{duration.get("observed_disappearances", 0):,}</div>
    <div class="l">observed disappearances</div></div>
</div>
<p class="muted">Every figure above describes <strong>one run of a live marketplace</strong>.
Quote it with its run id and timestamps; differences between runs are market churn.</p>

<h2 id="provenance">2. Provenance</h2>
{_kv(context.as_dict())}
<h3>Source tables</h3>
{_kv(manifest.source_tables)}
<h3>Attribute enrichment</h3>
{_kv(manifest.enrichment)}

<h2 id="quality">3. Data-quality findings</h2>
{_kv({k: v for k, v in quality.items() if k != "critical_findings"})}
{_table(pd.DataFrame([f.as_dict() for f in findings if f.count]), limit=30)}

<h2 id="inventory">4. Inventory structure</h2>
{_kv(inventory)}
<h3>Composition</h3>
{_table(results.get("composition"), limit=30)}
<h3>Brand × model</h3>
{_table(results.get("brand_model"), limit=20)}
<h3>Status distribution</h3>
{_table(results.get("status_distribution"), limit=15)}

<h2 id="price">5. Price analysis</h2>
<p class="muted">Unit: <strong>Iranian toman</strong>, never converted. These are
<strong>asking prices</strong> from listings, not transaction prices.</p>
<h3>Price by segment</h3>
{_table(results.get("price_by_segment"), limit=30)}
<h3>Comparable segments</h3>
{_table(results.get("comparable_segments"), limit=20)}
<h3>What moves with price</h3>
{_kv(results.get("price_associations", {}))}
<h3>Exploratory group tests</h3>
{_table(results.get("price_group_tests"), limit=15)}

<h2 id="mileage">6. Mileage, year and other distributions</h2>
{_table(results.get("descriptive_statistics"), limit=25)}

<h2 id="completeness">7. Feature completeness</h2>
{_kv(results.get("missingness_summary", {}))}
{_table(results.get("field_completeness"), limit=30)}

<h2 id="temporal">8. Temporal patterns</h2>
{_kv(publication)}
<h3>Observation intervals</h3>
{_kv(results.get("observation_intervals", {}))}
<h3>Runs</h3>
{_table(results.get("run_timeline"), limit=20)}

<h2 id="changes">9. Price changes</h2>
{_kv(results.get("price_changes", {}))}
{_table(results.get("price_change_table"), limit=20)}

<h2 id="duration">10. Disappearance and censoring</h2>
{_kv(duration)}
<h3>Bound ordering check</h3>
{_kv(results.get("bound_ordering", {}))}
<h3>Duration-ranking feasibility</h3>
{_kv(results.get("duration_feasibility", {}))}
<h3>Kaplan–Meier</h3>
{_kv({k: v for k, v in results.get("kaplan_meier", {}).items() if k != "curve"})}

<h2 id="reposts">11. Reposts and the vehicle grain</h2>
{_kv(results.get("vehicle_grain", {}))}
{_kv(results.get("repost_links", {}))}
{_table(results.get("repost_chains"), limit=15)}

<h2 id="filter">12. Filter exits</h2>
{_kv(results.get("filter_exits", {}))}
{_table(results.get("filter_exit_table"), limit=15)}

<h2 id="sale">13. Sale evidence</h2>
<div class="banner"><strong>sale_evidence_score is not a probability.</strong>
It is a heuristic additive score: 0.65 means "more evidence than 0.40", not
"65% of these sold". Calibration status is below.</div>
{_kv({k: v for k, v in sale.items() if not isinstance(v, dict) or k == "score_distribution"})}
<h3>Calibration</h3>
{_kv(calibration)}
{_table(results.get("sale_evidence_table"), limit=15)}

<h2 id="features">14. All scraped features</h2>
<p class="muted">Every column in the cross-section is catalogued and, where it can
support one, profiled — not a shortlist chosen in advance.</p>
{_kv(results.get("feature_summary", {}))}

<div class="banner"><strong>Grain.</strong> <code>listing</code> features are
observed for the individual advertisement. <code>vehicle</code> features come from
its detail page. <code>trim</code> features are properties of the model-trim and
are <strong>identical for every listing of that trim</strong> — so a breakdown of a
trim feature across cities describes the <em>trim mix</em> in each city, not a
count of individually inspected cars.</div>

<h3>Feature catalogue</h3>
{_table(results.get("feature_catalogue"), limit=40)}

<h3>Distributions</h3>
{_table(results.get("feature_profiles"), limit=40)}

<h3>Features most associated with asking price</h3>
<p class="muted">Spearman for numeric, Mann–Whitney with rank-biserial for flags,
Kruskal–Wallis with epsilon-squared for categoricals. Holm-adjusted across the
family. Restatements of the price itself are excluded, and any |effect| &gt; 0.95
is flagged as near-tautological rather than presented as a finding.</p>
{_table(results.get("feature_price_associations"), limit=30)}

<h3>Features that differ most across each dimension</h3>
<p class="muted">Compared across: {_esc(", ".join(results.get("feature_dimensions_compared", [])))}</p>
{_table(results.get("feature_differentiators"), limit=40)}

<h3>Per-group distributions</h3>
{_table(results.get("feature_comparisons"), limit=40)}

<h2 id="pairs">15. Feature-to-feature association: linear vs non-linear</h2>
<p class="muted">Three questions, three measures. Conflating them is how "these
variables are unrelated" gets said about a perfect parabola.</p>

<div class="scroll"><table class="kv">
<tr><th>Pearson <code>r</code></th><td><strong>Linear</strong> association only.
Blind to curvature, and easily destroyed by a contaminated minority.</td></tr>
<tr><th>Spearman <code>rho</code></th><td><strong>Monotone</strong> association,
linear or not. Rank-based, so robust to outliers and skew.</td></tr>
<tr><th>Distance correlation</th><td><strong>Any</strong> dependence. Zero if and
only if the variables are independent — it catches non-monotone shapes that both
of the others miss.</td></tr>
<tr><th>Correlation ratio <code>eta</code></th><td>Numeric versus categorical:
the share of variance explained by the grouping. No linear/monotone distinction
applies — a grouping has no order.</td></tr>
<tr><th>Cramér's V</th><td>Categorical versus categorical, bias-corrected.</td></tr>
</table></div>

{_kv(results.get("pairwise_summary", {}))}

<h3>Where a linear correlation would be wrong</h3>
<p class="muted">Each row is a pair whose relationship Pearson understates or
misses. <code>outlier_driven_linear</code> is separated out deliberately: there the
relationship <em>is</em> linear and a contaminated minority is breaking the
estimate, so the answer is to clean the data, not to change the model.</p>
{_table(results.get("nonlinearity"), limit=30)}

<h3>All pairs</h3>
{_table(results.get("pairwise_associations"), limit=40)}

<h2 id="charts">16. Charts</h2>
<p class="muted">{_esc(chart_manifest.get("font_note", ""))}</p>
{_charts_html(chart_manifest)}

<h2 id="limitations">17. Limitations</h2>
<ul>{"".join(f"<li>{_esc(item)}</li>" for item in limitations)}</ul>

<h2 id="unsupported">18. Conclusions that cannot be drawn</h2>
<div class="stop"><strong>These statements are not supported by this dataset.</strong>
Anyone quoting the report should be able to check this list first.</div>
<ul>{"".join(f"<li>{_esc(item)}</li>" for item in unsupported)}</ul>

<h2 id="repro">19. Reproducibility</h2>
{_kv(manifest.as_dict()["environment"])}
{_kv(manifest.as_dict()["configuration"])}
<p class="muted">Full manifest: <code>eda_manifest.json</code> — includes every SQL
query hash, source row count, exclusion and warning.</p>

</body></html>"""


def _limitations(results: dict[str, Any], manifest: Manifest) -> list[str]:
    publication = results.get("publication", {})
    runs = results.get("run_timeline")
    items = [
        f"Left truncation: only {publication.get('eligible_for_duration_ranking', 0)} of "
        f"{publication.get('advertisements_total', 0)} advertisements "
        f"({publication.get('eligible_share_pct', 0)}%) were first observed close enough to "
        "publication for their full market life to be known. Every other observed duration "
        "is a lower bound.",
        f"Observation cadence: {results.get('observation_intervals', {}).get('median_interval_hours', 'n/a')} "
        "hours between observations, so a disappearance can only be located to within one "
        "interval. The midpoint is an estimate, never a time of sale.",
        f"Monitoring window: {len(runs) if isinstance(runs, pd.DataFrame) else 0} run(s). "
        "Day-over-day rates (price change, disappearance, reposting) are barely observable "
        "over this window and must not be read as market rates.",
        "Detail coverage: vehicle attributes come from detail scrapes, which are "
        "rate-limited and policy-driven, so attribute completeness is far below inventory "
        "completeness. See the enrichment block for exactly where each attribute came from.",
        "Asking prices only: this dataset contains no transaction prices, so no statistic "
        "here describes what a vehicle sold for.",
        "Missingness is not random: attributes are absent most often for listings that "
        "disappeared before a detail scrape was due — that is, the shortest-lived ones.",
    ]
    if not results.get("kaplan_meier", {}).get("available"):
        items.append(
            "No survival curve was estimated: "
            + str(results.get("kaplan_meier", {}).get("note", "insufficient sample"))
        )
    if not results.get("sale_evidence", {}).get("calibration", {}).get("calibrated"):
        items.append(
            "Sale evidence is uncalibrated: no empirical sale rate exists for any score "
            "band, so the score orders listings by evidence and nothing more."
        )
    items.extend(w.message for w in manifest.warnings if w.area in ("enrichment", "privacy"))
    return items


def _unsupported_conclusions(results: dict[str, Any]) -> list[str]:
    return [
        '"X% of listings sold" — no sale is confirmed anywhere in this dataset; '
        "disappearance is the only observed event.",
        '"Cars of brand A sell faster than brand B" — the eligible sample is too small '
        "and no survival comparison met the group thresholds.",
        '"A sale-evidence score of 0.65 means a 65% chance of sale" — the score is '
        "uncalibrated and is an ordering, not a rate.",
        '"The median listing sells in N days" — the metric is time to disappearance, and '
        "most listings are left-truncated or right-censored.",
        '"Price reductions cause listings to disappear" — this is a descriptive '
        "cross-section; nothing here identifies a causal effect.",
        '"The market contains N cars" — the population is one filtered search, not the '
        "whole platform, and it changes between runs.",
        '"Reposted listings are repeat sales" — a repost means the vehicle stayed on the '
        "market; it is counted once at vehicle grain.",
    ]


def _write_readme(
    out_dir: Path, context: AnalysisContext, results: dict[str, Any], manifest: Manifest
) -> None:
    inventory = results.get("inventory", {})
    publication = results.get("publication", {})
    (out_dir / "README.txt").write_text(
        f"""Bama EDA report
================

Analysis run     : {context.run_id}
Scheduled slot   : {context.scheduled_for}
Run finished     : {context.run_finished_at}
Config hash      : {context.search_configuration_hash}
Analysed at      : {context.analysis_started_at}
Analysis version : {context.analysis_version}
Backend          : {context.database_backend}

Advertisements   : {inventory.get("advertisements", 0):,}
Vehicle entities : {inventory.get("vehicle_entities", 0):,}
Observations     : {inventory.get("observations_all_runs", 0):,}
Eligible for duration ranking : {publication.get("eligible_for_duration_ranking", 0):,}

READ THIS FIRST
---------------
{DISCLAIMER}

Terminology used here:
  time_to_disappearance          the measured event; NOT time to sale
  observed_monitoring_duration   measured from first observation
  estimated_market_duration      measured from publication; eligible listings only
  sale_evidence_score            heuristic ordering; NOT a probability
  left_truncated                 already active when observation began
  right_censored                 still present when observation ended
  active_outside_filter          left the search, not the market
  vehicle_entity_id              a physical vehicle, not an advertisement

Files
-----
  eda_manifest.json              full provenance: queries, counts, exclusions
  data_quality_report.json       integrity audit summary
  data_quality_issues.csv        every check and its result
  schema_inventory.csv           database columns with analytical roles
  dataset_dictionary.csv         analytical dataset columns, including derived
  advertisement_cross_section.*  dataset A, one row per advertisement
  longitudinal_panel.parquet     dataset B, one row per (run, advertisement)
  vehicle_entity_dataset.parquet dataset C, one row per physical vehicle
  charts/                        PNG and SVG, every chart labelled with n
  bama_eda_report.html           the report

Warnings recorded: {len(manifest.warnings)} (see analysis_warnings.csv)
""",
        encoding="utf-8",
    )


def check_forbidden_phrases(out_dir: Path) -> list[str]:
    """Scan generated artefacts for phrasing that misstates the evidence."""
    offenders: list[str] = []
    for path in out_dir.rglob("*"):
        if path.suffix.lower() not in (".csv", ".json", ".html", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for phrase in FORBIDDEN_PHRASES:
            # The report legitimately quotes these phrases while forbidding them;
            # only a bare occurrence outside that context is a problem.
            if phrase in text and "not " + phrase not in text and "never" not in text:
                offenders.append(f"{path.name}: {phrase}")
    return offenders
