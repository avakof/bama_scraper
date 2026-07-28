"""Analytical controls: filter exit, left truncation, vehicle grain, calibration.

These tests defend against *analytical* errors rather than crashes. Each one
corresponds to a way the system could produce a confident, well-formatted, wrong
answer to "which cars sell faster?".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from monitor_helpers import misses_of, run_day, status_of

from bama_monitor.analytics import (
    fastest_disappearing,
    filter_exit_report,
    left_truncated_disappearances,
    time_to_disappearance,
    truncation_summary,
)
from bama_monitor.config import MonitorConfig
from bama_monitor.db import Database
from bama_monitor.models import (
    AVAILABILITY_BY_VERDICT,
    AdStatus,
    DetailAvailability,
    DetailVerdict,
    EventType,
    FilterExitReason,
    PublishedAtSource,
)
from bama_monitor.publication import (
    classify_published_source,
    market_duration_bounds,
    truncation_status,
)
from bama_monitor.search_filter import (
    SearchFilter,
    assert_bounds_parsed,
    classify_absence,
    evaluate,
    parse_search_filter,
)
from bama_monitor.state_machine import AdState, on_seen
from bama_monitor.validation import (
    VALID_OUTCOMES,
    band_of,
    calibration,
    draw_sample,
    ingest_worksheet,
    write_worksheet,
)
from bama_monitor.vehicle_grain import (
    rebuild_vehicle_entities,
    summarize_vehicle,
    vehicle_durations,
)

T0 = datetime(2026, 7, 1, 11, 0, tzinfo=UTC)


def _header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")


def _fill_outcomes(path: Path, outcome: str) -> int:
    """Fill in `observed_outcome` the way a human would, via the CSV itself."""
    import csv as _csv

    text = path.read_text(encoding="utf-8-sig")
    rows = list(_csv.DictReader(text.splitlines()))
    for row in rows:
        row["observed_outcome"] = outcome
        row["outcome_source"] = "manual_check"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


REFERENCE_URL = (
    "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian"
)


# ---------------------------------------------------------------------------
# Filter exit: a listing can leave the search without leaving the market
# ---------------------------------------------------------------------------


class TestSearchFilterParsing:
    def test_reference_url_bounds(self) -> None:
        f = parse_search_filter(REFERENCE_URL)
        # Both are LOWER bounds, verified against the live endpoint.
        assert f.price_from == 1_000_000_000
        assert f.year_from == 1397
        assert f.price_to is None and f.year_to is None
        assert not f.is_empty

    def test_bounds_are_read_from_api_params_not_the_wrapper(self) -> None:
        """`parse_filters` nests the bounds; reading the top level finds nothing.

        A filter that parses to nothing makes every absence inconclusive, which is
        indistinguishable from "no listing ever left the filter" -- a silent
        degradation that would disable the whole false-removal guard.
        """
        f = parse_search_filter(REFERENCE_URL)
        assert assert_bounds_parsed(REFERENCE_URL, f) == []

    def test_declared_but_unparsed_bound_is_reported(self) -> None:
        empty = SearchFilter()
        warnings = assert_bounds_parsed(REFERENCE_URL, empty)
        assert len(warnings) == 2
        assert any("price" in w for w in warnings)
        assert any("year" in w for w in warnings)


class TestFilterEvaluation:
    filter = parse_search_filter(REFERENCE_URL)

    def test_listing_inside_the_filter_passes(self) -> None:
        reason, evidence = evaluate(
            {"price_toman": 1_500_000_000, "year_jalali": 1400}, self.filter
        )
        assert reason is None
        assert set(evidence["checked"]) == {"price_from", "year_from"}
        assert evidence["skipped"] == []

    def test_price_cut_below_the_bound_is_a_filter_exit(self) -> None:
        """The review's example: 1,050,000,000 -> 950,000,000."""
        reason, evidence = evaluate({"price_toman": 950_000_000, "year_jalali": 1400}, self.filter)
        assert reason is FilterExitReason.PRICE_BELOW_FILTER
        assert "950,000,000" in evidence["reason_detail"]

    def test_year_outside_the_bound_is_a_filter_exit(self) -> None:
        reason, _ = evaluate({"price_toman": 1_500_000_000, "year_jalali": 1396}, self.filter)
        assert reason is FilterExitReason.YEAR_OUTSIDE_FILTER

    def test_negotiable_price_is_not_a_price_below_the_bound(self) -> None:
        """Bama encodes 'negotiable' as price 0; that is not a cheap car."""
        reason, evidence = evaluate({"price_toman": 0, "year_jalali": 1400}, self.filter)
        assert reason is None
        assert "price_from" in evidence["skipped"]

    def test_a_gregorian_year_is_converted_not_rejected(self) -> None:
        """2021 where 1400 was expected must not manufacture a filter exit."""
        reason, _ = evaluate({"price_toman": 1_500_000_000, "year_text": "2021"}, self.filter)
        assert reason is None

    def test_a_tiny_shortfall_is_within_tolerance(self) -> None:
        reason, _ = evaluate({"price_toman": 999_500_000, "year_jalali": 1400}, self.filter)
        assert reason is None

    def test_missing_fields_are_recorded_as_skipped_not_passed(self) -> None:
        _, evidence = evaluate({"year_jalali": 1400}, self.filter)
        assert "price_from" in evidence["skipped"]
        assert "price_from" not in evidence["checked"]

    def test_unexplainable_absence_says_so(self) -> None:
        reason, evidence = classify_absence(
            {"price_toman": 1_500_000_000, "year_jalali": 1400}, self.filter
        )
        assert reason is FilterExitReason.SEARCH_INDEX_INCONSISTENCY
        assert "search index" in evidence["reason_detail"]


class TestFilterExitStopsTheMissCounter:
    def test_active_outside_filter_is_not_in_the_comparison_population(self) -> None:
        from bama_monitor.models import COMPARABLE_STATUSES, TERMINAL_STATUSES

        assert AdStatus.ACTIVE_OUTSIDE_FILTER not in COMPARABLE_STATUSES
        assert AdStatus.ACTIVE_OUTSIDE_FILTER in TERMINAL_STATUSES

    def test_filter_exit_is_not_a_disappearance(self) -> None:
        from bama_monitor.models import DISAPPEARED_STATUSES

        assert AdStatus.ACTIVE_OUTSIDE_FILTER not in DISAPPEARED_STATUSES
        assert AdStatus.LIKELY_REMOVED in DISAPPEARED_STATUSES

    def test_reentry_into_the_filter_is_its_own_event(self) -> None:
        state = AdState(
            status=AdStatus.ACTIVE_OUTSIDE_FILTER,
            consecutive_misses=0,
            total_seen_runs=3,
            total_missing_runs=1,
            first_seen_at=T0,
            last_seen_at=T0 + timedelta(days=1),
        )
        outcome = on_seen(state, observed_at=T0 + timedelta(days=5))
        assert outcome.state.status is AdStatus.ACTIVE
        kinds = [e for e, _ in outcome.events]
        # Not `reappeared`: it never left the market, only the search.
        assert EventType.FILTER_REENTRY in kinds
        assert EventType.REAPPEARED not in kinds

    def test_filter_exit_survival_row_is_censored_not_an_event(self) -> None:
        from bama_monitor.duration_estimation import survival_row

        row = survival_row(
            {
                "platform_ad_id": "x",
                "current_status": str(AdStatus.ACTIVE_OUTSIDE_FILTER),
                "first_seen_at": T0.isoformat(),
                "last_seen_at": (T0 + timedelta(days=2)).isoformat(),
                "first_missing_at": (T0 + timedelta(days=3)).isoformat(),
            }
        )
        assert row["event_observed"] == 0
        assert row["event_type"] == "left_search_filter"


class TestAvailabilitySeparation:
    """410 says the page is gone. It does not say the car was sold."""

    def test_every_verdict_maps_to_an_availability(self) -> None:
        for verdict in DetailVerdict:
            assert str(verdict) in AVAILABILITY_BY_VERDICT, verdict

    def test_gone_is_availability_not_status_and_not_a_sale(self) -> None:
        assert AVAILABILITY_BY_VERDICT["http_410"] is DetailAvailability.GONE
        assert AVAILABILITY_BY_VERDICT["http_404"] is DetailAvailability.GONE
        # The vocabularies stay disjoint apart from the shared "unknown", which
        # means the same thing in both: we could not tell.
        statuses = {str(s) for s in AdStatus} - {"unknown"}
        availabilities = {str(a) for a in DetailAvailability} - {"unknown"}
        assert statuses & availabilities == set()
        # In particular no availability value is a status: a page being gone is
        # never itself an observed lifecycle state.
        assert "gone" not in statuses

    def test_a_live_page_is_available(self) -> None:
        assert AVAILABILITY_BY_VERDICT["detail_page_still_active"] is DetailAvailability.AVAILABLE

    def test_our_own_parse_failure_is_not_a_server_error(self) -> None:
        """Found live: a TypeError in our parser was recorded as
        ``temporary_server_error`` and marked retryable on four pre-sale listings.

        Both halves were wrong. The site answered — it was our code that failed —
        and retrying an unparseable payload cannot succeed until the code changes.
        """
        assert DetailVerdict.PARSE_ERROR != DetailVerdict.TEMPORARY_ERROR
        assert AVAILABILITY_BY_VERDICT["local_parse_error"] is DetailAvailability.UNKNOWN

    def test_a_parse_failure_makes_no_claim_about_the_page(self) -> None:
        """It must not imply the page is unavailable, gone, or fine."""
        availability = AVAILABILITY_BY_VERDICT["local_parse_error"]
        assert availability not in (
            DetailAvailability.TEMPORARILY_UNAVAILABLE,
            DetailAvailability.GONE,
            DetailAvailability.AVAILABLE,
            DetailAvailability.REPORTS_UNAVAILABLE,
        )


# ---------------------------------------------------------------------------
# Left truncation
# ---------------------------------------------------------------------------


class TestPublicationProvenance:
    def test_absolute_jalali_date_is_reliable(self) -> None:
        source = classify_published_source("1405/4/28", 1.0)
        assert source is PublishedAtSource.DETAIL_ABSOLUTE

    def test_relative_phrase_is_reliable(self) -> None:
        assert classify_published_source("۳ روز پیش", 1.0) is PublishedAtSource.DETAIL_RELATIVE

    def test_moments_ago_is_coarse(self) -> None:
        """'دقایقی پیش' pins the day, not the hour."""
        assert classify_published_source("دقایقی پیش", 1.0) is PublishedAtSource.DETAIL_COARSE

    def test_nothing_parsed_is_unknown(self) -> None:
        assert classify_published_source("دقایقی پیش", None) is PublishedAtSource.UNKNOWN


class TestLeftTruncation:
    def test_baseline_listing_is_left_truncated_and_not_rankable(self) -> None:
        """Monitoring starts long after publication: earlier life unobserved."""
        status = truncation_status(
            published_at=T0 - timedelta(days=40),
            published_source=PublishedAtSource.DETAIL_ABSOLUTE,
            first_seen_at=T0,
        )
        assert status["left_truncated"] is True
        assert status["eligible_for_duration_ranking"] is False
        assert status["entry_delay_seconds"] == 40 * 86400

    def test_listing_published_just_before_first_sighting_is_rankable(self) -> None:
        status = truncation_status(
            published_at=T0 - timedelta(hours=20),
            published_source=PublishedAtSource.DETAIL_RELATIVE,
            first_seen_at=T0,
        )
        assert status["left_truncated"] is False
        assert status["eligible_for_duration_ranking"] is True

    def test_coarse_publication_time_is_never_rankable(self) -> None:
        status = truncation_status(
            published_at=T0 - timedelta(hours=1),
            published_source=PublishedAtSource.DETAIL_COARSE,
            first_seen_at=T0,
        )
        assert status["eligible_for_duration_ranking"] is False
        assert status["published_at_reliable"] is False

    def test_unknown_publication_defaults_to_truncated(self) -> None:
        """The unsafe default must be the conservative one."""
        status = truncation_status(published_at=None, published_source=None, first_seen_at=T0)
        assert status["left_truncated"] is True
        assert status["eligible_for_duration_ranking"] is False

    def test_publication_after_first_sighting_is_rejected(self) -> None:
        """Impossible ordering must not become a zero delay."""
        status = truncation_status(
            published_at=T0 + timedelta(days=3),
            published_source=PublishedAtSource.DETAIL_ABSOLUTE,
            first_seen_at=T0,
        )
        assert status["eligible_for_duration_ranking"] is False
        assert status["entry_delay_seconds"] is None

    def test_tolerance_boundary(self) -> None:
        inside = truncation_status(
            published_at=T0 - timedelta(hours=35),
            published_source=PublishedAtSource.DETAIL_RELATIVE,
            first_seen_at=T0,
        )
        outside = truncation_status(
            published_at=T0 - timedelta(hours=37),
            published_source=PublishedAtSource.DETAIL_RELATIVE,
            first_seen_at=T0,
        )
        assert inside["eligible_for_duration_ranking"] is True
        assert outside["eligible_for_duration_ranking"] is False


class TestMarketDurationRequiresEligibility:
    def test_ineligible_listing_gets_no_market_duration(self) -> None:
        """No number rather than a number with a caveat."""
        bounds = market_duration_bounds(
            published_at=T0 - timedelta(days=40),
            last_seen_at=T0 + timedelta(days=2),
            first_missing_at=T0 + timedelta(days=3),
            eligible=False,
        )
        assert set(bounds.values()) == {None}

    def test_eligible_listing_measures_from_publication(self) -> None:
        bounds = market_duration_bounds(
            published_at=T0 - timedelta(days=1),
            last_seen_at=T0 + timedelta(days=2),
            first_missing_at=T0 + timedelta(days=3),
            eligible=True,
        )
        # Publication is a day earlier than first sighting, so market duration
        # exceeds the observed one by exactly that day.
        assert bounds["market_minimum_days"] == 3.0
        assert bounds["market_maximum_days"] == 4.0
        assert bounds["market_estimate_days"] == 3.5


class TestRankingExcludesTruncatedListings:
    def _disappeared(self, db: Database, cfg: MonitorConfig) -> None:
        # Six listings stay present throughout. Two reasons, both deliberate
        # behaviour of the health gate: a run that discovers nothing is judged
        # unhealthy, and a run in which more than half the known population would
        # go missing is vetoed as implausible. So the population has to be large
        # enough that losing two listings is credible.
        keepers = [f"k{i}" for i in range(6)]
        run_day(db, cfg, day=1, keys=[*keepers, "baseline", "fresh"])
        run_day(db, cfg, day=2, keys=keepers)
        run_day(db, cfg, day=3, keys=keepers)
        # `baseline` was published five weeks before monitoring began; `fresh` the
        # day before it was first seen.
        db.execute(
            "UPDATE advertisements SET published_at=?, published_at_reliable=?,"
            " left_truncated=?, eligible_for_duration_ranking=?, entry_delay_seconds=?,"
            " sale_confidence=? WHERE platform_ad_id=?",
            [T0 - timedelta(days=35), True, True, False, 35 * 86400.0, 0.7, "baseline"],
        )
        db.execute(
            "UPDATE advertisements SET published_at=?, published_at_reliable=?,"
            " left_truncated=?, eligible_for_duration_ranking=?, entry_delay_seconds=?,"
            " sale_confidence=? WHERE platform_ad_id=?",
            [T0 - timedelta(hours=10), True, False, True, 36000.0, 0.7, "fresh"],
        )

    def test_ranking_contains_only_fully_observed_listings(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        self._disappeared(db, cfg)
        ranked = fastest_disappearing(db, min_confidence=0.65)
        ids = {r["advertisement_id"] for r in ranked}
        assert "fresh" in ids
        assert "baseline" not in ids, "a left-truncated listing must not be ranked"

    def test_excluded_listings_are_published_with_a_reason(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Excluded is not the same as absent, so the exclusions get their own file."""
        self._disappeared(db, cfg)
        excluded = left_truncated_disappearances(db, min_confidence=0.65)
        assert [r["advertisement_id"] for r in excluded] == ["baseline"]
        assert "35.0 days before first observation" in excluded[0]["excluded_from_ranking_because"]

    def test_both_duration_concepts_are_reported_side_by_side(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        self._disappeared(db, cfg)
        rows = {r["advertisement_id"]: r for r in time_to_disappearance(db)}
        baseline = rows["baseline"]
        fresh = rows["fresh"]
        # Observed duration exists for both.
        assert baseline["observed_monitoring_estimate_days"] is not None
        assert fresh["observed_monitoring_estimate_days"] is not None
        # Market duration only for the one whose whole life was seen.
        assert baseline["estimated_market_estimate_days"] is None
        assert fresh["estimated_market_estimate_days"] is not None
        # And it is longer than the observed one, by the pre-monitoring interval.
        assert fresh["estimated_market_estimate_days"] > fresh["observed_monitoring_estimate_days"]

    def test_truncation_summary_states_the_usable_share(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        self._disappeared(db, cfg)
        summary = truncation_summary(db)
        assert summary["advertisements_total"] == 8  # 6 keepers + baseline + fresh
        assert summary["eligible_for_duration_ranking"] == 1
        # Everything without a reliable publication time defaults to truncated.
        assert summary["left_truncated"] == 7


# ---------------------------------------------------------------------------
# Vehicle grain
# ---------------------------------------------------------------------------


class TestVehicleGrain:
    def _chain(self) -> list[dict]:
        """One physical car listed twice: A reposted as B."""
        return [
            {
                "id": 1,
                "platform_ad_id": "A",
                "vehicle_entity_id": "v1",
                "current_status": str(AdStatus.REPOSTED),
                "first_seen_at": T0.isoformat(),
                "last_seen_at": (T0 + timedelta(days=3)).isoformat(),
                "first_missing_at": (T0 + timedelta(days=4)).isoformat(),
                "published_at": (T0 - timedelta(hours=5)).isoformat(),
                "sale_confidence": 0.1,
                "sale_label": "unknown",
                "left_truncated": False,
            },
            {
                "id": 2,
                "platform_ad_id": "B",
                "vehicle_entity_id": "v1",
                "current_status": str(AdStatus.LIKELY_REMOVED),
                "first_seen_at": (T0 + timedelta(days=4)).isoformat(),
                "last_seen_at": (T0 + timedelta(days=9)).isoformat(),
                "first_missing_at": (T0 + timedelta(days=10)).isoformat(),
                "published_at": (T0 + timedelta(days=4)).isoformat(),
                "sale_confidence": 0.5,
                "sale_label": "possibly_sold",
                "left_truncated": False,
            },
        ]

    def test_duration_spans_the_whole_repost_chain(self) -> None:
        """Listing view: two lives of 3.5d and 5.5d. Vehicle view: one of ~9.5d."""
        summary = summarize_vehicle(self._chain())
        assert summary["listing_count"] == 2
        assert summary["repost_count"] == 1
        span_days = (
            summary["vehicle_last_seen_at"] - summary["vehicle_first_seen_at"]
        ).total_seconds() / 86400
        assert span_days == 9.0

    def test_vehicle_status_comes_from_the_latest_listing(self) -> None:
        summary = summarize_vehicle(self._chain())
        assert summary["current_status"] == str(AdStatus.LIKELY_REMOVED)

    def test_an_unsucceeded_repost_is_unknown_not_gone(self) -> None:
        """A chain whose last link is `reposted` has no recorded successor.

        Reporting that as a disappearance would count a car that is still listed
        somewhere as having left the market.
        """
        chain = self._chain()[:1]
        summary = summarize_vehicle(chain)
        assert summary["current_status"] == str(AdStatus.UNKNOWN)
        assert summary["vehicle_first_missing_at"] is None

    def test_vehicle_left_truncation_follows_the_earliest_listing(self) -> None:
        chain = self._chain()
        chain[0]["left_truncated"] = True
        assert summarize_vehicle(chain)["left_truncated"] is True

    def test_rebuild_is_idempotent(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["a", "b"])
        first = rebuild_vehicle_entities(db)
        second = rebuild_vehicle_entities(db)
        assert first == second == 2
        assert db.scalar("SELECT COUNT(*) FROM vehicle_entities") == 2

    def test_repost_is_a_covariate_not_an_outcome(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["a", "b"])
        db.execute(
            "UPDATE advertisements SET vehicle_entity_id=? WHERE platform_ad_id IN (?,?)",
            ["shared", "a", "b"],
        )
        rebuild_vehicle_entities(db)
        rows = vehicle_durations(db)
        assert len(rows) == 1, "two listings of one car must collapse to one vehicle"
        assert rows[0]["was_reposted"] == 1
        assert rows[0]["listing_count"] == 2


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestSaleScoreIsNotAProbability:
    def test_bands(self) -> None:
        assert band_of(0.0) == "0.00-0.39"
        assert band_of(0.40) == "0.40-0.64"
        assert band_of(0.65) == "0.65-0.84"
        assert band_of(0.95) == "0.85-1.00"

    def test_uncalibrated_until_labelled(self, db: Database, cfg: MonitorConfig) -> None:
        report = calibration(db)
        assert report["calibrated"] is False
        assert all(band["sale_rate"] is None for band in report["bands"])
        assert "not a probability" in report["interpretation"]

    def test_a_handful_of_labels_yields_no_rate(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        """Three labels is not a measurement."""
        keepers = [f"k{i}" for i in range(6)]
        run_day(db, cfg, day=1, keys=[*keepers, "a", "b", "c"])
        run_day(db, cfg, day=2, keys=keepers)
        run_day(db, cfg, day=3, keys=keepers)
        db.execute("UPDATE advertisements SET sale_confidence=0.7")
        sample = draw_sample(db, per_band=10)
        assert sample, "disappeared listings should be sampled"

        worksheet = tmp_path / "ws.csv"
        write_worksheet(sample, worksheet)
        assert "observed_outcome" in _header(worksheet)
        _fill_outcomes(worksheet, "sold")

        result = ingest_worksheet(db, worksheet)
        assert result["labelled"] == len(sample)
        report = calibration(db)
        assert report["total_labelled"] == len(sample)
        band = next(b for b in report["bands"] if b["band"] == "0.65-0.84")
        assert band["sale_rate"] is None
        assert "too few for a rate" in band["note"]
        assert report["calibrated"] is False

    def test_invalid_outcome_is_rejected(
        self, db: Database, cfg: MonitorConfig, tmp_path: Path
    ) -> None:
        keepers = [f"k{i}" for i in range(6)]
        run_day(db, cfg, day=1, keys=[*keepers, "a"])
        run_day(db, cfg, day=2, keys=keepers)
        run_day(db, cfg, day=3, keys=keepers)
        sample = draw_sample(db, per_band=5)
        assert sample
        worksheet = tmp_path / "ws.csv"
        write_worksheet(sample, worksheet)
        _fill_outcomes(worksheet, "definitely_sold")
        result = ingest_worksheet(db, worksheet)
        assert result["rejected"]
        assert "definitely_sold" in result["rejected"][0]

    def test_unknown_is_a_permitted_answer(self) -> None:
        """Guessing would poison the measurement this exists to make."""
        assert "unknown" in VALID_OUTCOMES


class TestFilterExitLedger:
    def test_exits_are_reported_with_their_reason(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["a", "b"])
        run_day(db, cfg, day=2, keys=["a"])
        db.execute(
            "UPDATE advertisements SET current_status=?, filter_exit_reason=?,"
            " filter_exit_at=?, detail_availability=? WHERE platform_ad_id=?",
            [
                str(AdStatus.ACTIVE_OUTSIDE_FILTER),
                str(FilterExitReason.PRICE_BELOW_FILTER),
                T0,
                str(DetailAvailability.OUTSIDE_FILTER),
                "b",
            ],
        )
        report = filter_exit_report(db)
        assert len(report) == 1
        assert report[0]["advertisement_id"] == "b"
        assert report[0]["filter_exit_reason"] == "price_below_filter"
        assert "NOT a disappearance" in report[0]["interpretation"]

    def test_a_filter_exit_stops_accumulating_misses(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """The whole point: it must not march on to `likely_removed`."""
        run_day(db, cfg, day=1, keys=["a", "b"])
        run_day(db, cfg, day=2, keys=["a"])
        assert misses_of(db, "b") == 1
        db.execute(
            "UPDATE advertisements SET current_status=?, consecutive_misses=0,"
            " filter_exit_reason=? WHERE platform_ad_id=?",
            [
                str(AdStatus.ACTIVE_OUTSIDE_FILTER),
                str(FilterExitReason.PRICE_BELOW_FILTER),
                "b",
            ],
        )
        run_day(db, cfg, day=3, keys=["a"])
        run_day(db, cfg, day=4, keys=["a"])
        assert misses_of(db, "b") == 0
        assert status_of(db, "b") == str(AdStatus.ACTIVE_OUTSIDE_FILTER)
        assert status_of(db, "b") != str(AdStatus.LIKELY_REMOVED)


class TestReportProvenance:
    def test_every_row_carries_the_run_identifiers(self, db: Database, cfg: MonitorConfig) -> None:
        from bama_monitor.reports import PROVENANCE_COLUMNS, generate_report

        run_day(db, cfg, day=1, keys=["a", "b"])
        manifest = generate_report(db, cfg)
        assert set(PROVENANCE_COLUMNS) <= set(manifest["provenance"]) | {
            "run_id",
            "scheduled_for",
            "run_started_at",
            "run_finished_at",
            "search_configuration_hash",
            "scraper_version",
            "monitor_version",
        }
        out = cfg.reports_dir / manifest["date"]
        header = (out / "new_ads.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        assert header.split(",")[: len(PROVENANCE_COLUMNS)] == PROVENANCE_COLUMNS

    def test_summary_states_that_counts_belong_to_one_run(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        import json

        from bama_monitor.reports import generate_report

        run_day(db, cfg, day=1, keys=["a"])
        manifest = generate_report(db, cfg)
        summary = json.loads(
            (cfg.reports_dir / manifest["date"] / "run_summary.json").read_text(encoding="utf-8")
        )
        notes = summary["measurement_notes"]
        assert "market churn, not scraper inconsistency" in notes["counts"]
        assert "NOT a calibrated probability" in notes["sale_evidence_score"]
        assert "left-truncated" in notes["left_truncation"]
        assert "not a disappearance" in notes["filter_exit"].lower()
        assert summary["provenance"]["run_id"] == manifest["run_id"]


class TestEveryAnalyticsViewIsReachable:
    """Each `--view` choice must actually run.

    A view whose flags are registered under the wrong subcommand parses fine and
    then dies on `args.limit`, which only shows up when someone runs that exact
    view. Enumerating the choices from the parser closes that gap.
    """

    def test_every_declared_view_executes(self, db: Database, cfg: MonitorConfig) -> None:
        from bama_monitor.cli import _cmd_analytics, build_parser

        keepers = [f"k{i}" for i in range(6)]
        run_day(db, cfg, day=1, keys=[*keepers, "gone"])
        run_day(db, cfg, day=2, keys=keepers)
        run_day(db, cfg, day=3, keys=keepers)

        parser = build_parser()
        analytics = parser._subparsers._group_actions[0].choices["analytics"]  # type: ignore[union-attr]
        views = next(a.choices for a in analytics._actions if a.dest == "view")
        assert len(views) >= 10

        for view in views:
            args = parser.parse_args(["analytics", "--view", view])
            assert _cmd_analytics(db, cfg, args) == 0, view

    def test_the_removed_view_name_is_gone(self) -> None:
        from bama_monitor.cli import build_parser

        parser = build_parser()
        analytics = parser._subparsers._group_actions[0].choices["analytics"]  # type: ignore[union-attr]
        views = next(a.choices for a in analytics._actions if a.dest == "view")
        assert "time_to_disappearance" in views
        assert "time_to_removal" not in views
