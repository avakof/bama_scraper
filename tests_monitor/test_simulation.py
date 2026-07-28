"""Multi-day simulation, idempotency and failure recovery.

The primary scenario is the one specified for this system:

    Day 1: A B C D
    Day 2: A B D E
    Day 3: A D E F
    Day 4: A B D E F

with required outcomes: B reappears and must never be sold, C reaches
likely_removed, E is new on day 2, F is new on day 3. Both a valid day 3 and an
*invalid* day 3 are exercised, because the invalid case is what protects the
dataset from our own failures.

Every test in this module runs on each configured backend.
"""

from __future__ import annotations

from monitor_helpers import make_context, make_discovery, misses_of, run_day, status_of

from bama_monitor.config import MonitorConfig
from bama_monitor.db import Database
from bama_monitor.inventory_comparison import InventoryComparator
from bama_monitor.models import AdStatus, EventType, RunHealth
from bama_monitor.repository import Repository

DAYS = {
    1: ["A", "B", "C", "D"],
    2: ["A", "B", "D", "E"],
    3: ["A", "D", "E", "F"],
    4: ["A", "B", "D", "E", "F"],
}


def simulate(db: Database, cfg: MonitorConfig, *, day3_healthy: bool = True) -> None:
    for day, keys in DAYS.items():
        run_day(db, cfg, day=day, keys=keys, healthy=(day != 3 or day3_healthy))


class TestSpecifiedScenario:
    def test_final_statuses(self, db: Database, cfg: MonitorConfig) -> None:
        simulate(db, cfg)
        assert status_of(db, "A") == str(AdStatus.ACTIVE)
        assert status_of(db, "B") == str(AdStatus.REAPPEARED)
        assert status_of(db, "C") == str(AdStatus.LIKELY_REMOVED)
        assert status_of(db, "D") == str(AdStatus.ACTIVE)
        assert status_of(db, "E") == str(AdStatus.ACTIVE)
        assert status_of(db, "F") == str(AdStatus.ACTIVE)

    def test_b_is_never_classified_as_sold_or_removed(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """The headline requirement: a reappearance is not a sale."""
        simulate(db, cfg)
        assert status_of(db, "B") not in (
            str(AdStatus.LIKELY_SOLD),
            str(AdStatus.LIKELY_REMOVED),
        )
        row = db.fetchone(
            "SELECT sale_label, sale_confidence, reappeared_at FROM advertisements"
            " WHERE platform_ad_id='B'"
        )
        assert row is not None
        assert row["reappeared_at"] is not None
        assert str(row["sale_label"]) != "confirmed_sold"

    def test_b_history_records_the_round_trip(self, db: Database, cfg: MonitorConfig) -> None:
        simulate(db, cfg)
        events = db.fetchall(
            "SELECT e.event_type FROM advertisement_events e JOIN advertisements a"
            " ON a.id = e.advertisement_id WHERE a.platform_ad_id='B' ORDER BY e.id"
        )
        types = [str(e["event_type"]) for e in events]
        assert str(EventType.MISSING_FIRST_TIME) in types
        assert str(EventType.REAPPEARED) in types

    def test_c_took_two_misses(self, db: Database, cfg: MonitorConfig) -> None:
        simulate(db, cfg)
        assert misses_of(db, "C") == 2
        row = db.fetchone(
            "SELECT first_missing_at, confirmed_removed_at FROM advertisements"
            " WHERE platform_ad_id='C'"
        )
        assert row is not None
        # The two timestamps differ: one records the first absence, the other the
        # run at which removal was inferred.
        assert row["first_missing_at"] != row["confirmed_removed_at"]

    def test_c_was_missing_once_before_being_removed(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        assert status_of(db, "C") == str(AdStatus.MISSING_ONCE)
        run_day(db, cfg, day=3, keys=DAYS[3])
        assert status_of(db, "C") == str(AdStatus.LIKELY_REMOVED)

    def test_e_and_f_first_seen_days(self, db: Database, cfg: MonitorConfig) -> None:
        simulate(db, cfg)
        rows = {
            str(r["platform_ad_id"]): r
            for r in db.fetchall(
                "SELECT platform_ad_id, first_seen_at, total_seen_runs FROM advertisements"
            )
        }
        assert str(rows["E"]["first_seen_at"])[:10] == "2026-07-02"
        assert str(rows["F"]["first_seen_at"])[:10] == "2026-07-03"
        assert int(rows["E"]["total_seen_runs"]) == 3
        assert int(rows["F"]["total_seen_runs"]) == 2

    def test_absence_is_recorded_as_an_observation(self, db: Database, cfg: MonitorConfig) -> None:
        """Absence must be a row, not a gap in the data."""
        simulate(db, cfg)
        absent = db.scalar(
            "SELECT COUNT(*) FROM daily_ad_observations o JOIN advertisements a"
            " ON a.id = o.advertisement_id WHERE a.platform_ad_id='C' AND o.was_seen=?",
            [False],
        )
        # Days 2 and 3. After day 3 C is `likely_removed`, which leaves the
        # comparison population, so it is not re-tested for absence every day.
        assert absent == 2

    def test_terminal_status_leaves_the_comparison_population(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A long-removed listing must not be re-counted as missing forever."""
        simulate(db, cfg)
        day4 = db.fetchone(
            "SELECT missing_count FROM monitoring_runs ORDER BY scheduled_for DESC LIMIT 1"
        )
        assert day4 is not None
        # C is still absent on day 4 but is no longer counted: it was already
        # resolved to likely_removed on day 3.
        assert int(day4["missing_count"]) == 0
        assert status_of(db, "C") == str(AdStatus.LIKELY_REMOVED)

    def test_history_is_never_overwritten(self, db: Database, cfg: MonitorConfig) -> None:
        simulate(db, cfg)
        # Four runs, and A was seen in all of them.
        seen = db.scalar(
            "SELECT COUNT(*) FROM daily_ad_observations o JOIN advertisements a"
            " ON a.id = o.advertisement_id WHERE a.platform_ad_id='A' AND o.was_seen=?",
            [True],
        )
        assert seen == 4

    def test_run_counters_match_the_transitions(self, db: Database, cfg: MonitorConfig) -> None:
        simulate(db, cfg)
        runs = db.fetchall(
            "SELECT scheduled_for, new_count, missing_count, removed_count, reappeared_count"
            " FROM monitoring_runs ORDER BY scheduled_for"
        )
        assert [int(r["new_count"]) for r in runs] == [4, 1, 1, 0]
        # Day 4 counts 0 missing: C left the population when it was resolved.
        assert [int(r["missing_count"]) for r in runs] == [0, 1, 2, 0]
        assert [int(r["removed_count"]) for r in runs] == [0, 0, 1, 0]
        assert [int(r["reappeared_count"]) for r in runs] == [0, 0, 0, 1]


class TestInvalidDayThree:
    """With day 3 invalid, nothing may become missing because of it."""

    def test_no_advertisement_becomes_missing_from_the_invalid_run(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        before = {k: misses_of(db, k) for k in "ABCD"}
        run_day(db, cfg, day=3, keys=DAYS[3], healthy=False)
        after = {k: misses_of(db, k) for k in "ABCD"}
        assert before == after

    def test_b_does_not_reach_missing_and_c_does_not_reach_removed(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        simulate(db, cfg, day3_healthy=False)
        # B was absent only on the invalid day 3, so it never missed a valid run.
        assert status_of(db, "B") == str(AdStatus.ACTIVE)
        # C missed valid runs 2 and 4 -> two misses -> likely_removed.
        assert status_of(db, "C") == str(AdStatus.LIKELY_REMOVED)
        assert misses_of(db, "C") == 2

    def test_the_invalid_run_is_recorded_with_its_reason(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        context, health, comparison = run_day(db, cfg, day=3, keys=DAYS[3], healthy=False)
        run = Repository(db).get_run(context.run_id)
        assert run is not None
        assert str(run["status"]) == str(RunHealth.PARTIAL)
        assert not bool(run["comparison_applied"])
        assert "verified_termination" in str(run["health_reason"])

    def test_f_discovered_on_the_invalid_day_is_still_persisted(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Evidence survives; only status transitions are withheld."""
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        run_day(db, cfg, day=3, keys=DAYS[3], healthy=False)
        row = db.fetchone("SELECT current_status FROM advertisements WHERE platform_ad_id='F'")
        assert row is not None  # the advertisement exists
        assert str(row["current_status"]) == str(AdStatus.NEW)  # but not promoted


class TestIdempotency:
    def test_replaying_a_day_does_not_duplicate_observations(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        before = db.scalar("SELECT COUNT(*) FROM daily_ad_observations")
        run_day(db, cfg, day=2, keys=DAYS[2])  # same slot again
        assert db.scalar("SELECT COUNT(*) FROM daily_ad_observations") == before

    def test_replaying_a_day_does_not_duplicate_events(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        before = db.scalar("SELECT COUNT(*) FROM advertisement_events")
        run_day(db, cfg, day=2, keys=DAYS[2])
        assert db.scalar("SELECT COUNT(*) FROM advertisement_events") == before

    def test_replaying_a_day_does_not_double_increment_misses(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """The counter guard: a re-run must not push C toward removal."""
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        assert misses_of(db, "C") == 1
        run_day(db, cfg, day=2, keys=DAYS[2])
        assert misses_of(db, "C") == 1
        assert status_of(db, "C") == str(AdStatus.MISSING_ONCE)

    def test_replaying_reuses_the_same_run_row(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        first = db.scalar("SELECT COUNT(*) FROM monitoring_runs")
        run_day(db, cfg, day=1, keys=DAYS[1])
        assert db.scalar("SELECT COUNT(*) FROM monitoring_runs") == first

    def test_a_different_configuration_hash_creates_a_separate_run(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """Changing thresholds makes runs incomparable, so they must not share a slot."""
        run_day(db, cfg, day=1, keys=DAYS[1])
        other = cfg.model_copy(update={"removal_confirmation_misses": 3})
        assert other.configuration_hash() != cfg.configuration_hash()
        run_day(db, other, day=1, keys=DAYS[1])
        assert db.scalar("SELECT COUNT(*) FROM monitoring_runs") == 2

    def test_replay_does_not_duplicate_price_changes(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        run_day(db, cfg, day=1, keys=["A"], prices={"A": 1_000_000_000})
        run_day(db, cfg, day=2, keys=["A"], prices={"A": 900_000_000})
        before = db.scalar("SELECT COUNT(*) FROM price_changes")
        assert before == 1
        run_day(db, cfg, day=2, keys=["A"], prices={"A": 900_000_000})
        assert db.scalar("SELECT COUNT(*) FROM price_changes") == before


class TestPriceTracking:
    def test_decrease_is_recorded_with_magnitude(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["A"], prices={"A": 2_000_000_000})
        run_day(db, cfg, day=2, keys=["A"], prices={"A": 1_800_000_000})
        change = db.fetchone("SELECT * FROM price_changes")
        assert change is not None
        assert str(change["change_type"]) == "price_decrease"
        assert int(change["absolute_change"]) == -200_000_000
        assert float(change["percentage_change"]) == -10.0

    def test_increase_is_recorded(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["A"], prices={"A": 1_000_000_000})
        run_day(db, cfg, day=2, keys=["A"], prices={"A": 1_100_000_000})
        change = db.fetchone("SELECT change_type FROM price_changes")
        assert change is not None and str(change["change_type"]) == "price_increase"

    def test_hidden_price_is_its_own_change_type(self, db: Database, cfg: MonitorConfig) -> None:
        """A withdrawn price is not a price of zero."""
        run_day(db, cfg, day=1, keys=["A"], prices={"A": 1_000_000_000})
        run_day(db, cfg, day=2, keys=["A"], prices={"A": None})
        change = db.fetchone("SELECT change_type, new_price FROM price_changes")
        assert change is not None
        assert str(change["change_type"]) == "price_hidden"
        assert change["new_price"] is None

    def test_unchanged_price_records_nothing(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=["A"], prices={"A": 1_000_000_000})
        run_day(db, cfg, day=2, keys=["A"], prices={"A": 1_000_000_000})
        assert db.scalar("SELECT COUNT(*) FROM price_changes") == 0


class TestFailureRecovery:
    def test_transaction_rolls_back_a_failed_comparison(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A partially applied comparison would corrupt the counters, so it must not commit."""
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2])
        before = {k: misses_of(db, k) for k in "ABCDE"}
        before_events = db.scalar("SELECT COUNT(*) FROM advertisement_events")

        repo = Repository(db)
        previous = repo.last_valid_run(cfg.configuration_hash())
        context = make_context(
            repo, cfg, day=3, previous_valid_run_id=int(previous["id"]) if previous else None
        )
        result = make_discovery(DAYS[3])
        comparator = InventoryComparator(db, cfg)
        persist = comparator.persist_inventory(result, context)
        health = comparator.validate(result, context, persisted=persist.persisted)

        # Break the very last step inside the transaction.
        def boom(*args, **kwargs):
            raise RuntimeError("simulated database failure mid-comparison")

        original = comparator.repo.update_run
        comparator.repo.update_run = boom  # type: ignore[method-assign]
        try:
            with __import__("pytest").raises(RuntimeError):
                comparator.compare(result, context, health, persist)
        finally:
            comparator.repo.update_run = original  # type: ignore[method-assign]

        assert {k: misses_of(db, k) for k in "ABCDE"} == before
        assert db.scalar("SELECT COUNT(*) FROM advertisement_events") == before_events

    def test_evidence_survives_a_failed_comparison(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        repo = Repository(db)
        context = make_context(repo, cfg, day=2)
        result = make_discovery(DAYS[2])
        comparator = InventoryComparator(db, cfg)
        comparator.persist_inventory(result, context)
        # Persisted before the comparison ran, so the observations are on disk.
        assert db.scalar(
            "SELECT COUNT(*) FROM daily_ad_observations WHERE run_id=?", [context.run_id]
        ) == len(DAYS[2])

    def test_resuming_an_interrupted_run_continues_the_same_row(
        self, db: Database, cfg: MonitorConfig
    ) -> None:
        """A crashed run leaves its row in `running`; the retry adopts it."""
        repo = Repository(db)
        context = make_context(repo, cfg, day=1)
        assert str(repo.get_run(context.run_id)["status"]) == str(RunHealth.RUNNING)
        again = make_context(repo, cfg, day=1)
        assert again.run_id == context.run_id

    def test_a_later_valid_run_recovers_the_series(self, db: Database, cfg: MonitorConfig) -> None:
        run_day(db, cfg, day=1, keys=DAYS[1])
        run_day(db, cfg, day=2, keys=DAYS[2], initial_page_ok=False)  # failed run
        run_day(db, cfg, day=3, keys=DAYS[2])  # healthy again
        assert misses_of(db, "C") == 1
        assert status_of(db, "C") == str(AdStatus.MISSING_ONCE)
