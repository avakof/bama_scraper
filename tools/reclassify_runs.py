"""Give existing runs truthful provenance.

Before the scheduler existed, every run was executed by hand. Some of those runs
were attributed to scheduled slots, which makes them look — to any report, and to
the comparison engine — like the day's genuine 13:00 observation. They are not.

This tool does not delete them. Deleting evidence to make a history look clean is
the opposite of what a monitoring system is for. It reclassifies them, and it
decides the classification from **evidence in the row**, not from an assumption:

* a run whose ``started_at`` is far from its ``scheduled_for`` cannot have been
  triggered at that slot;
* a run whose ``scheduled_for`` lies in the future relative to its own
  ``started_at`` is attributed to a slot that had not happened yet.

Both conditions are checked and reported before anything is written, and
``--apply`` is required to write.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bama_monitor.db import connect, parse_ts  # noqa: E402
from bama_monitor.models import TriggerType  # noqa: E402

#: A run started within this of its slot could plausibly have been triggered by it.
SCHEDULED_TOLERANCE = timedelta(minutes=30)


#: Runs executed within this of each other observed the same market state.
SAME_STATE_WINDOW = timedelta(minutes=60)


def classify_all(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify every run, using the whole set rather than each row alone.

    One judgement needs the full picture. Three runs executed nine minutes apart
    but attributed to three consecutive days are not three days of history: they
    observed one market state. The earliest of such a cluster is a genuine
    observation of that moment; the rest re-present it as other days, and *that*
    is what makes them synthetic — not their data, which is real.
    """
    decisions = [classify(run) for run in runs]

    ordered = sorted(
        (r for r in runs if parse_ts(r.get("started_at"))),
        key=lambda r: parse_ts(r["started_at"]),
    )
    by_id = {int(r["id"]): r for r in runs}
    cluster_leader: dict[int, int] = {}
    leader: dict[str, Any] | None = None
    for run in ordered:
        started = parse_ts(run["started_at"])
        if leader is None or started - parse_ts(leader["started_at"]) > SAME_STATE_WINDOW:
            leader = run
        cluster_leader[int(run["id"])] = int(leader["id"])

    for decision in decisions:
        run_id = decision["run_id"]
        leader_id = cluster_leader.get(run_id)
        if leader_id is None or leader_id == run_id:
            continue
        gap = parse_ts(by_id[run_id]["started_at"]) - parse_ts(by_id[leader_id]["started_at"])
        decision["trigger_type"] = str(TriggerType.DEPLOYMENT_TEST)
        decision["is_synthetic"] = True
        decision["release_slot"] = True
        decision["evidence"] = (
            f"executed {gap.total_seconds() / 60:.0f} min after run {leader_id}, so it "
            "observed the same market state; it was attributed to a different day, "
            "which would present one observation as several days of history"
        )
    # The cluster leader keeps its own evidence but is not synthetic: its data is a
    # genuine observation of the moment it ran, it simply never was a scheduled run.
    for decision in decisions:
        if cluster_leader.get(decision["run_id"]) == decision["run_id"]:
            decision["trigger_type"] = str(TriggerType.MANUAL)
            decision["is_synthetic"] = False
            decision["release_slot"] = True
            decision["evidence"] = (
                "earliest run of the pre-deployment cluster: a genuine observation of "
                "the market at its real start time, executed by hand. Real data, but "
                "never a scheduled execution, so it releases the slot"
            )
    return decisions


def classify(run: dict[str, Any]) -> dict[str, Any]:
    """Decide what a pre-existing run actually was, from its own timestamps."""
    slot = parse_ts(run.get("scheduled_for"))
    started = parse_ts(run.get("started_at"))
    run_id = int(run["id"])

    if slot is None:
        return {
            "run_id": run_id,
            "trigger_type": str(TriggerType.MANUAL),
            "is_synthetic": False,
            "release_slot": False,
            "evidence": "no scheduled_for: never claimed a slot",
        }
    if started is None:
        return {
            "run_id": run_id,
            "trigger_type": str(TriggerType.MANUAL),
            "is_synthetic": True,
            "release_slot": True,
            "evidence": "no started_at: cannot be shown to have run at its slot",
        }

    delta = started - slot
    minutes = delta.total_seconds() / 60.0

    if slot > started:
        # The slot had not arrived when the run executed. Its data is real, but
        # attributing it to that day would invent an observation.
        return {
            "run_id": run_id,
            "trigger_type": str(TriggerType.DEPLOYMENT_TEST),
            "is_synthetic": True,
            "release_slot": True,
            "evidence": (
                f"started {abs(minutes):.0f} min BEFORE its own slot "
                f"({started.isoformat()} < {slot.isoformat()}); the slot had not "
                "happened yet, so this is not an observation of it"
            ),
        }
    if abs(delta) <= SCHEDULED_TOLERANCE:
        return {
            "run_id": run_id,
            "trigger_type": str(TriggerType.SCHEDULED),
            "is_synthetic": False,
            "release_slot": False,
            "evidence": (
                f"started {minutes:+.0f} min from its slot, within the "
                f"{SCHEDULED_TOLERANCE} tolerance"
            ),
        }
    return {
        "run_id": run_id,
        "trigger_type": str(TriggerType.MANUAL),
        "is_synthetic": False,
        "release_slot": True,
        "evidence": (
            f"started {minutes:+.0f} min from its slot, outside the "
            f"{SCHEDULED_TOLERANCE} tolerance; executed by hand, so it must not "
            "hold the production slot"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument(
        "--apply", action="store_true", help="write the changes (default: report only)"
    )
    args = parser.parse_args()

    db = connect(args.database_url)
    try:
        runs = db.fetchall("SELECT * FROM monitoring_runs ORDER BY id")
        decisions = classify_all(runs)

        print(f"{'run':>4}  {'was':<10} {'becomes':<16} {'synthetic':<10} {'slot':<9} evidence")
        print("-" * 130)
        for run, decision in zip(runs, decisions, strict=False):
            print(
                f"{decision['run_id']:>4}  "
                f"{str(run.get('trigger_type') or 'unset'):<10} "
                f"{decision['trigger_type']:<16} "
                f"{str(decision['is_synthetic']):<10} "
                f"{'released' if decision['release_slot'] else 'owns':<9} "
                f"{decision['evidence']}"
            )

        if not args.apply:
            print("\nreport only; pass --apply to write")
            return 0

        for decision in decisions:
            fields: dict[str, Any] = {
                "trigger_type": decision["trigger_type"],
                "is_synthetic": decision["is_synthetic"],
            }
            if decision["release_slot"]:
                # `scheduled_for` is kept. It is evidence of what the run was
                # *attributed to*, and erasing it would destroy the record of the
                # mistake being corrected. The slot is released by classification:
                # `uq_production_slot` and the production run lookup both apply only
                # to scheduled/catch_up runs, so a genuine scheduled execution can
                # now claim the same slot without colliding with this row.
                fields["production_schedule_name"] = None
            assignments = ",".join(f"{k}=?" for k in fields)
            db.execute(
                f"UPDATE monitoring_runs SET {assignments} WHERE id=?",  # noqa: S608
                [*fields.values(), decision["run_id"]],
            )
        print(f"\napplied to {len(decisions)} run(s)")
        remaining = db.fetchall(
            "SELECT id, trigger_type, is_synthetic, scheduled_for FROM monitoring_runs ORDER BY id"
        )
        for row in remaining:
            print(
                f"  run {row['id']}: {row['trigger_type']} "
                f"synthetic={bool(row['is_synthetic'])} slot={row['scheduled_for']}"
            )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
