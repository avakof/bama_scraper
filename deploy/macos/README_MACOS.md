# macOS deployment — daily run at 13:00 Europe/Paris

The Linux `systemd` units in `deploy/` cannot run here. This directory is the
macOS equivalent: a **LaunchAgent** that keeps a long-running Python scheduler
alive, and a scheduler that owns the timing.

---

## Why launchd is not the timer

`launchd` can fire on a calendar (`StartCalendarInterval`), but it interprets that
calendar in the **machine's local timezone** and offers no way to pin another one.
A Mac set to anything except Europe/Paris would fire at the wrong moment, and a
Mac whose timezone changed — travel, a corporate profile — would drift silently.

So the split is:

```
launchd    keeps the process alive, restarts it, starts it at login
scheduler  owns the schedule, pinned to Europe/Paris with zoneinfo
```

The daemon builds `CronTrigger(hour=13, minute=0, timezone=ZoneInfo("Europe/Paris"))`.
Verified: with `TZ=Asia/Tehran` the next trigger is still `2026-07-29T11:00:00Z`.

---

## Install

```bash
./deploy/macos/install_launchd.sh
```

The installer resolves every absolute path itself, then **verifies rather than
assumes**:

1. resolves the repository and the virtualenv interpreter (fails if absent);
2. creates `deploy/macos/monitor.env` (mode 600) if missing;
3. connects to the database and reports the dialect and run count — aborts if it
   cannot;
4. creates and write-tests the log directory;
5. renders the plist and the shell wrappers with absolute paths;
6. runs `plutil -lint` and refuses any leftover `@PLACEHOLDER@`;
7. `launchctl bootout` any previous version;
8. `launchctl bootstrap gui/$(id -u)` → `enable` → `kickstart -k`;
9. waits up to 30 s for the process and **fails if it never appears**;
10. prints `launchctl print` and the next trigger in three clocks.

Writing a plist is not installing. If step 9 fails the installer exits non-zero
and prints the stderr log.

## Status

```bash
./deploy/macos/status_launchd.rendered.sh
```

Shows the plist, `launchctl print`, the live process, the next trigger, and the
log tail.

## Uninstall

```bash
./deploy/macos/uninstall_launchd.rendered.sh
```

Removes the agent. The database, `daily_snapshots/` and `logs/` are untouched.

## Logs

```bash
tail -f logs/scheduler.out.log     # scheduler + pipeline
tail -f logs/scheduler.err.log     # tracebacks
```

## Run manually, without touching the schedule

```bash
./deploy/macos/run_once.rendered.sh
# or
python -m bama_monitor.cli run-daily --trigger-type manual --export-snapshot
```

`--trigger-type manual` gives the run **no scheduled slot**. It cannot satisfy or
block the day's scheduled execution, and it never appears in genuine history.

---

## Catch-up

At startup the daemon computes today's 13:00 Paris slot and asks whether a
*genuine production* run exists for it. A manual run attributed to that slot does
not count — that is deliberate, and it is the defect this deployment fixes.

| situation | action |
|---|---|
| before 13:00 | wait for the trigger |
| after 13:00, within `catch_up_grace_minutes` (default 240) | start a `catch_up` run |
| beyond the grace window | record a missed slot, alert, **fabricate nothing** |

A catch-up run carries:

```
scheduled_for = the intended 13:00 Paris slot
started_at    = the real execution time
trigger_type  = catch_up
```

It is never described as an exact 13:00 observation.

---

## Sleep, shutdown and their limits

`launchd` restarts the agent at login and if the process dies. It cannot run a job
while the Mac is **off**, and a sleeping Mac may not wake for it.

* Asleep at 13:00 → the daemon reconciles when the machine wakes, and catches up
  if still inside the grace window.
* Off past the grace window → the slot is recorded in `missed_schedule_slots` with
  `resolution = outside_grace_window`, and an alert is raised. The observation is
  gone; inventing it would be worse than the gap.
* For unattended operation, disable App Nap for the process or keep the Mac awake
  around 13:00 (`caffeinate`, or Energy Saver → "Prevent automatic sleeping").

This is a genuine limitation of a laptop deployment, not a bug.

---

## Where things live

| what | where |
|---|---|
| database | `monitor_data/bama_monitor.sqlite`, or `$BAMA_MONITOR_DATABASE_URL` |
| daily snapshots | `daily_snapshots/YYYY-MM-DD/13-00_Europe-Paris/run_<id>/` |
| latest valid pointer | `daily_snapshots/latest_valid.json` |
| logs | `logs/scheduler.{out,err}.log` |
| secrets | `deploy/macos/monitor.env` (mode 600, never in git) |
| plist | `~/Library/LaunchAgents/com.bama.monitor.scheduler.plist` |

---

## Genuine history vs tests

Only `trigger_type IN ('scheduled', 'catch_up')` with `is_synthetic = 0` counts as
daily history — and not if the run was declined for lock contention, which records that
the trigger fired without observing anything. Everything else — manual runs, deployment tests, backfills,
simulations — is real work that is deliberately excluded from history and from EDA.

```bash
sqlite3 monitor_data/bama_monitor.sqlite \
  "SELECT id, trigger_type, is_synthetic, scheduled_for, started_at FROM monitoring_runs;"
```

The boundary is recorded in the `production_schedule` table
(`production_schedule_started_at`, `first_genuine_scheduled_run_id`).

## Verify tomorrow's run

```bash
# what should happen, and when
python -m bama_monitor.scheduler_daemon --print-next

# after 13:00 Paris, what did happen
sqlite3 monitor_data/bama_monitor.sqlite \
  "SELECT id, trigger_type, status, scheduled_for, started_at, discovered_count,
          detail_accounted_count, detail_coverage_rate
   FROM monitoring_runs WHERE trigger_type='scheduled' ORDER BY id DESC LIMIT 1;"

cat daily_snapshots/latest_valid.json
```

A run only advances `latest_valid.json` after passing every blocking validity
check, so that file is the honest answer to "is the mother dataset current".

## Restore after failure

```bash
# 1. what failed
tail -n 60 logs/scheduler.err.log
sqlite3 monitor_data/bama_monitor.sqlite \
  "SELECT id, status, health_reason FROM monitoring_runs ORDER BY id DESC LIMIT 3;"

# 2. clear a stale lock if the process was killed
sqlite3 monitor_data/bama_monitor.sqlite "SELECT * FROM run_locks;"
ls -la monitor_data/locks/

# 3. restart the agent
launchctl kickstart -k gui/$(id -u)/com.bama.monitor.scheduler

# 4. re-run a specific slot by hand, as a catch-up
python -m bama_monitor.cli run-daily \
  --trigger-type catch_up --scheduled-for 2026-07-29T11:00:00+00:00 --export-snapshot
```

A failed run keeps its partial evidence, keeps an accurate status, applies no
missing/removal transitions, and does **not** advance `latest_valid.json`.

---

## Failure modes handled

| failure | behaviour |
|---|---|
| network timeout | bounded retry; the run is marked `partial` or `failed`, no transitions applied |
| HTTP 403/429 | treated as a block: discovery stops gracefully, alert raised, nothing bypassed |
| parser change | health gate rejects (all-empty titles); no missing/removal transitions |
| database unavailable | installer refuses to install; at runtime the run fails and the lock is released |
| machine asleep | catch-up on wake inside the grace window, otherwise a recorded miss |
| scheduler restart | startup reconciliation decides; an existing genuine run is not duplicated |
| process crash | `KeepAlive` restarts it; the run lock has a TTL and is recovered |
| disk full | the export raises, the staging directory is removed, the pointer is unchanged |
| lock contention | the second run records `skipped_due_to_existing_run` with its real trigger type, does **not** claim the slot, and exits; reconciliation can still catch that slot up |
| invalid export | the staging directory is discarded; no partial day is published |
| missing venv | the wrapper exits 78 (`EX_CONFIG`) with the interpreter path it looked for |
