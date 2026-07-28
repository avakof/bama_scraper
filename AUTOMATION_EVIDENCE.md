# Automation evidence — daily run at 13:00 Europe/Paris on macOS

What was executed, what was observed, and what has **not** been observed yet.
Every claim is followed by the command that produced it.

Written 2026-07-28. Host `alis-macbook-air2.home`, macOS 25.5.0, Python 3.14.4,
SQLite 3.53.0.

---

## 1. What is installed

```
LaunchAgent   com.bama.monitor.scheduler
plist         ~/Library/LaunchAgents/com.bama.monitor.scheduler.plist
program       deploy/macos/scheduler_wrapper.rendered.sh
working dir   /Users/vakof/bama_scrape
logs          logs/scheduler.{out,err}.log
database      sqlite:////Users/vakof/bama_scrape/monitor_data/bama_monitor.sqlite
```

```console
$ launchctl print gui/501/com.bama.monitor.scheduler | grep -E "state|runs|exit"
	state = running
	runs = 3
	last exit code = 0

$ plutil -lint ~/Library/LaunchAgents/com.bama.monitor.scheduler.plist
... : OK
```

`RunAtLoad = true`, `KeepAlive = {SuccessfulExit: false}`, `ThrottleInterval = 60`.

The installer refuses to report success it has not observed: it connects to the
database first, runs `plutil -lint`, rejects any leftover `@PLACEHOLDER@`, and after
`bootstrap`/`enable`/`kickstart` waits up to 30 s for the process to appear, exiting
non-zero if it never does. Writing a plist is not installing.

## 2. Why launchd is not the timer

`StartCalendarInterval` is evaluated in the **machine's** timezone and cannot be
pinned to another one. launchd therefore only supervises the process; the schedule
lives in `scheduler_daemon.py` as
`CronTrigger(hour=13, minute=0, timezone=ZoneInfo("Europe/Paris"))`.

Proof the schedule does not depend on the machine clock:

```console
$ python -m bama_monitor.scheduler_daemon --print-next
{"europe_paris": "2026-07-29T13:00:00+02:00", "utc": "2026-07-29T11:00:00+00:00", ...}

$ TZ=Asia/Tehran python -m bama_monitor.scheduler_daemon --print-next
{"europe_paris": "2026-07-29T13:00:00+02:00", "utc": "2026-07-29T11:00:00+00:00", ...}
```

Identical UTC instant under a different system timezone.

The wrapper's guards were exercised rather than assumed:

```console
$ PYTHON=/nonexistent/python ./scheduler_wrapper.rendered.sh
FATAL: python interpreter not found at /nonexistent/python
exit code: 78            # EX_CONFIG
```

## 3. The three tests

### Test A — unit and integration suite

```console
$ ruff format --check src tests deep_scraper tests_monitor tools   # 135 files formatted
$ ruff check    src tests deep_scraper tests_monitor tools         # All checks passed
$ mypy src/bama_monitor src/bama_scraper deep_scraper/bama_deep    # 64 files, clean
$ mypy src/bama_eda                                                # 24 files, clean
$ pytest -q                                                        # 1239 passed
```

Every database test runs on **both** backends, SQLite and PostgreSQL.

### Test B — launchd actually fires a job, unattended

A temporary agent `com.bama.monitor.launchdtest` was installed with
`StartCalendarInterval {Hour: 15, Minute: 0}` local, then left alone. Nothing
kickstarted it.

```console
$ cat .../launchd_test/test.out.log
launchd test job started 2026-07-28T13:00:05Z pid=92092
marker written: {"fired_at_utc": "2026-07-28T13:00:05.862406+00:00", "run_id": 1, ...}
launchd test job finished

$ launchctl print gui/501/com.bama.monitor.launchdtest | grep -E "runs|exit"
	runs = 1
	last exit code = 0
```

Fired at 15:00:05 local — the scheduled minute. The job created a real run row with
**no scheduled slot**:

```
id  trigger_type     is_synthetic  scheduled_for  status  host_name
1   deployment_test  0             NULL           valid   alis-macbook-air2.home
```

The temporary agent was then `bootout`-ed and its plist deleted; `launchctl print`
now reports it does not exist.

### Test C — full production smoke run

`run 5`, launched by hand, `--trigger-type manual --export-snapshot`.

| | |
|---|---|
| trigger | `manual`, `is_synthetic = 0` |
| `scheduled_for` | **NULL** — a manual run takes no slot |
| started → finished | 14:46:38Z → 16:54:16Z (**127.6 min**) |
| discovered | 2,844 |
| detail checks | 2,844 (2,832 completed + 12 gone) |
| retryable failures | **0** |
| coverage | **1.0** |
| termination | `api_exhausted_after_2_empty_pages` |
| comparison | applied transactionally |
| new / missing / removed | 21 / 70 / 44 |
| validity gate | all 8 checks passed, `failed: []` |
| published | `daily_snapshots/2026-07-28/manual-16-46_Europe-Paris/run_5/` |
| pointer | `latest_valid.json` advanced, `in_genuine_history: false` |

The directory is labelled `manual-16-46`, not `13-00`, so a manual run can never be
mistaken for the day's scheduled observation on disk either.

## 4. Every run in the database

```
id  trigger_type     synth  slot                       started   finished  status  disc  acct  cov
1   manual           0      2026-07-28T11:00:00+00:00  08:30:33  08:34:14  valid   2817     0
2   deployment_test  1      2026-07-29T11:00:00+00:00  08:34:36  08:38:18  valid   2817     0
3   deployment_test  1      2026-07-30T11:00:00+00:00  08:39:21  08:43:35  valid   2818     0
4   catch_up         0      2026-07-28T11:00:00+00:00  12:30:42  14:42:28  valid   2849  2829  0.993
5   manual           0      (none)                     14:46:38  16:54:16  valid   2844  2844  1.0
```

Runs 1–3 pre-date the deployment and were **reclassified, not deleted**, from
evidence: runs 2 and 3 observed the same market state 4 and 9 minutes after run 1
while being attributed to different days, which is only possible for test executions.

**Run 4 is the catch-up.** The daemon started at 12:30, found the 13:00 Paris slot
already past and unfilled, and inside the 240-minute grace window created a
`catch_up` run carrying the intended slot as `scheduled_for` and the real execution
time as `started_at`. It is never described as a 13:00 observation.

On restart the daemon declined to duplicate it:

```
scheduler.reconcile action=none existing_run_id=4
  reason='genuine production run 4 already exists for this slot'
```

## 5. Defects found by running it for real

Four, all found because the pipeline was executed rather than reasoned about.

**1. A coded enum parsed as a string.** `detail.pre_sale_delivery` arrives as
`{"value": "SixToNineMonths", "display_name": "تحویل 6 تا 9 ماه"}` on pre-sale
listings, and was handed straight to a text normalizer:

```
TypeError: normalize() argument 2 must be str, not dict
```

20 of run 4's 2,849 checks failed on it. Both halves are now captured — the code is
stable across wording changes, the label is what a reader recognises — and the four
sampled advertisements that failed in run 4 all completed in run 5.

**2. Our own bug blamed on the server.** That `TypeError` was caught by a bare
`except` and recorded as `temporary_server_error`, **retryable**. Both parts were
false: the site answered correctly, and re-fetching a payload we cannot parse will
never succeed. Fetch and parse are now separate, with a `local_parse_error` verdict
that maps to `unknown` availability — a parse failure makes no claim about the page.
`parser_status` likewise now says `not_attempted` for a 404 instead of `parse_failed`.

**3. A declined run would have claimed the slot.** A scheduled trigger that fires
while a previous run holds the lock was recorded as `manual` — laundering the
provenance of a genuine trigger. Fixing that alone would have been worse: with its
real trigger restored it would have occupied `uq_production_slot` and permanently
blocked the day from ever being filled by a run that observed something. Migration
006 excludes skipped runs from the index, and they are excluded from history too.
This is not hypothetical here: a 2,850-advertisement census runs for over two hours.

**4. A stale-code footgun in the migration runner.** `_migration_files()` defaulted
to "every file". The daemon had been running since before the dialect filter was
written, so on reconnect it executed the **PostgreSQL** migration against SQLite.
Harmless by luck — SQLite 3.53 accepts `ALTER COLUMN ... DROP NOT NULL` and the
column was already nullable — but the next PostgreSQL-only statement would not have
been. The dialect argument is now required, and three tests pin the selection.

Two smaller export defects were found by reading the published files:
`filter_exits.csv` was three bytes of byte-order mark with no header (a zero-row day
was indistinguishable from a broken writer), and `scheduled_for` serialised as the
string `"None"` instead of JSON `null`. Both fixed and tested.

## 6. Correction made to the database

`production_schedule.first_genuine_scheduled_run_id` had been set to run 4 by the
scheduler's pre-fix code, which advanced the marker on a run that *finished* rather
than one that *published*. Run 4 is genuine history but failed its validity gate, so
it is not the first day of the mother dataset. The column is now `NULL` and the
reason is recorded in the row's `notes`. **Run 4 itself was not touched.**

## 7. Properties verified by execution, not assertion

```console
# reconstruction is deterministic
rows: 2894 | identical: True | content hash equal: True

# no future content leaks into an earlier day
sqptsgvr: run4 export -> snapshot 20 (run4=20, later=2811)  [OK]
hpo7w9sr: run4 export -> snapshot 39 (run4=39, later=2812)  [OK]
... 5/5 OK

# the gate refuses an incomplete run and publishes nothing
$ python -m bama_monitor.cli snapshot --run-id 4 --validate-only ; echo $?
"valid": false ... "detail": "20 retryable detail failure(s) outstanding"
1
$ ls daily_snapshots/latest_valid.json    # absent at that point
```

## 8. What has NOT been observed

**No unattended `trigger_type='scheduled'` run has happened yet.** The agent is
installed and running, launchd has been observed starting a job on its own, and the
daemon's next trigger is `2026-07-29T11:00:00Z`. Until a row with
`trigger_type='scheduled'` exists, this deployment does not claim one.

**No genuine day is in the mother dataset.** `latest_valid.json` points at run 5,
which is `manual` and marked `in_genuine_history: false`. Run 4 is genuine but failed
its gate. `first_published_run_id` is `null`, and that is the honest answer.

**The systemd units remain unverified** — there is no Linux host here.

**A laptop cannot guarantee an unattended run.** launchd restarts the agent at login
and on crash, but it cannot run a job while the Mac is off, and a sleeping Mac may
not wake for it. Past the 240-minute grace window the slot is written to
`missed_schedule_slots` with `resolution = outside_grace_window` and an alert is
raised. The observation is gone; inventing it would be worse than the gap.

**A ~2,850-advertisement census takes just over two hours** at the configured
politeness. That is the cost of checking every advertisement every day rather than a
subset, and it makes a same-day overlap a realistic event the lock now handles
truthfully.

---

## Checklist

1. **LaunchAgent installed and verified running** — yes; `launchctl print` reports
   `state = running`, `runs = 3`, `last exit code = 0`.
2. **launchd observed firing a job on its own** — yes; Test B fired at 15:00:05
   local, exit 0, marker written. Not kickstarted.
3. **Schedule pinned to Europe/Paris, independent of the machine clock** — yes;
   identical UTC trigger under `TZ=Asia/Tehran`.
4. **DST handled** — yes; slots computed in the zone and converted at the boundary,
   verified across both 2026–27 transitions.
5. **Startup reconciliation and catch-up work** — yes; run 4 was created for the
   missed 13:00 slot, and a later restart correctly declined to duplicate it.
6. **Catch-up is labelled as such, never as a 13:00 observation** — yes;
   `trigger_type='catch_up'`, slot and real start time both recorded.
7. **Missed slots recorded, never fabricated** — yes; `missed_schedule_slots` with
   `resolution`, plus an alert. No backdated history was created.
8. **Run provenance on every run** — yes; `trigger_type`, `is_synthetic`, host, pid,
   scheduler instance, phase timestamps.
9. **Manual runs cannot satisfy or block a scheduled slot** — yes; run 5 has
   `scheduled_for = NULL`, and slot uniqueness is a partial index over genuine
   triggers only.
10. **Fixed scheduled-slot idempotency** — yes; enforced by `uq_production_slot` at
    the database layer as well as in application logic, and proven live.
11. **Every advertisement detail-checked every run** — yes; run 5 checked all 2,844
    for a coverage rate of 1.0. Detail checks are stored separately from content
    snapshots, so "checked daily" is verifiable rather than asserted.
12. **Full-run validity gate blocks publication** — yes; run 4 failed on 20
    unresolved retryable failures and published nothing; run 5 passed all 8 checks.
13. **Daily snapshot folders with atomic rename** — yes; staging directory plus
    `os.replace`, `latest_valid.json` written the same way.
14. **Deterministic reconstruction with no future content** — yes; identical rebuild,
    and each row references the snapshot its own detail check pointed at.
15. **Genuine-history boundary recorded and honest** — yes; `first_genuine_run_id=4`,
    `first_published_run_id=null`, pre-deployment runs listed with true triggers.
16. **First unattended scheduled run** — **not yet observed.** Due
    2026-07-29T11:00:00Z. This report does not claim it.
