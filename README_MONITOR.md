# Bama longitudinal monitoring (`src/bama_monitor/`)

A daily monitoring system for one saved Bama.ir search. It records what the
inventory looked like each day, compares each day against every previous day, and
tracks how long individual listings stay on the market.

The single most important rule in this codebase:

> **A disappearance is an observation. A sale is an inference.**
> `deleted advertisement = disappeared from observed inventory`, never
> `deleted advertisement = definitely sold`.

Those two things live in different columns, are computed by different modules, are
written to different report files, and are never merged. Nothing in this system
produces "confirmed sold" from absence — the `confirmed_sold` label exists in the
enum solely so that an explicit platform statement would have somewhere to go, and
no code path assigns it.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Database schema and migrations](#2-database-schema-and-migrations)
3. [Scheduling and timezone handling](#3-scheduling-and-timezone-handling)
4. [State transitions](#4-state-transitions)
5. [Run-health rules](#5-run-health-rules)
6. [The comparison algorithm](#6-the-comparison-algorithm)
7. [Time to disappearance (interval censoring)](#7-time-to-disappearance-interval-censoring)
8. [Left truncation: what you may and may not measure](#8-left-truncation-what-you-may-and-may-not-measure)
9. [Filter exit: leaving the search is not leaving the market](#9-filter-exit-leaving-the-search-is-not-leaving-the-market)
10. [Sale-evidence methodology (and why it is not a probability)](#10-sale-evidence-methodology-and-why-it-is-not-a-probability)
11. [Repost detection and the vehicle grain](#11-repost-detection-and-the-vehicle-grain)
12. [Deployment](#12-deployment)
13. [Backup, retention and restore](#13-backup-retention-and-restore)
14. [CLI](#14-cli)
15. [Daily report](#15-daily-report)
16. [Testing and verification evidence](#16-testing-and-verification-evidence)
17. [Known limitations](#17-known-limitations)
18. [Daily snapshots and the mother dataset](#18-daily-snapshots-and-the-mother-dataset)

---

## 1. Architecture

```
src/bama_monitor/
├── models.py                domain vocabulary: AdStatus, RunHealth, EventType,
│                            DetailVerdict, SaleLabel + payload dataclasses
├── config.py                MonitorConfig, thresholds, weights, configuration_hash()
├── db.py                    dialect layer: PostgreSQL + SQLite, one SQL dialect
├── repository.py            all table access; no SQL anywhere else
├── state_machine.py         pure transition functions (no I/O)
├── run_health.py            is this run allowed to change any status?
├── inventory_comparison.py  the 6-step daily comparison, one transaction
├── duration_estimation.py   interval-censored durations + survival rows
├── sale_scoring.py          weighted, decomposed sale inference
├── repost_detection.py      multi-field vehicle fingerprint matching
├── detail_verification.py   classify why a missing listing's page did not load
├── scrapers.py              adapters over the existing scrapers (no re-implementation)
├── locking.py               file lock + DB lock row + PG advisory lock
├── alerts.py                sinks: structured log, webhook, email (adapter interface)
├── analytics.py             inventory, cohorts, market stats, survival export
├── reports.py               daily report tree + dashboard
├── runner.py                DailyRunner: orchestrates one run end to end
├── scheduler.py             DST-aware slot computation
├── search_filter.py         is this listing still inside the monitored search?
├── publication.py           published_at provenance + left-truncation verdict
├── vehicle_grain.py         listing -> physical-vehicle aggregation
├── validation.py            manual ground-truth labelling + score calibration
└── cli.py                   command surface
```

**Reuse, not duplication.** Discovery calls `bama_scraper.discovery.discover_via_api`
— the pass whose termination behaviour was verified against the live site. Detail
scraping calls the deep scraper's `parse_ad_api` / `parse_ad_html` / `merge_ad`
pipeline. `scrapers.py` contains adapters and one pure classification function; it
contains no parsing logic of its own.

Three interfaces are explicit, which is what makes the whole workflow testable
without a network:

```python
class DiscoveryScraper(Protocol):
    async def discover(self, search_url: str) -> DiscoveryResult: ...

class DetailScraper(Protocol):
    async def scrape(self, advertisement_url: str) -> DetailResult: ...

class InventoryComparator:
    def compare(self, result, context, health, persist) -> ComparisonResult: ...
```

`StaticDiscoveryScraper` / `StaticDetailScraper` implement the first two for the
simulation and the test suite. Everything else — health gating, transitions,
verification, scoring, reporting — is the production code path.

---

## 2. Database schema and migrations

PostgreSQL for production, SQLite for local development and tests. Both are
first-class: the whole suite runs on both and the multi-day simulation produces
byte-identical status outcomes on each.

Migrations are applied in filename order and recorded in `schema_migrations`:

* `001_monitoring_core.sql` — the 11 core tables;
* `002_analytical_controls.sql` — publication time and left-truncation columns,
  `detail_availability`, filter-exit columns, `vehicle_entities`,
  `sale_validation_samples`.

Each is written once with portable tokens
(`{{PK}}`, `{{TS}}`, `{{JSON}}`, `{{BOOL}}`, `{{NOW}}`, `{{EQ}}`) that `db.render_sql`
resolves per dialect. Query placeholders are always `?` and are rewritten to `%s`
for psycopg.

| Table | Purpose |
|---|---|
| `monitoring_runs` | one row per scheduled run: slot, timezone, health, counters, termination reason |
| `advertisements` | one row per platform ad id: current status, counters, first/last seen, sale inference |
| `daily_ad_observations` | **immutable**: was this ad seen in this run, with its card values |
| `advertisement_snapshots` | detail-page content, written only on change (see below) |
| `advertisement_events` | **immutable** audit log of every transition, with evidence |
| `price_changes` | old/new price, absolute and percentage change, changed_at |
| `detail_verifications` | why a missing ad's page did or did not load |
| `repost_links` | parent ↔ child link, matched fields, score |
| `scrape_errors` | per-stage failures with retryability |
| `alerts` | raised conditions with severity |
| `run_locks` | advisory lock rows with TTL |
| `vehicle_entities` | one row per **physical vehicle**, spanning linked reposts (§11) |
| `sale_validation_samples` | manually labelled outcomes, for calibrating the score (§10) |
| `advertisement_detail_checks` | **proof that each ad was looked at** on each run (§2.1) |
| `production_schedule` | the genuine-history boundary: when automation actually began |
| `missed_schedule_slots` | slots the machine could not run, recorded rather than invented |

Uniqueness constraints carry the correctness guarantees, so idempotency is enforced
by the database and not only by application logic:

| Constraint | Meaning |
|---|---|
| `uq_production_slot` | one **genuine** run per slot per search definition — a partial index over `trigger_type IN ('scheduled','catch_up') AND is_synthetic = false` |
| `uq_detail_check_run_ad (run_id, advertisement_id)` | one detail check per ad per run |
| `uq_ads_platform_id (platform, platform_ad_id)` | one row per platform advertisement |
| `uq_ads_canonical_url (canonical_url)` | URL is a second identity key |
| `uq_obs_run_ad (run_id, advertisement_id)` | one observation per ad per run |
| `uq_snapshot_run_hash (advertisement_id, run_id, content_hash)` | no duplicate snapshots |
| `uq_event_idem (advertisement_id, run_id, event_type)` | replaying a run cannot duplicate events |
| `uq_price_change (advertisement_id, run_id, change_type)` | one price event per ad per run |
| `uq_verify_run_ad (run_id, advertisement_id)` | one verification per ad per run |
| `uq_repost_pair (parent_ad_id, child_ad_id)` | one link per pair |

The production-slot index is *partial* on purpose. Migration 001 declared a blanket
`UNIQUE (scheduled_for, configuration_hash)` over every run, which was correct while
every run was a scheduled run and wrong the moment manual and test runs existed: a run
executed by hand and attributed to today's slot physically prevented that day's genuine
execution from ever being created. That is not a hypothetical — it is the error the
scheduler hit on its first real start — see `migrations/004_drop_legacy_slot_constraint.sql`. Migration 004 removes the blanket
constraint; `uq_production_slot` replaces it, and manual, test, backfill and simulation
runs are free to carry whatever attribution they were given because they can never be
mistaken for history.

**Snapshots are written only when** the listing is first discovered, its content
hash changed, a periodic refresh is due (3–7 days, jittered), the previous attempt
failed, or the ad is being verified while missing. Unchanged listings do not
accumulate identical rows.

### 2.1 A detail check is not a snapshot

These are two different claims and the schema keeps them apart:

| | question it answers | written |
|---|---|---|
| `advertisement_detail_checks` | *did we look at this ad today, and what happened?* | **every ad, every run** |
| `advertisement_snapshots` | *what did the page contain?* | only when the content changed |

Storing one row per ad per day in `detail_checks` costs ~2,850 small rows/day and is
what makes "checked daily" a verifiable statement rather than an assertion. A check
whose content was unchanged references the snapshot that still describes it
(`snapshot_id`), so no content is duplicated and no day is left unaccounted for.

`monitoring_runs.detail_checked_count`, `detail_accounted_count` and
`detail_coverage_rate` summarise the census, and `daily_snapshot.validate_run` refuses
to publish a day whose coverage falls below the configured floor.

### 2.2 Dialect-suffixed migrations

Most migrations are one file with substitution tokens. A few changes have no
portable spelling, so the runner also accepts `NNN_name.<dialect>.sql` and selects
only the files matching the connected backend:

```
005_nullable_scheduled_for.postgres.sql   ALTER COLUMN ... DROP NOT NULL
005_nullable_scheduled_for.sqlite.sql     create, copy, drop, rename
```

The dialect argument is **required**, not defaulted. A default of "every file" is a
footgun a long-running process finds eventually: the scheduler daemon had been up
since before the filter was written, so its in-memory module still walked every file
and executed the PostgreSQL-only migration against SQLite on reconnect. It was
harmless — SQLite 3.53 accepts `ALTER COLUMN ... DROP NOT NULL`, and the column was
already nullable — but the next PostgreSQL-only statement would not have been, and
that path is now closed and tested.

(The SQLite rebuild is kept even though 3.53 no longer needs it: it is the only form
that works on the older SQLite builds this may be restored onto.)

Apply migrations with `python -m bama_monitor.cli migrate`.

---

## 3. Scheduling and timezone handling

The run fires at **13:00 Europe/Paris every day**, which is a fixed *local* time and
therefore a moving UTC instant: **11:00Z in summer (CEST, UTC+2), 12:00Z in winter
(CET, UTC+1)**. Verified across both 2026–27 transitions:

```
2026-10-24  ->  11:00Z   local 13:00+02:00      2027-03-27  ->  12:00Z   local 13:00+01:00
2026-10-25  ->  12:00Z   local 13:00+01:00      2027-03-28  ->  11:00Z   local 13:00+02:00
```

`scheduler.py` computes slots in the configured zone and converts to UTC only at the
boundary, so:

* the spring-forward gap and the autumn-fall-back repeat are both handled
  (`fold=0` is used for the ambiguous repeated hour, so the *first* 13:00 wins);
* a run's identity is its **local slot**, not the UTC instant, so the "one run per
  day" guarantee survives a DST change;
* every timestamp in the database is timezone-aware UTC. PostgreSQL stores
  `TIMESTAMPTZ`; SQLite stores ISO-8601 text and `db.py` converts both ways, so
  callers never see the difference.

Preference order for scheduling, as specified: **systemd service + timer** (primary),
Docker triggered by systemd (secondary), cron (fallback), APScheduler only inside a
long-running process.

### 3.1 What is actually installed here: macOS + launchd

The systemd units are the Linux target. **This machine is macOS**, where they cannot
run, so the deployment that is genuinely installed and firing is the LaunchAgent in
[deploy/macos/](deploy/macos/) — see [README_MACOS.md](deploy/macos/README_MACOS.md).

launchd is deliberately **not** the timer. It can fire on a calendar
(`StartCalendarInterval`), but it reads that calendar in the *machine's local timezone*
and offers no way to pin another one; a Mac set to anything but Europe/Paris would fire
at the wrong moment, and one whose timezone changed would drift silently. So:

```
launchd    keeps the process alive, restarts it on crash, starts it at login
scheduler  owns the schedule — CronTrigger(13:00, tz=ZoneInfo("Europe/Paris"))
```

This is the "APScheduler inside a long-running process" option, chosen because on this
platform it is the only one that can honour the timezone requirement. Proof it is
independent of the machine clock: with `TZ=Asia/Tehran` the daemon still computes the
identical next trigger, `2026-07-29T11:00:00Z`.

`scheduler_daemon.py` also does what a timer cannot — on every start it **reconciles**:

| situation at startup | action |
|---|---|
| today's slot is still ahead | wait for the trigger |
| slot passed, a genuine run already exists | do nothing; never duplicate |
| slot passed, within `catch_up_grace_minutes` (240) | run it as `trigger_type='catch_up'`, keeping the intended slot as `scheduled_for` and the real time as `started_at` |
| slot passed, beyond the grace window | record it in `missed_schedule_slots`, raise an alert, **fabricate nothing** |

A laptop that is asleep or off at 13:00 is a real limitation of this deployment, not a
bug, and the missed-slot row is how it stays visible.

---

## 4. State transitions

`state_machine.py` holds pure functions — no database, no clock, no I/O — so the
rules can be tested exhaustively.

| From | Event | To |
|---|---|---|
| (unknown to us) | seen in a valid run | `new` |
| `new` | seen in a valid run | `active` |
| `active` | seen in a valid run | `active` |
| `reappeared` | seen in a valid run | `active` |
| `active` \| `new` \| `reappeared` | absent for 1 valid run | `missing_once` |
| `missing_once` | absent for another valid run | `likely_removed` |
| `likely_removed` | still absent | `likely_removed` (confidence rises) |
| `missing_once` \| `likely_removed` | seen again | `reappeared` |
| `likely_removed` | strong sale evidence | `likely_sold` |
| `likely_removed` | matching new advertisement found | `reposted` |
| `likely_sold` \| `reposted` | seen again | `reappeared` |
| `missing_once` \| `likely_removed` | live page fails a filter bound | `active_outside_filter` |
| `active_outside_filter` | seen in the search again | `active` (via `filter_reentry`) |

Load-bearing details:

* `REMOVAL_CONFIRMATION_MISSES = 2` (configurable). **One missing observation never
  produces a removal**, and never a sale.
* `likely_sold` is reachable **only from `likely_removed`**, never directly from an
  absence.
* Reappearance always wins. A listing seen again after being called removed, sold or
  reposted returns to `reappeared` and its sale confidence collapses. Observation
  beats inference, always.
* `active_outside_filter` halts the miss counter (§9): the listing is known alive
  and known outside the search, so its continued absence carries no information.
* Every transition writes an `advertisement_events` row carrying the evidence that
  justified it. The event log is append-only and is what a replay reconstructs from.

---

## 5. Run-health rules

> **Never update removal statuses when the daily scrape is incomplete or unhealthy.**

A run is `valid` only if *all* of these hold:

1. the search URL loaded (`initial_page_ok`);
2. termination was **verified** — discovery stopped because it saw the configured
   number of consecutive empty pages, not because it hit a ceiling;
3. the stabilization pass completed;
4. the count is plausible against the recent median (default: not more than 70%
   below) and above an absolute floor;
5. no CAPTCHA/block response was observed;
6. no critical parser failure (e.g. all titles empty — the classic selector-change
   signature);
7. the failed-page ratio is below threshold (default 20%);
8. the inventory was persisted;
9. the implied mass-disappearance is plausible (a run that would mark >50% of the
   known population missing is vetoed).

Statuses: `valid`, `invalid`, `partial`, `blocked`, `failed`, `skipped`,
`skipped_due_to_existing_run`, `running`. The distinction is operational: `blocked`
means back off, `partial` means retry.

**When a run is not valid:**

* the observed inventory is still persisted — evidence is never discarded;
* **no** advertisement's `consecutive_misses` moves;
* no absence rows are written, so an unhealthy run cannot leave a trace that looks
  like a disappearance;
* rows created for genuinely new listings are recorded but **not counted as
  discoveries** — the next valid run counts them, so the daily counters always agree
  with the event log;
* alerts are raised with the failed checks attached;
* the run is retried within bounds.

---

## 6. The comparison algorithm

`InventoryComparator` executes exactly this order:

1. **Persist** the discovered inventory (evidence first — it survives any later
   failure). Deduplicate by platform id, then by canonical URL.
2. **Detect card-level price changes** against the previous *seen* observation.
3. **Validate** run health.
4. If and only if valid, open **one transaction**:
   * apply seen transitions;
   * apply missing transitions;
   * write counters.
5. Commit. A failure anywhere rolls the whole comparison back — a half-applied
   comparison would leave some listings with an incremented miss counter and others
   not, with nothing downstream able to tell which.

The **comparison population** is restricted to advertisements that: belong to the
same normalized filter (`configuration_hash`), hold status `new` / `active` /
`reappeared` / `missing_once`, were last seen before the current valid run, are not
permanently excluded, and are expected to fall inside the search criteria.

**Idempotency.** The run row carries `comparison_applied`. Re-running a slot returns
the stored outcome, rebuilt from the immutable event log, instead of recomputing —
so `consecutive_misses` cannot be incremented twice and a listing cannot be pushed
to `likely_removed` a day early. Verified live: a full re-scrape of an
already-compared slot reported
`"transitions for this scheduled slot were already applied; returning the stored
outcome without re-incrementing counters"` and left every counter untouched.

**Overlap prevention** is three-layered, because any single layer can fail:
`flock` on a lock file, a `run_locks` row with a TTL, and `pg_try_advisory_lock` on
PostgreSQL. A run that finds the lock held records
`status = skipped_due_to_existing_run` and exits without touching anything.

---

## 7. Time to disappearance (interval censoring)

> The metric is named **time to disappearance**, never *time to sale*. What is
> measured is the interval in which a listing left the observed inventory. Why it
> left is a separate, inferred question with its own column — see §10.

Removal is never observed directly. With one scrape per day, all that is known is
that the listing was present at `last_seen_at` and absent at `first_missing_at`.
Reporting a single duration would fabricate up to a full observation interval, so
three quantities are stored:

```
minimum_active = last_seen_at     - first_seen_at
maximum_active = first_missing_at - first_seen_at
estimated      = midpoint(last_seen_at, first_missing_at) - first_seen_at
observation_interval_hours = first_missing_at - last_seen_at
```

**The midpoint is an estimate, never a time of sale.** It is always accompanied by
the bounds and by the interval width that produced it.

Listings still present — and listings that reappeared after an absence — are
**right-censored**: they have a lower bound only, `maximum` and `estimated` are
`None`. (A reappeared listing's old `first_missing_at` is history, not the end of
its life; closing the interval on it would place the upper bound below the lower
bound. The bound-ordering invariant `min ≤ est ≤ max` is asserted by tests across
every status shape.)

Every duration also records the basis it was measured from:

| Column group | Measured from | Available for |
|---|---|---|
| `duration_*`, `observed_monitoring_*` | first observation | every listing |
| `estimated_market_*` | seller's publication time | only fully observed listings (§8) |

`survival_dataset.csv` is Kaplan–Meier-ready and names the distinction explicitly:

> `event_observed = 1` means **the disappearance was observed**, not that a
> confirmed sale occurred.

`event_type` carries `disappearance_observed` / `reposted` /
`still_present_or_returned`, and the sale inference travels in separate
`sale_confidence` / `sale_label` columns.

---

## 8. Left truncation: what you may and may not measure

`first_seen_at` is when **monitoring** first saw a listing. `published_at` is when
the **seller** published it. These are not the same thing, and for every listing
that already existed on the first monitoring day they differ by an unknown amount:

```
monitoring begins   1 August
listing A first seen 1 August
listing A disappears 3 August      -> two days observed
                                      but A may have been listed since June
```

Such a listing is **left-truncated** (delayed entry). Its observed duration is a
*lower bound* on its time on market, so ranking it against a listing whose whole
life was watched systematically flatters the long-standing one.

The system therefore keeps two duration concepts apart and never merges them:

```
observed_monitoring_duration = disappearance interval - first_seen_at
estimated_market_duration    = disappearance interval - published_at
```

`estimated_market_duration` is computed **only** when both hold:

```python
published_at is not None and published_at_reliable
and abs(first_seen_at - published_at) <= entry_tolerance_hours   # default 36h
```

Anything else gets `None`, not a number with a caveat attached — a number with a
caveat is what ends up in a chart without the caveat.

### Publication-time provenance

Not every publication string is precise enough to subtract:

| `published_at_source` | Example | Usable? |
|---|---|---|
| `detail_absolute` | `1405/4/28` | yes |
| `detail_relative` | `۳ روز پیش` | yes |
| `detail_coarse` | `دقایقی پیش` | **no** — pins the day, not the hour |
| `unknown` | nothing parsed | no |

An impossible ordering (publication *after* first sighting) is rejected outright
rather than clamped to zero, so a parser error cannot smuggle a listing into the
rankings.

### Consequences in the outputs

* `fastest_disappearing.csv` contains **only** eligible listings.
* `left_truncated_excluded.csv` contains everything excluded, each row with
  `excluded_from_ranking_because`. An exclusion is published, not silently dropped.
* `survival_dataset.csv` carries `left_truncated` and `entry_delay_days` so a
  model can apply delayed-entry handling instead of assuming there is none.
* `analytics --view truncation_summary` reports the share of the corpus that can
  support a time-on-market claim at all.

For the cleanest "which cars disappear sooner?" analysis, use the eligible subset.
Baseline listings remain in inventory statistics, where they are perfectly valid.

---

## 9. Filter exit: leaving the search is not leaving the market

The monitored search has lower bounds on price (1,000,000,000 Toman) and year
(1397). A seller who cuts the asking price from 1,050,000,000 to 950,000,000
vanishes from the results while remaining entirely for sale.

Left alone, that absence would accumulate misses, reach `likely_removed`, and then
feed a sale inference — on evidence that says the opposite.

So when a missing listing's detail page still loads, its own fields are re-checked
against the bounds parsed from the search URL:

| Outcome | Status | Miss counter |
|---|---|---|
| fails a bound | `active_outside_filter` + `filter_exit` event with the reason | reset, stops |
| passes every bound | unchanged | continues |

Reasons recorded: `price_below_filter`, `price_above_filter`,
`year_outside_filter`, `category_changed`, `country_classification_changed`,
`search_index_inconsistency`.

The last one is the honest non-explanation: the listing matches everything that
could be checked and was still absent, which is an indexing effect rather than a
listing change. It is recorded as an **unexplained absence** and the status is left
exactly as it was — "we cannot explain this" must never be dressed up as
"explained".

`active_outside_filter` is deliberately outside `COMPARABLE_STATUSES` (its
continued absence carries no information) and outside `DISAPPEARED_STATUSES` (the
car is still for sale). In the survival export it is a **censoring** event,
`event_type = left_search_filter`, not a disappearance. If the listing later
matches again — the price goes back up — it re-enters through its own
`filter_reentry` event rather than being mislabelled a reappearance from absence.

Skipped checks are recorded. A listing whose price could not be read has
`skipped: ["price_from"]` in its evidence, so an unverifiable listing never looks
verified. And `price = 0`, which is how Bama encodes "negotiable", is treated as
*no price* rather than as a price below the minimum.

### Three vocabularies, kept disjoint

```
current_status        what the SEARCH showed      (observation)
detail_availability   what the PAGE said          (observation)
sale_label            what we INFER               (inference)
```

A 410 sets `detail_availability = gone`. It does not set a status and it does not
set a label. Observed live: a listing went missing, its page returned 410, and it
still only reached `missing_once` at score 0.35 — because one absence is one
absence, whatever the page says.

---

## 10. Sale-evidence methodology (and why it is not a probability)

`sale_scoring.py` computes `sale_confidence ∈ [0.0, 1.0]` from additive weighted
rules and **retains every component**, so any score can be explained.

| Evidence | Weight |
|---|---|
| absent from two consecutive valid runs | +0.30 |
| detail page explicitly reports unavailable | +0.15 |
| detail page persistently returns 404/410 | +0.15 |
| a price reduction shortly before disappearance | +0.10 |
| disappeared soon after publication | +0.10 |
| no repost detected from the same seller | +0.10 |
| absence persists 7+ days | +0.10 |
| **reappeared in the inventory** | **−0.30** |
| a repost of the same vehicle was found | −0.25 |
| detail page remains fully accessible | −0.20 |
| listing appears outside the original filter | −0.20 |

Labels (inclusive lower bounds): `unknown` < 0.40, `possibly_sold` ≥ 0.40,
`likely_sold` ≥ 0.65, `highly_likely_sold` ≥ 0.85. `confirmed_sold` is
**never** assigned by any code path.

Two consequences worth stating because they are deliberate:

* Absence alone (two misses + no repost) scores **0.40** — `possibly_sold`, which is
  below `likely_sold`. Disappearing is not, by itself, sufficient evidence of a sale.
* A still-accessible detail page *lowers* confidence. Observed live: a listing left
  the search results, its page returned HTTP 200, its confidence stayed **0.00**, and
  the next run showed it back in the inventory.

`sale_scoring.METHODOLOGY` carries this text in the codebase, and each score's
components are stored so a report can show the arithmetic.

### It is a score, not a probability

The exported column is named **`sale_evidence_score`**, never `sale_probability`
and never `sale_confidence`:

> A score of 0.65 means "more evidence than 0.40". It does **not** mean "65% of
> these sold". The weights were chosen by reasoning about evidence, not fitted to
> observed outcomes, so the number is an **ordering**, not a rate.

Turning it into a rate requires ground truth, and there is no automatic oracle for
"did this car sell". So the path is explicit and manual:

```bash
python -m bama_monitor.cli validation-sample --per-band 25     # stratified worksheet
#   ... a human fills in `observed_outcome` for each row ...
python -m bama_monitor.cli validation-ingest --worksheet reports/validation_worksheet.csv
python -m bama_monitor.cli calibration                         # measured rate per band
```

Permitted outcomes: `sold`, `not_sold`, `withdrawn`, `expired`, `reposted`,
`outside_filter`, `unknown`. `unknown` is a first-class answer — guessing would
poison the very measurement this exists to make.

`calibration` reports `sale_rate` per band **only** where at least 10 decided
labels exist, and sets `calibrated: false` until the sample is large enough. Until
that flips, no report in this system describes a score as a percentage chance of
sale, and neither should you.

---

## 11. Repost detection and the vehicle grain

An advertisement id identifies a *listing*, not a *vehicle*. Sellers relist, so
counting a repost as a sale would inflate every sale metric.

Matching uses a **multi-field vehicle fingerprint** — never a single field:
brand, model, trim, year, mileage (within tolerance and monotonically non-decreasing),
price (within tolerance), city, seller identity, body and interior colour,
description similarity, and image similarity via CDN-path hashing with a Hamming
distance threshold. Field weights, tolerances and the acceptance threshold are all
configurable.

On a match the system links `repost_parent_ad_id`, records a `possible_repost`
event, marks the original `reposted`, **reduces** sale confidence, and assigns a
shared `vehicle_entity_id` so the physical vehicle can be followed across listings.

> **A reposted advertisement is not counted as a sale.**

The false-positive guard is explicit and tested: two different cars of the same
model, year and city must not match on taxonomy alone.

### Two grains, because they answer different questions

| Grain | Key | A relisted car looks like |
|---|---|---|
| listing | `advertisement_id` | one disappearance + one new arrival |
| **vehicle** | `vehicle_entity_id` | one car still looking for a buyer |

Only the second supports "which cars sell faster?". `vehicle_grain.py` therefore
aggregates linked listings:

```
vehicle_first_seen_at = min(first_seen_at) across linked listings
vehicle_last_seen_at  = max(last_seen_at)  across linked listings
```

A repost is an **explanatory variable**, not an endpoint: the chain ends only when
its last listing disappears without a successor. Two listings of 3.5 and 5.5 days
become one vehicle life of 9–10 days, which is what actually happened.

Two deliberate details:

* the vehicle's status comes from its **latest** listing — an earlier listing
  marked `reposted` describes a chain link, not the car's fate;
* a chain whose last link is `reposted` with no recorded successor reports
  `unknown`, not a disappearance. The car is listed *somewhere*; we just have not
  found where.

`vehicle_durations.csv` carries `listing_count`, `repost_count` and `was_reposted`
so reposting can be used as a covariate rather than silently inflating sale counts.

---

## 12. Deployment

### macOS + launchd (what is installed and running here)

```
deploy/macos/com.bama.monitor.scheduler.plist   RunAtLoad, KeepAlive, template
deploy/macos/install_launchd.sh                 renders, installs, and VERIFIES
deploy/macos/uninstall_launchd.sh               bootout + remove
deploy/macos/status_launchd.sh                  plist, launchctl print, process, next slot
deploy/macos/run_once.sh                        one manual run, no slot, no schedule impact
deploy/macos/scheduler_wrapper.sh               absolute paths, env file, exec
```

```bash
./deploy/macos/install_launchd.sh
launchctl print gui/$(id -u)/com.bama.monitor.scheduler | grep state
```

The installer refuses to claim success it has not observed: it connects to the database
before installing, runs `plutil -lint`, rejects any unrendered `@PLACEHOLDER@`, then
after `bootstrap`/`enable`/`kickstart` waits up to 30 s for the process to appear and
**exits non-zero if it never does**. Writing a plist is not installing.

Details, catch-up semantics and the sleep/shutdown limitations:
[deploy/macos/README_MACOS.md](deploy/macos/README_MACOS.md).

### Run provenance: history vs everything else

Every run records how it was triggered, and only two kinds count as daily history:

| `trigger_type` | genuine history? | meaning |
|---|---|---|
| `scheduled` | **yes** | the daemon fired at the 13:00 Paris slot |
| `catch_up` | **yes** | the slot was missed and recovered inside the grace window |
| `manual` | no | a human ran it; `scheduled_for` is `NULL` |
| `deployment_test` | no | proving the plumbing works |
| `backfill` / `simulation` | no | reconstructed or synthetic |

A third case sits between them. A scheduled trigger that fires while a previous run
still holds the execution lock records `status = skipped_due_to_existing_run` with its
**real** trigger type — it genuinely was the schedule firing, not a human. But it
observed nothing, so it neither fulfils its slot nor appears in history, and the slot
stays available for a catch-up. That matters here: a ~2,850-advertisement census runs
for about two hours, so an overlap is a realistic event rather than a hypothetical.

`is_synthetic` is a separate axis: a run may be `scheduled` in shape yet synthetic in
origin, and only `is_synthetic = false` production triggers enter history. Analytical
exports and the EDA read genuine history only; the rest is real work, kept, and
deliberately excluded. The boundary itself is stored in `production_schedule`
(`production_schedule_started_at`, `first_genuine_scheduled_run_id`) so no reader has to
infer where automation began.

### systemd (primary, Linux)

```
deploy/bama-monitor.service     Type=oneshot, hardened, EnvironmentFile for secrets
deploy/bama-monitor.timer       OnCalendar=*-*-* 13:00:00, Timezone=Europe/Paris,
                                Persistent=true, RandomizedDelaySec=120
deploy/install_systemd.sh       installs units, enables the timer
```

The timer carries `Timezone=Europe/Paris`, so systemd itself performs the DST
arithmetic. `Persistent=true` means a run missed while the machine was off fires on
the next boot rather than being silently skipped. `RandomizedDelaySec=120` avoids a
thundering herd against the site at exactly 13:00:00.

```bash
sudo ./deploy/install_systemd.sh
systemctl list-timers bama-monitor.timer
journalctl -u bama-monitor.service -f
```

### Docker (secondary)

`deploy/Dockerfile`, `deploy/docker-compose.yml` (app + PostgreSQL), and
`deploy/bama-monitor-docker.service` for a systemd-triggered container run.

### Secrets

**No credentials are hard-coded.** The database URL comes from
`BAMA_MONITOR_DATABASE_URL`; alert webhook and SMTP settings come from the
environment via `EnvironmentFile=`. `deploy/config.example.yaml` documents every
setting and contains no secrets.

---

## 13. Backup, retention and restore

The monitoring history is **not reproducible**. An observation missed today can
never be recovered, because yesterday's inventory no longer exists anywhere. That
makes the backup part of the data model, not an operational nicety.

```bash
deploy/backup.sh dump                  # compressed custom-format dump + prune
deploy/backup.sh verify <file>         # readable, and contains the core tables
deploy/backup.sh restore <file> <db>   # restore into a NEW database
deploy/backup.sh drill                 # dump -> restore -> compare row counts
```

`drill` is the only one that proves anything: a dump nobody has restored is a
hypothesis. It dumps, restores into a scratch database, compares row counts across
`monitoring_runs`, `advertisements`, `daily_ad_observations` and
`advertisement_events`, then drops the scratch database.

**Executed against the real PostgreSQL 16.13 cluster:**

```
wrote bama-monitor-20260728T092130Z.dump (56K)
OK: dump is readable and contains all 4 core tables
restored into bama_restore_drill_76039
dropped scratch database
DRILL PASSED: row counts identical after restore
monitoring_runs=4 advertisements=7 daily_ad_observations=23 advertisement_events=38
```

Retention: every dump from the last `BAMA_RETAIN_DAILY` days (default 14) plus
weekly dumps for `BAMA_RETAIN_WEEKLY` weeks (default 8). Pruning is by mtime, so a
clock jump cannot delete everything at once.

`restore` deliberately refuses to write into an existing database — recovery goes
into a fresh one, so a mistaken restore cannot destroy the history it was meant to
protect. Promoting it is a separate, conscious step.

Scheduled by `deploy/bama-monitor-backup.timer` at **14:30 Europe/Paris** — after
the 13:00 run and its report, not before: a backup at 12:55 would omit the run
about to happen, leaving the freshest observation the least protected. The service
runs a full restore drill on Sundays.

---

## 14. CLI

```
python -m bama_monitor.cli [--config FILE] <command>

run-daily          full pipeline: lock, discover, persist, health-gate, compare,
                   detail-scrape, verify, repost-detect, score, finalize, report
discover-only      discovery + persistence, no comparison
compare            report the stored comparison for a run id (never re-applies)
verify-missing     verify detail pages of advertisements missing in a run
refresh-details    deep-scrape advertisements due a refresh
detect-reposts     repost detection over recent removals
report             generate the daily report tree
backfill           create runs for a date range (simulation / catch-up)
validate           check configuration, schedule, database and scraper wiring
migrate            apply database migrations
analytics          print an analytics view as JSON
status             summarise current inventory state
validation-sample  draw a stratified sample of disappearances to label by hand
validation-ingest  load a completed labelling worksheet
calibration        report the measured sale rate per evidence-score band
```

`analytics --view` accepts: `daily_inventory`, `time_to_disappearance`,
`fastest_disappearing`, `left_truncated_excluded`, `vehicle_durations`,
`filter_exits`, `truncation_summary`, `market`, `cohorts`, `survival`.

Common flags: `--database-url`, `--search-url`, `--timezone`, `--run-at`,
`--removal-confirmation-misses`, `--output-dir`, `--reports-dir`, `--scheduled-for`,
`--log-level`, `--dry-run`. Note `--config` is a *top-level* flag and must precede
the subcommand.

---

## 15. Daily report

`reports/YYYY-MM-DD/` contains:

```
run_summary.json          status, counters, health checks, interpretation, disclaimer
daily_inventory.csv       one row per day: totals and per-category counts
new_ads.csv               first-discovery transitions
missing_ads.csv           first-absence transitions
likely_removed_ads.csv    removal-confirmed transitions
reappeared_ads.csv        returns from absence
possible_reposts.csv      candidate relistings with matched fields
likely_sold_ads.csv       sale *inferences* with confidence
price_changes.csv         old/new/absolute/percentage
scrape_errors.csv         per-stage failures
time_to_disappearance.csv interval-censored durations, both bases side by side
fastest_disappearing.csv  ranked; evidence >= 0.65 AND fully observed only
left_truncated_excluded.csv  what the ranking excluded, and why
vehicle_durations.csv     vehicle grain: durations spanning linked reposts
filter_exits.csv          still for sale, outside the search - the false-removal ledger
survival_dataset.csv      Kaplan-Meier-ready export, with left_truncated
cohorts.csv               1/3/7/14/30-day cohort survival
market_by_{brand,model,year,city,seller_type,price_range,mileage_range}.csv
dashboard.html            single-page summary
manifest.json             row counts per file
```

Every CSV keeps a **stable header even when it has no rows**, so a consumer reading
`missing_ads.csv` daily never loses its columns on a quiet day.

**Every row is prefixed with the run identifiers** — `run_id`, `scheduled_for`,
`run_started_at`, `run_finished_at`, `search_configuration_hash`,
`scraper_version`, `monitor_version` — so a count can never be quoted without the
run that produced it. This matters because the inventory of a live marketplace
changes between runs:

> Not "the full pass found 2,816 advertisements", but
> **"monitoring run 1, executed 2026-07-28T11:00Z to 11:04Z under configuration
> hash `a3f1…`, observed 2,816 unique advertisements."**

Otherwise a reader interprets normal churn (2,816 one day, 2,817 the next) as a
scraper inconsistency. `run_summary.json` carries the same identifiers under
`provenance`, plus `measurement_notes` spelling out the four cautions above.

The dashboard keeps four categories visually and textually distinct, and never
combines them:

```
observed disappearance   the listing was absent from a valid run   (FACT)
likely removal           absent for the confirmation threshold      (inferred)
likely sale              a confidence score with its components     (inferred)
confirmed sale           never produced by this system              (0, always)
```

---

## 16. Testing and verification evidence

**335 tests**, executed on **SQLite and PostgreSQL 16.13**:

| File | Tests | Covers |
|---|---|---|
| `test_state_transitions.py` | 35 | every transition, terminal states, reappearance precedence |
| `test_run_health.py` | 57 | each unhealthy shape leaves counters untouched; recovery; live-adapter fidelity |
| `test_durations_and_scoring.py` | 42 | interval censoring, bound ordering, DST, scoring bands |
| `test_reposts.py` | 31 | fingerprint matching, false-positive prevention, evidence vocabulary |
| `test_simulation.py` | 56 | multi-day scenarios, idempotency, failure recovery |
| `test_locking_and_reports.py` | 52 | three-layer locking, overlap, report schema, dialect portability |
| `test_analytical_controls.py` | 62 | filter exit, left truncation, vehicle grain, calibration, provenance |

Gates: `ruff format --check`, `ruff check`, `mypy` — all clean.

### Multi-day simulation

`tools/simulate_days.py` runs the **real** `DailyRunner` with scripted scraper
adapters. `G` is a listing whose asking price drops under the monitored minimum on
day 3 — alive, but out of the search.

```
Day 1  seen=ABCDG   valid  applied=True  new=5 missing=0 removed=0 reappeared=0 filter_exits=0
Day 2  seen=ABDEG   valid  applied=True  new=1 missing=1 removed=0 reappeared=0 filter_exits=0
Day 3  seen=ADEF    valid  applied=True  new=1 missing=3 removed=1 reappeared=0 filter_exits=1
Day 4  seen=ABDEF   valid  applied=True  new=0 missing=0 removed=0 reappeared=1 filter_exits=0

A active   B reappeared   C likely_removed   D active   E active   F active
G active_outside_filter        <- absent from search, live page, price 900M < 1,000M bound
B is never sold or removed: True                                       RESULT: PASS
```

G is the important row: it was genuinely absent (an observed fact, recorded), its
live page explained why, and it never accumulated a second miss.

Variants, all passing identically on both backends:

* **invalid day 3** — day 3 is `partial`, `applied=False`, and **no** listing gains
  a miss from it; B stays `active` throughout; the row created on day 3 is counted
  as a discovery by the next valid run.
* **strong evidence** — C's page returns 410, demonstrating
  `likely_removed + strong evidence -> likely_sold` at confidence 0.65.

### Live verification

Executed against the real search URL (permitted by `robots.txt`, which disallows
only a campaign-banner path):

| Run | Result |
|---|---|
| bounded (2 pages) | 60 ads, `max_pages_reached` → **`partial`**, comparison **refused**, 60 rows persisted, 0 counted as discoveries |
| full pass, slot 1 | **2,816 ads**, `api_exhausted_after_2_empty_pages`, health **`valid`**, all checks passed, 3m45s |
| replay of slot 1 | full re-scrape of an already-compared slot: stored outcome returned, **no counter double-counted** |
| slot 2 | 2,817 ads, 2 new, **1 missing** → live verification: HTTP 200, `detail_page_still_active`, confidence **0.00**, status `missing_once` |
| slot 3 | 2,818 ads, **1 reappeared** (the ad from slot 2 returned — never called sold), 1 newly missing whose page returned **HTTP 410**, which still only produced `missing_once` at 0.35 because one absence is one absence |

---

## 17. Known limitations

**The systemd deployment is still unverified; the macOS one is not.** Keep the two
apart:

```
scheduler logic verified                      (DST across both 2026-27 transitions)
macOS launchd deployment INSTALLED + FIRING   (agent running, job observed to start)
systemd units prepared, NOT verified          (no Linux host available)
```

This machine is macOS, so `systemctl` could never be run and the **timer** has not
been observed firing. What *has* been observed is the LaunchAgent equivalent (§12,
[README_MACOS.md](deploy/macos/README_MACOS.md)): the agent is bootstrapped and
`launchctl print` reports `state = running`, launchd started the job on its own, and
the daemon's startup reconciliation created a genuine `catch_up` run for the missed
13:00 slot. The first *unattended* `trigger_type='scheduled'` run happens at the next
13:00 Europe/Paris, and until that row exists this README does not claim it.

Converting the systemd line requires a Linux host and one real scheduled run:

```bash
systemd-analyze verify deploy/bama-monitor.service deploy/bama-monitor.timer
sudo ./deploy/install_systemd.sh          # runs the verify + a restore drill
systemctl list-timers bama-monitor.timer
journalctl -u bama-monitor.service -f
```

and then confirming, on that run: it started at 13:00 Europe/Paris; the
`EnvironmentFile` was loaded; the working directory was right; PostgreSQL was
reachable; file permissions allowed the report tree to be written; the execution
lock was acquired **and released**; a deliberately failed run raised its alert; the
report landed on the expected persistent path; and the timer survived a logout and
a reboot. Until every one of those is observed, treat the deployment as prepared
rather than working.

**Sale-evidence scores are uncalibrated.** No labelled sample has been collected
yet, so `calibration` reports `calibrated: false` and every band's `sale_rate` is
`None`. The score orders listings by weight of evidence; it does not yet estimate a
rate. See §10 for the procedure that would change this.

**Left truncation limits the corpus that can answer "how long do cars take to
sell".** Every listing already active on the first monitoring day is excluded from
duration rankings, and on a fresh install that is *all* of them. The usable subset
grows only as newly published listings are observed from their publication onward;
`analytics --view truncation_summary` reports the share at any moment. Plan for the
first few weeks of monitoring to produce inventory statistics but no credible
time-on-market ranking.

**Filter-exit detection is only as good as the fields it can read.** A listing whose
price cannot be parsed has that check recorded as *skipped*, not passed, and its
absence stays unexplained rather than being attributed to the filter. That is the
safe direction, but it means some genuine filter exits will be recorded as
`search_index_inconsistency` instead.

**Publication times are frequently too coarse to use.** `دقایقی پیش` pins the day
but not the hour, so it is classified `detail_coarse` and never used for a market
duration. On the observed corpus this affects a substantial share of freshly posted
listings — exactly the ones most useful for duration analysis.

**Every advertisement is now detail-checked every run**, so the earlier cap of 6
pages per run no longer applies. The cost is real: a ~2,850-ad census at the
configured politeness runs for roughly two hours, and `detail_coverage_rate` is a
blocking validity check rather than a statistic. A run that cannot reach the coverage
floor is `partial` and does not advance the mother dataset.

**One observation per day bounds the resolution.** Every duration carries at least
24 hours of uncertainty, which is why the bounds are stored rather than a point
estimate. A listing that appears and disappears within a single day is never
observed at all.

**Disappearance is not sale ground truth.** Bama publishes no sale status. A listing can vanish because it sold, expired, was withdrawn, was moderated,
was edited outside the filter, was reposted, moved category, or because a scrape
failed. The scoring model weighs these; it cannot resolve them.

**Repost detection depends on the seller reusing recognisable attributes.** A
relisting with changed photos, a rounded price and a rewritten description may score
below the threshold and be missed — which biases sale counts *upward*. The reverse
error, two genuinely different cars matching, is guarded by requiring agreement
across several fields.

**The `total_count` field of Bama's search API is not trustworthy** — it kept growing
(2,816 → 2,820 → 2,850) while pages were already returning zero advertisements.
Termination is therefore based on consecutive empty pages, and the field is retained
only as evidence.

**Renaming an `EventType` value would need a data migration** if rows already exist.
`detail_restored` was renamed to `detail_still_active` (the page was never down, so
"restored" asserted something that had not happened) while the system had no
production data. Any future rename must ship an `UPDATE` alongside it.

**Historical observations are append-only.** Nothing overwrites a
`daily_ad_observations` or `advertisement_events` row. Correcting a mistaken
inference means writing a new event, never editing an old one.

---

## 18. Daily snapshots and the mother dataset

A run writes to the database; `daily_snapshot.py` is what turns a *valid* run into a
published day. The two steps are separate so that a bad day can never overwrite a good
one.

```
daily_snapshots/
├── 2026-07-28/
│   └── 13-00_Europe-Paris/
│       └── run_4/
│           ├── advertisements.csv        one row per ad checked that day
│           ├── detail_checks.csv         the census: outcome per ad
│           ├── observations.csv          card-level values as seen
│           ├── events.csv                transitions applied by this run
│           ├── run_metadata.json         provenance, counters, validity verdict
│           └── manifest.json             row counts + sha256 per file
└── latest_valid.json                     pointer to the newest day that PASSED
```

### Validity is a gate, not a label

`validate_run` runs blocking checks before anything is exported: the run reached a
terminal status, discovery terminated verifiably, detail coverage cleared the floor,
the run holds no unreleased lock, and its health did not forbid transitions. A run that
fails any of them is exported **only** under its own `run_<id>/` directory with the
failure recorded in `run_metadata.json`, and `latest_valid.json` is not moved. Partial
days are visible; they are never promoted.

### Reconstruction reads through the run's own check

`build_daily_dataset` joins each ad to the snapshot *its own detail check referenced*,
not to the newest snapshot in the table. That is the difference between "what the page
said on 28 July" and "what the page says now", and it is why a day can be rebuilt
identically months later.

### Writes are atomic

The export builds into a staging directory and finishes with a single `os.replace()`.
An interrupted export leaves no half-written day, and `latest_valid.json` is replaced
the same way — a reader either sees the old pointer or the new one, never a truncated
file.

### The two clocks

`scheduled_for` is the slot the day *belongs to*; `started_at`/`finished_at` are when
the machine actually did the work. Observations are attributed to the slot, which keeps
a catch-up run at 12:30 filed under the 13:00 day it recovers — while `run_metadata.json`
still shows the real execution time. Neither clock is allowed to overwrite the other,
and no advertisement is stamped with an artificial 13:00 timestamp.
