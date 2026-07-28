# Bama EDA (`src/bama_eda/`)

Reproducible exploratory analysis over the Bama monitoring database.

It is a **separate package** from the scraper and the monitor, and it is
**read-only**: `ReadOnlyDatabase` has no write path at all, so no analytical
convenience can change what gets collected.

---

## The one thing to read before quoting a number

> A disappearance from the observed inventory is an **observed fact**.
> A sale is an **inference**.

This analysis measures **time to disappearance**. It never reports a confirmed
sale, never converts the sale-evidence score into a probability, and never counts
a filter exit or a repost as a disappearance.

### Terminology

| Use | Never |
|---|---|
| `time_to_disappearance` | `time_to_sale` |
| `sale_evidence_score` | `sale_probability` |
| `observed_monitoring_duration` (from first observation) | "time on market" |
| `estimated_market_duration` (from publication, eligible rows only) | — |
| `first_seen_at` (when *we* first saw it) | "publication date" |
| `vehicle_entity_id` (a physical car) | advertisement id as a vehicle |
| midpoint of the censoring interval, labelled an estimate | "sale date" |

`models.FORBIDDEN_PHRASES` lists the banned phrasings and a test asserts no module
uses them as identifiers; `report_builder.check_forbidden_phrases` re-checks every
generated artefact.

---

## Quick start

```bash
pip install -e ".[eda]"          # matplotlib + scipy, on top of the base install

python -m bama_eda run \
  --database-url "sqlite:///monitor_data/bama_monitor.sqlite" \
  --run-id latest-valid \
  --enrich-attributes
```

Output lands in `reports/eda/<run_id>_<timestamp>/`.

---

## Source of truth

The **monitoring database** is authoritative, in this order:

1. PostgreSQL production database;
2. SQLite monitoring database (local development);
3. Parquet/CSV exports — only when no database is reachable.

The URL is read from `--database-url`, then `BAMA_EDA_DATABASE_URL`,
`BAMA_DATABASE_URL`, `BAMA_MONITOR_DATABASE_URL`, then the monitor's own default
SQLite path. **No credential is ever written to a config file or a manifest**; the
manifest records only the URL *shape*.

### Optional attribute enrichment (`--enrich-attributes`)

The monitoring schema keeps vehicle attributes in `advertisement_snapshots`,
written only when a detail page is actually scraped. Detail scraping is
rate-limited and policy-driven, so on a young installation snapshot coverage is a
few percent — on the database analysed below, **23 of 2,819 advertisements**.
Brand, model, condition and seller type simply cannot be answered from the
monitoring database alone.

`--enrich-attributes` joins the deep scraper's data. It is **opt-in and fully
provenanced**: the manifest records the source database, the row counts, the
temporal rule applied, every column added, and the coverage achieved. Doing this
silently would destroy auditability, which is why it is a flag.

Two joins happen:

* **per-advertisement attributes** from `ad_deep` — *every* analysable column, not
  a shortlist. Of 142 columns, 97 survive (12 private/blob, 19 entirely empty, 13
  constant, 1 too sparse), and each rejection is counted in the manifest.
* **trim-level technical specifications** — 113 features (sunroof, ABS, airbag
  count, ESC, BSD, 360° camera, kerb weight, boot capacity, performance) joined
  through the trim key, covering **95.3%** of advertisements.

The specifications are pivoted **here**, not read from the deep scraper's
`v_trim_specs_wide` view. The view projects each value as
`COALESCE(value_bool, value_num, value_text)`, and `value_num` is populated
opportunistically by the measure parser — so the string `(از پاییز 1402)`
("ESC, available from autumn 1402") surfaces as the bare number `1402.0`, which
in an analysis reads as a nonsensical ESC value. This package's projection is
type-aware: `value_bool` → present/absent, `value_num` only when the row is not a
String, otherwise `value_text`.

---

## Run selection

`latest-valid` picks the newest run satisfying **all** of:

```
status = 'valid'  AND  finished_at IS NOT NULL
AND comparison_applied  AND  discovered_count > 0
```

Each condition is applied only if the column actually exists, and any condition
that had to be skipped comes back as a warning — a run chosen under fewer
conditions than advertised is a different run. Override with `--run-id 123`; an
explicitly requested non-valid run is honoured *with* a warning.

---

## The three datasets

Three grains, never conflated — conflating them is how analyses double-count.

### A — `advertisement_cross_section` (one row per advertisement)

Identity, the observation from the selected run, the latest qualifying detail
snapshot, status, availability, sale inference, vehicle and seller attributes,
location, price, mileage, description and media statistics, publication
provenance, repost linkage.

**The no-future-snapshot rule.** A snapshot may only describe a run if it did not
come from the future. This is enforced twice:

```sql
s.run_id  <= :analysis_run_id     -- clock-free, and the real intent
s.scraped_at <= :reference_instant -- the timestamp rule
```

The second needs care. The monitor deliberately stamps observations and snapshots
with the run's **scheduled slot** (so duration bounds sit on a clean 24-hour grid)
while `finished_at` is the **real wall-clock** finish. For a run attributed to a
slot ahead of when it executed — a catch-up run, or a back-dated test run — the
slot is *later* than `finished_at`, and a literal `scraped_at <= finished_at`
discards every snapshot the run itself wrote. On the live database that rule alone
would have excluded **all 24** snapshots. So the reference instant is the later of
the two clocks, both counts are reported in the manifest, and the divergence is
raised as a warning rather than quietly resolved.

When several snapshots qualify, the latest wins via
`ROW_NUMBER() OVER (PARTITION BY advertisement_id ORDER BY scraped_at DESC, id DESC)`
— the `id` tiebreak makes the choice deterministic.

Every row carries `snapshot_scraped_at`, `snapshot_age_hours` and
`snapshot_is_stale`, because detail attributes are **not** contemporaneous with
the search observation, and `price_source` records whether a price came from the
detail page or the search card.

### B — `longitudinal_panel` (one row per run × advertisement)

`was_seen`, status before/after, card and detail price, price change, mileage,
filter-exit status, availability, sale evidence, event types, consecutive misses,
reappearance, repost indicators, observation timestamp, **run health**.

Observations from non-valid runs are kept as evidence but their absence signal is
blanked into `was_seen_trusted`: a scrape that failed halfway did not observe an
absence, it merely failed.

### C — `vehicle_entity_dataset` (one row per physical vehicle)

Aggregates linked advertisement ids across repost chains: `advertisement_count`,
`advertisement_ids`, publication bounds, `observed_monitoring_duration`,
`estimated_market_duration`, the three disappearance bounds, `was_reposted`,
`repost_count`, price range and change count, sale evidence, `left_truncated`,
`right_censored`, `eligible_for_duration_ranking`.

A reposted advertisement is **never** counted as a separate sold vehicle.

---

## Provenance

Every dataset row is prefixed with `run_id`, `scheduled_for`, `run_started_at`,
`run_finished_at`, `search_configuration_hash`, `scraper_version`,
`monitor_version`, `analysis_version`, `database_backend`. A source column of the
same name is **renamed**, not overwritten — an advertisement first discovered
under an older search definition keeps its own hash, and losing that difference
would hide the population change the hash exists to reveal.

`eda_manifest.json` records the selected database and run, source tables and row
counts, **every SQL query with its SHA-256**, output row counts, duplicates,
exclusions with reasons, warnings, package version, git commit, Python and
dependency versions, and the random seed.

---

## Integrity audit

30 adversarial checks across identity, referential integrity, temporal ordering,
value ranges, duration-bound ordering, semantic consistency, privacy and
provenance. Findings are `critical` / `error` / `warning` / `informational`;
a critical finding **stops** the analysis (exit code 5) with the quality report
already written.

**Every check is reported, including those that found nothing** — a report listing
only failures cannot be distinguished from one where the checks never ran.

The privacy gate deserves a note: it caught a real defect during development.
Pandas 3 gives string columns a dedicated `str` dtype, so the idiomatic
`dtype == object` test matches **nothing** and the sweep silently scanned zero
columns. `models.is_texty` is now the single place that decides what counts as
text. The cross-section also scrubs free text at build time, so the audit checks
the unexpected rather than being the only thing between a phone number and a CSV.

---

## Every feature, not a shortlist

`feature_analysis.py` discovers the analysable features from the data rather than
from a list written in advance, because a hand-picked list silently decides what
the analysis is allowed to notice. On the live database: **285 columns catalogued,
174 usable** — 73 numeric, 77 categorical, 24 boolean.

### Grain is tracked, and it changes what a number means

| grain | meaning | example |
|---|---|---|
| `listing` | observed for this advertisement | price, mileage, position |
| `vehicle` | from this advertisement's detail page | brand, colour, engine |
| `trim` | a property of the model-trim, **identical for every listing of that trim** | sunroof, ABS, kerb weight |

So "sunroof by city" over a `trim` feature describes the **trim mix** in that city,
not a count of individually inspected cars. Every table, chart and caveat that
touches a trim feature says so.

### Classification is stable across runs

A feature's kind is decided by dtype, by a present/absent **value vocabulary**
(`دارد`/`ندارد`, `present`/`absent`), or by a flag-shaped **name** — never by the
observed range. A count holding only 0 and 1 today may hold 5 tomorrow, and a
variable that silently switches from a share to a median between runs is not
reproducible.

Two traps this avoids:

* `1165 کیلوگرم` is a **measure** → numeric (the unit is stripped);
* `5 دنده دستی` ("5-speed manual") is a **description** → categorical. Reading it
  as the number 5 would discard manual-versus-automatic entirely.

The test: after removing the digits, a measure leaves at most one token of
residue; a description leaves more.

### What is produced

* `feature_catalogue.csv` — every column with kind, grain, coverage, and, for the
  unusable ones, **why** (dropping them would make "not analysed" look like "not
  present").
* `feature_profiles.csv` — a distribution per usable feature: median [IQR] for
  numeric, share present for flags, top levels for categoricals.
* `feature_level_distributions.csv` — long-format level counts.
* `feature_comparison_by_dimension.csv` — every feature across every group of
  every dimension (city, province, brand, seller type, fuel type, transmission,
  condition, price band). ~8,100 rows on the live data.
* `feature_differentiators.csv` — features ranked by how much they differ across
  each dimension: Kruskal–Wallis with epsilon-squared for numeric, chi-square with
  bias-corrected Cramér's V for categorical and boolean, Holm-adjusted.
* `feature_price_associations.csv` — every feature against asking price with the
  right test per kind.

### Tautologies are removed or flagged

Restatements of the price (`card_price_normalized`, `log10_price`, …) are excluded
from the price ranking outright — correlating a number with a copy of itself
measures nothing and crowds out real findings. A variable is never compared
against a bucketing of itself (`price_toman` vs `price_band`). Anything else with
|effect| > 0.95 is **flagged** `near_tautological` rather than silently dropped.

## Feature-to-feature association: linear vs non-linear

Three questions need three measures, and conflating them is how "these variables
are unrelated" gets said about a perfect parabola.

| measure | answers | blind to |
|---|---|---|
| Pearson `r` | is it **linear**? | curvature; wrecked by a contaminated minority |
| Spearman `rho` | is it **monotone**? | non-monotone shapes |
| distance correlation | is there **any** dependence? (0 iff independent) | nothing — but O(n²), so it runs on a deterministic subsample |
| correlation ratio `eta` | numeric vs categorical | — (no order to be linear in) |
| Cramér's V | categorical vs categorical | — |

On the live data, **12,930 pairs**: 2,610 numeric–numeric, 6,302
numeric–categorical, 4,018 categorical–categorical.

| shape | pairs |
|---|---|
| `no_detected_association` | 7,003 |
| `group_differences` | 4,687 |
| `approximately_linear` | 294 |
| `linear_within_trimmed_range` | 247 |
| `non_monotone_dependence` | 239 |
| `monotone_nonlinear` | 95 |

### Where Pearson alone would be wrong

`fuel_tank_capacity` vs `boot_capacity`: Pearson **0.01** — apparently unrelated.
Spearman **0.50**, distance correlation **0.65**. The relationship is real and
curved; a linear correlation reports nothing.

`stats_model_ad_count` vs `power`: Pearson 0.41, Spearman **0.08**, dCor **0.53**.
Non-monotone — both correlations understate it, for opposite reasons.

`feature_nonlinear_relationships.csv` ranks every such pair, and
`charts/45_nonlinear_pairs.png` draws them, because a number cannot show a shape.

### Two causes of a Pearson/Spearman gap, not one

A gap can mean the relationship is curved, or that the relationship is fine and a
small subpopulation on a different scale is destroying the estimate. The live data
has exactly that second case: ~1.6% of listings publish the year on the Gregorian
calendar among Jalali ones, putting two parallel clusters 621 apart in the
scatter. Pearson collapses to −0.02 while Spearman stays at 0.96.

A 1%/99% clip does not catch a 1.6% group — a 10%/90% clip does. Pairs where
Pearson recovers under clipping are labelled **`linear_within_trimmed_range`**,
named for what was measured rather than for a cause, because clipping cannot
distinguish a contaminated minority from curvature concentrated in the tails. Read
the scatter.

### Multiple comparisons at this scale

Benjamini–Hochberg FDR, not Holm. Over ~13,000 pairs Holm is so conservative that
nothing survives and the correction stops being informative; FDR bounds the
expected *share* of false discoveries instead of the chance of any.

Alias columns (`power` / `power_hp` / `power_text` / `power_raw` are one quantity
under four names) produce byte-identical statistics; **196** such duplicates are
collapsed out of the non-linearity report so the panel shows six distinct
relationships rather than the same one six times.

## Statistical discipline

* **Robust summaries.** Asking prices are strongly right-skewed (observed skew
  **17.8**), so medians and IQRs lead; means are reported with a note.
* **Spearman over Pearson**, and the live data shows exactly why: price against
  mileage is Pearson **−0.08** but Spearman **−0.48**. The relationship is
  monotone, not linear; the Pearson figure is the misleading one.
* **Effect sizes with every p-value** — rank-biserial for Mann-Whitney,
  epsilon-squared for Kruskal-Wallis, bias-corrected Cramér's V for categoricals.
* **Holm adjustment** across each test family.
* **Suppression**: groups below `min_group_size` are exported but labelled
  `insufficient_sample`; they never appear in a chart.
* **Exploratory means exploratory.** These tests were chosen after seeing the
  data. Nothing here is confirmatory and nothing here is causal.

---

## Censoring

Every observation gets exactly one class, because they need different handling:

| Class | Meaning |
|---|---|
| `fully_observed` | entry and disappearance both observed |
| `interval_censored` | disappeared inside a known interval |
| `right_censored` | still present when observation ended |
| `left_truncated` | already present when observation began (delayed entry) |
| `active_outside_filter` | left the search, not the market — a **censoring** event |
| `reappeared` / `reposted` / `unknown` | |

`min ≤ estimate ≤ max` is verified, not assumed.

The Kaplan–Meier estimator supports **delayed entry**: the risk set at time *t*
counts subjects with `entry < t ≤ exit`, so a listing already three weeks old when
monitoring began contributes only from week three rather than pretending it was
born when we first saw it. Greenwood's formula gives the variance; the band is
log-log transformed so it stays inside [0, 1].

Below `min_survival_group_size` subjects or `min_survival_events` events it
**refuses to estimate** and says why, rather than drawing a confident step
function over six points.

---

## Left truncation — the binding constraint

`first_seen_at` is when *monitoring* first saw a listing. `published_at` is when
the *seller* published it. For anything already active on day one they differ by
an unknown amount.

Two durations are kept apart and never merged:

```
observed_monitoring_duration = last_seen_at - first_seen_at    (every listing)
estimated_market_duration    = last_seen_at - published_at     (eligible only)
```

Eligible means the publication time is reliable **and** first observation was
within `entry_tolerance_hours` (36h) of it. Everything else gets `None`, not a
number with a caveat — a number with a caveat is what ends up in a chart without
the caveat.

Excluded rows are published to `left_truncated_excluded.csv` with a reason.

---

## Output

```
reports/eda/<run_id>_<timestamp>/
├── README.txt                        the terminology, up front
├── eda_manifest.json                 queries, counts, exclusions, environment
├── data_quality_report.json          audit summary
├── data_quality_issues.csv           all 30 checks and their results
├── schema_inventory.csv              every database column + analytical role
├── data_dictionary.csv               database-level dictionary
├── dataset_dictionary.csv            analytical columns, including derived ones
├── advertisement_cross_section.{csv,parquet}
├── longitudinal_panel.parquet
├── vehicle_entity_dataset.{csv,parquet}
├── field_completeness.csv            with an attributed cause per column
├── missingness_patterns.csv          joint missingness
├── descriptive_statistics.csv        count/missing/mean/sd/P01..P99/IQR/skew
├── categorical_distributions.csv
├── inventory_composition.csv
├── brand_model_combinations.csv
├── price_by_segment.csv              median/IQR/trimmed mean/CI + sample size
├── comparable_segments.csv           brand × model × year × mileage band
├── price_group_tests.csv             exploratory, with effect sizes
├── price_changes.csv
├── status_distribution.csv
├── status_transitions.csv
├── duration_summary.csv
├── duration_ranking_eligible.csv     or an explicit refusal
├── left_truncated_excluded.csv       what was excluded, and why
├── filter_exits.csv
├── repost_analysis.csv
├── sale_evidence_summary.csv
├── correlations.csv
├── feature_catalogue.csv          every column: kind, grain, coverage, usability
├── feature_profiles.csv           a distribution per usable feature
├── feature_level_distributions.csv
├── feature_comparison_by_dimension.csv
├── feature_differentiators.csv    what differs most across each dimension
├── feature_price_associations.csv
├── feature_pairwise_associations.csv   every pair: linear, monotone, general
├── feature_nonlinear_relationships.csv where a linear correlation would be wrong
├── analysis_warnings.csv
├── charts/                           45 charts, PNG + SVG
└── bama_eda_report.html              19 sections
```

Charts: every one states its sample size; grouped charts apply the minimum group
size and write the exclusion into the subtitle; count axes start at zero; log
scales are labelled. Persian labels render when a Persian-capable font is already
installed — **no font is bundled or downloaded** — and otherwise fall back to
English technical labels with a note, which beats tofu boxes.

An analysis the data cannot support produces a **placeholder chart explaining
why**, not a missing file: a missing chart reads as an oversight, a chart saying
"not enough data" reads as a finding.

---

## CLI

```bash
python -m bama_eda inspect-schema  --database-url "$BAMA_DATABASE_URL"
python -m bama_eda build-datasets  --run-id latest-valid
python -m bama_eda validate        --run-id latest-valid      # exit 1 if the audit fails
python -m bama_eda run             --run-id latest-valid --output-dir reports/eda
python -m bama_eda report          --run-id latest-valid
python -m bama_eda compare-runs    --run-id-a 100 --run-id-b 101
```

Options: `--database-url --run-id --output-dir --config --min-group-size
--min-survival-group-size --min-survival-events --snapshot-stale-hours
--include-left-truncated --enrich-attributes --enrichment-db --html/--no-html
--no-charts --export-csv/--no-export-csv --export-parquet/--no-export-parquet
--log-level`.

Exit codes: `0` ok, `1` audit failed or forbidden phrasing found, `2` usage,
`3` schema error, `4` run selection failed, `5` critical integrity finding.

---

## Testing

**235 tests** in `tests/bama_eda/`, run on SQLite and PostgreSQL 16.13. Fixtures
are synthetic and construct every scenario explicitly: valid/invalid/unfinished
runs, a future snapshot, two snapshots for one advertisement, an advertisement
with none, a filter exit, a disappearance, a reappearance, a repost chain, a
left-truncated listing, an eligible one, an unreliable publication timestamp, and
a planted contact number.

```bash
pytest tests/bama_eda -q
ruff format --check src/bama_eda tests/bama_eda
ruff check src/bama_eda tests/bama_eda
mypy --python-version 3.13 src/bama_eda
```

The mypy flag is required because numpy 2.5's stubs use `type` statements that
need Python ≥ 3.12, while the project's global `python_version` is 3.11 for the
collection packages.

---

## Notebook

`notebooks/bama_eda_overview.ipynb` is a **thin viewer**: every number comes from
a tested package function. No analytical logic lives only in a cell.

---

## Known limitations

These are properties of the data, not of the code.

* **Left truncation is severe on a young installation.** On the analysed database
  only **6 of 2,819** advertisements (0.21%) were first observed close enough to
  publication for their full market life to be known. The other 2,813 have
  observed durations that are lower bounds. The usable subset grows only as newly
  published listings are observed from publication onward.
* **Three monitoring runs.** Day-over-day dynamics — price changes, disappearance
  rates, repost rates — are barely observable and must not be read as market
  rates. Zero observed reposts means *the window was too short*, not that
  reposting is rare.
* **Detail coverage is thin.** 23 of 2,819 advertisements have a monitoring
  snapshot; vehicle attributes otherwise come from the enrichment join.
* **Missingness is not random.** Attributes are absent most often for listings
  that disappeared before a detail scrape was due — the shortest-lived ones. Any
  analysis restricted to complete rows is biased towards listings that lasted.
* **Asking prices only.** There are no transaction prices anywhere in this
  dataset.
* **Sale evidence is uncalibrated.** No labelled sample exists, so no empirical
  sale rate is reported for any band and none is interpolated.
* **One filtered search, not a market.** The population is
  `year ≥ 1397, price ≥ 1,000,000,000, passenger_car, iranian` — and it changes
  between runs.
* **Mixed calendars.** About 1.5% of cards publish a Gregorian year where a Jalali
  one is expected. These are converted (−621) rather than discarded, and
  `year_calendar_source` records which rows were converted.
