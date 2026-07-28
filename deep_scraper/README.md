# Deep per-car scraper (`deep_scraper/`)

A second-stage scraper that revisits every advertisement discovered by
`src/bama_scraper` and captures **everything publicly available about each car** —
including the ~110 technical feature items (sunroof, airbags, ABS/ESC, dimensions,
weight, fuel-tank capacity) that live on a completely different endpoint, plus
daily model price history and dealer profiles.

It is self-contained: it *reads* the existing `output/bama_ads.sqlite`
**read-only** and writes to its own database under `deep_scraper/data/`. The
original dataset is never modified, and the audit verifies that by comparing the
source file's SHA-256 before and after.

---

## 1. Why this exists

An audit of the first scraper's output found that it **discarded 42 of the 83
payload keys it already downloaded**, and that **19 declared columns were 0%
populated**. Investigating why turned up three public data sources it never
touched.

| Source | What it adds |
|---|---|
| `GET /cad/api/detail/<code>` | ~7 KB JSON, no auth. The **only** place `fuel` exists — which is exactly why `fuel_type` was empty. Also `province`, `is_pre_sale`, `ad_class_i_d`, `publish_networks`, `inspection_station_count`, full-resolution `images[].original`, float `dealer.score`. |
| `GET /nws/api/CarReview/getspecification` | **13 groups / ~110 typed items per model-trim** — the "all features" source. 59 distinct keys cover 2,760 of 2,846 ads, so ~180 requests enrich the whole corpus. |
| `GET /cad/api/price/detail` | ~10 series per model (market **and** factory price, per model-year, per equipment class), ~94 daily points each. |
| `GET /cad/api/Corporation/<id>` + `/ads/<id>` | Dealer profile and full live inventory; the inventory doubles as a coverage cross-check on the original discovery. |

Some gaps are **not** fixable and no column pretends otherwise: Bama exposes **no
view counts**, no per-ad price history, and no insurance / technical-inspection /
chassis-status fields (those are Divar concepts). The only route to insurance or
document status is natural-language extraction from the seller's description.

---

## 2. The numeric encoding that drives the schema

Bama serializes a decimal as **its display string with the decimal point
deleted**, so the implied scale is `10 ** (fractional digits in the text)` — it is
*not* a constant:

```
volume          value=17   text="1.7 لیتر"                 -> /10
fuelConsumption value=69   text="6.9 لیتر در صد کیلومتر"    -> /10
acceleration    value=12   text="12 ثانیه"                 -> /1   (12 s, NOT 1.2 s)
power           value=113  text="113 اسب‌بخار"              -> /1
```

Measured across the live corpus, acceleration is an integer on **1,305** ads and
one-decimal on **1,234**. A fixed `/10` would silently turn 12 seconds into 1.2
on more than a thousand cars.

Therefore **the text is authoritative and the integer is only a checksum.** Every
such field carries three columns — `*_text`, a parsed `REAL`, and `*_raw` — and
`numeric_agreement_json` records which class the reconciliation fell into
(`exact`, `scaled_x10`, `scaled_x100`, `mismatch`, `text_only`, `int_only`). A
mismatch keeps the text value and is surfaced rather than silently persisted.

---

## 3. Install and run

Requires the parent project to be installed (`pip install -e .` in the repo
root), which puts `bama_scraper` on the path. No extra install is needed for the
deep scraper itself.

```bash
# 1. import the discovered-ad inventory (read-only on the source DB)
python deep_scraper/run_deep.py seed

# 2. see exactly what the run will cost, without issuing anything
python deep_scraper/run_deep.py plan

# 3. phase A: both views of every ad, merged and cross-validated
python deep_scraper/run_deep.py ads --concurrency 2

# 4. enrichment phases (each keyed on values discovered in phase A)
python deep_scraper/run_deep.py specs
python deep_scraper/run_deep.py prices
python deep_scraper/run_deep.py dealers

# 5. rebuild the generated wide-specs view, audit, export
python deep_scraper/run_deep.py reindex
python deep_scraper/run_deep.py audit
python deep_scraper/run_deep.py export

# or everything end to end
python deep_scraper/run_deep.py run
```

Useful flags: `--limit N` (smoke tests), `--refresh`, `--retry-delisted`,
`--skip-api` / `--skip-html`, `--phases a,c`, `--dry-run`,
`--no-dealer-address`, `--max-requests`, `--log-level`.

Exit codes: `0` ok, `1` audit failed or blocked, `2` usage/seed error,
`130` interrupted.

---

## 4. Request budget

One `DeepFetcher` is shared by every phase, so its single semaphore bounds the
**global** rate even when phases interleave. The two requests per advertisement
are issued sequentially inside one worker task, which keeps the instantaneous
rate at `concurrency` rather than double it, and lets a 404 from the API skip the
HTML fetch entirely.

| Phase | Entities | Req/entity | Requests |
|---|---|---|---|
| A ads | 2,855 | 2 | 5,710 |
| B specs | 59 review keys | 3 | ~177 |
| C prices | 20 price keys + 2 reference | 1 | ~22 |
| D dealers | 40 | 2 | ~80 |
| | | | **≈6,000 (~60 min at concurrency 2)** |

`max_requests` (default 8,000) is a hard ceiling: once reached, the run stops with
`termination_reason = request_budget_exhausted`, so retries can never quietly
blow the budget.

---

## 5. Merge and cross-validation

Neither source is authoritative for everything, so `merge.py` declares precedence
once in `FIELD_RULES`:

| Class | Winner | Examples |
|---|---|---|
| Typed integers | **HTML** | `mileage_km`, `price_toman`, `year_jalali`, the `*_raw` checksums |
| API-exclusive | **API** | `fuel_type`, `province`, `is_pre_sale`, `publish_networks`, `images[].original` |
| Persian display strings | **API** | `title`, `body_status`, colours, `transmission` |
| Explicit condition | **JSON-LD** | `condition_new_used` (`NewCondition`/`UsedCondition`) |
| HTML-exclusive | **HTML** | `is_promoted` (= `media.badge`), `meta_slogan`, breadcrumb slugs |

Every field records its origin in `provenance_json`
(`A`/`H`/`L`/`B`=both-agreed/`-`=absent); a *disagreement* additionally writes a
row to `ad_field_conflicts`. The provenance map answers "where did this come
from?" cheaply for all fields; the conflicts table answers "what disagreed?"
without scanning ~170k rows.

**Comparison is deliberately not naive.** Validated offline against 2,845 real
ad pairs, strict equality reported ~8,500 conflicts, almost all spurious:

- `title` — the API writes `دنا، پلاس EF7`, the page writes `دنا پلاس EF7`
  (punctuation only) → compared with punctuation tolerance.
- `location_text` — the API gives `کرج / فردیس`, the page gives
  `کرج، البرز، فردیس` (different granularity) → compared as component *sets*.
- `breadcrumb_json` — the API emits `{"title","url"}`, the page emits
  `{"title","to","navigatable"}` (same trail, different key names) → compared by
  extracted `(title, link)` pairs.
- `published_text` / `published_ts` — relative to capture time → never a conflict.

After that the same 2,845 pairs produce **27 conflicts total**: 24 are image
counts changing between captures, 1 is a genuine seller price edit (confirmed
against the live site), 1 a redaction difference. Critical-conflict rate
0.035%, against a 1% audit threshold.

---

## 6. Privacy

Seller telephone numbers are **never stored**. Bama replicates the seller's text
in **three** places, and all three are scrubbed:

1. `content.phone` / `data.detail.phone` — a server-masked mobile; the key is
   removed outright.
2. `content.description` — free text sellers type numbers and Telegram handles
   into.
3. **`metadata.description` and JSON-LD `Product.description`** — Bama builds
   these *from* the seller's description, so a number leaks into them too. This
   was found by the project's own privacy gate on live ad `cuaxpwpu`, not by
   inspection.

`description` therefore holds the **scrubbed** text and `description_redactions`
counts what was removed. Redaction patterns are anchored on Iranian mobile
prefixes and guarded at both ends against hex characters — without that guard a
SHA-256 digest or image UUID matches a phone pattern by chance (measured: 5 false
positives against 2 genuine hits).

`deep audit` re-scans every stored text column and every raw payload; any hit
fails the run.

**Endpoint guards.** `endpoints.py` raises `ForbiddenEndpointError` *before a
socket is opened* for:

- `/cad/api/log/visit`, `/log/impression`, `/log/share`, `/event/api/v1/events` —
  write-only telemetry; calling them would inflate a seller's own view counters.
- `/cad/api/carad/*`, `/cad/api/price?`, `/prf/api/*` — authenticated (401).
- `/cad/api/Corporation/phone/*` — exists to reveal personal contact data.

`robots.txt` disallows exactly one path, `/uploads/BamaImages/CampaignBanner/`,
which `is_allowed_asset()` refuses; vehicle photos under
`/uploads/BamaImages/VehicleCarImages/` are permitted. Both rules are unit-tested
with a request log asserting **zero** requests were issued.

`dealer_address` is a business premises and is included by default; pass
`--no-dealer-address` to omit the column from every export.

---

## 7. Schema

`deep_scraper/data/bama_deep.sqlite`:

```
deep_runs, deep_checkpoint, deep_errors     bookkeeping
deep_queue                                  phase A work queue + per-source status
ad_deep                                     one row per advertisement (~150 columns)
ad_media_deep                               one row per image/video
ad_raw_attributes                           every payload key, tagged by source
ad_field_conflicts                          only where the sources disagreed
trim_reviews, trim_spec_groups, trim_specs  the ~110 feature items, LONG format
trim_specs_typed, spec_key_catalog          typed subset + generated data dictionary
price_keys, price_series, price_points       daily market/factory history
price_brands, price_hierarchy               the priced universe
dealers, dealer_ads                         profiles + inventories (coverage check)
v_trim_specs_wide                           generated wide pivot (see reindex)
```

### Why specs are stored long, not wide

The item set is **model-dependent** — an EV trim carries battery items an ICE trim
does not — so a 110-column table would be mostly NULL and would need a migration
for every new model. The keys are also Persian display strings the site can
reword: in long format a reworded key becomes a new row and the old data stays
attributable, whereas a renamed column silently starts yielding NULLs. Booleans
get a real `value_bool INTEGER`, so `WHERE item_slug='sunroof' AND value_bool=1`
works without string comparison. Cost is negligible (~6,500 rows).

Analysts still get wide: `deep reindex` regenerates `spec_key_catalog` and the
`v_trim_specs_wide` view from the data, and `deep export` writes
`trim_specs_wide.csv/.parquet` with deterministic column order. The wide schema is
a **command**, not a code change.

`item_slug` is part of a primary key, so it must be stable. Reviewed keys come
from `spec_slugs.py`; anything unmapped gets a deterministic `unk_<hash>` and is
reported by the audit so the mapping can be completed deliberately rather than
guessed. Slug matching is ZWNJ-insensitive because Bama's own group titles are
inconsistent about it (`سیستم‌‌ کمکی راننده` ships with two consecutive ZWNJ where
the item keys use one).

---

## 8. Resume behaviour

SQLite is the checkpoint. Every phase has its own queue and status column.

```
pending -> fetching -> completed | partial | delisted
                    -> retryable_error   (re-queued until max_retries)
                    -> permanent_error   (never re-queued)
```

`fetching` is deliberately **not** terminal: if the process dies mid-batch those
rows stay claimable even if `recover_stuck()` never runs. `--refresh` walks a
**fixed snapshot** (`all_queue()`), because a status-driven loop cannot terminate
when status is ignored — the same bug the parent project regression-tests.

Delisted handling keeps churn accounted for rather than lost:

| API | HTML | Status | Row written |
|---|---|---|---|
| 200 | 200 | `completed` | full merged row |
| 404/410 | 404/410 | `delisted` | minimal row, `is_delisted=1`, `delisted_at` |
| one 404 | other 200 | `partial` | single-source row + warning |
| 403/429 | — | stays `pending` | none, and **`attempts` is not bumped** — a block is the site's state, not the advertisement's |

---

## 9. Testing

```bash
pytest deep_scraper/tests -q
ruff check deep_scraper && ruff format --check deep_scraper
mypy deep_scraper/bama_deep
```

`pyproject.toml` sets `testpaths = ["tests"]`, so bare `pytest` collects only the
parent suite — run the path explicitly, or add `deep_scraper/tests` to
`testpaths`. Test modules are named `test_deep_*` to avoid pytest's
import-file-mismatch against the parent `tests/`.

No test touches the live site. Fixtures are real captured responses, scrubbed,
with fake contact data deliberately re-inserted into the privacy fixtures so
those tests exercise a realistic payload rather than an already-clean one.

Two offline tools re-validate the parsers against real data at **zero request
cost**, and both are gates in the build:

```bash
python deep_scraper/tools/sweep_corpus.py   # parse all offline ad records
python deep_scraper/tools/merge_report.py   # merge them against stored HTML payloads
```

---

## 10. Known limitations

- Bama exposes no view counts, no per-ad price history, and no
  insurance/inspection/chassis fields. Those columns do not exist here.
- Price-history dates are **year-less** (`18 فروردین`), so the year is inferred
  by walking the series backwards from an anchor. Every point records
  `year_inferred`, making the inference visible; a series spanning more than one
  new-year boundary would defeat it.
- `modified_date` is interpreted in local time, which is the site's own clock
  (Asia/Tehran).
- `seller.score` has no text sibling in the HTML, so a dot-stripped integer is
  recovered by bounds (`45 -> 4.5`) and flagged via `dealer_score_heuristic`. The
  JSON API's float is preferred whenever present.
- Specification data is per model-trim, not per car: two ads on the same trim
  share the same feature rows. Ads whose payload carries no `review_url`
  (86 of 2,846) have no feature data at all.
- `Corporation/ads` ignores pagination and returns the whole inventory; its
  `metadata.total_count` is wrong (reports 0), so `ads_listed` counts the array.
- Parsing depends on the `__NUXT_DATA__` devalue payload and the JSON API shapes.
  If either changes, records degrade to `partial` and are reported rather than
  failing silently.
