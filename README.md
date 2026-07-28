# Bama Scraper

A production-quality scraper for [bama.ir](https://bama.ir) filtered car search
results and their advertisement detail pages.

Reference target used throughout this document:

```text
https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian
```

---

## 1. Purpose

Given a Bama search URL, the scraper:

1. determines how the result set is actually delivered (JSON endpoint,
   pagination, or infinite scroll) by observing the live page;
2. enumerates **every** advertisement reachable from that search, incrementally
   and resumably;
3. visits every unique detail page and extracts all publicly visible fields;
4. normalizes Persian text and numbers while preserving the originals;
5. stores everything in SQLite and exports CSV / JSONL / Parquet;
6. runs an adversarial completeness audit that tries to disprove the claim
   "we collected everything".

It deliberately **does not** collect seller phone numbers or any other private
contact data, and it does not attempt to bypass access controls.

---

## 2. Installation

Requires Python 3.11+ (developed and executed on 3.14).

```bash
git clone <your-repo-url> bama_scrape
cd bama_scrape

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # runtime dependencies
pip install -e ".[dev]"            # + pytest / ruff / mypy
```

### Playwright browser

Playwright needs a real Chromium binary. This is a separate download:

```bash
playwright install chromium
```

Only `inspect`, `--mode scroll`, and the browser fallback need it. The default
`--mode auto` path uses plain HTTP once the endpoint has been confirmed.

Verify the install:

```bash
python -c "from playwright.sync_api import sync_playwright; \
  print(sync_playwright().start().chromium.executable_path)"
```

---

## 3. How the site actually works

Everything below was **measured against the live site**, not assumed.

### 3.1 Loading mechanism

The search page is a server-rendered Nuxt 3 application. The first 30 cards are
present in the initial HTML; scrolling then triggers `fetch` calls to a public
JSON endpoint:

```text
GET https://bama.ir/cad/api/search
      ?body=passenger_car
      &country=iranian
      &yearFrom=1397-2018
      &priceFrom=1000000000
      &pageIndex=<n>
      &pageSize=<n>
```

No authentication, cookie, or token is required. `robots.txt` disallows only
`/uploads/BamaImages/CampaignBanner/`; `/car` and the detail pages are
permitted, and no `Crawl-delay` is declared.

### 3.2 Traps this scraper works around

These are the non-obvious behaviours that break naive scrapers:

| Behaviour | Reality | Consequence |
|---|---|---|
| `metadata.total_count` | **Grows with every page** — a running "delivered so far" counter (31, 61, 91 …), and it *overshoots* past the end (2850, 2880 for a 2849-ad set) | Useless as a total, and it never settles on the real figure |
| `metadata.total_pages` | Grows with you (2, 3, 4 …), settling only at the end | Useless as a page count while paging |
| `metadata.has_next` | Does flip to `false` on the last page with ads — but it is a single signal sitting next to two counters that are demonstrably wrong | Recorded as corroborating evidence, not trusted as the sole stop rule |
| `pageSize` | Values above 30 return **HTTP 400** `"PageSize is not Valid"` | Batch size is capped at 30 regardless of what you request |
| `data.ads[]` | Mixes `type: "ad"` with `type: "banner"` campaign slots | Banners must be filtered out or they inflate counts |
| `price.type: "negotiable"` | Comes with `price: "0"` | Treating it as a number yields fake 0-Toman cars |
| Detail payload | Contains a partially masked `phone` field | Must be stripped, not persisted |
| DOM scrolling | Virtualizes: older cards are unmounted | Must extract *every cycle*, not once at the end |

Termination is therefore driven purely by **observed emptiness**: the walk stops
after `api_stale_pages` (default 2) consecutive pages that contain zero real
advertisements. The endpoint's own `has_next` is captured in the run summary
(`discovery.last_metadata`) so the two can be compared after the fact, but it is
never the thing that stops the loop.

Measured boundary for the reference target:

```text
page=93  real_ads=30  total_count=2821  total_pages=95  has_next=True
page=94  real_ads=25  total_count=2845  total_pages=95  has_next=False
page=95  real_ads=0   total_count=2850  total_pages=95  has_next=False
page=96  real_ads=0   total_count=2880  total_pages=96  has_next=False
```

Note `total_count` climbing to 2880 for a set that really contains ~2849, and
`real_ads` on page 94 varying between runs (29 → 27 → 25) as the live feed
changes underneath — both reasons to stop on evidence rather than on counters.

### 3.3 Detail pages

Detail pages are fully server-rendered. The complete advertisement object is
embedded in the Nuxt payload:

```html
<script type="application/json" id="__NUXT_DATA__">[ ... ]</script>
```

This is [devalue](https://github.com/Rich-Harris/devalue)-serialized: a flat
array where containers reference members by integer index and negative values
are sentinels (`-1` = undefined). `detail_parser.resolve_devalue` reconstructs
it, and the ad lives under the key `get-ad-pdp-car_<code>`.

Because the payload is in the raw HTTP response, detail scraping uses **httpx**
rather than a browser — far cheaper. A Playwright fallback triggers
automatically if the payload is ever missing.

Three JSON-LD blocks (`Product`, `WebPage`, `ImageObject`) are also captured as
corroborating evidence.

---

## 4. Interpreting the filters

The public URL parameters are **not** what the API receives. Verified mapping:

| URL parameter | API parameter | Meaning |
|---|---|---|
| `year=1397-2018,` | `yearFrom=1397-2018` | Production year **1397 (Jalali) / 2018 (Gregorian) and newer**. The trailing comma is an empty upper bound. Not "exactly 1397". |
| `price=1000000000` | `priceFrom=1000000000` | **Minimum** price of 1,000,000,000 Toman. A lower bound — *not* a maximum. |
| `body=passenger_car` | `body=passenger_car` | Passenger-car body type, passed through unchanged. |
| `country=iranian` | `country=iranian` | Iranian-manufactured vehicles, passed through unchanged. |

The `1397-2018` token encodes the same year in both calendars (1397 + 621 =
2018). The scraper stores this interpretation verbatim in `run_summary.json`
under `applied_filters`, and the audit asserts that no collected advertisement
violates either bound.

The target URL is never silently modified.

---

## 5. Commands

```bash
# 1. Inspect the live page: record XHR, detect the mechanism, save a screenshot
python -m bama_scraper inspect \
  --url "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian" \
  --headed

# 2. Discover every advertisement in the result set
python -m bama_scraper discover \
  --url "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian" \
  --mode auto

# 3. Fetch and parse every discovered detail page
python -m bama_scraper scrape-details --concurrency 2 --resume

# 4. Everything end-to-end (discover -> details -> verify -> export -> audit)
python -m bama_scraper run \
  --url "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian" \
  --mode auto --resume --concurrency 2

# 5. Re-export from the database without re-scraping
python -m bama_scraper export

# 6. Completeness audit (add --skip-second-pass to audit offline)
python -m bama_scraper validate

# 7. Recompute derived fields on stored records after a normalizer improves,
#    without refetching anything from the site
python -m bama_scraper backfill
```

`backfill` exists because some normalized fields are pure functions of raw text
that is already persisted. When a date or number parser is improved, stored rows
can be corrected in place. Relative phrases are re-anchored to each record's own
`scraped_at`, so a backfill run days later still produces the timestamp the
original scrape should have produced. Pass `--refresh` to recompute every row
rather than only the ones still missing a value.

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--url` | the reference target | Search URL to scrape |
| `--mode auto\|api\|pagination\|scroll` | `auto` | Discovery strategy; `auto` uses the JSON endpoint and falls back to scrolling |
| `--headed` / `--headless` | headless | Show the browser window |
| `--max-scrolls` | 2000 | Hard ceiling on scroll cycles |
| `--stale-cycles` | 8 | Consecutive no-progress cycles required to stop scrolling |
| `--max-runtime` | 7200 | Seconds per stage |
| `--concurrency` | 2 | Parallel detail fetches (deliberately low) |
| `--delay-min` / `--delay-max` | 0.4 / 1.2 | Random per-request delay |
| `--resume` | on | Skip advertisements already completed |
| `--refresh` | off | Re-scrape everything, including completed rows |
| `--download-images` | off | Also download image bytes (URLs are always stored) |
| `--output-dir` | `output` | Where the database and exports go |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--limit` | – | `scrape-details` only: stop after N ads (smoke tests) |

### Configuration file

```bash
cp config.example.yaml config.yaml
python -m bama_scraper run --config config.yaml
```

CLI flags override file values; file values override defaults.

---

## 6. Output schema

`output/` after a complete run:

| File | Contents |
|---|---|
| `bama_ads.sqlite` | **Source of truth** — all tables |
| `bama_ads.csv` | One row per advertisement, UTF-8 **with BOM** (Excel-safe) |
| `bama_ads.jsonl` | One JSON object per advertisement |
| `bama_ads.parquet` | Columnar export |
| `bama_media.csv` | Normalized media, one row per image/video |
| `bama_errors.csv` | Every failure with stage and message |
| `bama_discovered.csv` | Card-level discovery inventory |
| `run_summary.json` | Run metadata + completeness audit |
| `network_candidates.json` | Every JSON endpoint observed |
| `debug/` | Screenshots and gzipped HTML for failed parses |

### Tables

```text
runs                -- one row per invocation: filters, mode, termination reason
discovered_ads      -- card-level inventory; URL is the primary key
ad_details          -- parsed detail record as a JSON payload
ad_media            -- one row per image/video
raw_attributes      -- flattened label -> value pairs, including unmapped fields
scrape_errors       -- every error, with stage and type
network_candidates  -- endpoints seen while inspecting
discovery_checkpoint-- resumable progress markers
```

### Field conventions

* **Raw text is never discarded.** Normalized fields sit beside their source:
  `mileage_text="40,000 km"` / `mileage_km=40000`,
  `price_text="1,585,000,000"` / `price_toman=1585000000`,
  `year_text="1403"` / `year_jalali=1403` / `year_gregorian=2024`.
* **Missing is `null`, never `0`.** A negotiable or contact-for-price listing
  has `price_toman = null`. A genuine zero-kilometre car has `mileage_km = 0`
  and `is_zero_km = true` — the two cases are distinguishable.
* **Persian is preserved.** ZWNJ (U+200C) survives normalization because it is
  orthographically meaningful (`می‌رود`). Only Arabic character variants are
  folded to Persian (`ي`→`ی`, `ك`→`ک`) and digits converted to ASCII.
* **List columns** (`image_urls`, `badges`, `json_ld`, …) are JSON-encoded in
  the flat exports so one row always equals one advertisement.
* **Evidence** per record: `html_sha256`, `scraper_version`, `scraped_at`,
  `source` (which evidence source the fields came from) and `fetch_source`
  (how the HTML was retrieved) are tracked separately.

---

## 7. Resume behaviour

SQLite is the checkpoint. Every advertisement carries a status:

```text
discovered -> pending -> scraping -> completed
                              \-> retryable_error   (re-queued, bounded)
                              \-> permanent_error   (404/410, never re-queued)
```

On restart:

* `completed` rows are skipped unless `--refresh` is given;
* `retryable_error` rows are retried until `max_retries` (default 4) attempts;
* rows left in `scraping` by a killed process are recovered to `pending`;
* discovery resumes from the last checkpointed page.

Progress is committed continuously, so `Ctrl+C` is safe at any moment — the run
persists what it has and exits with code 130. Existing checkpoints are never
overwritten with empty values: upserts use `COALESCE`, so a later sighting that
lacks a price cannot erase a price already recorded.

---

## 8. Termination algorithm

### API discovery

The endpoint's own counters are unusable (§3.2), so termination is empirical:

1. request `pageIndex = n`;
2. count entries with `type == "ad"`;
3. zero real ads increments the stale counter, any ads reset it to 0;
4. stop when the stale counter reaches `api_stale_pages` (default 2).

A single empty page is never sufficient.

### Scroll discovery

Each cycle:

1. extract **all** cards currently in the DOM;
2. upsert them immediately (this is what defeats virtualization);
3. record unique count and document height;
4. scroll to 92 % of height, then to the true bottom;
5. wait for network/DOM activity;
6. click an enabled "load more" control if present;
7. re-measure.

A cycle counts as **stale** only when *every* one of these holds:

* no new unique advertisement URL, **and**
* no listing XHR fired, **and**
* document height did not grow, **and**
* no loading indicator is visible, **and**
* no enabled "load more" control exists, **and**
* the viewport is already at the bottom.

After `stale_cycles` (default 8) consecutive stale cycles, a **final
stabilization check** waits twice the settle time and re-extracts. If that finds
anything new, the counter resets and scrolling continues. Only when it finds
nothing is the run declared complete.

Bounded by `--max-scrolls` and `--max-runtime`; on an uncertain termination a
screenshot and full HTML are written to `output/debug/`.

### Example progress output

```text
page=92 entries=30 ads=30 new=30 unique_total=2790 total_count_field=2791 stale=0
page=93 entries=30 ads=30 new=30 unique_total=2820 total_count_field=2821 stale=0
page=94 entries=29 ads=29 new=29 unique_total=2849 total_count_field=2849 stale=0
page=95 entries=0  ads=0  new=0  unique_total=2849 total_count_field=2850 stale=1
page=96 entries=0  ads=0  new=0  unique_total=2849 total_count_field=2880 stale=2
```

Scroll mode prints the equivalent per-cycle line:

```text
cycle=18 visible=30 new=12 unique_total=426 height=28491 stale=0
```

---

## 9. Completeness verification

`validate` (and the tail of `run`) performs an audit designed to **fail**:

* counts: discovered, distinct IDs, duplicate URLs/IDs, completed, failed,
  with/without price, with/without images, with/without description;
* first and last discovered URL, cycles, stale cycles, termination reason,
  elapsed time, errors, parse warnings;
* **integrity assertions**, each of which must be zero:
  * orphan detail rows with no discovery record,
  * rows marked `completed` with no stored detail,
  * URLs from which no advertisement ID can be extracted,
  * ad-ID / URL mismatches,
  * advertisements priced **below** the `priceFrom` filter,
  * advertisements older than the `yearFrom` filter,
  * any record containing a `phone` key;
* an **independent second discovery pass**, diffed against the first. New
  advertisements are merged and detail-scraped, then verification repeats. The
  inventory is only called stable when a pass finds no previously unseen URL.

Live classifieds change during a run, so ads *disappearing* between passes does
not fail the audit; ads *appearing* triggers another round. Exact run start and
end timestamps are recorded.

`validate` exits non-zero when any check fails.

### Endpoint-vs-browser cross-check

Because discovery trusts a JSON endpoint, that endpoint was validated against
what the browser actually renders. A bounded Playwright scroll (12 cycles,
separate database) was diffed against the API inventory:

```text
browser pass:      600 unique ads over 12 cycles
present in API:    595
absent from API:     5
```

All five absentees were then found on `pageIndex=0` of a fresh API walk with
posted times of *دقایقی پیش* ("minutes ago") and *۱ ساعت پیش* — they were
**posted after discovery finished**, not omitted by the endpoint. Endpoint and
browser agree on 595/595 of the advertisements that existed at discovery time.

The full result is saved to `output/cross_check_report.json`. This is exactly
the churn the verification pass is built to absorb.

---

## 10. Reliability and request discipline

* concurrency defaults to **2**; the endpoint page walk is sequential;
* random 0.4–1.2 s delay before every request;
* bounded exponential backoff (tenacity), max 4 attempts;
* connection, request and navigation timeouts;
* HTTP pool and browser context recycled every 200 pages;
* 403/429 is treated as a **block**: the run stops gracefully, persists
  progress, and reports it rather than retrying harder. The abort is enforced
  *inside* the fetch, after the concurrency slot is acquired — every coroutine
  in a batch is already past any earlier check by the time the first response
  arrives, so a naive flag would still let the rest of the batch fire. Ads that
  were never sent go back to `pending` **without** consuming a retry attempt:
  a block is the site's state, not the advertisement's;
* 404/410 is permanent and never retried;
* failed parses save gzipped HTML to `output/debug/`.

---

## 11. Ethics and access limitations

* `robots.txt` was fetched and honoured. It disallows only
  `/uploads/BamaImages/CampaignBanner/`; `/car` and detail pages are allowed.
  No `Crawl-delay` is specified; conservative delays are used regardless.
* Only the same public JSON endpoint the website's own frontend calls is used.
  No authentication, no private endpoints, no access-control bypass.
* **No CAPTCHA, paywall, rate-limit or anti-automation control is
  circumvented.** If the site blocks, the scraper stops and reports it.
* **No private data is collected.** Seller phone numbers appear in the page
  payload and are explicitly stripped (`detail_parser._PRIVATE_KEYS`) before
  anything is written to disk. This is enforced by tests and re-checked by the
  audit. Exact street addresses and hidden contact details are likewise not
  collected; only the public city / province / neighbourhood labels are.
* Scraped content remains subject to Bama's terms of use and to Iranian and
  local law. You are responsible for how you use the output — particularly for
  republication or commercial use.

---

## 12. Observed live run

Executed 2026-07-27 against the reference target. Numbers below are measured,
not estimated.

| Stage | Result |
|---|---|
| Discovery | 2,849 unique ads over 97 API pages in 200 s; stopped after 2 consecutive empty pages |
| Detail scraping | 2,846 completed, 9 permanent errors, 0 retryable, 0 blocked; ~35 min at concurrency 2 |
| Verification | 3 passes: +5 ads, +1 ad, then **0 new → stable** at 2,855 discovered |
| Final dataset | 2,846 advertisements, 6,834 media rows |

The 9 permanent errors were all **HTTP 410 Gone** — listings deleted between
discovery and detail scraping. They are recorded in `bama_errors.csv` rather
than silently dropped: `2,846 completed + 9 gone = 2,855 discovered`.

Data quality of the final dataset:

```text
with price          2846 / 2846   (100%)   min = 1,000,000,000 Toman
with images         2294 / 2846   ( 81%)
with description    2461 / 2846   ( 86%)
publication date    2846 / 2846   (100%)   2025-11-09 .. 2026-07-27
parsed from payload 2846 / 2846   (100%)   partial parses: 0
phone data stored      0
```

`min price` landing exactly on 1,000,000,000 and `min year` exactly on 1397
is the empirical confirmation that both filters are **lower bounds**.

Integrity assertions, all zero: orphan detail rows, completed-without-detail,
unparseable URLs, ad-id/URL mismatches, ads priced below the filter, ads older
than the filter, records containing a phone key.

## 13. Known limitations

* `total_count` from the API is meaningless (§3.2); the only true total is the
  count the scraper itself accumulates.
* The result set is a live classifieds feed. Advertisements are added and
  removed while the scraper runs, so an exact count is only meaningful together
  with the run's start/end timestamps.
* Ordering is server-controlled and not guaranteed stable between passes; this
  is why deduplication uses the canonical URL rather than position.
* Deep pagination beyond the observed end was probed by binary search rather
  than exhaustively; the endpoint returns empty pages past the last one.
* `--download-images` stores bytes but performs no de-duplication of identical
  images across advertisements.
* Jalali→Gregorian conversion uses a fixed +621 offset, correct at year
  granularity but not for exact dates.
* Detail parsing depends on the `__NUXT_DATA__` structure. If Bama changes its
  frontend, the parser degrades to JSON-LD + DOM and marks records `partial`
  rather than failing silently — but coverage will drop until updated.

---

## 14. Debugging

```bash
# Watch the browser work
python -m bama_scraper discover --mode scroll --headed --log-level DEBUG

# Force the browser path even though the API works
python -m bama_scraper discover --mode scroll

# Audit stored data without touching the network
python -m bama_scraper validate --skip-second-pass

# Inspect a stored record
sqlite3 output/bama_ads.sqlite \
  "SELECT json_extract(payload,'\$.title'), json_extract(payload,'\$.price_toman')
   FROM ad_details LIMIT 5;"

# Read a saved failed parse
gunzip -c output/debug/<ad_id>.html.gz | less
```

`output/debug/` holds `inspect_*.png` / `.html` and gzipped HTML for every
failed or ambiguous parse.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Executable doesn't exist … chromium` | Browser not installed | `playwright install chromium` |
| `HTTP 400 PageSize is not Valid` | `api_page_size > 30` | Keep it at 30 or below |
| Discovery returns 0 ads | Filters produced an empty set, or the endpoint changed | Run `inspect` and compare `listing_endpoints` |
| `blocked_http_403` / `429` | The site is rate-limiting | Stop, wait, lower `--concurrency`, raise delays. Do **not** attempt to evade |
| Every `published_ts` is null | Site introduced a new relative-date phrasing | Extend `_JUST_NOW_MARKERS` / `_REL_UNITS` |
| Parquet export skipped | `pyarrow` missing | `pip install pyarrow` (the run still succeeds; the error is recorded in the manifest) |
| Persian shows as `????` in Excel | Opened the wrong file | Use `bama_ads.csv` — it carries a UTF-8 BOM |
| Run seems stuck | Delays plus 2849 pages ≈ 40 min | Watch the `details completed=…` progress line |

---

## 16. Testing

```bash
pytest -q                      # full suite
pytest tests/test_termination.py -q   # browser integration (local server only)
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/bama_scraper
```

Tests never depend on the live website. Fixtures in `tests/fixtures/` are
sanitized captures (phone numbers removed at fixture-build time). The browser
integration test serves a **local** synthetic page that reproduces virtualized
infinite scroll, so the scroll algorithm — including the virtualization and
stale-termination behaviour — is verified end-to-end offline.
