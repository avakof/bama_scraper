-- Longitudinal listing-monitoring schema.
--
-- Portable across PostgreSQL and SQLite via doubled-brace type placeholders
-- (see TYPE_TOKENS) that src/bama_monitor/db.py resolves per dialect.
--
-- Design rule enforced by this schema: observed facts and inferred outcomes live
-- in different columns. `was_seen`, `consecutive_misses` and the event log record
-- what happened; `sale_confidence` and `sale_label` record what is inferred from
-- it, and nothing in the schema lets absence alone be written as a sale.

-- ---------------------------------------------------------------- runs
CREATE TABLE IF NOT EXISTS monitoring_runs (
    id                    {{PK}},
    search_url            {{TEXT}} NOT NULL,
    started_at            {{TS}},
    finished_at           {{TS}},
    scheduled_for         {{TS}} NOT NULL,
    timezone              {{TEXT}} NOT NULL,
    status                {{TEXT}} NOT NULL,
    termination_reason    {{TEXT}},
    health_reason         {{TEXT}},
    discovered_count      {{INT}} DEFAULT 0,
    new_count             {{INT}} DEFAULT 0,
    active_count          {{INT}} DEFAULT 0,
    missing_count         {{INT}} DEFAULT 0,
    removed_count         {{INT}} DEFAULT 0,
    reappeared_count      {{INT}} DEFAULT 0,
    reposted_count        {{INT}} DEFAULT 0,
    likely_sold_count     {{INT}} DEFAULT 0,
    detail_success_count  {{INT}} DEFAULT 0,
    detail_failure_count  {{INT}} DEFAULT 0,
    duplicate_count       {{INT}} DEFAULT 0,
    error_count           {{INT}} DEFAULT 0,
    scraper_version       {{TEXT}},
    monitor_version       {{TEXT}},
    configuration_hash    {{TEXT}} NOT NULL,
    previous_valid_run_id {{FK}},
    comparison_applied    {{BOOL}} DEFAULT FALSE,
    evidence_path         {{TEXT}},
    created_at            {{TS}} NOT NULL,
    -- Idempotency: one run per (scheduled slot, search configuration). Re-running
    -- the same slot reuses the row instead of creating a parallel history.
    CONSTRAINT uq_runs_slot UNIQUE (scheduled_for, configuration_hash)
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON monitoring_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_scheduled ON monitoring_runs(scheduled_for);

-- ------------------------------------------------------- advertisements
-- One durable row per platform advertisement id. An advertisement id is NOT a
-- vehicle: `vehicle_entity_id` groups ids believed to describe the same car.
CREATE TABLE IF NOT EXISTS advertisements (
    id                    {{PK}},
    platform              {{TEXT}} NOT NULL DEFAULT 'bama',
    platform_ad_id        {{TEXT}} NOT NULL,
    canonical_url         {{TEXT}} NOT NULL,
    current_status        {{TEXT}} NOT NULL,
    first_seen_at         {{TS}},
    last_seen_at          {{TS}},
    first_missing_at      {{TS}},
    confirmed_removed_at  {{TS}},
    reappeared_at         {{TS}},
    suspected_sold_at     {{TS}},
    consecutive_misses    {{INT}} NOT NULL DEFAULT 0,
    total_seen_runs       {{INT}} NOT NULL DEFAULT 0,
    total_missing_runs    {{INT}} NOT NULL DEFAULT 0,
    first_run_id          {{FK}},
    last_seen_run_id      {{FK}},
    last_checked_run_id   {{FK}},
    latest_snapshot_id    {{FK}},
    repost_parent_ad_id   {{FK}},
    vehicle_entity_id     {{TEXT}},
    vehicle_fingerprint   {{TEXT}},
    -- Inference, not observation. 0.0-1.0, with `sale_label` as its bucket.
    sale_confidence       {{REAL}} DEFAULT 0.0,
    sale_label            {{TEXT}} DEFAULT 'unknown',
    last_detail_verdict   {{TEXT}},
    last_detail_checked_at {{TS}},
    detail_failure_count  {{INT}} NOT NULL DEFAULT 0,
    search_configuration_hash {{TEXT}},
    created_at            {{TS}} NOT NULL,
    updated_at            {{TS}} NOT NULL,
    CONSTRAINT uq_ads_platform_id UNIQUE (platform, platform_ad_id),
    CONSTRAINT uq_ads_canonical_url UNIQUE (canonical_url)
);
CREATE INDEX IF NOT EXISTS idx_ads_status ON advertisements(current_status);
CREATE INDEX IF NOT EXISTS idx_ads_last_seen ON advertisements(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_ads_vehicle ON advertisements(vehicle_entity_id);
CREATE INDEX IF NOT EXISTS idx_ads_fingerprint ON advertisements(vehicle_fingerprint);

-- -------------------------------------------------- daily observations
-- Append-only. One row per advertisement per valid run, including rows with
-- was_seen = FALSE, so absence is a recorded observation rather than an inferred
-- gap in the data.
CREATE TABLE IF NOT EXISTS daily_ad_observations (
    id                      {{PK}},
    run_id                  {{FK}} NOT NULL REFERENCES monitoring_runs(id),
    advertisement_id        {{FK}} NOT NULL REFERENCES advertisements(id),
    was_seen                {{BOOL}} NOT NULL,
    position                {{INT}},
    page_number             {{INT}},
    scroll_cycle            {{INT}},
    card_title              {{TEXT}},
    card_price_raw          {{TEXT}},
    card_price_normalized   {{INT}},
    card_year               {{TEXT}},
    card_mileage_raw        {{TEXT}},
    card_mileage_normalized {{INT}},
    card_location           {{TEXT}},
    card_image_url          {{TEXT}},
    card_hash               {{TEXT}},
    observed_at             {{TS}} NOT NULL,
    CONSTRAINT uq_obs_run_ad UNIQUE (run_id, advertisement_id)
);
CREATE INDEX IF NOT EXISTS idx_obs_ad ON daily_ad_observations(advertisement_id);
CREATE INDEX IF NOT EXISTS idx_obs_seen ON daily_ad_observations(run_id, was_seen);

-- ------------------------------------------------ detail snapshots
-- Versioned detail-page content. A new row is written only when something
-- actually changed, so `content_hash` history is a genuine change log.
CREATE TABLE IF NOT EXISTS advertisement_snapshots (
    id                       {{PK}},
    advertisement_id         {{FK}} NOT NULL REFERENCES advertisements(id),
    run_id                   {{FK}} REFERENCES monitoring_runs(id),
    scraped_at               {{TS}} NOT NULL,
    title                    {{TEXT}},
    brand                    {{TEXT}},
    model                    {{TEXT}},
    trim                     {{TEXT}},
    year                     {{TEXT}},
    price_raw                {{TEXT}},
    price_normalized         {{INT}},
    mileage_raw              {{TEXT}},
    mileage_normalized       {{INT}},
    description              {{TEXT}},
    seller_type              {{TEXT}},
    seller_name              {{TEXT}},
    city                     {{TEXT}},
    province                 {{TEXT}},
    body_color               {{TEXT}},
    interior_color           {{TEXT}},
    transmission             {{TEXT}},
    fuel_type                {{TEXT}},
    engine                   {{TEXT}},
    body_condition           {{TEXT}},
    chassis_condition        {{TEXT}},
    insurance                {{TEXT}},
    published_at_raw         {{TEXT}},
    published_at_normalized  {{TS}},
    attributes_json          {{JSON}},
    media_json               {{JSON}},
    content_hash             {{TEXT}},
    detail_page_status       {{TEXT}},
    parser_version           {{TEXT}},
    created_at               {{TS}} NOT NULL,
    -- Idempotency: the same content in the same run cannot be stored twice.
    CONSTRAINT uq_snapshot_run_hash UNIQUE (advertisement_id, run_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_snap_ad ON advertisement_snapshots(advertisement_id, scraped_at);

-- ------------------------------------------------------------ events
-- Immutable audit trail. Every status transition writes one row, with the
-- evidence that justified it, so any classification can be reconstructed.
CREATE TABLE IF NOT EXISTS advertisement_events (
    id                {{PK}},
    advertisement_id  {{FK}} NOT NULL REFERENCES advertisements(id),
    run_id            {{FK}} REFERENCES monitoring_runs(id),
    event_type        {{TEXT}} NOT NULL,
    previous_status   {{TEXT}},
    new_status        {{TEXT}},
    event_at          {{TS}} NOT NULL,
    evidence_json     {{JSON}},
    confidence        {{REAL}},
    created_at        {{TS}} NOT NULL,
    -- Idempotency key: replaying a run cannot duplicate its events.
    CONSTRAINT uq_event_idem UNIQUE (advertisement_id, run_id, event_type)
);
CREATE INDEX IF NOT EXISTS idx_events_ad ON advertisement_events(advertisement_id, event_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON advertisement_events(event_type);

-- ------------------------------------------------------- price history
CREATE TABLE IF NOT EXISTS price_changes (
    id                {{PK}},
    advertisement_id  {{FK}} NOT NULL REFERENCES advertisements(id),
    run_id            {{FK}} REFERENCES monitoring_runs(id),
    change_type       {{TEXT}} NOT NULL,
    old_price         {{INT}},
    new_price         {{INT}},
    absolute_change   {{INT}},
    percentage_change {{REAL}},
    old_price_raw     {{TEXT}},
    new_price_raw     {{TEXT}},
    changed_at        {{TS}} NOT NULL,
    created_at        {{TS}} NOT NULL,
    CONSTRAINT uq_price_change UNIQUE (advertisement_id, run_id, change_type)
);
CREATE INDEX IF NOT EXISTS idx_price_ad ON price_changes(advertisement_id, changed_at);

-- ------------------------------------------------ detail verification
-- Raw evidence for each check of a missing advertisement's detail page. Stored
-- separately from the inference so the verdict can be re-scored later.
CREATE TABLE IF NOT EXISTS detail_verifications (
    id                {{PK}},
    advertisement_id  {{FK}} NOT NULL REFERENCES advertisements(id),
    run_id            {{FK}} REFERENCES monitoring_runs(id),
    url               {{TEXT}},
    verdict           {{TEXT}} NOT NULL,
    http_status       {{INT}},
    evidence_json     {{JSON}},
    checked_at        {{TS}} NOT NULL,
    CONSTRAINT uq_verify_run_ad UNIQUE (advertisement_id, run_id)
);

-- ----------------------------------------------------- repost linkage
CREATE TABLE IF NOT EXISTS repost_links (
    id                 {{PK}},
    parent_ad_id       {{FK}} NOT NULL REFERENCES advertisements(id),
    child_ad_id        {{FK}} NOT NULL REFERENCES advertisements(id),
    run_id             {{FK}} REFERENCES monitoring_runs(id),
    score              {{REAL}} NOT NULL,
    matched_on_json    {{JSON}},
    evidence_json      {{JSON}},
    vehicle_entity_id  {{TEXT}},
    created_at         {{TS}} NOT NULL,
    CONSTRAINT uq_repost_pair UNIQUE (parent_ad_id, child_ad_id)
);

-- ------------------------------------------------------------ errors
CREATE TABLE IF NOT EXISTS scrape_errors (
    id                {{PK}},
    run_id            {{FK}} REFERENCES monitoring_runs(id),
    advertisement_id  {{FK}} REFERENCES advertisements(id),
    stage             {{TEXT}},
    url               {{TEXT}},
    error_type        {{TEXT}},
    error_message     {{TEXT}},
    retryable         {{BOOL}},
    attempt           {{INT}},
    screenshot_path   {{TEXT}},
    html_path         {{TEXT}},
    created_at        {{TS}} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_run ON scrape_errors(run_id);

-- ------------------------------------------------------------ alerts
CREATE TABLE IF NOT EXISTS alerts (
    id             {{PK}},
    run_id         {{FK}} REFERENCES monitoring_runs(id),
    severity       {{TEXT}} NOT NULL,
    alert_type     {{TEXT}} NOT NULL,
    message        {{TEXT}} NOT NULL,
    metadata_json  {{JSON}},
    created_at     {{TS}} NOT NULL,
    resolved_at    {{TS}}
);
CREATE INDEX IF NOT EXISTS idx_alerts_open ON alerts(resolved_at);

-- ------------------------------------------------------------- locks
-- Database-level mutual exclusion. Used together with a filesystem lock and,
-- on PostgreSQL, a session advisory lock: three independent layers, because a
-- second concurrent run would corrupt the miss counters.
CREATE TABLE IF NOT EXISTS run_locks (
    lock_name    {{TEXT}} PRIMARY KEY,
    holder       {{TEXT}} NOT NULL,
    run_id       {{FK}},
    acquired_at  {{TS}} NOT NULL,
    expires_at   {{TS}} NOT NULL,
    heartbeat_at {{TS}}
);
