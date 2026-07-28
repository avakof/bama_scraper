-- Production provenance, daily detail checking, and the genuine-history boundary.
--
-- Three problems this migration fixes, all of which let a report claim more than
-- the data supports:
--
-- 1. A run had no way to say HOW it was triggered. A manual run and a scheduled
--    run were indistinguishable, so a manual execution could occupy — and appear
--    to satisfy — a production scheduled slot.
-- 2. There was no record that an advertisement had been *checked* on a given day.
--    Snapshots are written only on change, so "no snapshot" was ambiguous between
--    "unchanged" and "never looked at".
-- 3. Nothing marked where genuine scheduled history begins, so pre-deployment
--    test runs could be counted as real daily observations.

-- ------------------------------------------------------------ run provenance
-- How the run was triggered. Only `scheduled` and `catch_up` form genuine daily
-- history; everything else is real work that must not be mistaken for it.
--   scheduled        fired by the installed scheduler at its slot
--   catch_up         fired late for a missed slot, inside the grace window
--   manual           a human ran it; never owns a slot
--   deployment_test  proving the plumbing works
--   backfill         reconstructing a range deliberately
--   simulation       synthetic data for testing
ALTER TABLE monitoring_runs ADD COLUMN trigger_type {{TEXT}} NOT NULL DEFAULT 'manual';

-- TRUE when this run's attribution to its slot is not a real observation of that
-- slot. Excluded from history and EDA by default.
ALTER TABLE monitoring_runs ADD COLUMN is_synthetic {{BOOL}} NOT NULL DEFAULT {{FALSE}};

-- The slot in Paris terms, denormalised so a partition label never has to be
-- recomputed from a UTC instant by a reader who may not know the rule.
ALTER TABLE monitoring_runs ADD COLUMN scheduled_date_paris {{TEXT}};
ALTER TABLE monitoring_runs ADD COLUMN scheduled_hour_paris {{TEXT}};

-- Which machine and process produced this run.
ALTER TABLE monitoring_runs ADD COLUMN host_name {{TEXT}};
ALTER TABLE monitoring_runs ADD COLUMN process_id {{INT}};
ALTER TABLE monitoring_runs ADD COLUMN scheduler_instance_id {{TEXT}};

-- Names the production schedule a slot belongs to, so a second schedule (a
-- different search, a different hour) cannot collide with this one.
ALTER TABLE monitoring_runs ADD COLUMN production_schedule_name {{TEXT}};

-- Phase boundaries. A daily snapshot is not one instant: discovery and detail
-- scraping take tens of minutes, and labelling every row 13:00 would be a lie.
ALTER TABLE monitoring_runs ADD COLUMN discovery_started_at {{TS}};
ALTER TABLE monitoring_runs ADD COLUMN discovery_finished_at {{TS}};
ALTER TABLE monitoring_runs ADD COLUMN detail_started_at {{TS}};
ALTER TABLE monitoring_runs ADD COLUMN detail_finished_at {{TS}};

-- Detail-coverage accounting. `detail_accounted_count` must equal
-- `discovered_count` for a run to be valid.
ALTER TABLE monitoring_runs ADD COLUMN detail_accounted_count {{INT}} NOT NULL DEFAULT 0;
ALTER TABLE monitoring_runs ADD COLUMN detail_gone_count {{INT}} NOT NULL DEFAULT 0;
ALTER TABLE monitoring_runs ADD COLUMN detail_permanent_failure_count {{INT}} NOT NULL DEFAULT 0;
ALTER TABLE monitoring_runs ADD COLUMN detail_retryable_failure_count {{INT}} NOT NULL DEFAULT 0;
ALTER TABLE monitoring_runs ADD COLUMN detail_coverage_rate {{REAL}};

-- Where the exported daily snapshot for this run landed.
ALTER TABLE monitoring_runs ADD COLUMN snapshot_export_path {{TEXT}};

CREATE INDEX IF NOT EXISTS idx_runs_trigger ON monitoring_runs(trigger_type);
CREATE INDEX IF NOT EXISTS idx_runs_genuine ON monitoring_runs(is_synthetic, trigger_type);

-- One genuine production execution per (schedule, slot). Partial, so a manual
-- run, a deployment test and a simulation can all share a slot value without
-- colliding with the production run or with each other — which is exactly the
-- confusion that let a manual run look like the day's scheduled execution.
CREATE UNIQUE INDEX IF NOT EXISTS uq_production_slot
    ON monitoring_runs (production_schedule_name, configuration_hash, scheduled_for)
    WHERE trigger_type IN ('scheduled', 'catch_up') AND is_synthetic = {{FALSE}};

-- ------------------------------------------------------ daily detail checks
-- One row per (run, advertisement): PROOF that the advertisement was looked at
-- on that day. Distinct from a snapshot, which is written only when the content
-- changed — so without this table, "no snapshot today" cannot be told apart from
-- "never checked today".
CREATE TABLE IF NOT EXISTS advertisement_detail_checks (
    id                  {{PK}},
    run_id              {{FK}} NOT NULL REFERENCES monitoring_runs(id),
    advertisement_id    {{FK}} NOT NULL REFERENCES advertisements(id),
    checked_at          {{TS}} NOT NULL,
    detail_http_status  {{INT}},
    -- available | gone | reports_unavailable | temporarily_unavailable
    -- | blocked | outside_filter | unknown
    detail_availability {{TEXT}},
    -- ok | parse_failed | not_attempted
    parser_status       {{TEXT}},
    content_hash        {{TEXT}},
    -- The immutable snapshot this check's content corresponds to. When content is
    -- unchanged the check points at the EXISTING snapshot rather than duplicating
    -- a multi-kilobyte payload.
    snapshot_id         {{FK}} REFERENCES advertisement_snapshots(id),
    content_changed     {{BOOL}} NOT NULL DEFAULT {{FALSE}},
    attempt_count       {{INT}} NOT NULL DEFAULT 1,
    duration_ms         {{INT}},
    -- completed | gone | permanent_failure | retryable_failure | skipped
    outcome             {{TEXT}} NOT NULL,
    error_type          {{TEXT}},
    error_message       {{TEXT}},
    created_at          {{TS}} NOT NULL,
    -- Replaying a run cannot double-count a check.
    CONSTRAINT uq_detail_check_run_ad UNIQUE (run_id, advertisement_id)
);
CREATE INDEX IF NOT EXISTS idx_detail_checks_run ON advertisement_detail_checks(run_id);
CREATE INDEX IF NOT EXISTS idx_detail_checks_ad ON advertisement_detail_checks(advertisement_id);
CREATE INDEX IF NOT EXISTS idx_detail_checks_outcome ON advertisement_detail_checks(outcome);

-- --------------------------------------------------- genuine history boundary
-- Where real scheduled observation starts. Everything before it is real work but
-- not daily history, and reports must say so rather than quietly counting it.
CREATE TABLE IF NOT EXISTS production_schedule (
    name                         {{TEXT}} PRIMARY KEY,
    search_configuration_hash    {{TEXT}} NOT NULL,
    timezone                     {{TEXT}} NOT NULL,
    run_at_local                 {{TEXT}} NOT NULL,
    production_schedule_started_at {{TS}},
    first_genuine_scheduled_run_id {{FK}} REFERENCES monitoring_runs(id),
    installed_by                 {{TEXT}},
    host_name                    {{TEXT}},
    notes                        {{TEXT}},
    created_at                   {{TS}} NOT NULL,
    updated_at                   {{TS}} NOT NULL
);

-- ------------------------------------------------------- missed-slot ledger
-- A slot the machine was not available for. Recorded rather than fabricated: the
-- absence of an observation is itself an observation about the pipeline.
CREATE TABLE IF NOT EXISTS missed_schedule_slots (
    id                  {{PK}},
    production_schedule_name {{TEXT}},
    scheduled_for       {{TS}} NOT NULL,
    scheduled_date_paris {{TEXT}},
    detected_at         {{TS}} NOT NULL,
    minutes_late        {{REAL}},
    -- caught_up | outside_grace_window | machine_unavailable
    resolution          {{TEXT}} NOT NULL,
    catch_up_run_id     {{FK}} REFERENCES monitoring_runs(id),
    note                {{TEXT}},
    created_at          {{TS}} NOT NULL,
    CONSTRAINT uq_missed_slot UNIQUE (production_schedule_name, scheduled_for)
);
CREATE INDEX IF NOT EXISTS idx_missed_slots ON missed_schedule_slots(scheduled_for);
