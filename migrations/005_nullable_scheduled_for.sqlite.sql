-- SQLite: make monitoring_runs.scheduled_for nullable.
--
-- A manual, test or backfill run has no scheduled slot, and saying so with NULL
-- is more honest than inventing an attribution. The column was declared NOT NULL
-- back when every run was a scheduled run.
--
-- SQLite cannot ALTER a column's nullability, so the table is rebuilt: create,
-- copy, drop, rename. The migration runner disables foreign keys around any
-- script containing RENAME TO and re-checks integrity afterwards.

CREATE TABLE monitoring_runs_nullable (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    search_url            TEXT NOT NULL,
    started_at            TEXT,
    finished_at           TEXT,
    scheduled_for         TEXT,
    timezone              TEXT NOT NULL,
    status                TEXT NOT NULL,
    termination_reason    TEXT,
    health_reason         TEXT,
    discovered_count      INTEGER DEFAULT 0,
    new_count             INTEGER DEFAULT 0,
    active_count          INTEGER DEFAULT 0,
    missing_count         INTEGER DEFAULT 0,
    removed_count         INTEGER DEFAULT 0,
    reappeared_count      INTEGER DEFAULT 0,
    reposted_count        INTEGER DEFAULT 0,
    likely_sold_count     INTEGER DEFAULT 0,
    detail_success_count  INTEGER DEFAULT 0,
    detail_failure_count  INTEGER DEFAULT 0,
    duplicate_count       INTEGER DEFAULT 0,
    error_count           INTEGER DEFAULT 0,
    scraper_version       TEXT,
    monitor_version       TEXT,
    configuration_hash    TEXT NOT NULL,
    previous_valid_run_id INTEGER,
    comparison_applied    INTEGER DEFAULT FALSE,
    evidence_path         TEXT,
    created_at            TEXT NOT NULL, trigger_type TEXT NOT NULL DEFAULT 'manual', is_synthetic INTEGER NOT NULL DEFAULT 0, scheduled_date_paris TEXT, scheduled_hour_paris TEXT, host_name TEXT, process_id INTEGER, scheduler_instance_id TEXT, production_schedule_name TEXT, discovery_started_at TEXT, discovery_finished_at TEXT, detail_started_at TEXT, detail_finished_at TEXT, detail_accounted_count INTEGER NOT NULL DEFAULT 0, detail_gone_count INTEGER NOT NULL DEFAULT 0, detail_permanent_failure_count INTEGER NOT NULL DEFAULT 0, detail_retryable_failure_count INTEGER NOT NULL DEFAULT 0, detail_coverage_rate REAL, snapshot_export_path TEXT
);

INSERT INTO monitoring_runs_nullable (id, search_url, started_at, finished_at, scheduled_for, timezone, status, termination_reason, health_reason, discovered_count, new_count, active_count, missing_count, removed_count, reappeared_count, reposted_count, likely_sold_count, detail_success_count, detail_failure_count, duplicate_count, error_count, scraper_version, monitor_version, configuration_hash, previous_valid_run_id, comparison_applied, evidence_path, created_at, trigger_type, is_synthetic, scheduled_date_paris, scheduled_hour_paris, host_name, process_id, scheduler_instance_id, production_schedule_name, discovery_started_at, discovery_finished_at, detail_started_at, detail_finished_at, detail_accounted_count, detail_gone_count, detail_permanent_failure_count, detail_retryable_failure_count, detail_coverage_rate, snapshot_export_path)
    SELECT id, search_url, started_at, finished_at, scheduled_for, timezone, status, termination_reason, health_reason, discovered_count, new_count, active_count, missing_count, removed_count, reappeared_count, reposted_count, likely_sold_count, detail_success_count, detail_failure_count, duplicate_count, error_count, scraper_version, monitor_version, configuration_hash, previous_valid_run_id, comparison_applied, evidence_path, created_at, trigger_type, is_synthetic, scheduled_date_paris, scheduled_hour_paris, host_name, process_id, scheduler_instance_id, production_schedule_name, discovery_started_at, discovery_finished_at, detail_started_at, detail_finished_at, detail_accounted_count, detail_gone_count, detail_permanent_failure_count, detail_retryable_failure_count, detail_coverage_rate, snapshot_export_path FROM monitoring_runs;

DROP TABLE monitoring_runs;
ALTER TABLE monitoring_runs_nullable RENAME TO monitoring_runs;

CREATE INDEX IF NOT EXISTS idx_runs_status ON monitoring_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_scheduled ON monitoring_runs(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_runs_trigger ON monitoring_runs(trigger_type);
CREATE INDEX IF NOT EXISTS idx_runs_genuine ON monitoring_runs(is_synthetic, trigger_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_production_slot
    ON monitoring_runs (production_schedule_name, configuration_hash, scheduled_for)
    WHERE trigger_type IN ('scheduled', 'catch_up') AND is_synthetic = 0;
