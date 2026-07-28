-- PostgreSQL: make monitoring_runs.scheduled_for nullable.
--
-- A manual, test or backfill run has no scheduled slot, and NULL says that
-- honestly. One statement here; SQLite needs a full table rebuild, which is why
-- this migration is dialect-suffixed.

ALTER TABLE monitoring_runs ALTER COLUMN scheduled_for DROP NOT NULL;
