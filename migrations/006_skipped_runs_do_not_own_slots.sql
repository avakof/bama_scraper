-- A run skipped for lock contention must not own the slot it was aiming at.
--
-- The scenario is concrete on this deployment. A full daily detail census of
-- ~2,850 advertisements runs for roughly two hours at the configured politeness.
-- If one day's run stalls, the next day's 13:00 trigger finds the execution lock
-- still held, declines, and records `skipped_due_to_existing_run`.
--
-- That row is evidence that the trigger fired. It is not an observation of the
-- market: no page was fetched, no advertisement was checked, nothing was compared.
-- Two things follow, and migration 003 got both wrong:
--
--   1. It was written with `trigger_type='manual'`, because `skipped_run` never
--      passed the real trigger through. A scheduled trigger that fired and was
--      declined then looked, forever after, like somebody running the pipeline by
--      hand. That is exactly the kind of provenance laundering the rest of this
--      schema exists to prevent.
--
--   2. With the truthful trigger restored, the row would occupy `uq_production_slot`
--      and permanently block the slot from ever being filled by a run that actually
--      observed something. A declined attempt would masquerade as the day's data.
--
-- So the index now excludes skipped runs. The slot stays unfulfilled, startup
-- reconciliation still sees it as needing a catch-up, several declined attempts for
-- one slot are all recordable, and the skip keeps its real trigger type.
--
-- Index-only change: both dialects support DROP INDEX / CREATE INDEX directly, so
-- no table rebuild is involved and no row is rewritten.

DROP INDEX IF EXISTS uq_production_slot;

CREATE UNIQUE INDEX IF NOT EXISTS uq_production_slot
    ON monitoring_runs (production_schedule_name, configuration_hash, scheduled_for)
    WHERE trigger_type IN ('scheduled', 'catch_up')
      AND is_synthetic = {{FALSE}}
      AND status <> 'skipped_due_to_existing_run';
