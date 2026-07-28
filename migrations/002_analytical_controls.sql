-- Analytical controls: left truncation, filter exit, availability separation
-- and the vehicle grain.
--
-- Motivation for each block is recorded here because these columns exist to stop
-- specific *analytical* errors, not to store new observations.

-- ---------------------------------------------------------------- publication
-- `first_seen_at` is when MONITORING first saw the listing. `published_at` is
-- when the SELLER published it. For any advertisement that already existed on
-- the first monitoring day these differ by an unknown amount, and treating the
-- former as the latter understates time-on-market without bound.
ALTER TABLE advertisements ADD COLUMN published_at {{TS}};
-- Where the publication time came from and how far it can be trusted:
--   'detail_absolute'  parsed from an absolute Jalali date        -> reliable
--   'detail_relative'  parsed from a relative phrase (روز پیش)     -> reliable
--   'detail_coarse'    only a coarse phrase was available          -> not reliable
--   NULL               never obtained
ALTER TABLE advertisements ADD COLUMN published_at_source {{TEXT}};
ALTER TABLE advertisements ADD COLUMN published_at_reliable {{BOOL}} NOT NULL DEFAULT {{FALSE}};

-- TRUE when the listing was already active at its first observation, i.e. its
-- prior lifetime is unknown. Such rows are LEFT-TRUNCATED (delayed entry) and
-- must not be mixed into time-on-market rankings without accounting for it.
ALTER TABLE advertisements ADD COLUMN left_truncated {{BOOL}} NOT NULL DEFAULT {{TRUE}};
-- Cached result of the eligibility rule: publication time known AND the listing
-- entered observation close enough to publication that its full life was seen.
ALTER TABLE advertisements ADD COLUMN eligible_for_duration_ranking {{BOOL}} NOT NULL DEFAULT {{FALSE}};
-- Seconds between publication and first observation. 0 for a listing first seen
-- on the day it was published; large for a baseline listing.
ALTER TABLE advertisements ADD COLUMN entry_delay_seconds {{REAL}};

CREATE INDEX IF NOT EXISTS idx_ads_published ON advertisements(published_at);
CREATE INDEX IF NOT EXISTS idx_ads_rankable ON advertisements(eligible_for_duration_ranking);

-- --------------------------------------------------------- detail availability
-- Deliberately separate from `current_status` (observation) and `sale_label`
-- (inference). A 410 says the page is gone; it does not say why, and it is not a
-- status transition on its own.
--   available | gone | reports_unavailable | temporarily_unavailable
--   | blocked | outside_filter | unknown
ALTER TABLE advertisements ADD COLUMN detail_availability {{TEXT}};

-- --------------------------------------------------------------- filter exit
-- A listing can leave the monitored search while remaining perfectly alive: the
-- seller drops the price under `priceFrom`, edits the year, or the category
-- changes. That is NOT a disappearance from the market and must not accumulate
-- misses.
--   price_below_filter | price_above_filter | year_outside_filter
--   | category_changed | country_classification_changed
--   | search_index_inconsistency
ALTER TABLE advertisements ADD COLUMN filter_exit_reason {{TEXT}};
ALTER TABLE advertisements ADD COLUMN filter_exit_at {{TS}};
ALTER TABLE advertisements ADD COLUMN filter_exit_evidence_json {{JSON}};
CREATE INDEX IF NOT EXISTS idx_ads_filter_exit ON advertisements(filter_exit_reason);

-- ------------------------------------------------------------- vehicle grain
-- An advertisement id identifies a LISTING. Reposts mean several listings can
-- describe one physical vehicle, and a duration measured per listing restarts
-- at every repost. Vehicle-level rows carry the span across linked listings.
CREATE TABLE IF NOT EXISTS vehicle_entities (
    vehicle_entity_id   {{TEXT}} PRIMARY KEY,
    first_ad_id         {{FK}} REFERENCES advertisements(id),
    listing_count       {{INT}} NOT NULL DEFAULT 1,
    repost_count        {{INT}} NOT NULL DEFAULT 0,
    vehicle_first_seen_at {{TS}},
    vehicle_last_seen_at  {{TS}},
    vehicle_first_missing_at {{TS}},
    vehicle_published_at  {{TS}},
    current_status      {{TEXT}},
    -- Inference carried at the vehicle grain too, so a repost chain is not
    -- counted as several sales.
    sale_evidence_score {{REAL}} DEFAULT 0.0,
    sale_label          {{TEXT}} DEFAULT 'unknown',
    left_truncated      {{BOOL}} NOT NULL DEFAULT {{TRUE}},
    created_at          {{TS}} NOT NULL,
    updated_at          {{TS}} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vehicles_status ON vehicle_entities(current_status);

-- ------------------------------------------------------- validation sampling
-- Manual ground-truth labelling. Sale confidence is a heuristic evidence score
-- until it has been checked against real outcomes; this table is where those
-- checks live.
CREATE TABLE IF NOT EXISTS sale_validation_samples (
    id                  {{PK}},
    advertisement_id    {{FK}} NOT NULL REFERENCES advertisements(id),
    sampled_at          {{TS}} NOT NULL,
    sampled_run_id      {{FK}} REFERENCES monitoring_runs(id),
    predicted_score     {{REAL}},
    predicted_label     {{TEXT}},
    -- Filled in by a human: sold | not_sold | withdrawn | expired | reposted
    -- | outside_filter | unknown
    observed_outcome    {{TEXT}},
    outcome_source      {{TEXT}},
    labelled_at         {{TS}},
    labelled_by         {{TEXT}},
    notes               {{TEXT}},
    CONSTRAINT uq_validation_sample UNIQUE (advertisement_id, sampled_run_id)
);
CREATE INDEX IF NOT EXISTS idx_validation_outcome ON sale_validation_samples(observed_outcome);
