"""Dataset-level data dictionary.

``schema_inspection`` documents the *database*. This documents the *analytical
datasets* — the columns an analyst actually reads — including the derived ones
that exist nowhere in the schema and are therefore the easiest to misread.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .models import AnalyticalRole

#: Explicit descriptions for derived columns. Anything derived and undocumented is
#: an invitation to guess, and the guesses are usually the dangerous readings.
DERIVED_COLUMNS: dict[str, tuple[str, AnalyticalRole]] = {
    "price_toman": (
        "Asking price in toman, taken from the detail snapshot when one exists and "
        "from the search card otherwise. See price_source. NOT a transaction price.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "price_source": (
        "Which source supplied price_toman: detail_snapshot or search_card.",
        AnalyticalRole.PROVENANCE,
    ),
    "log10_price": (
        "log10 of price_toman where positive, else NaN. Used because asking prices "
        "are strongly right-skewed.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "mileage_km": (
        "Odometer reading in km; 0 means a zero-kilometre car.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "year_jalali": (
        "Production year on the Jalali calendar. Values published as Gregorian were "
        "converted; see year_calendar_source.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "year_calendar_source": (
        "jalali_as_published or gregorian_converted (-621 applied).",
        AnalyticalRole.PROVENANCE,
    ),
    "snapshot_scraped_at": (
        "When the joined detail snapshot was scraped.",
        AnalyticalRole.TIMESTAMP,
    ),
    "snapshot_age_hours": (
        "Hours between the snapshot and the analysis run's reference instant. "
        "Detail attributes are NOT contemporaneous with the search observation.",
        AnalyticalRole.PROVENANCE,
    ),
    "snapshot_is_stale": (
        "True when snapshot_age_hours exceeds the configured threshold.",
        AnalyticalRole.PROVENANCE,
    ),
    "listing_age_days_observed": (
        "Days from first observation to the run. A LOWER BOUND on time on market for "
        "any left-truncated listing.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "censoring_class": (
        "fully_observed | interval_censored | right_censored | left_truncated | "
        "active_outside_filter | reappeared | reposted | unknown.",
        AnalyticalRole.PROVENANCE,
    ),
    "observed_monitoring_duration": (
        "Days from FIRST OBSERVATION to last sighting. Defined for every listing; a "
        "lower bound where the listing predates monitoring.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "estimated_market_duration": (
        "Days from PUBLICATION to last sighting. Populated only where the whole "
        "market life was observed; NaN otherwise, by design.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "event_observed": (
        "1 = disappearance from the observed inventory was observed. NOT a confirmed sale.",
        AnalyticalRole.OUTCOME_OBSERVATION,
    ),
    "sale_evidence_score": (
        "Heuristic additive score in [0,1]. An ORDERING of evidence, not a probability.",
        AnalyticalRole.HEURISTIC_INFERENCE,
    ),
    "sale_confidence": (
        "The monitor's stored heuristic score; exported as sale_evidence_score. Not a probability.",
        AnalyticalRole.HEURISTIC_INFERENCE,
    ),
    "left_truncated": (
        "True when the listing was already active at first observation, so its earlier "
        "life was never seen.",
        AnalyticalRole.PROVENANCE,
    ),
    "eligible_for_duration_ranking": (
        "True only when publication time is reliable AND first observation was close "
        "enough to publication. Gates estimated_market_duration.",
        AnalyticalRole.PROVENANCE,
    ),
    "was_seen_trusted": (
        "was_seen, blanked for observations from non-valid runs. A failed scrape did "
        "not observe an absence.",
        AnalyticalRole.OUTCOME_OBSERVATION,
    ),
    "observation_run_id": (
        "The run this observation belongs to (panel grain).",
        AnalyticalRole.IDENTIFIER,
    ),
    "run_id": (
        "The ANALYSIS run this dataset describes (provenance stamp).",
        AnalyticalRole.PROVENANCE,
    ),
    "vehicle_entity_id": (
        "Identifier for a PHYSICAL VEHICLE across relistings. Not an advertisement id.",
        AnalyticalRole.IDENTIFIER,
    ),
    "media_count": (
        "Number of images/videos attached to the listing.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
    "description_length": (
        "Characters in the scrubbed description.",
        AnalyticalRole.NORMALIZED_NUMERIC,
    ),
}

#: Prefix meanings, so a reader can tell where any column came from.
PREFIX_MEANING: dict[str, str] = {
    "obs_": "from the search-card observation in the analysis run",
    "snap_": "from the joined detail snapshot (monitoring database)",
    "deep_": "from the deep scraper database via explicit, provenanced enrichment",
    "events_": "count of that event type for the advertisement up to the analysis run",
    "run_": "attribute of the observation's run (panel only)",
    "ad_": "attribute of the advertisement row",
    "source_": "a source column renamed because it collided with a provenance column",
}


def build(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (dataset, column) with meaning, role and coverage."""
    records: list[dict[str, Any]] = []
    for dataset_name, frame in datasets.items():
        if frame is None or frame.empty:
            continue
        total = len(frame)
        for column in frame.columns:
            description, role = _describe(column)
            non_null = int(frame[column].notna().sum())
            records.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "non_null_count": non_null,
                    "null_pct": round((total - non_null) / total * 100, 2) if total else None,
                    "distinct_count": int(frame[column].nunique(dropna=True)),
                    "analytical_role": str(role),
                    "description": description,
                    "example": _example(frame[column]),
                }
            )
    return pd.DataFrame(records)


def _describe(column: str) -> tuple[str, AnalyticalRole]:
    if column in DERIVED_COLUMNS:
        return DERIVED_COLUMNS[column]
    for prefix, meaning in PREFIX_MEANING.items():
        if column.startswith(prefix):
            base = column[len(prefix) :]
            from .schema_inspection import classify_column

            return f"{base}: {meaning}", classify_column(base, "", False)
    from .schema_inspection import classify_column

    return "", classify_column(column, "", False)


def _example(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return ""
    text = str(values.iloc[0])
    return text[:47] + "..." if len(text) > 50 else text
