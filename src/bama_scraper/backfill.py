"""Recompute derived fields on already-stored records.

Some normalized fields are pure functions of raw text that is already
persisted. When a normalizer improves, those records can be corrected in place
rather than refetched -- cheaper for us and for the site.

Relative phrases are re-anchored to each record's own ``scraped_at``, so a
backfill run days later still produces the timestamp the original scrape
should have produced.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .logging_config import get_logger
from .normalization import parse_published_time
from .storage import Storage

log = get_logger(__name__)


def _scraped_at_ts(value: Any) -> float:
    if isinstance(value, str):
        try:
            return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            pass
    return time.time()


def backfill_published_ts(storage: Storage, *, only_missing: bool = True) -> dict[str, int]:
    """Recompute ``published_ts`` from the stored ``published_text``.

    Returns counts of rows examined, updated, and still unresolved.
    """
    rows = list(storage.conn.execute("SELECT ad_id, payload FROM ad_details"))
    examined = updated = unresolved = 0
    pending: list[tuple[str, str]] = []

    for row in rows:
        payload = json.loads(row["payload"])
        if only_missing and payload.get("published_ts") is not None:
            continue
        examined += 1
        text = payload.get("published_text")
        if not text:
            unresolved += 1
            continue
        ts = parse_published_time(text, now_ts=_scraped_at_ts(payload.get("scraped_at")))
        if ts is None:
            unresolved += 1
            continue
        payload["published_ts"] = ts
        pending.append((json.dumps(payload, ensure_ascii=False), row["ad_id"]))
        updated += 1

    if pending:
        storage.conn.executemany("UPDATE ad_details SET payload=? WHERE ad_id=?", pending)
        storage.conn.commit()

    log.info("backfill.published_ts", examined=examined, updated=updated, unresolved=unresolved)
    return {"examined": examined, "updated": updated, "unresolved": unresolved}
