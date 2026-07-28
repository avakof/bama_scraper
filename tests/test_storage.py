"""Storage, checkpointing, resume semantics and export formats."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from bama_scraper.export import export_all, export_sqlite_copy
from bama_scraper.models import AdDetail, AdMedia, DiscoveredAd
from bama_scraper.storage import Storage

SEARCH_URL = "https://bama.ir/car?year=1397-2018,&price=1000000000"


def make_ad(n: int, **kw) -> DiscoveredAd:
    code = f"code{n:04d}"
    return DiscoveredAd(
        ad_id=code,
        url=f"https://bama.ir/car/detail-{code}-pride-1400",
        source_search_url=SEARCH_URL,
        card_title=f"پراید {n}",
        discovery_position=n,
        discovery_cycle=0,
        **kw,
    )


def make_detail(n: int, **kw) -> AdDetail:
    code = f"code{n:04d}"
    return AdDetail(
        ad_id=code,
        url=f"https://bama.ir/car/detail-{code}-pride-1400",
        title=f"پراید {n}",
        **kw,
    )


@pytest.fixture
def store(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "test.sqlite")
    s.start_run("run1", SEARCH_URL, {"priceFrom": "1000000000"}, "api", "1.0.0")
    yield s
    s.close()


class TestUpsertAndDedup:
    def test_new_ads_counted_once(self, store: Storage) -> None:
        assert store.upsert_discovered([make_ad(1), make_ad(2)], "run1") == 2
        assert store.count_discovered() == 2

    def test_reinserting_same_url_adds_no_rows(self, store: Storage) -> None:
        store.upsert_discovered([make_ad(1)], "run1")
        assert store.upsert_discovered([make_ad(1)], "run1") == 0
        assert store.count_discovered() == 1

    def test_duplicate_sighting_is_recorded(self, store: Storage) -> None:
        store.upsert_discovered([make_ad(1)], "run1")
        store.upsert_discovered([make_ad(1)], "run1")
        row = store.conn.execute("SELECT times_seen FROM discovered_ads").fetchone()
        assert row["times_seen"] == 2

    def test_upsert_does_not_erase_existing_values(self, store: Storage) -> None:
        store.upsert_discovered([make_ad(1, price_toman=1_500_000_000)], "run1")
        store.upsert_discovered([make_ad(1)], "run1")  # later sighting lacks a price
        row = store.conn.execute("SELECT price_toman FROM discovered_ads").fetchone()
        assert row["price_toman"] == 1_500_000_000

    def test_empty_batch(self, store: Storage) -> None:
        assert store.upsert_discovered([], "run1") == 0


class TestStatusLifecycle:
    def test_default_status(self, store: Storage) -> None:
        store.upsert_discovered([make_ad(1)], "run1")
        assert store.stats()["pending"] == 1

    def test_transition_to_completed(self, store: Storage) -> None:
        ad = make_ad(1)
        store.upsert_discovered([ad], "run1")
        store.mark_status(ad.url, "completed", bump_attempt=True)
        stats = store.stats()
        assert stats["completed"] == 1 and stats["pending"] == 0

    def test_retryable_is_requeued_until_max_attempts(self, store: Storage) -> None:
        ad = make_ad(1)
        store.upsert_discovered([ad], "run1")
        for _ in range(3):
            store.mark_status(ad.url, "retryable_error", "boom", bump_attempt=True)
        assert len(store.claim_pending(10, max_attempts=4)) == 1
        store.mark_status(ad.url, "retryable_error", "boom", bump_attempt=True)
        assert store.claim_pending(10, max_attempts=4) == []

    def test_permanent_error_never_requeued(self, store: Storage) -> None:
        ad = make_ad(1)
        store.upsert_discovered([ad], "run1")
        store.mark_status(ad.url, "permanent_error", "404", bump_attempt=True)
        assert store.claim_pending(10) == []

    def test_completed_skipped_unless_refresh(self, store: Storage) -> None:
        ad = make_ad(1)
        store.upsert_discovered([ad], "run1")
        store.mark_status(ad.url, "completed")
        assert store.claim_pending(10) == []
        assert len(store.claim_pending(10, refresh=True)) == 1


class TestCrashRecovery:
    def test_rows_left_in_scraping_are_recovered(self, store: Storage) -> None:
        ads = [make_ad(i) for i in range(3)]
        store.upsert_discovered(ads, "run1")
        store.mark_many_scraping([a.url for a in ads])
        # simulate a crash here: rows are stuck in 'scraping'
        assert store.recover_stuck() == 3
        assert len(store.claim_pending(10)) == 3

    def test_resume_across_reopen_keeps_progress(self, tmp_path: Path) -> None:
        db = tmp_path / "resume.sqlite"
        first = Storage(db)
        first.start_run("run1", SEARCH_URL, {}, "api", "1.0.0")
        ads = [make_ad(i) for i in range(5)]
        first.upsert_discovered(ads, "run1")
        first.mark_status(ads[0].url, "completed")
        first.mark_status(ads[1].url, "completed")
        first.close()

        # a fresh process opens the same database
        second = Storage(db)
        assert second.count_discovered() == 5
        assert second.stats()["completed"] == 2
        remaining = second.claim_pending(10)
        assert len(remaining) == 3
        assert all(r["url"] not in (ads[0].url, ads[1].url) for r in remaining)
        second.close()

    def test_checkpoint_roundtrip(self, store: Storage) -> None:
        store.set_checkpoint("run1", "api_last_page", 42)
        assert store.get_checkpoint("run1", "api_last_page") == 42
        store.set_checkpoint("run1", "api_last_page", 43)
        assert store.get_checkpoint("run1", "api_last_page") == 43

    def test_missing_checkpoint_is_none(self, store: Storage) -> None:
        assert store.get_checkpoint("run1", "nope") is None

    def test_existing_checkpoint_not_clobbered_by_new_run(self, store: Storage) -> None:
        store.set_checkpoint("run1", "k", 1)
        store.set_checkpoint("run2", "k", 2)
        assert store.get_checkpoint("run1", "k") == 1


class TestDetailPersistence:
    def test_save_and_read_back(self, store: Storage) -> None:
        ad = make_ad(1)
        store.upsert_discovered([ad], "run1")
        detail = make_detail(1, price_toman=1_200_000_000, image_urls=["https://x/1.jpg"])
        media = [AdMedia(ad_id=detail.ad_id, position=0, url="https://x/1.jpg")]
        store.save_detail(detail, media)
        rows = list(store.iter_details())
        assert len(rows) == 1
        assert rows[0]["price_toman"] == 1_200_000_000
        assert store.stats()["media"] == 1

    def test_resaving_replaces_media_without_duplicating(self, store: Storage) -> None:
        detail = make_detail(1)
        store.save_detail(detail, [AdMedia(ad_id="code0001", position=0, url="a")])
        store.save_detail(detail, [AdMedia(ad_id="code0001", position=0, url="b")])
        assert store.stats()["media"] == 1

    def test_persian_text_survives_roundtrip(self, store: Storage) -> None:
        detail = make_detail(1, description="توضیحات فارسی با ZWNJ می‌رود")
        store.save_detail(detail, [])
        assert list(store.iter_details())[0]["description"] == "توضیحات فارسی با ZWNJ می‌رود"

    def test_raw_attributes_stored(self, store: Storage) -> None:
        detail = make_detail(1, raw_attributes={"vehicle.color": "سفید", "n": 3})
        store.save_detail(detail, [])
        rows = {
            r["key"]: r["value"]
            for r in store.conn.execute("SELECT key, value FROM raw_attributes")
        }
        assert rows["vehicle.color"] == "سفید"


class TestErrors:
    def test_errors_logged(self, store: Storage) -> None:
        store.log_error("run1", "code0001", "u", "detail", "http_error", "500")
        assert store.stats()["errors"] == 1


class TestExports:
    @pytest.fixture
    def populated(self, tmp_path: Path) -> Storage:
        s = Storage(tmp_path / "e.sqlite")
        s.start_run("run1", SEARCH_URL, {}, "api", "1.0.0")
        ads = [make_ad(i) for i in range(3)]
        s.upsert_discovered(ads, "run1")
        for i, ad in enumerate(ads):
            s.save_detail(
                make_detail(
                    i,
                    price_toman=1_000_000_000 + i,
                    description="توضیحات فارسی",
                    image_urls=[f"https://x/{i}.jpg"],
                    badges=["خانواده"],
                    raw_attributes={"k": "v"},
                ),
                [AdMedia(ad_id=f"code{i:04d}", position=0, url=f"https://x/{i}.jpg")],
            )
            s.mark_status(ad.url, "completed")
        s.log_error("run1", "code0001", "u", "detail", "http_error", "500")
        yield s
        s.close()

    def test_all_files_written(self, populated: Storage, tmp_path: Path) -> None:
        out = tmp_path / "out"
        manifest = export_all(populated, out)
        for name in (
            "bama_ads.csv",
            "bama_ads.jsonl",
            "bama_media.csv",
            "bama_errors.csv",
            "network_candidates.json",
        ):
            assert (out / name).exists(), name
        assert manifest["rows"] == 3

    def test_jsonl_one_row_per_ad_and_utf8(self, populated: Storage, tmp_path: Path) -> None:
        out = tmp_path / "out"
        export_all(populated, out)
        lines = (out / "bama_ads.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert record["description"] == "توضیحات فارسی"

    def test_csv_has_bom_for_excel(self, populated: Storage, tmp_path: Path) -> None:
        out = tmp_path / "out"
        export_all(populated, out)
        assert (out / "bama_ads.csv").read_bytes().startswith(b"\xef\xbb\xbf")

    def test_csv_readable_and_persian_intact(self, populated: Storage, tmp_path: Path) -> None:
        import pandas as pd

        out = tmp_path / "out"
        export_all(populated, out)
        frame = pd.read_csv(out / "bama_ads.csv", encoding="utf-8-sig")
        assert len(frame) == 3
        assert frame["description"].iloc[0] == "توضیحات فارسی"

    def test_parquet_roundtrip(self, populated: Storage, tmp_path: Path) -> None:
        import pandas as pd

        out = tmp_path / "out"
        manifest = export_all(populated, out)
        assert "parquet" in manifest["files"], manifest.get("parquet_error")
        frame = pd.read_parquet(out / "bama_ads.parquet")
        assert len(frame) == 3
        assert frame["description"].iloc[0] == "توضیحات فارسی"

    def test_list_columns_json_encoded_keeping_one_row_per_ad(
        self, populated: Storage, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        export_all(populated, out)
        record = json.loads((out / "bama_ads.jsonl").read_text(encoding="utf-8").split("\n")[0])
        assert json.loads(record["image_urls"]) == ["https://x/0.jpg"]

    def test_media_csv_is_normalized_separately(self, populated: Storage, tmp_path: Path) -> None:
        import pandas as pd

        out = tmp_path / "out"
        export_all(populated, out)
        media = pd.read_csv(out / "bama_media.csv", encoding="utf-8-sig")
        assert len(media) == 3 and set(media["ad_id"]) == {"code0000", "code0001", "code0002"}

    def test_errors_csv(self, populated: Storage, tmp_path: Path) -> None:
        import pandas as pd

        out = tmp_path / "out"
        export_all(populated, out)
        assert len(pd.read_csv(out / "bama_errors.csv", encoding="utf-8-sig")) == 1

    def test_sqlite_snapshot_is_queryable(self, populated: Storage, tmp_path: Path) -> None:
        target = tmp_path / "out" / "bama_ads.sqlite"
        export_sqlite_copy(populated, target)
        conn = sqlite3.connect(target)
        assert conn.execute("SELECT COUNT(*) FROM ad_details").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM discovered_ads").fetchone()[0] == 3
        conn.close()


class TestRefreshSnapshot:
    """Regression: --refresh used to stop after a single batch.

    ``claim_pending(refresh=True)`` ignores status, so it returns the same rows
    forever; the pipeline therefore walks an explicit snapshot instead.
    """

    def test_all_ads_returns_every_row_in_discovery_order(self, store: Storage) -> None:
        store.upsert_discovered([make_ad(i) for i in range(5)], "run1")
        rows = store.all_ads()
        assert len(rows) == 5
        assert [r["url"] for r in rows] == [make_ad(i).url for i in range(5)]

    def test_all_ads_includes_completed_rows(self, store: Storage) -> None:
        ads = [make_ad(i) for i in range(3)]
        store.upsert_discovered(ads, "run1")
        for ad in ads:
            store.mark_status(ad.url, "completed")
        assert store.claim_pending(10) == []  # nothing left by status
        assert len(store.all_ads()) == 3  # but refresh still sees all

    def test_snapshot_chunking_covers_everything_exactly_once(self, store: Storage) -> None:
        store.upsert_discovered([make_ad(i) for i in range(57)], "run1")
        queue = store.all_ads()
        seen, size = [], 25
        while queue:
            seen.extend(r["url"] for r in queue[:size])
            queue = queue[size:]
        assert len(seen) == 57 and len(set(seen)) == 57


class TestBackfill:
    """Improved normalizers must be applicable without refetching the site."""

    def test_recomputes_missing_published_ts(self, store: Storage) -> None:
        from bama_scraper.backfill import backfill_published_ts

        store.save_detail(
            make_detail(
                1, published_text="1405/4/28", published_ts=None, scraped_at="2026-07-27T12:00:00"
            ),
            [],
        )
        result = backfill_published_ts(store)
        assert result["updated"] == 1
        record = list(store.iter_details())[0]
        assert record["published_ts"] is not None

    def test_relative_dates_anchor_to_the_original_scrape_time(self, store: Storage) -> None:
        from bama_scraper.backfill import backfill_published_ts

        store.save_detail(
            make_detail(
                2, published_text="2 روز پیش", published_ts=None, scraped_at="2026-07-27T12:00:00"
            ),
            [],
        )
        backfill_published_ts(store)
        import time as _t

        expected = _t.mktime(_t.strptime("2026-07-27T12:00:00", "%Y-%m-%dT%H:%M:%S")) - 172800
        assert list(store.iter_details())[0]["published_ts"] == expected

    def test_already_populated_rows_are_left_alone(self, store: Storage) -> None:
        from bama_scraper.backfill import backfill_published_ts

        store.save_detail(make_detail(3, published_text="دیروز", published_ts=999.0), [])
        assert backfill_published_ts(store)["updated"] == 0
        assert list(store.iter_details())[0]["published_ts"] == 999.0

    def test_unresolvable_text_is_counted_not_invented(self, store: Storage) -> None:
        from bama_scraper.backfill import backfill_published_ts

        store.save_detail(make_detail(4, published_text="زمان نامشخص", published_ts=None), [])
        result = backfill_published_ts(store)
        assert result["updated"] == 0 and result["unresolved"] == 1
        assert list(store.iter_details())[0]["published_ts"] is None
