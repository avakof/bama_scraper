"""Storage schema, queues, checkpoints, and read-only seeding."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from bama_deep.endpoints import (
    ForbiddenEndpointError,
    ad_api_url,
    assert_allowed,
    dealer_id_from_url,
    is_allowed_asset,
    normalize_price_key,
    normalize_review_key,
    parse_review_key,
)
from bama_deep.seed import SeedError, file_sha256, open_source_readonly, seed_from_source
from bama_deep.storage import DeepStorage


@pytest.fixture
def store(tmp_path: Path) -> DeepStorage:
    s = DeepStorage(tmp_path / "deep.sqlite")
    s.start_run(
        "r1", source_db="src.sqlite", source_sha256="abc", phases="a", version="t", config={}
    )
    yield s
    s.close()


def make_source(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Build a minimal stand-in for the original scraper's database."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE discovered_ads (url TEXT PRIMARY KEY, ad_id TEXT, status TEXT)")
    conn.executemany("INSERT INTO discovered_ads VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


class TestSchema:
    def test_reopen_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "x.sqlite"
        DeepStorage(path).close()
        second = DeepStorage(path)
        assert second.count("deep_queue") == 0
        second.close()

    def test_all_expected_tables_exist(self, store: DeepStorage) -> None:
        names = {
            r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in (
            "deep_runs",
            "deep_checkpoint",
            "deep_errors",
            "deep_queue",
            "ad_deep",
            "ad_media_deep",
            "ad_raw_attributes",
            "ad_field_conflicts",
            "trim_reviews",
            "trim_spec_groups",
            "trim_specs",
            "trim_specs_typed",
            "spec_key_catalog",
            "price_keys",
            "price_series",
            "price_points",
            "price_brands",
            "price_hierarchy",
            "dealers",
            "dealer_ads",
        ):
            assert table in names, table


class TestQueue:
    def test_seed_and_claim(self, store: DeepStorage) -> None:
        rows = [
            (f"code{i:04d}", f"https://bama.ir/car/detail-code{i:04d}-x", "api", "completed")
            for i in range(5)
        ]
        assert store.seed_queue(rows) == 5
        assert len(store.claim_ads(10)) == 5

    def test_seed_is_idempotent(self, store: DeepStorage) -> None:
        rows = [("a1", "u", "api", "completed")]
        assert store.seed_queue(rows) == 1
        assert store.seed_queue(rows) == 0
        assert store.count("deep_queue") == 1

    def test_terminal_statuses_are_not_reclaimed(self, store: DeepStorage) -> None:
        store.seed_queue([(f"a{i}", "u", "api", "") for i in range(4)])
        for ad_id, status in (
            ("a0", "completed"),
            ("a1", "partial"),
            ("a2", "delisted"),
            ("a3", "permanent_error"),
        ):
            store.set_ad_status(ad_id, status)
        assert store.claim_ads(10) == []

    def test_retryable_requeued_until_max_attempts(self, store: DeepStorage) -> None:
        store.seed_queue([("a1", "u", "api", "")])
        for _ in range(3):
            store.set_ad_status("a1", "retryable_error", error="boom", bump_attempt=True)
        assert len(store.claim_ads(10, max_attempts=4)) == 1
        store.set_ad_status("a1", "retryable_error", error="boom", bump_attempt=True)
        assert store.claim_ads(10, max_attempts=4) == []

    def test_crashed_fetching_rows_stay_claimable(self, store: DeepStorage) -> None:
        # 'fetching' is deliberately NOT terminal: if the process dies mid-batch the
        # rows must remain claimable even if recover_stuck() never runs.
        store.seed_queue([(f"a{i}", "u", "api", "") for i in range(3)])
        store.mark_ads_fetching(["a0", "a1", "a2"])
        assert len(store.claim_ads(10)) == 3

    def test_recover_stuck_normalizes_status(self, store: DeepStorage) -> None:
        store.seed_queue([(f"a{i}", "u", "api", "") for i in range(3)])
        store.mark_ads_fetching(["a0", "a1", "a2"])
        assert store.recover_stuck() == 3
        assert store.status_breakdown("deep_queue") == {"pending": 3}
        assert len(store.claim_ads(10)) == 3

    def test_requeue_delisted_is_opt_in(self, store: DeepStorage) -> None:
        store.seed_queue([("a1", "u", "api", "")])
        store.set_ad_status("a1", "delisted")
        assert store.claim_ads(10) == []
        assert store.requeue_delisted() == 1
        assert len(store.claim_ads(10)) == 1

    def test_all_queue_is_a_fixed_snapshot(self, store: DeepStorage) -> None:
        # --refresh must walk a snapshot; a status-driven loop cannot terminate.
        store.seed_queue([(f"a{i}", "u", "api", "") for i in range(6)])
        for i in range(6):
            store.set_ad_status(f"a{i}", "completed")
        assert store.claim_ads(10) == []
        assert len(store.all_queue()) == 6


class TestCheckpoints:
    def test_roundtrip(self, store: DeepStorage) -> None:
        store.set_checkpoint("r1", "k", 42)
        assert store.get_checkpoint("r1", "k") == 42

    def test_dict_value(self, store: DeepStorage) -> None:
        store.set_checkpoint("r1", "k", {"a": [1, 2]})
        assert store.get_checkpoint("r1", "k") == {"a": [1, 2]}

    def test_overwrite(self, store: DeepStorage) -> None:
        store.set_checkpoint("r1", "k", 1)
        store.set_checkpoint("r1", "k", 2)
        assert store.get_checkpoint("r1", "k") == 2

    def test_missing(self, store: DeepStorage) -> None:
        assert store.get_checkpoint("r1", "nope") is None

    def test_runs_are_isolated(self, store: DeepStorage) -> None:
        store.set_checkpoint("r1", "k", 1)
        store.set_checkpoint("r2", "k", 2)
        assert store.get_checkpoint("r1", "k") == 1


class TestAdRecords:
    def _record(self, ad_id: str = "a1", **kw) -> dict:
        base = {"ad_id": ad_id, "url": "https://bama.ir/car/detail-a1-x", "title": "پراید"}
        base.update(kw)
        return base

    def test_save_and_read(self, store: DeepStorage) -> None:
        store.save_ad(
            self._record(fuel_type="بنزینی"),
            [{"kind": "image", "position": 0, "original_url": "https://x/1.jpg"}],
            [("api", "detail.fuel", "بنزینی")],
            [],
        )
        row = store.conn.execute("SELECT * FROM ad_deep").fetchone()
        assert row["fuel_type"] == "بنزینی"
        assert store.count("ad_media_deep") == 1
        assert store.count("ad_raw_attributes") == 1

    def test_resave_replaces_media_without_duplicating(self, store: DeepStorage) -> None:
        rec = self._record()
        store.save_ad(rec, [{"kind": "image", "position": 0, "original_url": "a"}], [], [])
        store.save_ad(rec, [{"kind": "image", "position": 0, "original_url": "b"}], [], [])
        assert store.count("ad_deep") == 1
        assert store.count("ad_media_deep") == 1
        assert store.conn.execute("SELECT original_url FROM ad_media_deep").fetchone()[0] == "b"

    def test_conflicts_replaced_not_accumulated(self, store: DeepStorage) -> None:
        rec = self._record()
        conflict = {"field": "price_toman", "api_value": 1, "html_value": 2, "severity": "critical"}
        store.save_ad(rec, [], [], [conflict])
        store.save_ad(rec, [], [], [conflict])
        assert store.count("ad_field_conflicts") == 1

    def test_persian_roundtrip(self, store: DeepStorage) -> None:
        store.save_ad(self._record(description="توضیحات با ZWNJ می‌رود"), [], [], [])
        assert (
            store.conn.execute("SELECT description FROM ad_deep").fetchone()[0]
            == "توضیحات با ZWNJ می‌رود"
        )


class TestDerivedQueues:
    def test_derive_review_and_price_and_dealer_keys(self, store: DeepStorage) -> None:
        for i in range(3):
            store.save_ad(
                {
                    "ad_id": f"a{i}",
                    "url": "u",
                    "review_key": "/car-reviews/dena/plusef7p-specs-1481-6mt",
                    "review_url": "/car-reviews/dena/plusef7p-specs-1481-6mt",
                    "price_key": "dena|plusef7p|6mt",
                    "price_url": "/price/dena_plusef7p_6mt",
                    "dealer_id": 921,
                    "dealer_link": "/dealer/921",
                },
                [],
                [],
                [],
            )
        # Three ads sharing one model-trim must produce ONE fetch target each.
        assert store.derive_review_keys() == 1
        assert store.derive_price_keys() == 1
        assert store.derive_dealer_ids() == 1
        assert store.conn.execute("SELECT ad_count_in_dataset FROM trim_reviews").fetchone()[0] == 3

    def test_price_key_parts_are_split(self, store: DeepStorage) -> None:
        store.save_ad(
            {
                "ad_id": "a1",
                "url": "u",
                "price_key": "dena|plusef7p|6mt",
                "price_url": "/price/dena_plusef7p_6mt",
            },
            [],
            [],
            [],
        )
        store.derive_price_keys()
        row = store.conn.execute("SELECT brand, model, trim FROM price_keys").fetchone()
        assert (row["brand"], row["model"], row["trim"]) == ("dena", "plusef7p", "6mt")

    def test_ads_without_join_keys_are_not_queued(self, store: DeepStorage) -> None:
        store.save_ad({"ad_id": "a1", "url": "u"}, [], [], [])
        assert store.derive_review_keys() == 0
        assert store.derive_price_keys() == 0
        assert store.derive_dealer_ids() == 0


class TestSeeding:
    def test_seeds_from_real_shaped_source(self, tmp_path: Path) -> None:
        src = make_source(
            tmp_path / "src.sqlite",
            [
                ("https://bama.ir/car/detail-aaa11111-pride-1400", "aaa11111", "completed"),
                ("https://bama.ir/car/detail-bbb22222-tara-1403", "bbb22222", "permanent_error"),
            ],
        )
        store = DeepStorage(tmp_path / "deep.sqlite")
        result = seed_from_source(store, src)
        # HTTP-410 ads from the original run are re-checked, not inherited as failures.
        assert result["seeded"] == 2
        assert store.count("deep_queue") == 2
        store.close()

    def test_only_completed_filter(self, tmp_path: Path) -> None:
        src = make_source(
            tmp_path / "src.sqlite",
            [
                ("https://bama.ir/car/detail-aaa11111-x", "aaa11111", "completed"),
                ("https://bama.ir/car/detail-bbb22222-x", "bbb22222", "permanent_error"),
            ],
        )
        store = DeepStorage(tmp_path / "deep.sqlite")
        assert seed_from_source(store, src, only_completed=True)["seeded"] == 1
        store.close()

    def test_reseed_is_idempotent(self, tmp_path: Path) -> None:
        src = make_source(
            tmp_path / "src.sqlite", [("https://bama.ir/car/detail-aaa11111-x", "aaa11111", "c")]
        )
        store = DeepStorage(tmp_path / "deep.sqlite")
        seed_from_source(store, src)
        assert seed_from_source(store, src)["seeded"] == 0
        store.close()

    def test_api_url_is_derived_per_ad(self, tmp_path: Path) -> None:
        src = make_source(
            tmp_path / "src.sqlite", [("https://bama.ir/car/detail-aaa11111-x", "aaa11111", "c")]
        )
        store = DeepStorage(tmp_path / "deep.sqlite")
        seed_from_source(store, src)
        assert store.conn.execute("SELECT api_url FROM deep_queue").fetchone()[0] == ad_api_url(
            "aaa11111"
        )
        store.close()

    def test_source_is_opened_read_only(self, tmp_path: Path) -> None:
        src = make_source(tmp_path / "src.sqlite", [("u", "aaa11111", "c")])
        conn = open_source_readonly(src)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO discovered_ads VALUES ('x','y','z')")
        conn.close()

    def test_source_file_unchanged_by_seeding(self, tmp_path: Path) -> None:
        src = make_source(
            tmp_path / "src.sqlite", [("https://bama.ir/car/detail-aaa11111-x", "aaa11111", "c")]
        )
        before = file_sha256(src)
        store = DeepStorage(tmp_path / "deep.sqlite")
        seed_from_source(store, src)
        store.close()
        assert file_sha256(src) == before

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        store = DeepStorage(tmp_path / "deep.sqlite")
        with pytest.raises(SeedError):
            seed_from_source(store, tmp_path / "nope.sqlite")
        store.close()


class TestEndpointGuards:
    @pytest.mark.parametrize(
        "url",
        [
            "https://bama.ir/cad/api/log/visit",
            "https://bama.ir/cad/api/log/impression",
            "https://bama.ir/cad/api/log/share",
            "https://bama.ir/event/api/v1/events",
            "https://bama.ir/cad/api/carad/abc123",
            "https://bama.ir/cad/api/carad/abc123/details",
            "https://bama.ir/cad/api/price?pageIndex=0",
            "https://bama.ir/cad/api/Corporation/phone/921",
            "https://bama.ir/prf/api/user/userinfo",
        ],
    )
    def test_forbidden_urls_raise(self, url: str) -> None:
        with pytest.raises(ForbiddenEndpointError):
            assert_allowed(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://bama.ir/cad/api/detail/abc123",
            "https://bama.ir/cad/api/price/detail?brand=dena&model=x&trim=y",
            "https://bama.ir/cad/api/price/brand",
            "https://bama.ir/cad/api/Corporation/921",
            "https://bama.ir/cad/api/Corporation/ads/921",
            "https://bama.ir/nws/api/CarReview/getspecification?id=1&trimname=x",
            "https://bama.ir/car/detail-abc123-pride-1400",
        ],
    )
    def test_permitted_urls_pass(self, url: str) -> None:
        assert_allowed(url)  # must not raise

    def test_campaign_banner_assets_refused(self) -> None:
        # The single robots.txt Disallow.
        assert not is_allowed_asset(
            "https://cdn-sth1.bama.ir/uploads/BamaImages/CampaignBanner/183/x.jpg"
        )

    def test_vehicle_photos_allowed(self) -> None:
        assert is_allowed_asset(
            "https://cdn-sth1.bama.ir/uploads/BamaImages/VehicleCarImages/abc/CarImage_x.jpg"
        )


class TestJoinKeys:
    def test_review_key_canonicalization(self) -> None:
        assert (
            normalize_review_key("/car-reviews/Dena/plusef7p-specs-1481-6mt/")
            == "/car-reviews/dena/plusef7p-specs-1481-6mt"
        )

    @pytest.mark.parametrize(
        "key,model,trim,ids",
        [
            ("/car-reviews/dena/plusef7p-specs-1481-6mt", "plusef7p", "6mt", "1481"),
            ("/car-reviews/samand/soren-specs-1113-459-plusxu7p", "soren", "plusxu7p", "1113-459"),
            ("/car-reviews/shahin/g-specs-1360", "g", None, "1360"),
        ],
    )
    def test_all_three_review_shapes_parse(
        self, key: str, model: str, trim: str | None, ids: str
    ) -> None:
        parts = parse_review_key(key)
        assert parts["model_slug"] == model
        assert parts["trim_slug"] == trim
        assert parts["extra_segment"] == ids

    def test_price_key(self) -> None:
        key, parts = normalize_price_key("/price/dena_plusef7p_6mt")
        assert key == "dena|plusef7p|6mt"
        assert parts == {"brand": "dena", "model": "plusef7p", "trim": "6mt"}

    def test_price_key_with_multipart_trim(self) -> None:
        key, parts = normalize_price_key("/price/kmc_eagle_1.5liter")
        assert key == "kmc|eagle|1.5liter"

    @pytest.mark.parametrize(
        "url,key,trim",
        [
            ("/price/shahin_g", "shahin|g|", ""),
            ("/price/shahin_gl", "shahin|gl|", ""),
            ("/price/farda_511", "farda|511|", ""),
        ],
    )
    def test_single_trim_models_have_two_segments(self, url: str, key: str, trim: str) -> None:
        # Verified live: the endpoint accepts an empty trim and returns the full
        # series set. Requiring three segments silently dropped 63 ads.
        got_key, parts = normalize_price_key(url)
        assert got_key == key
        assert parts["trim"] == trim

    def test_malformed_price_url_yields_no_key(self) -> None:
        assert normalize_price_key("/price/onlybrand")[0] is None
        assert normalize_price_key(None)[0] is None
        assert normalize_price_key("")[0] is None

    def test_dealer_id(self) -> None:
        assert dealer_id_from_url("/dealer/6007") == 6007
        assert dealer_id_from_url("https://bama.ir/dealer/921") == 921
        assert dealer_id_from_url(None) is None
        assert dealer_id_from_url("/car/detail-x") is None
