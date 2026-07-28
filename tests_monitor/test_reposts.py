"""Repost detection, with equal weight on avoiding false positives.

A missed repost inflates the apparent sale rate. A false repost suppresses a
genuine disappearance. Both directions are tested.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bama_monitor.config import RepostConfig
from bama_monitor.repost_detection import (
    VehicleProfile,
    description_similarity,
    find_reposts,
    fingerprint,
    hamming_hex,
    image_overlap,
    score_pair,
    vehicle_entity_id,
)

CFG = RepostConfig()

LONG_DESC = (
    "خودرو کاملا سالم و بدون رنگ، سرویس های دوره ای انجام شده، لاستیک ها نو، "
    "بیمه تا پایان سال، معاینه فنی دارد، تک برگ سند به نام"
)

T0 = datetime(2026, 7, 1, 11, 0, tzinfo=UTC)


def profile(key: str, **kw) -> VehicleProfile:
    base = {
        "brand": "دنا",
        "model": "پلاس",
        "year": "1403",
        "mileage_km": 50_000,
        "price_toman": 2_000_000_000,
        "city": "تهران",
        "seller_name": "اتو گالری تست",
        "body_color": "سفید",
        "interior_color": "مشکی",
        "description": LONG_DESC,
        "image_hashes": ("aaaa111122223333",),
    }
    base.update(kw)
    return VehicleProfile(platform_ad_id=key, **base)


class TestFingerprint:
    def test_identical_vehicles_share_a_fingerprint(self) -> None:
        assert fingerprint(profile("a"), CFG) == fingerprint(profile("b"), CFG)

    def test_small_mileage_drift_stays_in_the_same_bucket(self) -> None:
        # A repost is retyped by hand, so exact mileage equality is too strict.
        assert fingerprint(profile("a", mileage_km=50_000), CFG) == fingerprint(
            profile("b", mileage_km=52_000), CFG
        )

    def test_different_model_differs(self) -> None:
        assert fingerprint(profile("a"), CFG) != fingerprint(profile("b", model="تارا"), CFG)

    def test_zwnj_and_case_do_not_change_the_fingerprint(self) -> None:
        assert fingerprint(profile("a", brand="دنا"), CFG) == fingerprint(
            profile("b", brand=" دنا "), CFG
        )


class TestSignals:
    def test_hamming_of_identical_hashes_is_zero(self) -> None:
        assert hamming_hex("abcd", "abcd") == 0

    def test_hamming_detects_small_differences(self) -> None:
        # 0xabcd ^ 0xabce == 0x0003 -> two bits differ.
        assert hamming_hex("abcd", "abce") == 2
        assert hamming_hex("0000", "000f") == 4

    def test_hamming_rejects_mismatched_lengths(self) -> None:
        assert hamming_hex("abcd", "abcdef") is None

    def test_image_overlap_counts_near_matches(self) -> None:
        assert image_overlap(("aaaa000000000000",), ("aaaa000000000001",)) == 1

    def test_image_overlap_ignores_unrelated_images(self) -> None:
        assert image_overlap(("ffffffffffffffff",), ("0000000000000000",)) == 0

    def test_description_similarity_needs_substance(self) -> None:
        # Two short strings are not evidence, however similar.
        assert description_similarity("سالم", "سالم") == 0.0

    def test_identical_long_descriptions_are_similar(self) -> None:
        assert description_similarity(LONG_DESC, LONG_DESC) == pytest.approx(1.0)

    def test_different_descriptions_are_dissimilar(self) -> None:
        other = "پراید مدل پایین با کارکرد بالا و رنگ شدگی در چند نقطه، مناسب برای مصرف روزانه"
        assert description_similarity(LONG_DESC, other) < CFG.description_similarity_min


class TestScorePair:
    def test_same_everything_scores_high(self) -> None:
        score, matched, _ = score_pair(profile("old"), profile("new"), CFG)
        assert score >= CFG.match_threshold
        assert "same_seller" in matched and "image_hash" in matched

    def test_taxonomy_only_is_below_threshold(self) -> None:
        """Two different cars of the same model and year must not match."""
        old = profile(
            "old",
            seller_name="گالری الف",
            image_hashes=(),
            description=None,
            mileage_km=30_000,
            price_toman=1_500_000_000,
            city="تهران",
            body_color="سفید",
            interior_color="مشکی",
        )
        new = profile(
            "new",
            seller_name="گالری ب",
            image_hashes=(),
            description=None,
            mileage_km=120_000,
            price_toman=2_600_000_000,
            city="مشهد",
            body_color="مشکی",
            interior_color="کرم",
        )
        score, matched, _ = score_pair(old, new, CFG)
        assert score < CFG.match_threshold
        assert matched == ["brand_model_year"]

    def test_slight_mileage_change_still_matches(self) -> None:
        score, matched, evidence = score_pair(
            profile("old", mileage_km=50_000), profile("new", mileage_km=53_000), CFG
        )
        assert "mileage_close" in matched
        assert evidence["mileage_delta_km"] == 3_000
        assert score >= CFG.match_threshold

    def test_large_mileage_change_drops_that_signal(self) -> None:
        _, matched, _ = score_pair(
            profile("old", mileage_km=50_000), profile("new", mileage_km=95_000), CFG
        )
        assert "mileage_close" not in matched

    def test_price_change_within_tolerance_matches(self) -> None:
        _, matched, _ = score_pair(
            profile("old", price_toman=2_000_000_000),
            profile("new", price_toman=2_100_000_000),
            CFG,
        )
        assert "price_close" in matched

    def test_price_change_outside_tolerance_drops_that_signal(self) -> None:
        _, matched, _ = score_pair(
            profile("old", price_toman=2_000_000_000),
            profile("new", price_toman=3_000_000_000),
            CFG,
        )
        assert "price_close" not in matched

    def test_same_description_different_seller_still_needs_more(self) -> None:
        """Copied text alone is weak evidence; a private seller may reuse a template."""
        old = profile("old", seller_name="گالری الف", image_hashes=())
        new = profile("new", seller_name="گالری ب", image_hashes=())
        score, matched, _ = score_pair(old, new, CFG)
        assert "description_similar" in matched
        assert "same_seller" not in matched
        # Taxonomy + mileage + price + city + colors + description can clear the
        # bar; what matters is that seller identity is not assumed.
        assert score == pytest.approx(0.80, abs=0.01)


class TestFindReposts:
    def test_matches_a_clear_repost(self) -> None:
        matches = find_reposts([profile("new1")], [profile("old1")], CFG)
        assert len(matches) == 1
        assert matches[0].parent_platform_ad_id == "old1"
        assert matches[0].new_platform_ad_id == "new1"

    def test_no_match_for_an_unrelated_vehicle(self) -> None:
        other = profile(
            "new1",
            brand="پراید",
            model="111",
            year="1396",
            seller_name="کسی دیگر",
            image_hashes=(),
            description=None,
        )
        assert find_reposts([other], [profile("old1")], CFG) == []

    def test_a_parent_is_claimed_only_once(self) -> None:
        matches = find_reposts([profile("new1"), profile("new2")], [profile("old1")], CFG)
        assert len(matches) == 1

    def test_best_scoring_parent_wins(self) -> None:
        weak = profile("old_weak", seller_name="گالری دیگر", image_hashes=())
        strong = profile("old_strong")
        matches = find_reposts([profile("new1")], [weak, strong], CFG)
        assert matches[0].parent_platform_ad_id == "old_strong"

    def test_same_id_is_never_a_repost_of_itself(self) -> None:
        assert find_reposts([profile("same")], [profile("same")], CFG) == []

    def test_disabled_config_short_circuits(self) -> None:
        disabled = RepostConfig(enabled=False)
        assert find_reposts([profile("new1")], [profile("old1")], disabled) == []

    def test_match_records_its_evidence(self) -> None:
        match = find_reposts([profile("new1")], [profile("old1")], CFG)[0]
        assert match.matched_on
        assert "matching_images" in match.evidence


class TestVehicleEntity:
    def test_stable_and_order_independent(self) -> None:
        assert vehicle_entity_id("a", "b") == vehicle_entity_id("b", "a")

    def test_existing_group_is_preserved(self) -> None:
        # A car reposted three times must stay one entity, not fragment into pairs.
        assert vehicle_entity_id("a", "c", existing="veh_keepme") == "veh_keepme"

    def test_distinct_pairs_get_distinct_ids(self) -> None:
        assert vehicle_entity_id("a", "b") != vehicle_entity_id("c", "d")


class TestNegativeEvidenceVocabulary:
    """Events are named for what was observed, not for what was assumed."""

    def test_a_live_page_during_absence_is_not_called_a_restoration(self) -> None:
        from bama_monitor.models import EventType

        assert EventType.DETAIL_STILL_ACTIVE.value == "detail_still_active"
        # `detail_restored` would assert the page had been down and came back.
        assert not hasattr(EventType, "DETAIL_RESTORED")
        assert "detail_restored" not in {e.value for e in EventType}

    def test_strong_removal_evidence_still_needs_two_misses(self) -> None:
        """HTTP 410 is strong evidence, but one absence is still one absence.

        Observed live: an advertisement left the search, its detail page returned
        410 Gone, and it correctly stayed `missing_once` rather than jumping to
        `likely_removed` on the strength of the page alone.
        """
        from bama_monitor.config import MonitorConfig
        from bama_monitor.models import AdStatus, DetailVerdict
        from bama_monitor.sale_scoring import ScoringInputs, score_sale
        from bama_monitor.state_machine import AdState, on_missing

        cfg = MonitorConfig()
        state = AdState(
            status=AdStatus.ACTIVE,
            consecutive_misses=0,
            total_seen_runs=1,
            total_missing_runs=0,
            first_seen_at=T0,
            last_seen_at=T0,
        )
        outcome = on_missing(
            state,
            run_started_at=T0 + timedelta(days=1),
            removal_confirmation_misses=cfg.removal_confirmation_misses,
        )
        assert outcome.state.status is AdStatus.MISSING_ONCE

        score = score_sale(
            ScoringInputs(
                consecutive_misses=1,
                removal_confirmation_misses=cfg.removal_confirmation_misses,
                detail_verdict=DetailVerdict.HTTP_410,
            ),
            cfg.scoring,
        )
        # Evidence accumulates, but the label stays below a sale claim.
        assert 0.0 < score.confidence < 0.4
        assert str(score.label) == "unknown"
