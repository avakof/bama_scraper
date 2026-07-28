"""Repost detection: linking a removed advertisement to a new one for the same car.

A seller who removes a listing and republishes the same vehicle produces a
disappearance that looks exactly like a sale. Detecting that is therefore a
precision requirement, not a nice-to-have: an undetected repost inflates the sale
rate, and a *false* repost suppresses a genuine one.

The design consequence is that no single field can decide a match. Scores are
accumulated from several independent signals — taxonomy (brand/model/year),
seller identity, mileage and price proximity, city, image perceptual hashes and
description similarity — and a configurable threshold must be cleared. Two
different cars of the same model and year will share the taxonomy signal alone,
which is deliberately not enough.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .config import RepostConfig
from .models import RepostMatch


@dataclass(slots=True)
class VehicleProfile:
    """Normalized attributes used for fingerprinting and matching."""

    platform_ad_id: str
    brand: str | None = None
    model: str | None = None
    trim: str | None = None
    year: str | None = None
    mileage_km: int | None = None
    price_toman: int | None = None
    city: str | None = None
    seller_name: str | None = None
    seller_type: str | None = None
    body_color: str | None = None
    interior_color: str | None = None
    description: str | None = None
    image_hashes: tuple[str, ...] = ()

    def taxonomy_key(self) -> str | None:
        if not (self.brand and self.model and self.year):
            return None
        return f"{_norm(self.brand)}|{_norm(self.model)}|{_norm(self.year)}"


def _norm(value: str | None) -> str:
    """Lowercase, collapse whitespace, drop ZWNJ so Persian variants align."""
    if not value:
        return ""
    text = value.replace("‌", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def bucket(value: int | None, size: int) -> int | None:
    return None if value is None or size <= 0 else value // size


def fingerprint(profile: VehicleProfile, cfg: RepostConfig) -> str:
    """Coarse fingerprint for cheap candidate lookup.

    Deliberately built from *bucketed* mileage and price rather than exact values,
    because a repost is usually re-entered by hand and rarely identical. The
    fingerprint narrows candidates; the scorer decides.
    """
    parts = [
        _norm(profile.brand),
        _norm(profile.model),
        _norm(profile.year),
        str(bucket(profile.mileage_km, cfg.mileage_bucket_km)),
        _norm(profile.city),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def description_similarity(left: str | None, right: str | None) -> float:
    """Ratio in ``[0, 1]``; 0 when either side is missing or trivially short."""
    a, b = _norm(left), _norm(right)
    if len(a) < 25 or len(b) < 25:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def hamming_hex(left: str, right: str) -> int | None:
    """Hamming distance between two equal-length hex perceptual hashes."""
    if not left or not right or len(left) != len(right):
        return None
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return None


def image_overlap(left: tuple[str, ...], right: tuple[str, ...], *, max_distance: int = 8) -> int:
    """Count image pairs that are near-identical by perceptual hash.

    Exact-equality would miss re-uploaded or re-compressed photos, which is the
    common case in a repost, so a small Hamming distance counts as a match.
    """
    if not left or not right:
        return 0
    matches = 0
    for a in left:
        for b in right:
            if a == b:
                matches += 1
                break
            distance = hamming_hex(a, b)
            if distance is not None and distance <= max_distance:
                matches += 1
                break
    return matches


def score_pair(
    parent: VehicleProfile, child: VehicleProfile, cfg: RepostConfig
) -> tuple[float, list[str], dict[str, Any]]:
    """Score one candidate pair. Returns ``(score, matched_signals, evidence)``."""
    score = 0.0
    matched: list[str] = []
    evidence: dict[str, Any] = {}

    parent_taxonomy, child_taxonomy = parent.taxonomy_key(), child.taxonomy_key()
    if parent_taxonomy and parent_taxonomy == child_taxonomy:
        score += cfg.weight_brand_model_year
        matched.append("brand_model_year")
        evidence["taxonomy"] = parent_taxonomy

    if parent.seller_name and _norm(parent.seller_name) == _norm(child.seller_name):
        score += cfg.weight_same_seller
        matched.append("same_seller")
        evidence["seller"] = parent.seller_name

    if parent.mileage_km is not None and child.mileage_km is not None:
        delta = abs(parent.mileage_km - child.mileage_km)
        evidence["mileage_delta_km"] = delta
        if delta <= cfg.mileage_tolerance_km:
            score += cfg.weight_mileage_close
            matched.append("mileage_close")

    if parent.price_toman and child.price_toman:
        ratio = abs(parent.price_toman - child.price_toman) / max(parent.price_toman, 1)
        evidence["price_delta_ratio"] = round(ratio, 4)
        if ratio <= cfg.price_tolerance_ratio:
            score += cfg.weight_price_close
            matched.append("price_close")

    if parent.city and _norm(parent.city) == _norm(child.city):
        score += cfg.weight_city
        matched.append("city")

    overlap = image_overlap(parent.image_hashes, child.image_hashes)
    if overlap:
        score += cfg.weight_image_hash
        matched.append("image_hash")
        evidence["matching_images"] = overlap

    similarity = description_similarity(parent.description, child.description)
    if similarity >= cfg.description_similarity_min:
        score += cfg.weight_description_similar
        matched.append("description_similar")
    if similarity:
        evidence["description_similarity"] = round(similarity, 4)

    if (
        parent.body_color
        and _norm(parent.body_color) == _norm(child.body_color)
        and parent.interior_color
        and _norm(parent.interior_color) == _norm(child.interior_color)
    ):
        score += cfg.weight_colors
        matched.append("colors")

    return round(min(score, 1.0), 4), matched, evidence


def find_reposts(
    candidates: list[VehicleProfile],
    parents: list[VehicleProfile],
    cfg: RepostConfig,
) -> list[RepostMatch]:
    """Match each new advertisement against recently removed ones.

    Only the single best-scoring parent above the threshold is linked, so one
    removed listing cannot be claimed by several new ones.
    """
    if not cfg.enabled:
        return []
    matches: list[RepostMatch] = []
    claimed: set[str] = set()

    for child in candidates:
        best: RepostMatch | None = None
        for parent in parents:
            if parent.platform_ad_id == child.platform_ad_id:
                continue
            if parent.platform_ad_id in claimed:
                continue
            score, matched, evidence = score_pair(parent, child, cfg)
            if score < cfg.match_threshold:
                continue
            # Taxonomy alone must never be sufficient: two different cars of the
            # same model and year would otherwise be linked.
            if matched == ["brand_model_year"]:
                continue
            if best is None or score > best.score:
                best = RepostMatch(
                    new_platform_ad_id=child.platform_ad_id,
                    parent_platform_ad_id=parent.platform_ad_id,
                    score=score,
                    matched_on=matched,
                    evidence=evidence,
                )
        if best is not None:
            claimed.add(best.parent_platform_ad_id)
            matches.append(best)
    return matches


def vehicle_entity_id(parent_id: str, child_id: str, existing: str | None = None) -> str:
    """Stable identifier grouping advertisement ids for one physical vehicle.

    An existing group id is preserved so a vehicle reposted three times keeps one
    entity across all of them, rather than fragmenting into pairs.
    """
    if existing:
        return existing
    seed = "|".join(sorted((parent_id, child_id)))
    return "veh_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


METHODOLOGY = """\
Repost-detection methodology
============================

Problem: a seller removes a listing and republishes the same car under a new
advertisement id. The disappearance is indistinguishable from a sale unless the
new listing is linked back to the old one.

Approach: a coarse fingerprint (brand, model, year, bucketed mileage, city) is
used to shortlist candidates cheaply; then each candidate pair is scored across
independent signals, each contributing a configured weight:

    brand + model + year        0.30
    same seller                 0.20
    matching image hashes       0.20
    mileage within tolerance    0.15
    similar description         0.15
    price within tolerance      0.10
    same city                   0.05
    matching body + interior    0.05

A pair must clear `match_threshold` (default 0.70) to be recorded, and taxonomy
alone is explicitly rejected -- two different cars of the same model and year
would otherwise match. Each removed advertisement can be claimed by at most one
new advertisement, and only the highest-scoring pair is linked.

A confirmed link has three effects: the new advertisement records
`repost_parent_ad_id`, the parent moves to status `reposted`, and the parent's
sale confidence is *reduced*. A repost means the vehicle is still on the market,
so it is never counted as a sale.

Both advertisements are also assigned a shared `vehicle_entity_id`, which is what
downstream survival analysis groups on -- that column, not the advertisement id,
represents the physical car.
"""
