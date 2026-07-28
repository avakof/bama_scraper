"""Pydantic schemas for discovered ads, detail records and run metadata.

Every field that may be absent on a real listing is optional. Missing data is
represented as ``None`` and is never coerced to ``0`` or ``""``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AdStatus = Literal[
    "discovered",
    "pending",
    "scraping",
    "completed",
    "retryable_error",
    "permanent_error",
]

DiscoveryMode = Literal["auto", "api", "pagination", "scroll"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class DiscoveredAd(_Base):
    """A listing as seen on the search-results surface (card level)."""

    ad_id: str
    url: str
    source_search_url: str

    card_title: str | None = None
    card_subtitle: str | None = None

    price_text: str | None = None
    price_toman: int | None = None
    price_type: str | None = None

    year_text: str | None = None
    year_jalali: int | None = None
    year_gregorian: int | None = None

    mileage_text: str | None = None
    mileage_km: int | None = None

    location_text: str | None = None
    thumbnail_url: str | None = None

    is_promoted: bool | None = None
    discovery_position: int | None = None
    discovery_cycle: int | None = None
    discovered_at: str | None = None

    status: AdStatus = "discovered"


class AdMedia(_Base):
    """One media item belonging to an advertisement."""

    ad_id: str
    position: int
    kind: Literal["image", "video"] = "image"
    url: str | None = None
    original_url: str | None = None
    thumb_url: str | None = None
    alt: str | None = None


class AdDetail(_Base):
    """Fully parsed detail page. Raw text is preserved next to normalized values."""

    # --- identity -----------------------------------------------------------
    ad_id: str
    url: str
    canonical_url: str | None = None
    title: str | None = None
    brand: str | None = None
    brand_fa: str | None = None
    model: str | None = None
    model_fa: str | None = None
    trim: str | None = None
    trim_fa: str | None = None
    year_text: str | None = None
    year_jalali: int | None = None
    year_gregorian: int | None = None
    vehicle_category: str | None = None
    body_type: str | None = None
    body_type_fa: str | None = None
    manufacturer_country: str | None = None

    # --- price & sale terms -------------------------------------------------
    price_text: str | None = None
    price_toman: int | None = None
    price_currency: str = "IRT"
    price_type: str | None = None
    is_negotiable: bool | None = None
    is_installment: bool | None = None
    down_payment_text: str | None = None
    down_payment_toman: int | None = None
    installment_amount_text: str | None = None
    installment_amount_toman: int | None = None
    installment_months: int | None = None
    is_exchange: bool | None = None
    is_presale: bool | None = None
    price_hidden: bool | None = None
    delivery_days: int | None = None

    # --- vehicle attributes -------------------------------------------------
    mileage_text: str | None = None
    mileage_km: int | None = None
    is_zero_km: bool | None = None
    transmission: str | None = None
    fuel_type: str | None = None
    engine: str | None = None
    engine_volume_text: str | None = None
    drivetrain: str | None = None
    body_color: str | None = None
    interior_color: str | None = None
    body_status: str | None = None
    chassis_condition: str | None = None
    paint_condition: str | None = None
    technical_condition: str | None = None
    insurance_text: str | None = None
    insurance_months_remaining: int | None = None
    inspection_text: str | None = None
    cylinders: str | None = None
    doors: str | None = None
    power_text: str | None = None
    torque_text: str | None = None
    acceleration_text: str | None = None
    fuel_consumption_text: str | None = None

    # --- advertisement info -------------------------------------------------
    description: str | None = None
    seller_type: str | None = None
    seller_name: str | None = None
    seller_authenticated: bool | None = None
    dealer_url: str | None = None
    province: str | None = None
    city: str | None = None
    neighbourhood: str | None = None
    location_text: str | None = None
    published_text: str | None = None
    published_ts: float | None = None
    modified_date: str | None = None
    view_count: int | None = None
    badges: list[str] = Field(default_factory=list)
    is_promoted: bool | None = None

    # --- media --------------------------------------------------------------
    primary_image_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    thumbnail_urls: list[str] = Field(default_factory=list)
    video_urls: list[str] = Field(default_factory=list)
    media_count: int | None = None

    # --- raw evidence -------------------------------------------------------
    raw_attributes: dict[str, Any] = Field(default_factory=dict)
    json_ld: list[dict[str, Any]] = Field(default_factory=list)
    html_sha256: str | None = None
    scraper_version: str | None = None
    scraped_at: str | None = None
    parse_status: Literal["ok", "partial", "failed"] = "ok"
    parse_warnings: list[str] = Field(default_factory=list)
    #: Which evidence source the fields came from: nuxt_data | json_ld | dom.
    source: str | None = None
    #: How the HTML was retrieved: httpx | browser. Kept separate from
    #: ``source`` so the parse evidence is never overwritten by transport info.
    fetch_source: str | None = None


class NetworkCandidate(_Base):
    """A JSON/XHR endpoint observed while inspecting the live search page."""

    url: str
    method: str
    status: int | None = None
    resource_type: str | None = None
    is_listing_endpoint: bool = False
    top_level_keys: list[str] = Field(default_factory=list)
    ad_count: int | None = None
    notes: str | None = None


class RunSummary(_Base):
    """Everything needed to audit a single run after the fact."""

    run_id: str
    started_at: str
    finished_at: str | None = None
    target_url: str
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    mode: str | None = None
    scraper_version: str | None = None

    total_discovered: int = 0
    duplicate_urls: int = 0
    duplicate_ad_ids: int = 0
    details_completed: int = 0
    details_failed: int = 0
    with_price: int = 0
    without_price: int = 0
    with_images: int = 0
    without_images: int = 0
    first_url: str | None = None
    last_url: str | None = None
    cycles: int = 0
    stale_cycles_at_end: int = 0
    termination_reason: str | None = None
    elapsed_seconds: float | None = None
    errors: int = 0
    parse_warnings: int = 0
    verification_pass: dict[str, Any] = Field(default_factory=dict)
