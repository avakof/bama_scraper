"""Runtime configuration for the deep scraper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEEP_VERSION = "1.0.0"

#: Repository root, derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

DEFAULT_REFERER = (
    "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian"
)


class DeepConfig(BaseModel):
    """All tunables. Every field is overridable from the CLI."""

    model_config = ConfigDict(extra="forbid")

    #: Existing scraper database, opened read-only as the seed inventory.
    source_db: Path = REPO_ROOT / "output" / "bama_ads.sqlite"
    output_dir: Path = REPO_ROOT / "deep_scraper" / "data"

    # request discipline -- mirrors the base scraper's conservative defaults
    concurrency: int = 2
    delay_min: float = 0.4
    delay_max: float = 1.2
    max_retries: int = 4
    request_timeout: float = 45.0
    recycle_context_every: int = 200
    user_agent: str = DEFAULT_USER_AGENT
    referer: str = DEFAULT_REFERER

    # budget / bounds
    max_runtime: float = 10800.0
    #: Hard ceiling so retries can never silently blow the request budget.
    max_requests: int = 8000

    # behaviour
    phases: str = "a,b,c,d"
    skip_api: bool = False
    skip_html: bool = False
    refresh: bool = False
    retry_delisted: bool = False
    download_images: bool = False
    dry_run: bool = False

    # audit thresholds
    max_critical_conflict_rate: float = 0.01

    # export options
    raw_description: bool = False
    no_dealer_address: bool = False

    log_level: str = "INFO"
    json_logs: bool = False

    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.output_dir / "bama_deep.sqlite"

    @property
    def snapshot_path(self) -> Path:
        return self.output_dir / "bama_deep_snapshot.sqlite"

    @property
    def exports_dir(self) -> Path:
        return self.output_dir / "exports"

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    def enabled_phases(self) -> set[str]:
        return {p.strip().lower() for p in self.phases.split(",") if p.strip()}

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None, **overrides: Any) -> DeepConfig:
    """Build a config from an optional YAML file plus CLI overrides.

    ``None`` overrides are dropped so unset flags do not clobber file values.
    """
    data: dict[str, Any] = {}
    if path:
        candidate = Path(path)
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("source_db", "output_dir"):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return DeepConfig(**data)
