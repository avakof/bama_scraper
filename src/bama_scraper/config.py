"""Runtime configuration: defaults, YAML file, and CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

SCRAPER_VERSION = "1.0.0"

DEFAULT_URL = (
    "https://bama.ir/car?year=1397-2018,&price=1000000000&body=passenger_car&country=iranian"
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class Config(BaseModel):
    """All tunables. Every field is overridable from the CLI."""

    model_config = ConfigDict(extra="forbid")

    url: str = DEFAULT_URL
    mode: str = "auto"

    # browser
    headed: bool = False
    user_agent: str = DEFAULT_USER_AGENT
    locale: str = "fa-IR"
    viewport_width: int = 1440
    viewport_height: int = 900

    # discovery / scrolling
    max_scrolls: int = 2000
    stale_cycles: int = 8
    max_runtime: float = 7200.0
    scroll_pause_ms: int = 1200
    settle_ms: int = 2500
    api_page_size: int = 30
    #: consecutive empty API pages required before declaring the end
    api_stale_pages: int = 2

    # detail scraping
    concurrency: int = 2
    delay_min: float = 0.4
    delay_max: float = 1.2
    max_retries: int = 4
    request_timeout: float = 45.0
    navigation_timeout: float = 60.0
    recycle_context_every: int = 200

    # behaviour
    resume: bool = True
    refresh: bool = False
    download_images: bool = False

    # output
    output_dir: Path = Path("output")
    log_level: str = "INFO"
    save_failed_html: bool = True

    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.output_dir / "bama_ads.sqlite"

    @property
    def debug_dir(self) -> Path:
        return self.output_dir / "debug"

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.debug_dir, self.raw_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config` from an optional YAML file plus CLI overrides.

    ``None`` overrides are dropped so that unset CLI flags do not clobber
    values coming from the YAML file.
    """
    data: dict[str, Any] = {}
    if path:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    clean = {k: v for k, v in overrides.items() if v is not None}
    data.update(clean)
    if "output_dir" in data and data["output_dir"] is not None:
        data["output_dir"] = Path(data["output_dir"])
    return Config(**data)
