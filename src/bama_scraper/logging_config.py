"""Structured logging setup (structlog over stdlib logging)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog once per process."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    # Third-party chatter would otherwise drown the progress lines.
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "bama_scraper") -> Any:
    return structlog.get_logger(name)
