"""Typed coercion helpers for untrusted JSON.

Every response field may be absent, ``null`` or the wrong shape. These helpers
make that explicit at the call site and keep the parsers free of repeated
``isinstance`` ladders.
"""

from __future__ import annotations

from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Return ``value`` if it is a list, else an empty list."""
    return value if isinstance(value, list) else []


def sub_dict(node: Any, *path: str) -> dict[str, Any]:
    """Walk a nested path, returning an empty dict at the first non-dict."""
    current: Any = node
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def unwrap_data(payload: Any) -> Any:
    """Unwrap Bama's ``{status, errors, metadata, data}`` envelope when present.

    Some endpoints return the envelope and some return the payload directly, so
    both shapes have to be accepted.
    """
    if isinstance(payload, dict) and "data" in payload and len(payload) <= 5:
        return payload["data"]
    return payload
