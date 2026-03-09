"""Small local utility helpers used by the mirrored SDK."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


def run_sync(coro: Any, *, label: str | None = None) -> Any:
    """Run a coroutine from sync code."""
    del label
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Synchronous wrapper called from an active event loop")


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load TOML configuration into a mapping."""
    import tomllib

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("TOML root must decode to a mapping")
    return payload

