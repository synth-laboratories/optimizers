from __future__ import annotations

from typing import Any, ClassVar, Protocol


class HostedOptimizerConfig(Protocol):
    """Submit-ready hosted optimizer config."""

    algorithm: ClassVar[Any]

    def to_config_json(self) -> dict[str, Any]: ...

