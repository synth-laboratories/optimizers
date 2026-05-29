from __future__ import annotations

from typing import Any

__version__: str


class SynthOptimizerError(RuntimeError):
    error_code: str


class ConfigError(SynthOptimizerError): ...


class ContainerContractError(SynthOptimizerError): ...


class ProposerError(SynthOptimizerError): ...


class CacheMissError(SynthOptimizerError): ...


class CacheFullError(SynthOptimizerError): ...


class CacheCorruptError(SynthOptimizerError): ...


class BudgetExceededError(SynthOptimizerError): ...


class OptimizerDiskBudgetError(SynthOptimizerError): ...


class CancelledError(SynthOptimizerError): ...


class RunFailedError(SynthOptimizerError): ...


class InvariantError(SynthOptimizerError): ...


class EventCompareError(SynthOptimizerError): ...


class StateTransitionError(SynthOptimizerError): ...


class OptimizerIoError(SynthOptimizerError): ...


class OptimizerJsonError(SynthOptimizerError): ...


class OptimizerTomlDecodeError(SynthOptimizerError): ...


class OptimizerHttpError(SynthOptimizerError): ...


class OptimizerSqliteError(SynthOptimizerError): ...


class GepaRun:
    config_path: str

    @staticmethod
    def from_toml(path: str) -> GepaRun: ...

    def execute(self) -> GepaRunResult: ...


class GepaRunResult:
    best_candidate: dict[str, Any]
    manifest_path: str
    event_feed_path: str
    normalized_event_feed_path: str
    cache_profile_path: str
    candidate_registry_path: str
    frontier_path: str
    score_chart_path: str
    run_registry_path: str
    workspace_db_path: str
    artifact_refs: list[dict[str, Any]]
    cost_usd: float
    usage: dict[str, Any]
    state_history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]: ...


def events_replay(path: str) -> str: ...


def events_compare(left: str, right: str) -> bool: ...


# The standing HTTP service is the public worker/workspace surface. Queue
# submission, claiming, run lifecycle control, and workspace status are driven
# through its routes rather than direct native bindings.
def gepa_serve(
    db_path: str,
    bind_addr: str = "127.0.0.1:8879",
    worker_id: str | None = None,
    lease_seconds: int = 3600,
) -> None: ...
