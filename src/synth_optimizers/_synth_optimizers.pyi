from __future__ import annotations

from typing import Any

__version__: str


class SynthOptimizerError(RuntimeError):
    error_code: str


class ConfigError(SynthOptimizerError): ...


class ContainerContractError(SynthOptimizerError): ...


class ProposerError(SynthOptimizerError): ...


class CacheMissError(SynthOptimizerError): ...


class CancelledError(SynthOptimizerError): ...


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


def gepa_serve(
    db_path: str,
    bind_addr: str = "127.0.0.1:8879",
    worker_id: str | None = None,
    lease_seconds: int = 3600,
) -> None: ...


def gepa_service_run_next(
    db_path: str,
    worker_id: str = "synth-gepa-worker",
    lease_seconds: int = 3600,
) -> dict[str, Any]: ...

def gepa_service_tick(
    db_path: str,
    worker_id: str = "synth-gepa-worker",
    lease_seconds: int = 3600,
) -> dict[str, Any]: ...


def gepa_service_recover(db_path: str) -> dict[str, Any]: ...


def workspace_status(path: str) -> dict[str, Any]: ...


def workspace_submit_run_request(
    db_path: str,
    config_path: str,
    priority: int = 0,
) -> dict[str, Any]: ...


def workspace_claim_next_run_request(
    db_path: str,
    lease_id: str,
    worker_id: str | None = None,
    lease_seconds: int = 3600,
) -> dict[str, Any] | None: ...


def workspace_heartbeat_run_request(
    db_path: str,
    request_id: str,
    lease_id: str,
    lease_seconds: int = 3600,
) -> dict[str, Any] | None: ...


def workspace_start_run_request(db_path: str, request_id: str) -> dict[str, Any]: ...


def workspace_complete_run_request(db_path: str, request_id: str) -> dict[str, Any]: ...


def workspace_fail_run_request(
    db_path: str,
    request_id: str,
    error_message: str,
    reason_code: str | None = None,
) -> dict[str, Any]: ...


def workspace_cancel_run_request(
    db_path: str,
    request_id: str,
    reason: str = "cancelled",
) -> dict[str, Any]: ...


def workspace_recover_expired_run_requests(db_path: str) -> list[dict[str, Any]]: ...


def workspace_claim_next_optimizer_job(
    db_path: str,
    run_id: str,
    lease_id: str,
    worker_id: str | None = None,
    lease_seconds: int = 300,
) -> dict[str, Any] | None: ...


def workspace_claim_optimizer_job(
    db_path: str,
    run_id: str,
    job_id: str,
    lease_id: str,
    worker_id: str | None = None,
    lease_seconds: int = 300,
) -> dict[str, Any] | None: ...


def workspace_mark_optimizer_job_running(
    db_path: str,
    run_id: str,
    job_id: str,
    lease_id: str,
    lease_seconds: int = 300,
) -> dict[str, Any] | None: ...


def workspace_heartbeat_optimizer_job(
    db_path: str,
    run_id: str,
    job_id: str,
    lease_id: str,
    lease_seconds: int = 300,
) -> dict[str, Any] | None: ...


def workspace_recover_expired_optimizer_jobs(
    db_path: str,
    run_id: str,
) -> list[dict[str, Any]]: ...
