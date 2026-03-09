"""High-level local prompt-learning job wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import PolicyCandidate, PolicyCandidatePage, PromptLearningResult
from .configs.prompt_learning import PromptLearningConfig
from .runtime import RUNTIME, get_candidate_typed, list_candidates_typed
from .utils import load_toml, run_sync
from ..policy.v1 import PolicyOptimizationOfflineJob


@dataclass
class PromptLearningJobConfig:
    """Configuration for a local prompt-learning job."""

    backend_url: str
    api_key: str
    config_path: Path | None = None
    config_dict: dict[str, Any] | None = None
    container_api_key: str | None = field(default=None, repr=False)
    container_key: str | None = field(default=None, repr=False)
    container_worker_token: str | None = field(default=None, repr=False)
    allow_experimental: bool | None = None
    overrides: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        has_path = self.config_path is not None
        has_dict = self.config_dict is not None
        if has_path == has_dict:
            raise ValueError("Provide exactly one of config_path or config_dict")
        if self.config_path is not None and not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if self.container_api_key is None and isinstance(self.container_key, str):
            stripped = self.container_key.strip()
            self.container_api_key = stripped or None

    def materialize(self) -> dict[str, Any]:
        payload = load_toml(self.config_path) if self.config_path is not None else dict(self.config_dict or {})
        if self.overrides:
            payload = _merge_overrides(payload, dict(self.overrides))
        return payload


def _merge_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    for dotted_key, value in overrides.items():
        current = merged
        parts = str(dotted_key).split(".")
        for part in parts[:-1]:
            existing = current.get(part)
            if not isinstance(existing, dict):
                existing = {}
                current[part] = existing
            current = existing
        current[parts[-1]] = value
    return merged


class PromptLearningJobPoller:
    """Small local poller."""

    def poll_job(self, job_id: str) -> dict[str, Any]:
        return RUNTIME.get_job(job_id).status_payload()


class PromptLearningJob:
    """High-level local prompt-learning job."""

    def __init__(
        self,
        config: PromptLearningJobConfig,
        job_id: str | None = None,
        skip_health_check: bool = False,
    ) -> None:
        self.config = config
        self._job_id = job_id
        self._skip_health_check = skip_health_check
        self._offline_job: PolicyOptimizationOfflineJob | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: str | None = None,
        api_key: str | None = None,
        container_worker_token: str | None = None,
        allow_experimental: bool | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "PromptLearningJob":
        return cls(
            PromptLearningJobConfig(
                config_path=Path(config_path),
                backend_url=str(backend_url or "local://prompt-opt"),
                api_key=str(api_key or "local"),
                container_worker_token=container_worker_token,
                allow_experimental=allow_experimental,
                overrides=overrides,
            )
        )

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        backend_url: str | None = None,
        api_key: str | None = None,
        container_worker_token: str | None = None,
        allow_experimental: bool | None = None,
        overrides: dict[str, Any] | None = None,
        skip_health_check: bool = False,
    ) -> "PromptLearningJob":
        return cls(
            PromptLearningJobConfig(
                config_dict=dict(config_dict),
                backend_url=str(backend_url or "local://prompt-opt"),
                api_key=str(api_key or "local"),
                container_worker_token=container_worker_token,
                allow_experimental=allow_experimental,
                overrides=overrides,
            ),
            skip_health_check=skip_health_check,
        )

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: str | None = None,
        api_key: str | None = None,
    ) -> "PromptLearningJob":
        config = PromptLearningJobConfig(
            config_dict={"prompt_learning": {"algorithm": "gepa"}},
            backend_url=str(backend_url or "local://prompt-opt"),
            api_key=str(api_key or "local"),
        )
        return cls(config=config, job_id=job_id)

    def _kind_from_payload(self, payload: dict[str, Any]) -> str:
        parsed = PromptLearningConfig.from_mapping(payload)
        return "mipro_offline" if parsed.algorithm == "mipro" else "gepa_offline"

    def submit(self) -> str:
        if self._job_id is not None:
            raise RuntimeError(f"Job already submitted: {self._job_id}")
        payload = self.config.materialize()
        system_name = _resolve_system_name(payload)
        offline_job = PolicyOptimizationOfflineJob.create(
            kind=self._kind_from_payload(payload),
            technique="discrete_optimization",
            system_name=system_name,
            config=payload,
            metadata={"skip_health_check": self._skip_health_check},
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
        )
        self._offline_job = offline_job
        self._job_id = offline_job.job_id
        return offline_job.job_id

    @property
    def job_id(self) -> str | None:
        return self._job_id

    def _ensure_offline_job(self) -> PolicyOptimizationOfflineJob:
        if self._job_id is None:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        if self._offline_job is None:
            self._offline_job = PolicyOptimizationOfflineJob.get(
                self._job_id,
                backend_url=self.config.backend_url,
                api_key=self.config.api_key,
            )
        return self._offline_job

    async def get_status_async(self) -> dict[str, Any]:
        return await self._ensure_offline_job().status_async()

    def get_status(self) -> dict[str, Any]:
        return run_sync(self.get_status_async())

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: list[Any] | None = None,
        on_event: Any | None = None,
    ) -> PromptLearningResult:
        payload = await self._ensure_offline_job().stream_until_complete_async(
            timeout=timeout,
            interval=interval,
            handlers=handlers,
            on_event=on_event,
        )
        return PromptLearningResult.from_response(self._job_id or "", payload)

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: list[Any] | None = None,
        on_event: Any | None = None,
    ) -> PromptLearningResult:
        return run_sync(
            self.stream_until_complete_async(
                timeout=timeout,
                interval=interval,
                handlers=handlers,
                on_event=on_event,
            )
        )

    async def get_results_async(self) -> dict[str, Any]:
        return dict(RUNTIME.get_job(self._job_id or "").result_payload)

    def get_results(self) -> dict[str, Any]:
        return run_sync(self.get_results_async())

    async def list_candidates_async(
        self,
        *,
        algorithm: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        sort: str | None = None,
        include: str | None = None,
    ) -> dict[str, Any]:
        del algorithm, mode, cursor, sort, include
        return RUNTIME.list_candidates(self._job_id or "", limit=limit, status=status)

    async def list_candidates_typed_async(
        self,
        *,
        algorithm: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        sort: str | None = None,
        include: str | None = None,
    ) -> PolicyCandidatePage:
        del algorithm, mode, cursor, sort, include
        return list_candidates_typed(self._job_id or "", limit=limit, status=status)

    def list_candidates(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self.list_candidates_async(**kwargs))

    def list_candidates_typed(self, **kwargs: Any) -> PolicyCandidatePage:
        return run_sync(self.list_candidates_typed_async(**kwargs))

    async def get_candidate_async(self, candidate_id: str) -> dict[str, Any]:
        return RUNTIME.get_candidate(self._job_id or "", candidate_id)

    async def get_candidate_typed_async(self, candidate_id: str) -> PolicyCandidate:
        return get_candidate_typed(self._job_id or "", candidate_id)

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        return run_sync(self.get_candidate_async(candidate_id))

    def get_candidate_typed(self, candidate_id: str) -> PolicyCandidate:
        return run_sync(self.get_candidate_typed_async(candidate_id))

    async def submit_candidates_async(self, **kwargs: Any) -> dict[str, Any]:
        return await self._ensure_offline_job().submit_candidates_async(**kwargs)

    def submit_candidates(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self.submit_candidates_async(**kwargs))

    async def get_state_baseline_info_async(self) -> dict[str, Any]:
        return await self._ensure_offline_job().get_state_baseline_info_async()

    def get_state_baseline_info(self) -> dict[str, Any]:
        return run_sync(self.get_state_baseline_info_async())

    async def get_state_envelope_async(self) -> dict[str, Any]:
        return await self._ensure_offline_job().get_state_envelope_async()

    def get_state_envelope(self) -> dict[str, Any]:
        return run_sync(self.get_state_envelope_async())


def _resolve_system_name(payload: dict[str, Any]) -> str:
    top_level = payload.get("prompt_learning", payload)
    if isinstance(top_level, dict):
        system_name = top_level.get("system_name") or top_level.get("env_name")
        if isinstance(system_name, str) and system_name.strip():
            return system_name.strip()
        algorithm = top_level.get("algorithm")
        if isinstance(algorithm, str) and algorithm.strip():
            return f"{algorithm.strip()}-local-job"
    return "prompt-learning-local-job"
