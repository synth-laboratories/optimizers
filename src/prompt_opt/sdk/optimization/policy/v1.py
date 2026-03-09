"""Local mirror of Synth offline policy optimization jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..internal.runtime import RUNTIME


@dataclass
class PolicyOptimizationOfflineJob:
    """Canonical local client for offline prompt optimization jobs."""

    job_id: str
    backend_url: str
    api_key: str
    system_id: str | None = None
    system_name: str | None = None
    timeout: float = 30.0
    api_version: str = "v1"

    @classmethod
    async def create_async(
        cls,
        *,
        technique: Literal["discrete_optimization"] = "discrete_optimization",
        kind: Literal["gepa_offline", "mipro_offline", "eval"],
        system_name: str,
        system_id: str | None = None,
        reuse_system: bool = True,
        config_mode: Literal["DEFAULT", "FULL"] = "DEFAULT",
        config: dict[str, Any],
        container_worker_token: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_start: bool = True,
        backend_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        api_version: str | None = None,
        prompt_opt_version: str = "v1",
        prompt_opt_fallback_policy: str | None = None,
    ) -> "PolicyOptimizationOfflineJob":
        del config_mode, container_worker_token, prompt_opt_version, prompt_opt_fallback_policy
        if not system_name or not system_name.strip():
            raise ValueError("system_name is required")
        record = RUNTIME.create_job(
            kind=kind,
            technique=technique,
            system_name=system_name.strip(),
            system_id=system_id,
            reuse_system=bool(reuse_system),
            config=config,
            metadata=metadata,
            auto_start=bool(auto_start),
            api_version=str(api_version or "v1"),
        )
        return cls(
            job_id=record.job_id,
            backend_url=str(backend_url or "local://prompt-opt"),
            api_key=str(api_key or "local"),
            system_id=record.system_id,
            system_name=record.system_name,
            timeout=timeout,
            api_version=str(api_version or "v1"),
        )

    @classmethod
    def create(cls, **kwargs: Any) -> "PolicyOptimizationOfflineJob":
        from ..internal.utils import run_sync

        return run_sync(cls.create_async(**kwargs))

    @classmethod
    async def get_async(
        cls,
        job_id: str,
        *,
        backend_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        api_version: str | None = None,
    ) -> "PolicyOptimizationOfflineJob":
        record = RUNTIME.get_job(job_id)
        return cls(
            job_id=record.job_id,
            backend_url=str(backend_url or "local://prompt-opt"),
            api_key=str(api_key or "local"),
            system_id=record.system_id,
            system_name=record.system_name,
            timeout=timeout,
            api_version=str(api_version or record.api_version),
        )

    @classmethod
    def get(cls, job_id: str, **kwargs: Any) -> "PolicyOptimizationOfflineJob":
        from ..internal.utils import run_sync

        return run_sync(cls.get_async(job_id, **kwargs))

    @classmethod
    async def list_async(
        cls,
        *,
        state: str | None = None,
        kind: Literal["gepa_offline", "mipro_offline", "eval"] | None = None,
        system_id: str | None = None,
        system_name: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        backend_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        api_version: str | None = None,
    ) -> dict[str, Any]:
        del created_after, created_before, cursor, backend_url, api_key, timeout, api_version
        return RUNTIME.list_jobs(
            state=state,
            kind=kind,
            system_id=system_id,
            system_name=system_name,
            limit=limit,
        )

    @classmethod
    def list(cls, **kwargs: Any) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(cls.list_async(**kwargs))

    async def status_async(self) -> dict[str, Any]:
        return RUNTIME.get_job(self.job_id).status_payload()

    def status(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.status_async())

    async def events_async(self, *, since_seq: int = 0, limit: int = 500) -> dict[str, Any]:
        record = RUNTIME.get_job(self.job_id)
        items = [event for event in record.events if int(event.get("seq", 0)) > since_seq]
        return {"items": items[: max(1, limit)], "next_cursor": None}

    def events(self, *, since_seq: int = 0, limit: int = 500) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.events_async(since_seq=since_seq, limit=limit))

    async def artifacts_async(self) -> dict[str, Any]:
        return dict(RUNTIME.get_job(self.job_id).artifacts_payload)

    def artifacts(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.artifacts_async())

    async def checkpoint_async(self) -> dict[str, Any]:
        return dict(RUNTIME.get_job(self.job_id).checkpoint_payload)

    def checkpoint(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.checkpoint_async())

    async def submit_candidates_async(
        self,
        *,
        algorithm_kind: str,
        candidates: list[dict[str, Any]],
        proposal_session_id: str | None = None,
        proposer_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind, proposal_session_id, proposer_metadata
        record = RUNTIME.get_job(self.job_id)
        inserted: list[dict[str, Any]] = []
        for raw_candidate in candidates or []:
            candidate_id = str(raw_candidate.get("candidate_id") or raw_candidate.get("id") or "")
            if not candidate_id:
                candidate_id = f"submitted_{len(record.candidates) + 1}"
            payload = dict(raw_candidate)
            payload.setdefault("candidate_id", candidate_id)
            payload.setdefault("job_id", record.job_id)
            payload.setdefault("system_id", record.system_id)
            payload.setdefault("system_name", record.system_name)
            payload.setdefault("status", "proposed")
            payload.setdefault("created_at", record.created_at)
            payload.setdefault("updated_at", record.created_at)
            payload.setdefault("avg_reward", None)
            record.candidates[candidate_id] = payload
            record.candidate_versions[candidate_id] = max(record.candidate_versions.values(), default=0) + 1
            inserted.append(payload)
        record._refresh_envelopes()
        return {"job_id": self.job_id, "items": inserted}

    def submit_candidates(self, **kwargs: Any) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.submit_candidates_async(**kwargs))

    async def get_state_baseline_info_async(self) -> dict[str, Any]:
        return dict(RUNTIME.get_job(self.job_id).baseline_info)

    def get_state_baseline_info(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.get_state_baseline_info_async())

    async def get_state_envelope_async(self) -> dict[str, Any]:
        return dict(RUNTIME.get_job(self.job_id).state_envelope)

    def get_state_envelope(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.get_state_envelope_async())

    async def list_trial_queue_async(self) -> dict[str, Any]:
        return {"items": list(RUNTIME.get_job(self.job_id).trial_queue), "next_cursor": None}

    def list_trial_queue(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.list_trial_queue_async())

    async def enqueue_trial_async(
        self,
        *,
        trial: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        record = RUNTIME.get_job(self.job_id)
        trial_payload = dict(trial or {})
        trial_payload.setdefault("trial_id", f"trial_{len(record.trial_queue) + 1}")
        record.trial_queue.append(trial_payload)
        return trial_payload

    def enqueue_trial(self, **kwargs: Any) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.enqueue_trial_async(**kwargs))

    async def update_trial_async(
        self,
        trial_id: str,
        *,
        patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        record = RUNTIME.get_job(self.job_id)
        for trial in record.trial_queue:
            if str(trial.get("trial_id")) == str(trial_id):
                trial.update(dict(patch or {}))
                return dict(trial)
        raise ValueError(f"Unknown trial_id={trial_id!r}")

    def update_trial(self, trial_id: str, **kwargs: Any) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.update_trial_async(trial_id, **kwargs))

    async def cancel_trial_async(
        self,
        trial_id: str,
        *,
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        record = RUNTIME.get_job(self.job_id)
        before = len(record.trial_queue)
        record.trial_queue = [trial for trial in record.trial_queue if str(trial.get("trial_id")) != str(trial_id)]
        return {"trial_id": trial_id, "removed": before - len(record.trial_queue)}

    def cancel_trial(self, trial_id: str, *, algorithm_kind: str | None = None) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.cancel_trial_async(trial_id, algorithm_kind=algorithm_kind))

    async def reorder_trials_async(
        self,
        *,
        trial_ids: list[str],
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        record = RUNTIME.get_job(self.job_id)
        order = {trial_id: index for index, trial_id in enumerate(trial_ids)}
        record.trial_queue.sort(key=lambda trial: order.get(str(trial.get("trial_id")), len(order)))
        return {"items": list(record.trial_queue)}

    def reorder_trials(self, **kwargs: Any) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.reorder_trials_async(**kwargs))

    async def apply_default_trial_plan_async(
        self,
        *,
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        return {"items": list(RUNTIME.get_job(self.job_id).trial_queue)}

    def apply_default_trial_plan(self, *, algorithm_kind: str | None = None) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.apply_default_trial_plan_async(algorithm_kind=algorithm_kind))

    async def get_rollout_queue_async(self) -> dict[str, Any]:
        return {"items": list(RUNTIME.get_job(self.job_id).rollout_queue), "next_cursor": None}

    def get_rollout_queue(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.get_rollout_queue_async())

    async def set_rollout_queue_policy_async(
        self,
        *,
        policy_patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        record = RUNTIME.get_job(self.job_id)
        record.metadata.setdefault("rollout_queue_policy", {}).update(dict(policy_patch or {}))
        return dict(record.metadata["rollout_queue_policy"])

    def set_rollout_queue_policy(
        self,
        *,
        policy_patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(
            self.set_rollout_queue_policy_async(
                policy_patch=policy_patch,
                algorithm_kind=algorithm_kind,
            )
        )

    async def get_rollout_dispatch_metrics_async(self) -> dict[str, Any]:
        record = RUNTIME.get_job(self.job_id)
        return {
            "queued": len(record.rollout_queue),
            "completed": len(record.seed_evals),
        }

    def get_rollout_dispatch_metrics(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.get_rollout_dispatch_metrics_async())

    async def get_rollout_limiter_status_async(self) -> dict[str, Any]:
        return {"limited": False, "reason": None}

    def get_rollout_limiter_status(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.get_rollout_limiter_status_async())

    async def retry_rollout_dispatch_async(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        return {"dispatch_id": dispatch_id, "retried": True}

    def retry_rollout_dispatch(self, dispatch_id: str, *, algorithm_kind: str | None = None) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.retry_rollout_dispatch_async(dispatch_id, algorithm_kind=algorithm_kind))

    async def drain_rollout_queue_async(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        del algorithm_kind
        record = RUNTIME.get_job(self.job_id)
        count = len(record.rollout_queue)
        if cancel_queued:
            record.rollout_queue = []
        return {"drained": count, "cancelled": bool(cancel_queued)}

    def drain_rollout_queue(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: str | None = None,
    ) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(
            self.drain_rollout_queue_async(
                cancel_queued=cancel_queued,
                algorithm_kind=algorithm_kind,
            )
        )

    async def pause_async(self) -> dict[str, Any]:
        return RUNTIME.update_job_state(self.job_id, action="pause")

    def pause(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.pause_async())

    async def resume_async(self) -> dict[str, Any]:
        return RUNTIME.update_job_state(self.job_id, action="resume")

    def resume(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.resume_async())

    async def cancel_async(self) -> dict[str, Any]:
        return RUNTIME.update_job_state(self.job_id, action="cancel")

    def cancel(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.cancel_async())

    async def restart_from_checkpoint_async(self) -> dict[str, Any]:
        record = RUNTIME.get_job(self.job_id)
        restarted = await self.create_async(
            technique=record.technique,
            kind=record.kind,
            system_name=record.system_name,
            system_id=record.system_id,
            reuse_system=True,
            config=record.raw_config,
            metadata=record.metadata,
            auto_start=True,
            backend_url=self.backend_url,
            api_key=self.api_key,
            timeout=self.timeout,
            api_version=self.api_version,
        )
        self.job_id = restarted.job_id
        return {"child_job_id": restarted.job_id, "parent_job_id": record.job_id}

    def restart_from_checkpoint(self) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.restart_from_checkpoint_async())

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 2.0,
        handlers: list[Any] | None = None,
        on_event: Any | None = None,
    ) -> dict[str, Any]:
        del handlers, on_event
        return RUNTIME.wait_for_terminal(self.job_id, timeout=timeout, interval=interval)

    def stream_until_complete(self, **kwargs: Any) -> dict[str, Any]:
        from ..internal.utils import run_sync

        return run_sync(self.stream_until_complete_async(**kwargs))

