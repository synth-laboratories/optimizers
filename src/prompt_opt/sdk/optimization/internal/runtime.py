"""Shared in-process runtime for local offline GEPA and MIPRO jobs."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib import request
from uuid import uuid4

from ..models import PolicyCandidate, PolicyCandidatePage, PromptLearningResult
from .configs.prompt_learning import (
    GEPAConfig,
    MIPROAlgorithmConfig,
    MessagePatternConfig,
    PromptCandidateConfig,
    PromptLearningConfig,
    PromptStageConfig,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Cannot block on a coroutine from an active event loop")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _clean_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _render_pattern(pattern: str, values: dict[str, Any]) -> str:
    rendered = pattern
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", _normalize_text(value))
    return rendered


def _first_message_text(stage: PromptStageConfig) -> str:
    if not stage.messages:
        return ""
    ordered = sorted(stage.messages, key=lambda item: item.order)
    return ordered[0].pattern


def _find_instruction_index(stage: PromptStageConfig) -> int:
    ordered = sorted(enumerate(stage.messages), key=lambda item: item[1].order)
    for original_index, message in ordered:
        if message.role == "system":
            return original_index
    return 0


def _extract_stage_text(stage: PromptStageConfig) -> str:
    if not stage.messages:
        return ""
    return stage.messages[_find_instruction_index(stage)].pattern


def _normalize_instruction_fragment(value: str) -> str:
    return "\n".join(line.strip() for line in value.strip().splitlines() if line.strip())


def _append_stage_text_once(base_text: str, fragment: str) -> str:
    normalized_fragment = _normalize_instruction_fragment(fragment)
    if not normalized_fragment:
        return base_text.strip()
    paragraphs = {
        _normalize_instruction_fragment(part)
        for part in base_text.split("\n\n")
        if _normalize_instruction_fragment(part)
    }
    if normalized_fragment in paragraphs:
        return base_text.strip()
    if not base_text.strip():
        return fragment.strip()
    return f"{base_text.rstrip()}\n\n{fragment.strip()}".strip()


def _replace_stage_text(stage: PromptStageConfig, updated_text: str) -> PromptStageConfig:
    messages = list(stage.messages)
    if not messages:
        messages = [MessagePatternConfig(role="system", pattern=updated_text, order=0)]
        return PromptStageConfig(
            id=stage.id,
            name=stage.name,
            messages=messages,
            wildcards=dict(stage.wildcards),
            metadata=dict(stage.metadata),
        )

    target_index = _find_instruction_index(stage)
    updated_messages: list[MessagePatternConfig] = []
    for index, message in enumerate(messages):
        if index == target_index:
            updated_messages.append(
                MessagePatternConfig(role=message.role, pattern=updated_text, order=message.order)
            )
        else:
            updated_messages.append(message)
    return PromptStageConfig(
        id=stage.id,
        name=stage.name,
        messages=updated_messages,
        wildcards=dict(stage.wildcards),
        metadata=dict(stage.metadata),
    )


@dataclass(frozen=True)
class AtomicInstructionTransform:
    """One stage-local prompt edit."""

    transform_id: str
    stage_id: str
    description: str
    appended_text: str


@dataclass
class CandidateBundle:
    """Internal multi-stage candidate representation."""

    candidate_id: str
    stages: list[PromptStageConfig]
    parent_id: str | None
    generation: int
    transform_ids_by_stage: dict[str, str | None]
    created_at: str
    avg_reward: float | None = None
    status: str = "proposed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def rendered_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for stage in self.stages:
            ordered = sorted(stage.messages, key=lambda item: item.order)
            for message in ordered:
                rendered.append(
                    {
                        "role": message.role,
                        "content": _render_pattern(message.pattern, example),
                        "stage_id": stage.id,
                        "stage_name": stage.name,
                    }
                )
        return rendered

    def candidate_content(self) -> str:
        parts: list[str] = []
        for stage in self.stages:
            stage_name = stage.name or stage.id or "stage"
            text = _extract_stage_text(stage)
            parts.append(f"[{stage_name}]\n{text}".strip())
        return "\n\n".join(part for part in parts if part.strip())

    def legacy_prompt_map(self) -> dict[str, str]:
        content = self.candidate_content()
        legacy: dict[str, str] = {"candidate_content": content}
        if self.stages:
            first_stage = self.stages[0]
            legacy["system_prompt"] = _extract_stage_text(first_stage)
            legacy["prompt"] = legacy["system_prompt"]
            legacy["instruction"] = legacy["system_prompt"]
        return legacy

    def to_payload(
        self,
        *,
        job_id: str,
        system_id: str,
        system_name: str,
        algorithm: str,
    ) -> dict[str, Any]:
        candidate_payload = {
            "candidate_id": self.candidate_id,
            "job_id": job_id,
            "system_id": system_id,
            "system_name": system_name,
            "algorithm": algorithm,
            "mode": "offline",
            "status": self.status,
            "avg_reward": self.avg_reward,
            "created_at": self.created_at,
            "updated_at": _utc_now_iso(),
            "generation": self.generation,
            "parent_id": self.parent_id,
            "metadata": dict(self.metadata),
            "candidate": {
                "candidate_id": self.candidate_id,
                "stages": [
                    {
                        "id": stage.id,
                        "name": stage.name,
                        "messages": [
                            {
                                "role": message.role,
                                "pattern": message.pattern,
                                "content": message.pattern,
                                "order": message.order,
                            }
                            for message in sorted(stage.messages, key=lambda item: item.order)
                        ],
                        "wildcards": dict(stage.wildcards),
                        "metadata": dict(stage.metadata),
                    }
                    for stage in self.stages
                ],
            },
            "stages": [
                {
                    "id": stage.id,
                    "name": stage.name,
                    "messages": [
                        {
                            "role": message.role,
                            "pattern": message.pattern,
                            "content": message.pattern,
                            "order": message.order,
                        }
                        for message in sorted(stage.messages, key=lambda item: item.order)
                    ],
                    "wildcards": dict(stage.wildcards),
                    "metadata": dict(stage.metadata),
                }
                for stage in self.stages
            ],
            "transform_ids_by_stage": dict(self.transform_ids_by_stage),
            "candidate_content": self.candidate_content(),
            "content": self.candidate_content(),
            "artifact_payload": {"text": self.candidate_content()},
            "messages": self.rendered_messages({}),
        }
        candidate_payload.update(self.legacy_prompt_map())
        return candidate_payload


@dataclass
class SeedEvalRecord:
    candidate_id: str
    split: str
    seed: int
    reward: float
    rollout_id: str
    success: bool
    metadata: dict[str, Any]
    created_at: str

    def to_payload(self, job_id: str) -> dict[str, Any]:
        return {
            "seed_eval_id": f"{job_id}:{self.candidate_id}:{self.split}:{self.seed}:{self.rollout_id}",
            "job_id": job_id,
            "candidate_id": self.candidate_id,
            "split": self.split,
            "seed": self.seed,
            "reward": self.reward,
            "avg_reward": self.reward,
            "success": self.success,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


class AsyncDiscreteTpe:
    """Small async-friendly discrete TPE implementation."""

    def __init__(
        self,
        *,
        seed: int = 0,
        gamma: float = 0.3,
        n_candidates: int = 32,
        n_startup_trials: int = 5,
        alpha: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        self._rng = random.Random(seed)
        self._gamma = max(0.05, min(gamma, 0.95))
        self._n_candidates = max(4, int(n_candidates))
        self._n_startup_trials = max(1, int(n_startup_trials))
        self._alpha = max(float(alpha), 1e-6)
        self._epsilon = max(float(epsilon), 1e-9)
        self._trials: list[tuple[dict[str, Any], float]] = []
        self._seen_signatures: set[tuple[tuple[str, str], ...]] = set()

    @staticmethod
    def _signature(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((key, repr(value)) for key, value in config.items()))

    async def tell(self, config: dict[str, Any], score: float) -> None:
        signature = self._signature(config)
        self._seen_signatures.add(signature)
        self._trials.append((dict(config), float(score)))

    def _uniform_config(self, search_space: dict[str, list[Any]]) -> dict[str, Any]:
        return {key: self._rng.choice(values) for key, values in search_space.items()}

    def _startup_config(self, search_space: dict[str, list[Any]]) -> dict[str, Any]:
        candidate: dict[str, Any] = {}
        trial_index = len(self._trials)
        for dimension_index, (key, values) in enumerate(sorted(search_space.items())):
            if not values:
                continue
            offset = 1 if len(values) > 1 and values[0] is None else 0
            choice_index = (trial_index + dimension_index + offset) % len(values)
            candidate[key] = values[choice_index]
        return candidate

    def _smoothed_probability(
        self,
        *,
        counts: Counter,
        total: int,
        choices: list[Any],
    ) -> dict[Any, float]:
        denominator = total + self._alpha * max(len(choices), 1)
        return {
            choice: (counts.get(repr(choice), 0) + self._alpha) / max(denominator, self._epsilon)
            for choice in choices
        }

    async def suggest(
        self,
        search_space: dict[str, list[Any]],
        *,
        taboo_signatures: set[tuple[tuple[str, str], ...]] | None = None,
    ) -> dict[str, Any] | None:
        taboo = taboo_signatures or set()
        if not search_space:
            return {}

        if len(self._trials) < self._n_startup_trials:
            candidate = self._startup_config(search_space)
            signature = self._signature(candidate)
            if signature not in self._seen_signatures and signature not in taboo:
                return candidate
            for _ in range(256):
                candidate = self._uniform_config(search_space)
                signature = self._signature(candidate)
                if signature not in self._seen_signatures and signature not in taboo:
                    return candidate
            return None

        ranked = sorted(self._trials, key=lambda item: item[1], reverse=True)
        split_index = max(1, int(math.ceil(len(ranked) * self._gamma)))
        good = ranked[:split_index]
        bad = ranked[split_index:] or ranked[-1:]

        per_dimension_good: dict[str, Counter] = {}
        per_dimension_bad: dict[str, Counter] = {}
        for dimension, _choices in search_space.items():
            per_dimension_good[dimension] = Counter(repr(config[dimension]) for config, _score in good)
            per_dimension_bad[dimension] = Counter(repr(config[dimension]) for config, _score in bad)

        best_candidate: dict[str, Any] | None = None
        best_acquisition = float("-inf")
        for _ in range(self._n_candidates):
            candidate: dict[str, Any] = {}
            acquisition = 0.0
            for dimension, choices in search_space.items():
                good_probs = self._smoothed_probability(
                    counts=per_dimension_good[dimension],
                    total=len(good),
                    choices=choices,
                )
                bad_probs = self._smoothed_probability(
                    counts=per_dimension_bad[dimension],
                    total=len(bad),
                    choices=choices,
                )
                weights = [
                    max((good_probs[choice] / max(bad_probs[choice], self._epsilon)), self._epsilon)
                    for choice in choices
                ]
                picked = self._rng.choices(choices, weights=weights, k=1)[0]
                candidate[dimension] = picked
                acquisition += math.log(good_probs[picked] + self._epsilon) - math.log(
                    bad_probs[picked] + self._epsilon
                )

            signature = self._signature(candidate)
            if signature in self._seen_signatures or signature in taboo:
                continue
            if acquisition > best_acquisition:
                best_candidate = candidate
                best_acquisition = acquisition

        if best_candidate is not None:
            return best_candidate

        for _ in range(256):
            candidate = self._uniform_config(search_space)
            signature = self._signature(candidate)
            if signature not in self._seen_signatures and signature not in taboo:
                return candidate
        return None


@dataclass
class LocalJobRecord:
    """Mutable in-memory job state."""

    job_id: str
    kind: str
    technique: str
    system_id: str
    system_name: str
    config: PromptLearningConfig
    raw_config: dict[str, Any]
    metadata: dict[str, Any]
    auto_start: bool
    created_at: str
    api_version: str = "v1"
    state: str = "created"
    result_payload: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_versions: dict[str, int] = field(default_factory=dict)
    seed_evals: list[SeedEvalRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    artifacts_payload: dict[str, Any] = field(default_factory=dict)
    trial_queue: list[dict[str, Any]] = field(default_factory=list)
    rollout_queue: list[dict[str, Any]] = field(default_factory=list)
    baseline_info: dict[str, Any] = field(default_factory=dict)
    state_envelope: dict[str, Any] = field(default_factory=dict)
    checkpoint_payload: dict[str, Any] = field(default_factory=dict)
    best_candidate_id: str | None = None
    selected_candidate_id: str | None = None
    baseline_candidate_id: str | None = None
    best_reward: float | None = None
    error_message: str | None = None
    local_runtime_options: dict[str, Any] = field(default_factory=dict)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock()),
        repr=False,
    )
    _next_event_seq: int = field(default=1, repr=False)
    _next_candidate_version: int = field(default=1, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)

    def append_event(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._condition:
            event = {
                "seq": self._next_event_seq,
                "event_type": event_type,
                "timestamp": _utc_now_iso(),
                "job_id": self.job_id,
                "system_id": self.system_id,
                "payload": dict(payload or {}),
            }
            self._next_event_seq += 1
            self.events.append(event)
            self._condition.notify_all()
            return event

    def wait_if_paused(self) -> None:
        with self._condition:
            while self.state == "paused" and not self._cancel_requested:
                self._condition.wait(timeout=0.2)
            if self._cancel_requested:
                raise RuntimeError("job_cancelled")

    def check_cancelled(self) -> None:
        with self._condition:
            if self._cancel_requested:
                raise RuntimeError("job_cancelled")

    def set_state(self, state: str) -> None:
        with self._condition:
            self.state = state
            self._condition.notify_all()

    def request_cancel(self) -> None:
        with self._condition:
            self._cancel_requested = True
            self._condition.notify_all()

    def register_candidate(self, bundle: CandidateBundle, *, algorithm: str) -> dict[str, Any]:
        with self._condition:
            payload = bundle.to_payload(
                job_id=self.job_id,
                system_id=self.system_id,
                system_name=self.system_name,
                algorithm=algorithm,
            )
            self.candidates[bundle.candidate_id] = payload
            self.candidate_versions[bundle.candidate_id] = self._next_candidate_version
            self._next_candidate_version += 1
            self.selected_candidate_id = bundle.candidate_id
            if self.baseline_candidate_id is None:
                self.baseline_candidate_id = bundle.candidate_id
            self._refresh_envelopes()
            self._condition.notify_all()
            return payload

    def update_candidate_score(
        self,
        candidate_id: str,
        *,
        reward: float,
        status: str = "evaluated",
    ) -> None:
        with self._condition:
            candidate = self.candidates[candidate_id]
            candidate["avg_reward"] = reward
            candidate["score"] = reward
            candidate["status"] = status
            candidate["updated_at"] = _utc_now_iso()
            if self.best_reward is None or reward >= self.best_reward:
                self.best_reward = reward
                self.best_candidate_id = candidate_id
                self.selected_candidate_id = candidate_id
            self._refresh_envelopes()
            self._condition.notify_all()

    def record_seed_eval(self, seed_eval: SeedEvalRecord) -> None:
        with self._condition:
            self.seed_evals.append(seed_eval)
            self._refresh_envelopes()
            self._condition.notify_all()

    def _refresh_envelopes(self) -> None:
        lever_summary = {
            "prompt_lever_id": "prompt_lever",
            "candidate_lever_versions": dict(self.candidate_versions),
            "best_candidate_id": self.best_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "lever_count": len(self.candidates),
            "mutation_count": max(0, len(self.candidates) - 1),
            "latest_version": max(self.candidate_versions.values(), default=0),
        }
        state_value = {
            "job_id": self.job_id,
            "system_id": self.system_id,
            "algorithm_kind": "mipro" if "mipro" in self.kind else "gepa",
            "execution_mode": "retrieved",
            "best_candidate_id": self.best_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "best_reward": self.best_reward,
            "rollout_count": len(self.seed_evals),
            "candidate_lever_versions": dict(self.candidate_versions),
            "lever_summary": lever_summary,
            "candidates": dict(self.candidates),
            "seed_evals": [item.to_payload(self.job_id) for item in self.seed_evals],
        }
        self.state_envelope = {
            "job_id": self.job_id,
            "system_id": self.system_id,
            "state": state_value,
            "baseline_info": dict(self.baseline_info),
        }
        self.checkpoint_payload = {
            "job_id": self.job_id,
            "system_id": self.system_id,
            "created_at": self.created_at,
            "state": state_value,
            "config": self.raw_config,
        }
        best_candidate = (
            dict(self.candidates[self.best_candidate_id]) if self.best_candidate_id in self.candidates else None
        )
        self.result_payload = {
            "job_id": self.job_id,
            "status": self.state,
            "system": {"id": self.system_id, "name": self.system_name},
            "best_reward": self.best_reward,
            "best_score": self.best_reward,
            "best_candidate_id": self.best_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "best_candidate": best_candidate,
            "lever_summary": lever_summary,
            "lever_versions": dict(self.candidate_versions),
            "best_lever_version": self.candidate_versions.get(
                self.best_candidate_id or self.baseline_candidate_id or "",
            ),
            "result": {
                "best_candidate": best_candidate,
                "best_reward": self.best_reward,
                "lever_summary": lever_summary,
                "lever_versions": dict(self.candidate_versions),
            },
        }
        if self.error_message:
            self.result_payload["error"] = self.error_message

    def status_payload(self) -> dict[str, Any]:
        with self._condition:
            payload = {
                "job_id": self.job_id,
                "kind": self.kind,
                "technique": self.technique,
                "state": self.state,
                "status": self.state,
                "created_at": self.created_at,
                "updated_at": _utc_now_iso(),
                "system": {"id": self.system_id, "name": self.system_name},
                "best_reward": self.best_reward,
                "best_score": self.best_reward,
                "best_candidate_id": self.best_candidate_id,
                "selected_candidate_id": self.selected_candidate_id,
                "baseline_candidate_id": self.baseline_candidate_id,
                "lever_summary": self.result_payload.get("lever_summary", {}),
                "lever_versions": self.result_payload.get("lever_versions", {}),
                "best_lever_version": self.result_payload.get("best_lever_version"),
            }
            if self.error_message:
                payload["error"] = self.error_message
            return payload


class LocalPromptLearningRuntime:
    """Singleton local runtime backing the mirrored SDK."""

    TERMINAL_STATES = {"succeeded", "failed", "cancelled"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, LocalJobRecord] = {}
        self._systems_by_name: dict[str, str] = {}
        self._systems_by_id: dict[str, str] = {}

    def list_jobs(
        self,
        *,
        state: str | None = None,
        kind: str | None = None,
        system_id: str | None = None,
        system_name: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        with self._lock:
            items = list(self._jobs.values())
        filtered: list[dict[str, Any]] = []
        for job in items:
            if state and job.state != state:
                continue
            if kind and job.kind != kind:
                continue
            if system_id and job.system_id != system_id:
                continue
            if system_name and job.system_name != system_name:
                continue
            filtered.append(job.status_payload())
        filtered.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return {"items": filtered[: max(1, limit)], "next_cursor": None}

    def get_job(self, job_id: str) -> LocalJobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise ValueError(f"Unknown local job_id={job_id!r}") from exc

    def create_job(
        self,
        *,
        kind: str,
        technique: str,
        system_name: str,
        system_id: str | None,
        reuse_system: bool,
        config: dict[str, Any],
        metadata: dict[str, Any] | None,
        auto_start: bool,
        api_version: str,
    ) -> LocalJobRecord:
        parsed_config = PromptLearningConfig.from_mapping(config)
        local_runtime_options = self._resolve_local_runtime_options(config, parsed_config)

        with self._lock:
            resolved_system_id = system_id
            if reuse_system and system_name in self._systems_by_name:
                resolved_system_id = self._systems_by_name[system_name]
            if not resolved_system_id:
                resolved_system_id = _new_id("system")
            job_id = _new_id("pl")
            record = LocalJobRecord(
                job_id=job_id,
                kind=kind,
                technique=technique,
                system_id=resolved_system_id,
                system_name=system_name,
                config=parsed_config,
                raw_config=dict(config),
                metadata=dict(metadata or {}),
                auto_start=auto_start,
                created_at=_utc_now_iso(),
                api_version=api_version,
                local_runtime_options=local_runtime_options,
            )
            record.append_event("prompt_learning.created", {"status": "created"})
            self._jobs[job_id] = record
            self._systems_by_name[system_name] = resolved_system_id
            self._systems_by_id[resolved_system_id] = system_name

        if auto_start:
            self.start_job(job_id)
        return record

    def _resolve_local_runtime_options(
        self,
        raw_config: dict[str, Any],
        parsed_config: PromptLearningConfig,
    ) -> dict[str, Any]:
        root_payload = dict(raw_config.get("prompt_learning", raw_config))
        options = root_payload.get("local_runtime")
        if not isinstance(options, dict):
            options = {}
        mipro_payload = root_payload.get("mipro")
        if isinstance(mipro_payload, dict):
            if "parallel_batches" in mipro_payload and "parallel_batches" not in options:
                options["parallel_batches"] = mipro_payload["parallel_batches"]
            if "parallel_batch_size" in mipro_payload and "parallel_batch_size" not in options:
                options["parallel_batch_size"] = mipro_payload["parallel_batch_size"]
        if parsed_config.container_url and "container_url" not in options:
            options["container_url"] = parsed_config.container_url
        if parsed_config.execution_mode and "execution_mode" not in options:
            options["execution_mode"] = parsed_config.execution_mode
        return options

    def start_job(self, job_id: str) -> None:
        record = self.get_job(job_id)
        with record._condition:
            if record._thread is not None and record._thread.is_alive():
                return
            record.state = "queued"
            record.append_event("prompt_learning.queued", {"status": "queued"})
            record._thread = threading.Thread(
                target=self._run_job,
                args=(job_id,),
                daemon=True,
            )
            record._thread.start()

    def _run_job(self, job_id: str) -> None:
        record = self.get_job(job_id)
        record.set_state("running")
        record.append_event("prompt_learning.running", {"status": "running"})
        try:
            if "mipro" in record.kind:
                result = _run_sync(self._run_mipro(record))
            else:
                result = _run_sync(self._run_gepa(record))
            with record._condition:
                if record.state not in self.TERMINAL_STATES:
                    record.state = "succeeded"
                if isinstance(result, dict):
                    record.result_payload.update(result)
                record._refresh_envelopes()
            record.append_event(
                "prompt_learning.succeeded",
                {"best_candidate_id": record.best_candidate_id, "best_reward": record.best_reward},
            )
        except RuntimeError as exc:
            if str(exc) == "job_cancelled":
                with record._condition:
                    record.state = "cancelled"
                    record.error_message = "cancelled"
                    record._refresh_envelopes()
                record.append_event("prompt_learning.cancelled", {"status": "cancelled"})
                return
            with record._condition:
                record.state = "failed"
                record.error_message = str(exc)
                record._refresh_envelopes()
            record.append_event("prompt_learning.failed", {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            with record._condition:
                record.state = "failed"
                record.error_message = str(exc)
                record._refresh_envelopes()
            record.append_event("prompt_learning.failed", {"error": str(exc)})

    def _resolve_algorithm_config(self, record: LocalJobRecord) -> GEPAConfig | MIPROAlgorithmConfig:
        if "mipro" in record.kind:
            if record.config.mipro is None:
                return MIPROAlgorithmConfig(
                    initial_candidate=PromptCandidateConfig(stages=[]),
                    execution_mode=record.config.execution_mode,
                )
            return record.config.mipro
        if record.config.gepa is None:
            return GEPAConfig(
                initial_candidate=PromptCandidateConfig(stages=[]),
                execution_mode=record.config.execution_mode,
            )
        return record.config.gepa

    def _seed_examples(
        self,
        task_data: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if task_data is None:
            return ([], [])

        train_examples = _as_list_of_dicts(task_data.train_examples)
        val_examples = _as_list_of_dicts(task_data.validation_examples)
        generic_examples = _as_list_of_dicts(task_data.examples)
        examples_by_seed = dict(task_data.examples_by_seed)
        train_pools = task_data.train_pools
        train_seeds: list[int] = []
        if train_pools is not None:
            train_seeds.extend(int(seed) for seed in train_pools.reflection_seeds)
            train_seeds.extend(int(seed) for seed in train_pools.pareto_seeds)
            train_seeds.extend(int(seed) for seed in train_pools.seeds)
        validation_seeds = [int(seed) for seed in task_data.validation_seeds]

        if not train_examples and not val_examples and generic_examples:
            split_index = max(1, len(generic_examples) - max(1, len(generic_examples) // 4))
            train_examples = generic_examples[:split_index]
            val_examples = generic_examples[split_index:] or generic_examples[:1]

        if examples_by_seed:
            if not train_examples:
                train_examples = [
                    {"seed": seed, **dict(examples_by_seed.get(str(seed), {}))}
                    for seed in train_seeds
                ]
            if not val_examples:
                val_examples = [
                    {"seed": seed, **dict(examples_by_seed.get(str(seed), {}))}
                    for seed in validation_seeds
                ]

        if not train_examples and train_seeds:
            train_examples = [{"seed": seed} for seed in train_seeds]
        if not val_examples and validation_seeds:
            val_examples = [{"seed": seed} for seed in validation_seeds]

        return (train_examples, val_examples or train_examples[:1])

    def _baseline_candidate(self, record: LocalJobRecord) -> CandidateBundle:
        algorithm_config = self._resolve_algorithm_config(record)
        initial_candidate = getattr(algorithm_config, "initial_candidate", None)
        stages = list(getattr(initial_candidate, "stages", []) or [])
        if not stages:
            stages = [
                PromptStageConfig(
                    id="default",
                    name="Default",
                    messages=[MessagePatternConfig(role="system", pattern="You are a helpful assistant.", order=0)],
                    wildcards={},
                    metadata={},
                )
            ]
        for index, stage in enumerate(stages):
            if not stage.id:
                stage.id = f"stage_{index}"
            if stage.name is None:
                stage.name = stage.id

        bundle = CandidateBundle(
            candidate_id="baseline",
            stages=stages,
            parent_id=None,
            generation=0,
            transform_ids_by_stage={stage.id or f"stage_{index}": None for index, stage in enumerate(stages)},
            created_at=_utc_now_iso(),
            metadata={"kind": "baseline"},
            status="evaluated",
        )
        payload = record.register_candidate(bundle, algorithm="mipro" if "mipro" in record.kind else "gepa")
        record.baseline_info = {
            "baseline_candidate_id": bundle.candidate_id,
            "candidate": payload,
            "stages": payload.get("stages", []),
        }
        record._refresh_envelopes()
        return bundle

    def _build_request_payload(
        self,
        record: LocalJobRecord,
        example: dict[str, Any],
        candidate_payload: dict[str, Any],
        *,
        split: str,
    ) -> dict[str, Any]:
        trace_correlation_id = f"{record.job_id}:{candidate_payload['candidate_id']}:{split}:{uuid4().hex[:8]}"
        env_seed = example.get("seed")
        request_payload = {
            "trace_correlation_id": trace_correlation_id,
            "env": {
                "env_name": record.system_name,
                "seed": env_seed,
                "config": {
                    "example": dict(example),
                    "split": split,
                },
            },
            "policy": {
                "policy_name": "prompt-opt-local",
                "config": {
                    "candidate_id": candidate_payload["candidate_id"],
                    "candidate": candidate_payload,
                    "candidate_content": candidate_payload.get("candidate_content"),
                    "stages": candidate_payload.get("stages"),
                },
            },
            "on_done": "reset",
            "safety": {"max_time_s": 300},
        }
        custom_builder = record.local_runtime_options.get("request_builder")
        if callable(custom_builder):
            payload = custom_builder(
                example=example,
                candidate=candidate_payload,
                trace_correlation_id=trace_correlation_id,
                split=split,
            )
            if isinstance(payload, dict):
                return payload
        return request_payload

    async def _evaluate_candidate(
        self,
        record: LocalJobRecord,
        candidate_id: str,
        examples: list[dict[str, Any]],
        *,
        split: str,
    ) -> float:
        candidate_payload = self.get_candidate(record.job_id, candidate_id)
        adapter = record.local_runtime_options.get("adapter")
        task_model = record.local_runtime_options.get("task_model")
        score_fn = record.local_runtime_options.get("score_fn")
        container_url = record.local_runtime_options.get("container_url")
        score_extractor = record.local_runtime_options.get("score_extractor")
        headers = dict(record.local_runtime_options.get("headers") or {})
        timeout_seconds = float(record.local_runtime_options.get("timeout_seconds", 30.0))
        parallel_enabled = bool(record.local_runtime_options.get("parallel_batches", True))
        max_parallel = int(
            record.local_runtime_options.get(
                "parallel_batch_size",
                record.local_runtime_options.get(
                    "max_concurrent_rollouts",
                    record.local_runtime_options.get("max_concurrent", 100),
                ),
            )
        )
        concurrency = max(1, min(len(examples), max_parallel if parallel_enabled else 1))

        async def evaluate_example(index: int, example: dict[str, Any]) -> SeedEvalRecord:
            await asyncio.to_thread(record.wait_if_paused)
            await asyncio.to_thread(record.check_cancelled)

            if adapter is not None:
                eval_batch = await asyncio.to_thread(
                    adapter.evaluate,
                    [example],
                    candidate_payload,
                    False,
                )
                score_list = getattr(eval_batch, "scores", None) or [0.0]
                reward = float(score_list[0])
                metadata = {
                    "adapter_output": getattr(eval_batch, "outputs", []),
                    "adapter_objective_scores": getattr(eval_batch, "objective_scores", []),
                }
            elif callable(task_model):
                messages = candidate_payload.get("candidate", {}).get("stages") or candidate_payload.get("stages", [])
                rendered_messages: list[dict[str, Any]] = []
                for stage in messages:
                    if not isinstance(stage, dict):
                        continue
                    for message in sorted(stage.get("messages", []), key=lambda item: item.get("order", 0)):
                        content = _render_pattern(str(message.get("pattern") or message.get("content") or ""), example)
                        rendered_messages.append(
                            {
                                "role": message.get("role", "user"),
                                "content": content,
                                "stage_id": stage.get("id"),
                            }
                        )
                prompt_text = "\n\n".join(item["content"] for item in rendered_messages if item["content"])
                task_output = await _maybe_await(task_model(prompt_text))
                if callable(score_fn):
                    reward = float(await _maybe_await(score_fn(example, candidate_payload, task_output)))
                else:
                    expected = example.get("answer", example.get("expected", example.get("label")))
                    reward = 1.0 if _normalize_text(task_output).strip() == _normalize_text(expected).strip() else 0.0
                metadata = {"expected": example.get("answer", example.get("expected")), "predicted": task_output}
            elif container_url:
                payload = self._build_request_payload(record, example, candidate_payload, split=split)
                body = json.dumps(payload).encode("utf-8")
                http_headers = {"content-type": "application/json", **headers}
                http_request = request.Request(
                    f"{str(container_url).rstrip('/')}/rollout",
                    data=body,
                    headers=http_headers,
                    method="POST",
                )
                with await asyncio.to_thread(request.urlopen, http_request, timeout=timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                if callable(score_extractor):
                    reward = float(score_extractor(response_payload))
                else:
                    metrics = response_payload.get("metrics") or response_payload.get("reward_info") or {}
                    reward = float(metrics.get("outcome_reward", metrics.get("reward", 0.0)))
                metadata = {
                    "response": response_payload,
                    "request": payload,
                }
            else:
                raise RuntimeError("Local runtime requires either local_runtime.task_model or container_url")

            reward = float(reward)
            return SeedEvalRecord(
                candidate_id=candidate_id,
                split=split,
                seed=int(example.get("seed", index)),
                reward=reward,
                rollout_id=f"{split}-{candidate_id}-{index}",
                success=True,
                metadata=metadata,
                created_at=_utc_now_iso(),
            )

        rewards: list[float] = []
        if concurrency == 1:
            for index, example in enumerate(examples):
                seed_eval = await evaluate_example(index, example)
                rewards.append(seed_eval.reward)
                record.record_seed_eval(seed_eval)
        else:
            semaphore = asyncio.Semaphore(concurrency)
            tasks = []
            try:
                for index, example in enumerate(examples):
                    async def run_one(i: int = index, ex: dict[str, Any] = example) -> SeedEvalRecord:
                        async with semaphore:
                            return await evaluate_example(i, ex)
                    tasks.append(asyncio.create_task(run_one()))
                for task in asyncio.as_completed(tasks):
                    seed_eval = await task
                    rewards.append(seed_eval.reward)
                    record.record_seed_eval(seed_eval)
            except Exception:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return sum(rewards) / max(1, len(rewards))

    def get_candidate(self, job_id: str, candidate_id: str) -> dict[str, Any]:
        record = self.get_job(job_id)
        with record._condition:
            try:
                return dict(record.candidates[candidate_id])
            except KeyError as exc:
                raise ValueError(f"Unknown candidate_id={candidate_id!r} for local job {job_id!r}") from exc

    def list_candidates(
        self,
        job_id: str,
        *,
        limit: int = 100,
        status: str | None = None,
    ) -> dict[str, Any]:
        record = self.get_job(job_id)
        with record._condition:
            items = list(record.candidates.values())
        if status is not None:
            items = [item for item in items if item.get("status") == status]
        items.sort(key=lambda item: str(item.get("created_at", "")))
        return {"items": items[: max(1, limit)], "next_cursor": None}

    def list_seed_evals(
        self,
        job_id: str,
        *,
        candidate_id: str | None = None,
        split: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        record = self.get_job(job_id)
        payloads = [item.to_payload(job_id) for item in record.seed_evals]
        if candidate_id is not None:
            payloads = [item for item in payloads if item.get("candidate_id") == candidate_id]
        if split is not None:
            payloads = [item for item in payloads if item.get("split") == split]
        return {"items": payloads[: max(1, limit)], "next_cursor": None}

    def update_job_state(self, job_id: str, *, action: str) -> dict[str, Any]:
        record = self.get_job(job_id)
        if action == "pause":
            record.set_state("paused")
            record.append_event("prompt_learning.paused", {"status": "paused"})
        elif action == "resume":
            if record.state == "created":
                self.start_job(job_id)
            else:
                record.set_state("running")
                record.append_event("prompt_learning.resumed", {"status": "running"})
        elif action == "cancel":
            record.request_cancel()
        return record.status_payload()

    async def _run_mipro(self, record: LocalJobRecord) -> dict[str, Any]:
        baseline = self._baseline_candidate(record)
        train_examples, val_examples = self._seed_examples(record.config.task_data)
        if not train_examples and not val_examples:
            raise RuntimeError("No task_data examples or seeds available for local MIPRO execution")
        if not val_examples:
            val_examples = train_examples[:1]

        baseline_score = await self._evaluate_candidate(record, baseline.candidate_id, val_examples, split="validation")
        record.update_candidate_score(baseline.candidate_id, reward=baseline_score)
        record.append_event(
            "prompt_learning.baseline.evaluated",
            {"candidate_id": baseline.candidate_id, "reward": baseline_score},
        )

        cfg = record.config.mipro or MIPROAlgorithmConfig()
        best_candidate_id = baseline.candidate_id
        best_score = baseline_score
        patience = 0
        transform_bank: dict[str, list[AtomicInstructionTransform]] = {}
        taboo: set[tuple[tuple[str, str], ...]] = set()
        tpe = AsyncDiscreteTpe(seed=int(cfg.seed), n_startup_trials=max(3, min(8, cfg.num_candidates)))

        baseline_stages = [
            PromptStageConfig.model_validate(stage.model_dump(mode="python"))
            if hasattr(stage, "model_dump")
            else PromptStageConfig.model_validate(stage)
            for stage in baseline.stages
        ]
        max_iterations = max(
            1,
            int(cfg.max_iterations or 1),
            int((cfg.termination_conditions.total_rollouts if cfg.termination_conditions else 0) // max(1, len(val_examples))) if cfg.termination_conditions and cfg.termination_conditions.total_rollouts else 1,
        )

        for iteration in range(max_iterations):
            record.wait_if_paused()
            record.check_cancelled()
            for stage in baseline_stages:
                stage_id = stage.id or "stage"
                if stage_id not in transform_bank:
                    transform_bank[stage_id] = []
                next_transforms = self._propose_stage_transforms(
                    stage,
                    train_examples,
                    iteration=iteration,
                    existing_count=len(transform_bank[stage_id]),
                )
                existing_ids = {item.transform_id for item in transform_bank[stage_id]}
                for transform in next_transforms:
                    if transform.transform_id not in existing_ids:
                        transform_bank[stage_id].append(transform)
                        existing_ids.add(transform.transform_id)

            search_space = {
                stage.id or "stage": [None] + [item.transform_id for item in transform_bank[stage.id or "stage"]]
                for stage in baseline_stages
            }
            config_choice = await tpe.suggest(search_space, taboo_signatures=taboo)
            if config_choice is None:
                break

            candidate_bundle = self._materialize_candidate(
                baseline=baseline,
                transform_bank=transform_bank,
                config_choice=config_choice,
                generation=iteration + 1,
                metadata={"algorithm": "mipro", "iteration": iteration},
            )
            record.register_candidate(candidate_bundle, algorithm="mipro")
            candidate_score = await self._evaluate_candidate(
                record,
                candidate_bundle.candidate_id,
                val_examples,
                split="validation",
            )
            record.update_candidate_score(candidate_bundle.candidate_id, reward=candidate_score)
            await tpe.tell(config_choice, candidate_score)

            improved = candidate_score > best_score + float(cfg.min_improvement)
            record.append_event(
                "prompt_learning.candidate.evaluated",
                {
                    "candidate_id": candidate_bundle.candidate_id,
                    "reward": candidate_score,
                    "iteration": iteration,
                    "improved": improved,
                },
            )
            record.append_event(
                "prompt_learning.iteration.completed",
                {
                    "iteration": iteration,
                    "candidate_id": candidate_bundle.candidate_id,
                    "reward": candidate_score,
                },
            )
            if improved:
                best_candidate_id = candidate_bundle.candidate_id
                best_score = candidate_score
                patience = 0
            else:
                patience += 1
            if patience >= max(1, int(cfg.early_stop_rounds)):
                break

        return {
            "best_candidate_id": best_candidate_id,
            "best_reward": best_score,
            "best_candidate": self.get_candidate(record.job_id, best_candidate_id),
        }

    async def _run_gepa(self, record: LocalJobRecord) -> dict[str, Any]:
        baseline = self._baseline_candidate(record)
        train_examples, val_examples = self._seed_examples(record.config.task_data)
        if not train_examples and not val_examples:
            raise RuntimeError("No task_data examples or seeds available for local GEPA execution")
        if not val_examples:
            val_examples = train_examples[:1]

        baseline_score = await self._evaluate_candidate(record, baseline.candidate_id, val_examples, split="validation")
        record.update_candidate_score(baseline.candidate_id, reward=baseline_score)
        best_candidate_id = baseline.candidate_id
        best_score = baseline_score

        cfg = record.config.gepa or GEPAConfig()
        population = cfg.population or type("Population", (), {"num_generations": 1, "children_per_generation": 4})()
        num_generations = max(1, int(population.num_generations))
        children_per_generation = max(1, int(population.children_per_generation))
        current_parent = baseline

        for generation in range(1, num_generations + 1):
            record.wait_if_paused()
            record.check_cancelled()
            record.append_event("prompt_learning.generation.started", {"generation": generation})
            generation_candidates: list[tuple[str, float]] = []
            base_transforms: dict[str, list[AtomicInstructionTransform]] = {}
            generation_parent = current_parent
            next_parent = current_parent
            for stage in generation_parent.stages:
                stage_id = stage.id or "stage"
                base_transforms[stage_id] = self._propose_stage_transforms(
                    stage,
                    train_examples,
                    iteration=generation,
                    existing_count=0,
                )

            for child_index in range(children_per_generation):
                config_choice: dict[str, Any] = {}
                for stage in current_parent.stages:
                    stage_id = stage.id or "stage"
                    transforms = base_transforms[stage_id]
                    picked = transforms[child_index % len(transforms)] if transforms else None
                    config_choice[stage_id] = None if picked is None else picked.transform_id
                candidate_bundle = self._materialize_candidate(
                    baseline=generation_parent,
                    transform_bank=base_transforms,
                    config_choice=config_choice,
                    generation=generation,
                    metadata={"algorithm": "gepa", "generation": generation},
                )
                record.register_candidate(candidate_bundle, algorithm="gepa")
                reward = await self._evaluate_candidate(
                    record,
                    candidate_bundle.candidate_id,
                    val_examples,
                    split="validation",
                )
                record.update_candidate_score(candidate_bundle.candidate_id, reward=reward)
                generation_candidates.append((candidate_bundle.candidate_id, reward))
                record.append_event(
                    "prompt_learning.candidate.evaluated",
                    {
                        "candidate_id": candidate_bundle.candidate_id,
                        "generation": generation,
                        "reward": reward,
                    },
                )
                if reward >= best_score:
                    best_candidate_id = candidate_bundle.candidate_id
                    best_score = reward
                    next_parent = candidate_bundle
                    record.append_event(
                        "prompt_learning.frontier.updated",
                        {
                            "generation": generation,
                            "best_candidate_id": best_candidate_id,
                            "best_reward": best_score,
                    },
                )

            current_parent = next_parent
            record.append_event(
                "prompt_learning.generation.completed",
                {
                    "generation": generation,
                    "best_candidate_id": best_candidate_id,
                    "best_reward": best_score,
                    "evaluated_candidates": generation_candidates,
                },
            )

        return {
            "best_candidate_id": best_candidate_id,
            "best_reward": best_score,
            "best_candidate": self.get_candidate(record.job_id, best_candidate_id),
        }

    def _propose_stage_transforms(
        self,
        stage: PromptStageConfig,
        train_examples: list[dict[str, Any]],
        *,
        iteration: int,
        existing_count: int,
    ) -> list[AtomicInstructionTransform]:
        base_text = _extract_stage_text(stage)
        labels = sorted(
            {
                _clean_key(example.get("answer") or example.get("expected") or example.get("label"))
                for example in train_examples
                if _clean_key(example.get("answer") or example.get("expected") or example.get("label"))
            }
        )
        label_hint = f"Return exactly one of: {', '.join(labels)}." if labels else "Return only the final answer."
        choices = [
            "Be concise and deterministic.",
            "Follow the requested output schema exactly.",
            "Think carefully, then respond with only the final answer.",
            label_hint,
            "State no preamble and no extra explanation.",
            "Prefer the highest-confidence answer consistent with the examples.",
        ]
        transforms: list[AtomicInstructionTransform] = []
        stage_id = stage.id or "stage"
        for offset, text in enumerate(choices):
            transform_id = f"{stage_id}-tx-{existing_count + iteration}-{offset}"
            if text.strip() in base_text:
                continue
            transforms.append(
                AtomicInstructionTransform(
                    transform_id=transform_id,
                    stage_id=stage_id,
                    description=text,
                    appended_text=text,
                )
            )
        return transforms or [
            AtomicInstructionTransform(
                transform_id=f"{stage_id}-tx-{existing_count + iteration}-fallback",
                stage_id=stage_id,
                description="Restate task requirements clearly.",
                appended_text="Restate the task requirements clearly and respond precisely.",
            )
        ]

    def _materialize_candidate(
        self,
        *,
        baseline: CandidateBundle,
        transform_bank: dict[str, list[AtomicInstructionTransform]],
        config_choice: dict[str, Any],
        generation: int,
        metadata: dict[str, Any],
    ) -> CandidateBundle:
        transform_lookup = {
            transform.transform_id: transform
            for transforms in transform_bank.values()
            for transform in transforms
        }
        stages: list[PromptStageConfig] = []
        transform_ids_by_stage: dict[str, str | None] = {}
        for stage in baseline.stages:
            stage_id = stage.id or "stage"
            selected_transform_id = config_choice.get(stage_id)
            transform_ids_by_stage[stage_id] = selected_transform_id
            if selected_transform_id is None:
                stages.append(stage)
                continue
            transform = transform_lookup[selected_transform_id]
            base_text = _extract_stage_text(stage)
            updated_text = _append_stage_text_once(base_text, transform.appended_text)
            stages.append(_replace_stage_text(stage, updated_text))
        return CandidateBundle(
            candidate_id=_new_id("cand"),
            stages=stages,
            parent_id=baseline.candidate_id,
            generation=generation,
            transform_ids_by_stage=transform_ids_by_stage,
            created_at=_utc_now_iso(),
            metadata=dict(metadata),
        )

    def create_prompt_learning_result(self, job_id: str) -> PromptLearningResult:
        record = self.get_job(job_id)
        return PromptLearningResult.from_response(job_id, record.result_payload)

    def wait_for_terminal(
        self,
        job_id: str,
        *,
        timeout: float,
        interval: float,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout
        record = self.get_job(job_id)
        while time.time() < deadline:
            status = record.status_payload()
            if status["status"] in self.TERMINAL_STATES:
                return record.result_payload or status
            time.sleep(interval)
        raise TimeoutError(f"Local job {job_id} did not reach a terminal state within {timeout} seconds")


RUNTIME = LocalPromptLearningRuntime()


def list_candidates_typed(job_id: str, *, limit: int = 100, status: str | None = None) -> PolicyCandidatePage:
    return PolicyCandidatePage.from_dict(RUNTIME.list_candidates(job_id, limit=limit, status=status))


def get_candidate_typed(job_id: str, candidate_id: str) -> PolicyCandidate:
    return PolicyCandidate.from_dict(RUNTIME.get_candidate(job_id, candidate_id))
