"""GEPA-compatible local optimizer facade backed by the local offline SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from prompt_opt.adapters.synth_offline import SynthOfflineLearningAdapter
from prompt_opt.sdk.optimization.internal.prompt_learning import PromptLearningJob

try:
    from synth_ai.gepa.core.result import GEPAResult as SynthGEPAResult
except Exception:  # pragma: no cover - allows standalone use
    SynthGEPAResult = None


class LocalGEPAAdapterProtocol(Protocol):
    """Protocol compatible with GEPA adapter-style calls."""

    def evaluate(
        self,
        batch: list[Mapping[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ...


@dataclass(frozen=True)
class LocalGEPAResult:
    """Minimal GEPA-style result when synth_ai is unavailable."""

    candidates: list[dict[str, str]]
    parents: list[list[int | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[int, float]]
    per_val_instance_best_candidates: dict[int, set[int]]
    discovery_eval_counts: list[int]
    total_metric_calls: int

    @property
    def best_idx(self) -> int:
        return max(range(len(self.val_aggregate_scores)), key=self.val_aggregate_scores.__getitem__)

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[Mapping[str, Any]],
    valset: list[Mapping[str, Any]] | None = None,
    adapter: LocalGEPAAdapterProtocol | None = None,
    task_lm: str | Any | None = None,
    evaluator: Any | None = None,
    reflection_lm: str | Any | None = None,
    max_metric_calls: int | None = None,
    stop_callbacks: Any | None = None,
    **_: Any,
) -> Any:
    """Local GEPA-compatible optimize function.

    This function is shaped for `gepa-ai` compatibility and can be used as a
    drop-in local replacement where a simple adapter-based optimization loop is
    sufficient.
    """
    del task_lm, evaluator, reflection_lm, stop_callbacks
    if not seed_candidate:
        raise ValueError("seed_candidate must contain at least one entry.")
    if not trainset:
        raise ValueError("trainset must contain at least one item.")
    if adapter is None:
        raise ValueError("adapter is required for local/offline mode.")

    eval_set: list[Mapping[str, Any]] = list(valset if valset is not None else trainset)
    budget = max(1, int(max_metric_calls or 8))
    stages = [
        {
            "id": key,
            "name": key,
            "messages": [{"role": "system", "pattern": value, "order": 0}],
            "wildcards": {},
        }
        for key, value in seed_candidate.items()
    ]
    if not stages:
        stages = [
            {
                "id": "default",
                "name": "default",
                "messages": [{"role": "system", "pattern": "You are a helpful assistant.", "order": 0}],
                "wildcards": {},
            }
        ]

    prompt_learning_config = {
        "prompt_learning": {
            "algorithm": "gepa",
            "execution_mode": "retrieved",
            "task_data": {
                "train_examples": [dict(item) for item in trainset],
                "validation_examples": [dict(item) for item in eval_set],
            },
            "gepa": {
                "initial_candidate": {"stages": stages},
                "population": {
                    "initial_size": 1,
                    "num_generations": 1,
                    "children_per_generation": max(1, budget - 1),
                },
                "termination_conditions": {
                    "total_rollouts": budget * max(1, len(eval_set)),
                },
            },
            "local_runtime": {
                "adapter": adapter,
            },
        }
    }
    job = PromptLearningJob.from_dict(
        prompt_learning_config,
        backend_url="local://prompt-opt",
        api_key="local",
    )
    job.submit()
    job.stream_until_complete(timeout=300.0, interval=0.05).to_dict()
    candidates_page = job.list_candidates(limit=budget + 1)
    raw_candidates = candidates_page.get("items", [])

    candidates: list[dict[str, str]] = []
    parents: list[list[int | None]] = []
    scores: list[float] = []
    subscores: list[dict[int, float]] = []
    eval_counts: list[int] = []
    metric_calls_total = 0
    candidate_index_by_id: dict[str, int] = {}

    seed_eval_items = job.get_state_envelope().get("state", {}).get("seed_evals", [])
    per_candidate_per_seed: dict[str, dict[int, float]] = {}
    for seed_eval in seed_eval_items:
        if not isinstance(seed_eval, dict):
            continue
        candidate_id = str(seed_eval.get("candidate_id", ""))
        seed = int(seed_eval.get("seed", 0))
        reward = float(seed_eval.get("reward", 0.0))
        per_candidate_per_seed.setdefault(candidate_id, {})[seed] = reward

    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        stage_items = raw_candidate.get("stages") or raw_candidate.get("candidate", {}).get("stages") or []
        candidate_map: dict[str, str] = {}
        for stage in stage_items:
            if not isinstance(stage, dict):
                continue
            stage_key = str(stage.get("id") or stage.get("name") or f"stage_{len(candidate_map)}")
            for message in stage.get("messages", []):
                if isinstance(message, dict) and message.get("role") == "system":
                    text = message.get("pattern") or message.get("content")
                    if isinstance(text, str):
                        candidate_map[stage_key] = text
                        break
        if not candidate_map:
            candidate_map = {
                "default": str(raw_candidate.get("candidate_content", "")),
            }
        candidate_id = str(raw_candidate.get("candidate_id", ""))
        candidate_index_by_id[candidate_id] = len(candidates)
        candidates.append(candidate_map)
        parent_id = raw_candidate.get("parent_id")
        parents.append([candidate_index_by_id[parent_id]] if isinstance(parent_id, str) and parent_id in candidate_index_by_id else [None])
        candidate_subscores = per_candidate_per_seed.get(candidate_id, {})
        subscores.append(dict(candidate_subscores))
        score = float(raw_candidate.get("avg_reward", 0.0) or 0.0)
        scores.append(score)
        eval_counts.append(len(candidate_subscores))
        metric_calls_total += len(candidate_subscores)

    per_instance_best: dict[int, set[int]] = {}
    for example_idx in range(len(eval_set)):
        best_score = max((item.get(example_idx, float("-inf")) for item in subscores), default=float("-inf"))
        per_instance_best[example_idx] = {
            candidate_idx
            for candidate_idx, item in enumerate(subscores)
            if item.get(example_idx, float("-inf")) == best_score
        }

    if SynthGEPAResult is not None:
        return SynthGEPAResult(
            candidates=candidates,
            parents=parents,
            val_aggregate_scores=scores,
            val_subscores=subscores,
            per_val_instance_best_candidates=per_instance_best,
            discovery_eval_counts=eval_counts,
            total_metric_calls=metric_calls_total,
        )

    return LocalGEPAResult(
        candidates=candidates,
        parents=parents,
        val_aggregate_scores=scores,
        val_subscores=subscores,
        per_val_instance_best_candidates=per_instance_best,
        discovery_eval_counts=eval_counts,
        total_metric_calls=metric_calls_total,
    )


__all__ = [
    "LocalGEPAAdapterProtocol",
    "LocalGEPAResult",
    "SynthOfflineLearningAdapter",
    "optimize",
]
