"""Synth Container adapter for local prompt-opt evaluations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib import request


RolloutRequestBuilder = Callable[[Mapping[str, Any], dict[str, str], str], dict[str, Any]]
RolloutScoreExtractor = Callable[[dict[str, Any]], float]


def default_rollout_request_builder(
    *,
    example: Mapping[str, Any],
    candidate: dict[str, str],
    trace_correlation_id: str,
    env_name: str = "prompt-opt-local",
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a minimal Synth rollout request using the canonical container contract."""
    env_seed = seed
    if env_seed is None:
        raw_seed = example.get("seed")
        if isinstance(raw_seed, int):
            env_seed = raw_seed

    return {
        "trace_correlation_id": trace_correlation_id,
        "env": {
            "env_name": env_name,
            "seed": env_seed,
            "config": {
                "example": dict(example),
            },
        },
        "policy": {
            "policy_name": "prompt-opt",
            "config": {
                "candidate": dict(candidate),
            },
        },
        "on_done": "reset",
        "safety": {"max_time_s": 300},
    }


def default_rollout_score_extractor(response_payload: dict[str, Any]) -> float:
    """Extract the rollout reward from a Synth Container response payload."""
    reward_info = response_payload.get("metrics")
    if not isinstance(reward_info, dict):
        reward_info = response_payload.get("reward_info")
    if not isinstance(reward_info, dict):
        raise ValueError("rollout response is missing metrics/reward_info")
    reward = reward_info.get("outcome_reward")
    if reward is None:
        raise ValueError("rollout response metrics are missing outcome_reward")
    return float(reward)


@dataclass(frozen=True)
class ContainerEvaluationBatch:
    """Batch output shaped to GEPA adapter expectations."""

    outputs: list[dict[str, Any]]
    scores: list[float]
    trajectories: list[dict[str, Any] | None] | None = None
    objective_scores: list[dict[str, float]] | None = None


class SynthContainerLearningAdapter:
    """Evaluate candidates against a Synth-compatible Container rollout endpoint."""

    def __init__(
        self,
        *,
        container_url: str,
        request_builder: RolloutRequestBuilder,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        score_extractor: RolloutScoreExtractor = default_rollout_score_extractor,
    ) -> None:
        self._container_url = container_url.rstrip("/")
        self._request_builder = request_builder
        self._api_key = api_key
        self._headers = dict(headers or {})
        self._timeout_seconds = float(timeout_seconds)
        self._score_extractor = score_extractor

    def _post_rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"content-type": "application/json", **self._headers}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        http_request = request.Request(
            f"{self._container_url}/rollout",
            data=body,
            headers=headers,
            method="POST",
        )
        with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def evaluate(
        self,
        batch: list[Mapping[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> ContainerEvaluationBatch:
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any] | None] = []
        objective_scores: list[dict[str, float]] = []

        for index, example in enumerate(batch):
            correlation_id = f"prompt-opt-{index}-{uuid.uuid4().hex}"
            rollout_request = self._request_builder(example, candidate, correlation_id)
            rollout_response = self._post_rollout(rollout_request)
            score = self._score_extractor(rollout_response)
            reward_info = rollout_response.get("metrics") or rollout_response.get("reward_info") or {}
            outcome_objectives = reward_info.get("outcome_objectives")
            outputs.append(
                {
                    "request": rollout_request,
                    "response": rollout_response,
                }
            )
            scores.append(score)
            trajectories.append(rollout_response.get("trace") if capture_traces else None)
            objective_scores.append(
                dict(outcome_objectives) if isinstance(outcome_objectives, dict) else {"reward": score}
            )

        return ContainerEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: ContainerEvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        reflective_rows: list[dict[str, Any]] = []
        for idx, (output, score) in enumerate(zip(eval_batch.outputs, eval_batch.scores)):
            reflective_rows.append(
                {
                    "index": idx,
                    "candidate": dict(candidate),
                    "score": float(score),
                    "output": output,
                }
            )
        return {component: tuple(reflective_rows) for component in components_to_update}


__all__ = [
    "ContainerEvaluationBatch",
    "SynthContainerLearningAdapter",
    "default_rollout_request_builder",
    "default_rollout_score_extractor",
]
