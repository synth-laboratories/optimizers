"""Local MIPRO entrypoint backed by the mirrored offline SDK runtime."""

from __future__ import annotations

from typing import Any, Callable

from prompt_opt.sdk.optimization.internal.prompt_learning import PromptLearningJob


_PROPOSER_BACKENDS = ["single_prompt", "rlm"]


def proposer_backends() -> list[str]:
    """Return the supported public proposer backend tokens."""
    return list(_PROPOSER_BACKENDS)


def _policy_to_initial_candidate(initial_policy: dict[str, Any]) -> dict[str, Any]:
    template = str(initial_policy.get("template", "")).strip() or "You are a helpful assistant."
    return {
        "stages": [
            {
                "id": "default",
                "name": "Default",
                "messages": [
                    {"role": "system", "pattern": template, "order": 0},
                    {"role": "user", "pattern": "{input}", "order": 1},
                ],
                "wildcards": {},
            }
        ]
    }


def _extract_best_policy(result_payload: dict[str, Any]) -> dict[str, Any]:
    best_candidate = result_payload.get("best_candidate")
    if not isinstance(best_candidate, dict):
        return {"template": ""}
    stages = best_candidate.get("stages") or best_candidate.get("candidate", {}).get("stages") or []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for message in stage.get("messages", []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                text = message.get("pattern") or message.get("content")
                if isinstance(text, str):
                    return {"template": text}
    content = best_candidate.get("candidate_content")
    if isinstance(content, str):
        return {"template": content}
    return {"template": ""}


def run_mipro(
    *,
    config: dict[str, Any],
    initial_policy: dict[str, Any],
    dataset: dict[str, Any],
    task_llm: Callable[[str], str],
) -> dict[str, Any]:
    """Run local offline MIPRO and return a JSON-compatible result payload."""
    examples = []
    for item in dataset.get("examples", []):
        if isinstance(item, dict):
            examples.append(
                {
                    "input": item.get("input"),
                    "answer": item.get("expected"),
                    "expected": item.get("expected"),
                    "metadata": dict(item.get("metadata") or {}),
                }
            )
    prompt_learning_config = {
        "prompt_learning": {
            "algorithm": "mipro",
            "execution_mode": "retrieved",
            "task_data": {
                "examples": examples,
            },
            "mipro": {
                "initial_candidate": _policy_to_initial_candidate(initial_policy),
                "num_candidates": int(config.get("num_candidates", 8)),
                "max_iterations": int(config.get("max_iterations", 8)),
                "early_stop_rounds": int(config.get("early_stop_rounds", 3)),
                "min_improvement": float(config.get("min_improvement", 1e-6)),
                "seed": int(config.get("seed", 0)),
                "proposer_backend": str(config.get("proposer_backend", "single_prompt")),
                "termination_conditions": {
                    "total_rollouts": int(config.get("max_iterations", 8)) * max(1, len(examples)),
                },
            },
            "local_runtime": {
                "task_model": task_llm,
            },
        }
    }

    job = PromptLearningJob.from_dict(
        prompt_learning_config,
        backend_url="local://prompt-opt",
        api_key="local",
    )
    job_id = job.submit()
    result = job.stream_until_complete(timeout=300.0, interval=0.05).to_dict()
    return {
        "run_id": job_id,
        "initial_policy": dict(initial_policy),
        "best_policy": _extract_best_policy(result),
        "best_score": result.get("best_reward"),
        "history": job.get_results().get("result", {}).get("history", []),
        "job_result": result,
    }
