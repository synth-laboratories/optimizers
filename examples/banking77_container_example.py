"""Run local offline GEPA and MIPRO on Banking77 via a Synth in-process container."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from datasets import load_dataset
from prompt_opt.sdk.optimization.policy.v1 import PolicyOptimizationOfflineJob
from synth_ai.container import InProcessContainer
from synth_ai.sdk.container import ContainerConfig, RolloutResponseBuilder, create_container
from synth_ai.sdk.container.contracts import RolloutRequest

DEFAULT_LABELS = [
    "card_arrival",
    "cash_withdrawal_charge",
    "beneficiary_not_allowed",
    "pending_transfer",
]
DEFAULT_ALGORITHMS = ["gepa", "mipro"]


def _load_banking77_examples(
    *,
    selected_labels: list[str],
    train_per_label: int,
    held_out_per_label: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = load_dataset("EXRdurable/banking77", split="test")
    label_names = dataset.features["label"].names
    selected_ids = {label_names.index(name) for name in selected_labels}

    train_examples: list[dict[str, Any]] = []
    held_out_examples: list[dict[str, Any]] = []
    per_label_counts = {label_id: 0 for label_id in selected_ids}
    max_per_label = max(0, train_per_label) + max(0, held_out_per_label)

    for row in dataset:
        label_id = int(row["label"])
        if label_id not in selected_ids:
            continue
        if per_label_counts[label_id] >= max_per_label:
            continue
        per_label_counts[label_id] += 1
        example = {
            "seed": sum(per_label_counts.values()) - 1,
            "input": row["text"],
            "answer": label_names[label_id],
            "labels": selected_labels,
        }
        if per_label_counts[label_id] <= train_per_label:
            train_examples.append(example)
        else:
            held_out_examples.append(example)
        if all(count >= max_per_label for count in per_label_counts.values()):
            break

    return train_examples, held_out_examples


def _classify(prompt: str) -> str:
    prompt_lower = prompt.lower()
    text_region = prompt_lower.split("available labels:", 1)[0]

    if "beneficiar" in text_region or "recipient" in text_region and "not" in text_region:
        label = "beneficiary_not_allowed"
    elif "withdraw" in text_region and ("fee" in text_region or "charged" in text_region):
        label = "cash_withdrawal_charge"
    elif "pending transfer" in text_region or "transfer" in text_region and "pending" in text_region:
        label = "pending_transfer"
    elif "card" in text_region:
        label = "card_arrival"
    else:
        label = "beneficiary_not_allowed"

    if "return exactly one of" in prompt_lower or "output schema exactly" in prompt_lower:
        return label
    return f"The correct label is {label}."


def _extract_instruction(candidate: dict[str, Any]) -> str:
    stages = candidate.get("stages") or candidate.get("candidate", {}).get("stages") or []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for message in stage.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "system":
                text = message.get("pattern") or message.get("content")
                if isinstance(text, str):
                    return text
    return str(candidate.get("candidate_content", ""))


def _build_container_app(
    *,
    selected_labels: list[str],
    train_per_label: int,
    held_out_per_label: int,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    train_examples, held_out_examples = _load_banking77_examples(
        selected_labels=selected_labels,
        train_per_label=train_per_label,
        held_out_per_label=held_out_per_label,
    )
    all_examples = train_examples + held_out_examples

    def provide_taskset_description() -> dict[str, Any]:
        return {"id": "banking77-local", "splits": ["eval"], "sizes": {"eval": len(all_examples)}}

    def provide_task_instances(seeds: list[int]) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for seed in seeds:
            example = all_examples[seed % len(all_examples)]
            instances.append(
                {
                    "task": {"id": "banking77", "name": "Banking77"},
                    "dataset": {"id": "banking77", "split": "eval", "index": seed},
                    "task_metadata": dict(example),
                }
            )
        return instances

    async def rollout(request: RolloutRequest, _fastapi_request: Any):
        example_payload = dict(request.env.config.get("example", {}))
        candidate_payload = dict(request.policy.config.get("candidate", {}))
        labels = ", ".join(example_payload.get("labels", []))
        system_prompt = _extract_instruction(candidate_payload)
        user_input = str(example_payload.get("input", "")).strip()
        rendered_prompt = (
            f"{system_prompt}\n\nCustomer query:\n{user_input}\n\nAvailable labels: {labels}"
        )
        predicted = _classify(rendered_prompt)
        expected = str(example_payload.get("answer", "")).strip()
        reward = 1.0 if predicted.strip() == expected else 0.0
        return RolloutResponseBuilder.trace_only(
            trace_correlation_id=request.trace_correlation_id,
            reward=reward,
            trace={
                "metadata": {"trace_correlation_id": request.trace_correlation_id},
                "event_history": [
                    {
                        "type": "lm_call",
                        "llm_request": {"messages": [{"role": "user", "content": rendered_prompt}]},
                        "llm_response": {"message": {"role": "assistant", "content": predicted}},
                    }
                ],
            },
            details={"expected_answer": expected, "predicted_answer": predicted},
        )

    app = create_container(
        ContainerConfig(
            app_id="banking77-local",
            name="Banking77 Local",
            description="Banking77 local container for prompt-opt",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=rollout,
            cors_origins=["*"],
        )
    )
    return app, train_examples, held_out_examples


def _offline_config(
    *,
    algorithm: str,
    container_url: str,
    train_examples: list[dict[str, Any]],
    total_rollouts: int,
    num_generations: int,
    children_per_generation: int,
    num_candidates: int,
    max_iterations: int,
) -> dict[str, Any]:
    return {
        "prompt_learning": {
            "algorithm": algorithm,
            "execution_mode": "retrieved",
            "container_url": container_url,
            "task_data": {
                "train_examples": train_examples,
                "validation_examples": train_examples,
            },
            algorithm: {
                "initial_candidate": {
                    "stages": [
                        {
                            "id": "banking77_main",
                            "name": "Banking77 Classification",
                            "messages": [
                                {
                                    "role": "system",
                                    "order": 0,
                                    "pattern": "Classify the customer support query into exactly one Banking77 label from the provided set.",
                                }
                            ],
                            "wildcards": {},
                        }
                    ]
                },
                "termination_conditions": {"total_rollouts": total_rollouts},
                "population": {
                    "initial_size": 1,
                    "num_generations": num_generations,
                    "children_per_generation": children_per_generation,
                },
                "num_candidates": num_candidates,
                "max_iterations": max_iterations,
            },
        }
    }


def _score_instruction(instruction: str, examples: list[dict[str, Any]]) -> float:
    correct = 0
    for example in examples:
        labels = ", ".join(example.get("labels", []))
        rendered_prompt = (
            f"{instruction}\n\nCustomer query:\n{example['input']}\n\nAvailable labels: {labels}"
        )
        predicted = _classify(rendered_prompt)
        if predicted.strip() == str(example["answer"]).strip():
            correct += 1
    return correct / max(1, len(examples))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=DEFAULT_ALGORITHMS,
        default=list(DEFAULT_ALGORITHMS),
        help="Subset of local offline optimizers to run.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Banking77 labels to include in the local regression harness.",
    )
    parser.add_argument("--train-per-label", type=int, default=4)
    parser.add_argument("--held-out-per-label", type=int, default=2)
    parser.add_argument("--total-rollouts", type=int, default=48)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--children-per-generation", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--min-held-out-delta",
        type=float,
        default=0.0,
        help="Raise if an optimizer does not beat the held-out baseline by more than this value.",
    )
    return parser.parse_args()


async def _run_harness(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    app, train_examples, held_out_examples = _build_container_app(
        selected_labels=list(args.labels),
        train_per_label=int(args.train_per_label),
        held_out_per_label=int(args.held_out_per_label),
    )
    previous_auth_mode = os.environ.get("SYNTH_CONTAINER_AUTH_MODE")
    os.environ["SYNTH_CONTAINER_AUTH_MODE"] = "optional_local"
    try:
        async with InProcessContainer(app=app, tunnel_mode="local") as container:
            runs: dict[str, dict[str, Any]] = {}
            for algorithm in args.algorithms:
                kind = f"{algorithm}_offline"
                job = await PolicyOptimizationOfflineJob.create_async(
                    kind=kind,
                    system_name=f"banking77-{algorithm}",
                    config=_offline_config(
                        algorithm=algorithm,
                        container_url=container.url or "",
                        train_examples=train_examples,
                        total_rollouts=int(args.total_rollouts),
                        num_generations=int(args.num_generations),
                        children_per_generation=int(args.children_per_generation),
                        num_candidates=int(args.num_candidates),
                        max_iterations=int(args.max_iterations),
                    ),
                    backend_url="local://prompt-opt",
                    api_key="local",
                )
                result = await job.stream_until_complete_async(
                    timeout=float(args.timeout),
                    interval=0.05,
                )
                state_envelope = await job.get_state_envelope_async()
                candidates = state_envelope["state"]["candidates"]
                best_candidate = candidates[result["best_candidate_id"]]
                baseline_candidate = candidates["baseline"]
                baseline_instruction = _extract_instruction(baseline_candidate)
                best_instruction = _extract_instruction(best_candidate)
                train_baseline_score = float(baseline_candidate.get("avg_reward") or 0.0)
                train_best_score = float(result["best_reward"] or 0.0)
                held_out_baseline_score = _score_instruction(baseline_instruction, held_out_examples)
                held_out_best_score = _score_instruction(best_instruction, held_out_examples)
                held_out_delta = held_out_best_score - held_out_baseline_score
                train_delta = train_best_score - train_baseline_score
                if held_out_delta <= float(args.min_held_out_delta):
                    raise RuntimeError(
                        f"{algorithm} failed to improve Banking77 locally on held-out: "
                        f"baseline={held_out_baseline_score:.3f} best={held_out_best_score:.3f} "
                        f"required_delta>{float(args.min_held_out_delta):.3f}"
                    )
                runs[algorithm] = {
                    "baseline_candidate_id": "baseline",
                    "train_baseline_score": train_baseline_score,
                    "best_candidate_id": result["best_candidate_id"],
                    "train_best_score": train_best_score,
                    "train_delta": train_delta,
                    "held_out_baseline_score": held_out_baseline_score,
                    "held_out_best_score": held_out_best_score,
                    "held_out_delta": held_out_delta,
                    "best_instruction": best_instruction,
                }

            return runs
    finally:
        if previous_auth_mode is None:
            os.environ.pop("SYNTH_CONTAINER_AUTH_MODE", None)
        else:
            os.environ["SYNTH_CONTAINER_AUTH_MODE"] = previous_auth_mode


def main() -> None:
    args = _parse_args()
    results = asyncio.run(_run_harness(args))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
