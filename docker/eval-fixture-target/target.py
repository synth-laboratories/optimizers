"""Deterministic `eval.target.v1` fixture target.

It exists so the runner can be proved end to end — result parsing, gates, live
events, cancellation, resume, and semaphore behaviour — without depending on a
benchmark substrate. The task is a tiny deterministic corridor: the policy must
walk to a seed-determined target square before the step budget runs out.

The contract this file implements is the whole point: `/input/trial.json` in,
`/output/result.json` out, live lines on `/output/events.jsonl`, and rig health
(`status`) kept strictly separate from the policy's outcome
(`benchmark_status`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

INPUT = Path("/input")
OUTPUT = Path("/output")
CORRIDOR = 16
STEP_BUDGET = 40


def emit(event: str, **fields: Any) -> None:
    with (OUTPUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "at": time.time(), **fields}) + "\n")
        handle.flush()


def load_policy(entrypoint: str) -> Any:
    """Import the mounted candidate. Never imported by the runner itself."""

    module_name, _, attribute = entrypoint.partition(":")
    path = INPUT / "policy" / f"{module_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"policy module {module_name}.py is not in the mounted candidate")
    spec = importlib.util.spec_from_file_location(f"candidate_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load policy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, attribute or "Policy")
    return factory()


def trace(record: dict[str, Any]) -> None:
    """Append one step to the durable rollout trace.

    Every step is written, not sampled: the trace is what makes a score
    re-checkable after the run, so it is evidence rather than a debug aid.
    """

    with (OUTPUT / "trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def rollout(policy: Any, seed: int, trial_id: str) -> tuple[float, int, bool]:
    """A deterministic corridor walk. Same seed, same target, every time."""

    target = seed % CORRIDOR
    position = (seed * 7) % CORRIDOR
    for step in range(STEP_BUDGET):
        if position == target:
            trace({"trial_id": trial_id, "step": step, "position": position, "terminal": True})
            return 1.0 - step / (2 * STEP_BUDGET), step, True
        observation = {
            "position": position,
            "corridor": CORRIDOR,
            "step": step,
            "target": target,
        }
        action = policy.act(observation)
        if action == "left":
            position = (position - 1) % CORRIDOR
        elif action == "right":
            position = (position + 1) % CORRIDOR
        elif action != "stay":
            raise ValueError(f"policy returned an unknown action: {action!r}")
        trace(
            {
                "trial_id": trial_id,
                "step": step,
                "observation": observation,
                "action": action,
                "next_position": position,
                "terminal": False,
            }
        )
        if step % 8 == 0:
            emit("progress", step=step, position=position, target=target)
    return 0.0, STEP_BUDGET, False


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trial = json.loads((INPUT / "trial.json").read_text(encoding="utf-8"))
    trial_id = trial["trial_id"]
    started = time.time()
    gates = []
    emit("trial.started", trial_id=trial_id, seed=trial["seed"], scenario=trial["scenario"])

    try:
        policy = load_policy(trial["candidate"]["entrypoint"])
        gates.append({"id": "policy_loaded", "passed": True})
    except Exception as error:  # noqa: BLE001 - a bad candidate is not a rig failure
        gates.append({"id": "policy_loaded", "passed": False})
        write_result(
            trial_id,
            status="evaluated",
            benchmark_status="invalid",
            metrics={},
            gates=gates,
            started=started,
            error=f"{type(error).__name__}: {error}",
        )
        return 0

    try:
        reward, steps, reached = rollout(policy, int(trial["seed"]), trial_id)
    except Exception as error:  # noqa: BLE001 - the policy misbehaved, the rig did not
        gates.append({"id": "verifier_completed", "passed": False})
        write_result(
            trial_id,
            status="evaluated",
            benchmark_status="failed",
            metrics={},
            gates=gates,
            started=started,
            error=f"{type(error).__name__}: {error}",
        )
        return 0

    gates.append({"id": "verifier_completed", "passed": True})
    emit("trial.finished", trial_id=trial_id, reward=reward, steps=steps, reached=reached)
    write_result(
        trial_id,
        status="evaluated",
        benchmark_status="passed" if reached else "failed",
        metrics={"reward": reward, "steps": float(steps)},
        gates=gates,
        started=started,
        error=None,
    )
    return 0


def write_result(
    trial_id: str,
    *,
    status: str,
    benchmark_status: str,
    metrics: dict[str, float],
    gates: list[dict[str, Any]],
    started: float,
    error: str | None,
) -> None:
    payload = {
        "schema_version": "eval.container-result.v1",
        "trial_id": trial_id,
        "status": status,
        "benchmark_status": benchmark_status,
        "metrics": metrics,
        "gates": gates,
        "usage": {
            "cost_usd": None,
            "rollouts": 1,
            "wall_time_ms": int((time.time() - started) * 1000),
        },
        "artifacts": [
            {"role": "trace", "path": "trace.jsonl"},
            {"role": "events", "path": "events.jsonl"},
        ],
    }
    if error:
        payload["error"] = error
    (OUTPUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
