"""GameBench/Harbor `eval.target.v1` target.

Internally this is Harbor: the Harbor workspace layout, the Harbor candidate
location, and the Harbor verifier surface (`/logs/verifier/result.json` and
`/logs/verifier/reward.txt`). From `eval`'s side none of that is visible — it
is a container that reads `/input/trial.json` and writes the standard result,
trace, and evidence tree.

Adopting Harbor's output is the point, not a detail. Harbor's receipts land in
`/logs/verifier`, which is invisible to the matrix; this target copies them
into `/output` with their digests so a Harbor trial is queryable through the
same evidence layout as every other target, and a trial is not terminal until
that adoption has happened.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/opt/eval")
from policy_setup import CandidateError, resolve_policy, summarize_usage  # noqa: E402

INPUT = Path("/input")
OUTPUT = Path("/output")
WORKSPACE = Path("/workspace")
LOGS = Path("/logs/verifier")
TASK = "craftax-singleplayer"
CANDIDATE_SUBDIR = "craftax"
TASK_DIR = WORKSPACE / "gamebench" / "tasks" / TASK
VERIFIER = Path("/opt/harbor/verify_single_candidate.py")

SCENARIOS: dict[str, dict[str, Any]] = {
    "craftax-default-48x48": {
        "task_template": "tasks/craftax_default_template.json",
        "max_steps": 500,
        "world": "craftax_default (48x48, 9 levels)",
    },
}


def emit(event: str, **fields: Any) -> None:
    with (OUTPUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "at": time.time(), **fields}) + "\n")
        handle.flush()


def workspace_fingerprint() -> str:
    """Digest the GameBench tree so a candidate cannot quietly edit the task."""

    hasher = hashlib.sha256()
    root = WORKSPACE / "gamebench"
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return hasher.hexdigest()


def write_result(
    trial_id: str,
    *,
    status: str,
    benchmark_status: str | None,
    metrics: dict[str, float],
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    started: float,
    error: str | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "eval.container-result.v1",
        "trial_id": trial_id,
        "status": status,
        "benchmark_status": benchmark_status,
        "metrics": metrics,
        "gates": gates,
        "usage": {
            "rollouts": 1,
            "wall_time_ms": int((time.time() - started) * 1000),
            **(usage or {"cost_usd": None}),
        },
        "artifacts": artifacts,
    }
    if error:
        payload["error"] = error
    (OUTPUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def adopt_harbor_evidence() -> list[dict[str, Any]]:
    """Copy Harbor's receipts into the normalized evidence tree."""

    adopted: list[dict[str, Any]] = []
    destination = OUTPUT / "verifier"
    destination.mkdir(parents=True, exist_ok=True)
    for name, role in (("result.json", "harbor_result"), ("reward.txt", "harbor_reward")):
        source = LOGS / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            adopted.append({"role": role, "path": f"verifier/{name}"})
    return adopted


def write_trace(report_path: Path, trial_id: str, seed: int) -> bool:
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    shutil.copy2(report_path, OUTPUT / "verifier" / "report.json")
    with (OUTPUT / "trace.jsonl").open("w", encoding="utf-8") as handle:
        summaries = report.get("episode_summaries") or []
        for index, episode in enumerate(report.get("episodes") or []):
            handle.write(
                json.dumps(
                    {
                        "kind": "episode",
                        "trial_id": trial_id,
                        "seed": seed,
                        "rollout_id": episode.get("rollout_id"),
                        "summary": summaries[index] if index < len(summaries) else {},
                        "reward_info": episode.get("reward_info"),
                        "success_status": episode.get("success_status"),
                    }
                )
                + "\n"
            )
            for step, event in enumerate(episode.get("events") or []):
                handle.write(
                    json.dumps(
                        {
                            "kind": "event",
                            "trial_id": trial_id,
                            "seed": seed,
                            "rollout_id": episode.get("rollout_id"),
                            "index": step,
                            "event": event,
                        }
                    )
                    + "\n"
                )
            if episode.get("state") is not None:
                handle.write(
                    json.dumps(
                        {
                            "kind": "final_state",
                            "trial_id": trial_id,
                            "seed": seed,
                            "state": episode["state"],
                        }
                    )
                    + "\n"
                )
    return True


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "verifier").mkdir(parents=True, exist_ok=True)
    trial = json.loads((INPUT / "trial.json").read_text(encoding="utf-8"))
    trial_id = trial["trial_id"]
    seed = int(trial["seed"])
    started = time.time()
    gates: list[dict[str, Any]] = []
    emit("trial.started", trial_id=trial_id, seed=seed, scenario=trial["scenario"])

    scenario = SCENARIOS.get(trial["scenario"])
    if scenario is None:
        write_result(
            trial_id,
            status="failed",
            benchmark_status=None,
            metrics={},
            gates=gates,
            artifacts=[],
            started=started,
            error=f"unknown scenario {trial['scenario']!r} for the Harbor target",
        )
        return 0

    before = workspace_fingerprint()

    # Stage the candidate where Harbor expects it.
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    candidate_dir = WORKSPACE / "candidates" / CANDIDATE_SUBDIR
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate = candidate_dir / "heuristic_policy.py"
    try:
        source, policy_env = resolve_policy(trial, INPUT, work)
    except (CandidateError, Exception) as error:  # noqa: BLE001 - bad candidate, healthy rig
        gates.append({"id": "policy_loaded", "passed": False})
        write_result(
            trial_id,
            status="evaluated",
            benchmark_status="invalid",
            metrics={},
            gates=gates,
            artifacts=[],
            started=started,
            error=f"{type(error).__name__}: {error}",
        )
        return 0
    shutil.copy2(source, candidate)
    gates.append({"id": "policy_loaded", "passed": True})

    suite_path = work / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema": "gamebench.craftax.policy_sweep.v1",
                "suite_id": f"eval_harbor_{trial['scenario']}_{seed}",
                "task_template": scenario["task_template"],
                "seeds": [seed],
                "max_steps": scenario["max_steps"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    emit("verifier.started", seed=seed, world=scenario["world"])
    completed = subprocess.run(  # noqa: S603 - fixed interpreter, image-owned verifier
        [
            sys.executable,
            str(VERIFIER),
            "--workspace",
            str(WORKSPACE),
            "--task",
            TASK,
            "--candidate",
            str(candidate),
            "--suite",
            str(suite_path),
            "--work",
            str(work),
            "--result",
            str(LOGS / "result.json"),
            "--reward",
            str(LOGS / "reward.txt"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **policy_env},
    )
    (OUTPUT / "verifier" / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    usage = summarize_usage(work)
    if usage["calls"]:
        (OUTPUT / "usage.jsonl").write_bytes((work / "usage.jsonl").read_bytes())

    artifacts = adopt_harbor_evidence()
    harbor_result_path = OUTPUT / "verifier" / "result.json"
    harbor_reward_path = OUTPUT / "verifier" / "reward.txt"
    verifier_completed = harbor_result_path.is_file()
    gates.append({"id": "verifier_completed", "passed": verifier_completed})

    reward: float | None = None
    if harbor_reward_path.is_file():
        try:
            reward = float(harbor_reward_path.read_text(encoding="utf-8").strip())
        except ValueError:
            reward = None
    gates.append({"id": "reward_emitted", "passed": reward is not None})

    after = workspace_fingerprint()
    gates.append({"id": "workspace_intact", "passed": after == before})

    if write_trace(work / "report.json", trial_id, seed):
        artifacts.append({"role": "trace", "path": "trace.jsonl"})
        artifacts.append({"role": "verifier", "path": "verifier/report.json"})
    artifacts.append({"role": "events", "path": "events.jsonl"})
    if (OUTPUT / "usage.jsonl").is_file():
        artifacts.append({"role": "usage", "path": "usage.jsonl"})

    if not verifier_completed:
        write_result(
            trial_id,
            status="failed",
            benchmark_status=None,
            metrics={},
            gates=gates,
            artifacts=artifacts,
            started=started,
            usage=usage,
            error=(
                "Harbor verifier produced no result.json; "
                f"exit {completed.returncode}: {(completed.stderr or '')[-1500:]}"
            ),
        )
        return 0

    harbor = json.loads(harbor_result_path.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    if reward is not None:
        metrics["verifier_reward"] = reward
    if isinstance(harbor.get("achievement_count"), (int, float)):
        metrics["achievements"] = float(harbor["achievement_count"])
    emit(
        "verifier.finished",
        reward=reward,
        passed=harbor.get("passed"),
        cost_usd=usage["cost_usd"],
    )
    write_result(
        trial_id,
        status="evaluated",
        benchmark_status="passed" if harbor.get("passed") else "failed",
        metrics=metrics,
        gates=gates,
        artifacts=artifacts,
        started=started,
        usage=usage,
        error=harbor.get("error"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
