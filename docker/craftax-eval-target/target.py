"""Craftax `eval.target.v1` target.

One trial is one candidate code policy on one seed of one pinned Craftax world,
evaluated in-container against the GameBench native Rust engine (`gold_rust`).
No host substrate and no network: the engine ships inside the image, so
`network = none` holds and two trials of the same run cannot drift apart
because something outside the container changed.

The scenario names the world it actually uses. A Craftax number that does not
say which board produced it is not a result, and the 48x48 default board and a
9x9 fixture room are not the same benchmark.

Everything the rollout did is written to `/output/trace.jsonl` — every engine
event of every episode — and the sweep's own report is kept verbatim as the
verifier artifact, so a score can be re-derived later rather than trusted.

Isolation boundary: the eval container *is* the sandbox. GameBench's own
per-episode sandbox needs unprivileged user namespaces (bubblewrap), which are
not available inside a container on every host, so the sweep runs unsupervised
in a private working directory and this wrapper — not the sweep, and not the
candidate — writes `/output` after the sweep has exited. That is why the
Craftax smoke recipe is report-only: it measures candidates honestly under a
container boundary, but it is not a promotion-grade adversarial rig.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/opt/eval")
from policy_setup import CandidateError, resolve_policy, summarize_usage  # noqa: E402

INPUT = Path("/input")
OUTPUT = Path("/output")
TASK_DIR = Path("/opt/gamebench/tasks/craftax-singleplayer")
SWEEP = TASK_DIR / "scripts" / "run_policy_sweep.py"

# Scenario id -> the exact board and step budget it means. The id is part of
# the evidence: reading it tells you what was measured.
SCENARIOS: dict[str, dict[str, Any]] = {
    "craftax-default-48x48": {
        "task_template": "tasks/craftax_default_template.json",
        "max_steps": 500,
        "world": "craftax_default (48x48, 9 levels)",
    },
    "craftax-policy-dev-9x9": {
        "task_template": "tasks/policy_dev_template.json",
        "max_steps": 80,
        "world": "fixture_room (9x9) — a smoke board, not a benchmark",
    },
}

# The sweep's own exit codes, so a bad candidate is never reported as a rig
# failure and a rig failure is never blamed on the candidate.
EXIT_CANDIDATE_POLICY_FAILURE = 40
EXIT_CANDIDATE_EPISODE_TIMEOUT = 41


def emit(event: str, **fields: Any) -> None:
    with (OUTPUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "at": time.time(), **fields}) + "\n")
        handle.flush()


def write_result(
    trial_id: str,
    *,
    status: str,
    benchmark_status: str,
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


def episode_steps(report: dict[str, Any]) -> int | None:
    """Steps the episode actually ran, from the engine's own final state."""

    steps: list[int] = []
    for episode in report.get("episodes") or []:
        state = (episode.get("state") or {}).get("private") or {}
        index = state.get("step_index")
        if isinstance(index, int):
            steps.append(index)
    return max(steps) if steps else None


def with_policy_coverage(usage: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Say how much of the episode the candidate's policy actually chose.

    An LLM policy that spends its budget does not end the episode — it returns
    `noop` for every remaining step. The reward that follows is the fallback's,
    not the model's, and a mean that silently mixes the two is not evidence
    about the candidate. A policy that never exhausted its budget covers the
    whole episode, which is the ordinary case and reads as such.
    """

    steps = episode_steps(report)
    exhausted_at = usage.get("exhausted_at_ply")
    covered = steps if exhausted_at is None else min(exhausted_at, steps or exhausted_at)
    return {
        **usage,
        "episode_steps": steps,
        "policy_steps": covered,
        "filler_steps": None if steps is None or covered is None else max(0, steps - covered),
        "policy_step_fraction": (
            None if not steps or covered is None else round(covered / steps, 4)
        ),
    }


def write_trace(report: dict[str, Any], trial_id: str, seed: int) -> None:
    """Every engine event of every episode, one JSON object per line."""

    with (OUTPUT / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for index, episode in enumerate(report.get("episodes") or []):
            summary = (
                (report.get("episode_summaries") or [{}])[index]
                if index < len(report.get("episode_summaries") or [])
                else {}
            )
            handle.write(
                json.dumps(
                    {
                        "kind": "episode",
                        "trial_id": trial_id,
                        "seed": seed,
                        "rollout_id": episode.get("rollout_id"),
                        "summary": summary,
                        "reward_info": episode.get("reward_info"),
                        "success_status": episode.get("success_status"),
                        "status_detail": episode.get("status_detail"),
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
                            "rollout_id": episode.get("rollout_id"),
                            "state": episode["state"],
                        }
                    )
                    + "\n"
                )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
            benchmark_status="invalid",
            metrics={},
            gates=gates,
            artifacts=[],
            started=started,
            error=f"unknown scenario {trial['scenario']!r} for the Craftax target",
        )
        return 0

    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    try:
        policy_path, policy_env = resolve_policy(trial, INPUT, work)
        gates.append({"id": "policy_loaded", "passed": True})
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

    suite = {
        "schema": "gamebench.craftax.policy_sweep.v1",
        "suite_id": f"eval_{trial['scenario']}_{seed}",
        "task_template": scenario["task_template"],
        "seeds": [seed],
        "max_steps": scenario["max_steps"],
    }
    # Private working directory: the candidate never runs with /output in view,
    # and this wrapper publishes evidence only once the sweep has exited.
    suite_path = work / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    work_report = work / "report.json"
    replay_dir = work / "replays"
    report_path = OUTPUT / "verifier" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    emit("rollout.started", seed=seed, world=scenario["world"], max_steps=scenario["max_steps"])
    completed = subprocess.run(  # noqa: S603 - fixed interpreter, image-owned script
        [
            sys.executable,
            str(SWEEP),
            "--policy",
            str(policy_path),
            "--suite",
            str(suite_path),
            "--output",
            str(work_report),
            "--include-trace",
            "--replay-dir",
            str(replay_dir),
            "--lane",
            "rust",
        ],
        cwd=str(TASK_DIR),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **policy_env},
    )
    if work_report.is_file():
        report_path.write_bytes(work_report.read_bytes())
    (OUTPUT / "suite.json").write_bytes(suite_path.read_bytes())
    if replay_dir.is_dir():
        output_frames = OUTPUT / "frames"
        output_frames.mkdir(parents=True, exist_ok=True)
        for replay in replay_dir.glob("*.gif"):
            (output_frames / replay.name).write_bytes(replay.read_bytes())
    (OUTPUT / "verifier" / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    usage = summarize_usage(work)
    if usage["calls"]:
        (OUTPUT / "usage.jsonl").write_bytes((work / "usage.jsonl").read_bytes())
    if completed.returncode == EXIT_CANDIDATE_POLICY_FAILURE:
        gates.append({"id": "verifier_completed", "passed": False})
        write_result(
            trial_id,
            status="evaluated",
            benchmark_status="failed",
            metrics={},
            gates=gates,
            artifacts=_artifacts(report_path),
            started=started,
            error="candidate policy raised during rollout",
            usage=usage,
        )
        return 0
    if completed.returncode == EXIT_CANDIDATE_EPISODE_TIMEOUT:
        gates.append({"id": "verifier_completed", "passed": False})
        write_result(
            trial_id,
            status="evaluated",
            benchmark_status="failed",
            metrics={},
            gates=gates,
            artifacts=_artifacts(report_path),
            started=started,
            error="candidate policy exceeded the episode deadline",
            usage=usage,
        )
        return 0
    if completed.returncode != 0 or not report_path.is_file():
        # The rig, not the policy: report it as such so it is never scored.
        write_result(
            trial_id,
            status="failed",
            benchmark_status=None,
            metrics={},
            gates=gates,
            artifacts=_artifacts(report_path),
            started=started,
            error=f"policy sweep exited {completed.returncode}: {(completed.stderr or '')[-2000:]}",
            usage=usage,
        )
        return 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    write_trace(report, trial_id, seed)
    gates.append({"id": "verifier_completed", "passed": True})
    reward = float(report.get("mean_reward", 0.0))
    achievements = float(report.get("unique_achievement_count", 0))
    usage = with_policy_coverage(usage, report)
    if usage["budget_exhausted"]:
        emit(
            "policy.budget_exhausted",
            reason=usage["budget_exhausted"],
            at_ply=usage["exhausted_at_ply"],
            filler_steps=usage["filler_steps"],
        )
    emit(
        "rollout.finished",
        reward=reward,
        achievements=achievements,
        cost_usd=usage["cost_usd"],
        policy_step_fraction=usage["policy_step_fraction"],
    )
    write_result(
        trial_id,
        status="evaluated",
        benchmark_status="passed" if reward > 0 else "failed",
        metrics={"reward": reward, "achievements": achievements},
        gates=gates,
        artifacts=_artifacts(report_path),
        started=started,
        usage=usage,
    )
    return 0


def _artifacts(report_path: Path) -> list[dict[str, Any]]:
    artifacts = [
        {"role": "trace", "path": "trace.jsonl"},
        {"role": "events", "path": "events.jsonl"},
        {"role": "suite", "path": "suite.json"},
        {"role": "usage", "path": "usage.jsonl"},
    ]
    if report_path.is_file():
        artifacts.append({"role": "verifier", "path": "verifier/report.json"})
    for replay in sorted((OUTPUT / "frames").glob("*.gif")):
        artifacts.append(
            {
                "role": "replay",
                "path": f"frames/{replay.name}",
                "media_type": "image/gif",
            }
        )
    return [entry for entry in artifacts if (OUTPUT / entry["path"]).is_file()]


if __name__ == "__main__":
    sys.exit(main())
