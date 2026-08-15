"""Harbor verifier for one candidate on one seed.

The shipped Harbor bundle (`code_policy_deo_hillclimb`) scores a *leaderboard*:
an agent writes many candidates and the verifier ranks them. That is not one
policy on one seed, so `eval` cannot use it directly and stay inside
`trial_mode = one-policy-one-seed`.

This verifier keeps Harbor's contract and drops its multi-candidate framing:
same workspace layout (`/workspace/gamebench`, `/workspace/candidates/<task>`),
same output surface (`/logs/verifier/result.json` and `/logs/verifier/reward.txt`),
same `harbor_reward` field the Harbor scorer emits — for exactly one candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BENCHMARK_FAMILY = "gamebench.code_policy_single_candidate"


def write_result(result_path: Path, reward_path: Path, payload: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reward_path.write_text(f"{payload.get('harbor_reward', 0.0)}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--task", default="craftax-singleplayer")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--work", default="/tmp/work")
    parser.add_argument("--result", default="/logs/verifier/result.json")
    parser.add_argument("--reward", default="/logs/verifier/reward.txt")
    args = parser.parse_args()

    task_dir = Path(args.workspace) / "gamebench" / "tasks" / args.task
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "report.json"
    result_path = Path(args.result)
    reward_path = Path(args.reward)

    completed = subprocess.run(  # noqa: S603 - fixed interpreter, image-owned script
        [
            sys.executable,
            str(task_dir / "scripts" / "run_policy_sweep.py"),
            "--policy",
            args.candidate,
            "--suite",
            args.suite,
            "--output",
            str(report_path),
            "--include-trace",
            "--lane",
            "python",
        ],
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    (work / "sweep.stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    if completed.returncode != 0 or not report_path.is_file():
        write_result(
            result_path,
            reward_path,
            {
                "benchmark_family": BENCHMARK_FAMILY,
                "harbor_reward": 0.0,
                "passed": False,
                "candidate_attributable": completed.returncode in (40, 41),
                "error": f"policy sweep exited {completed.returncode}",
                "stderr_tail": (completed.stderr or "")[-2000:],
            },
        )
        return completed.returncode or 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    reward = float(report.get("mean_reward", 0.0))
    write_result(
        result_path,
        reward_path,
        {
            "benchmark_family": BENCHMARK_FAMILY,
            "score_metric": "achievement_success_score",
            "harbor_reward": reward,
            "passed": reward > 0.0,
            "achievement_count": report.get("unique_achievement_count"),
            "achievements": report.get("unique_achievements"),
            "seeds": report.get("seeds"),
            "suite_id": report.get("suite_id"),
            "policy_sha256": report.get("policy_sha256"),
            "report_path": str(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
