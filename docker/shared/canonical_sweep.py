"""Trusted frontend agent driving the pinned native engine, never candidate code.

Code-policy trials retain GameBench's scrubbed subprocess sandbox. This lane
imports only the image-owned canonical agent; credentials never enter arbitrary
candidate code. The agent owns planning/history and the engine owns all facts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/gamebench/tasks/craftax-singleplayer")
from canonical_policy import choose_actions  # noqa: E402
from containers.codepolicy.rust_repl_session import RustReplSession  # noqa: E402


def run(suite: dict, replay_dir: Path) -> dict:
    if len(suite["seeds"]) != 1:
        raise ValueError("one canonical agent process per trial required")
    seed = suite["seeds"][0]
    task = json.loads(Path(suite["task_template"]).read_text())
    events = []
    repl = RustReplSession()
    try:
        latest = repl.reset(
            task=task,
            seed=seed,
            readout_mode="full",
            replay={"enabled": True, "stride": 10, "max_frames": 160, "delay_cs": 8},
        )
        initial = latest["readout"]
        readout = initial
        for ply in range(suite["max_steps"]):
            if latest.get("terminated") or latest.get("truncated"):
                break
            decision = choose_actions(
                observation_text=readout["observation_text"],
                session={},
                valid_actions=readout.get("valid_actions", []),
                readout=readout,
                seed=seed,
                ply=ply,
            )
            if decision.get("stop_episode"):
                break  # partial evidence fails completion/coverage gates; never fill with noops
            actions = decision.get("actions")
            if not isinstance(actions, list) or len(actions) != 1:
                raise ValueError("canonical harness must choose exactly one native action")
            latest = repl.step(actions[0], readout_mode="full")
            readout = latest["readout"]
            private = readout["private"]
            row = {
                "seed": seed,
                "ply": ply,
                "steps_executed": 1,
                "at": time.time(),
                "action": actions[0],
                "actions": actions,
                "reward_total": private["total_reward"],
                "resources": readout["public"].get("inventory", {}),
                "achievements": sorted(private.get("achievements", [])),
                "terminated": bool(latest.get("terminated")),
                "truncated": bool(latest.get("truncated")),
                "grid_hash": readout.get("grid_hash"),
                "readout": readout,
            }
            events.append(row)
            with Path(os.environ["GAMEBENCH_ROLLOUT_EVENT_PATH"]).open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay = repl.save_replay(replay_dir / f"{seed}.gif")
    finally:
        repl.close()
    private = readout["private"]
    achievements = sorted(private.get("achievements", []))
    reward = float(private["total_reward"])
    episode = {
        "rollout_id": f"rust-repl-{seed}",
        "success_status": "success" if achievements else "failure",
        "status_detail": "terminated"
        if latest.get("terminated")
        else "truncated"
        if latest.get("truncated")
        else "partial",
        "reward_info": {
            "outcome_reward": reward,
            "details": {
                "seed": seed,
                "steps": private["step_index"],
                "achievements": achievements,
                "achievement_count": len(achievements),
                "lane": "rust",
                "engine_mode": "rust_repl",
                "engine_clone": False,
            },
        },
        "initial_state": initial,
        "state": {"public": readout["public"], "private": private},
        "events": events,
        "replay": replay,
    }
    return {
        "schema": "gamebench.craftax.policy_sweep_summary.v1",
        "max_steps": suite["max_steps"],
        "mean_reward": reward,
        "unique_achievement_count": len(achievements),
        "unique_achievements": achievements,
        "achievement_frequency": {a: 1 for a in achievements},
        "episodes": [episode],
        "episode_summaries": [
            {
                "seed": seed,
                "rollout_id": episode["rollout_id"],
                "reward": reward,
                "achievement_count": len(achievements),
                "achievements": achievements,
                "replay": replay,
                "supervision": "trusted_frontend_agent",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(json.loads(args.suite.read_text()), args.replay_dir)
    args.output.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
