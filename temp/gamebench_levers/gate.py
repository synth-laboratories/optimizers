#!/usr/bin/env python3
"""Headroom gate: prove a (game, mode) target is searchable before spending budget.

For each game it boots the real code stack and scores two policies over the whole
train split: the seed GEPA starts from, and a hand-written reference. It fails the
target when the seed is already at the reference (nothing to learn) or when the
reference cannot beat the seed (the reward does not pay for better code).

    uv run python gate.py                # all games
    uv run python gate.py sokoban rogue  # a subset
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Any

import httpx

from gamebench_levers import GAMES
from gamebench_levers.references import reference_policy
from gamebench_levers.stack import start_stack


def _score(orch: str, candidate_id: str | None, split: str, count: int) -> list[float]:
    rewards: list[float] = []
    for index in range(count):
        body: dict[str, Any] = {"task_id": f"{split}:{index}"}
        if candidate_id:
            body["candidate_id"] = candidate_id
        record = httpx.post(f"{orch}/rollout", json=body, timeout=900).json()
        rewards.append(float(record.get("reward") or 0.0))
    return rewards


def gate_game(game: str) -> dict[str, Any]:
    started = time.perf_counter()
    stack = start_stack(game, "code")
    try:
        taskset = httpx.get(f"{stack.orch_url}/taskset", timeout=30).json()
        train = int(taskset["splits"]["train"])
        seed_rewards = _score(stack.orch_url, None, "train", train)
        registered = httpx.post(
            f"{stack.orch_url}/candidates",
            json={"candidate_id": "reference", "lever_bundle": {"values": {"policy_script": reference_policy(game)}}},
            timeout=180,
        ).json()
        if registered.get("status") != "registered":
            return {"game": game, "ok": False, "why": "reference failed to apply", "apply_report": registered}
        ref_rewards = _score(stack.orch_url, "reference", "train", train)
    finally:
        stack.stop()

    seed_mean = statistics.mean(seed_rewards)
    ref_mean = statistics.mean(ref_rewards)
    return {
        "game": game,
        "train_tasks": len(seed_rewards),
        "seed_mean": round(seed_mean, 4),
        "reference_mean": round(ref_mean, 4),
        "seed_rewards": [round(value, 3) for value in seed_rewards],
        "reference_rewards": [round(value, 3) for value in ref_rewards],
        "headroom": round(ref_mean - seed_mean, 4),
        "ok": ref_mean > seed_mean,
        "why": "" if ref_mean > seed_mean else "reference does not beat seed: reward does not pay for better code",
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    games = [arg for arg in sys.argv[1:] if arg in GAMES] or list(GAMES)
    results = []
    for game in games:
        try:
            result = gate_game(game)
        except Exception as exc:  # noqa: BLE001
            result = {"game": game, "ok": False, "why": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    print("\n=== headroom gate ===")
    for result in results:
        mark = "PASS" if result.get("ok") else "FAIL"
        print(
            f"{mark} {result['game']:12s} seed={result.get('seed_mean')} "
            f"reference={result.get('reference_mean')} headroom={result.get('headroom')} {result.get('why','')}"
        )
    return 0 if all(result.get("ok") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
