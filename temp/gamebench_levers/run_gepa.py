#!/usr/bin/env python3
"""GEPA search over GameBench lever containers.

    uv run python run_gepa.py --game sokoban --mode code
    uv run python run_gepa.py --game craftax --mode code --generations 3
    uv run python run_gepa.py --all-code

Modes:
  code    searches `policy_script` (whole_file.v1 / unified_diff.v1)
  harness searches `harness_module` (harness_restart.v1) + `system_prompt`

The proposer is GEPA's default Codex app-server. Do not swap in a
chat-completions "real proposer" -- that is a different experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KEY_FILES = (
    Path.home() / "Documents/GitHub/backend/.env.local",
    Path.home() / "Documents/GitHub/backend/.env",
    ROOT / ".env.local",
)
# Harness mode targets the harness only. With two target modules the Codex proposer
# can return null for one of them, and the engine rejects the whole proposal with
# `proposal index=0 is not a valid proposal object: invalid type: null, expected a
# string`, killing the run. `system_prompt` stays advertised on /program either way.
TARGETS = {"code": ["policy_script"], "harness": ["harness_module"]}


def _load_dotenv(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        if key.strip():
            loaded[key.strip()] = value.strip().strip("'").strip('"')
    return loaded


def ensure_llm_keys() -> str:
    """Harness episodes call a model every turn; never silently skip them."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        return "environment"
    for path in KEY_FILES:
        if not path.is_file():
            continue
        parsed = _load_dotenv(path)
        for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "GAMEBENCH_LLM_MODEL"):
            value = parsed.get(key) or ""
            if value and not os.environ.get(key):
                os.environ[key] = value
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
            return str(path)
    raise SystemExit("OPENAI_API_KEY or OPENROUTER_API_KEY is required (backend/.env.local)")


def _reward(candidate: dict[str, Any]) -> float:
    for key in ("train_reward", "minibatch_reward", "heldout_reward"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _summarize(result: Any) -> dict[str, Any]:
    registry_path = Path(result.candidate_registry_path)
    candidates = json.loads(registry_path.read_text()) if registry_path.is_file() else []
    seed = next(
        (item for item in candidates if not item.get("parent_id") or str(item.get("source") or "").startswith("seed")),
        candidates[0] if candidates else {},
    )
    best = result.best_candidate if isinstance(result.best_candidate, dict) else {}
    seed_reward, best_reward = _reward(seed), _reward(best)
    return {
        "seed_candidate_id": seed.get("candidate_id"),
        "best_candidate_id": best.get("candidate_id"),
        "seed_reward": seed_reward,
        "best_reward": best_reward,
        "uplift": round(best_reward - seed_reward, 4),
        "candidate_count": len(candidates),
        "children": [
            {
                "candidate_id": item.get("candidate_id"),
                "parent_id": item.get("parent_id"),
                "train_reward": item.get("train_reward"),
                "minibatch_reward": item.get("minibatch_reward"),
                "heldout_reward": item.get("heldout_reward"),
            }
            for item in candidates
        ],
        "manifest_path": str(result.manifest_path),
        "candidate_registry_path": str(registry_path),
    }


def run_search(
    game: str,
    mode: str,
    *,
    generations: int,
    proposals: int,
    minibatch: int,
    max_rollouts: int,
    output_dir: Path,
    max_steps: int | None,
    max_train: int | None = None,
    max_heldout: int | None = None,
) -> dict[str, Any]:
    from gamebench_levers.stack import start_stack
    from synth_containers import ContainerConnection
    from synth_optimizers.gepa import (
        CacheConfig, GepaBudgetConfig, GepaConfig, GepaPipeline, GepaTaskPools,
        RolloutTransport, RunSettings, TasksetSelection, UsageRegistrationConfig,
    )

    stack = start_stack(game, mode, max_steps=max_steps)
    run_id = f"gepa_{game}_{mode}_{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    try:
        import httpx

        taskset = httpx.get(f"{stack.orch_url}/taskset", timeout=30).json()
        n_train = int(taskset["splits"]["train"])
        n_heldout = int(taskset["splits"]["heldout"])
        if max_train:
            n_train = min(n_train, max_train)
        if max_heldout:
            n_heldout = min(n_heldout, max_heldout)
        train_ids = [f"train:{i}" for i in range(n_train)]
        heldout_ids = [f"heldout:{i}" for i in range(n_heldout)]

        # A minibatch pool the same size as the minibatch makes every proposal see
        # the identical rows; lift then measures nothing. Keep the pool strictly larger.
        minibatch = max(1, min(minibatch, max(1, len(train_ids) - 1)))
        if len(train_ids) <= minibatch:
            raise SystemExit(f"{game}: minibatch pool {len(train_ids)} must exceed minibatch_size {minibatch}")

        print(
            f"[{game}/{mode}] {run_id} orch={stack.orch_url} "
            f"train={len(train_ids)} heldout={len(heldout_ids)} minibatch={minibatch}",
            flush=True,
        )
        config = GepaConfig(
            container=ContainerConnection(url=stack.orch_url),
            taskset=TasksetSelection(
                train_split="train", heldout_split="heldout",
                train_ids=train_ids, heldout_ids=heldout_ids,
            ),
            task_pools=GepaTaskPools(
                # The engine requires minibatch to be a subset of reflection.
                pareto=train_ids, minibatch=train_ids,
                reflection=train_ids, heldout=heldout_ids,
            ),
            policy=None,
            target_modules=TARGETS[mode],
            pipeline=GepaPipeline.sync_serial(
                rollout_transport=RolloutTransport.SYNC, rollout_timeout_seconds=900,
            ),
            budgets=GepaBudgetConfig(
                max_generations=generations,
                proposals_per_generation=proposals,
                minibatch_size=minibatch,
                max_total_rollouts=max_rollouts,
            ),
            cache=CacheConfig(mode="off"),
            run=RunSettings(run_id=run_id, output_dir=output_dir),
            usage_registration=UsageRegistrationConfig(enabled=False),
        )
        summary = _summarize(config.execute())
        summary.update({
            "game": game, "mode": mode, "run_id": run_id,
            "train_tasks": len(train_ids), "heldout_tasks": len(heldout_ids),
            "minibatch_size": minibatch,
            "elapsed_s": round(time.perf_counter() - started, 1),
        })
        return summary
    finally:
        stack.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None)
    parser.add_argument("--mode", default="code", choices=["code", "harness"])
    parser.add_argument("--all-code", action="store_true", help="every game in code mode")
    parser.add_argument("--all-harness", action="store_true", help="every game in harness mode")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--max-rollouts", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=None)
    # Harness rollouts cost one model call per turn, so a game with twice the tasks
    # costs twice the wall-clock. Cap the split to keep games comparable.
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-heldout", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    os.environ.setdefault("SYNTH_OPTIMIZERS_VL_PROJECT", "0")
    print(f"llm keys loaded from {ensure_llm_keys()}")

    from gamebench_levers import GAMES

    jobs: list[tuple[str, str]] = []
    if args.all_code:
        jobs += [(game, "code") for game in GAMES]
    if args.all_harness:
        jobs += [(game, "harness") for game in GAMES]
    if not jobs:
        if not args.game:
            parser.error("pass --game, --all-code or --all-harness")
        jobs = [(args.game, args.mode)]

    output_dir = Path(args.output_dir or ROOT / "runs" / "gamebench_levers")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for game, mode in jobs:
        try:
            row = run_search(
                game, mode,
                generations=args.generations, proposals=args.proposals,
                minibatch=args.minibatch, max_rollouts=args.max_rollouts,
                output_dir=output_dir, max_steps=args.max_steps,
                max_train=args.max_train, max_heldout=args.max_heldout,
            )
        except Exception as exc:  # noqa: BLE001
            row = {"game": game, "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps(row, indent=2, default=str), flush=True)

    print("\n=== GEPA search ===")
    failed = False
    for row in rows:
        label = f"{row['game']}/{row['mode']}"
        if row.get("error"):
            print(f"ERROR {label}: {row['error']}")
            failed = True
            continue
        uplift = float(row.get("uplift") or 0.0)
        mark = "CLIMB" if uplift > 0 else "flat "
        print(
            f"{mark} {label:22s} seed={row['seed_reward']:<10} best={row['best_reward']:<10} "
            f"uplift={uplift:<8} candidates={row['candidate_count']} {row['elapsed_s']}s"
        )
    summary_path = Path(args.output_dir or ROOT / "runs" / "gamebench_levers") / "search_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
