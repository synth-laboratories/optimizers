#!/usr/bin/env python3
"""Run GEPA search on the toy Craftax stacks (code + ReAct prompt)."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from craftax_levers.stack import start_stack
from synth_containers import ContainerConnection
from synth_optimizers.gepa import (
    CacheConfig,
    GepaBudgetConfig,
    GepaConfig,
    GepaPipeline,
    GepaTaskPools,
    RolloutTransport,
    RunSettings,
    TasksetSelection,
    UsageRegistrationConfig,
)

ROOT = Path(__file__).resolve().parents[2]
KEY_FILES = (
    Path.home() / "Documents/GitHub/backend/.env.local",
    Path.home() / "Documents/GitHub/backend/.env",
    Path.home() / "Documents/GitHub/evals/.env",
    ROOT / ".env.local",
    ROOT / ".env",
)


def _load_dotenv(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            loaded[key] = value
    return loaded


def _ensure_llm_keys() -> str:
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        return "environment"
    for path in KEY_FILES:
        if not path.is_file():
            continue
        parsed = _load_dotenv(path)
        applied = False
        for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "CRAFTAX_LLM_MODEL"):
            value = parsed.get(key) or ""
            if value and not os.environ.get(key):
                os.environ[key] = value
                applied = True
        if applied and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            return str(path)
    raise SystemExit(
        "OPENAI_API_KEY or OPENROUTER_API_KEY is required; "
        "put it in the environment or backend/.env.local"
    )


def _reward(candidate: dict) -> float:
    for key in ("train_reward", "minibatch_reward", "heldout_reward"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _summarize(result) -> dict:
    registry_path = Path(result.candidate_registry_path)
    candidates = json.loads(registry_path.read_text()) if registry_path.is_file() else []
    seed = next(
        (
            item
            for item in candidates
            if not item.get("parent_id") or str(item.get("source") or "").startswith("seed")
        ),
        candidates[0] if candidates else {},
    )
    best = result.best_candidate if isinstance(result.best_candidate, dict) else {}
    seed_reward = _reward(seed)
    best_reward = _reward(best)
    return {
        "best_candidate_id": best.get("candidate_id"),
        "seed_candidate_id": seed.get("candidate_id"),
        "seed_reward": seed_reward,
        "best_reward": best_reward,
        "uplift": best_reward - seed_reward,
        "candidate_count": len(candidates),
        "children": [
            {
                "candidate_id": item.get("candidate_id"),
                "parent_id": item.get("parent_id"),
                "source": item.get("source"),
                "train_reward": item.get("train_reward"),
                "minibatch_reward": item.get("minibatch_reward"),
            }
            for item in candidates
        ],
        "manifest_path": result.manifest_path,
        "candidate_registry_path": str(registry_path),
    }


def _config(url: str, *, target_modules: list[str], run_id: str, output_dir: Path) -> GepaConfig:
    ids = ["train:0"]
    heldout = ["heldout:0"]
    return GepaConfig(
        container=ContainerConnection(url=url),
        taskset=TasksetSelection(
            train_split="train",
            heldout_split="heldout",
            train_ids=ids,
            heldout_ids=heldout,
        ),
        task_pools=GepaTaskPools(
            pareto=ids,
            minibatch=ids,
            reflection=ids,
            heldout=heldout,
        ),
        policy=None,
        target_modules=target_modules,
        pipeline=GepaPipeline.sync_serial(
            rollout_transport=RolloutTransport.SYNC,
            rollout_timeout_seconds=600,
        ),
        budgets=GepaBudgetConfig(
            max_generations=2,
            proposals_per_generation=1,
            minibatch_size=1,
            max_total_rollouts=24,
        ),
        cache=CacheConfig(mode="off"),
        run=RunSettings(run_id=run_id, output_dir=output_dir),
        usage_registration=UsageRegistrationConfig(enabled=False),
    )


def _run(mode: str, target_modules: list[str], output_dir: Path) -> dict:
    stack = start_stack(mode)
    run_id = f"gepa_craftax_{mode}_{uuid.uuid4().hex[:8]}"
    print(f"starting {mode} search {run_id} orch={stack.orch_url}", flush=True)
    try:
        result = _config(
            stack.orch_url,
            target_modules=target_modules,
            run_id=run_id,
            output_dir=output_dir,
        ).execute()
        summary = _summarize(result)
        summary.update({"mode": mode, "run_id": run_id, "orch_url": stack.orch_url})
        return summary
    finally:
        stack.stop()


def main() -> int:
    os.environ.setdefault("SYNTH_OPTIMIZERS_VL_PROJECT", "0")
    key_source = _ensure_llm_keys()
    print(f"llm keys loaded from {key_source}")
    output_dir = Path(os.environ.get("GEPA_CRAFTAX_OUTPUT", ROOT / "runs" / "craftax_levers"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run("code", ["policy_script"], output_dir),
        _run("react", ["react_system_prompt"], output_dir),
    ]
    failed = False
    for row in rows:
        print(json.dumps(row, indent=2, default=str))
        if not row.get("best_candidate_id"):
            print(f"{row['mode']}: GEPA did not produce a candidate", file=sys.stderr)
            failed = True
        elif float(row.get("uplift") or 0.0) <= 0.0:
            print(
                f"{row['mode']}: no train uplift "
                f"(seed={row.get('seed_reward')} best={row.get('best_reward')})",
                file=sys.stderr,
            )
            failed = True
        else:
            print(
                f"{row['mode']}: {row['seed_reward']} -> {row['best_reward']} "
                f"(+{row['uplift']}) over {row['candidate_count']} candidates"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
