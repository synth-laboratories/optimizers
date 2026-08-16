"""Resolving a candidate into the policy file the GameBench sweep runs.

Two candidate kinds reach these targets:

* `python-code.craftax-choose-actions.v1` — the candidate *is* the policy.
* `llm-policy.v1` — the candidate is a small TOML naming an allowlisted model
  and reasoning effort; the policy code is the image's, not the candidate's.

The second is why cost is trustworthy: the candidate never supplies the route
or the rates, only a choice from the allowlist the recipe declared, so a trial
cannot quietly bill against a different model than the one it reports.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

LLM_POLICY = Path("/opt/eval/llm_policy.py")


class CandidateError(ValueError):
    """The candidate does not satisfy the target's policy contract."""


def resolve_policy(trial: dict[str, Any], input_dir: Path, work: Path) -> tuple[Path, dict]:
    """Return (policy file, extra process env) for this candidate."""

    candidate = trial["candidate"]
    kind = candidate.get("kind", "python-code.craftax-choose-actions.v1")
    if kind == "llm-policy.v1":
        return _llm_policy(trial, input_dir, work)
    return _code_policy(candidate, input_dir), {}


def _code_policy(candidate: dict[str, Any], input_dir: Path) -> Path:
    entrypoint = str(candidate["entrypoint"])
    module_name, _, attribute = entrypoint.partition(":")
    if (attribute or "choose_actions") != "choose_actions":
        raise CandidateError(f"Craftax code policies must expose choose_actions, not {attribute!r}")
    path = input_dir / "policy" / f"{module_name}.py"
    if not path.is_file():
        raise CandidateError(f"policy module {module_name}.py is not in the mounted candidate")
    return path


def _llm_policy(trial: dict[str, Any], input_dir: Path, work: Path) -> tuple[Path, dict]:
    config_path = input_dir / "policy" / "policy.toml"
    if not config_path.is_file():
        raise CandidateError("an llm-policy.v1 candidate must contain policy.toml")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model_id = str(config.get("model", "")).strip()
    effort = str(config.get("effort", "")).strip()
    if not model_id:
        raise CandidateError("policy.toml must name a model")

    allowlist = {entry["id"]: entry for entry in trial.get("models") or []}
    route = allowlist.get(model_id)
    if route is None:
        raise CandidateError(
            f"model {model_id!r} is not in this recipe's allowlist "
            f"({', '.join(sorted(allowlist)) or 'none'})"
        )
    if effort and effort not in (route.get("efforts") or []):
        raise CandidateError(
            f"effort {effort!r} is not permitted for {model_id} "
            f"({', '.join(route.get('efforts') or []) or 'none'})"
        )
    budget = trial.get("budget") or {}
    if not budget:
        raise CandidateError("a paid policy needs a recipe-declared budget")

    usage_path = work / "usage.jsonl"
    env = {
        "EVAL_LLM_ROUTE": route["route"],
        "EVAL_LLM_MODEL": model_id,
        "EVAL_LLM_EFFORT": effort,
        "EVAL_LLM_SECRET_NAME": route["secret"],
        "EVAL_LLM_TEMPERATURE": str(config.get("temperature", 0)),
        "EVAL_LLM_PLAN_MIN": str(config.get("plan_min", 5)),
        "EVAL_LLM_PLAN_MAX": str(config.get("plan_max", 20)),
        "EVAL_LLM_MAX_CALLS": str(budget["max_llm_calls"]),
        "EVAL_LLM_MAX_USD": str(budget["max_usd"]),
        "EVAL_LLM_USD_PER_1M_INPUT": str(route["usd_per_1m_input"]),
        "EVAL_LLM_USD_PER_1M_OUTPUT": str(route["usd_per_1m_output"]),
        "EVAL_LLM_USD_PER_1M_CACHED_INPUT": str(route["usd_per_1m_cached_input"]),
        "EVAL_LLM_USAGE_PATH": str(usage_path),
    }
    if not os.environ.get(route["secret"], "").strip():
        raise CandidateError(
            f"{route['secret']} is not present in the trial container; the recipe "
            f"must declare it and the eval home must supply it"
        )
    return LLM_POLICY, env


def summarize_usage(work: Path) -> dict[str, Any]:
    """Fold the policy's per-call records into one trial usage summary.

    Tokens are provider-reported and dollars come from the recipe's declared
    rate. A trial with no records reports zeros for a policy that made no calls
    — which is a real fact about a code policy, not a missing measurement.
    """

    path = work / "usage.jsonl"
    summary = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.0,
        "route_errors": 0,
        "llm_seconds": 0.0,
        # Null for a policy that played its whole episode. Set once the model
        # stopped choosing and the harness started filling, so a reader can tell
        # a score the model earned from one a fallback coasted to.
        "budget_exhausted": None,
        "exhausted_at_ply": None,
    }
    if not path.is_file():
        return summary
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "budget_exhausted":
            summary["budget_exhausted"] = str(entry.get("reason") or "budget exhausted")
            ply = entry.get("ply")
            summary["exhausted_at_ply"] = int(ply) if isinstance(ply, int) else None
            continue
        summary["llm_seconds"] += float(entry.get("elapsed_s") or 0.0)
        if entry.get("error"):
            summary["route_errors"] += 1
            continue
        summary["calls"] += 1
        summary["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        summary["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        summary["cached_tokens"] += int(entry.get("cached_tokens") or 0)
        summary["cost_usd"] += float(entry.get("usd") or 0.0)
    summary["cost_usd"] = round(summary["cost_usd"], 6)
    summary["llm_seconds"] = round(summary["llm_seconds"], 3)
    return summary
