"""An LLM Craftax policy, in the shape the GameBench sweep already calls.

The sweep asks a policy module for `choose_actions(...)` and applies the batch
it returns. That is all this is: the same entry point, backed by a chat
completion instead of a heuristic, so an LLM candidate needs no change to the
evaluator, the target, or the eval runner.

Everything that decides cost — the model, the route, the reasoning effort, the
rates, and the caps — is set by the trusted target wrapper from the recipe's
model allowlist. The candidate contributes only which allowlisted model and
effort to use. A policy cannot point this at another endpoint.

Every call's tokens are appended to a usage file the wrapper reads back, so the
dollars a trial reports are computed from provider-reported tokens and a
recipe-declared rate, never estimated after the fact.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

ROUTE = os.environ.get("EVAL_LLM_ROUTE", "")
MODEL = os.environ.get("EVAL_LLM_MODEL", "")
EFFORT = os.environ.get("EVAL_LLM_EFFORT", "")
TEMPERATURE = float(os.environ.get("EVAL_LLM_TEMPERATURE", "0") or 0)
PLAN_MIN = int(os.environ.get("EVAL_LLM_PLAN_MIN", "5") or 5)
PLAN_MAX = int(os.environ.get("EVAL_LLM_PLAN_MAX", "20") or 20)
MAX_CALLS = int(os.environ.get("EVAL_LLM_MAX_CALLS", "40") or 40)
MAX_USD = float(os.environ.get("EVAL_LLM_MAX_USD", "0.25") or 0.25)
USAGE_PATH = os.environ.get("EVAL_LLM_USAGE_PATH", "/tmp/work/usage.jsonl")
API_KEY = os.environ.get(os.environ.get("EVAL_LLM_SECRET_NAME", "OPENAI_API_KEY"), "")
USD_IN = float(os.environ.get("EVAL_LLM_USD_PER_1M_INPUT", "0") or 0)
USD_OUT = float(os.environ.get("EVAL_LLM_USD_PER_1M_OUTPUT", "0") or 0)
USD_CACHED = float(os.environ.get("EVAL_LLM_USD_PER_1M_CACHED_INPUT", "0") or 0)

# Completion headroom must clear the reasoning budget, or every plan truncates
# and the trial measures the cap instead of the model.
_HEADROOM = {"none": 2_048, "minimal": 4_096, "low": 4_096, "medium": 16_384, "high": 32_768}
_TIMEOUT = {"none": 90.0, "low": 120.0, "medium": 240.0, "high": 360.0}

SYSTEM_PROMPT = (
    "You are playing Craftax, a 2D survival and crafting game, through a "
    "symbolic text interface.\n"
    "Goal: unlock as many achievements as possible — collect wood, place a "
    "crafting table, make tools, mine stone/coal/iron, and survive.\n"
    "You will be shown the current observation and the legal actions.\n"
    f'Reply with ONLY a JSON object: {{"actions": [...], "rationale": "..."}} '
    f"where actions is a list of {PLAN_MIN} to {PLAN_MAX} action names from the "
    "legal set. Every action you submit is executed in order before you are "
    "asked again, so commit to a plan you believe in and keep the rationale to "
    "one short sentence."
)

_STATE: dict[str, Any] = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "usd": 0.0,
    "budget_exhausted": None,
    "exhausted_at_ply": None,
    "filler_steps": 0,
    "errors": 0,
}


def _cost(prompt: int, completion: int, cached: int) -> float:
    fresh = max(0, prompt - cached)
    return (
        fresh * USD_IN / 1_000_000
        + cached * USD_CACHED / 1_000_000
        + completion * USD_OUT / 1_000_000
    )


def _record(entry: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(USAGE_PATH) or ".", exist_ok=True)
    with open(USAGE_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _post(body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        ROUTE,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _complete(messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
    headroom = _HEADROOM.get(EFFORT, 8_192)
    timeout = _TIMEOUT.get(EFFORT, 180.0)
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": headroom,
    }
    if EFFORT:
        body["reasoning_effort"] = EFFORT
    if TEMPERATURE:
        body["temperature"] = TEMPERATURE
    try:
        payload = _post(body, timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        # A reasoning route that rejects an optional knob should cost one retry,
        # not the whole trial.
        if error.code == 400 and ("temperature" in detail or "reasoning" in detail):
            body.pop("temperature", None)
            payload = _post(body, timeout)
        else:
            if error.code in {401, 403}:
                code = "provider_auth_rejected"
            elif error.code == 429:
                code = "provider_rate_limited"
            elif error.code >= 500:
                code = "provider_unavailable"
            else:
                code = f"policy_route_error_{error.code}"
            raise RuntimeError(f"{code}: {detail[:400]}") from error
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    text = ""
    for choice in payload.get("choices") or []:
        text = (choice.get("message") or {}).get("content") or ""
        if text:
            break
    return text, {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
    }


def _parse_plan(text: str, valid_actions: list[str]) -> list[str]:
    """Take the model's plan, keep only legal actions, and bound its length."""

    plan: list[str] = []
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            raw = payload.get("actions")
            if isinstance(raw, list):
                plan = [str(item).strip().lower() for item in raw]
        except json.JSONDecodeError:
            plan = []
    if not plan:
        # Fall back to bare action words in order; a malformed reply is still a
        # decision if it names legal moves.
        legal = {action.lower() for action in valid_actions}
        plan = [word for word in re.findall(r"[a-z_]+", text.lower()) if word in legal]
    legal = {action.lower(): action for action in valid_actions}
    plan = [legal[action] for action in plan if action in legal]
    return plan[:PLAN_MAX]


def choose_actions(
    *,
    observation_text: str,
    session: dict[str, Any],
    valid_actions: list[str],
    engine: Any = None,
    readout: dict[str, Any],
    seed: int,
    ply: int,
) -> dict[str, Any]:
    fallback = "noop" if "noop" in valid_actions else valid_actions[0]

    # Past the budget this is no longer the candidate's policy — it is `noop`
    # wearing the candidate's name. Record the moment it stops so the trial can
    # say how much of the episode the model actually played; a score averaged
    # over model plays and filler steps, presented as one number, is not
    # evidence about the model.
    def _exhaust(reason: str) -> dict[str, Any]:
        if not _STATE["budget_exhausted"]:
            _STATE["budget_exhausted"] = reason
            _STATE["exhausted_at_ply"] = ply
            _record({"event": "budget_exhausted", "reason": reason, "ply": ply})
        _STATE["filler_steps"] += 1
        return {
            "actions": [],
            "rationale": reason,
            "stop_episode": True,
            "stop_reason": reason,
        }

    if _STATE["budget_exhausted"]:
        return _exhaust(_STATE["budget_exhausted"])
    if _STATE["calls"] >= MAX_CALLS:
        return _exhaust(f"call cap reached ({MAX_CALLS})")
    if _STATE["usd"] >= MAX_USD:
        return _exhaust(f"spend cap reached (${MAX_USD})")

    user = (
        f"Step {ply}, seed {seed}.\n\n"
        f"{observation_text}\n\n"
        f"Legal actions: {', '.join(valid_actions)}\n"
        f"Reply with {PLAN_MIN}-{PLAN_MAX} actions as JSON."
    )
    started = time.time()
    try:
        text, usage = _complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
    except Exception as error:  # noqa: BLE001 - a route failure is recorded, not hidden
        _STATE["errors"] += 1
        _record(
            {
                "ply": ply,
                "seed": seed,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_s": round(time.time() - started, 3),
            }
        )
        if _STATE["errors"] >= 3:
            # A dead route stops the model just as surely as a spent budget, and
            # the rest of the episode is filler either way. Record it the same.
            return _exhaust(f"route failed {_STATE['errors']} times")
        return {"actions": [fallback], "rationale": "route error"}

    call_usd = _cost(usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"])
    _STATE["calls"] += 1
    _STATE["prompt_tokens"] += usage["prompt_tokens"]
    _STATE["completion_tokens"] += usage["completion_tokens"]
    _STATE["cached_tokens"] += usage["cached_tokens"]
    _STATE["usd"] += call_usd
    plan = _parse_plan(text, valid_actions)
    _record(
        {
            "ply": ply,
            "seed": seed,
            "model": MODEL,
            "effort": EFFORT,
            "elapsed_s": round(time.time() - started, 3),
            "usd": call_usd,
            "plan": plan,
            **usage,
        }
    )
    if not plan:
        return {"actions": [fallback], "rationale": "no legal action parsed"}
    return {"actions": plan, "rationale": "llm plan"}
