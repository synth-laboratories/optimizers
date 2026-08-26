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
COMPACT_AFTER_TOKENS = int(os.environ.get("EVAL_LLM_COMPACT_AFTER_TOKENS", "3200") or 3200)
COMPACT_TAIL_TURNS = max(1, int(os.environ.get("EVAL_LLM_COMPACT_TAIL_TURNS", "2") or 2))
USAGE_PATH = os.environ.get("EVAL_LLM_USAGE_PATH", "/tmp/work/usage.jsonl")
API_KEY = os.environ.get(os.environ.get("EVAL_LLM_SECRET_NAME", "OPENAI_API_KEY"), "")
USD_IN = float(os.environ.get("EVAL_LLM_USD_PER_1M_INPUT", "0") or 0)
USD_OUT = float(os.environ.get("EVAL_LLM_USD_PER_1M_OUTPUT", "0") or 0)
USD_CACHED = float(os.environ.get("EVAL_LLM_USD_PER_1M_CACHED_INPUT", "0") or 0)
SYSTEM_APPEND = os.environ.get("EVAL_LLM_SYSTEM_APPEND", "").strip()[:1000]

# Completion headroom must clear the reasoning budget, or every plan truncates
# and the trial measures the cap instead of the model.
_HEADROOM = {"none": 2_048, "minimal": 4_096, "low": 4_096, "medium": 16_384, "high": 32_768}
_TIMEOUT = {"none": 90.0, "low": 120.0, "medium": 240.0, "high": 360.0}

SYSTEM_PROMPT = (
    "You are playing Craftax, a 2D survival and crafting game, through a "
    "symbolic text interface.\n"
    "Goal: unlock as many achievements as possible — collect wood, place a "
    "crafting table, make tools, mine stone/coal/iron, and survive.\n"
    "You will receive the opening observation once. After each action batch, "
    "the craftax_interact tool result contains the resulting game state.\n"
    "Think briefly in the assistant content, then call craftax_interact exactly "
    f"once with {PLAN_MIN} to {PLAN_MAX} legal actions. Every action you submit "
    "is executed in order before the next tool result, so commit to a coherent "
    "short plan."
)
if SYSTEM_APPEND:
    SYSTEM_PROMPT += "\n\nAdditional policy instruction:\n" + SYSTEM_APPEND

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
    "history": [],
    "opening_message": None,
    "pending_tool_call": None,
    "turns": [],
    "compactions": 0,
    "last_prompt_tokens": 0,
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


def _tool_spec(valid_actions: list[str]) -> list[dict[str, Any]]:
    # Stable ordering keeps the tool-schema portion of the prompt byte-identical
    # across calls, which lets both MLX and hosted endpoints reuse prefix KV.
    valid_actions = sorted(dict.fromkeys(valid_actions))
    return [{"type": "function", "function": {
        "name": "craftax_interact",
        "description": "Execute one bounded batch of legal Craftax actions.",
        "parameters": {"type": "object", "properties": {"actions": {
            "type": "array", "items": {"type": "string", "enum": valid_actions},
            "minItems": PLAN_MIN, "maxItems": PLAN_MAX,
        }}, "required": ["actions"], "additionalProperties": False},
    }}]


def _complete(
    messages: list[dict[str, Any]], valid_actions: list[str]
) -> tuple[dict[str, Any], dict[str, int | None]]:
    headroom = _HEADROOM.get(EFFORT, 8_192)
    timeout = _TIMEOUT.get(EFFORT, 180.0)
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": headroom,
        "tools": _tool_spec(valid_actions),
        "tool_choice": "required",
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
            raise RuntimeError(f"policy_route_error {error.code}: {detail[:400]}") from error
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    message: dict[str, Any] = {}
    for choice in payload.get("choices") or []:
        candidate = choice.get("message") or {}
        if candidate:
            message = candidate
            break
    cached = details.get("cached_tokens")
    cached_tokens = int(cached) if cached is not None else None
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    return message, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        # Null means the endpoint did not expose cache accounting. It must not
        # be rewritten to zero: "not reported" and "reported no cache hit" are
        # different experimental facts.
        "cached_tokens": cached_tokens,
        "cache_telemetry_reported": cached is not None,
        "uncached_prompt_tokens": prompt_tokens - cached_tokens
        if cached_tokens is not None
        else None,
    }


def _parse_tool_call(message: dict[str, Any], valid_actions: list[str]) -> tuple[dict[str, Any], list[str]]:
    calls = message.get("tool_calls") or []
    if calls:
        call = calls[0]
        function = call.get("function") or {}
        if function.get("name") != "craftax_interact":
            raise ValueError("expected craftax_interact tool call")
        arguments = function.get("arguments") or "{}"
        payload = json.loads(arguments) if isinstance(arguments, str) else arguments
    else:
        text = str(message.get("content") or "")
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if not match:
            raise ValueError("missing craftax_interact tool call")
        envelope = json.loads(match.group(1))
        if envelope.get("name") != "craftax_interact":
            raise ValueError("expected craftax_interact tool call")
        payload = envelope.get("arguments") or {}
        call = {"id": f"call_{_STATE['calls'] + 1}", "type": "function", "function": {
            "name": "craftax_interact", "arguments": json.dumps(payload),
        }}
    raw = payload.get("actions") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        raise ValueError("tool arguments.actions must be an array")
    legal = {action.lower(): action for action in valid_actions}
    actions = [legal[str(item).strip().lower()] for item in raw if str(item).strip().lower() in legal]
    if not actions:
        raise ValueError("tool call contained no legal actions")
    return call, actions[:PLAN_MAX]


def _state_result(observation_text: str, valid_actions: list[str], ply: int) -> str:
    return (
        f"State after the previous action batch (environment step {ply}):\n"
        f"{observation_text}\n\nLegal actions: {', '.join(valid_actions)}"
    )


def _maybe_compact() -> None:
    history = _STATE["history"]
    if _STATE["last_prompt_tokens"] < COMPACT_AFTER_TOKENS:
        return
    completed = list(_STATE["turns"])
    if len(completed) <= COMPACT_TAIL_TURNS:
        return
    dropped = completed[:-COMPACT_TAIL_TURNS]
    retained = completed[-COMPACT_TAIL_TURNS:]
    summary = ["Compacted append-only Craftax history:"]
    for turn in dropped:
        thinking = str(turn.get("thinking") or "").strip().replace("\n", " ")[:240]
        summary.append(f"- actions={json.dumps(turn.get('actions') or [])}; thinking={thinking or '—'}")
    opening = _STATE.get("opening_message") or history[1]
    rebuilt: list[dict[str, Any]] = [history[0], opening, {
        "role": "user",
        "content": "\n".join(summary) + "\n\nContinue from the retained recent tool states.",
    }]
    for turn in retained:
        rebuilt.extend([turn["assistant"], turn["tool"]])
    _STATE["history"] = rebuilt
    _STATE["turns"] = retained
    _STATE["compactions"] += 1
    _record({
        "event": "context_compacted",
        "prompt_tokens_before": _STATE["last_prompt_tokens"],
        "dropped_turns": len(dropped),
        "retained_turns": len(retained),
        "compaction_count": _STATE["compactions"],
    })


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
        f"Call craftax_interact with {PLAN_MIN}-{PLAN_MAX} actions."
    )
    if not _STATE["history"]:
        opening = {"role": "user", "content": user}
        _STATE["opening_message"] = opening
        _STATE["history"] = [{"role": "system", "content": SYSTEM_PROMPT}, opening]
    elif _STATE["pending_tool_call"] is not None:
        pending = _STATE["pending_tool_call"]
        tool = {
            "role": "tool",
            "tool_call_id": str(pending["call"].get("id") or "call"),
            "content": _state_result(observation_text, valid_actions, ply),
        }
        _STATE["history"].append(tool)
        pending["tool"] = tool
        _STATE["turns"].append(pending)
        _STATE["pending_tool_call"] = None
        _maybe_compact()
    started = time.time()
    try:
        message, usage = _complete(_STATE["history"], valid_actions)
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

    elapsed_s = max(time.time() - started, 0.000001)
    billable_cached = usage["cached_tokens"] or 0
    call_usd = _cost(usage["prompt_tokens"], usage["completion_tokens"], billable_cached)
    _STATE["calls"] += 1
    _STATE["prompt_tokens"] += usage["prompt_tokens"]
    _STATE["completion_tokens"] += usage["completion_tokens"]
    _STATE["cached_tokens"] += billable_cached
    _STATE["usd"] += call_usd
    _STATE["last_prompt_tokens"] = usage["prompt_tokens"]
    thinking = str(message.get("reasoning_content") or message.get("content") or "").strip()
    try:
        call, plan = _parse_tool_call(message, valid_actions)
    except (ValueError, json.JSONDecodeError) as error:
        plan = _parse_plan(str(message.get("content") or ""), valid_actions)
        call = {"id": f"call_{_STATE['calls']}", "type": "function", "function": {
            "name": "craftax_interact", "arguments": json.dumps({"actions": plan}),
        }}
        if not plan:
            _record({"event": "invalid_tool_call", "ply": ply, "error": str(error)})
    assistant = {"role": "assistant", "content": thinking or None, "tool_calls": [call]}
    _STATE["history"].append(assistant)
    _STATE["pending_tool_call"] = {
        "assistant": assistant, "call": call, "actions": plan, "thinking": thinking,
    }
    _record(
        {
            "event": "policy.call",
            "ply": ply,
            "seed": seed,
            "model": MODEL,
            "effort": EFFORT,
            "elapsed_s": round(elapsed_s, 3),
            "tokens_per_second": round(usage["completion_tokens"] / elapsed_s, 3),
            "usd": call_usd,
            "plan": plan,
            "observation_text": observation_text,
            "thinking": thinking[:4000],
            "tool_call": call,
            "context_messages": len(_STATE["history"]),
            "context_completed_turns": len(_STATE["turns"]),
            "context_compactions": _STATE["compactions"],
            **usage,
        }
    )
    if not plan:
        return {"actions": [fallback], "rationale": "no legal action parsed"}
    return {"actions": plan, "rationale": "llm plan"}
