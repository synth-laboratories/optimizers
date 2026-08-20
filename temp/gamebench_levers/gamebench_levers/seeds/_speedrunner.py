"""Shared SpeedRunner harness scaffold.

SpeedRunner-style actor (arXiv:2608.11338): the LLM picks a *public skill* by name;
the skill is an ordinary program that expands into primitive env steps with no
further model call. Optimizing this file is harness search, not prompt search --
GEPA can add skills, change the arbiter, or rewrite the observation summary.
"""

from __future__ import annotations

PREFIX = '''\
"""SpeedRunner actor. The LLM picks a public skill; skills run primitives alone."""

import json

MAX_LLM_CALLS = 12


def summarize(obs):
    """Compact view handed to the model. Keep it small and decision-relevant."""
    state = obs.get("state") or {}
    return json.dumps(
        {
            "tick": obs.get("tick"),
            "score": obs.get("score"),
            "ascii": obs.get("ascii"),
            "achievements": obs.get("achievements"),
            "state": {key: state.get(key) for key in SUMMARY_KEYS if key in state},
            "public_skills": list(PUBLIC_SKILLS),
        },
        default=str,
    )


def parse_choice(text):
    """Pull {"skill": ...} or {"action": ...} out of the model reply."""
    raw = text or ""
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except Exception:
            data = {}
        skill = str(data.get("skill") or "")
        if skill in PUBLIC_SKILLS:
            return {"kind": "skill", "value": skill}
        action = data.get("action")
        if action:
            return {"kind": "action", "value": action}
    for name in PUBLIC_SKILLS:
        if name in raw:
            return {"kind": "skill", "value": name}
    return {"kind": "action", "value": FALLBACK_ACTION}
'''

SUFFIX = '''

def run_episode(env, prompt, seed=0, max_steps=None, llm=None):
    """Entry point the policy service calls. Must return a dict with `reward`."""
    if llm is None:
        raise RuntimeError("SpeedRunner harness requires an llm")
    obs = env.reset(seed, max_steps)
    events = []
    tool_calls = []
    skills_used = []
    calls = 0
    messages = [{"role": "system", "content": prompt or ""}]
    while not obs.get("done") and calls < MAX_LLM_CALLS:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Observation:\\n"
                    + summarize(obs)
                    + "\\nReply with JSON only: {\\"skill\\": \\"<one public skill>\\"} "
                    "or {\\"action\\": \\"<one primitive action>\\"}."
                ),
            }
        )
        text = llm(messages)
        calls += 1
        events.append({"type": "llm_request", "response": text})
        messages.append({"role": "assistant", "content": text})
        choice = parse_choice(text)
        if choice["kind"] == "skill":
            skills_used.append(choice["value"])
            events.append({"type": "skill_start", "skill": choice["value"], "tick": obs.get("tick")})
            obs, ran = PUBLIC_SKILLS[choice["value"]](env, obs)
            tool_calls.extend(ran)
            events.append({"type": "skill_end", "skill": choice["value"], "primitives": len(ran)})
        else:
            action = choice["value"]
            stepped = env.step(action)
            obs = stepped["obs"]
            tool_calls.append({"tick": obs.get("tick"), "action": action})
            events.append({"type": "env_step", "action": action, "reward": stepped.get("reward")})
    info = env.last_info or {}
    return {
        "reward": float(info.get("outcome_reward") or obs.get("score") or 0.0),
        "events": events,
        "tool_calls": tool_calls,
        "achievements": list(obs.get("achievements") or []),
        "architecture": "speedrunner_skill_arbiter",
        "skills_used": skills_used,
        "llm_calls": calls,
    }
'''


def build(body: str) -> str:
    """Assemble a complete harness script from a per-game skill library."""
    return PREFIX + body + SUFFIX
