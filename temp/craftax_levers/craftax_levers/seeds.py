"""Seed policy_script / ReAct+LLM loop text. These are the GEPA search objects."""

SEED_POLICY = '''\
def act(obs):
    """Noop policy. Does not collect wood on the seed-0 fixture."""
    return "noop"
'''

GREEDY_POLICY = '''\
def act(obs):
    """Collect adjacent wood; otherwise step toward the nearest W and skip lava."""
    if obs.get("adjacent", {}).get("wood"):
        return "collect"
    grid = obs["grid"]
    x, y = int(obs["x"]), int(obs["y"])
    targets = [
        (gx, gy)
        for gy, row in enumerate(grid)
        for gx, ch in enumerate(row)
        if ch == "W"
    ]
    if not targets:
        return "noop"
    tx, ty = min(targets, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
    order = []
    if tx > x:
        order.append("right")
    if tx < x:
        order.append("left")
    if ty > y:
        order.append("down")
    if ty < y:
        order.append("up")
    lava = {
        (gx, gy)
        for gy, row in enumerate(grid)
        for gx, ch in enumerate(row)
        if ch == "L"
    }
    delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    for action in order:
        dx, dy = delta[action]
        if (x + dx, y + dy) not in lava:
            return action
    return "noop"
'''

SEED_PROMPT = (
    "You are a Craftax survival agent. Always reply with exactly: action: noop. "
    "Do not collect wood. Do not move."
)

WOOD_PROMPT = (
    "You are a Craftax survival agent. Collect wood first. Never step on lava. "
    "If wood is adjacent, collect it. Otherwise move toward the nearest wood."
)

# LLM ReAct: every tick calls llm(), steps parse_action(that text) only.
SEED_HARNESS = '''\
"""LLM ReAct loop. Action is parse_action(llm(messages)) only."""

ACTIONS = ("collect", "up", "down", "left", "right", "noop")


def format_obs(obs):
    return (
        f"tick={obs['tick']} pos=({obs['x']},{obs['y']}) "
        f"grid={obs['grid']} inv={obs['inventory']} "
        f"adjacent_wood={str(obs.get('adjacent', {}).get('wood')).lower()}"
    )


def parse_action(text):
    lowered = (text or "").lower()
    for action in ACTIONS:
        if f"action: {action}" in lowered:
            return action
    return "noop"


def run_episode(env, prompt, seed=0, max_steps=16, llm=None):
    if llm is None:
        raise RuntimeError("ReAct script requires llm")
    obs = env.reset(seed, max_steps)
    events = []
    tool_calls = []
    messages = [{"role": "system", "content": prompt or ""}]
    while not obs.get("done"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "Observation:\\n"
                    + format_obs(obs)
                    + "\\nReply with one Thought line and one Action line "
                    "(collect|up|down|left|right|noop)."
                ),
            }
        )
        text = llm(messages)
        events.append({"type": "llm_request", "messages": list(messages), "response": text})
        action = parse_action(text)
        messages.append({"role": "assistant", "content": text})
        tool_calls.append({"tick": obs["tick"], "action": action})
        events.append({"type": "tool_call", "action": action, "tick": obs["tick"]})
        stepped = env.step(action)
        obs = stepped["obs"]
        events.append({"type": "env_step", "reward": stepped.get("reward")})
        if stepped.get("terminated") or obs.get("done"):
            break
    info = env.last_info
    return {
        "reward": float(info.get("outcome_reward") or 0.0),
        "events": events,
        "tool_calls": tool_calls,
        "achievements": list(info.get("achievements") or []),
        "death_cause": info.get("death_cause"),
        "architecture": "react_llm_thought_action",
        "llm_calls": len([e for e in events if e.get("type") == "llm_request"]),
    }
'''

# SpeedRunner-style actor (arXiv:2608.11338): LLM selects a public skill; the skill
# is an executable program over primitive env actions. No LLM call inside the skill.
# Ship a different library by POSTing a new harness_module (harness_restart.v1).

_SPEEDRUNNER_PREFIX = '''\
"""SpeedRunner actor. LLM chooses a public skill; skills run deterministic primitives."""

import json

ACTIONS = ("collect", "up", "down", "left", "right", "noop")
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def format_obs(obs):
    adj = obs.get("adjacent") or {}
    return json.dumps(
        {
            "tick": obs["tick"],
            "pos": [obs["x"], obs["y"]],
            "grid": obs["grid"],
            "wood": (obs.get("inventory") or {}).get("wood"),
            "adjacent_wood": bool(adj.get("wood")),
            "adjacent_lava": bool(adj.get("lava")),
            "legal": obs.get("legal_actions"),
            "public_skills": list(PUBLIC_SKILLS),
        }
    )


def parse_choice(text):
    raw = text or ""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {"kind": "action", "value": "noop"}
    data = json.loads(raw[start : end + 1])
    skill = str(data.get("skill") or "")
    if skill in PUBLIC_SKILLS:
        return {"kind": "skill", "value": skill}
    action = str(data.get("action") or "noop")
    if action in ACTIONS:
        return {"kind": "action", "value": action}
    return {"kind": "action", "value": "noop"}


def _wood_cells(obs):
    grid = obs["grid"]
    return [
        (gx, gy)
        for gy, row in enumerate(grid)
        for gx, ch in enumerate(row)
        if ch == "W"
    ]


def _step_toward_wood(obs):
    if obs.get("adjacent", {}).get("wood"):
        return "collect"
    targets = _wood_cells(obs)
    if not targets:
        return "noop"
    x, y = int(obs["x"]), int(obs["y"])
    tx, ty = min(targets, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
    order = []
    if tx > x:
        order.append("right")
    if tx < x:
        order.append("left")
    if ty > y:
        order.append("down")
    if ty < y:
        order.append("up")
    lava = {
        (gx, gy)
        for gy, row in enumerate(obs["grid"])
        for gx, ch in enumerate(row)
        if ch == "L"
    }
    for action in order:
        dx, dy = DELTAS[action]
        if (x + dx, y + dy) not in lava:
            return action
    return "noop"


def _emit_primitive(env, obs, events, tool_calls, action, skill):
    tool_calls.append({"tick": obs["tick"], "action": action, "skill": skill})
    events.append({"type": "skill_primitive", "skill": skill, "action": action, "tick": obs["tick"]})
    stepped = env.step(action)
    events.append({"type": "env_step", "reward": stepped.get("reward"), "skill": skill})
    return stepped["obs"], stepped
'''

_SPEEDRUNNER_RUN = '''\

def run_episode(env, prompt, seed=0, max_steps=16, llm=None):
    if llm is None:
        raise RuntimeError("ReAct script requires llm")
    obs = env.reset(seed, max_steps)
    events = []
    tool_calls = []
    docs = "\\n".join(f"- {name}: {doc}" for name, doc in PUBLIC_SKILLS.items())
    messages = [
        {
            "role": "system",
            "content": (
                (prompt or "")
                + "\\nYou are a SpeedRunner actor. Public skills:\\n"
                + docs
                + "\\nReply ONLY JSON {\\"skill\\": name} to run a program, or "
                + "{\\"action\\": collect|up|down|left|right|noop} for one primitive."
            ),
        }
    ]
    while not obs.get("done"):
        messages.append({"role": "user", "content": "Observation JSON:\\n" + format_obs(obs)})
        text = llm(messages)
        events.append({"type": "llm_request", "messages": list(messages), "response": text})
        choice = parse_choice(text)
        messages.append({"role": "assistant", "content": text})
        if choice["kind"] == "skill":
            events.append({"type": "skill_invoke", "skill": choice["value"], "tick": obs["tick"]})
            obs = SKILL_IMPL[choice["value"]](env, obs, events, tool_calls)
        else:
            action = choice["value"]
            tool_calls.append({"tick": obs["tick"], "action": action})
            events.append({"type": "tool_call", "action": action, "tick": obs["tick"]})
            stepped = env.step(action)
            obs = stepped["obs"]
            events.append({"type": "env_step", "reward": stepped.get("reward")})
        if obs.get("done"):
            break
    info = env.last_info
    return {
        "reward": float(info.get("outcome_reward") or 0.0),
        "events": events,
        "tool_calls": tool_calls,
        "achievements": list(info.get("achievements") or []),
        "death_cause": info.get("death_cause"),
        "architecture": "react_llm_speedrunner",
        "llm_calls": len([e for e in events if e.get("type") == "llm_request"]),
        "skill_invokes": len([e for e in events if e.get("type") == "skill_invoke"]),
    }
'''

SPEEDRUNNER_SKILLS: dict[str, tuple[str, str]] = {
    "stand_still": (
        "Do nothing until the episode ends. Never collect wood.",
        '''
def stand_still(env, obs, events, tool_calls):
    while not obs.get("done"):
        obs, stepped = _emit_primitive(env, obs, events, tool_calls, "noop", "stand_still")
        if stepped.get("terminated") or obs.get("done"):
            break
    return obs
''',
    ),
    "walk_into_lava": (
        "Walk into the nearest lava cell. Ends the episode with reward 0.",
        '''
def walk_into_lava(env, obs, events, tool_calls):
    while not obs.get("done"):
        lava = [
            (gx, gy)
            for gy, row in enumerate(obs["grid"])
            for gx, ch in enumerate(row)
            if ch == "L"
        ]
        if not lava:
            break
        x, y = int(obs["x"]), int(obs["y"])
        tx, ty = min(lava, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        if tx > x:
            action = "right"
        elif tx < x:
            action = "left"
        elif ty > y:
            action = "down"
        elif ty < y:
            action = "up"
        else:
            break
        obs, stepped = _emit_primitive(env, obs, events, tool_calls, action, "walk_into_lava")
        if stepped.get("terminated") or obs.get("done"):
            break
    return obs
''',
    ),
    "collect_adjacent_once": (
        "If wood is adjacent, collect it once, then return. Does not chase remaining wood.",
        '''
def collect_adjacent_once(env, obs, events, tool_calls):
    if obs.get("done"):
        return obs
    action = "collect" if obs.get("adjacent", {}).get("wood") else "noop"
    obs, _stepped = _emit_primitive(env, obs, events, tool_calls, action, "collect_adjacent_once")
    return obs
''',
    ),
    "collect_all_wood": (
        "Collect every remaining wood on the visible grid. Never step on lava.",
        '''
def collect_all_wood(env, obs, events, tool_calls):
    while not obs.get("done") and _wood_cells(obs):
        action = _step_toward_wood(obs)
        obs, stepped = _emit_primitive(env, obs, events, tool_calls, action, "collect_all_wood")
        if stepped.get("terminated") or obs.get("done"):
            break
    return obs
''',
    ),
}

MAX_WOOD_PROMPT = (
    "Maximize wood collected. Never step on lava. "
    "If collect_all_wood is available, call that skill immediately."
)


def make_speedrunner_harness(skill_names: list[str]) -> str:
    unknown = [name for name in skill_names if name not in SPEEDRUNNER_SKILLS]
    if unknown:
        raise ValueError(f"unknown SpeedRunner skills: {unknown}")
    docs = {name: SPEEDRUNNER_SKILLS[name][0] for name in skill_names}
    impls = "\n".join(SPEEDRUNNER_SKILLS[name][1].strip("\n") for name in skill_names)
    mapping = ", ".join(f'"{name}": {name}' for name in skill_names)
    return (
        _SPEEDRUNNER_PREFIX
        + f"\nPUBLIC_SKILLS = {docs!r}\n\n"
        + impls
        + f"\n\nSKILL_IMPL = {{{mapping}}}\n"
        + _SPEEDRUNNER_RUN
    )


SPEEDRUNNER_HARNESS = make_speedrunner_harness(["collect_all_wood"])

