"""Sokoban seeds."""

from __future__ import annotations

from gamebench_levers.seeds._speedrunner import build

SEED_POLICY = '''\
def act(obs):
    """Seed policy: always push up. Solves nothing on the medium bank."""
    return "up"
'''

SEED_PROMPT = (
    "You are playing Sokoban. Push every box ($) onto a goal (.). "
    "Pick one public skill per turn."
)

_SKILLS = '''
SUMMARY_KEYS = ("player", "boxes", "goals", "boxes_on_target", "num_boxes", "steps_left")
FALLBACK_ACTION = "up"
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _walk(env, obs, target, limit=12):
    """Step the player toward `target` by greedy Manhattan descent."""
    ran = []
    for _ in range(limit):
        state = obs.get("state") or {}
        pr, pc = state.get("player") or [0, 0]
        tr, tc = target
        if (pr, pc) == (tr, tc) or obs.get("done"):
            break
        action = None
        if pr != tr:
            action = "down" if tr > pr else "up"
        elif pc != tc:
            action = "right" if tc > pc else "left"
        if action is None:
            break
        stepped = env.step(action)
        obs = stepped["obs"]
        ran.append({"tick": obs.get("tick"), "action": action})
    return obs, ran


def skill_approach_box(env, obs):
    """Walk next to the first box that is not yet on a goal."""
    state = obs.get("state") or {}
    goals = {tuple(g) for g in (state.get("goals") or [])}
    boxes = [tuple(b) for b in (state.get("boxes") or []) if tuple(b) not in goals]
    if not boxes:
        return obs, []
    return _walk(env, obs, boxes[0])


def skill_push_up(env, obs):
    """Push four times upward."""
    ran = []
    for _ in range(4):
        if obs.get("done"):
            break
        stepped = env.step("up")
        obs = stepped["obs"]
        ran.append({"tick": obs.get("tick"), "action": "up"})
    return obs, ran


PUBLIC_SKILLS = {
    "approach_box": skill_approach_box,
    "push_up": skill_push_up,
}
'''

SEED_HARNESS = build(_SKILLS)
