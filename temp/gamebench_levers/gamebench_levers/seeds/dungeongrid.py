"""DungeonGrid seeds."""

from __future__ import annotations

from gamebench_levers.seeds._speedrunner import build

SEED_POLICY = '''\
def act(obs):
    """Seed policy: pass every turn. Advances no quest."""
    return "end_turn"
'''

SEED_PROMPT = (
    "You are playing DungeonGrid. Advance the quest: take gold, cast spells, "
    "unlock achievements. Pick one public skill per turn."
)

_SKILLS = '''
SUMMARY_KEYS = (
    "hero", "legal", "monsters", "doors", "chests", "objective",
    "gold_collected", "spells_cast", "invalid_actions", "steps_left",
)
FALLBACK_ACTION = "end_turn"


def _act(env, obs, action):
    stepped = env.step(action)
    return stepped["obs"], [{"tick": stepped["obs"].get("tick"), "action": action}]


def _ap(obs):
    return int(((obs.get("state") or {}).get("legal") or {}).get("ap") or 0)


def skill_step_east(env, obs):
    """Spend the turn's AP walking east, then end the turn to refresh.

    A refused move does not consume AP, so this stops on the first rejection --
    otherwise one skill call spins until the episode's attempt budget is gone.
    """
    ran = []
    while _ap(obs) > 0 and not obs.get("done"):
        obs, one = _act(env, obs, "move:east")
        ran.extend(one)
        if (obs.get("state") or {}).get("last_applied") is False:
            break
    obs, one = _act(env, obs, "end_turn")
    return obs, ran + one


def skill_guard(env, obs):
    """Guard, then end the turn."""
    obs, ran = _act(env, obs, "guard")
    obs, one = _act(env, obs, "end_turn")
    return obs, ran + one


PUBLIC_SKILLS = {
    "step_east": skill_step_east,
    "guard": skill_guard,
}
'''

SEED_HARNESS = build(_SKILLS)
