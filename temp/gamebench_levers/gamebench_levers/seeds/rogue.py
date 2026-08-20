"""Rogue seeds."""

from __future__ import annotations

from gamebench_levers.seeds._speedrunner import build

SEED_POLICY = '''\
def act(obs):
    """Seed policy: rest in place. Scouts nothing beyond the starting room view."""
    return "rest"
'''

SEED_PROMPT = (
    "You are playing Rogue. Explore the level, take gold, and descend the stairs. "
    "Pick one public skill per turn."
)

_SKILLS = '''
SUMMARY_KEYS = (
    "hero", "visible_items", "visible_monsters", "hp", "purse",
    "dungeon_level", "objective", "scout_score", "steps_left",
)
FALLBACK_ACTION = "rest"


def _run(env, obs, action, count):
    ran = []
    for _ in range(count):
        if obs.get("done"):
            break
        stepped = env.step(action)
        obs = stepped["obs"]
        ran.append({"tick": obs.get("tick"), "action": action})
    return obs, ran


def skill_walk_east(env, obs):
    """Walk east across the room; shaped reward pays for newly seen tiles."""
    return _run(env, obs, "right", 6)


def skill_walk_west(env, obs):
    """Walk west across the room."""
    return _run(env, obs, "left", 6)


PUBLIC_SKILLS = {
    "walk_east": skill_walk_east,
    "walk_west": skill_walk_west,
}
'''

SEED_HARNESS = build(_SKILLS)
