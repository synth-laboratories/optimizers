"""Craftax seeds."""

from __future__ import annotations

from gamebench_levers.seeds._speedrunner import build

SEED_POLICY = '''\
def act(obs):
    """Seed policy: stand still. Unlocks nothing."""
    return "noop"
'''

SEED_PROMPT = (
    "You are playing Craftax in a 9x9 room. Unlock as many achievements as you can. "
    "Pick one public skill per turn."
)

_SKILLS = '''
SUMMARY_KEYS = (
    "front_tile", "local_map", "inventory", "near_crafting_table",
    "remaining_ladder", "steps_left", "health",
)
FALLBACK_ACTION = "noop"


def _nearest(local_map, glyph):
    """Offset of the closest `glyph` in the egocentric map, or None."""
    radius = len(local_map) // 2
    best = None
    for y, row in enumerate(local_map):
        for x, cell in enumerate(row):
            if cell == glyph:
                dist = abs(x - radius) + abs(y - radius)
                if dist and (best is None or dist < best[0]):
                    best = (dist, x - radius, y - radius)
    return best


def _toward(env, obs, glyph, limit=10):
    """Move until `glyph` is the tile directly in front."""
    ran = []
    for _ in range(limit):
        state = obs.get("state") or {}
        if obs.get("done"):
            break
        best = _nearest(state.get("local_map") or [], glyph)
        if best is None:
            break
        _, dx, dy = best
        action = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
        stepped = env.step(action)
        obs = stepped["obs"]
        ran.append({"tick": obs.get("tick"), "action": action})
        if ((obs.get("state") or {}).get("front_tile")) == GLYPH_TILE.get(glyph):
            break
    return obs, ran


GLYPH_TILE = {"T": "tree", "S": "stone", "c": "coal", "i": "iron", ">": "ladder_down", "C": "cow"}


def skill_gather_wood(env, obs):
    """Walk to a tree and chop it a few times."""
    obs, ran = _toward(env, obs, "T")
    for _ in range(3):
        if obs.get("done"):
            break
        stepped = env.step("do")
        obs = stepped["obs"]
        ran.append({"tick": obs.get("tick"), "action": "do"})
    return obs, ran


def skill_place_table(env, obs):
    """Place a crafting table on the tile in front."""
    stepped = env.step("place_table")
    return stepped["obs"], [{"tick": stepped["obs"].get("tick"), "action": "place_table"}]


PUBLIC_SKILLS = {
    "gather_wood": skill_gather_wood,
    "place_table": skill_place_table,
}
'''

SEED_HARNESS = build(_SKILLS)
