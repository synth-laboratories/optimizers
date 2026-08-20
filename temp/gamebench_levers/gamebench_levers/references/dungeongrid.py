"""Reference DungeonGrid policy: path around walls, act on what is adjacent.

`legal['directions']` always lists all four compass points -- it is not filtered by
passability -- so a policy that trusts it walks into walls and burns the attempt
budget on rejected moves. This one paths on the ascii map instead.
"""

from collections import deque

DELTAS = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
TARGET_GLYPHS = ("I", "C", "D")


def _grid(obs):
    return [list(row) for row in (obs.get("ascii") or [])]


def _passable(grid, x, y):
    if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[y]):
        return False
    return grid[y][x] != "#"


def _first_step(grid, start, goals):
    """BFS from start; return the direction name of the first step toward a goal."""
    if not goals:
        return None
    seen = {start}
    queue = deque((start, name) for name in ())
    for name, (dx, dy) in DELTAS.items():
        nxt = (start[0] + dx, start[1] + dy)
        if _passable(grid, *nxt):
            seen.add(nxt)
            queue.append((nxt, name))
    while queue:
        (pos, first) = queue.popleft()
        if pos in goals:
            return first
        for dx, dy in DELTAS.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in seen and _passable(grid, *nxt):
                seen.add(nxt)
                queue.append((nxt, first))
    return None


def act(obs):
    state = obs["state"]
    legal = state.get("legal") or {}
    if int(legal.get("ap") or 0) <= 0:
        return "end_turn"
    # Never repeat an action the engine just refused; end the turn to refresh AP
    # and re-plan from the new state.
    if state.get("last_applied") is False:
        return "end_turn"

    chests = legal.get("adjacent_chests") or []
    if chests:
        return f"interact:{chests[0]}"
    monsters = legal.get("adjacent_monsters") or []
    if monsters:
        return f"attack_melee:{monsters[0]}"
    doors = legal.get("adjacent_doors") or []
    if doors:
        return f"open_door:{doors[0]}"

    spells = legal.get("spells") or []
    if spells and int(state.get("spells_cast") or 0) < 2:
        return f"cast:{spells[0]}@self"

    hero = (state.get("hero") or {}).get("pos") or {}
    start = (int(hero.get("x", 0)), int(hero.get("y", 0)))
    grid = _grid(obs)
    goals = {
        (x, y)
        for y, row in enumerate(grid)
        for x, cell in enumerate(row)
        if cell in TARGET_GLYPHS
    }
    step = _first_step(grid, start, goals)
    if step:
        return f"move:{step}"
    return "end_turn"
