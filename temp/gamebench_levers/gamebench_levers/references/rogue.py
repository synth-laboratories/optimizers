"""Reference Rogue policy: sweep the room so shaped reward pays for new tiles."""

_MOVES = ("right", "down", "left", "up")


def act(obs):
    state = obs["state"]
    hero = state.get("hero") or [0, 0]
    items = state.get("visible_items") or {}
    rows = obs.get("ascii") or []

    # Stand on an item, then pick it up.
    key = f"{hero[0]},{hero[1]}"
    if key in items:
        return "pickup"
    if items:
        target = min(
            (tuple(int(part) for part in pos.split(",")) for pos in items),
            key=lambda p: abs(p[0] - hero[0]) + abs(p[1] - hero[1]),
        )
        if target[0] != hero[0]:
            return "down" if target[0] > hero[0] else "up"
        if target[1] != hero[1]:
            return "right" if target[1] > hero[1] else "left"

    # Stairs end the level well.
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell == "%":
                if (r, c) == (hero[0], hero[1]):
                    return "descend"
                if r != hero[0]:
                    return "down" if r > hero[0] else "up"
                if c != hero[1]:
                    return "right" if c > hero[1] else "left"

    return _MOVES[(obs.get("tick") or 0) // 4 % len(_MOVES)]
