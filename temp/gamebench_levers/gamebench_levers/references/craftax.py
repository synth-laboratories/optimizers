"""Reference Craftax policy: climb the early tech tree."""

GLYPH_TILE = {"T": "tree", "S": "stone", "c": "coal", "i": "iron", ">": "ladder_down", "C": "cow", "Z": "zombie"}


def _nearest(local_map, glyph):
    radius = len(local_map) // 2
    best = None
    for y, row in enumerate(local_map):
        for x, cell in enumerate(row):
            if cell == glyph:
                dist = abs(x - radius) + abs(y - radius)
                if dist and (best is None or dist < best[0]):
                    best = (dist, x - radius, y - radius)
    return best


def _toward(best):
    _, dx, dy = best
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def act(obs):
    state = obs["state"]
    inv = state["inventory"]
    front = state["front_tile"]
    local = state["local_map"]
    done = set(obs["achievements"])
    near_table = state.get("near_crafting_table")

    if front == "tree" and inv["wood"] < 6:
        return "do"
    if front == "stone" and inv["pickaxe"] >= 1 and inv["stone"] < 4:
        return "do"
    if front == "coal" and inv["pickaxe"] >= 1:
        return "do"
    if front == "iron" and inv["pickaxe"] >= 2:
        return "do"
    if front == "cow":
        return "do"
    if front == "zombie" and inv["sword"] >= 1:
        return "do"
    if front == "water" and "collect_drink" not in done:
        return "do"
    if inv.get("sapling", 0) > 0 and "place_plant" not in done and front == "grass":
        return "place_plant"
    if inv["wood"] >= 5 and "place_table" not in done and front == "grass":
        return "place_table"
    if near_table:
        if inv["wood"] >= 1 and inv["pickaxe"] < 1:
            return "make_wood_pickaxe"
        if inv["wood"] >= 1 and inv["sword"] < 1:
            return "make_wood_sword"
        if inv["wood"] >= 1 and inv["stone"] >= 1 and inv["pickaxe"] < 2:
            return "make_stone_pickaxe"
        if inv["wood"] >= 1 and inv["stone"] >= 1 and inv["sword"] < 2:
            return "make_stone_sword"
    if inv["stone"] >= 1 and "place_stone" not in done and front == "grass":
        return "place_stone"
    if front == "ladder_down" and inv["pickaxe"] >= 1:
        return "descend"

    wants = []
    if inv["wood"] < 6:
        wants.append("T")
    if inv["pickaxe"] >= 1 and inv["stone"] < 4:
        wants.append("S")
    if inv["pickaxe"] >= 1 and "collect_coal" not in done:
        wants.append("c")
    if inv["pickaxe"] >= 2 and "collect_iron" not in done:
        wants.append("i")
    if "eat_cow" not in done:
        wants.append("C")
    if inv["sword"] >= 1 and "defeat_zombie" not in done:
        wants.append("Z")
    wants.append(">")
    for glyph in wants:
        found = _nearest(local, glyph)
        if found:
            return _toward(found)
    return "right" if obs["tick"] % 2 else "down"
