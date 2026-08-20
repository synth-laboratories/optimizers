def act(obs):
    """Walk to the box, then push it toward the nearest goal."""
    st = obs["state"]
    pr, pc = st["player"]
    goals = {tuple(g) for g in st["goals"]}
    boxes = [tuple(b) for b in st["boxes"]]
    walls = {tuple(w) for w in st["walls"]}
    todo = [b for b in boxes if b not in goals]
    if not todo:
        return "up"
    box = min(todo, key=lambda b: abs(b[0] - pr) + abs(b[1] - pc))
    goal = min(goals, key=lambda g: abs(g[0] - box[0]) + abs(g[1] - box[1]))
    dr, dc = goal[0] - box[0], goal[1] - box[1]
    if dr and abs(dr) >= abs(dc):
        stand = (box[0] - (1 if dr > 0 else -1), box[1])
        push = "down" if dr > 0 else "up"
    elif dc:
        stand = (box[0], box[1] - (1 if dc > 0 else -1))
        push = "right" if dc > 0 else "left"
    else:
        return "up"
    if (pr, pc) == stand:
        return push
    if stand in walls:
        return "up"
    if pr != stand[0]:
        return "down" if stand[0] > pr else "up"
    if pc != stand[1]:
        return "right" if stand[1] > pc else "left"
    return "up"
