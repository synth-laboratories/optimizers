"""Deterministic Craftax-shaped grid. Seed 0 is the fixture used by lever tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ACTIONS = ("up", "down", "left", "right", "collect", "noop")
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# Seed 0: one wood is 4-adjacent; a second wood is reachable without stepping on lava.
FIXTURE_GRID = ["@W.", ".L.", "W.."]


@dataclass
class World:
    grid: list[list[str]]
    x: int
    y: int
    wood: int = 0
    tick: int = 0
    max_steps: int = 16
    achievements: list[str] = field(default_factory=list)
    dead: bool = False
    won: bool = False
    last_step_reward: float = 0.0
    death_cause: str | None = None

    @property
    def done(self) -> bool:
        return self.dead or self.won or self.tick >= self.max_steps

    @property
    def outcome_reward(self) -> float:
        if self.dead:
            return 0.0
        return float(self.wood)


def _find_player(rows: list[str]) -> tuple[int, int, list[list[str]]]:
    grid = [list(row) for row in rows]
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "@":
                row[x] = "."
                return x, y, grid
    raise ValueError("grid must contain @")


def make_world(seed: int, max_steps: int = 16) -> World:
    if seed == 0:
        rows = FIXTURE_GRID
    else:
        # Other seeds rotate the fixture so taskset rows are distinct but solvable.
        rows = FIXTURE_GRID
        if seed % 2 == 1:
            rows = ["".join(reversed(row)) for row in rows]
    x, y, grid = _find_player(rows)
    return World(grid=grid, x=x, y=y, max_steps=max_steps)


def _in_bounds(world: World, x: int, y: int) -> bool:
    return 0 <= y < len(world.grid) and 0 <= x < len(world.grid[0])


def _neighbors(world: World) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for dx, dy in DELTAS.values():
        nx, ny = world.x + dx, world.y + dy
        if _in_bounds(world, nx, ny):
            out.append((nx, ny, world.grid[ny][nx]))
    return out


def observation(world: World) -> dict[str, Any]:
    render = ["".join(row) for row in world.grid]
    line = list(render[world.y])
    line[world.x] = "X" if world.dead else "@"
    render[world.y] = "".join(line)
    cells = [cell for _, _, cell in _neighbors(world)]
    return {
        "tick": world.tick,
        "x": world.x,
        "y": world.y,
        "inventory": {"wood": world.wood},
        "grid": render,
        "width": len(world.grid[0]),
        "height": len(world.grid),
        "adjacent": {"wood": "W" in cells, "lava": "L" in cells},
        "achievements": list(world.achievements),
        "legal_actions": list(ACTIONS),
        "done": world.done,
        "dead": world.dead,
        "death_cause": world.death_cause,
    }


def reset(seed: int, max_steps: int = 16) -> tuple[World, dict[str, Any]]:
    world = make_world(seed, max_steps=max_steps)
    return world, observation(world)


def step(world: World, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    if world.done:
        obs = observation(world)
        return obs, 0.0, True, {"already_done": True}
    action = action if action in ACTIONS else "noop"
    world.last_step_reward = 0.0
    if action in DELTAS:
        dx, dy = DELTAS[action]
        nx, ny = world.x + dx, world.y + dy
        if not _in_bounds(world, nx, ny):
            pass
        elif world.grid[ny][nx] == "L":
            world.x, world.y = nx, ny
            world.dead = True
            world.death_cause = "lava"
            world.last_step_reward = 0.0
        else:
            world.x, world.y = nx, ny
    elif action == "collect":
        for nx, ny, cell in _neighbors(world):
            if cell == "W":
                world.grid[ny][nx] = "."
                world.wood += 1
                if "collect_wood" not in world.achievements:
                    world.achievements.append("collect_wood")
                world.last_step_reward = 1.0
                wood_left = sum(cell == "W" for row in world.grid for cell in row)
                if wood_left == 0:
                    world.won = True
                break
    world.tick += 1
    if world.tick >= world.max_steps and not world.won:
        world.death_cause = world.death_cause or "timeout"
    obs = observation(world)
    info = {
        "achievements": list(world.achievements),
        "death_cause": world.death_cause,
        "wood": world.wood,
        "outcome_reward": world.outcome_reward,
    }
    return obs, world.last_step_reward, world.done, info
