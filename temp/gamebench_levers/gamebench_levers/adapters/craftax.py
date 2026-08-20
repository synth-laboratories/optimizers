"""Craftax adapter. Engine: gamebench tasks/craftax-singleplayer gold_python.

World choice matters and is load-bearing. The shipped `policy_dev_small` default
carries `densities: {tree: 0.16, water: 0.05}`; densities *scale* the vanilla
generator, so that world holds 1-3 trees across the whole map and no coal or
iron — a code policy cannot climb a tech tree it cannot reach. The full
`craftax_default` 48x48x9 world has resources but a view radius of 4, so reward
stays flat at 0-1 achievements for hundreds of steps.

This adapter instead uses the purpose-built 9x9 `fixture_room` shape and seeds
its own room variants: every task is guaranteed tree/stone/coal/iron/ladder/cow,
so the achievement ladder is dense and each seed is a distinct layout.
"""

from __future__ import annotations

import random
from typing import Any

from gamebench_levers.adapters.base import ascii_rows, step_result, uniform_obs

GAME = "craftax"
ENV_ID = "gamebench.craftax-singleplayer"
DEFAULT_MAX_STEPS = 200
ROOM = 9

TRAIN_SEEDS = (101, 102, 103, 104, 105, 106)
HELDOUT_SEEDS = (201, 202, 203)

# Ordered early-game ladder these rooms make reachable.
LADDER = (
    "collect_wood",
    "collect_sapling",
    "place_plant",
    "place_table",
    "make_wood_pickaxe",
    "make_wood_sword",
    "collect_stone",
    "place_stone",
    "make_stone_pickaxe",
    "make_stone_sword",
    "collect_coal",
    "collect_iron",
    "eat_cow",
    "defeat_zombie",
    "collect_drink",
    "descend",
)

ACTIONS = (
    "noop", "left", "right", "up", "down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "rest", "descend", "ascend",
)


def _room_task(seed: int, max_steps: int) -> dict[str, Any]:
    """Deterministic 9x9 room variant. Every variant is resource-complete."""
    rng = random.Random(seed)
    cells = [(x, y) for x in range(1, ROOM - 1) for y in range(1, ROOM - 1)]
    rng.shuffle(cells)
    take = iter(cells)
    player = next(take)
    # Floor the whole room first: the fixture world otherwise generates lava, and
    # an entity spawned onto lava trips CraftaxInvariantError at reset.
    tiles: list[dict[str, Any]] = [
        {"pos": [x, y], "kind": "grass"} for x in range(ROOM) for y in range(ROOM)
    ]
    placed: dict[tuple[int, int], str] = {}
    for kind, count in (("tree", 5), ("stone", 3), ("coal", 1), ("iron", 1), ("water", 1), ("ladder_down", 1)):
        for _ in range(count):
            placed[next(take)] = kind
    for pos, kind in placed.items():
        tiles.append({"pos": list(pos), "kind": kind})
    entities = [
        {"kind": "cow", "pos": list(next(take)), "level": 0, "health": 2},
        {"kind": "zombie", "pos": list(next(take)), "level": 0, "health": 4},
    ]
    return {
        "schema": "gamebench.task.craftax.v1",
        "task_id": f"craftax_room_{seed}",
        "world": {
            "use_default": "fixture_room",
            "max_steps": int(max_steps),
            "initial_state": {
                "player": {"pos": list(player), "direction": [1, 0], "level": 0},
                "inventory": {"wood": 0, "stone": 0, "coal": 0, "iron": 0, "pickaxe": 0, "sword": 0},
                "tiles": tiles,
                "entities": entities,
            },
        },
        "rules": {"base": "symbolic_no_homeostasis"},
        "readouts": {"profile": "symbolic_compact"},
    }


class CraftaxSession:
    """One episode. Score is the count of unique achievements unlocked."""

    def __init__(self, seed: int, max_steps: int) -> None:
        from gold_python.engine import CraftaxEngine  # noqa: PLC0415

        self.engine = CraftaxEngine()
        self.engine.reset_from_task(_room_task(seed, max_steps), seed_override=int(seed))
        self.seed = int(seed)
        self.max_steps = int(max_steps)

    @property
    def achievements(self) -> list[str]:
        return sorted(self.engine.private.achievements or [])

    @property
    def outcome_reward(self) -> float:
        return float(len(self.engine.private.achievements or []))

    @property
    def done(self) -> bool:
        return bool(self.engine.private.terminated or self.engine.private.truncated)

    def _near(self, tile: str) -> bool:
        """Crafting requires standing next to a table (and a furnace for iron)."""
        try:
            return bool(self.engine.near_tile({tile}))
        except Exception:  # noqa: BLE001
            return False

    def observation(self) -> dict[str, Any]:
        readout = self.engine.symbolic_readout()
        obs = (readout.get("public") or {}).get("observation") or {}
        player = obs.get("player") or {}
        return uniform_obs(
            game=GAME,
            tick=int(self.engine.private.step_index),
            done=self.done,
            ascii_rows=ascii_rows(readout.get("ascii")),
            legal_actions=list(ACTIONS),
            score=self.outcome_reward,
            achievements=self.achievements,
            text=readout.get("observation_text"),
            state={
                "player": player,
                "front_tile": player.get("front_tile"),
                "local_map": obs.get("local_map") or [],
                "inventory": obs.get("inventory") or {},
                "max_steps": self.max_steps,
                "steps_left": max(0, self.max_steps - int(self.engine.private.step_index)),
                "health": (obs.get("inventory") or {}).get("health"),
                "near_crafting_table": self._near("crafting_table"),
                "near_furnace": self._near("furnace"),
                "ladder": list(LADDER),
                "remaining_ladder": [name for name in LADDER if name not in set(self.achievements)],
            },
        )

    def step(self, action: Any) -> dict[str, Any]:
        name = str(action or "noop").strip().lower()
        before = self.outcome_reward
        self.engine.step(name)
        after = self.outcome_reward
        return step_result(
            obs=self.observation(),
            reward=after - before,
            terminated=bool(self.engine.private.terminated),
            truncated=bool(self.engine.private.truncated),
            info={
                "outcome_reward": after,
                "achievements": self.achievements,
                "done_reason": self.engine.private.done_reason,
                "invalid_action_count": int(self.engine.private.invalid_action_count or 0),
            },
        )


def make_session(seed: int, max_steps: int = DEFAULT_MAX_STEPS) -> CraftaxSession:
    return CraftaxSession(seed, max_steps)


def spec() -> dict[str, Any]:
    return {
        "game": GAME,
        "env_id": ENV_ID,
        "action_space": list(ACTIONS),
        "action_type": "string",
        "max_horizon": DEFAULT_MAX_STEPS,
        "achievements": list(LADDER),
        "objective": "Unlock as many Craftax achievements as possible. Score is the count of unique achievements.",
        "train_seeds": list(TRAIN_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "observation_notes": (
            "9x9 room; every seed holds 5 tree, 3 stone, 1 coal, 1 iron, 1 water, 1 ladder_down, "
            "a cow and a zombie. obs['state']['local_map'] is an egocentric view (P player, T tree, "
            "S stone, c coal, i iron, C cow, Z zombie, > ladder_down, ~ or _ water, o out of bounds). "
            "'do' acts on obs['state']['front_tile'] -- move toward a resource until it is the front "
            "tile, then 'do'. legal_actions is the full vocabulary, NOT a legality filter. Recipes: "
            "place_table costs 2 wood and needs an empty walkable front tile; make_wood_pickaxe and "
            "make_wood_sword cost 1 wood each; stone tools cost 1 wood + 1 stone. Every make_* "
            "requires standing next to a crafting table -- obs['state']['near_crafting_table'] says "
            "whether you are. So gather wood first, place the table, then craft without walking away. "
            "Mining stone/coal needs pickaxe>=1; iron needs pickaxe>=2. Tools do not stack past their "
            "tier. descend needs to face a ladder_down ('>')."
        ),
    }
