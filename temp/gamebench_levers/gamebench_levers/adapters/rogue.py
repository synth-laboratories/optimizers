"""Rogue adapter. Engine: gamebench tasks/rogue-singleplayer gold_python.

Tasks come from the shipped `policy_dev_v2` sweep, which carries eight inline
rooms (grid + seed + max_steps + objective). Reward is the engine's own
`synth_shaped_reward`: dense, and it pays for scouting tiles, picking up items,
killing monsters and descending -- so a code policy has a gradient from step one.
"""

from __future__ import annotations

import json
from typing import Any

from gamebench_levers.adapters import task_dir
from gamebench_levers.adapters.base import ascii_rows, step_result, uniform_obs

GAME = "rogue"
ENV_ID = "gamebench.rogue-singleplayer"
SUITE = "policy_dev_v2"
DEFAULT_MAX_STEPS = 80

# Friendly names -> native rogue command characters.
MOVES = {
    "left": "h", "down": "j", "up": "k", "right": "l",
    "upleft": "y", "upright": "u", "downleft": "b", "downright": "n",
}
VERBS = {"descend": ">", "pickup": ",", "search": "s", "rest": ".", "noop": "."}
ALIASES = {**MOVES, **VERBS}
ACTIONS = tuple(ALIASES) + tuple(ALIASES.values())


def _suite() -> dict[str, Any]:
    path = task_dir(GAME) / "defaults" / "policy_sweep" / f"{SUITE}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _tasks() -> list[dict[str, Any]]:
    return list(_suite().get("tasks") or [])


def _split_seeds() -> tuple[tuple[int, ...], tuple[int, ...]]:
    seeds = [int(task["seed"]) for task in _tasks()]
    return tuple(seeds[:5]), tuple(seeds[5:])


TRAIN_SEEDS, HELDOUT_SEEDS = _split_seeds()


def _task_for(seed: int, max_steps: int | None) -> dict[str, Any]:
    entries = _tasks()
    match = next((task for task in entries if int(task["seed"]) == int(seed)), None)
    if match is None:
        match = entries[int(seed) % len(entries)]
    steps = int(max_steps or match.get("max_steps") or DEFAULT_MAX_STEPS)
    task: dict[str, Any] = {
        "schema": "gamebench.task.rogue.v1",
        "task_id": str(match["task_id"]),
        "seed": int(match["seed"]),
        "grid": list(match["grid"]),
        "rules": {"base": "modern_rogue_core", "overrides": {"max_steps": steps}},
        "objective": str(match.get("objective") or "descend"),
        "readouts": {"symbolic": "ascii", "visual": False},
        "checkpoint_every_n_steps": 1,
    }
    for key in ("monsters", "traps", "level_objects"):
        if key in match:
            task[key] = match[key]
    return task


class RogueSession:
    """One episode. Score is the engine's synth_shaped_reward."""

    def __init__(self, seed: int, max_steps: int | None) -> None:
        from gold_python.engine import RogueEngine  # noqa: PLC0415
        from task_resolve import resolve_task  # noqa: PLC0415

        task = _task_for(seed, max_steps)
        self.task_id = task["task_id"]
        self.objective = task["objective"]
        self.max_steps = int(task["rules"]["overrides"]["max_steps"])
        self.engine = RogueEngine()
        self.engine.reset(resolve_task(task, seed_override=int(task["seed"])))
        self.seed = int(task["seed"])

    @property
    def outcome_reward(self) -> float:
        return float(getattr(self.engine.private, "synth_shaped_reward", 0.0) or 0.0)

    @property
    def done(self) -> bool:
        return bool(self.engine.private.terminated or self.engine.private.truncated)

    def observation(self) -> dict[str, Any]:
        readout = self.engine.symbolic_readout()
        public = readout.get("public") or {}
        private = readout.get("private") or {}
        progress = readout.get("progress_metrics") or {}
        return uniform_obs(
            game=GAME,
            tick=int(private.get("step_index") or 0),
            done=self.done,
            ascii_rows=ascii_rows(public.get("terrain") or readout.get("ascii")),
            legal_actions=list(MOVES) + list(VERBS),
            score=self.outcome_reward,
            achievements=list(progress.get("achievement_names") or [])[:40],
            state={
                "hero": list(public.get("hero") or []),
                "visible_items": public.get("visible_items") or {},
                "visible_monsters": public.get("visible_monsters") or {},
                "hp": private.get("hp"),
                "max_hp": private.get("max_hp"),
                "purse": private.get("purse"),
                "dungeon_level": private.get("dungeon_level"),
                "objective": self.objective,
                "scout_score": progress.get("scout_score"),
                "shaped_reward": self.outcome_reward,
                "max_steps": self.max_steps,
                "steps_left": max(0, self.max_steps - int(private.get("step_index") or 0)),
                "task_id": self.task_id,
            },
        )

    def step(self, action: Any) -> dict[str, Any]:
        raw = str(action or ".").strip()
        command = ALIASES.get(raw.lower(), raw)
        if command not in set(ALIASES.values()):
            command = "."
        before = self.outcome_reward
        self.engine.step(command)
        after = self.outcome_reward
        private = self.engine.private
        return step_result(
            obs=self.observation(),
            reward=after - before,
            terminated=bool(private.terminated),
            truncated=bool(private.truncated),
            info={
                "outcome_reward": after,
                "achievements": list(getattr(private, "acquired_item_classes", []) or []),
                "terminal_reason": private.terminal_reason,
                "dungeon_level": private.dungeon_level,
                "purse": private.purse,
                "scout_score": getattr(private, "scout_score", 0),
            },
        )


def make_session(seed: int, max_steps: int | None = None) -> RogueSession:
    return RogueSession(seed, max_steps)


def spec() -> dict[str, Any]:
    return {
        "game": GAME,
        "env_id": ENV_ID,
        "action_space": list(MOVES) + list(VERBS),
        "action_type": "string",
        "max_horizon": DEFAULT_MAX_STEPS,
        "achievements": ["scout.tile_seen", "item.pickup", "monster.kill", "level.descend"],
        "objective": "Explore, collect gold, and descend. Score is the engine's synth_shaped_reward.",
        "train_seeds": list(TRAIN_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "observation_notes": (
            "obs['ascii'] is the known terrain ('|' and '-' walls, '.' floor, '%' stairs down, "
            "'*' gold, ':' food, '^' trap). obs['state']['hero'] is [row, col]. "
            "visible_items / visible_monsters are keyed 'row,col'. Actions: left/right/up/down plus "
            "the four diagonals, and pickup (step onto an item first), search, rest, descend "
            "(stand on '%'). Shaped reward pays mostly for newly seen tiles, so exploring the room "
            "beats standing still; gold and descending pay extra."
        ),
    }
