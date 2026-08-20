"""Sokoban adapter. Engine: gamebench tasks/sokoban-singleplayer gold_python."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gamebench_levers.adapters import task_dir
from gamebench_levers.adapters.base import ascii_rows, step_result, uniform_obs

GAME = "sokoban"
ENV_ID = "gamebench.sokoban-singleplayer"
ACTIONS = ("up", "down", "left", "right")
LEVEL_BANK = "curriculum_medium"
DEFAULT_MAX_STEPS = 120

# curriculum_medium holds 10 levels (1-2 boxes, optimal 5-8); index = seed % 10.
# Easy/ultra_easy solve in 1-2 moves and leave GEPA no headroom.
TRAIN_SEEDS = (0, 1, 2, 3, 4, 5, 6)
HELDOUT_SEEDS = (7, 8, 9)

WALL, FLOOR, TARGET, BOX_ON_TARGET, BOX, PLAYER, PLAYER_ON_TARGET = 0, 1, 2, 3, 4, 5, 6
MILESTONES = ("first_push", "first_box_on_target", "level_complete")


def _task_template() -> dict[str, Any]:
    return json.loads((task_dir(GAME) / "tasks" / "policy_dev_template.json").read_text(encoding="utf-8"))


class SokobanSession:
    """One episode. Score is box-on-target fraction, 1.0 when solved."""

    def __init__(self, seed: int, max_steps: int) -> None:
        from gold_python.engine import SokobanEngine  # noqa: PLC0415
        from task_resolve import resolve_task  # noqa: PLC0415

        task = _task_template()
        task.setdefault("map", {})["use_default"] = LEVEL_BANK
        task["map"]["seed"] = int(seed)
        task.setdefault("rules", {}).setdefault("overrides", {})["max_steps"] = int(max_steps)
        self.engine = SokobanEngine()
        self.engine.reset(resolve_task(task, seed_override=int(seed)))
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.unlocked: set[str] = set()
        self._num_boxes = len(self.engine.boxes)
        self._pushes = 0

    # -- scoring ---------------------------------------------------------
    @property
    def _on_target(self) -> int:
        return sum(1 for box in self.engine.boxes if box in self.engine.goals)

    @property
    def solved(self) -> bool:
        return self._num_boxes > 0 and self._on_target == self._num_boxes

    @property
    def outcome_reward(self) -> float:
        if self.solved:
            return 1.0
        if self._num_boxes <= 0:
            return 0.0
        return round(0.85 * (self._on_target / self._num_boxes), 6)

    @property
    def done(self) -> bool:
        return bool(self.engine.private.terminated or self.engine.private.truncated)

    # -- surface ---------------------------------------------------------
    def _refresh_milestones(self) -> None:
        if self._pushes:
            self.unlocked.add("first_push")
        if self._on_target:
            self.unlocked.add("first_box_on_target")
        if self.solved:
            self.unlocked.add("level_complete")

    def observation(self) -> dict[str, Any]:
        readout = self.engine.symbolic_readout()
        public = readout.get("public") or {}
        room = public.get("room_state") or []
        goals = [[r, c] for r, row in enumerate(room) for c, cell in enumerate(row) if cell in (TARGET, BOX_ON_TARGET, PLAYER_ON_TARGET)]
        walls = [[r, c] for r, row in enumerate(room) for c, cell in enumerate(row) if cell == WALL]
        self._refresh_milestones()
        return uniform_obs(
            game=GAME,
            tick=int(self.engine.private.step_index),
            done=self.done,
            ascii_rows=ascii_rows(readout.get("ascii")),
            legal_actions=list(self.engine.valid_actions()) or list(ACTIONS),
            score=self.outcome_reward,
            achievements=sorted(self.unlocked),
            state={
                "player": list(public.get("player") or []),
                "boxes": [list(box) for box in (public.get("boxes") or [])],
                "goals": goals,
                "walls": walls,
                "room_state": room,
                "boxes_on_target": self._on_target,
                "num_boxes": self._num_boxes,
                "solved": self.solved,
                "max_steps": self.max_steps,
                "steps_left": max(0, self.max_steps - int(self.engine.private.step_index)),
            },
        )

    def step(self, action: Any) -> dict[str, Any]:
        before_on_target = self._on_target
        before_boxes = set(self.engine.boxes)
        name = str(action or "").strip().lower()
        if name not in ACTIONS:
            name = "up"
        before = self.outcome_reward
        self.engine.step(name)
        if set(self.engine.boxes) != before_boxes:
            self._pushes += 1
        self._refresh_milestones()
        after = self.outcome_reward
        return step_result(
            obs=self.observation(),
            reward=after - before,
            terminated=bool(self.engine.private.terminated),
            truncated=bool(self.engine.private.truncated),
            info={
                "outcome_reward": self.outcome_reward,
                "achievements": sorted(self.unlocked),
                "boxes_on_target": self._on_target,
                "num_boxes": self._num_boxes,
                "solved": self.solved,
                "pushed": self._on_target != before_on_target,
            },
        )


def make_session(seed: int, max_steps: int = DEFAULT_MAX_STEPS) -> SokobanSession:
    return SokobanSession(seed, max_steps)


def spec() -> dict[str, Any]:
    return {
        "game": GAME,
        "env_id": ENV_ID,
        "action_space": list(ACTIONS),
        "action_type": "string",
        "max_horizon": DEFAULT_MAX_STEPS,
        "achievements": list(MILESTONES),
        "objective": "Push every box ($) onto a goal (.). Score is boxes-on-target fraction; 1.0 when solved.",
        "train_seeds": list(TRAIN_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "observation_notes": (
            "obs['state'] has player [row,col], boxes [[row,col]..], goals, walls, "
            "room_state ints (0 wall,1 floor,2 goal,3 box-on-goal,4 box,5 player,6 player-on-goal). "
            "Actions move the player; a box in front is pushed if the cell beyond is free."
        ),
    }
