"""DungeonGrid adapter. Engine: gamebench tasks/dungeongrid-singleplayer gold_python.

Reward deliberately departs from the shipped sweep composite. That composite adds
`armor` (which counts remaining HP) and `step_bonus` (max_actions - steps), so a
policy that does nothing keeps full HP, spends no steps, and scores well. Search
against it rewards inaction. This adapter scores progress only -- gold, unlocked
achievements, spells cast and the scenario's own reward, minus invalid actions.
"""

from __future__ import annotations

import json
from typing import Any

from gamebench_levers.adapters import task_dir
from gamebench_levers.adapters.base import ascii_rows, step_result, uniform_obs

GAME = "dungeongrid"
ENV_ID = "gamebench.dungeongrid-singleplayer"
DEFAULT_MAX_STEPS = 60

WEIGHTS = {"gold": 2.0, "achievements": 1.5, "spells": 2.0, "engine_reward": 1.0, "invalid_penalty": 0.1}

DIRECTIONS = ("north", "south", "east", "west")
ACTION_KINDS = (
    "move", "open_door", "cast", "search_traps", "inspect_tile",
    "interact", "attack_melee", "use_item", "give_item", "guard", "end_turn", "message",
)


def _scenarios(prefix: str) -> list[str]:
    root = task_dir(GAME) / "defaults" / "scenarios"
    return sorted(path.stem for path in root.glob(f"{prefix}*.json"))


TRAIN_SCENARIOS = tuple(_scenarios("sp_train_"))
HELDOUT_SCENARIOS = tuple(_scenarios("sp_heldout_"))
TRAIN_SEEDS = tuple(range(len(TRAIN_SCENARIOS)))
HELDOUT_SEEDS = tuple(range(len(HELDOUT_SCENARIOS)))


def _scenario(seed: int, split: str) -> dict[str, Any]:
    names = TRAIN_SCENARIOS if split == "train" else HELDOUT_SCENARIOS
    name = names[int(seed) % len(names)]
    path = task_dir(GAME) / "defaults" / "scenarios" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_action(action: Any) -> dict[str, Any]:
    """Accept a native dict, or a compact string: 'move:east', 'cast:ward_circle@self'."""
    if isinstance(action, dict):
        return dict(action)
    raw = str(action or "end_turn").strip()
    if not raw:
        return {"type": "end_turn"}
    kind, _, rest = raw.partition(":")
    kind = kind.strip()
    rest = rest.strip()
    if kind not in ACTION_KINDS:
        return {"type": "end_turn"}
    if kind == "move":
        return {"type": "move", "direction": rest or "east"}
    if kind == "cast":
        spell, _, target = rest.partition("@")
        return {"type": "cast", "target": target or "self", "payload": {"spell": spell}}
    if kind == "message":
        return {"type": "message", "target": "party", "payload": {"text": rest or "ping"}}
    if kind in {"guard", "end_turn", "search_traps", "inspect_tile"}:
        return {"type": kind}
    return {"type": kind, "target": rest}


class DungeonGridSession:
    """One episode. Score rewards progress, not survival-by-inaction."""

    def __init__(self, seed: int, max_steps: int, split: str = "train") -> None:
        from gold_python import DungeonGridSession as Engine  # noqa: PLC0415

        scenario = _scenario(seed, split)
        scenario["max_steps"] = int(max_steps)
        self.scenario_id = str(scenario.get("scenario_id") or "")
        self.session = Engine(scenario)
        self.seed = int(seed)
        self.split = split
        self.max_steps = int(max_steps)
        self.invalid = 0
        # The engine advances step_index only on an applied, non-end_turn action, so
        # rejected actions and bare end_turns would loop forever. Bound the episode
        # by attempts as well.
        self.attempts = 0
        self.attempt_budget = int(max_steps) * 4
        self.last_action: Any = None
        self.last_applied: bool | None = None
        self.last_reject_reason: str | None = None

    @property
    def outcome_reward(self) -> float:
        session = self.session
        gold = float(getattr(session, "gold_collected", 0) or 0)
        spells = float(getattr(session, "spells_cast", 0) or 0)
        achievements = float(len(session.achievements or []))
        score = (
            WEIGHTS["gold"] * gold
            + WEIGHTS["achievements"] * achievements
            + WEIGHTS["spells"] * spells
            + WEIGHTS["engine_reward"] * float(session.total_reward or 0.0)
            - WEIGHTS["invalid_penalty"] * float(self.invalid)
        )
        return round(score, 6)

    @property
    def done(self) -> bool:
        return bool(
            self.session.done
            or self.session.step_index >= self.max_steps
            or self.attempts >= self.attempt_budget
        )

    def _reject_reason(self) -> str | None:
        """The engine records why an action bounced only in its event log."""
        for event in reversed(self.session.event_log[-6:]):
            if event.get("kind") == "action_rejected":
                return str((event.get("payload") or {}).get("reason") or "rejected")
        return "rejected"

    def observation(self) -> dict[str, Any]:
        state = self.session.rich_state()
        hero = (state.get("heroes") or {}).get(state.get("active_agent")) or {}
        return uniform_obs(
            game=GAME,
            tick=int(state.get("step_index") or 0),
            done=self.done,
            ascii_rows=ascii_rows((state.get("map") or {}).get("ascii")),
            legal_actions=list(ACTION_KINDS),
            score=self.outcome_reward,
            achievements=list(state.get("achievements") or []),
            state={
                "hero": hero,
                "active_agent": state.get("active_agent"),
                "legal": state.get("legal_actions") or {},
                "monsters": state.get("monsters") or {},
                "doors": state.get("doors") or {},
                "chests": state.get("chests") or {},
                "traps": state.get("traps") or {},
                "objective": state.get("objective"),
                "gold_collected": state.get("gold_collected"),
                "spells_cast": state.get("spells_cast"),
                "total_reward": state.get("total_reward"),
                "invalid_actions": self.invalid,
                "last_action": self.last_action,
                "last_applied": self.last_applied,
                "last_reject_reason": self.last_reject_reason,
                "attempts": self.attempts,
                "attempts_left": max(0, self.attempt_budget - self.attempts),
                "scenario_id": self.scenario_id,
                "max_steps": self.max_steps,
                "steps_left": max(0, self.max_steps - int(state.get("step_index") or 0)),
            },
        )

    def step(self, action: Any) -> dict[str, Any]:
        parsed = parse_action(action)
        before = self.outcome_reward
        self.attempts += 1
        result = self.session.step(parsed)
        self.last_action = action
        self.last_applied = bool(result.get("applied"))
        self.last_reject_reason = None
        if not result.get("applied"):
            self.invalid += 1
            self.last_reject_reason = self._reject_reason()
        after = self.outcome_reward
        return step_result(
            obs=self.observation(),
            reward=after - before,
            terminated=bool(self.session.done),
            truncated=bool(
                not self.session.done
                and (self.session.step_index >= self.max_steps or self.attempts >= self.attempt_budget)
            ),
            info={
                "outcome_reward": after,
                "achievements": list(self.session.achievements or []),
                "applied": bool(result.get("applied")),
                "invalid_actions": self.invalid,
                "terminal_reason": self.session.terminal_reason,
                "success": bool(self.session.success),
            },
        )


def make_session(seed: int, max_steps: int = DEFAULT_MAX_STEPS, split: str = "train") -> DungeonGridSession:
    return DungeonGridSession(seed, max_steps, split)


def spec() -> dict[str, Any]:
    return {
        "game": GAME,
        "env_id": ENV_ID,
        "action_space": list(ACTION_KINDS),
        "action_type": "string_or_dict",
        "max_horizon": DEFAULT_MAX_STEPS,
        "achievements": ["movement.first_step", "coordination.guard_used", "coordination.message_sent"],
        "objective": "Advance the quest: take gold, unlock achievements, cast spells, earn scenario reward.",
        "train_seeds": list(TRAIN_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "observation_notes": (
            "Actions are compact strings: 'move:east|west|north|south', 'attack_melee:<monster_id>', "
            "'open_door:<door_id>', 'interact:<chest_id>', 'cast:<spell>@<target|self>', "
            "'search_traps', 'inspect_tile', 'guard', 'end_turn' (a raw action dict also works). "
            "obs['state']['legal'] lists exactly what is available right now: ap, directions, "
            "adjacent_doors, adjacent_chests, adjacent_monsters, spells, can_interact_objective, "
            "can_escape. Every action costs AP; when ap hits 0 you must 'end_turn' to refresh, and "
            "actions attempted without AP are rejected and cost score. The episode also ends "
            "after obs['state']['attempts_left'] more attempts, applied or not, so looping on a "
            "rejected action burns the budget. After every step obs['state'] carries "
            "last_action, last_applied and last_reject_reason -- read them and try something else "
            "rather than repeating a move the engine just refused."
        ),
    }
