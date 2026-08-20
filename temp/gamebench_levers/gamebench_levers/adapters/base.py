"""Adapter contract shared by every GameBench lever container.

An adapter turns one GameBench engine into a uniform episodic surface:

    session = adapter.make_session(task_id, max_steps)
    obs     = session.observation()
    result  = session.step(action)      # {obs, reward, terminated, truncated, info}

`reward` is the per-step delta; `info["outcome_reward"]` is the episode score the
optimizer maximizes. Observations are JSON and uniform in their top-level keys so
one code policy or harness shape reads across all four games.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


UNIFORM_OBS_KEYS = (
    "game",
    "tick",
    "done",
    "ascii",
    "legal_actions",
    "score",
    "achievements",
    "state",
)


@runtime_checkable
class Session(Protocol):
    def observation(self) -> dict[str, Any]: ...

    def step(self, action: Any) -> dict[str, Any]: ...

    @property
    def outcome_reward(self) -> float: ...

    @property
    def done(self) -> bool: ...


def ascii_rows(value: Any) -> list[str]:
    """GameBench readouts spell ascii as a newline string or a row list. Normalize."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.splitlines()
    return [str(row) for row in value]


def uniform_obs(
    *,
    game: str,
    tick: int,
    done: bool,
    ascii_rows: list[str],
    legal_actions: list[str],
    score: float,
    achievements: list[str],
    state: dict[str, Any],
    text: str | None = None,
) -> dict[str, Any]:
    """The observation every adapter returns. Extra game detail rides in `state`."""
    obs: dict[str, Any] = {
        "game": game,
        "tick": int(tick),
        "done": bool(done),
        "ascii": list(ascii_rows),
        "legal_actions": list(legal_actions),
        "score": float(score),
        "achievements": list(achievements),
        "state": state,
    }
    if text is not None:
        obs["text"] = text
    return obs


def step_result(
    *,
    obs: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "obs": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }
