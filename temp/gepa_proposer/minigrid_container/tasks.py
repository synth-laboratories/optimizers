"""MiniGrid episode seeds for GEPA.

Default env is Empty-5x5 (navigate to the green goal). DoorKey is available
via MINIGRID_ENV_ID. Reward is positive only on goal.
"""

from __future__ import annotations

TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
HELDOUT_SEEDS = [100, 101, 102, 103]

DEFAULT_SYSTEM_PROMPT = (
    "You are a MiniGrid agent. Reach the green goal square (G). "
    "Each turn you see an ASCII grid (# wall, . empty, K key, D locked door, O open door, G goal, >v<^ you) "
    "plus your facing direction and what you carry. "
    "Respond ONLY with JSON {\"action\": \"<left|right|forward|pickup|drop|toggle|done>\"}. "
    "Turn until the useful cell is ahead, then forward. "
    "If a key is present, pick it up (face it, pickup) before toggling a locked door. "
    "Do not repeat an action that just did nothing."
)


def rows_for(split: str) -> list[dict[str, str | int]]:
    seeds = TRAIN_SEEDS if split == "train" else HELDOUT_SEEDS
    prefix = "train" if split == "train" else "heldout"
    return [
        {
            "task_id": f"{prefix}:{seed}",
            "example_id": f"{prefix}:{seed}",
            "split": split,
            "seed": int(seed),
        }
        for seed in seeds
    ]


def episode_seed(task_id: str, split: str, seed: int) -> int:
    _, _, rest = str(task_id).partition(":")
    if rest.isdigit():
        return int(rest)
    ids = TRAIN_SEEDS if split == "train" else HELDOUT_SEEDS
    return ids[seed % len(ids)]
