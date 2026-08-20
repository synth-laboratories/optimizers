"""Per-game adapters. Import exactly one per process: gold_python collides across games."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

GAMEBENCH_ROOT = Path(
    os.environ.get("GAMEBENCH_ROOT", "/Users/joshuapurtell/Documents/GitHub/gamebench")
)
TASKS_ROOT = GAMEBENCH_ROOT / "tasks"

TASK_DIRS = {
    "sokoban": "sokoban-singleplayer",
    "craftax": "craftax-singleplayer",
    "rogue": "rogue-singleplayer",
    "dungeongrid": "dungeongrid-singleplayer",
}


def task_dir(game: str) -> Path:
    try:
        return TASKS_ROOT / TASK_DIRS[game]
    except KeyError:
        raise KeyError(f"unknown game: {game}") from None


def install_path(game: str) -> Path:
    """Put one game's gold_python/shared on sys.path. One game per process."""
    root = task_dir(game)
    if not root.is_dir():
        raise FileNotFoundError(f"gamebench task dir missing: {root}")
    for path in (root, root / "gold_python", root / "shared", root / "scripts"):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return root


def load(game: str):
    """Install the game's path, then import its adapter module."""
    install_path(game)
    return importlib.import_module(f"gamebench_levers.adapters.{game}")
