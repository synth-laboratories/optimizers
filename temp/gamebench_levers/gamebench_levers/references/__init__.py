"""Reference code policies used only by the headroom gate.

These are NOT seeds and are never handed to GEPA. They exist to answer one
question before any search budget is spent: does this game reward better code at
all? A seed already at the ceiling cannot show uplift, and a game where the
reference scores no better than the seed is a broken target, not a hard one.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def reference_policy(game: str) -> str:
    path = _DIR / f"{game}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no reference policy for {game}")
    return path.read_text(encoding="utf-8")
