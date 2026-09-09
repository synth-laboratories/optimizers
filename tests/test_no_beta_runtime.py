from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "synth_optimizers"

FORBIDDEN = (
    "SYNTH_OPTIMIZERS_BETA_URL",
    "OPTIMIZERS_BETA_URL",
    "OPTIMIZERS_BETA_SERVICE_TOKEN",
    "BetaSftExecutorClient",
    "from optimizers_beta",
    "import optimizers_beta",
)


def test_public_runtime_does_not_contact_optimizers_beta() -> None:
    hits: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO).as_posix()
        for token in FORBIDDEN:
            if token in text:
                hits.append(f"{relative}: {token}")
    assert hits == []
