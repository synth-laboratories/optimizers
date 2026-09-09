"""P0-9 lock — Python half.

A 2,000-line cap on every ``.py`` file under ``src/`` and ``tests/``, with an
explicit allowlist of the files that are already over it. Each allowlist entry
records a ceiling, so an offender may only shrink: adding lines to one of these
files fails here, and a new file crossing the cap has no allowlist to hide
behind.

Run: ``uv run pytest tests/code_quality/test_file_size_cap.py -q``
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("src", "tests")

#: Lines. Decision D-X-2 in the v0.7 structure review, alongside the 600-line
#: renderer cap in Workshop.
MAX_LINES = 2_000

#: Files already over the cap, with the count they may not exceed. Entries
#: leave this list one of two ways: the file drops under the cap (then the
#: entry must be deleted, which this test enforces), or the file is split.
#:
#: ``cli.py`` carries eight argparse trees including the `mapo`/`reflexion`/
#: `gelo` surfaces that P4-3 removes; ``o11y.py`` embeds the board's HTML/JS.
ALLOWLIST: dict[str, int] = {
    "src/synth_optimizers/cli.py": 2_567,
    "src/synth_optimizers/hosted.py": 2_014,
    "src/synth_optimizers/o11y.py": 3_236,
}


def _line_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            counts[relative] = len(path.read_text().splitlines())
    return counts


def test_allowlist_is_sorted() -> None:
    assert list(ALLOWLIST) == sorted(ALLOWLIST), "ALLOWLIST must be sorted"


def test_no_unlisted_file_is_over_the_cap() -> None:
    offenders = {
        path: lines
        for path, lines in _line_counts().items()
        if lines > MAX_LINES and path not in ALLOWLIST
    }
    assert offenders == {}, (
        f"these files are over the {MAX_LINES}-line cap and are not allowlisted. Split "
        f"them; do not add them to the list without a review: {offenders}"
    )


def test_allowlisted_files_only_shrink() -> None:
    counts = _line_counts()
    grown = {
        path: (counts[path], ceiling)
        for path, ceiling in ALLOWLIST.items()
        if path in counts and counts[path] > ceiling
    }
    assert grown == {}, (
        "an allowlisted file grew (actual, ceiling). These files may only shrink — put "
        f"the new code in a new module instead of raising the ceiling: {grown}"
    )


def test_allowlist_has_no_stale_entries() -> None:
    counts = _line_counts()
    stale = {
        path: counts.get(path)
        for path in ALLOWLIST
        if path not in counts or counts[path] <= MAX_LINES
    }
    assert stale == {}, f"remove these from ALLOWLIST — the list only shrinks: {stale}"
