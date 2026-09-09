"""P0-5 lock, Python half.

Proves that ``contracts/event_vocabulary.json`` is the union of what this repo
can actually emit: the Python eval worker feed is scanned out of the source, the
Rust half is checked for shape (the Rust ``observability::vocabulary`` tests own
its content), and the committed file must agree with both.

Runs without a maturin build: the package ``__init__`` imports the native
extension, so the module under test is loaded directly from ``src/``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "synth_optimizers"


def _load(name: str):
    """Import ``synth_optimizers.<name>``, native extension or not.

    A built package is used as-is. Without one (`synth_optimizers/__init__.py`
    imports `._synth_optimizers` at module scope) the module is loaded straight
    from `src/` under a private package name, so this test stays runnable
    before a maturin build and never replaces the real package for the rest of
    the session.
    """

    try:
        return importlib.import_module(f"synth_optimizers.{name}")
    except Exception:
        pass
    alias = "_synth_optimizers_src"
    package = sys.modules.get(alias)
    if package is None:
        package = types.ModuleType(alias)
        package.__path__ = [str(SRC)]
        package.__package__ = alias
        package.__version__ = "0.0.0-source-load"
        sys.modules[alias] = package
        sys.modules.setdefault(f"{alias}._synth_optimizers", types.ModuleType("native"))
    spec = importlib.util.spec_from_file_location(f"{alias}.{name}", SRC / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{alias}.{name}"] = module
    spec.loader.exec_module(module)
    return module


o11y = _load("o11y")


def _python_emit_sites() -> dict[str, list[str]]:
    """Every literal event name reaching an ``EventLog.emit`` call under ``src/``.

    An emit whose first argument is not a literal fails the test: the vocabulary
    cannot be honest about a name it cannot see.
    """

    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "emit":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                found.setdefault(first.value, []).append(where)
            else:
                pytest.fail(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: emit() event name is not a "
                    "string literal; the event vocabulary scan cannot resolve it"
                )
    return found


def test_python_event_types_match_the_emitters() -> None:
    scanned = set(_python_emit_sites())
    declared = set(o11y.PYTHON_EVENT_TYPES)
    assert scanned - declared == set(), (
        "these event names are emitted but missing from PYTHON_EVENT_TYPES: "
        f"{sorted(scanned - declared)}"
    )
    assert declared - scanned == set(), (
        "these event names are declared but nothing emits them — delete them, do not add "
        f"an emitter to satisfy this test: {sorted(declared - scanned)}"
    )


def test_declared_python_event_types_are_sorted_and_unique() -> None:
    declared = list(o11y.PYTHON_EVENT_TYPES)
    assert declared == sorted(set(declared))


def test_committed_vocabulary_equals_the_computed_union() -> None:
    committed = json.loads((REPO_ROOT / "contracts" / "event_vocabulary.json").read_text())
    assert committed == o11y.build_event_vocabulary(), (
        "contracts/event_vocabulary.json is stale; regenerate with "
        "`uv run python -m synth_optimizers.o11y --write-event-vocabulary`"
    )


def test_committed_vocabulary_is_sorted_and_well_formed() -> None:
    committed = json.loads((REPO_ROOT / "contracts" / "event_vocabulary.json").read_text())
    assert committed["schema_version"] == o11y.EVENT_VOCABULARY_SCHEMA
    entries = committed["event_types"]
    names = [entry["event_type"] for entry in entries]
    assert names == sorted(set(names)), "event_vocabulary.json must be sorted and unique"
    for entry in entries:
        assert entry["emitter"] in {"rust", "python"}
        assert entry["feeds"], f"{entry['event_type']} has no feed"
        assert entry["feeds"] == sorted(set(entry["feeds"]))
        for feed in entry["feeds"]:
            assert feed in committed["feeds"], f"{feed} is not described in feeds"


def test_python_half_of_the_committed_vocabulary() -> None:
    committed = json.loads((REPO_ROOT / "contracts" / "event_vocabulary.json").read_text())
    python_names = {
        entry["event_type"] for entry in committed["event_types"] if entry["emitter"] == "python"
    }
    assert python_names == set(o11y.PYTHON_EVENT_TYPES)


def test_exported_path_resolves() -> None:
    assert o11y.event_vocabulary_path().is_file()
    assert o11y.load_event_vocabulary()["schema_version"] == o11y.EVENT_VOCABULARY_SCHEMA
