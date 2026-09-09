"""P0-4 lock, Python half.

`gepa run --proposer-*` used to be exported as `SYNTH_OPTIMIZERS_PROPOSER_*` and
read back by the Rust config loader after the TOML was parsed. That made the
process environment a second config authority for every run. The flags now
mutate the loaded config in process, and no `SYNTH_OPTIMIZERS_*` /
`GEPA_PLATFORM_*` variable is written anywhere under `src/`.

Runs without a maturin build: modules are loaded directly from `src/`.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

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


def _assignment_targets_under_src() -> dict[str, list[str]]:
    """Every `os.environ[...] = ...` / `os.environ.setdefault(...)` name in `src/`."""

    found: dict[str, list[str]] = {}

    def record(name: str, path: Path, lineno: int) -> None:
        found.setdefault(name, []).append(f"{path.relative_to(REPO_ROOT)}:{lineno}")

    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "environ"
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        record(target.slice.value, path, node.lineno)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                record(node.args[0].value, path, node.lineno)
    return found


#: `SYNTH_OPTIMIZERS_TERMINAL` selects the Rust terminal renderer for a CLI run.
#: It is presentation, never run config, and the CLI restores it afterwards.
ALLOWED_ENV_WRITES = {"SYNTH_OPTIMIZERS_TERMINAL"}


def test_no_run_config_override_is_written_to_the_environment() -> None:
    offenders = {
        name: where
        for name, where in _assignment_targets_under_src().items()
        if name.startswith(("SYNTH_OPTIMIZERS_", "GEPA_PLATFORM_"))
        and name not in ALLOWED_ENV_WRITES
    }
    assert offenders == {}, (
        "these run-config overrides are still written to the environment; the CLI must "
        f"mutate the loaded config instead: {offenders}"
    )


def test_proposer_flags_mutate_the_config_in_process() -> None:
    cli = _load("cli")

    class Config:
        class Proposer:
            execution_mode = "local_process"
            model = "model-from-toml"
            reasoning_effort = "medium"
            service_tier = None
            auth_mode = "api_key"
            codex_home = None

        proposer = Proposer()

    class Args:
        proposer_execution_mode = "Docker"
        proposer_model = " model-from-flag "
        proposer_reasoning_effort = "HIGH"
        proposer_service_tier = "Flex"
        proposer_auth_mode = "chat-gpt"
        proposer_codex_home = "/tmp/codex"

    config = Config()
    cli._apply_proposer_overrides(config, Args())
    assert config.proposer.execution_mode == "docker"
    assert config.proposer.model == "model-from-flag"
    assert config.proposer.reasoning_effort == "high"
    assert config.proposer.service_tier == "flex"
    assert config.proposer.auth_mode == "chat_gpt"
    assert config.proposer.codex_home == "/tmp/codex"


def test_absent_flags_leave_the_config_alone() -> None:
    cli = _load("cli")

    class Config:
        class Proposer:
            execution_mode = "local_process"
            model = "model-from-toml"
            reasoning_effort = "medium"
            service_tier = None
            auth_mode = "api_key"
            codex_home = None

        proposer = Proposer()

    class Args:
        proposer_execution_mode = None
        proposer_model = None
        proposer_reasoning_effort = None
        proposer_service_tier = None
        proposer_auth_mode = None
        proposer_codex_home = None

    config = Config()
    cli._apply_proposer_overrides(config, Args())
    assert config.proposer.model == "model-from-toml"
    assert config.proposer.auth_mode == "api_key"


def test_one_backend_url_name() -> None:
    gepa = _load("gepa")
    assert gepa.BACKEND_BASE_URL_ENV == "SYNTH_BACKEND_URL"
    aliases = (
        "SYNTH_BACKEND_URL_OVERRIDE",
        "SYNTH_API_URL",
        "DEV_SYNTH_BACKEND_URL",
        "DEV_BACKEND_URL",
        "PROD_SYNTH_BACKEND_URL",
        "PROD_BACKEND_URL",
        '"BACKEND_URL"',
    )
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text()
        for alias in aliases:
            assert alias not in text, (
                f"{path.relative_to(REPO_ROOT)} still reads backend URL alias {alias}; "
                "there is one name"
            )
