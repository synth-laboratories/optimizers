"""The app-owned `eval` home: config, pins, candidate store, run evidence.

Everything configurable about local `eval` lives in TOML inside this directory,
never in environment variables, so a run's settings are inspectable after the
fact and survive a restart. The Desktop app owns the location and passes it on
the worker manifest; agents never name it.

Layout::

    <home>/runtime.toml        container runtime + global concurrency ceiling
    <home>/pins.toml           operator-pinned image digests, per catalog recipe
    <home>/semaphore/          global lease store, shared by every local run
    <home>/candidates/<id>/    immutable staged candidate sets
    <home>/runs/<run_id>/      sealed manifests and trial evidence
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .models import EvalContractError
from .recipes import EvalRecipe, catalog

LEGACY_DEFAULT_RUNTIME_TOML = """\
# Local `eval` runtime. The Desktop app owns this file; it is not agent input.

# OCI runtime used to launch pinned target images.
container_runtime = "docker"

# Hard ceiling on trial containers running at once across *all* local eval runs.
max_concurrent_trials = 2

# Seconds a semaphore lease survives without a heartbeat before it is reclaimed.
lease_ttl_seconds = 120
"""

DEFAULT_RUNTIME_TOML = LEGACY_DEFAULT_RUNTIME_TOML.replace(
    "max_concurrent_trials = 2", "max_concurrent_trials = 10"
)

DEFAULT_SECRETS_TOML = """\
# Credentials a recipe explicitly declares. Only a name a recipe lists in its
# `secrets` can ever be read from here, and only that name reaches the trial
# container. A value is never written to a run manifest, an event, or evidence.
#
# [secrets]
# OPENAI_API_KEY = "sk-..."
"""

DEFAULT_PINS_TOML = """\
# Operator-pinned image digests for catalog recipes. A pin may only supply the
# digest of the image the catalog already names: it cannot change the image,
# the command, the mounts, the limits, or the selection rule.
#
# [pins."eval.fixture.policy-smoke.v1"]
# image_digest = "sha256:..."
"""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    container_runtime: str
    max_concurrent_trials: int
    lease_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class EvalHome:
    root: Path
    config: RuntimeConfig

    @classmethod
    def open(cls, root: Path | str, *, create: bool = True) -> EvalHome:
        path = Path(root).expanduser()
        if create:
            path.mkdir(parents=True, exist_ok=True)
            for name, default in (
                ("runtime.toml", DEFAULT_RUNTIME_TOML),
                ("pins.toml", DEFAULT_PINS_TOML),
                ("secrets.toml", DEFAULT_SECRETS_TOML),
            ):
                target = path / name
                if not target.exists():
                    target.write_text(default, encoding="utf-8")
                elif name == "runtime.toml":
                    # v0.8's ten-lane NanoHorizon recipe cannot be live if an
                    # untouched older app default silently caps it at two.
                    # Migrate only the byte-identical app-owned default; an
                    # operator-edited ceiling remains authoritative.
                    current = target.read_text(encoding="utf-8")
                    if current == LEGACY_DEFAULT_RUNTIME_TOML:
                        target.write_text(DEFAULT_RUNTIME_TOML, encoding="utf-8")
        return cls(root=path, config=_load_runtime(path / "runtime.toml"))

    @property
    def semaphore_dir(self) -> Path:
        return self.root / "semaphore"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def pins(self) -> dict[str, str]:
        path = self.root / "pins.toml"
        if not path.is_file():
            return {}
        payload = tomllib.loads(path.read_text(encoding="utf-8")).get("pins", {})
        pins: dict[str, str] = {}
        for recipe_id, value in payload.items():
            digest = value.get("image_digest") if isinstance(value, dict) else None
            if isinstance(digest, str) and digest.startswith("sha256:"):
                pins[recipe_id] = digest
        return pins

    def resolve_secret(self, name: str, *, declared: tuple[str, ...]) -> str:
        """Resolve one recipe-declared credential. Never a name it did not declare."""

        if name not in declared:
            raise EvalContractError(
                f"{name} is not declared by this recipe; a trial only ever receives "
                f"credentials the recipe named"
            )
        path = self.root / "secrets.toml"
        if path.is_file():
            table = tomllib.loads(path.read_text(encoding="utf-8")).get("secrets", {})
            value = table.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = os.environ.get(name, "")
        if value.strip():
            return value.strip()
        raise EvalContractError(
            f"{name} is required by this recipe but is set neither in "
            f"{path} nor in the Desktop process environment"
        )

    def catalog(self) -> tuple[EvalRecipe, ...]:
        """The shipped catalog with operator pins applied to digests only."""

        pins = self.pins()
        resolved = []
        for recipe in catalog():
            digest = pins.get(recipe.id, recipe.image_digest)
            resolved.append(
                recipe if digest == recipe.image_digest else _with_digest(recipe, digest)
            )
        return tuple(resolved)

    def recipe(self, recipe_id: str) -> EvalRecipe:
        for recipe in self.catalog():
            if recipe.id == recipe_id:
                return recipe
        raise EvalContractError(f"unknown eval recipe: {recipe_id}")

    def write_pin(self, recipe_id: str, digest: str) -> None:
        """Record an operator pin. Unknown recipes and bad digests are refused."""

        if not any(recipe.id == recipe_id for recipe in catalog()):
            raise EvalContractError(f"unknown eval recipe: {recipe_id}")
        if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
            raise EvalContractError("pin digest must be sha256:<64 hex chars>")
        pins = self.pins()
        pins[recipe_id] = digest
        lines = [DEFAULT_PINS_TOML]
        for key in sorted(pins):
            lines.append(f'[pins."{key}"]\nimage_digest = "{pins[key]}"\n')
        (self.root / "pins.toml").write_text("\n".join(lines), encoding="utf-8")


def _with_digest(recipe: EvalRecipe, digest: str | None) -> EvalRecipe:
    return EvalRecipe(
        id=recipe.id,
        title=recipe.title,
        description=recipe.description,
        task=recipe.task,
        policy_kind=recipe.policy_kind,
        image=recipe.image,
        image_digest=digest,
        target=recipe.target,
        scenarios=recipe.scenarios,
        screening_seeds=recipe.screening_seeds,
        confirmation_seeds=recipe.confirmation_seeds,
        selection=recipe.selection,
        limits=recipe.limits,
        secrets=recipe.secrets,
        prerequisites=recipe.prerequisites,
        models=recipe.models,
        budget=recipe.budget,
    )


def _load_runtime(path: Path) -> RuntimeConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    runtime = payload.get("container_runtime", "docker")
    if runtime not in {"docker", "podman"}:
        raise EvalContractError(
            f"{path}: container_runtime must be docker or podman, got {runtime!r}"
        )
    concurrency = payload.get("max_concurrent_trials", 2)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise EvalContractError(f"{path}: max_concurrent_trials must be a positive integer")
    ttl = payload.get("lease_ttl_seconds", 120)
    if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < 5:
        raise EvalContractError(f"{path}: lease_ttl_seconds must be an integer >= 5")
    return RuntimeConfig(
        container_runtime=runtime, max_concurrent_trials=concurrency, lease_ttl_seconds=ttl
    )
