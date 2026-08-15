"""Trusted local recipe catalog for `eval`.

A recipe is the only place an image, a seed schedule, a metric, a gate set, a
resource ceiling, or a selection rule may come from.  Agents choose a recipe id
from this catalog; they never supply an image, a command, a path, or an
environment variable.

The catalog is deliberately separate from hosted capability discovery: `eval`
is a local algorithm and must not appear in the hosted algorithm enum.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import (
    EvalContractError,
    ModelRoute,
    TrialBudget,
    SeedLedger,
    SelectionSpec,
    TargetManifest,
    TrialLimits,
    _identifier,
    _seed_tuple,
    _text,
    _text_tuple,
    digest_of,
)

CATALOG_DIR = Path(__file__).resolve().parent / "catalog"


@dataclass(frozen=True, slots=True)
class EvalRecipe:
    id: str
    title: str
    description: str
    task: str
    policy_kind: str
    image: str
    image_digest: str | None
    target: TargetManifest
    scenarios: tuple[str, ...]
    screening_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    selection: SelectionSpec
    limits: TrialLimits
    secrets: tuple[str, ...]
    prerequisites: tuple[str, ...]
    models: tuple[ModelRoute, ...]
    budget: TrialBudget | None

    @property
    def pinned_reference(self) -> str:
        """The exact image an executor may run.  A tag alone is not enough."""

        if not self.image_digest:
            raise EvalContractError(
                f"recipe {self.id} has no pinned image digest; publish and pin the target first"
            )
        return f"{self.image}@{self.image_digest}"

    @property
    def available(self) -> bool:
        return self.image_digest is not None

    @property
    def unavailable_reason(self) -> str | None:
        if self.available:
            return None
        return "target image is not published and pinned yet"

    def seed_ledger(self, *, sealed_at: str) -> SeedLedger:
        return SeedLedger(
            screening=self.screening_seeds,
            confirmation=self.confirmation_seeds,
            scenarios=self.scenarios,
            sealed_at=sealed_at,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "algorithmId": "eval",
            "task": self.task,
            "availability": "available" if self.available else "unavailable",
            "availabilityReason": self.unavailable_reason,
            "policyKind": self.policy_kind,
            "image": self.image,
            "imageDigest": self.image_digest,
            "targetManifest": self.target.to_json(),
            "targetManifestDigest": digest_of(self.target.to_json()),
            "credentialInputs": list(self.secrets),
            "prerequisites": list(self.prerequisites),
            "models": [model.to_json() for model in self.models],
            "budget": self.budget.to_json() if self.budget else None,
            "limits": {
                "scenarios": list(self.scenarios),
                "screeningSeeds": list(self.screening_seeds),
                "confirmationSeeds": list(self.confirmation_seeds),
                "trials": self.trial_count(),
                **self.limits.to_json(),
                "selection": self.selection.to_json(),
            },
        }

    def trial_count(self) -> int:
        """The whole matrix up front, so a run's ceiling is legible before it starts."""

        candidates_unknown = 1
        per_stage = len(self.scenarios) * candidates_unknown
        return per_stage * (len(self.screening_seeds) + len(self.confirmation_seeds))

    @classmethod
    def from_mapping(cls, value: Any, *, source: str) -> EvalRecipe:
        if not isinstance(value, dict):
            raise EvalContractError(f"{source}: recipe must be a table")
        target = TargetManifest.from_mapping(value.get("target"))
        policy_kind = _identifier(value.get("policy_kind"), field_name="policy_kind")
        if policy_kind not in target.policy_kinds:
            raise EvalContractError(
                f"{source}: policy_kind {policy_kind!r} is not accepted by the target"
            )
        selection = SelectionSpec.from_mapping(value.get("selection"))
        target.metric(selection.primary_metric)
        digest = value.get("image_digest")
        if digest is not None:
            digest = _text(digest, field_name="image_digest")
            if not digest.startswith("sha256:"):
                raise EvalContractError(f"{source}: image_digest must be a sha256 digest")
        confirmation = _seed_tuple(
            value.get("confirmation_seeds", []), field_name="confirmation_seeds"
        )
        if selection.decision_mode == "promote" and not confirmation:
            raise EvalContractError(f"{source}: a promoting recipe must declare confirmation seeds")
        models = tuple(ModelRoute.from_mapping(entry) for entry in value.get("models", []) or ())
        secrets = _text_tuple(value.get("secrets", []), field_name="secrets")
        for model in models:
            if model.secret not in secrets:
                raise EvalContractError(
                    f"{source}: model {model.id} needs secret {model.secret}, which the "
                    f"recipe does not declare"
                )
        raw_budget = value.get("budget")
        budget = TrialBudget.from_mapping(raw_budget) if raw_budget else None
        if models and budget is None:
            # A paid policy without a declared ceiling is how a smoke run turns
            # into an unbounded bill.
            raise EvalContractError(f"{source}: a recipe with paid models must declare a budget")
        return cls(
            id=_identifier(value.get("id"), field_name="recipe.id"),
            title=_text(value.get("title"), field_name="title"),
            description=_text(value.get("description"), field_name="description"),
            task=_identifier(value.get("task"), field_name="task"),
            policy_kind=policy_kind,
            image=_text(value.get("image"), field_name="image"),
            image_digest=digest,
            target=target,
            scenarios=_text_tuple(value.get("scenarios"), field_name="scenarios"),
            screening_seeds=_seed_tuple(value.get("screening_seeds"), field_name="screening_seeds"),
            confirmation_seeds=confirmation,
            selection=selection,
            limits=TrialLimits.from_mapping(value.get("limits")),
            secrets=secrets,
            prerequisites=tuple(value.get("prerequisites", []) or ()),
            models=models,
            budget=budget,
        )


@lru_cache(maxsize=1)
def catalog() -> tuple[EvalRecipe, ...]:
    """Every recipe shipped with this build, in stable id order."""

    recipes: list[EvalRecipe] = []
    for path in sorted(CATALOG_DIR.glob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        recipes.append(EvalRecipe.from_mapping(payload.get("recipe"), source=path.name))
    ids = [recipe.id for recipe in recipes]
    if len(set(ids)) != len(ids):
        raise EvalContractError("duplicate recipe id in the eval catalog")
    return tuple(sorted(recipes, key=lambda recipe: recipe.id))


def get_recipe(recipe_id: str) -> EvalRecipe:
    for recipe in catalog():
        if recipe.id == recipe_id:
            return recipe
    raise EvalContractError(f"unknown eval recipe: {recipe_id}")
