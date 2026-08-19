"""Typed hosted-training model catalog contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class HostedTrainingModel:
    model_id: str
    label: str
    provider: str
    provider_revision: str
    architecture: str
    max_context_length: int
    rank: Mapping[str, int]
    algorithms: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HostedTrainingModel":
        algorithms = payload.get("algorithms")
        rank = payload.get("rank")
        if not isinstance(algorithms, Mapping) or not isinstance(rank, Mapping):
            raise ValueError("hosted training model is missing algorithms or rank")
        return cls(
            model_id=str(payload["model_id"]),
            label=str(payload["label"]),
            provider=str(payload["provider"]),
            provider_revision=str(payload["provider_revision"]),
            architecture=str(payload["architecture"]),
            max_context_length=int(payload["max_context_length"]),
            rank={str(key): int(value) for key, value in rank.items()},
            algorithms={
                str(key): dict(value)
                for key, value in algorithms.items()
                if isinstance(value, Mapping)
            },
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class HostedTrainingModelCatalog:
    catalog_revision: str
    live_preflight_required: bool
    models: tuple[HostedTrainingModel, ...]
    total: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HostedTrainingModelCatalog":
        models = payload.get("models")
        if not isinstance(models, Sequence) or isinstance(models, str | bytes):
            raise ValueError("hosted training model catalog is missing models")
        return cls(
            catalog_revision=str(payload["catalog_revision"]),
            live_preflight_required=payload.get("live_preflight_required") is True,
            models=tuple(
                HostedTrainingModel.from_payload(model)
                for model in models
                if isinstance(model, Mapping)
            ),
            total=int(payload.get("total", 0)),
        )


__all__ = ["HostedTrainingModel", "HostedTrainingModelCatalog"]
