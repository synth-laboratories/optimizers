"""Typed local models mirroring common Synth optimization payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyCandidate:
    """Best-effort typed wrapper around a candidate payload."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyCandidate":
        return cls(payload=dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def candidate_id(self) -> str | None:
        value = self.payload.get("candidate_id")
        return str(value) if isinstance(value, str) else None


@dataclass(frozen=True)
class PolicyCandidatePage:
    """Best-effort typed wrapper around candidate list payloads."""

    items: list[PolicyCandidate] = field(default_factory=list)
    next_cursor: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyCandidatePage":
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raw_items = payload.get("candidates", [])
        items = [
            PolicyCandidate.from_dict(item)
            for item in raw_items
            if isinstance(item, dict)
        ]
        next_cursor = payload.get("next_cursor")
        return cls(items=items, next_cursor=str(next_cursor) if isinstance(next_cursor, str) else None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class PromptLearningResult:
    """Typed prompt-learning result payload."""

    job_id: str
    payload: dict[str, Any]

    @classmethod
    def from_response(cls, job_id: str, payload: dict[str, Any]) -> "PromptLearningResult":
        return cls(job_id=job_id, payload=dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

