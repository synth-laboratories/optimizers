from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OptimizerEventType(StrEnum):
    RUN_QUEUED = "optimizer.run.queued"
    RUN_STARTED = "optimizer.run.started"
    RUN_COMPLETED = "optimizer.run.completed"
    RUN_FAILED = "optimizer.run.failed"
    RUN_CANCELLED = "optimizer.run.cancelled"
    CURSOR_UPDATED = "optimizer.cursor.updated"
    STATE_SLICE_UPDATED = "optimizer.state_slice.updated"
    CANDIDATE_ADDED = "optimizer.candidate.added"
    CANDIDATE_UPDATED = "optimizer.candidate.updated"
    CANDIDATE_ACCEPTED = "optimizer.candidate.accepted"
    CANDIDATE_REJECTED = "optimizer.candidate.rejected"
    FRONTIER_UPDATED = "optimizer.frontier.updated"
    ROLLOUT_UPDATED = "optimizer.rollout.updated"
    MODEL_CALL_UPDATED = "optimizer.model_call.updated"
    LOG = "optimizer.log"
    HEARTBEAT = "optimizer.heartbeat"
    ERROR = "optimizer.error"


class OptimizerItemType(StrEnum):
    RUN = "run"
    CURSOR = "cursor"
    CANDIDATE = "candidate"
    FRONTIER_CELL = "frontier_cell"
    ROLLOUT = "rollout"
    MODEL_CALL = "model_call"
    LOG = "log"


class OptimizerStateSliceKind(StrEnum):
    CURSOR = "cursor"
    CANDIDATES = "candidates"
    FRONTIER = "frontier"
    BOARD = "board"
    AGENTS = "agents"
    THEMES = "themes"
    DATA_ENGINE = "data-engine"
    LOGS = "logs"
    USAGE = "usage"


class OptimizerLogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class OptimizerItem:
    type: OptimizerItemType | str
    id: str | None = None
    status: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "OptimizerItem | None":
        if payload is None:
            return None
        item_type = str(payload.get("type") or "").strip()
        if not item_type:
            return None
        return cls(
            type=_enum_or_raw(OptimizerItemType, item_type),
            id=_str_or_none(payload.get("id")),
            status=_str_or_none(payload.get("status")),
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class OptimizerEvent:
    type: OptimizerEventType | str
    sequence_number: int
    run_id: str
    algorithm: str | None = None
    created_at: str | None = None
    item: OptimizerItem | None = None
    delta: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OptimizerEvent":
        event_type = str(
            payload.get("type") or payload.get("event_type") or "optimizer.event"
        ).strip()
        sequence_number = payload.get("sequence_number", payload.get("_seq", payload.get("seq", 0)))
        return cls(
            type=_enum_or_raw(OptimizerEventType, event_type),
            sequence_number=_int_or_zero(sequence_number),
            run_id=str(payload.get("run_id") or ""),
            algorithm=_str_or_none(payload.get("algorithm")),
            created_at=_str_or_none(payload.get("created_at") or payload.get("ts")),
            item=OptimizerItem.from_payload(_mapping_or_none(payload.get("item"))),
            delta=dict(_mapping_or_none(payload.get("delta")) or {}),
            error=_mapping_or_none(payload.get("error")),
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class OptimizerStateSlice:
    slice: OptimizerStateSliceKind | str
    run_id: str
    algorithm: str | None = None
    cursor_seq: int = 0
    updated_at: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OptimizerStateSlice":
        raw_slice = str(payload.get("slice") or "").strip()
        return cls(
            slice=_enum_or_raw(OptimizerStateSliceKind, raw_slice) if raw_slice else raw_slice,
            run_id=str(payload.get("run_id") or ""),
            algorithm=_str_or_none(payload.get("algorithm")),
            cursor_seq=_int_or_zero(payload.get("cursor_seq")),
            updated_at=_str_or_none(payload.get("updated_at")),
            data=dict(_mapping_or_none(payload.get("data")) or {}),
            raw=dict(payload),
        )


def state_slice_value(slice_kind: OptimizerStateSliceKind | str) -> str:
    return slice_kind.value if isinstance(slice_kind, OptimizerStateSliceKind) else str(slice_kind)


def _enum_or_raw(enum_type: type[StrEnum], value: str) -> StrEnum | str:
    try:
        return enum_type(value)
    except ValueError:
        return value


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None
