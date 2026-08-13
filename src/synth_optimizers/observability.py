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
    sequence_number: int | None
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
        sequence_number = _first_present(payload, "sequence_number", "_seq", "seq")
        return cls(
            type=_enum_or_raw(OptimizerEventType, event_type),
            sequence_number=_int_or_none(sequence_number),
            run_id=str(payload.get("run_id") or payload.get("optimizer_run_id") or ""),
            algorithm=_str_or_none(payload.get("algorithm")),
            created_at=_str_or_none(payload.get("created_at") or payload.get("ts")),
            item=OptimizerItem.from_payload(_mapping_or_none(payload.get("item"))),
            delta=dict(_mapping_or_none(payload.get("delta")) or {}),
            error=_mapping_or_none(payload.get("error")),
            raw=dict(payload),
        )


def container_child_eval_ref(
    rollout_id: str,
    stream_id: str,
    reward_url: str,
) -> dict[str, Any]:
    """Containers resource ref for an optimizer child eval. No NEV/frames."""
    return {
        "schema": "synth.resource-ref.v1",
        "kind": "container_rollout",
        "id": rollout_id,
        "attributes": {
            "stream_id": stream_id,
            "reward_url": reward_url,
        },
    }


def gepa_policy_ref(*, proposer_model: str, harness: str = "gepa") -> dict[str, Any]:
    """Policy = harness + config. Missing proposer_model fails closed."""
    model = str(proposer_model or "").strip()
    if not model:
        raise ValueError("policy_ref.config.proposer_model is required")
    return {
        "harness": harness,
        "config": {"proposer_model": model},
    }


@dataclass
class InMemoryRunLog:
    """One optimizer_event.v1 spool. Missing sequence/reward stay None."""

    run_id: str
    policy_ref: dict[str, Any]
    log_id: str = ""
    live: bool = True
    _events: list[OptimizerEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.log_id:
            self.log_id = f"optimizer_event.v1:{self.run_id}"

    def append(self, payload: Mapping[str, Any]) -> OptimizerEvent:
        if not self.live:
            raise RuntimeError(f"run {self.run_id} is sealed")
        event = OptimizerEvent.from_payload({**dict(payload), "run_id": self.run_id})
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[OptimizerEvent, ...]:
        return tuple(self._events)


class DualGepaHub:
    """Two concurrent GEPA-shaped logs. Not a hosted Banking77 job (A3)."""

    def __init__(self) -> None:
        self._runs: dict[str, InMemoryRunLog] = {}

    def start(self, run_id: str, policy_ref: Mapping[str, Any]) -> InMemoryRunLog:
        if run_id in self._runs:
            raise ValueError(f"duplicate optimizer_run_id {run_id}")
        log = InMemoryRunLog(run_id=run_id, policy_ref=dict(policy_ref))
        log.append({"type": "optimizer.run.started", "sequence_number": 1, "algorithm": "gepa"})
        self._runs[run_id] = log
        return log

    def get(self, run_id: str) -> InMemoryRunLog:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown optimizer_run_id {run_id}") from exc

    def flip_read(
        self, first: str, second: str
    ) -> tuple[tuple[OptimizerEvent, ...], tuple[OptimizerEvent, ...]]:
        """Open visual A then B. Both stay live; logs do not cross."""
        a = self.get(first).events
        b = self.get(second).events
        if not self.get(first).live or not self.get(second).live:
            raise RuntimeError("flip_read stalled a live run")
        return a, b



@dataclass(frozen=True, slots=True)
class OptimizerStateSlice:
    slice: OptimizerStateSliceKind | str
    run_id: str
    algorithm: str | None = None
    cursor_seq: int | None = None
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
            cursor_seq=_int_or_none(payload.get("cursor_seq")),
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


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None
