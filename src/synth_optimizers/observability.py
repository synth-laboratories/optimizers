from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

A3_TASK = "banking77"
GEPA_PROPOSER_HARNESS = "gepa_proposer"
BANKING77_EVAL_HARNESS = "banking77_eval"
LUNA_MED_POLICY_CONFIG = "luna_med"
SOL_MED_POLICY_CONFIG = "sol_med"


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
    PROPOSER_DELTA = "proposer.delta"
    CHILD_ROLLOUT_ATTACHED = "optimizer.child_rollout.attached"
    CANDIDATE_EVALUATION_ALLOCATED = "optimizer.candidate_evaluation.allocated"


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


def optimizer_event_log_id(optimizer_run_id: str) -> str:
    """One spool id per optimizer_run_id. Dual-write lives on this log."""
    run_id = str(optimizer_run_id).strip()
    if not run_id:
        raise ValueError("optimizer_run_id is required")
    return f"optimizer_event.v1:{run_id}"


def policy_ref(*, harness: str, config: str, code: str | None = None) -> dict[str, Any]:
    """Recipe pin. luna_med / sol_med are configs, not tasks. No harness_ref."""
    harness = str(harness).strip()
    config = str(config).strip()
    if not harness:
        raise ValueError("policy_ref.harness is required")
    if not config:
        raise ValueError("policy_ref.config is required")
    ref: dict[str, Any] = {"harness": harness, "config": config}
    if code is not None:
        ref["code"] = code
    return ref


def gepa_proposer_policy_ref(config: str) -> dict[str, Any]:
    return policy_ref(harness=GEPA_PROPOSER_HARNESS, config=config)


def banking77_eval_policy_ref(
    *, config: str | None = None, code: str | None = None
) -> dict[str, Any]:
    if config is None and code is None:
        raise ValueError("banking77 child eval policy_ref requires config or code")
    return policy_ref(harness=BANKING77_EVAL_HARNESS, config=config or "candidate", code=code)


def container_child_eval_ref(
    rollout_id: str,
    stream_id: str,
    reward_url: str,
    *,
    role: str = "candidate_evaluation",
) -> dict[str, Any]:
    """Containers resource ref for an optimizer child eval. No signed blob, NEV, or frames."""
    return {
        "schema": "synth.resource-ref.v1",
        "kind": "container_rollout",
        "id": rollout_id,
        "role": role,
        "attributes": {
            "stream_id": stream_id,
            "reward_url": reward_url,
        },
    }


PROPOSER_DELTA_EVENT_TYPE = "proposer.delta"
DEFAULT_PROPOSER_DELTA_CHANNEL = "content"
CHILD_ROLLOUT_ATTACHED_EVENT_TYPE = "optimizer.child_rollout.attached"


def proposer_delta_payload(
    generation: int,
    text: str,
    *,
    channel: str = DEFAULT_PROPOSER_DELTA_CHANNEL,
) -> dict[str, Any]:
    """Workshop consumer contract: generation + channel + text on proposer.delta."""
    if not str(channel).strip():
        channel = DEFAULT_PROPOSER_DELTA_CHANNEL
    return {
        "generation": generation,
        "channel": channel,
        "text": text,
    }


LUNA_MED_PROPOSER_POLICY_REF = gepa_proposer_policy_ref(LUNA_MED_POLICY_CONFIG)
SOL_MED_PROPOSER_POLICY_REF = gepa_proposer_policy_ref(SOL_MED_POLICY_CONFIG)


@dataclass
class InMemoryRunLog:
    """A bounded test spool for one ``optimizer_event.v1`` run.

    This is deliberately not a GEPA executor. It models the ownership and
    cursor rules used by the service while live tests switch between two
    independently advancing runs.
    """

    run_id: str
    policy_ref: dict[str, Any]
    log_id: str = ""
    live: bool = True
    _events: list[OptimizerEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_id = str(self.run_id).strip()
        if not self.run_id:
            raise ValueError("optimizer_run_id is required")
        self.policy_ref = _validated_policy_ref(self.policy_ref)
        expected_log_id = optimizer_event_log_id(self.run_id)
        if self.log_id and self.log_id != expected_log_id:
            raise ValueError(
                f"log_id {self.log_id!r} does not belong to optimizer run {self.run_id!r}"
            )
        self.log_id = expected_log_id

    def append(self, payload: Mapping[str, Any]) -> OptimizerEvent:
        if not self.live:
            raise RuntimeError(f"run {self.run_id} is sealed")
        source = dict(payload)
        supplied_run_id = str(source.get("run_id") or source.get("optimizer_run_id") or "").strip()
        if supplied_run_id and supplied_run_id != self.run_id:
            raise ValueError(f"event for {supplied_run_id!r} cannot be appended to {self.run_id!r}")
        event = OptimizerEvent.from_payload({**source, "run_id": self.run_id})
        if event.sequence_number is None:
            raise ValueError("optimizer_event.v1 sequence_number is required for a live spool")
        if event.sequence_number < 1:
            raise ValueError("optimizer_event.v1 sequence_number must be positive")
        if self._events:
            last_sequence = self._events[-1].sequence_number
            if last_sequence is None or event.sequence_number <= last_sequence:
                raise ValueError(
                    "optimizer_event.v1 sequence_number must advance monotonically "
                    f"({event.sequence_number} <= {last_sequence})"
                )
        self._events.append(event)
        return event

    def next_sequence(self) -> int:
        if not self._events:
            return 1
        last_sequence = self._events[-1].sequence_number
        if last_sequence is None:
            raise ValueError("optimizer_event.v1 live spool cannot resume from a missing sequence")
        return last_sequence + 1

    def append_proposer_delta(
        self,
        generation: int,
        text: str,
        *,
        channel: str = DEFAULT_PROPOSER_DELTA_CHANNEL,
    ) -> OptimizerEvent:
        return self.append(
            {
                "type": PROPOSER_DELTA_EVENT_TYPE,
                "sequence_number": self.next_sequence(),
                "algorithm": "gepa",
                "delta": proposer_delta_payload(generation, text, channel=channel),
            }
        )

    def append_child_eval(
        self,
        rollout_id: str,
        stream_id: str,
        reward_url: str,
    ) -> OptimizerEvent:
        return self.append(
            {
                "type": CHILD_ROLLOUT_ATTACHED_EVENT_TYPE,
                "sequence_number": self.next_sequence(),
                "algorithm": "gepa",
                "delta": {
                    "child_resource_ref": container_child_eval_ref(
                        rollout_id, stream_id, reward_url
                    )
                },
            }
        )

    def events_after(self, after_sequence: int = 0) -> tuple[OptimizerEvent, ...]:
        """Return an append-only cursor page without mutating live ownership."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        return tuple(
            event
            for event in self._events
            if event.sequence_number is not None and event.sequence_number > after_sequence
        )

    def seal(self) -> None:
        self.live = False

    @property
    def events(self) -> tuple[OptimizerEvent, ...]:
        return tuple(self._events)


class DualGepaHub:
    """Two or more isolated GEPA-shaped live logs used by switching tests.

    The hub proves spool isolation and cursor continuity only. A real A3 run
    still requires two Banking77 GEPA executions against Containers.
    """

    def __init__(self) -> None:
        self._runs: dict[str, InMemoryRunLog] = {}

    def start(self, run_id: str, proposer_policy_ref: Mapping[str, Any]) -> InMemoryRunLog:
        run_id = str(run_id).strip()
        if not run_id:
            raise ValueError("optimizer_run_id is required")
        if run_id in self._runs:
            raise ValueError(f"duplicate optimizer_run_id {run_id}")
        log = InMemoryRunLog(run_id=run_id, policy_ref=dict(proposer_policy_ref))
        log.append(
            {
                "type": "optimizer.run.started",
                "sequence_number": 1,
                "algorithm": "gepa",
                "delta": {"policy_ref": dict(log.policy_ref), "task": A3_TASK},
            }
        )
        log.append_proposer_delta(
            0,
            f"{run_id} proposer sample",
            channel=DEFAULT_PROPOSER_DELTA_CHANNEL,
        )
        self._runs[run_id] = log
        return log

    def start_luna_and_sol(
        self,
        luna_run_id: str = "gepa_luna",
        sol_run_id: str = "gepa_sol",
    ) -> tuple[InMemoryRunLog, InMemoryRunLog]:
        return (
            self.start(luna_run_id, LUNA_MED_PROPOSER_POLICY_REF),
            self.start(sol_run_id, SOL_MED_PROPOSER_POLICY_REF),
        )

    def get(self, run_id: str) -> InMemoryRunLog:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown optimizer_run_id {run_id}") from exc

    def flip_read(
        self,
        first: str,
        second: str,
        *,
        first_after: int = 0,
        second_after: int = 0,
    ) -> tuple[tuple[OptimizerEvent, ...], tuple[OptimizerEvent, ...]]:
        """Read A then B without sealing, draining, or sharing either spool."""
        first_log = self.get(first)
        second_log = self.get(second)
        if not first_log.live or not second_log.live:
            raise RuntimeError("flip_read requires both optimizer runs to remain live")
        return (
            first_log.events_after(first_after),
            second_log.events_after(second_after),
        )


def _validated_policy_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    if "harness_ref" in value:
        raise ValueError("policy_ref uses harness, not harness_ref")
    allowed = {"harness", "config", "code"}
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unsupported policy_ref fields: {sorted(extra)}")
    return policy_ref(
        harness=str(value.get("harness") or ""),
        config=str(value.get("config") or ""),
        code=_str_or_none(value.get("code")),
    )


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
