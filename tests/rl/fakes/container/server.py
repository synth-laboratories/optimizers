"""The HTTP surface: state machine, handshake, and route dispatch.

One lock guards all mutable state and no work happens in the background, so
an attempt only advances when the executor polls it. Time is read from an
injected clock; nothing here sleeps.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

from synth_optimizers.contracts.rl_clauses import (
    ALL_CLAUSES,
    HANDSHAKE_SCHEMA_VERSION,
    MANDATORY_CLAUSES,
)
from synth_optimizers.contracts.rl_identity import (
    AgentInstance,
    RolloutReceipt,
    TopologyError,
)
from synth_optimizers.contracts.rl_records import InferenceCall, RecordError, digest
from synth_optimizers.rl.capabilities import ClauseResult, canonical_capability_hash
from synth_optimizers.rl.handshake import Obligations, TaskResolution, compute_agreement_digest

from .client import ContainerClient
from .codecs import (
    _call_payload,
    _capability_payload,
    _episode_payload,
    _receipt_payload,
    _renderer_payload,
    _reward_payload,
    _segment_payload,
    _topology_payload,
)
from .config import (
    ALIAS_REFS,
    CONTRACT_VERSION,
    CORRELATION_FIELDS,
    DECLARED_ROUTES,
    Clock,
    ContainerConfig,
)
from .evidence import (
    _Attempt,
    _behavior,
    _context_segments,
    _episodes,
    _instance_calls,
    _judge_call,
    _Refusal,
    _reward,
)

class _State:
    """All mutable container state. Guarded by one lock; no background work."""

    def __init__(self, cfg: ContainerConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self.lock = threading.Lock()
        self.capability_epoch = 0
        self.handshakes: dict[str, dict[str, Any]] = {}
        self.revoked: set[str] = set()
        self.policy_configs: dict[str, dict[str, Any]] = {}
        self.policy_sets: dict[str, dict[str, Any]] = {}
        self.attempts: dict[str, _Attempt] = {}
        self.idempotency: dict[str, str] = {}
        self.states: dict[str, str] = {}
        self.polls: dict[str, int] = {}
        self.leases: dict[str, float] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.terminals: dict[str, int] = {}
        self.traces: dict[str, dict[str, Any]] = {}
        self.rewards: dict[str, dict[str, Any] | None] = {}
        self.refusals: dict[str, str] = {}
        self.counter = 0
        self.request_log: list[tuple[str, str]] = []

    # -- capabilities --------------------------------------------------- #

    def capability_document(self) -> dict[str, Any]:
        """The advertisement as it goes on the wire: hash included, no envelope."""

        document = _capability_payload(self.cfg, capability_epoch=self.capability_epoch)
        document["capability_hash"] = canonical_capability_hash(document)
        return document

    def capability_hash(self) -> str:
        return canonical_capability_hash(
            _capability_payload(self.cfg, capability_epoch=self.capability_epoch)
        )

    # -- events --------------------------------------------------------- #

    def emit(self, rollout_id: str, kind: str, **extra: Any) -> None:
        log = self.events.setdefault(rollout_id, [])
        log.append(
            {
                "cursor": len(log) + 1,
                "kind": kind,
                "at": self.clock.rfc3339(),
                "rollout_id": rollout_id,
                **extra,
            }
        )
        if kind in {"episode", "failure", "cancellation"}:
            self.terminals[rollout_id] = self.terminals.get(rollout_id, 0) + 1


# --------------------------------------------------------------------------- #
# Handshake
# --------------------------------------------------------------------------- #


def _clause_verdicts(
    state: _State, request: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = state.cfg
    plan = dict(request.get("run_plan") or {})
    requested_slots = int(plan.get("max_execution_slots") or plan.get("group_size") or 1)
    verdicts: dict[str, tuple[str, str]] = {clause: ("accepted", "") for clause in ALL_CLAUSES}

    if not cfg.tito_supported:
        verdicts["evidence.tito"] = ("unsupported", "container does not speak token in/out")
    if not cfg.artifact_by_reference:
        verdicts["evidence.artifact_reference"] = ("unsupported", "traces are always inline")
    if cfg.settlement_window_seconds <= 0:
        verdicts["reward.settlement_window"] = ("unsupported", "scored state does not lag")
    if not cfg.quiescence_supported:
        verdicts["reward.horizon_quiescence"] = (
            "degraded",
            "cannot kill agent-authored background processes; "
            "serves a horizon-clipped state snapshot instead",
        )
    if not cfg.lease_renewable and cfg.lease_ttl_seconds < cfg.horizon_seconds:
        verdicts["lifecycle.lease_renewal"] = (
            "rejected",
            f"lease ttl {cfg.lease_ttl_seconds}s cannot be extended and is shorter than "
            f"the declared horizon {cfg.horizon_seconds}s; an hour-scale episode may not "
            "depend on an HTTP request staying open",
        )
    if requested_slots > cfg.advertised_concurrency:
        verdicts["lifecycle.concurrency"] = (
            "degraded",
            f"{cfg.advertised_concurrency} leases available, {requested_slots} requested",
        )
    if len(cfg.topology.agent_instances) < 2:
        verdicts["topology.channels"] = ("unsupported", "single-instance topology")
        verdicts["topology.opponent_pinning"] = ("unsupported", "no opponent instances")
        verdicts["topology.minimum_roster"] = ("unsupported", "single-instance topology")
    if not cfg.topology.opponent_instances:
        verdicts["topology.opponent_pinning"] = ("unsupported", "no opponent instances")

    requested_profile = dict(request.get("renderer_profile") or {})
    if requested_profile:
        declared = _renderer_payload(cfg.renderer_profile)
        mismatched = [
            name
            for name in ("profile_id", "config_digest", "tokenizer_digest")
            if name in requested_profile and requested_profile[name] != declared[name]
        ]
        if mismatched:
            verdicts["policy.renderer_profile_match"] = (
                "rejected",
                f"renderer profile differs on {sorted(mismatched)}",
            )

    if abs(cfg.clock_skew_seconds) > cfg.skew_tolerance_seconds:
        verdicts[cfg.skew_clause_id] = (
            "rejected",
            f"measured skew {cfg.clock_skew_seconds}s exceeds tolerance "
            f"{cfg.skew_tolerance_seconds}s and the horizon is when reward is read",
        )

    for clause_id, (verdict, reason) in cfg.clause_overrides.items():
        verdicts[clause_id] = (verdict, reason)

    clauses = [
        {"clause_id": clause_id, "verdict": verdict, "reason": reason}
        for clause_id, (verdict, reason) in verdicts.items()
    ]
    obligations = {
        "max_concurrency": cfg.advertised_concurrency,
        "lease_ttl_seconds": cfg.lease_ttl_seconds,
        "lease_renewable": cfg.lease_renewable,
        "deferred_scoring": cfg.deferred_scoring,
        "quiescence": cfg.quiescence_supported,
        "settlement_window_seconds": cfg.settlement_window_seconds,
        "partial_roster": cfg.partial_roster_disposition,
        "probe_binding": cfg.probe_binding_supported,
        "prompt_budget_policy": cfg.prompt_budget_policy,
        "horizon": {
            "horizon_kind": cfg.horizon.horizon_kind,
            "value": cfg.horizon.value,
            "value_seconds": cfg.horizon_seconds,
            "seconds_per_unit": cfg.horizon.seconds_per_unit,
            "time_dilation": cfg.horizon.time_dilation,
        },
    }
    return clauses, obligations


class _EchoRequest:
    """The requirement document as received, for the canonical digest.

    ``compute_agreement_digest`` needs the executor's request payload and
    nothing else about it; the container holds the bytes it was sent, so it
    hands them back verbatim rather than re-deriving a request it did not
    author.
    """

    __slots__ = ("_payload",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = {
            key: value for key, value in payload.items() if key != "renew_of"
        }

    def to_payload(self) -> dict[str, Any]:
        return dict(self._payload)


def _agreement_digest(
    state: _State,
    request: Mapping[str, Any],
    *,
    handshake_id: str,
    capability_hash: str,
    clauses: Sequence[Mapping[str, Any]],
    obligations: Mapping[str, Any],
    resolution: Sequence[Mapping[str, Any]],
) -> str:
    """The digest both sides compute, from the shared handshake module."""

    return compute_agreement_digest(
        _EchoRequest(request),  # type: ignore[arg-type]
        handshake_id=handshake_id,
        capability_hash=capability_hash,
        renderer_fingerprint=state.cfg.renderer_profile.fingerprint,
        taskset_resolution=[
            TaskResolution(
                task_id=str(row["task_id"]),
                content_digest=str(row["content_digest"]),
                topology_ref=str(row["topology_ref"]),
            )
            for row in resolution
        ],
        obligations=Obligations.from_payload(obligations),
        clauses=[
            ClauseResult(
                clause_id=str(row["clause_id"]),
                verdict=str(row["verdict"]),
                reason=str(row.get("reason") or ""),
                source="container",
            )
            for row in clauses
        ],
    )


def _renew(state: _State, handshake_id: str) -> tuple[int, dict[str, Any]]:
    """Extend an agreement. A renewal never mints a new one.

    The handshake id and the agreement digest are what the executor's ledger
    gates every attempt on, so a renewal that changed either would not be a
    renewal: it would silently replace the agreement the run is already
    executing under.
    """

    cfg = state.cfg
    record = dict(state.handshakes[handshake_id])
    record["expires_at"] = state.clock.rfc3339(cfg.handshake_ttl_seconds)
    record["expires_at_offset"] = state.clock.now() + cfg.handshake_ttl_seconds
    record["clock"] = {
        "container_time": state.clock.rfc3339(cfg.clock_skew_seconds),
        "measured_skew_seconds": cfg.clock_skew_seconds,
        "tolerance_seconds": cfg.skew_tolerance_seconds,
    }
    record["renewed"] = True
    state.handshakes[handshake_id] = record
    return 200, dict(record)


def _handshake(state: _State, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    cfg = state.cfg
    renew_of = request.get("renew_of")
    capability_hash = state.capability_hash()
    if renew_of:
        prior = state.handshakes.get(str(renew_of))
        if prior is None:
            return 404, {"error": "unknown_handshake", "reason": str(renew_of)}
        if prior["capability_hash"] != capability_hash:
            return 409, {
                "error": "capability_document_changed",
                "reason": "capability hash changed since acceptance; fail closed",
                "prior_capability_hash": prior["capability_hash"],
                "capability_hash": capability_hash,
            }
        return _renew(state, str(renew_of))
    clauses, obligations = _clause_verdicts(state, request)
    by_id = {clause["clause_id"]: clause for clause in clauses}
    rejected_mandatory = [
        clause_id
        for clause_id in MANDATORY_CLAUSES
        if by_id[clause_id]["verdict"] == "rejected"
    ]
    degraded = [clause["clause_id"] for clause in clauses if clause["verdict"] == "degraded"]
    requested_ids = tuple(dict.fromkeys(request.get("taskset", {}).get("task_ids") or ()))
    rows = requested_ids or cfg.task_ids
    resolution = [
        {
            "task_id": task_id,
            "content_digest": digest([cfg.taskset_id, cfg.taskset_version, task_id], length=32),
            "topology_ref": cfg.topology.topology_id,
            "task_family": cfg.task_family,
        }
        for task_id in rows
        if task_id in cfg.task_ids
    ]
    accept_degraded = set(request.get("accept_degraded") or ())
    unaccepted_degraded = [clause for clause in degraded if clause not in accept_degraded]
    accepted = not rejected_mandatory and not unaccepted_degraded
    state.counter += 1
    handshake_id = f"hs_{digest([cfg.container_id, state.counter, request], length=16)}"
    agreement_digest = _agreement_digest(
        state,
        request,
        handshake_id=handshake_id,
        capability_hash=capability_hash,
        clauses=clauses,
        obligations=obligations,
        resolution=resolution,
    )
    record = {
        "schema_version": HANDSHAKE_SCHEMA_VERSION,
        "handshake_id": handshake_id,
        "accepted": accepted,
        "clauses": clauses,
        "rejected_mandatory_clauses": rejected_mandatory,
        "degraded_clauses": degraded,
        "unaccepted_degraded_clauses": unaccepted_degraded,
        "obligations": obligations,
        "taskset_resolution": resolution,
        "capability_hash": capability_hash,
        "agreement_digest": agreement_digest,
        "expires_at": state.clock.rfc3339(cfg.handshake_ttl_seconds),
        "expires_at_offset": state.clock.now() + cfg.handshake_ttl_seconds,
        "clock": {
            "container_time": state.clock.rfc3339(cfg.clock_skew_seconds),
            "measured_skew_seconds": cfg.clock_skew_seconds,
            "tolerance_seconds": cfg.skew_tolerance_seconds,
        },
    }
    if accepted:
        state.handshakes[handshake_id] = record
    return 200, record


def _check_handshake(state: _State, body: Mapping[str, Any]) -> dict[str, Any] | None:
    handshake_id = body.get("handshake_id")
    if not handshake_id:
        raise _Refused(400, "handshake_absent", "no handshake_id on the attempt")
    record = state.handshakes.get(str(handshake_id))
    if record is None:
        raise _Refused(403, "handshake_unknown", f"unknown handshake {handshake_id!r}")
    if str(handshake_id) in state.revoked:
        raise _Refused(403, "handshake_revoked", f"handshake {handshake_id!r} was revoked")
    if state.clock.now() > float(record["expires_at_offset"]):
        raise _Refused(403, "handshake_expired", f"handshake {handshake_id!r} expired")
    supplied = body.get("agreement_digest")
    if supplied is not None and supplied != record["agreement_digest"]:
        raise _Refused(
            409,
            "agreement_digest_mismatch",
            "attempt agreement digest does not match its handshake",
        )
    return record


class _Refused(Exception):
    def __init__(self, status: int, code: str, reason: str) -> None:
        self.status = status
        self.code = code
        self.reason = reason
        super().__init__(reason)


# --------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------- #


def _seal(state: _State, attempt: _Attempt) -> None:
    """Build the sealed trace and the reward once, deterministically."""

    cfg = state.cfg
    probe = bool(attempt.policy_config.get("probe"))
    transport = str(attempt.policy_config.get("transport") or cfg.sampling_transport)
    topology = cfg.topology
    missing = cfg.defects.missing_instance_id
    calls: list[InferenceCall] = []
    instances: list[AgentInstance | None]
    if len(topology.agent_instances) == 1 and not topology.opponent_instances:
        instances = [topology.agent_instances[0]]
    else:
        instances = list(topology.agent_instances)
    live_ids: list[str] = []
    for instance in instances:
        if instance is not None and instance.agent_instance_id == missing:
            continue
        if instance is not None:
            live_ids.append(instance.agent_instance_id)
        calls.extend(
            _instance_calls(
                cfg, attempt=attempt, instance=instance, transport=transport, probe=probe
            )
        )
    if cfg.judge_spans:
        calls.append(_judge_call(cfg, attempt))

    channels = []
    for channel in topology.communication_channels:
        dropped = channel.channel_id == cfg.defects.dropped_channel_id
        channels.append(
            {
                "channel_id": channel.channel_id,
                "scope": channel.scope,
                "trainable_for_author": channel.trainable_for_author,
                "message_count": 0 if dropped else 2 * len(live_ids),
            }
        )

    call_payloads = [_call_payload(call) for call in calls]
    trace_digest = digest(
        {
            "rollout_id": attempt.rollout_id,
            "calls": call_payloads,
            "channels": channels,
            "container": cfg.container_id,
        },
        length=32,
    )
    episodes = _episodes(cfg, attempt, calls, trace_digest=trace_digest, probe=probe)
    context = _context_segments(calls)
    trace = {
        "rollout_id": attempt.rollout_id,
        "task_id": attempt.task_id,
        "sealed": True,
        "schema_version": "trace.v5",
        "trace_digest": trace_digest,
        "renderer_profile": _renderer_payload(cfg.renderer_profile),
        "wire_api": cfg.wire_api,
        "sampling_transport": transport,
        "probe": probe,
        "turn_model": topology.turn_model,
        "actuation_model": topology.actuation_model,
        "declared_channels": channels,
        "correlation": dict(attempt.correlation),
        "instances": [
            {
                "agent_instance_id": instance.agent_instance_id,
                "role_id": instance.role_id,
                "policy_type_id": instance.policy_type_id,
                "team_id": instance.team_id,
                "trainable": instance.trainable,
                "pinned_identity": instance.pinned_identity,
                "present": instance.agent_instance_id in live_ids,
                "absent_at_tick": None if instance.agent_instance_id in live_ids else 0,
                "last_live_tick": None if instance.agent_instance_id in live_ids else 0,
            }
            for instance in topology.agent_instances
        ],
        "calls": call_payloads,
        "episodes": [_episode_payload(episode) for episode in episodes],
        # Foreign authorship is explicit, not implied by a zero mask.
        "context_segments": [_segment_payload(segment) for segment in context],
    }
    state.traces[attempt.rollout_id] = trace
    scored_offset = max(0.0, state.clock.now() - attempt.admitted_at)
    record = _reward(cfg, attempt, trace_digest=trace_digest, scored_offset=scored_offset)
    state.rewards[attempt.rollout_id] = None if record is None else _reward_payload(record)


def _group_pin_fields(state: _State, attempt: _Attempt) -> dict[str, Any]:
    cfg = state.cfg
    revision = int(attempt.correlation.get("policy_revision") or 0)
    transport = str(attempt.policy_config.get("transport") or cfg.sampling_transport)
    handshake = state.handshakes.get(attempt.handshake_id, {})
    match_set = cfg.defects.match_set_drift or attempt.correlation.get("match_set_revision_id")
    return {
        "behavior_fingerprint": _behavior(cfg, revision, transport).value,
        "policy_revision": revision,
        "wire_api": cfg.wire_api,
        "sampling_transport": transport,
        "policy_kind": cfg.policy_kind,
        "model_family": cfg.model_family,
        "container_image_digest": cfg.image_digest,
        "container_contract_hash": cfg.contract_hash,
        "handshake_agreement_digest": handshake.get("agreement_digest", ""),
        "task_family": cfg.task_family,
        "topology_id": cfg.topology.topology_id,
        "policy_set_revision_id": attempt.correlation.get("policy_set_revision"),
        "match_set_revision_id": match_set,
    }


def _submit(state: _State, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    cfg = state.cfg
    record = _check_handshake(state, body)
    assert record is not None
    key = str(body.get("idempotency_key") or "")
    if not key:
        raise _Refused(400, "idempotency_key_required", "an attempt must carry a key")
    correlation = {
        name: body.get("correlation", {}).get(name)
        for name in CORRELATION_FIELDS
        if name in (body.get("correlation") or {})
    }
    if key in state.idempotency:
        rollout_id = state.idempotency[key]
        attempt = state.attempts[rollout_id]
        return 202, {
            "rollout_id": rollout_id,
            "state": state.states[rollout_id],
            "idempotent_replay": True,
            "lease_expires_at_offset": state.leases.get(rollout_id, 0.0),
            "correlation": dict(attempt.correlation),
        }
    active = sum(
        1 for rid, value in state.states.items() if value in {"queued", "running", "awaiting_score"}
    )
    if active >= cfg.advertised_concurrency:
        raise _Refused(
            429,
            "concurrency_exhausted",
            f"{active} active attempts at advertised concurrency {cfg.advertised_concurrency}",
        )
    task_id = str(body.get("task_id") or "")
    if task_id not in cfg.task_ids:
        raise _Refused(404, "unknown_task", f"task {task_id!r} is not in the taskset")
    config_id = str(body.get("policy_config_id") or body.get("policy_set_id") or "")
    policy_config = state.policy_configs.get(config_id) or state.policy_sets.get(config_id)
    if policy_config is None:
        raise _Refused(409, "policy_not_bound", f"policy binding {config_id!r} is not bound")
    state.counter += 1
    rollout_id = f"ro_{digest([cfg.container_id, key, state.counter], length=16)}"
    replaces = dict(body.get("replaces") or {})
    attempt = _Attempt(
        rollout_id=rollout_id,
        idempotency_key=key,
        task_id=task_id,
        correlation=dict(correlation),
        handshake_id=str(body["handshake_id"]),
        policy_config=dict(policy_config),
        admitted_at=state.clock.now(),
        replaced_attempt_id=replaces.get("attempt_id"),
        replacement_index=int(replaces.get("index") or 0),
        replacement_reason=replaces.get("reason"),
    )
    state.attempts[rollout_id] = attempt
    state.idempotency[key] = rollout_id
    state.states[rollout_id] = "running"
    state.polls[rollout_id] = 0
    state.leases[rollout_id] = state.clock.now() + cfg.lease_ttl_seconds
    state.emit(rollout_id, "admitted", correlation=dict(correlation))
    return 202, {
        "rollout_id": rollout_id,
        "state": "running",
        "accepted": True,
        "idempotent_replay": False,
        "lease_expires_at": state.clock.rfc3339(cfg.lease_ttl_seconds),
        "lease_expires_at_offset": state.leases[rollout_id],
        "handshake_id": attempt.handshake_id,
        "correlation": dict(attempt.correlation),
        "group_pin_fields": _group_pin_fields(state, attempt),
    }


def _advance(state: _State, rollout_id: str) -> dict[str, Any]:
    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    current = state.states[rollout_id]
    if current in {"completed", "failed", "cancelled"}:
        return _state_payload(state, rollout_id)
    state.polls[rollout_id] += 1
    if state.clock.now() > state.leases[rollout_id]:
        state.states[rollout_id] = "failed"
        state.refusals[rollout_id] = "lease_expired"
        state.emit(rollout_id, "failure", code="lease_expired")
        return _state_payload(state, rollout_id)
    if state.polls[rollout_id] >= cfg.polls_until_terminal and current == "running":
        try:
            _seal(state, attempt)
        except _Refusal as refusal:
            state.states[rollout_id] = "failed"
            state.refusals[rollout_id] = refusal.code
            state.emit(rollout_id, "failure", code=refusal.code, reason=refusal.reason)
            return _state_payload(state, rollout_id)
        if cfg.deferred_scoring:
            state.states[rollout_id] = "awaiting_score"
            state.emit(rollout_id, "horizon_reached")
        else:
            state.states[rollout_id] = "scored"
            state.emit(rollout_id, "scored")
    return _state_payload(state, rollout_id)


def _state_payload(state: _State, rollout_id: str) -> dict[str, Any]:
    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    trace = state.traces.get(rollout_id)
    rows = (trace or {}).get("instances", [])
    present = {row["agent_instance_id"] for row in rows if row["present"]}
    return {
        "rollout_id": rollout_id,
        "state": state.states[rollout_id],
        "terminal": state.states[rollout_id] in {"completed", "failed", "cancelled"},
        "terminal_count": state.terminals.get(rollout_id, 0),
        "failure_code": state.refusals.get(rollout_id),
        "lease_expires_at": state.clock.rfc3339(state.leases[rollout_id] - state.clock.now()),
        "lease_expires_at_offset": state.leases[rollout_id],
        "lease_expired": state.clock.now() > state.leases[rollout_id],
        "handshake_id": attempt.handshake_id,
        "correlation": dict(attempt.correlation),
        "group_pin_fields": _group_pin_fields(state, attempt),
        "instance_liveness": [
            {
                "agent_instance_id": instance.agent_instance_id,
                "live": (
                    instance.agent_instance_id in present
                    if trace
                    else instance.agent_instance_id != cfg.defects.missing_instance_id
                ),
            }
            for instance in cfg.topology.agent_instances
        ],
    }


def _receipt(state: _State, rollout_id: str) -> dict[str, Any]:
    """Identity plus digests. Never raw tokens, and never a second terminal."""

    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    trace = state.traces.get(rollout_id) or {}
    reward = state.rewards.get(rollout_id) or {}
    handshake = state.handshakes.get(attempt.handshake_id, {})
    revision = int(attempt.correlation.get("policy_revision") or 0)
    transport = str(attempt.policy_config.get("transport") or cfg.sampling_transport)
    return _receipt_payload(
        RolloutReceipt(
            rollout_id=rollout_id,
            proxy_request_id=f"prid_{digest([rollout_id, 'attempt'], length=12)}",
            group_id=str(attempt.correlation.get("group_id") or ""),
            sample_index=int(attempt.correlation.get("sample_index") or 0),
            policy_revision=revision,
            behavior_fingerprint=_behavior(cfg, revision, transport).value,
            terminal_status=state.states[rollout_id],
            trace_digest=str(trace.get("trace_digest") or ""),
            evidence_digest=digest(trace.get("calls") or [], length=32),
            reward_id=reward.get("reward_id"),
            handshake_id=attempt.handshake_id,
            agreement_digest=str(handshake.get("agreement_digest") or ""),
            agent_instance_id=attempt.correlation.get("agent_instance_id"),
            team_id=attempt.correlation.get("team_id"),
            probe=bool(attempt.policy_config.get("probe")),
            replaced_attempt_id=attempt.replaced_attempt_id,
            replacement_index=attempt.replacement_index,
            replacement_reason=attempt.replacement_reason,
            metadata={"container_id": cfg.container_id},
        )
    )


def _finalize(state: _State, rollout_id: str) -> dict[str, Any]:
    cfg = state.cfg
    attempt = state.attempts[rollout_id]
    if state.states[rollout_id] in {"completed", "failed", "cancelled"}:
        payload = _state_payload(state, rollout_id)
        payload["already_terminal"] = True
        payload["receipt"] = _receipt(state, rollout_id)
        return payload
    if rollout_id not in state.traces:
        _seal(state, attempt)
    state.states[rollout_id] = "completed"
    state.emit(rollout_id, "episode", trace_digest=state.traces[rollout_id]["trace_digest"])
    payload = _state_payload(state, rollout_id)
    payload["already_terminal"] = False
    quiesced = cfg.quiescence_supported and not cfg.defects.unquiesced_deferred_program
    payload["snapshot"] = {
        "horizon_kind": cfg.horizon.horizon_kind,
        "horizon_value": cfg.horizon.value,
        "clipped": (not cfg.quiescence_supported)
        and not cfg.defects.unquiesced_deferred_program,
        "quiescence_attested": quiesced,
        "agent_authored_programs_killed": quiesced,
        "taken_at_offset": state.clock.now() - attempt.admitted_at,
    }
    payload["trace_digest"] = state.traces[rollout_id]["trace_digest"]
    payload["receipt"] = _receipt(state, rollout_id)
    return payload


def _handle(
    state: _State,
    method: str,
    path: str,
    query: Mapping[str, list[str]],
    body: Any,
) -> tuple[int, Any]:
    cfg = state.cfg
    body = body if isinstance(body, dict) else {}
    segments = [part for part in path.split("/") if part]

    if method == "GET" and path == "/metadata":
        return 200, {
            "metadata": {
                "optimizer_contracts": {
                    "cispo": {"version": CONTRACT_VERSION, **dict(DECLARED_ROUTES)}
                }
            }
        }
    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "container_id": cfg.container_id,
            "container_version": cfg.taskset_version,
            "image_digest": cfg.image_digest,
            "contract_version": CONTRACT_VERSION,
        }
    if method == "GET" and path == "/training/capabilities":
        # The document is the body: the client hands the response straight to
        # ``CapabilityDocument.from_payload``, envelope-free.
        return 200, state.capability_document()
    if method == "POST" and path == "/training/handshake":
        return _handshake(state, body)
    if method == "GET" and path == "/taskset":
        return 200, {
            "taskset_id": cfg.taskset_id,
            "version": cfg.taskset_version,
            "splits": cfg.declared_splits,
            "task_family": cfg.task_family,
        }
    if method == "POST" and path == "/taskset/tasks":
        # ``ROUTE_METHODS`` declares this route POST: a row request carries a
        # list of ids and a split, not a query string.
        asked = body.get("ids")
        if isinstance(asked, str):
            asked = _csv([asked])
        requested = tuple(dict.fromkeys(tuple(str(item) for item in asked or ()) or cfg.task_ids))
        unknown = [task_id for task_id in requested if task_id not in cfg.task_ids]
        if unknown:
            return 404, {"error": "unknown_task", "reason": f"{unknown}"}
        return 200, {
            "rows": [
                {
                    "task_id": task_id,
                    "topology_ref": cfg.topology.topology_id,
                    "task_family": cfg.task_family,
                    "seed": index,
                    "content_digest": digest(
                        [cfg.taskset_id, cfg.taskset_version, task_id], length=32
                    ),
                }
                for index, task_id in enumerate(requested)
            ]
        }
    if method == "GET" and len(segments) == 2 and segments[0] == "topologies":
        if segments[1] != cfg.topology.topology_id:
            return 404, {"error": "unknown_topology", "reason": segments[1]}
        payload = _topology_payload(cfg)
        payload["minimum_viable_roster"] = {
            team.team_id: team.minimum_viable_roster for team in cfg.topology.teams
        }
        return 200, payload
    if method == "POST" and path == "/policy-configs":
        kind = str(body.get("kind") or "trainable")
        if kind == "probe" and not cfg.probe_binding_supported:
            return 409, {"error": "probe_unsupported", "reason": "no probe binding kind"}
        transport = str(body.get("transport") or cfg.sampling_transport)
        if transport == "tokens_in_tokens_out" and not cfg.tito_supported:
            return 409, {"error": "transport_unsupported", "reason": transport}
        if any(name in body for name in ("api_key", "bearer_token", "credential")):
            return 400, {"error": "embedded_credential", "reason": "credentials never inline"}
        state.counter += 1
        config_id = f"pc_{digest([cfg.container_id, state.counter, body], length=16)}"
        record = {
            "config_id": config_id,
            "kind": kind,
            "probe": kind == "probe",
            "transport": transport,
            "policy_revision": int(body.get("policy_revision") or 0),
            "renderer_profile": _renderer_payload(cfg.renderer_profile),
            "sampler_ready": True,
            "sampler_origin": f"/samplers/{config_id}",
            "immutable": True,
        }
        state.policy_configs[config_id] = record
        return 200, record
    if method == "POST" and path == "/policy-sets":
        bindings = list(body.get("bindings") or [])
        declared = {
            instance.agent_instance_id: instance for instance in cfg.topology.agent_instances
        }
        bound_ids = {str(item.get("agent_instance_id")) for item in bindings}
        if bound_ids != set(declared):
            return 409, {
                "error": "partial_roster_binding",
                "reason": "no episode may start with a half-bound roster",
                "missing": sorted(set(declared) - bound_ids),
                "unknown": sorted(bound_ids - set(declared)),
            }
        for item in bindings:
            instance = declared[str(item["agent_instance_id"])]
            ref = str(item.get("policy_ref") or "")
            if not instance.trainable and ref.strip().lower() in ALIAS_REFS:
                return 409, {
                    "error": "alias_opponent_binding",
                    "reason": f"opponent {instance.agent_instance_id} may not resolve {ref!r}",
                }
        state.counter += 1
        set_id = f"ps_{digest([cfg.container_id, state.counter, body], length=16)}"
        record = {
            "config_id": set_id,
            "policy_set_id": set_id,
            "policy_set_revision_id": str(body.get("policy_set_revision_id") or set_id),
            "kind": str(body.get("kind") or "trainable"),
            "probe": str(body.get("kind") or "") == "probe",
            "transport": str(body.get("transport") or cfg.sampling_transport),
            "policy_revision": int(body.get("policy_revision") or 0),
            "atomic": True,
            "bindings": [
                {
                    "agent_instance_id": instance.agent_instance_id,
                    "trainable": instance.trainable,
                    "parameter_group_id": cfg.topology.parameter_groups.get(
                        instance.policy_type_id
                    ),
                    "pinned_identity": instance.pinned_identity,
                    "renderer_profile": _renderer_payload(cfg.renderer_profile),
                }
                for instance in cfg.topology.agent_instances
            ],
        }
        state.policy_sets[set_id] = record
        return 200, record
    if method == "POST" and path == "/rollout":
        return _submit(state, body)

    if segments[:1] == ["rollouts"] and len(segments) >= 2:
        rollout_id = segments[1]
        if rollout_id not in state.attempts:
            return 404, {"error": "unknown_rollout", "reason": rollout_id}
        tail = segments[2:]
        if method == "GET" and not tail:
            return 200, _advance(state, rollout_id)
        if method == "GET" and tail == ["events"]:
            cursor = int((query.get("cursor") or ["0"])[0])
            log = state.events.get(rollout_id, [])
            return 200, {
                "events": [event for event in log if event["cursor"] > cursor],
                "next_cursor": log[-1]["cursor"] if log else cursor,
            }
        if method == "POST" and tail == ["renew"]:
            if not cfg.lease_renewable:
                return 409, {
                    "error": "lease_not_renewable",
                    "reason": "the container advertised a non-renewable lease",
                }
            if state.states[rollout_id] in {"completed", "failed", "cancelled"}:
                return 409, {"error": "not_renewable", "reason": state.states[rollout_id]}
            if state.clock.now() > state.leases[rollout_id]:
                return 409, {"error": "lease_expired", "reason": "renew after expiry"}
            state.leases[rollout_id] = state.clock.now() + cfg.lease_ttl_seconds
            state.emit(rollout_id, "lease_renewed")
            return 200, {
                "rollout_id": rollout_id,
                "lease_expires_at": state.clock.rfc3339(cfg.lease_ttl_seconds),
                "lease_expires_at_offset": state.leases[rollout_id],
            }
        if method == "POST" and tail == ["finalize"]:
            return 200, _finalize(state, rollout_id)
        if method == "POST" and tail == ["terminate"]:
            if state.states[rollout_id] in {"completed", "failed", "cancelled"}:
                payload = _state_payload(state, rollout_id)
                payload["already_terminal"] = True
                payload["receipt"] = _receipt(state, rollout_id)
                return 200, payload
            state.states[rollout_id] = "cancelled"
            state.refusals[rollout_id] = str(body.get("reason") or "cancelled")
            state.emit(rollout_id, "cancellation", reason=state.refusals[rollout_id])
            payload = _state_payload(state, rollout_id)
            payload["already_terminal"] = False
            payload["receipt"] = _receipt(state, rollout_id)
            return 200, payload
        if method == "GET" and tail == ["trace"]:
            trace = state.traces.get(rollout_id)
            if trace is None:
                return 409, {"error": "trace_not_sealed", "reason": state.states[rollout_id]}
            if cfg.artifact_by_reference:
                return 200, {
                    "rollout_id": rollout_id,
                    "inline": False,
                    "trace_digest": trace["trace_digest"],
                    "trace_ref": f"/rollouts/{rollout_id}/trace/body",
                }
            return 200, {"inline": True, **trace}
        if method == "GET" and tail == ["trace", "body"]:
            trace = state.traces.get(rollout_id)
            if trace is None:
                return 409, {"error": "trace_not_sealed", "reason": state.states[rollout_id]}
            return 200, {"inline": True, **trace}
        if method == "GET" and tail == ["artifacts"]:
            trace = state.traces.get(rollout_id)
            inventory = [
                {
                    "artifact_id": f"{rollout_id}:trace",
                    "role": "trace",
                    "digest": (trace or {}).get("trace_digest", ""),
                    "bytes": len(json.dumps(trace or {})),
                    "fetch_handle": f"/rollouts/{rollout_id}/trace/body",
                    "by_reference": cfg.artifact_by_reference,
                }
            ]
            if cfg.artifact_by_reference:
                inventory.append(
                    {
                        "artifact_id": f"{rollout_id}:recording",
                        "role": "recording",
                        "digest": digest([rollout_id, "recording"], length=32),
                        "bytes": 734003200,
                        "fetch_handle": f"/rollouts/{rollout_id}/trace/body",
                        "by_reference": True,
                    }
                )
            return 200, {"rollout_id": rollout_id, "artifacts": inventory}
        return 405, {"error": "route_not_declared", "reason": path}

    if path == "/reward" and method in {"GET", "POST"}:
        rollout_id = str(body.get("rollout_id") or (query.get("rollout_id") or [""])[0])
        if rollout_id not in state.attempts:
            return 404, {"error": "unknown_rollout", "reason": rollout_id}
        if cfg.deferred_scoring and state.states[rollout_id] != "completed":
            return 202, {
                "rollout_id": rollout_id,
                "state": "pending",
                "reason": "deferred verifier has not settled",
            }
        payload = state.rewards.get(rollout_id)
        if payload is None:
            if rollout_id not in state.traces:
                return 409, {"error": "reward_not_ready", "reason": state.states[rollout_id]}
            return 422, {
                "error": "missing_evidence",
                "reason": "no reward for a sealed trace; absent is not zero",
            }
        return 200, payload

    return 404, {"error": "route_not_declared", "reason": path}


def _csv(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    for value in values:
        out.extend(part for part in value.split(",") if part)
    return tuple(out)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _State

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        return

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "bad_json", "reason": "body is not JSON"})
            return
        with self.state.lock:
            self.state.request_log.append((method, parsed.path))
            try:
                status, payload = _handle(self.state, method, parsed.path, query, body)
            except _Refused as refused:
                status, payload = refused.status, {
                    "error": refused.code,
                    "reason": refused.reason,
                }
            except _Refusal as refusal:
                status, payload = 422, {"error": refusal.code, "reason": refusal.reason}
            except (RecordError, TopologyError) as error:
                status, payload = 500, {"error": "record_error", "reason": str(error)}
        self._respond(status, payload)

    def _respond(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")


class RunningContainer:
    """A live fake. ``base_url`` plus ``shutdown()`` is the whole contract."""

    __slots__ = ("config", "clock", "_state", "_server", "_thread")

    def __init__(self, config: ContainerConfig, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self._state = _State(config, clock)
        handler = type("_BoundHandler", (_Handler,), {"state": self._state})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def declared_routes(self) -> Mapping[str, str]:
        return dict(DECLARED_ROUTES)

    @property
    def requested_paths(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._state.request_log)

    def client(self, **kwargs: Any) -> ContainerClient:
        return ContainerClient(self.base_url, **kwargs)

    def set_reward_source(
        self,
        source: Mapping[Any, float] | Callable[[str, int], float] | None,
    ) -> None:
        """Re-declare the per-attempt measure this container will emit.

        A single constant makes every attempt in a group tie, and a tied group
        carries no ordering, so a caller that needs variance says so here
        rather than reaching into the container's private state.
        """

        with self._state.lock:
            self._state.cfg = replace(self._state.cfg, reward_value_by_sample=source)
            self.config = self._state.cfg

    def revoke_handshake(self, handshake_id: str) -> None:
        """The container may revoke a handshake when it degrades."""

        with self._state.lock:
            self._state.revoked.add(handshake_id)

    def bump_capability_epoch(self) -> str:
        """Change the capability document so a renewal must fail closed."""

        with self._state.lock:
            self._state.capability_epoch += 1
            return self._state.capability_hash()

    @property
    def attempt_count(self) -> int:
        """How many logical attempts the container actually admitted."""

        with self._state.lock:
            return len(self._state.attempts)

    def terminal_count(self, rollout_id: str) -> int:
        with self._state.lock:
            return self._state.terminals.get(rollout_id, 0)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "RunningContainer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()


def serve(config: ContainerConfig, *, clock: Clock | None = None) -> RunningContainer:
    """Start one fake container.

    The whole construction API: pass a configuration, get back a running
    server with ``base_url``, ``client()``, and ``shutdown()``. Bound to
    ``127.0.0.1`` on an ephemeral port; no outbound network, no real sleeping.
    """

    return RunningContainer(config, clock or Clock())
