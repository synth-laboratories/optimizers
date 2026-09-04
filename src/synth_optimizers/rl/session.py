"""The admitted container, for one run, over the declared contract client.

This is the executor's only door to a container. It performs the ordered
startup the design note requires -- health, metadata, capabilities and their
hash, taskset rows, handshake, renderer equality, probe -- and refuses to go
one step further than the container has agreed to. Nothing here creates a
provider session or issues a paid request: that is the caller's, and the whole
point of this module is that the caller cannot reach it early.

Two rules hold everywhere below:

* **A rejected mandatory clause stops the run here**, before a session exists
  and before a single token is paid for. The typed error carries the clause
  list so the receipt can name it.
* **Absent is never zero.** A missing reward, an unsealed trace, a probe
  episode that looks trainable: each raises. Degrading one of those into a
  neutral value is how a run optimizes nothing and still looks healthy.

No task, harness, environment or algorithm name appears in this module.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from ..contracts.rl_clauses import MANDATORY_CLAUSES
from ..contracts.rl_identity import GroupPin, RolloutReceipt, TaskSpec, Topology
from ..contracts.rl_records import (
    BehaviorFingerprint,
    CompactionProvenance,
    EvidenceError,
    HorizonEvidence,
    InferenceCall,
    RendererProfile,
    RewardChannel,
    RewardRecord,
    SamplingProfile,
    TrainableEpisode,
    TrainableSegment,
    assert_strict_prefix,
)
from .capabilities import (
    CapabilityDocument,
    ClauseResult,
    ExecutorRequirements,
    assert_preflight_passed,
    check_requirements,
)
from .config import RunConfig
from .contract import ContainerClient, ContainerContract, preflight_contract
from .handshake import (
    Agreement,
    ClauseRejected,
    HandshakeLedger,
    HandshakeRequest,
    HandshakeVerdict,
    Obligations,
    OptimizerIdentity,
    PolicyRequest,
    RunPlan,
    TopologyExpectation,
    build_request,
    evaluate_handshake,
    format_rfc3339,
)
from .ports import PortError, SamplerOrigin
from .probe import ProbeAttempt, ProbeReport, validate_probe
from .store import RunIdentity

SESSION_SCHEMA_VERSION = "cispo.session.v1"

#: How many times a degraded handshake may be lowered and re-sent before the
#: executor concludes the container and the run plan cannot meet.
MAX_HANDSHAKE_ATTEMPTS = 4

OPTIMIZER_NAME = "synth_optimizers.cispo"


class SessionError(PortError):
    """The container seam refused. Never degraded into a zero reward."""


class EvidenceNotReady(SessionError):
    """The attempt has not settled yet. Poll again; do not treat it as failed."""


class ProbeRefused(SessionError):
    """The unpaid evidence path could not be walked, so no paid one may be."""


# --------------------------------------------------------------------------- #
# Injected time
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RunClock:
    """Monotone seconds for the queues, wall time for the agreement. Never sleeps."""

    epoch: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    elapsed: float = 0.0

    def now(self) -> float:
        return self.elapsed

    def utc(self) -> datetime:
        return self.epoch + timedelta(seconds=self.elapsed)

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise SessionError("a monotone clock cannot go backwards")
        self.elapsed += float(seconds)
        return self.elapsed


@dataclass(slots=True)
class LiveRunClock:
    """Production clock: process-monotone durations and real UTC timestamps."""

    _monotonic_origin: float = field(default_factory=time.monotonic)
    _utc_origin: datetime = field(default_factory=lambda: datetime.now(UTC))

    def now(self) -> float:
        return time.monotonic() - self._monotonic_origin

    def utc(self) -> datetime:
        return self._utc_origin + timedelta(seconds=self.now())


# --------------------------------------------------------------------------- #
# Wire decoders: container payloads in, shared records out
# --------------------------------------------------------------------------- #


def _ints(payload: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value = payload.get(name) or ()
    return tuple(int(item) for item in value)


def call_from_payload(payload: Mapping[str, Any]) -> InferenceCall:
    """One per-call record, rebuilt from the sealed trace."""

    compaction = payload.get("compaction")
    return InferenceCall(
        call_id=str(payload["call_id"]),
        proxy_request_id=str(payload["proxy_request_id"]),
        rollout_id=str(payload["rollout_id"]),
        group_id=str(payload.get("group_id") or ""),
        sample_index=int(payload.get("sample_index") or 0),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        policy_revision=int(payload["policy_revision"]),
        wire_api=str(payload["wire_api"]),
        sampling_transport=str(payload["sampling_transport"]),
        token_capture_provenance=str(payload["token_capture_provenance"]),
        prompt_token_ids=_ints(payload, "prompt_token_ids"),
        generation_token_ids=_ints(payload, "generation_token_ids"),
        generation_logprobs=tuple(
            float(item) for item in payload.get("generation_logprobs") or ()
        ),
        sampled_mask=_ints(payload, "sampled_mask"),
        finish_reason=str(payload["finish_reason"]),
        stop_token_ids=_ints(payload, "stop_token_ids"),
        content_mask=_ints(payload, "content_mask"),
        renderer_profile_fingerprint=str(payload.get("renderer_profile_fingerprint") or ""),
        trainable=bool(payload.get("trainable", True)),
        author_kind=str(payload.get("author_kind") or "policy"),
        branch_id=str(payload.get("branch_id") or "root"),
        parent_branch_id=payload.get("parent_branch_id"),
        compaction=(
            CompactionProvenance(
                rule=str(compaction["rule"]),
                divergence_index=int(compaction["divergence_index"]),
                removed_message_indices=tuple(
                    int(item) for item in compaction.get("removed_message_indices") or ()
                ),
                authored_by_policy=bool(compaction.get("authored_by_policy")),
            )
            if compaction
            else None
        ),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        role_id=payload.get("role_id"),
        policy_type_id=payload.get("policy_type_id"),
        parameter_group_id=payload.get("parameter_group_id"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        effect_tick_start=payload.get("effect_tick_start"),
        effect_tick_end=payload.get("effect_tick_end"),
        wire_request=dict(payload.get("wire_request") or {}),
        wire_response=dict(payload.get("wire_response") or {}),
        usage=dict(payload.get("usage") or {}),
        created_at=str(payload.get("created_at") or ""),
    )


def segment_from_payload(payload: Mapping[str, Any]) -> TrainableSegment:
    return TrainableSegment(
        token_ids=_ints(payload, "token_ids"),
        loss_mask=_ints(payload, "loss_mask"),
        behavior_logprobs=tuple(float(item) for item in payload.get("behavior_logprobs") or ()),
        branch_id=str(payload.get("branch_id") or "root"),
        parameter_group_id=payload.get("parameter_group_id"),
        agent_instance_id=payload.get("agent_instance_id"),
        call_ids=tuple(str(item) for item in payload.get("call_ids") or ()),
        author_kind=str(payload.get("author_kind") or "policy"),
        role_id=payload.get("role_id"),
        policy_type_id=payload.get("policy_type_id"),
        team_id=payload.get("team_id"),
        policy_revision=payload.get("policy_revision"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        effect_tick_start=payload.get("effect_tick_start"),
        effect_tick_end=payload.get("effect_tick_end"),
    )


def episode_from_payload(payload: Mapping[str, Any]) -> TrainableEpisode:
    return TrainableEpisode(
        rollout_id=str(payload["rollout_id"]),
        task_id=str(payload["task_id"]),
        seed=int(payload.get("seed") or 0),
        policy_revision=int(payload["policy_revision"]),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        segments=tuple(segment_from_payload(row) for row in payload.get("segments") or ()),
        terminal_status=str(payload["terminal_status"]),
        usage=dict(payload.get("usage") or {}),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        policy_set_revision_id=payload.get("policy_set_revision_id"),
        root_rollout_id=payload.get("root_rollout_id"),
        trace_digest=str(payload.get("trace_digest") or ""),
        probe=bool(payload.get("probe")),
    )


def reward_from_payload(payload: Mapping[str, Any]) -> RewardRecord:
    horizon = payload.get("horizon")
    return RewardRecord(
        reward_id=str(payload["reward_id"]),
        rollout_id=str(payload["rollout_id"]),
        trace_digest=str(payload.get("trace_digest") or ""),
        channels=tuple(
            RewardChannel(
                channel_id=str(row["channel_id"]),
                team_id=row.get("team_id"),
                measure=float(row["measure"]),
                rank=row.get("rank"),
            )
            for row in payload.get("channels") or ()
        ),
        optimized_channel=str(payload["optimized_channel"]),
        terminal_status=str(payload["terminal_status"]),
        evaluation_plan_id=str(payload["evaluation_plan_id"]),
        horizon=(
            None
            if not horizon
            else HorizonEvidence(
                horizon_kind=str(horizon["horizon_kind"]),
                horizon_value=float(horizon["horizon_value"]),
                scored_at_offset_seconds=float(horizon["scored_at_offset_seconds"]),
                clipped=bool(horizon["clipped"]),
                quiescence_attested=bool(horizon["quiescence_attested"]),
                settlement_window_seconds=float(horizon.get("settlement_window_seconds") or 0.0),
                credited_settlement_seconds=float(
                    horizon.get("credited_settlement_seconds") or 0.0
                ),
            )
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def receipt_from_payload(payload: Mapping[str, Any]) -> RolloutReceipt:
    return RolloutReceipt(
        rollout_id=str(payload["rollout_id"]),
        proxy_request_id=str(payload["proxy_request_id"]),
        group_id=str(payload.get("group_id") or ""),
        sample_index=int(payload.get("sample_index") or 0),
        policy_revision=int(payload.get("policy_revision") or 0),
        behavior_fingerprint=str(payload["behavior_fingerprint"]),
        terminal_status=str(payload["terminal_status"]),
        trace_digest=str(payload.get("trace_digest") or ""),
        evidence_digest=str(payload.get("evidence_digest") or ""),
        reward_id=payload.get("reward_id"),
        handshake_id=str(payload.get("handshake_id") or ""),
        agreement_digest=str(payload.get("agreement_digest") or ""),
        agent_instance_id=payload.get("agent_instance_id"),
        team_id=payload.get("team_id"),
        probe=bool(payload.get("probe")),
        replaced_attempt_id=payload.get("replaced_attempt_id"),
        replacement_index=int(payload.get("replacement_index") or 0),
        replacement_reason=payload.get("replacement_reason"),
        metadata=dict(payload.get("metadata") or {}),
    )


# --------------------------------------------------------------------------- #
# Startup record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HandshakeExchange:
    """One requirement document and the verdict it drew, kept for the receipt."""

    request: Mapping[str, Any]
    verdict: Mapping[str, Any]
    outcome: str


@dataclass(frozen=True, slots=True)
class StartupRecord:
    """Everything the ordered startup learned, in the order it learned it."""

    health: Mapping[str, Any]
    metadata: Mapping[str, Any]
    contract: ContainerContract
    capability: CapabilityDocument
    clauses: tuple[ClauseResult, ...]
    taskset: Mapping[str, Any]
    task_rows: tuple[Mapping[str, Any], ...]
    exchanges: tuple[HandshakeExchange, ...]
    agreement: Agreement
    probe: ProbeReport | None
    probe_cost: float = 0.0
    renewals: tuple[Mapping[str, Any], ...] = ()
    revocations: tuple[Mapping[str, Any], ...] = ()

    @property
    def container_image_digest(self) -> str:
        return self.capability.container_image_digest

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "health": dict(self.health),
            "metadata": dict(self.metadata),
            "contract": {
                "version": self.contract.version,
                "contract_hash": self.contract.contract_hash,
                "routes": dict(self.contract.route_table.declared),
            },
            "capabilities": dict(self.capability.raw),
            "capability_hash": self.capability.content_hash,
            "container_image_digest": self.capability.container_image_digest,
            "executor_clauses": [item.to_payload() for item in self.clauses],
            "taskset": dict(self.taskset),
            "task_rows": [dict(row) for row in self.task_rows],
            "handshake": {
                "exchanges": [
                    {
                        "request": dict(item.request),
                        "verdict": dict(item.verdict),
                        "outcome": item.outcome,
                    }
                    for item in self.exchanges
                ],
                "agreement": self.agreement.to_receipt(),
                "renewals": [dict(item) for item in self.renewals],
                "revocations": [dict(item) for item in self.revocations],
            },
            "probe": (
                None
                if self.probe is None
                else {
                    **self.probe.to_payload(),
                    "trainable": False,
                    "cost": self.probe_cost,
                    "cost_attribution": "handshake_overhead",
                }
            ),
        }


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SubmittedAttempt:
    """What the container said when it accepted one attempt."""

    rollout_id: str
    idempotency_key: str
    policy_binding_id: str
    accepted: Mapping[str, Any]
    group_pin_fields: Mapping[str, Any] = field(default_factory=dict)


def _origin_payload(origin: SamplerOrigin) -> dict[str, Any]:
    """Send the origin field for field, not as a bare URL.

    A container checks that the path carries a per-attempt id, and it cannot
    decide that from a URL string without assuming a path layout — a global
    ``/v1`` would pass. It also stamps evidence with the behavior fingerprint
    the origin names, which reaches it nowhere else. The same dataclass exists
    on both sides; flattening it here threw away the two fields that make the
    binding checkable.
    """

    return {
        "base_url": origin.base_url,
        "credential": origin.credential,
        "policy_revision": origin.policy_revision,
        "behavior_fingerprint": origin.behavior_fingerprint,
        "proxy_request_id": origin.proxy_request_id,
        "wire_api": origin.wire_api,
        "sampling_transport": origin.sampling_transport,
        "expires_at": origin.expires_at,
    }


class ContractContainerSession:
    """A :class:`~.ports.ContainerSession` over the declared route surface."""

    def __init__(
        self,
        client: ContainerClient,
        *,
        config: RunConfig,
        startup: StartupRecord,
        ledger: HandshakeLedger,
        clock: RunClock,
        request_builder: HandshakeRequest,
    ) -> None:
        self._client = client
        self._config = config
        self._startup = startup
        self._ledger = ledger
        self._clock = clock
        self._request = request_builder
        self._agreement = startup.agreement
        self._submitted: dict[str, SubmittedAttempt] = {}
        self._by_key: dict[str, str] = {}
        self._renewals: list[Mapping[str, Any]] = list(startup.renewals)
        self._call_count = 0

    # -- identity ----------------------------------------------------------

    @property
    def handshake_id(self) -> str:
        return self._agreement.handshake_id

    @property
    def agreement_digest(self) -> str:
        return self._agreement.agreement_digest

    @property
    def agreement(self) -> Agreement:
        return self._agreement

    @property
    def obligations(self) -> Obligations:
        return self._agreement.obligations

    @property
    def capability(self) -> CapabilityDocument:
        return self._startup.capability

    @property
    def topology(self) -> Topology:
        return self._startup.capability.topology

    @property
    def startup(self) -> StartupRecord:
        return self._startup

    @property
    def submissions(self) -> Mapping[str, SubmittedAttempt]:
        return dict(self._submitted)

    def run_identity(self, run_id: str, *, plan_hash: str) -> RunIdentity:
        """The binding a resume must reproduce exactly or be refused."""

        return RunIdentity(
            run_id=run_id,
            container_contract_hash=self._startup.contract.contract_hash,
            container_image_digest=self._startup.capability.container_image_digest,
            algorithm_plan_hash=plan_hash,
            renderer_fingerprint=self._startup.capability.renderer_profile.fingerprint,
            handshake_agreement_digest=self._agreement.agreement_digest,
            capability_hash=self._startup.capability.content_hash,
        )

    # -- discovery ---------------------------------------------------------

    def tasks(self, *, split: str, task_ids: Sequence[str]) -> tuple[TaskSpec, ...]:
        """Task rows, each one carrying the digest the agreement resolved."""

        rows = {str(row["task_id"]): row for row in self._startup.task_rows}
        wanted = tuple(task_ids) or tuple(rows)
        specs: list[TaskSpec] = []
        for index, task_id in enumerate(wanted):
            row = rows.get(task_id)
            if row is None:
                raise SessionError(f"task {task_id!r} is not a row of the resolved taskset")
            digest = self._agreement.task_digest(task_id)
            declared = str(row.get("content_digest") or "")
            if declared and declared != digest:
                raise SessionError(
                    f"task {task_id!r} row digest {declared} does not match the agreed "
                    f"digest {digest}; the taskset moved under the agreement"
                )
            specs.append(
                TaskSpec(
                    task_id=task_id,
                    split=split,
                    seed=int(row.get("seed") if row.get("seed") is not None else index),
                    group_id="",
                    task_family=str(row.get("task_family") or ""),
                    content_digest=digest,
                    topology_ref=row.get("topology_ref"),
                    tags={"index": index},
                )
            )
        return tuple(specs)

    # -- attempts ----------------------------------------------------------

    def bind(
        self,
        origins: Mapping[str, SamplerOrigin],
        *,
        pin: GroupPin,
        probe: bool = False,
    ) -> Mapping[str, Any]:
        """Bind one policy, or the whole declared roster, for one attempt.

        A trainable instance is routed to the origin of its own parameter group;
        a non-trainable one keeps the immutable identity the container declared.
        No credential is ever inlined: the per-attempt identity lives in the
        origin's path, and that is what is bound.
        """

        topology = self.topology
        kind = "probe" if probe else "trainable"
        first = next(iter(origins.values()))
        if len(topology.agent_instances) < 2:
            return self._client.bind_policy(
                {
                    "kind": kind,
                    "policy_revision": first.policy_revision,
                    "transport": first.sampling_transport,
                    "wire_api": first.wire_api,
                    # The container stamps its evidence with this identity, and
                    # a probe has no origin to carry it in, so it is named at
                    # the top level for both kinds rather than only nested.
                    "behavior_fingerprint": first.behavior_fingerprint,
                    "model_family": self._config.model.family,
                    "model_id": self._config.model.id,
                    "sampler_origin": _origin_payload(first),
                    "policy_ref": first.credential,
                    "handshake_id": self.handshake_id,
                    "agreement_digest": self.agreement_digest,
                }
            )
        bindings = []
        for instance in topology.agent_instances:
            if instance.trainable:
                group = topology.parameter_group_for(instance.agent_instance_id)
                # Multi-agent rosters need one conversation route per seat,
                # even when two seats share weights.  Sharing the parameter
                # group's route makes the second seat look like an edited
                # history of the first seat's conversation.
                origin = origins.get(instance.agent_instance_id) or origins.get(group, first)
                policy_ref = origin.credential
                instance_origin = _origin_payload(origin)
            else:
                instance_origin = None
                policy_ref = instance.pinned_identity or ""
                if not policy_ref:
                    raise SessionError(
                        f"opponent {instance.agent_instance_id!r} declares no pinned "
                        "identity; an unpinned opponent is not a reproducible sample"
                    )
            bindings.append(
                {
                    "agent_instance_id": instance.agent_instance_id,
                    "policy_ref": policy_ref,
                    "sampler_origin": instance_origin,
                    "trainable": instance.trainable,
                }
            )
        return self._client.bind_policy_set(
            {
                "kind": kind,
                "policy_revision": first.policy_revision,
                "transport": first.sampling_transport,
                "behavior_fingerprint": first.behavior_fingerprint,
                "model_family": self._config.model.family,
                "model_id": self._config.model.id,
                "policy_set_revision_id": pin.policy_set_revision_id or "policy-set-0",
                "match_set_revision_id": pin.match_set_revision_id,
                "bindings": bindings,
                "handshake_id": self.handshake_id,
                "agreement_digest": self.agreement_digest,
            }
        )

    def submit(
        self,
        task: TaskSpec,
        origin: SamplerOrigin,
        *,
        pin: GroupPin,
        sample_index: int,
        idempotency_key: str,
    ) -> str:
        """One attempt, one origin. Idempotent by key."""

        return self.submit_roster(
            task,
            {"": origin},
            pin=pin,
            sample_index=sample_index,
            idempotency_key=idempotency_key,
        )

    def submit_roster(
        self,
        task: TaskSpec,
        origins: Mapping[str, SamplerOrigin],
        *,
        pin: GroupPin,
        sample_index: int,
        idempotency_key: str,
        agent_instance_id: str | None = None,
        team_id: str | None = None,
        probe: bool = False,
    ) -> str:
        """One attempt with one origin per trainable parameter group."""

        if not origins:
            raise SessionError("an attempt needs at least one bound sampler origin")
        self._ledger.assert_admissible(
            self.handshake_id, self.agreement_digest, now=self._clock.utc()
        )
        for origin in origins.values():
            if origin.behavior_fingerprint != pin.behavior_fingerprint:
                raise SessionError(
                    "sampler origin behavior fingerprint "
                    f"{origin.behavior_fingerprint} does not match the group pin's "
                    f"{pin.behavior_fingerprint}"
                )
            if origin.policy_revision != pin.policy_revision:
                raise SessionError(
                    f"sampler origin is revision {origin.policy_revision}, the pin is "
                    f"{pin.policy_revision}"
                )
        binding = self.bind(origins, pin=pin, probe=probe)
        binding_id = str(binding.get("config_id") or binding.get("policy_set_id") or "")
        if not binding_id:
            raise SessionError("container returned no policy binding id")
        request = {
            "task_id": task.task_id,
            "idempotency_key": idempotency_key,
            "policy_config_id": binding_id,
            "handshake_id": self.handshake_id,
            "agreement_digest": self.agreement_digest,
            "correlation": {
                "run_id": pin.run_id,
                "group_id": pin.group_id,
                "sample_index": int(sample_index),
                "seed": int(task.seed),
                "policy_revision": int(pin.policy_revision),
                "agent_instance_id": agent_instance_id,
                "team_id": team_id,
                "policy_set_revision": pin.policy_set_revision_id,
                "match_set_revision_id": pin.match_set_revision_id,
            },
        }
        accepted = self._client.submit_rollout(request)
        rollout_id = str(accepted.get("rollout_id") or "")
        if not rollout_id:
            raise SessionError("container accepted an attempt without naming a rollout id")
        known = self._by_key.get(idempotency_key)
        if known is not None and known != rollout_id:
            raise SessionError(
                f"idempotency key {idempotency_key!r} produced a second logical attempt: "
                f"{known} then {rollout_id}"
            )
        self._by_key[idempotency_key] = rollout_id
        # An idempotent replay answers with less than the acceptance did, so the
        # first acceptance is the record kept.
        self._submitted.setdefault(
            rollout_id,
            SubmittedAttempt(
                rollout_id=rollout_id,
                idempotency_key=idempotency_key,
                policy_binding_id=binding_id,
                accepted=dict(accepted),
                group_pin_fields=dict(accepted.get("group_pin_fields") or {}),
            ),
        )
        return rollout_id

    def poll(self, rollout_id: str) -> Mapping[str, Any]:
        return self._client.rollout_state(rollout_id)

    def reward_payload(self, rollout_id: str) -> Mapping[str, Any]:
        """The raw reward receipt. Pending is a state, not a zero."""

        return self._client.reward(rollout_id)

    def events(self, rollout_id: str, *, cursor: str | None = None) -> Mapping[str, Any]:
        return self._client.rollout_events(rollout_id, cursor=cursor)

    def renew(self, rollout_id: str) -> Mapping[str, Any]:
        payload = self._client.renew_rollout(rollout_id, {"handshake_id": self.handshake_id})
        self._renewals.append({"rollout_id": rollout_id, **dict(payload)})
        return payload

    def finalize(self, rollout_id: str) -> Mapping[str, Any]:
        return self._client.finalize_rollout(rollout_id, {"handshake_id": self.handshake_id})

    def terminate(self, rollout_id: str, *, reason: str) -> RolloutReceipt:
        payload = self._client.terminate_rollout(
            rollout_id, {"reason": reason, "handshake_id": self.handshake_id}
        )
        receipt = payload.get("receipt")
        if not isinstance(receipt, Mapping):
            raise SessionError(f"terminating {rollout_id} sealed no receipt")
        return receipt_from_payload(receipt)

    # -- evidence ----------------------------------------------------------

    def trace(self, rollout_id: str) -> Mapping[str, Any]:
        payload = self._client.trace(rollout_id)
        if payload.get("inline") is False:
            reference = str(payload.get("trace_ref") or "")
            fetch = getattr(self._client, "fetch_reference", None)
            if not reference or fetch is None:
                raise SessionError(
                    f"rollout {rollout_id} stored its trace by reference at "
                    f"{reference!r} and this client cannot fetch a reference"
                )
            body = fetch(reference)
            if body.get("trace_digest") != payload.get("trace_digest"):
                raise SessionError(
                    "trace reference digest does not match its inventory entry"
                )
            return body
        return payload

    def evidence(self, rollout_id: str) -> tuple[TrainableEpisode, RewardRecord]:
        """The sealed episode and its reward, both validated before they leave."""

        trace = self.trace(rollout_id)
        if not trace.get("sealed"):
            raise EvidenceNotReady(f"rollout {rollout_id} has not sealed its trace")
        trace_digest = str(trace.get("trace_digest") or "")
        if not trace_digest:
            raise SessionError(f"rollout {rollout_id} sealed a trace with no digest")
        episode = self._episode(rollout_id, trace)
        reward = self._reward(rollout_id, trace_digest)
        return self._align_team(episode, reward), reward

    def _align_team(
        self, episode: TrainableEpisode, reward: RewardRecord
    ) -> TrainableEpisode:
        """Drop a team the reward does not measure separately.

        A single-team topology names its team on every trajectory while its
        reward carries one untargeted channel. Carrying the team forward would
        send the credit estimator looking for a per-team channel that does not
        exist, so an untargeted optimized channel means the team is not a
        comparison key for this episode. A team the reward *does* split on is
        always kept.
        """

        if episode.team_id is None:
            return episode
        if any(channel.team_id == episode.team_id for channel in reward.channels):
            return episode
        optimized = next(
            (
                channel
                for channel in reward.channels
                if channel.channel_id == reward.optimized_channel
            ),
            None,
        )
        if optimized is not None and optimized.team_id is None:
            return replace(episode, team_id=None)
        raise SessionError(
            f"rollout {episode.rollout_id} names team {episode.team_id!r} but its reward "
            f"carries no channel for that team and no untargeted optimized channel"
        )

    def _episode(self, rollout_id: str, trace: Mapping[str, Any]) -> TrainableEpisode:
        rows = trace.get("episodes") or ()
        episodes = [episode_from_payload(row) for row in rows]
        if not episodes:
            raise SessionError(f"rollout {rollout_id} sealed no trainable episode")
        calls = [call_from_payload(row) for row in trace.get("calls") or ()]
        self._call_count += len(calls)
        self._validate_calls(rollout_id, calls)
        merged = self._merge(rollout_id, episodes, trace)
        if merged.probe:
            raise SessionError(
                f"rollout {rollout_id} is probe evidence; a probe may never enter a group"
            )
        try:
            merged.validate()
        except EvidenceError as error:
            raise SessionError(f"rollout {rollout_id} evidence is not admissible: {error}") from (
                error
            )
        return merged

    def _validate_calls(self, rollout_id: str, calls: Sequence[InferenceCall]) -> None:
        by_instance: dict[str | None, list[InferenceCall]] = {}
        for call in calls:
            by_instance.setdefault(call.agent_instance_id, []).append(call)
        for stream in by_instance.values():
            for previous, following in zip(stream, stream[1:], strict=False):
                if previous.branch_id != following.branch_id:
                    continue
                try:
                    assert_strict_prefix(previous, following)
                except EvidenceError as error:
                    raise SessionError(
                        f"rollout {rollout_id} breaks the strict-prefix rule: {error}"
                    ) from error
        for call in calls:
            if not call.trainable:
                continue
            try:
                call.validate_for_training()
            except EvidenceError as error:
                raise SessionError(
                    f"rollout {rollout_id} call {call.call_id} is not trainable evidence: "
                    f"{error}"
                ) from error

    def _merge(
        self,
        rollout_id: str,
        episodes: Sequence[TrainableEpisode],
        trace: Mapping[str, Any],
    ) -> TrainableEpisode:
        """One rollout is one episode, whatever the roster size.

        A joint episode arrives as one trajectory per policy-authored instance.
        The batch is built per parameter group from segment authorship, so the
        instances are merged here rather than in the assembler, which would
        otherwise have to be told what a roster is.
        """

        if len(episodes) == 1:
            return episodes[0]
        segments: list[TrainableSegment] = []
        usage: dict[str, Any] = {"instances": len(episodes)}
        prompt = 0
        completion = 0
        for episode in episodes:
            if episode.rollout_id != rollout_id:
                raise SessionError(
                    f"trace of {rollout_id} carries a trajectory for {episode.rollout_id}"
                )
            segments.extend(episode.segments)
            prompt += int(episode.usage.get("prompt_tokens") or 0)
            completion += int(episode.usage.get("completion_tokens") or 0)
        usage["prompt_tokens"] = prompt
        usage["completion_tokens"] = completion
        teams = {episode.team_id for episode in episodes if episode.team_id}
        if len(teams) > 1:
            raise SessionError(
                f"rollout {rollout_id} mixes trainable teams {sorted(teams)}; a group is "
                "one comparison on one team"
            )
        head = episodes[0]
        return TrainableEpisode(
            rollout_id=rollout_id,
            task_id=head.task_id,
            seed=head.seed,
            policy_revision=head.policy_revision,
            behavior_fingerprint=head.behavior_fingerprint,
            segments=tuple(segments),
            terminal_status=head.terminal_status,
            usage=usage,
            agent_instance_id=None,
            team_id=next(iter(teams)) if teams else None,
            policy_set_revision_id=head.policy_set_revision_id,
            root_rollout_id=rollout_id,
            trace_digest=str(trace.get("trace_digest") or ""),
            probe=any(episode.probe for episode in episodes),
        )

    def _reward(self, rollout_id: str, trace_digest: str) -> RewardRecord:
        payload = self._client.reward(rollout_id)
        state = str(payload.get("state") or "")
        if state == "pending" or "reward_id" not in payload:
            raise EvidenceNotReady(
                f"rollout {rollout_id} has no settled reward receipt yet; "
                "an absent reward is never a zero"
            )
        record = reward_from_payload(payload)
        try:
            record.validate(episode_trace_digest=trace_digest)
        except EvidenceError as error:
            raise SessionError(
                f"rollout {rollout_id} reward receipt is not admissible: {error}"
            ) from error
        if self.obligations.quiescence and (
            record.horizon is None or not record.horizon.quiescence_attested
        ):
            raise SessionError(
                f"rollout {rollout_id} reward carries no quiescence attestation, which "
                "the agreement obliged"
            )
        return record

    # -- agreement maintenance ---------------------------------------------

    def rehandshake(self) -> Agreement:
        """Re-read the capability document and re-verify the agreement.

        Used by resume. A changed capability hash, contract or agreement digest
        fails closed: that is a new run with a lineage edge, not a continuation.
        """

        payload = _capability_payload(self._client.capabilities())
        capability = CapabilityDocument.from_payload(payload)
        capability.assert_unchanged(self._startup.capability.content_hash)
        verdict = HandshakeVerdict.from_payload(
            self._client.handshake({**self._request.to_payload(), "renew_of": self.handshake_id})
        )
        agreement = self._ledger.renew(
            self.handshake_id,
            capability=capability,
            verdict=verdict,
            now=self._clock.utc(),
        )
        self._agreement = agreement
        self._renewals.append(
            {
                "handshake_id": agreement.handshake_id,
                "agreement_digest": agreement.agreement_digest,
                "capability_hash": agreement.capability_hash,
                "expires_at": format_rfc3339(agreement.expires_at),
                "renewed_at": format_rfc3339(self._clock.utc()),
            }
        )
        return agreement

    def receipt(self) -> dict[str, Any]:
        payload = self._startup.to_receipt()
        payload["handshake"]["renewals"] = [dict(item) for item in self._renewals]
        payload["calls_observed"] = self._call_count
        return payload


# --------------------------------------------------------------------------- #
# Ordered startup
# --------------------------------------------------------------------------- #


def _capability_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The capability document, whether or not it is served in an envelope."""

    inner = payload.get("capabilities")
    return inner if isinstance(inner, Mapping) else payload


def _requirements(
    config: RunConfig,
    document: CapabilityDocument,
    renderer_profile: RendererProfile,
) -> ExecutorRequirements:
    """What this run needs, in the capability module's own vocabulary."""

    horizon = config.pipeline.expected_horizon_seconds
    return ExecutorRequirements(
        renderer_profile=renderer_profile,
        min_concurrency=max(1, config.pipeline.max_execution_slots),
        horizon_seconds=horizon if horizon is not None else float(document.horizon.value),
        optimized_channel=config.reward.optimized_channel,
        sampling_transport=config.model.sampling_transport,
        wire_api=config.model.wire_api,
        split=config.taskset.train_split,
        expected_taskset_id=config.taskset.taskset_id,
        expected_topology_ref=config.topology.expected_topology_id,
        minimum_viable_roster=config.topology.minimum_viable_roster,
        require_quiescence=config.reward.require_quiescence,
        require_settlement_window=config.reward.require_settlement_window,
    )


def _run_plan(config: RunConfig, document: CapabilityDocument) -> RunPlan:
    plan = config.expanded_plan()
    horizon = config.pipeline.expected_horizon_seconds
    return RunPlan(
        group_size=plan.rollout.cardinality,
        groups_per_step=plan.groups_per_step,
        max_execution_slots=config.pipeline.max_execution_slots,
        maximum_policy_lag=config.pipeline.maximum_policy_lag,
        target_train_updates=config.plan.target_train_updates,
        expected_horizon_seconds=(
            horizon if horizon is not None else float(document.horizon.value)
        ),
    )


def _trainable_teams(config: RunConfig, document: CapabilityDocument) -> tuple[str, ...]:
    declared = config.topology.trainable_teams
    if declared:
        return declared
    return tuple(team.team_id for team in document.topology.teams if team.trainable)


def start_session(
    client: ContainerClient,
    config: RunConfig,
    *,
    renderer_profile: RendererProfile,
    clock: RunClock,
    optimizer_version: str = "0.2.20",
    sampling: SamplingProfile | None = None,
    probe_runner: "Callable[..., ProbeReport] | None" = None,
) -> ContractContainerSession:
    """Health, metadata, capabilities, tasks, handshake, renderer, probe -- in order.

    Returns an admitted session, or raises before a provider session could ever
    have been created. Nothing in this function spends money.
    """

    health = client.health()
    metadata = client.metadata()
    contract = preflight_contract(metadata)

    capability_payload = _capability_payload(client.capabilities())
    document = CapabilityDocument.from_payload(capability_payload)
    requirements = _requirements(config, document, renderer_profile)
    clauses = check_requirements(document, requirements, contract=contract)
    assert_preflight_passed(clauses)

    taskset = client.taskset()
    # A training run consumes the train rows; the paired-evaluation command
    # consumes the held-out rows through this same admitted session. Resolve
    # both allowlists up front so a disjoint held-out set is actually reachable,
    # while preserving first-seen order and never exposing either row's gold.
    task_ids = tuple(
        dict.fromkeys((*config.taskset.train_ids, *config.taskset.evaluation_ids))
    )
    rows_payload = client.taskset_tasks(
        {"ids": list(task_ids), "split": config.taskset.train_split}
    )
    rows = tuple(dict(row) for row in rows_payload.get("rows") or ())
    if not rows:
        raise SessionError("container resolved no taskset rows for this run")
    resolved_ids = tuple(str(row["task_id"]) for row in rows)

    request = build_request(
        run_id=config.run_id,
        optimizer=OptimizerIdentity(name=OPTIMIZER_NAME, version=optimizer_version),
        policy=PolicyRequest(
            provider=config.model.provider,
            model_id=config.model.id,
            transport=config.model.sampling_transport,
        ),
        requirements=requirements,
        topology=TopologyExpectation(
            expected_topology_id=(
                config.topology.expected_topology_id or document.topology.topology_id
            ),
            trainable_teams=_trainable_teams(config, document),
            partial_roster=config.topology.partial_roster,
        ),
        run_plan=_run_plan(config, document),
        task_ids=task_ids or resolved_ids,
        taskset_id=str(taskset.get("taskset_id") or document.discovery.taskset_id),
        now=clock.utc(),
    )

    exchanges: list[HandshakeExchange] = []
    ledger = HandshakeLedger(clock=clock.utc)
    agreement: Agreement | None = None
    for _attempt in range(MAX_HANDSHAKE_ATTEMPTS):
        payload = client.handshake(request.to_payload())
        verdict = HandshakeVerdict.from_payload(payload)
        decision = evaluate_handshake(
            request,
            verdict,
            capability=document,
            contract=contract,
            executor_clauses=clauses,
            now=clock.utc(),
        )
        exchanges.append(
            HandshakeExchange(
                request=request.to_payload(), verdict=dict(payload), outcome=decision.outcome
            )
        )
        if decision.outcome == "renegotiate":
            if decision.next_request is None:  # pragma: no cover - guarded upstream
                raise SessionError("renegotiation produced no lowered run plan")
            request = decision.next_request
            continue
        agreement = ledger.admit(decision)
        break
    if agreement is None:
        raise ClauseRejected(
            (
                ClauseResult(
                    clause_id="lifecycle.concurrency",
                    verdict="rejected",
                    reason=(
                        "the run plan could not be lowered to a plan the container "
                        f"accepts within {MAX_HANDSHAKE_ATTEMPTS} handshakes"
                    ),
                ),
            )
        )

    # 6. Renderer-profile equality against the profile the training session uses.
    document.renderer_profile.assert_matches(renderer_profile)

    startup = StartupRecord(
        health=dict(health),
        metadata=dict(metadata),
        contract=contract,
        capability=document,
        clauses=clauses,
        taskset=dict(taskset),
        task_rows=rows,
        exchanges=tuple(exchanges),
        agreement=agreement,
        probe=None,
    )
    session = ContractContainerSession(
        client,
        config=config,
        startup=startup,
        ledger=ledger,
        clock=clock,
        request_builder=request,
    )

    runner = probe_runner or run_probe
    report = runner(
        session,
        config=config,
        renderer_profile=renderer_profile,
        sampling=sampling or SamplingProfile(),
    )
    session._startup = _with_probe(startup, report)  # noqa: SLF001 - same module
    return session


def _with_probe(startup: StartupRecord, report: ProbeReport) -> StartupRecord:
    return StartupRecord(
        health=startup.health,
        metadata=startup.metadata,
        contract=startup.contract,
        capability=startup.capability,
        clauses=startup.clauses,
        taskset=startup.taskset,
        task_rows=startup.task_rows,
        exchanges=startup.exchanges,
        agreement=startup.agreement,
        probe=report,
        probe_cost=0.0,
    )


# --------------------------------------------------------------------------- #
# The probe: the whole evidence path, at zero provider cost
# --------------------------------------------------------------------------- #


def _probe_origin(config: RunConfig, behavior: BehaviorFingerprint) -> SamplerOrigin:
    """A synthetic origin. The probe binding returns canned generations."""

    return SamplerOrigin(
        base_url="probe://local",
        credential=f"probe/{config.run_id}",
        policy_revision=0,
        behavior_fingerprint=behavior.value,
        proxy_request_id=f"probe::{config.run_id}",
        wire_api=config.model.wire_api,
        sampling_transport=config.model.sampling_transport,
    )


def _probe_pin(config: RunConfig, session: ContractContainerSession, behavior: str) -> GroupPin:
    capability = session.capability
    return GroupPin(
        group_id=f"probe::{config.run_id}",
        run_id=config.run_id,
        algorithm_plan_hash=config.expanded_plan().plan_hash,
        behavior_fingerprint=behavior,
        policy_revision=0,
        wire_api=config.model.wire_api,
        sampling_transport=config.model.sampling_transport,
        policy_kind=config.model.policy_kind,
        model_family=config.model.family,
        container_image_digest=capability.container_image_digest,
        container_contract_hash=session.startup.contract.contract_hash,
        handshake_agreement_digest=session.agreement_digest,
        task_family=str(session.startup.task_rows[0].get("task_family") or ""),
        cardinality=1,
        topology_id=capability.topology.topology_id,
    )


def run_probe(
    session: ContractContainerSession,
    *,
    config: RunConfig,
    renderer_profile: RendererProfile,
    sampling: SamplingProfile,
) -> ProbeReport:
    """Walk submit, state, events, renew, trace, reward, finalize, terminate.

    Plus an idempotent resubmit of the same key and one cancellation. Validates
    shape, not quality, and refuses to let the run continue when the container
    declares no probe binding: a paid canary is a decision the operator makes,
    not one this function makes on their behalf.
    """

    capability = session.capability
    policy_block = capability.raw.get("policy")
    supported = True
    if isinstance(policy_block, Mapping):
        supported = bool(policy_block.get("probe_binding", True))
    if not supported:
        raise ProbeRefused(
            "container declares no probe policy binding; the evidence path cannot be "
            "walked at zero cost and this run refuses to spend on a canary implicitly"
        )
    behavior = BehaviorFingerprint(
        renderer_profile=renderer_profile,
        model_family=config.model.family,
        model_id=config.model.id,
        policy_revision=0,
        wire_api=config.model.wire_api,
        sampling_transport=config.model.sampling_transport,
        sampling=sampling,
    )
    pin = _probe_pin(config, session, behavior.value)
    origin = _probe_origin(config, behavior)
    task = session.tasks(split=config.taskset.train_split, task_ids=())[0]
    key = f"probe::{config.run_id}::0"
    operations: set[str] = set()

    rollout_id = session.submit_roster(
        task, {"probe": origin}, pin=pin, sample_index=0, idempotency_key=key, probe=True
    )
    operations.add("submit")

    cursors: list[int] = []
    state: Mapping[str, Any] = {}
    for _poll in range(8):
        state = session.poll(rollout_id)
        operations.add("state")
        events = session.events(rollout_id)
        operations.add("events")
        rows = events.get("events") or ()
        if rows:
            cursors.append(int(rows[-1]["cursor"]))
        if str(state.get("state")) in {"scored", "awaiting_score"} or state.get("terminal"):
            break
    session.renew(rollout_id)
    operations.add("renew")
    finalized = session.finalize(rollout_id)
    trace = session.trace(rollout_id)
    operations.add("trace")
    reward_payload = session.reward_payload(rollout_id)
    operations.add("reward")
    operations.add("finalize")

    events = session.events(rollout_id)
    rows = events.get("events") or ()
    if rows:
        cursors.append(int(rows[-1]["cursor"]))
    terminal_states = {"episode", "failure", "cancellation"}
    terminal_kinds = tuple(
        str(row["kind"]) for row in rows if str(row["kind"]) in terminal_states
    )

    resubmit = session.submit_roster(
        task, {"probe": origin}, pin=pin, sample_index=0, idempotency_key=key, probe=True
    )
    operations.add("idempotent_resubmit")

    cancel_key = f"{key}::cancel"
    cancelled = session.submit_roster(
        task,
        {"probe": origin},
        pin=pin,
        sample_index=0,
        idempotency_key=cancel_key,
        probe=True,
    )
    session.terminate(cancelled, reason="probe_cancellation")
    operations.add("cancellation")
    operations.add("terminate")

    calls = [call_from_payload(row) for row in trace.get("calls") or ()]
    episodes = [episode_from_payload(row) for row in trace.get("episodes") or ()]
    if not episodes:
        raise ProbeRefused("probe attempt sealed no episode")
    # A joint probe is validated on one instance's stream: prefix consistency is
    # a property of one conversation, not of a roster.
    episode = episodes[0]
    instance = episode.agent_instance_id
    stream = [call for call in calls if call.agent_instance_id == instance] or calls
    attempt = ProbeAttempt(
        rollout_id=rollout_id,
        behavior=behavior,
        calls=tuple(stream),
        episode=episode,
        reward=reward_from_payload(reward_payload),
        event_cursors=tuple(cursors),
        terminal_results=terminal_kinds or ("episode",),
        operations=frozenset(operations),
        resubmit_rollout_id=resubmit,
        cancelled_rollout_id=cancelled,
        trace_digest=str(trace.get("trace_digest") or ""),
        metadata={"finalize": dict(finalized), "state": dict(state)},
    )
    return validate_probe(
        attempt,
        expected_profile=renderer_profile,
        quiescence_accepted=session.obligations.quiescence,
    )


def mandatory_clause_ids() -> tuple[str, ...]:
    """The clause set every requirement document must name."""

    return tuple(MANDATORY_CLAUSES)
