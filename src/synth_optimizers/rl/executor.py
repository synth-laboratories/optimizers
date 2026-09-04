"""The loop: admission, queues, training and artifacts, driven as one run.

Everything this module does is generic. The algorithm arrives as an expanded
:class:`~.plan.AlgorithmPlan`; the container arrives as a
:class:`~.ports.ContainerSession`; the provider arrives as a
:class:`~.ports.PolicyBinder`; the renderer and the token capture arrive as a
:class:`~.ports.SamplerGateway`. There is no branch on a task, a harness, an
environment or an algorithm name anywhere below, and a second preset runs
through the identical code path.

The order is the design note's, and the order is the point:

1. Ordered startup, in :mod:`.session`: health, metadata, capabilities and
   hash, taskset rows, handshake, renderer equality, probe. A rejected
   mandatory clause stops here, before a provider session exists.
2. Register the run's binding in the durable journal.
3. Register the baseline revision through the binder, before anything is
   admitted -- a rollout may not reference a revision the catalog has not seen.
4. Admit per sample under a group pin, never whole groups.
5. Dispatch, submit, poll, renew, finalize, read evidence, validate, admit to
   the scored queue, complete groups.
6. At the train dequeue gate: recheck staleness *now*, build the batch through
   the assembler, train once per parameter group, publish atomically, and bind
   the new revision for every later group.
7. Emit the receipt directory the note requires.

Pause, drain, resume and stop are first-class at every one of those
boundaries, and resume re-handshakes before a single attempt is re-admitted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.rl_identity import GroupPin, TaskSpec
from ..contracts.rl_records import RewardRecord, TrainableEpisode
from . import assembly, credit
from .assembly import AssemblyError, EvidenceBundle, TrainingBatch
from .config import RunConfig
from .leases import LeaseBook, LeaseSizing, StragglerPolicy
from .lifecycle import RunLifecycle
from .plan import AlgorithmPlan
from .ports import (
    AttemptFacts,
    PolicyBinder,
    PolicyRevision,
    SamplerGateway,
    SamplerOrigin,
    TrainOutcome,
)
from .queues import AttemptRequest, GateRejection, QueueCapacities, QueueEngine, QueuePolicy
from .session import ContractContainerSession, EvidenceNotReady, RunClock, SessionError
from .store import (
    GROUP_TRAIN_READY,
    JournalStore,
    RunIdentity,
)

EXECUTOR_SCHEMA_VERSION = "cispo.executor.v1"

#: Every file the note's "Required run artifacts" list resolves to, mapped from
#: the bullet it satisfies. The manifest written at the end of a run carries
#: this map, so a reader can check the list off against the directory.
RUN_ARTIFACTS: Mapping[str, str] = {
    "effective redacted configuration and expanded plan with its hash": "effective_config.json",
    "the group pin for every group": "group_pins.jsonl",
    "rejected-group records with the field that caused the rejection": "rejected_groups.jsonl",
    "lifecycle transition log with the re-handshake performed at each resume": "lifecycle.jsonl",
    "replay source runs, accepted staleness and comparison": "replay.json",
    "container metadata, contract, capability response and hash": "container.json",
    "the handshake pair, obligations, digests, expiry, skew, renewals": "handshake.json",
    "probe or canary validation record, marked non-trainable, with its cost": "probe.json",
    "container/image digest and relevant repository commits": "provenance.json",
    "baseline and trained policy revisions or policy-set manifests": "policy_revisions.json",
    "append-only checkpoint catalog": "checkpoint_catalog.jsonl",
    "sampler-weight and training-state references with digests": "checkpoint_artifacts.jsonl",
    "checkpoint lineage edges": "checkpoint_lineage.jsonl",
    "verified resume source resolution": "resume_resolution.json",
    "independently verified resume artifact identity": "resume_artifact_identity.json",
    "evaluation manifests referencing immutable ids": "evaluation_manifest.json",
    "resolved topology, channels, rosters and partial-roster disposition": "topology.json",
    "match-set manifest naming every opponent's pinned identity": "match_set.json",
    "horizon, scored-read time, clipping, settlement, quiescence": "horizon.jsonl",
    "per-instance liveness ledger": "instance_liveness.jsonl",
    "per-team reward channels with measure, rank and optimized channel": "team_rewards.jsonl",
    "renderer profile, transport, prompt budget and compaction spans": "renderer.json",
    "queue transition journal and aggregate queue metrics": "queue_journal.jsonl",
    "aggregate queue metrics": "queue_metrics.json",
    "group membership, rewards, advantages, staleness and skip decisions": "groups.jsonl",
    "provider usage, training-token counts, cost and request ids": "provider_usage.json",
    "sampling TPS by call and weighted aggregate": "sampling_tps.json",
    "container reward receipts": "reward_receipts.jsonl",
    "sealed Trace V5 references or bundles": "traces.jsonl",
    "paired baseline/trained evaluation rows and summary": "evaluation_rows.json",
    "cleanup receipt listing what was removed and retained": "cleanup.json",
}

#: Reasons a run stops. None of them is an exception path.
STOP_REASONS: tuple[str, ...] = (
    "target_train_updates_reached",
    "sampled_group_budget_exhausted",
    "drained",
    "stopped",
    "no_progress",
)


class ExecutorError(RuntimeError):
    """The loop refused to continue. Never degraded into an empty update."""


# --------------------------------------------------------------------------- #
# Records the loop keeps for its own receipt
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AttemptEvidence:
    """One completed attempt: what it cost, what it proved, when it was read."""

    attempt_id: str
    rollout_id: str
    group_id: str
    sample_index: int
    task_id: str
    episode: TrainableEpisode
    reward: RewardRecord
    reward_payload: Mapping[str, Any]
    trace_digest: str
    instance_liveness: tuple[Mapping[str, Any], ...]
    submitted_at: float
    scored_at: float
    usage: Mapping[str, Any]

    @property
    def generated_tokens(self) -> int:
        return int(self.usage.get("completion_tokens") or 0)

    @property
    def seconds(self) -> float:
        return max(self.scored_at - self.submitted_at, 0.0)


@dataclass(frozen=True, slots=True)
class GroupOutcome:
    """One group's whole story, whatever happened to it."""

    group_id: str
    pin: GroupPin
    disposition: str
    staleness: int | None = None
    rewards: tuple[float, ...] = ()
    advantages: tuple[float, ...] = ()
    zero_variance: bool = False
    skipped: bool = False
    members: tuple[str, ...] = ()
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "pin": asdict(self.pin),
            "pin_digest": self.pin.pin_digest,
            "disposition": self.disposition,
            "staleness": self.staleness,
            "rewards": list(self.rewards),
            "advantages": list(self.advantages),
            "zero_variance": self.zero_variance,
            "skipped": self.skipped,
            "members": list(self.members),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class UpdateRecord:
    """One published update: what was packed, trained, and published."""

    update_id: str
    round_index: int
    group_ids: tuple[str, ...]
    parameter_groups: tuple[str, ...]
    steps: int
    outcomes: Mapping[str, TrainOutcome]
    revisions: Mapping[str, PolicyRevision]
    advantage_digest: str
    composition_digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "round_index": self.round_index,
            "group_ids": list(self.group_ids),
            "parameter_groups": list(self.parameter_groups),
            "provider_steps": self.steps,
            "advantage_digest": self.advantage_digest,
            "composition_digest": self.composition_digest,
            "outcomes": {
                name: {
                    "request_ids": list(outcome.request_ids),
                    "examples": outcome.examples,
                    "tokens": outcome.tokens,
                    "provider_cost": outcome.provider_cost,
                    "metrics": dict(outcome.metrics),
                }
                for name, outcome in self.outcomes.items()
            },
            "revisions": {
                name: _revision_payload(revision) for name, revision in self.revisions.items()
            },
        }


@dataclass(frozen=True, slots=True)
class RunReport:
    """What the run did, in numbers a caller can assert on."""

    run_id: str
    plan_hash: str
    stop_reason: str
    updates: tuple[UpdateRecord, ...]
    groups: tuple[GroupOutcome, ...]
    sampled_groups: int
    receipt_directory: Path
    final_revisions: Mapping[str, PolicyRevision]
    lifecycle_state: str

    @property
    def trained_groups(self) -> tuple[str, ...]:
        return tuple(item.group_id for item in self.groups if item.disposition == "trained")

    @property
    def skipped_groups(self) -> tuple[str, ...]:
        return tuple(item.group_id for item in self.groups if item.disposition == "skipped")

    @property
    def stale_groups(self) -> tuple[str, ...]:
        return tuple(item.group_id for item in self.groups if item.disposition == "stale")


def _revision_payload(revision: PolicyRevision) -> dict[str, Any]:
    return {
        "revision": revision.revision,
        "revision_id": revision.revision_id,
        "checkpoint_id": revision.checkpoint_id,
        "parameter_group_id": revision.parameter_group_id,
        "sampler_reference": revision.sampler_reference,
        "behavior_fingerprint": revision.behavior_fingerprint,
        "training_state_reference": revision.training_state_reference,
        "policy_set_revision_id": revision.policy_set_revision_id,
        "metadata": dict(revision.metadata),
    }


# --------------------------------------------------------------------------- #
# The executor
# --------------------------------------------------------------------------- #


class ContainerRunExecutor:
    """One run of the container-first plane, from baseline to receipt."""

    def __init__(
        self,
        *,
        config: RunConfig,
        session: ContractContainerSession,
        gateway: SamplerGateway,
        binder: PolicyBinder,
        clock: RunClock,
        receipts: str | Path,
        catalog_rows: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        lineage_rows: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self.plan: AlgorithmPlan = config.expanded_plan()
        self.session = session
        self.gateway = gateway
        self.binder = binder
        self.clock = clock
        self.receipts = Path(receipts)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self._catalog_rows = catalog_rows
        self._lineage_rows = lineage_rows

        self.run_id = config.run_id
        self.identity: RunIdentity = session.run_identity(
            self.run_id, plan_hash=self.plan.plan_hash
        )
        self.store = JournalStore(self.receipts / "queue_journal.sqlite3", clock=clock)
        self.store.register_run(self.identity)
        self.lifecycle = RunLifecycle(
            self.store,
            self.run_id,
            terminate=self._terminate_attempt,
            rehandshake=self._rehandshake,
            clock=clock,
        )
        horizon = session.capability.horizon
        sizing = LeaseSizing(
            heartbeat_interval_seconds=config.pipeline.heartbeat_interval_seconds,
            missed_heartbeats_allowed=config.pipeline.missed_heartbeats_allowed,
            quiescence_seconds=config.pipeline.quiescence_seconds,
            artifact_collection_seconds=config.pipeline.artifact_collection_seconds,
            grace_seconds=config.reward.horizon_grace_seconds,
            seconds_per_unit=None,
        )
        self.leases = LeaseBook(
            self.store,
            horizon=horizon,
            sizing=sizing,
            straggler=StragglerPolicy(
                max_replacements=config.pipeline.straggler_max_replacements
            ),
            clock=clock,
        )
        obliged = max(1, session.obligations.max_concurrency)
        self.queues = QueueEngine(
            self.store,
            run_id=self.run_id,
            policy=QueuePolicy(
                capacities=QueueCapacities(
                    rollout=config.pipeline.rollout_queue_capacity,
                    score=config.pipeline.score_queue_capacity,
                    scored_result=config.pipeline.scored_result_queue_capacity,
                    train_ready=config.pipeline.train_ready_capacity,
                ),
                max_staleness=config.pipeline.maximum_policy_lag,
                max_in_flight=min(config.pipeline.max_execution_slots, obliged),
                max_open_groups=config.pipeline.max_open_groups,
                stale_disposition=config.pipeline.stale_disposition,
            ),
            leases=self.leases,
            lifecycle=self.lifecycle,
        )

        self.tasks: tuple[TaskSpec, ...] = session.tasks(
            split=config.taskset.train_split, task_ids=config.taskset.train_ids
        )
        if not self.tasks:
            raise ExecutorError("the run resolved no task rows to sample")

        self.revisions: dict[str, PolicyRevision] = {}
        self.baseline_revisions: dict[str, PolicyRevision] = {}
        self.current_revision = 0
        self.sampled_groups = 0
        self.updates: list[UpdateRecord] = []
        self.group_outcomes: list[GroupOutcome] = []
        self.evidence: dict[str, AttemptEvidence] = {}
        self._pins: dict[str, GroupPin] = {}
        self._rollouts: dict[str, str] = {}
        self._attempts: dict[str, str] = {}
        self._origins: dict[str, tuple[str, ...]] = {}
        self._pending_declarations: dict[str, tuple[Mapping[str, SamplerOrigin], TaskSpec]] = {}
        self._submitted_at: dict[str, float] = {}
        self._pending_groups: list[str] = []
        self._pending_recycled: list[GateRejection] = []
        self._recycled: list[Mapping[str, Any]] = []
        self._rehandshakes: list[Mapping[str, Any]] = []
        self._stop_reason = ""
        self._round_index = 0

    # -- parameter groups --------------------------------------------------

    @property
    def parameter_groups(self) -> tuple[str, ...]:
        """Every trainable parameter group the container's topology declares."""

        topology = self.session.topology
        declared = topology.trainable_parameter_groups()
        if declared:
            return declared
        return ("pg_solo",)

    # -- baseline ----------------------------------------------------------

    def register_baseline(self) -> Mapping[str, PolicyRevision]:
        """Catalog the baseline through the binder before anything is admitted."""

        if self.revisions:
            return dict(self.revisions)
        for parameter_group in self.parameter_groups:
            revision = self.binder.baseline(
                run_id=self.run_id, parameter_group_id=parameter_group
            )
            if revision.parameter_group_id != parameter_group:
                raise ExecutorError(
                    f"binder returned a baseline for {revision.parameter_group_id!r} when "
                    f"{parameter_group!r} was asked for"
                )
            self.revisions[parameter_group] = revision
            self.baseline_revisions[parameter_group] = revision
        self.current_revision = min(
            revision.revision for revision in self.revisions.values()
        )
        return dict(self.revisions)

    # -- lifecycle ---------------------------------------------------------

    def pause(self, *, reason: str = "operator") -> str:
        return self.lifecycle.pause(reason=reason)

    def drain(self, *, reason: str = "operator") -> str:
        return self.lifecycle.drain(reason=reason)

    def resume(self) -> str:
        return self.lifecycle.resume()

    def stop(self, *, reason: str = "operator") -> str:
        report = self.lifecycle.stop(reason=reason)
        self._stop_reason = self._stop_reason or "stopped"
        return report.to_state

    def _terminate_attempt(self, attempt_id: str) -> None:
        rollout_id = self._rollouts.get(attempt_id)
        if rollout_id is None:
            return
        self.session.terminate(rollout_id, reason="lifecycle_terminate")
        self._close_origin(attempt_id)

    def _rehandshake(self) -> RunIdentity:
        """Resume re-handshakes first, then verifies the binding it came back with."""

        agreement = self.session.rehandshake()
        identity = self.session.run_identity(self.run_id, plan_hash=self.plan.plan_hash)
        self._rehandshakes.append(
            {
                "at": self.clock.now(),
                "handshake_id": agreement.handshake_id,
                "agreement_digest": agreement.agreement_digest,
                "capability_hash": agreement.capability_hash,
                "binding_digest": identity.binding_digest,
            }
        )
        return identity

    # -- admission ---------------------------------------------------------

    def _pin_for(self, group_id: str, task: TaskSpec) -> GroupPin:
        capability = self.session.capability
        revision = self.revisions[self.parameter_groups[0]]
        return GroupPin(
            group_id=group_id,
            run_id=self.run_id,
            algorithm_plan_hash=self.plan.plan_hash,
            behavior_fingerprint=revision.behavior_fingerprint,
            policy_revision=revision.revision,
            wire_api=self.config.model.wire_api,
            sampling_transport=self.config.model.sampling_transport,
            policy_kind=self.config.model.policy_kind,
            model_family=self.config.model.family,
            container_image_digest=capability.container_image_digest,
            container_contract_hash=self.session.startup.contract.contract_hash,
            handshake_agreement_digest=self.session.agreement_digest,
            task_family=task.task_family,
            cardinality=self.plan.rollout.cardinality,
            policy_set_revision_id=revision.policy_set_revision_id,
            match_set_revision_id=self.config.opponents.match_set_revision,
            topology_id=capability.topology.topology_id,
            # A roster-wide group pin names the immutable policy-set revision.
            # Its component revision id is only meaningful for a one-component
            # set; carrying the first component's id into every gateway route
            # makes a valid second parameter group look like a rebind.
            policy_revision_id=(
                revision.revision_id if len(self.parameter_groups) == 1 else None
            ),
        )

    def _next_task(self) -> TaskSpec:
        return self.tasks[self.sampled_groups % len(self.tasks)]

    def admit_group(self) -> str | None:
        """Open one group and admit every one of its samples, per sample."""

        if not self.lifecycle.gates.admit:
            return None
        if self.sampled_groups >= self.config.maximum_sampled_groups:
            return None
        if len(self.queues.open_groups()) >= self.config.pipeline.max_open_groups:
            return None
        task = self._next_task()
        index = self.sampled_groups
        group_id = f"{self.run_id}::g{index:04d}"
        pin = self._pin_for(group_id, task)
        self._pins[group_id] = pin
        for sample_index in range(pin.cardinality):
            request = AttemptRequest(
                idempotency_key=f"{group_id}::s{sample_index}",
                pin=pin,
                sample_index=sample_index,
                task_id=task.task_id,
                # The seed identifies the declared task instance. Group
                # samples repeat that same instance; sample_index,
                # idempotency, and the sampler provide rollout diversity.
                seed=task.seed,
                metadata={"task_family": task.task_family, "split": task.split},
            )
            self.queues.admit(request)
        self.sampled_groups += 1
        return group_id

    def _readmit_recycled(self, rejection: GateRejection) -> str | None:
        """A recycled group returns slots, not tasks: re-admit under a fresh pin.

        The slots wait when there is no room for another open group; they are
        never re-minted, because the queue engine may not invent a task
        identity and neither may this loop.
        """

        if self.sampled_groups >= self.config.maximum_sampled_groups:
            return None
        if len(self.queues.open_groups()) >= self.config.pipeline.max_open_groups:
            return None
        source = self._pins[rejection.group_id]
        task = next(
            (item for item in self.tasks if item.task_id == rejection.slots[0].task_id),
            self._next_task(),
        )
        index = self.sampled_groups
        group_id = f"{self.run_id}::g{index:04d}"
        pin = self._pin_for(group_id, task)
        self._pins[group_id] = pin
        for slot in rejection.slots:
            self.queues.admit(
                AttemptRequest(
                    idempotency_key=f"{group_id}::s{slot.sample_index}",
                    pin=pin,
                    sample_index=slot.sample_index,
                    task_id=slot.task_id,
                    seed=slot.seed,
                    metadata={"recycled_from": source.group_id},
                )
            )
        self.sampled_groups += 1
        self._recycled.append(
            {
                "from_group": rejection.group_id,
                "to_group": group_id,
                "slots": [asdict(slot) for slot in rejection.slots],
                "staleness": rejection.staleness,
            }
        )
        return group_id

    # -- dispatch ----------------------------------------------------------

    def _origins_for(
        self, pin: GroupPin, *, attempt_id: str, sample_index: int, task: TaskSpec
    ) -> Mapping[str, SamplerOrigin]:
        """One origin per trainable parameter group, bound before submission.

        The attempt facts name the executor's own attempt id: the container has
        not minted a rollout id yet, and cannot, because the origin is what it
        will sample through.
        """

        facts = AttemptFacts(rollout_id=attempt_id, task_id=task.task_id, seed=task.seed)
        origins: dict[str, SamplerOrigin] = {}
        trainable_instances = tuple(
            item for item in self.session.topology.agent_instances if item.trainable
        )
        route_keys = (
            tuple((item.agent_instance_id, self.session.topology.parameter_group_for(
                item.agent_instance_id
            )) for item in trainable_instances)
            if len(self.session.topology.agent_instances) > 1
            else tuple((group, group) for group in self.revisions)
        )
        for route_key, parameter_group in route_keys:
            revision = self.revisions[parameter_group]
            proxy_request_id = f"{pin.group_id}::s{sample_index}::{route_key}"
            origins[route_key] = self.gateway.bind(
                revision,
                pin=pin,
                sample_index=sample_index,
                proxy_request_id=proxy_request_id,
                attempt=facts,
            )
        return origins

    def dispatch_once(self) -> int:
        """Send what the queue says may go now. Never a whole group at a time."""

        ready = self.queues.next_dispatch(limit=self.config.pipeline.max_execution_slots)
        sent = 0
        for attempt in ready:
            pin = self._pins[attempt.group_id]
            task = next(item for item in self.tasks if item.task_id == attempt.task_id)
            task = TaskSpec(
                task_id=task.task_id,
                split=task.split,
                seed=attempt.seed,
                group_id=attempt.group_id,
                task_family=task.task_family,
                content_digest=task.content_digest,
                topology_ref=task.topology_ref,
                tags=task.tags,
            )
            origins = self._origins_for(
                pin,
                attempt_id=attempt.attempt_id,
                sample_index=attempt.sample_index,
                task=task,
            )
            self.queues.dispatch(attempt.attempt_id, holder="executor")
            try:
                rollout_id = self.session.submit_roster(
                    task,
                    origins,
                    pin=pin,
                    sample_index=attempt.sample_index,
                    idempotency_key=attempt.idempotency_key,
                )
            except SessionError as error:
                self.queues.fail(attempt.attempt_id, reason=f"submit_refused: {error}")
                continue
            # Declaring while the provider call holds its route lock would
            # serialize dispatch. Settle the provisional ID after completion.
            self._pending_declarations[attempt.attempt_id] = (origins, task)
            self._rollouts[attempt.attempt_id] = rollout_id
            self._attempts[rollout_id] = attempt.attempt_id
            self._origins[attempt.attempt_id] = tuple(
                origin.proxy_request_id for origin in origins.values()
            )
            self._submitted_at[attempt.attempt_id] = self.clock.now()
            sent += 1
        return sent

    def _declare(
        self, origins: Mapping[str, SamplerOrigin], *, rollout_id: str, task: TaskSpec
    ) -> None:
        """Hand the container's rollout id back to a gateway that wants it.

        ``bind`` happens before ``submit`` -- the origin is what is submitted --
        so the rollout id arrives by this second door, which the port declares.
        """

        for origin in origins.values():
            self.gateway.declare_attempt(
                origin.proxy_request_id,
                rollout_id=rollout_id,
                task_id=task.task_id,
                seed=task.seed,
            )

    def _close_origin(self, attempt_id: str) -> None:
        self._pending_declarations.pop(attempt_id, None)
        proxy_request_ids = self._origins.pop(attempt_id, ())
        for proxy_request_id in proxy_request_ids:
            self.gateway.close(proxy_request_id)

    # -- progress ----------------------------------------------------------

    def progress_once(self) -> int:
        """Poll, renew, finalize and validate every attempt the container holds."""

        moved = 0
        for attempt in self.queues.in_flight():
            rollout_id = self._rollouts.get(attempt.attempt_id)
            if rollout_id is None:
                continue
            state = self.session.poll(rollout_id)
            name = str(state.get("state") or "")
            if not state.get("terminal") and name not in {"scored", "awaiting_score"}:
                self._heartbeat(attempt.attempt_id, rollout_id)
                continue
            if name == "awaiting_score" and attempt.state == "running":
                self.queues.report_awaiting_score(attempt.attempt_id)
                self.session.finalize(rollout_id)
                moved += 1
                continue
            if attempt.state == "running":
                self.session.finalize(rollout_id)
            if self._accept(attempt.attempt_id, rollout_id, state):
                moved += 1
        return moved

    def _heartbeat(self, attempt_id: str, rollout_id: str) -> None:
        lease = self.leases.lease_for(attempt_id)
        if lease is None:
            return
        due = lease.expires_at - self.leases.heartbeat_ttl_seconds / 2
        if self.clock.now() < due:
            return
        self.session.renew(rollout_id)
        self.queues.heartbeat(attempt_id)

    def _accept(self, attempt_id: str, rollout_id: str, state: Mapping[str, Any]) -> bool:
        try:
            episode, reward = self.session.evidence(rollout_id)
        except EvidenceNotReady:
            return False
        except SessionError as error:
            self.queues.report_scored(attempt_id, payload={"rollout_id": rollout_id})
            self.queues.reject_evidence(attempt_id, reason=f"invalid_evidence: {error}")
            self._close_origin(attempt_id)
            return True
        attempt = self.store.attempt(attempt_id)
        pending = self._pending_declarations.pop(attempt_id, None)
        if pending is not None:
            origins, task = pending
            self._declare(origins, rollout_id=rollout_id, task=task)
        payload = dict(self.session.reward_payload(rollout_id))
        self.queues.report_scored(attempt_id, payload={"rollout_id": rollout_id})
        self.evidence[attempt_id] = AttemptEvidence(
            attempt_id=attempt_id,
            rollout_id=rollout_id,
            group_id=attempt.group_id,
            sample_index=attempt.sample_index,
            task_id=attempt.task_id,
            episode=episode,
            reward=reward,
            reward_payload=payload,
            trace_digest=episode.trace_digest,
            instance_liveness=tuple(
                dict(row) for row in state.get("instance_liveness") or ()
            ),
            submitted_at=self._submitted_at.get(attempt_id, self.clock.now()),
            scored_at=self.clock.now(),
            usage=dict(episode.usage),
        )
        self.queues.accept_evidence(
            attempt_id,
            payload={
                "rollout_id": rollout_id,
                "trace_digest": episode.trace_digest,
                "reward_id": reward.reward_id,
            },
        )
        self._close_origin(attempt_id)
        return True

    # -- training ----------------------------------------------------------

    def _bundles(self, group_id: str, staleness: int) -> tuple[EvidenceBundle, ...]:
        pin = self._pins[group_id]
        topology = self.session.topology
        bundles: list[EvidenceBundle] = []
        for attempt in self.queues.group_members(group_id):
            record = self.evidence.get(attempt.attempt_id)
            if record is None:
                raise ExecutorError(
                    f"group {group_id} left the gate without evidence for "
                    f"{attempt.attempt_id}"
                )
            bundles.append(
                EvidenceBundle(
                    group_id=group_id,
                    sample_index=attempt.sample_index,
                    pin=pin,
                    episode=record.episode,
                    reward=record.reward,
                    root_rollout_id=record.rollout_id,
                    topology=topology if topology.is_multi_policy else None,
                    staleness_steps=staleness,
                    source_run_id=self.run_id,
                )
            )
        return tuple(bundles)

    def _preview_credit(self, bundles: Sequence[EvidenceBundle]) -> credit.GroupCredit:
        """The group's own advantage vector, so a skip is decided before packing.

        The assembler recomputes this from the trainable spans it admits; this
        preview exists so a group that carries no ordering can be recorded and
        replaced instead of raising out of the assembler.
        """

        samples = []
        for bundle in bundles:
            channel = _resolved_channel(bundle.reward, bundle.episode.team_id)
            tokens = sum(
                segment.trainable_tokens
                for segment in bundle.episode.segments
                if segment.author_kind == "policy"
            )
            samples.append(
                credit.CreditSample(
                    sample_key=bundle.episode.rollout_id,
                    reward=channel.measure,
                    length=tokens,
                    reward_channel_id=channel.channel_id,
                    team_id=bundle.episode.team_id,
                )
            )
        return credit.estimate(self.plan.credit, samples)

    def train_once(self) -> UpdateRecord | None:
        """Dequeue, gate on staleness, pack, train, publish. One update or none."""

        if not self.lifecycle.gates.train:
            return None
        while len(self._pending_groups) < self.plan.groups_per_step:
            outcome = self.queues.train_dequeue(current_policy_revision=self.current_revision)
            for rejection in outcome.rejected:
                self._record_stale(rejection)
            if outcome.released is None:
                break
            group_id = outcome.released.group_id
            staleness = int(outcome.staleness or 0)
            bundles = self._bundles(group_id, staleness)
            preview = self._preview_credit(bundles)
            if preview.skipped:
                self.group_outcomes.append(
                    GroupOutcome(
                        group_id=group_id,
                        pin=self._pins[group_id],
                        disposition="skipped",
                        staleness=staleness,
                        rewards=preview.rewards,
                        advantages=preview.advantages,
                        zero_variance=preview.zero_variance,
                        skipped=True,
                        members=preview.sample_keys,
                        reason="zero_advantage_group",
                    )
                )
                continue
            self._pending_groups.append(group_id)
        if len(self._pending_groups) < self.plan.groups_per_step:
            return None
        return self._train(tuple(self._pending_groups))

    def _record_stale(self, rejection: GateRejection) -> None:
        self.group_outcomes.append(
            GroupOutcome(
                group_id=rejection.group_id,
                pin=self._pins[rejection.group_id],
                disposition="stale",
                staleness=rejection.staleness,
                members=tuple(slot.attempt_id for slot in rejection.slots),
                reason=(
                    f"staleness {rejection.staleness} exceeds the bound "
                    f"{self.config.pipeline.maximum_policy_lag} at the dequeue gate"
                ),
            )
        )
        self._pending_recycled.append(rejection)

    def _train(self, group_ids: Sequence[str]) -> UpdateRecord:
        bundles: list[EvidenceBundle] = []
        staleness = 0
        for group_id in group_ids:
            group = self.store.group(group_id)
            assert group is not None
            gap = self.current_revision - group.policy_revision
            staleness = max(staleness, gap)
            bundles.extend(self._bundles(group_id, gap))
        try:
            batch: TrainingBatch = assembly.assemble(
                self.plan,
                bundles,
                round_index=self._round_index,
                off_policy=self.config.offline.replaying,
                accepted_staleness=staleness,
            )
        except AssemblyError as error:
            for group_id in group_ids:
                self.group_outcomes.append(
                    GroupOutcome(
                        group_id=group_id,
                        pin=self._pins[group_id],
                        disposition="skipped",
                        skipped=True,
                        reason=f"assembly refused the batch: {error}",
                    )
                )
            self._pending_groups.clear()
            raise
        steps = len(batch.steps)
        if steps > self.plan.max_steps_per_round:
            raise ExecutorError(
                f"the packed batch needs {steps} provider steps but the plan's ceiling "
                f"is {self.plan.max_steps_per_round} per round"
            )
        update_id = f"{self.run_id}::u{len(self.updates):04d}"
        outcomes: dict[str, TrainOutcome] = {}
        for parameter_group in batch.parameter_groups:
            outcomes[parameter_group.parameter_group_id] = self.binder.train(
                parameter_group_id=parameter_group.parameter_group_id,
                batch=[_item_payload(item) for item in parameter_group.items],
                update_id=update_id,
                plan_hash=self.plan.plan_hash,
            )
        published = self.binder.publish(
            run_id=self.run_id,
            update_id=update_id,
            parameter_groups=tuple(outcomes),
            outcome=outcomes,
        )
        if set(published) != set(outcomes):
            raise ExecutorError(
                "publication is atomic: the binder published "
                f"{sorted(published)} for parameter groups {sorted(outcomes)}"
            )
        self.revisions.update(published)
        self.current_revision = min(item.revision for item in self.revisions.values())
        for provenance in batch.provenance:
            self.group_outcomes.append(
                GroupOutcome(
                    group_id=provenance.group_id,
                    pin=self._pins[provenance.group_id],
                    disposition="skipped" if provenance.skipped else "trained",
                    staleness=staleness,
                    rewards=provenance.rewards,
                    advantages=provenance.advantages,
                    zero_variance=provenance.zero_variance,
                    skipped=provenance.skipped,
                    members=provenance.rollout_ids,
                    reason=provenance.credit_kind,
                )
            )
        record = UpdateRecord(
            update_id=update_id,
            round_index=self._round_index,
            group_ids=tuple(group_ids),
            parameter_groups=tuple(sorted(outcomes)),
            steps=steps,
            outcomes=outcomes,
            revisions=dict(published),
            advantage_digest=batch.advantage_digest,
            composition_digest=batch.composition_digest,
        )
        self.updates.append(record)
        self._round_index += 1
        self._pending_groups.clear()
        return record

    # -- the loop ----------------------------------------------------------

    def tick(self) -> Mapping[str, Any]:
        """One pass over every stage. Production never stops for training."""

        self.queues.sweep()
        admitted = 0
        while self.lifecycle.gates.admit and self.queues.has_capacity("rollout"):
            # Returned slots go back before new ones are minted: a recycled
            # group is work that already left the pipeline.
            if self._pending_recycled:
                if self._readmit_recycled(self._pending_recycled[0]) is None:
                    break
                self._pending_recycled.pop(0)
                admitted += 1
                continue
            if self.admit_group() is None:
                break
            admitted += 1
        sent = self.dispatch_once() if self.lifecycle.gates.dispatch else 0
        moved = self.progress_once() if self.lifecycle.gates.score else 0
        self.queues.promote_ready_groups()
        trained = None
        if len(self.updates) < self.config.plan.target_train_updates:
            trained = self.train_once()
        return {
            "admitted_groups": admitted,
            "dispatched": sent,
            "progressed": moved,
            "update": None if trained is None else trained.update_id,
            "lifecycle": self.lifecycle.state,
            "policy_revision": self.current_revision,
        }

    def _finished(self) -> str:
        if len(self.updates) >= self.config.plan.target_train_updates:
            return "target_train_updates_reached"
        state = self.lifecycle.state
        if state in {"drained", "stopped"}:
            return "drained" if state == "drained" else "stopped"
        outstanding = any(self.lifecycle.outstanding_drain_work().values())
        if (
            self.sampled_groups >= self.config.maximum_sampled_groups
            and not outstanding
            and not self.queues.next_dispatch(limit=1)
        ):
            return "sampled_group_budget_exhausted"
        return ""

    def run(
        self,
        *,
        max_ticks: int = 256,
        on_tick: Callable[["ContainerRunExecutor", Mapping[str, Any]], None] | None = None,
        poll_interval_seconds: float = 0.0,
    ) -> RunReport:
        """Drive to the target update count, the group budget, or a control."""

        if poll_interval_seconds < 0:
            raise ExecutorError("poll_interval_seconds cannot be negative")
        self.register_baseline()
        reason = ""
        for _index in range(max_ticks):
            report = self.tick()
            if on_tick is not None:
                on_tick(self, report)
            reason = self._finished()
            if reason:
                break
            if poll_interval_seconds:
                # Real HTTP containers may return from submission before their
                # provider worker has completed. Pace observation without
                # advancing the injected logical clock used by leases/tests.
                time.sleep(poll_interval_seconds)
        else:
            reason = "no_progress"
        return self.finish(reason or "no_progress")

    def finish(self, reason: str) -> RunReport:
        """Close the run, leave the receipts complete, and report."""

        state = self.lifecycle.state
        if state == "draining" and not any(self.lifecycle.outstanding_drain_work().values()):
            self.lifecycle.finish_drain()
        elif state not in {"stopped", "drained"}:
            self.lifecycle.stop(reason=reason)
        for attempt_id in list(self._origins):
            self._close_origin(attempt_id)
        self._stop_reason = reason
        directory = self.write_receipts(reason)
        return RunReport(
            run_id=self.run_id,
            plan_hash=self.plan.plan_hash,
            stop_reason=reason,
            updates=tuple(self.updates),
            groups=tuple(self.group_outcomes),
            sampled_groups=self.sampled_groups,
            receipt_directory=directory,
            final_revisions=dict(self.revisions),
            lifecycle_state=self.lifecycle.state,
        )

    # -- receipts ----------------------------------------------------------

    def _write(self, name: str, payload: Any) -> None:
        path = self.receipts / name
        if name.endswith(".jsonl"):
            rows = payload or []
            text = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
            path.write_text(text, encoding="utf-8")
            return
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )

    def write_receipts(self, reason: str = "") -> Path:
        """Every artifact the note's list names, in one self-contained directory."""

        startup = self.session.receipt()
        plan_payload = self.plan.to_dict()
        self._write(
            "effective_config.json",
            {
                "schema_version": EXECUTOR_SCHEMA_VERSION,
                "config": self.config.redacted_payload(),
                "expanded_plan": plan_payload,
                "plan_hash": self.plan.plan_hash,
                "shared_dimension_hash": self.plan.shared_dimension_hash,
                "stop_reason": reason or self._stop_reason,
            },
        )
        self._write(
            "group_pins.jsonl",
            [
                {"group_id": group_id, "pin": asdict(pin), "pin_digest": pin.pin_digest}
                for group_id, pin in sorted(self._pins.items())
            ],
        )
        self._write(
            "rejected_groups.jsonl",
            [
                {**item.to_payload(), "rejected_field": "policy_revision"}
                for item in self.group_outcomes
                if item.disposition in {"stale", "skipped"}
            ]
            + list(self._recycled),
        )
        self._write(
            "lifecycle.jsonl",
            [
                {
                    "cursor": row.cursor,
                    "at": row.at,
                    "control": row.subject,
                    "from_state": row.from_state,
                    "to_state": row.to_state,
                    "reason": row.reason,
                    "detail": dict(row.detail),
                }
                for row in self.store.lifecycle_events(self.run_id)
            ]
            + [{"control": "resume_rehandshake", **dict(item)} for item in self._rehandshakes],
        )
        self._write(
            "replay.json",
            {
                "mode": self.config.offline.mode,
                "source_run_ids": list(self.config.offline.source_run_ids),
                "accepted_staleness": self.config.offline.accepted_staleness,
                "advantage_digests": {
                    record.update_id: record.advantage_digest for record in self.updates
                },
                "composition_digests": {
                    record.update_id: record.composition_digest for record in self.updates
                },
            },
        )
        self._write(
            "container.json",
            {
                "health": startup["health"],
                "metadata": startup["metadata"],
                "contract": startup["contract"],
                "capabilities": startup["capabilities"],
                "capability_hash": startup["capability_hash"],
                "executor_clauses": startup["executor_clauses"],
                "taskset": startup["taskset"],
                "task_rows": startup["task_rows"],
            },
        )
        self._write("handshake.json", startup["handshake"])
        self._write("probe.json", startup["probe"])
        self._write(
            "provenance.json",
            {
                "container_image_digest": startup["container_image_digest"],
                "container_contract_hash": self.session.startup.contract.contract_hash,
                "optimizer": {"name": "synth_optimizers.cispo", "schema": EXECUTOR_SCHEMA_VERSION},
                "run_binding_digest": self.identity.binding_digest,
            },
        )
        first_published = {
            name: _revision_payload(revision)
            for name, revision in (self.updates[0].revisions if self.updates else {}).items()
        }
        self._write(
            "policy_revisions.json",
            {
                "baseline": {
                    name: _revision_payload(revision)
                    for name, revision in self._baseline_revisions().items()
                },
                "trained": {
                    name: _revision_payload(revision)
                    for name, revision in self.revisions.items()
                },
                "first_published": first_published,
                "updates": [record.to_payload() for record in self.updates],
            },
        )
        self._write("checkpoint_catalog.jsonl", self._catalog_payload())
        self._write("checkpoint_artifacts.jsonl", self._artifact_payload())
        self._write("checkpoint_lineage.jsonl", self._lineage_payload())
        resolution_receipts = getattr(self.binder, "resolution_receipts", lambda: ())
        artifact_identity = getattr(self.binder, "resume_artifact_identity", lambda: {})
        self._write(
            "resume_resolution.json",
            {"resolutions": [dict(row) for row in resolution_receipts()]},
        )
        self._write("resume_artifact_identity.json", artifact_identity())
        self._write(
            "evaluation_manifest.json",
            {
                "paired": self.config.evaluation.paired,
                "baseline_samples": self.config.evaluation.baseline_samples,
                "trained_samples": self.config.evaluation.trained_samples,
                "fixed_match_set": self.config.evaluation.fixed_match_set,
                "split": self.config.taskset.evaluation_split,
                "task_ids": list(self.config.taskset.evaluation_ids),
                "baseline_checkpoint_ids": [
                    revision.checkpoint_id for revision in self._baseline_revisions().values()
                ],
                "trained_checkpoint_ids": [
                    revision.checkpoint_id for revision in self.revisions.values()
                ],
                "match_set_revision_id": self.config.opponents.match_set_revision,
            },
        )
        topology = self.session.topology
        self._write(
            "topology.json",
            {
                "topology_id": topology.topology_id,
                "turn_model": topology.turn_model,
                "actuation_model": topology.actuation_model,
                "reward_relation": topology.reward_relation,
                "parameter_groups": dict(topology.parameter_groups),
                "trainable_instances": [
                    instance.agent_instance_id for instance in topology.trainable_instances
                ],
                "non_trainable_instances": [
                    instance.agent_instance_id for instance in topology.opponent_instances
                ],
                "teams": [
                    {
                        "team_id": team.team_id,
                        "trainable": team.trainable,
                        "minimum_viable_roster": team.minimum_viable_roster,
                    }
                    for team in topology.teams
                ],
                "communication_channels": [
                    {
                        "channel_id": channel.channel_id,
                        "scope": channel.scope,
                        "trainable_for_author": channel.trainable_for_author,
                    }
                    for channel in topology.communication_channels
                ],
                "partial_roster_disposition": self.config.topology.partial_roster,
            },
        )
        self._write(
            "match_set.json",
            {
                "match_set_revision_id": self.config.opponents.match_set_revision,
                "allow_alias_resolution": self.config.opponents.allow_alias_resolution,
                "opponents": [
                    {
                        "agent_instance_id": instance.agent_instance_id,
                        "role_id": instance.role_id,
                        "team_id": instance.team_id,
                        "pinned_identity": instance.pinned_identity,
                    }
                    for instance in topology.opponent_instances
                ],
                "groups": sorted(self._pins),
            },
        )
        self._write("horizon.jsonl", [self._horizon_row(item) for item in self._records()])
        self._write(
            "instance_liveness.jsonl",
            [
                {
                    "rollout_id": record.rollout_id,
                    "group_id": record.group_id,
                    "sample_index": record.sample_index,
                    "admitted": True,
                    "instances": list(record.instance_liveness),
                    "terminal_status": record.episode.terminal_status,
                    "entered_batch": record.group_id in set(self._trained_group_ids()),
                }
                for record in self._records()
            ],
        )
        self._write(
            "team_rewards.jsonl",
            [
                {
                    "rollout_id": record.rollout_id,
                    "optimized_channel": record.reward.optimized_channel,
                    "channels": [
                        {
                            "channel_id": channel.channel_id,
                            "team_id": channel.team_id,
                            "measure": channel.measure,
                            "rank": channel.rank,
                        }
                        for channel in record.reward.channels
                    ],
                }
                for record in self._records()
            ],
        )
        profile = self.session.capability.renderer_profile
        self._write(
            "renderer.json",
            {
                "profile_id": profile.profile_id,
                "package": profile.package,
                "package_version": profile.package_version,
                "config_digest": profile.config_digest,
                "tokenizer_id": profile.tokenizer_id,
                "tokenizer_digest": profile.tokenizer_digest,
                "fingerprint": profile.fingerprint,
                # Identity is what the container declared; agreement is whether
                # a renderer here was ever shown to produce the same tokens. A
                # receipt that records the first without the second reads as
                # though the second happened.
                "agreement_proven": profile.agreement_proven,
                "canary_digest": profile.canary_digest,
                "sampling_transport": self.config.model.sampling_transport,
                "wire_api": self.config.model.wire_api,
                "prompt_budget_policy": self.session.capability.raw.get("policy", {}).get(
                    "prompt_budget_policy", "refuse"
                ),
                "compaction_spans": self._compaction_spans(),
            },
        )
        self._write(
            "queue_journal.jsonl",
            [
                {
                    "cursor": row.cursor,
                    "kind": row.kind,
                    "subject": row.subject,
                    "at": row.at,
                    "from_state": row.from_state,
                    "to_state": row.to_state,
                    "from_queue": row.from_queue,
                    "to_queue": row.to_queue,
                    "reason": row.reason,
                    "detail": dict(row.detail),
                }
                for row in self.store.journal_since(0, run_id=self.run_id)
            ],
        )
        self._write("queue_metrics.json", self._queue_metrics())
        self._write("groups.jsonl", [item.to_payload() for item in self.group_outcomes])
        self._write("provider_usage.json", self._provider_usage())
        self._write("sampling_tps.json", self._sampling_tps())
        self._write(
            "reward_receipts.jsonl",
            [dict(record.reward_payload) for record in self._records()],
        )
        self._write(
            "traces.jsonl",
            [
                {
                    "rollout_id": record.rollout_id,
                    "trace_digest": record.trace_digest,
                    "segments": len(record.episode.segments),
                    "usage": dict(record.usage),
                }
                for record in self._records()
            ],
        )
        self._write(
            "evaluation_rows.json",
            {
                "paired": self.config.evaluation.paired,
                "rows": [],
                "summary": {
                    "baseline_revisions": sorted(self._baseline_revisions()),
                    "trained_revisions": sorted(self.revisions),
                    "note": (
                        "rows are written by the evaluation pass, which resolves the "
                        "immutable ids in evaluation_manifest.json"
                    ),
                },
            },
        )
        self._write(
            "cleanup.json",
            {
                "removed": [
                    {"kind": "sampler_origin", "id": proxy}
                    for proxy in sorted(
                        item for origins in self._origins.values() for item in origins
                    )
                ],
                "retained": [
                    {"kind": "queue_journal", "path": "queue_journal.sqlite3"},
                    {"kind": "receipt_directory", "path": str(self.receipts)},
                ],
                "cancelled_attempts": [
                    attempt.attempt_id
                    for attempt in self.store.attempts_in_state("cancelled", run_id=self.run_id)
                ],
            },
        )
        self._write(
            "manifest.json",
            {
                "schema_version": EXECUTOR_SCHEMA_VERSION,
                "run_id": self.run_id,
                "plan_hash": self.plan.plan_hash,
                "stop_reason": reason or self._stop_reason,
                "artifacts": dict(RUN_ARTIFACTS),
                "files": sorted(path.name for path in self.receipts.iterdir()),
            },
        )
        return self.receipts

    # -- receipt helpers ---------------------------------------------------

    def _records(self) -> tuple[AttemptEvidence, ...]:
        return tuple(
            sorted(self.evidence.values(), key=lambda item: (item.group_id, item.sample_index))
        )

    def _trained_group_ids(self) -> tuple[str, ...]:
        return tuple(item.group_id for item in self.group_outcomes if item.disposition == "trained")

    def _baseline_revisions(self) -> Mapping[str, PolicyRevision]:
        """The revisions the run started from, whatever it published later."""

        return dict(self.baseline_revisions)

    def _horizon_row(self, record: AttemptEvidence) -> Mapping[str, Any]:
        horizon = record.reward.horizon
        return {
            "rollout_id": record.rollout_id,
            "horizon_kind": None if horizon is None else horizon.horizon_kind,
            "horizon_value": None if horizon is None else horizon.horizon_value,
            "scored_at_offset_seconds": (
                None if horizon is None else horizon.scored_at_offset_seconds
            ),
            "clipped": None if horizon is None else horizon.clipped,
            "quiescence_attested": None if horizon is None else horizon.quiescence_attested,
            "settlement_window_seconds": (
                None if horizon is None else horizon.settlement_window_seconds
            ),
            "credited_settlement_seconds": (
                None if horizon is None else horizon.credited_settlement_seconds
            ),
        }

    def _compaction_spans(self) -> int:
        return sum(
            1
            for record in self._records()
            for segment in record.episode.segments
            if segment.branch_id != "root"
        )

    def _queue_metrics(self) -> Mapping[str, Any]:
        return {
            "depths": {
                name: self.queues.depth(name)
                for name in ("rollout", "score", "scored_result", "train_ready")
            },
            "capacities": {
                "rollout": self.config.pipeline.rollout_queue_capacity,
                "score": self.config.pipeline.score_queue_capacity,
                "scored_result": self.config.pipeline.scored_result_queue_capacity,
                "train_ready": self.config.pipeline.train_ready_capacity,
            },
            "max_staleness": self.config.pipeline.maximum_policy_lag,
            "sampled_groups": self.sampled_groups,
            "maximum_sampled_groups": self.config.maximum_sampled_groups,
            "train_ready_groups": [
                group.group_id
                for group in self.store.groups_in_state(GROUP_TRAIN_READY, run_id=self.run_id)
            ],
            "unfillable_groups": list(self.queues.unfillable_groups()),
            "recycled": list(self._recycled),
        }

    def _provider_usage(self) -> Mapping[str, Any]:
        rows = []
        cost = 0.0
        cost_missing = False
        tokens = 0
        examples = 0
        for record in self.updates:
            for parameter_group, outcome in record.outcomes.items():
                outcome_cost_missing = bool(outcome.metrics.get("cost_missing", False))
                if outcome_cost_missing:
                    cost_missing = True
                    receipted_cost: float | None = None
                else:
                    receipted_cost = outcome.provider_cost
                    cost += outcome.provider_cost
                tokens += outcome.tokens
                examples += outcome.examples
                rows.append(
                    {
                        "update_id": record.update_id,
                        "parameter_group_id": parameter_group,
                        "request_ids": list(outcome.request_ids),
                        "examples": outcome.examples,
                        "training_tokens": outcome.tokens,
                        "provider_cost": receipted_cost,
                        "cost_missing": outcome_cost_missing,
                        "metrics": dict(outcome.metrics),
                    }
                )
        return {
            "train_calls": rows,
            "totals": {
                "provider_cost": None if cost_missing else cost,
                "cost_missing": cost_missing,
                "training_tokens": tokens,
                "examples": examples,
                "train_calls": len(rows),
                "probe_cost": self.session.startup.probe_cost,
            },
        }

    def _sampling_tps(self) -> Mapping[str, Any]:
        rows = []
        total_tokens = 0
        service_seconds = 0.0
        submitted: list[float] = []
        scored: list[float] = []
        for record in self._records():
            seconds = record.seconds
            tokens = record.generated_tokens
            total_tokens += tokens
            service_seconds += seconds
            submitted.append(record.submitted_at)
            scored.append(record.scored_at)
            rows.append(
                {
                    "rollout_id": record.rollout_id,
                    "generated_tokens": tokens,
                    "seconds": seconds,
                    "tokens_per_second": (tokens / seconds) if seconds > 0 else None,
                }
            )
        makespan_seconds = (
            max(max(scored) - min(submitted), 0.0) if submitted and scored else 0.0
        )
        rollout_count = len(rows)
        return {
            "by_call": rows,
            "clock_source": type(self.clock).__name__,
            "service_time_semantics": "sum_of_per_call_submit_to_score_seconds",
            "makespan_semantics": "earliest_submit_to_latest_score_seconds",
            "service_time_generated_tps": (
                (total_tokens / service_seconds) if service_seconds > 0 else None
            ),
            # Backward-compatible aliases. These have always described summed
            # per-call service time, not concurrent wall-clock throughput.
            "weighted_aggregate_tps": (
                (total_tokens / service_seconds) if service_seconds > 0 else None
            ),
            "generated_tokens": total_tokens,
            "sampling_seconds": service_seconds,
            "service_time_seconds": service_seconds,
            "makespan_seconds": makespan_seconds,
            "rollout_count": rollout_count,
            "end_to_end_generated_tps": (
                (total_tokens / makespan_seconds) if makespan_seconds > 0 else None
            ),
            "end_to_end_rollouts_per_second": (
                (rollout_count / makespan_seconds) if makespan_seconds > 0 else None
            ),
        }

    def _catalog_payload(self) -> list[Mapping[str, Any]]:
        if self._catalog_rows is not None:
            return [dict(row) for row in self._catalog_rows()]
        rows = [
            {
                "checkpoint_id": revision.checkpoint_id,
                "parameter_group_id": name,
                "policy_revision_id": revision.revision_id,
                "revision": revision.revision,
                "publication_status": "published",
                "role": "baseline",
                "run_id": revision.metadata.get("run_id"),
                "update_id": revision.metadata.get("update_id"),
                "parent_checkpoint_id": revision.metadata.get("parent_checkpoint_id"),
                "sampler_reference": revision.sampler_reference,
                "sampler_digest": revision.metadata.get("sampler_digest"),
                "training_state_reference": revision.training_state_reference,
                "training_state_digest": revision.metadata.get("training_state_digest"),
                "policy_set_revision_id": revision.policy_set_revision_id,
            }
            for name, revision in sorted(self._baseline_revisions().items())
        ]
        for record in self.updates:
            for name, revision in sorted(record.revisions.items()):
                rows.append(
                    {
                        "checkpoint_id": revision.checkpoint_id,
                        "parameter_group_id": name,
                        "policy_revision_id": revision.revision_id,
                        "revision": revision.revision,
                        "publication_status": "published",
                        "role": "trained",
                        "run_id": revision.metadata.get("run_id"),
                        "parent_checkpoint_id": revision.metadata.get("parent_checkpoint_id"),
                        "sampler_reference": revision.sampler_reference,
                        "sampler_digest": revision.metadata.get("sampler_digest"),
                        "training_state_reference": revision.training_state_reference,
                        "training_state_digest": revision.metadata.get("training_state_digest"),
                        "update_id": record.update_id,
                        "policy_set_revision_id": revision.policy_set_revision_id,
                    }
                )
        return rows

    def _artifact_payload(self) -> list[Mapping[str, Any]]:
        rows: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for name, revision in list(self._baseline_revisions().items()) + [
            (name, revision)
            for record in self.updates
            for name, revision in record.revisions.items()
        ]:
            if revision.checkpoint_id in seen:
                continue
            seen.add(revision.checkpoint_id)
            rows.append(
                {
                    "checkpoint_id": revision.checkpoint_id,
                    "parameter_group_id": name,
                    "sampler_weights": {
                        "ref": revision.sampler_reference,
                        "digest": revision.metadata.get("sampler_digest", ""),
                        "retained": True,
                    },
                    "training_state": {
                        "ref": revision.training_state_reference,
                        "digest": revision.metadata.get("training_state_digest", ""),
                        "retained": self.config.artifacts.retain_training_state,
                    },
                }
            )
        return rows

    def _lineage_payload(self) -> list[Mapping[str, Any]]:
        if self._lineage_rows is not None:
            return [dict(row) for row in self._lineage_rows()]
        rows: list[Mapping[str, Any]] = []
        for name, revision in sorted(self._baseline_revisions().items()):
            parent = revision.metadata.get("parent_checkpoint_id")
            if parent:
                rows.append(
                    {
                        "child_checkpoint_id": revision.checkpoint_id,
                        "parent_checkpoint_id": parent,
                        "relation": "resumed_from",
                        "run_id": self.run_id,
                        "update_id": revision.metadata.get("update_id"),
                        "parameter_group_id": name,
                        "policy_type_ids": list(revision.metadata.get("policy_type_ids") or ()),
                        "train_call_ids": [],
                        "policy_set_revision_id": revision.policy_set_revision_id,
                    }
                )
        parents = {
            name: revision.checkpoint_id
            for name, revision in self._baseline_revisions().items()
        }
        for record in self.updates:
            for name, revision in sorted(record.revisions.items()):
                rows.append(
                    {
                        "child_checkpoint_id": revision.checkpoint_id,
                        "parent_checkpoint_id": parents.get(name),
                        "relation": "trained_from",
                        "run_id": self.run_id,
                        "update_id": record.update_id,
                        "parameter_group_id": name,
                        "policy_type_ids": [
                            policy_type
                            for policy_type, group in self.session.topology.parameter_groups.items()
                            if group == name
                        ],
                        "train_call_ids": list(
                            record.outcomes[name].request_ids if name in record.outcomes else ()
                        ),
                        "policy_set_revision_id": revision.policy_set_revision_id,
                    }
                )
                parents[name] = revision.checkpoint_id
        return rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolved_channel(reward: RewardRecord, team_id: str | None):
    """The channel this group is compared on. Recorded, never guessed."""

    by_id = {channel.channel_id: channel for channel in reward.channels}
    optimized = by_id.get(reward.optimized_channel)
    if optimized is None:
        raise ExecutorError(
            f"reward {reward.reward_id} names channel {reward.optimized_channel!r} "
            "which it does not carry"
        )
    if team_id is None or optimized.team_id == team_id:
        return optimized
    candidates = [channel for channel in reward.channels if channel.team_id == team_id]
    if len(candidates) == 1:
        return candidates[0]
    raise ExecutorError(
        f"reward {reward.reward_id} has {len(candidates)} channels for team {team_id!r}; "
        "the optimized channel for a team must be unambiguous"
    )


def _item_payload(item: Any) -> dict[str, Any]:
    """One packed span, as the provider-facing mapping the binder consumes."""

    return {
        "parameter_group_id": item.parameter_group_id,
        "group_id": item.group_id,
        "rollout_id": item.rollout_id,
        "root_rollout_id": item.root_rollout_id,
        "sample_index": item.sample_index,
        "branch_id": item.branch_id,
        "agent_instance_id": item.agent_instance_id,
        "token_ids": list(item.token_ids),
        "loss_mask": list(item.loss_mask),
        "behavior_logprobs": list(item.behavior_logprobs),
        "advantage": item.advantage,
        "root_rollout_weight": item.root_rollout_weight,
        "same_policy_weight": item.same_policy_weight,
        "loss_weight": item.loss_weight,
        "trainable_tokens": item.trainable_tokens,
        "policy_revision": item.policy_revision,
        "staleness_steps": item.staleness_steps,
        "call_ids": list(item.call_ids),
    }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Everything ``execute`` needs beyond the ports, kept out of its signature."""

    receipts: Path
    max_ticks: int = 256
    poll_interval_seconds: float = 0.0
    on_tick: Callable[[ContainerRunExecutor, Mapping[str, Any]], None] | None = None
    catalog_rows: Callable[[], Sequence[Mapping[str, Any]]] | None = None
    lineage_rows: Callable[[], Sequence[Mapping[str, Any]]] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def execute(
    config: RunConfig,
    session: ContractContainerSession,
    gateway: SamplerGateway,
    binder: PolicyBinder,
    *,
    clock: RunClock,
    plan: ExecutionPlan,
) -> RunReport:
    """Run one admitted session to its target, and leave the receipts behind."""

    executor = ContainerRunExecutor(
        config=config,
        session=session,
        gateway=gateway,
        binder=binder,
        clock=clock,
        receipts=plan.receipts,
        catalog_rows=plan.catalog_rows,
        lineage_rows=plan.lineage_rows,
    )
    return executor.run(
        max_ticks=plan.max_ticks,
        on_tick=plan.on_tick,
        poll_interval_seconds=plan.poll_interval_seconds,
    )
