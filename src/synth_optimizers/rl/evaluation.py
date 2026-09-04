"""The paired evaluation entrypoint.

One number on its own is not a result. A trained policy's mean reward is only
readable next to a baseline's mean reward earned on the *same* held-out seeds,
through the *same* roster, and -- where the topology is competitive -- against
the *same* pinned opponent set. So this module runs two arms rather than one,
and refuses any pair whose two halves are not comparable.

Everything the arms will load is resolved and verified first. A selector goes
through :class:`~synth_optimizers.rl.resolver.EvaluationResolver`, which turns
it into immutable ids and checks artifact existence, digest, role, and
renderer/tokenizer compatibility. Only after both arms and the pinned match set
have passed does the first attempt start. A missing artifact, a digest that
disagrees, or a component bound in the wrong role is an evidence failure and
ends the evaluation; nothing here falls back to whatever is newest.

The receipt records the requested selector *and* the immutable id it resolved
to, the provider sampler references actually loaded, the seeds, the per-arm
rewards, and the paired summary. The result is then written back to the catalog
as an :class:`~synth_optimizers.rl.catalog.EvaluationBinding`: an append-only
relation on the checkpoints, never a mutation of them.

Nothing here names a task, a harness, an environment, or a provider.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.rl_identity import GroupPin, TaskSpec
from ..contracts.rl_records import EvidenceError, RewardRecord, TrainableEpisode, digest
from .catalog import SAMPLER_ROLE, EvaluationBinding, utc_now
from .ports import (
    AttemptFacts,
    ContainerSession,
    PolicyBinder,
    PolicyRevision,
    SamplerGateway,
    SamplerOrigin,
)
from .resolver import (
    CompatibilityRequirement,
    EvaluationResolver,
    Resolution,
    ResolutionScope,
    ResolvedOpponent,
)

EVALUATION_RECEIPT_SCHEMA_VERSION = "cispo.evaluation_receipt.v1"

BASELINE_ARM = "baseline"
TRAINED_ARM = "trained"
ARMS: tuple[str, str] = (BASELINE_ARM, TRAINED_ARM)

#: Terminal states a container may report. Anything else is still in flight.
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

#: Scored and awaiting-score attempts are ready for the explicit finalization
#: barrier.  Immediate scorers may expose ``awaiting_score`` until that barrier
#: seals the episode and publishes its reward.  Deferred scorers may still
#: refuse evidence after finalization; that remains a typed evidence failure,
#: never a fabricated zero.
FINALIZABLE_STATES = TERMINAL_STATES | {"scored", "awaiting_score"}

# The shipped container configs declare a 60-second expected attempt horizon.
# At the default 250 ms cadence, 240 observations cover that envelope.
DEFAULT_POLL_LIMIT = 240
DEFAULT_POLL_INTERVAL_SECONDS = 0.25


class EvaluationError(EvidenceError):
    """The evaluation cannot produce a comparable number. Never degraded."""


class ArmComparabilityError(EvaluationError):
    """The two arms would not be measuring the same thing."""


class SamplerReferenceMismatchError(EvaluationError):
    """The binder loaded a provider reference the catalog did not catalogue."""


class RosterBindingError(EvaluationError):
    """A roster slot has no policy in the resolution that was supposed to bind it."""


class AttemptFailedError(EvaluationError):
    """An attempt did not reach a scored terminal state. Absent is not zero."""


@dataclass(frozen=True, slots=True)
class RosterSlot:
    """One instance the evaluation binds. Identical across both arms."""

    agent_instance_id: str
    parameter_group_id: str
    policy_type_id: str | None = None
    team_id: str | None = None
    role_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_instance_id": self.agent_instance_id,
            "parameter_group_id": self.parameter_group_id,
            "policy_type_id": self.policy_type_id,
            "team_id": self.team_id,
            "role_id": self.role_id,
        }


@dataclass(frozen=True, slots=True)
class HeldOutSeed:
    """One held-out unit of work. Both arms run exactly this list."""

    task_id: str
    seed: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.task_id, self.seed)

    def to_payload(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class PinTemplate:
    """The run-invariant half of a group pin.

    The arm-varying half -- policy revision, policy-set revision, behavior
    fingerprint -- is filled in per arm, so two arms differ in exactly the
    fields that are supposed to differ and in no others.
    """

    run_id: str
    algorithm_plan_hash: str
    wire_api: str
    sampling_transport: str
    policy_kind: str
    model_family: str
    container_image_digest: str
    container_contract_hash: str
    task_family: str
    topology_id: str | None = None

    def pin(
        self,
        *,
        group_id: str,
        behavior_fingerprint: str,
        policy_revision: int,
        policy_revision_id: str | None,
        cardinality: int,
        handshake_agreement_digest: str,
        policy_set_revision_id: str | None = None,
        match_set_revision_id: str | None = None,
    ) -> GroupPin:
        return GroupPin(
            group_id=group_id,
            run_id=self.run_id,
            algorithm_plan_hash=self.algorithm_plan_hash,
            behavior_fingerprint=behavior_fingerprint,
            policy_revision=policy_revision,
            wire_api=self.wire_api,
            sampling_transport=self.sampling_transport,
            policy_kind=self.policy_kind,
            model_family=self.model_family,
            container_image_digest=self.container_image_digest,
            container_contract_hash=self.container_contract_hash,
            handshake_agreement_digest=handshake_agreement_digest,
            task_family=self.task_family,
            cardinality=cardinality,
            policy_set_revision_id=policy_set_revision_id,
            match_set_revision_id=match_set_revision_id,
            topology_id=self.topology_id,
            policy_revision_id=policy_revision_id,
        )


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """What to evaluate, against what, on which held-out seeds."""

    evaluation_id: str
    baseline_selector: str
    trained_selector: str
    seeds: tuple[HeldOutSeed, ...]
    roster: tuple[RosterSlot, ...]
    pin: PinTemplate
    split: str = "heldout"
    #: The pinned match set both arms play. A competitive topology requires it.
    match_set_selector: str | None = None
    scope: ResolutionScope | None = None
    #: Which reward channel is the measure. Defaults to the optimized channel.
    reward_channel: str | None = None
    metric_name: str = "mean_reward"
    poll_limit: int = DEFAULT_POLL_LIMIT

    def __post_init__(self) -> None:
        if not str(self.evaluation_id).strip():
            raise EvaluationError("an evaluation must carry an id")
        if not self.seeds:
            raise EvaluationError(
                "a paired evaluation needs at least one held-out seed; an empty "
                "held-out set produces a summary of nothing"
            )
        if len({seed.key for seed in self.seeds}) != len(self.seeds):
            raise EvaluationError("held-out seeds must be distinct (task_id, seed) pairs")
        if not self.roster:
            raise EvaluationError("a paired evaluation needs a roster to bind")
        groups = [slot.parameter_group_id for slot in self.roster]
        if len(set(groups)) != len(groups):
            raise EvaluationError(
                f"roster binds parameter group(s) twice: {sorted(groups)}; one slot per group"
            )
        if self.poll_limit < 1:
            raise EvaluationError("poll_limit must be positive")

    @property
    def task_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for seed in self.seeds:
            if seed.task_id not in seen:
                seen.append(seed.task_id)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class AttemptRow:
    """One scored attempt on one arm. The unit both arms are compared over."""

    arm: str
    task_id: str
    seed: int
    sample_index: int
    rollout_id: str
    proxy_request_id: str
    reward: float
    reward_channel: str
    terminal_status: str
    checkpoint_ids: tuple[str, ...]
    sampler_references: tuple[str, ...]
    trace_digest: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "seed": self.seed,
            "sample_index": self.sample_index,
            "rollout_id": self.rollout_id,
            "proxy_request_id": self.proxy_request_id,
            "reward": self.reward,
            "reward_channel": self.reward_channel,
            "terminal_status": self.terminal_status,
            "checkpoint_ids": list(self.checkpoint_ids),
            "sampler_references": list(self.sampler_references),
            "trace_digest": self.trace_digest,
        }


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm: what it resolved to, what it loaded, and what it scored."""

    arm: str
    requested_selector: str
    resolution: Resolution
    #: The provider sampler references the binder actually returned.
    loaded_sampler_references: tuple[str, ...]
    attempts: tuple[AttemptRow, ...]

    @property
    def resolved_id(self) -> str:
        return self.resolution.resolved_id

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(attempt.reward for attempt in self.attempts)

    @property
    def mean_reward(self) -> float:
        return statistics.fmean(self.rewards) if self.attempts else 0.0

    def reward_for(self, seed: HeldOutSeed) -> float:
        for attempt in self.attempts:
            if attempt.task_id == seed.task_id and attempt.seed == seed.seed:
                return attempt.reward
        raise EvaluationError(
            f"arm {self.arm} scored no attempt for task {seed.task_id!r} seed {seed.seed}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "requested_selector": self.requested_selector,
            "resolved_id": self.resolved_id,
            "resolution": self.resolution.to_receipt(),
            "catalogued_sampler_references": list(self.resolution.loaded_refs),
            "loaded_sampler_references": list(self.loaded_sampler_references),
            "attempts": [attempt.to_payload() for attempt in self.attempts],
            "attempt_count": len(self.attempts),
            "mean_reward": self.mean_reward,
        }


@dataclass(frozen=True, slots=True)
class PairedRow:
    """The two arms' rewards on one held-out seed, side by side."""

    task_id: str
    seed: int
    baseline_reward: float
    trained_reward: float

    @property
    def delta(self) -> float:
        return self.trained_reward - self.baseline_reward

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "baseline_reward": self.baseline_reward,
            "trained_reward": self.trained_reward,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class PairedSummary:
    """The comparison. Uplift is recorded, never required."""

    rows: tuple[PairedRow, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise EvaluationError("a paired summary needs at least one paired row")

    @property
    def pairs(self) -> int:
        return len(self.rows)

    @property
    def baseline_mean(self) -> float:
        return statistics.fmean(row.baseline_reward for row in self.rows)

    @property
    def trained_mean(self) -> float:
        return statistics.fmean(row.trained_reward for row in self.rows)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(row.delta for row in self.rows)

    @property
    def delta_stdev(self) -> float:
        deltas = [row.delta for row in self.rows]
        return statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    @property
    def wins(self) -> int:
        return sum(1 for row in self.rows if row.delta > 0)

    @property
    def losses(self) -> int:
        return sum(1 for row in self.rows if row.delta < 0)

    @property
    def ties(self) -> int:
        return sum(1 for row in self.rows if row.delta == 0)

    def to_payload(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "baseline_mean": self.baseline_mean,
            "trained_mean": self.trained_mean,
            "mean_delta": self.mean_delta,
            "delta_stdev": self.delta_stdev,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "rows": [row.to_payload() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    """``cispo.evaluation_receipt.v1``. Selector, resolution, refs, and rewards."""

    evaluation_id: str
    split: str
    seeds: tuple[HeldOutSeed, ...]
    roster: tuple[RosterSlot, ...]
    baseline: ArmResult
    trained: ArmResult
    summary: PairedSummary
    bindings: tuple[EvaluationBinding, ...]
    match_set_selector: str | None = None
    match_set_revision_id: str | None = None
    opponents: tuple[ResolvedOpponent, ...] = ()
    metric_name: str = "mean_reward"
    handshake_id: str = ""
    agreement_digest: str = ""
    created_at: str = ""
    schema_version: str = EVALUATION_RECEIPT_SCHEMA_VERSION

    @property
    def selector_resolutions(self) -> tuple[tuple[str, str], ...]:
        """``(requested selector, immutable id)`` for every policy this loaded."""

        return tuple(
            (arm.requested_selector, arm.resolved_id) for arm in (self.baseline, self.trained)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at,
            "split": self.split,
            "metric_name": self.metric_name,
            "handshake_id": self.handshake_id,
            "agreement_digest": self.agreement_digest,
            "seeds": [seed.to_payload() for seed in self.seeds],
            "roster": [slot.to_payload() for slot in self.roster],
            "match_set": {
                "requested_selector": self.match_set_selector,
                "match_set_revision_id": self.match_set_revision_id,
                "opponents": [opponent.to_payload() for opponent in self.opponents],
            },
            "arms": {
                BASELINE_ARM: self.baseline.to_payload(),
                TRAINED_ARM: self.trained.to_payload(),
            },
            "paired_summary": self.summary.to_payload(),
            "evaluation_bindings": [binding.to_payload() for binding in self.bindings],
        }

    def write(self, directory: str | Path) -> Path:
        """Write the receipt into a run's artifact directory."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{self.evaluation_id}.evaluation.json"
        path.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


@dataclass(slots=True)
class _BoundArm:
    """Everything one arm needs, resolved and loaded, before it runs."""

    arm: str
    resolution: Resolution
    revisions: Mapping[str, PolicyRevision]
    loaded_refs: tuple[str, ...] = ()
    attempts: list[AttemptRow] = field(default_factory=list)


class PairedEvaluation:
    """Run a baseline arm and a trained arm over one identical held-out set."""

    def __init__(
        self,
        resolver: EvaluationResolver,
        *,
        session: ContainerSession,
        gateway: SamplerGateway,
        binder: PolicyBinder,
        clock: Any = utc_now,
    ) -> None:
        self._resolver = resolver
        self._session = session
        self._gateway = gateway
        self._binder = binder
        self._clock = clock

    # ------------------------------------------------------------------ run

    def run(self, request: EvaluationRequest) -> EvaluationReceipt:
        """Resolve, verify, then run both arms. Refusals happen before attempts."""

        requirement = CompatibilityRequirement.from_renderer_profile(
            self._gateway.renderer_profile,
            container_contract_hash=request.pin.container_contract_hash,
        )
        # --- verification, all of it, before a single attempt is submitted ---
        baseline = self._resolve(request.baseline_selector, request, requirement)
        trained = self._resolve(request.trained_selector, request, requirement)
        pinned_match, opponents = self._pinned_match_set(request, requirement, baseline, trained)
        self._assert_comparable(request, baseline, trained, opponents)

        bound = {
            BASELINE_ARM: self._load(BASELINE_ARM, baseline, request),
            TRAINED_ARM: self._load(TRAINED_ARM, trained, request),
        }
        tasks = self._tasks(request)

        # --- only now does anything run ---
        for arm in ARMS:
            self._run_arm(bound[arm], request, tasks, pinned_match)

        results = {
            arm: ArmResult(
                arm=arm,
                requested_selector=(
                    request.baseline_selector if arm == BASELINE_ARM else request.trained_selector
                ),
                resolution=bound[arm].resolution,
                loaded_sampler_references=bound[arm].loaded_refs,
                attempts=tuple(bound[arm].attempts),
            )
            for arm in ARMS
        }
        summary = PairedSummary(
            rows=tuple(
                PairedRow(
                    task_id=seed.task_id,
                    seed=seed.seed,
                    baseline_reward=results[BASELINE_ARM].reward_for(seed),
                    trained_reward=results[TRAINED_ARM].reward_for(seed),
                )
                for seed in request.seeds
            )
        )
        bindings = self._record(request, results, summary)
        return EvaluationReceipt(
            evaluation_id=request.evaluation_id,
            split=request.split,
            seeds=request.seeds,
            roster=request.roster,
            baseline=results[BASELINE_ARM],
            trained=results[TRAINED_ARM],
            summary=summary,
            bindings=bindings,
            match_set_selector=request.match_set_selector,
            match_set_revision_id=pinned_match,
            opponents=opponents,
            metric_name=request.metric_name,
            handshake_id=self._session.handshake_id,
            agreement_digest=self._session.agreement_digest,
            created_at=self._clock(),
        )

    # ----------------------------------------------------- resolve / verify

    def _resolve(
        self,
        selector: str,
        request: EvaluationRequest,
        requirement: CompatibilityRequirement,
    ) -> Resolution:
        """Immutable ids and verified artifacts, or a typed refusal."""

        return self._resolver.resolve(
            selector,
            role=SAMPLER_ROLE,
            compatibility=requirement,
            scope=request.scope,
        )

    def _pinned_match_set(
        self,
        request: EvaluationRequest,
        requirement: CompatibilityRequirement,
        baseline: Resolution,
        trained: Resolution,
    ) -> tuple[str | None, tuple[ResolvedOpponent, ...]]:
        """The one opponent set both arms play, or none for a solo topology."""

        if request.match_set_selector is not None:
            match = self._resolver.resolve_match_set(
                request.match_set_selector,
                role=SAMPLER_ROLE,
                compatibility=requirement,
                scope=request.scope,
            )
            return match.resolved_id, match.opponents
        arms_match = {
            arm.match_set_revision_id for arm in (baseline, trained) if arm.match_set_revision_id
        }
        if not arms_match:
            return None, ()
        if len(arms_match) > 1:
            raise ArmComparabilityError(
                "the arms name different match-set revisions "
                f"{sorted(arms_match)}; a reward earned against one opponent set is not "
                "comparable to a reward earned against another"
            )
        only = next(iter(arms_match))
        opponents = baseline.opponents or trained.opponents
        return only, opponents

    def _assert_comparable(
        self,
        request: EvaluationRequest,
        baseline: Resolution,
        trained: Resolution,
        opponents: tuple[ResolvedOpponent, ...],
    ) -> None:
        """Refuse a pair whose halves are not measuring the same thing."""

        for arm, resolution in ((BASELINE_ARM, baseline), (TRAINED_ARM, trained)):
            for slot in request.roster:
                try:
                    policy = resolution.policy_for_group(slot.parameter_group_id)
                except EvidenceError as error:
                    raise RosterBindingError(
                        f"arm {arm} selector {resolution.requested_selector!r} resolves no "
                        f"policy for roster slot {slot.agent_instance_id!r} "
                        f"(parameter group {slot.parameter_group_id!r}): {error}"
                    ) from error
                if (
                    slot.policy_type_id is not None
                    and slot.policy_type_id not in policy.policy_type_ids
                ):
                    raise RosterBindingError(
                        f"arm {arm} binds {policy.checkpoint_id} to roster slot "
                        f"{slot.agent_instance_id!r}, which serves policy type "
                        f"{slot.policy_type_id!r} and the checkpoint does not"
                    )
        baseline_compat = {policy.compatibility.renderer_profile for policy in baseline.policies}
        trained_compat = {policy.compatibility.renderer_profile for policy in trained.policies}
        if baseline_compat != trained_compat:
            raise ArmComparabilityError(
                f"the arms render differently: {sorted(baseline_compat)} against "
                f"{sorted(trained_compat)}; their tokens do not mean the same thing"
            )
        for arm, resolution in ((BASELINE_ARM, baseline), (TRAINED_ARM, trained)):
            if not opponents or not resolution.opponents:
                continue
            theirs = tuple(sorted(_opponent_key(item) for item in resolution.opponents))
            shared = tuple(sorted(_opponent_key(item) for item in opponents))
            if theirs != shared:
                raise ArmComparabilityError(
                    f"arm {arm} pins opponents {list(theirs)}, the evaluation pins "
                    f"{list(shared)}; both arms must play the identical pinned match set"
                )

    # ------------------------------------------------------------- loading

    def _load(
        self, arm: str, resolution: Resolution, request: EvaluationRequest
    ) -> _BoundArm:
        """Ask the binder for the revisions, then check it loaded what we resolved."""

        revisions = dict(self._binder.resolve(resolution.resolved_id))
        catalogued = {policy.parameter_group_id: policy for policy in resolution.policies}
        loaded: list[str] = []
        for slot in request.roster:
            revision = revisions.get(slot.parameter_group_id)
            if revision is None:
                raise RosterBindingError(
                    f"arm {arm}: the binder returned no revision for parameter group "
                    f"{slot.parameter_group_id!r} of {resolution.resolved_id}"
                )
            policy = catalogued[slot.parameter_group_id]
            if revision.checkpoint_id != policy.checkpoint_id:
                raise SamplerReferenceMismatchError(
                    f"arm {arm}: the binder loaded checkpoint {revision.checkpoint_id}, the "
                    f"catalog resolved {policy.checkpoint_id}"
                )
            if revision.sampler_reference != policy.artifact.ref:
                raise SamplerReferenceMismatchError(
                    f"arm {arm}: checkpoint {policy.checkpoint_id} was loaded from "
                    f"{revision.sampler_reference!r}, catalogued as {policy.artifact.ref!r}"
                )
            loaded.append(revision.sampler_reference)
        return _BoundArm(
            arm=arm,
            resolution=resolution,
            revisions=revisions,
            loaded_refs=tuple(loaded),
        )

    def _tasks(self, request: EvaluationRequest) -> Mapping[str, TaskSpec]:
        specs = self._session.tasks(split=request.split, task_ids=list(request.task_ids))
        by_id = {spec.task_id: spec for spec in specs}
        missing = [task_id for task_id in request.task_ids if task_id not in by_id]
        if missing:
            raise EvaluationError(
                f"held-out split {request.split!r} does not carry task(s) {missing}; "
                "both arms must run the identical held-out set"
            )
        for held in request.seeds:
            declared = by_id[held.task_id].seed
            if declared != held.seed:
                raise EvaluationError(
                    f"held-out task {held.task_id!r} is seeded {declared} in split "
                    f"{request.split!r}, the evaluation asked for {held.seed}; the seed is "
                    "the container's, and both arms run the identical one"
                )
        return by_id

    # ------------------------------------------------------------- running

    def _run_arm(
        self,
        bound: _BoundArm,
        request: EvaluationRequest,
        tasks: Mapping[str, TaskSpec],
        match_set_revision_id: str | None,
    ) -> None:
        primary = request.roster[0]
        revision = bound.revisions[primary.parameter_group_id]
        group_id = f"{request.evaluation_id}::{bound.arm}"
        for index, held in enumerate(request.seeds):
            pin = request.pin.pin(
                group_id=group_id,
                behavior_fingerprint=revision.behavior_fingerprint,
                policy_revision=revision.revision,
                policy_revision_id=revision.revision_id,
                cardinality=len(request.seeds),
                handshake_agreement_digest=self._session.agreement_digest,
                policy_set_revision_id=bound.resolution.policy_set_revision_id,
                match_set_revision_id=match_set_revision_id,
            )
            origins = self._bind_roster(bound, request, group_id, index, pin, held)
            try:
                row = self._attempt(
                    bound,
                    request,
                    tasks[held.task_id],
                    held,
                    index,
                    pin,
                    origins[primary.parameter_group_id],
                )
            finally:
                for origin in origins.values():
                    self._gateway.close(origin.proxy_request_id)
            bound.attempts.append(row)

    def _bind_roster(
        self,
        bound: _BoundArm,
        request: EvaluationRequest,
        group_id: str,
        sample_index: int,
        pin: GroupPin,
        held: HeldOutSeed,
    ) -> dict[str, SamplerOrigin]:
        """One origin per roster slot, named by the attempt it will actually run.

        The facts carry the evaluation's own attempt id: the container has not
        minted a rollout id yet, and the captured evidence has to name the task
        and seed this arm ran rather than one inferred from the pin.
        """

        attempt_id = f"{group_id}::s{sample_index}"
        facts = AttemptFacts(rollout_id=attempt_id, task_id=held.task_id, seed=held.seed)
        origins: dict[str, SamplerOrigin] = {}
        for slot in request.roster:
            revision = bound.revisions[slot.parameter_group_id]
            proxy_request_id = _proxy_request_id(group_id, sample_index, slot.agent_instance_id)
            origins[slot.parameter_group_id] = self._gateway.bind(
                revision,
                pin=pin,
                sample_index=sample_index,
                proxy_request_id=proxy_request_id,
                attempt=facts,
            )
        return origins

    def _attempt(
        self,
        bound: _BoundArm,
        request: EvaluationRequest,
        task: TaskSpec,
        held: HeldOutSeed,
        sample_index: int,
        pin: GroupPin,
        origin: SamplerOrigin,
    ) -> AttemptRow:
        rollout_id = self._session.submit(
            task,
            origin,
            pin=pin,
            sample_index=sample_index,
            idempotency_key=_idempotency_key(request.evaluation_id, bound.arm, held, sample_index),
        )
        state: Mapping[str, Any] = {}
        for _ in range(request.poll_limit):
            state = self._session.poll(rollout_id)
            if state.get("terminal") or str(state.get("state") or "") in FINALIZABLE_STATES:
                break
            # Async containers return from submission before their provider
            # worker. Pace observation so the bounded poll count represents a
            # real opportunity to finish instead of a localhost hot spin.
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
        else:
            self._session.terminate(rollout_id, reason="evaluation_poll_limit")
            raise AttemptFailedError(
                f"arm {bound.arm} attempt {rollout_id} on task {held.task_id!r} seed "
                f"{held.seed} never reached a terminal state; an unfinished attempt is "
                "not a zero"
            )
        self._session.finalize(rollout_id)
        episode, reward = self._session.evidence(rollout_id)
        return self._row(bound, held, sample_index, rollout_id, origin, episode, reward, request)

    def _row(
        self,
        bound: _BoundArm,
        held: HeldOutSeed,
        sample_index: int,
        rollout_id: str,
        origin: SamplerOrigin,
        episode: TrainableEpisode,
        reward: RewardRecord,
        request: EvaluationRequest,
    ) -> AttemptRow:
        reward.validate()
        channel = request.reward_channel or reward.optimized_channel
        scored = {"completed", "scored"}
        if episode.terminal_status not in scored and reward.terminal_status not in scored:
            raise AttemptFailedError(
                f"arm {bound.arm} attempt {rollout_id} terminated "
                f"{episode.terminal_status!r}; a failed attempt is not a zero reward"
            )
        return AttemptRow(
            arm=bound.arm,
            task_id=held.task_id,
            seed=held.seed,
            sample_index=sample_index,
            rollout_id=rollout_id,
            proxy_request_id=origin.proxy_request_id,
            reward=reward.value(channel),
            reward_channel=channel,
            terminal_status=reward.terminal_status,
            checkpoint_ids=bound.resolution.checkpoint_ids,
            sampler_references=bound.loaded_refs,
            trace_digest=episode.trace_digest,
        )

    # ------------------------------------------------------------ recording

    def _record(
        self,
        request: EvaluationRequest,
        results: Mapping[str, ArmResult],
        summary: PairedSummary,
    ) -> tuple[EvaluationBinding, ...]:
        """Append one relation per arm. The checkpoint records are untouched."""

        bindings: list[EvaluationBinding] = []
        for arm in ARMS:
            metrics: dict[str, float] = {
                request.metric_name: results[arm].mean_reward,
                "paired_attempts": float(summary.pairs),
            }
            if arm == TRAINED_ARM:
                metrics["paired_mean_delta"] = summary.mean_delta
            bindings.append(
                self._resolver.record_evaluation(
                    f"{request.evaluation_id}::{arm}",
                    results[arm].resolution,
                    metrics=metrics,
                )
            )
        return tuple(bindings)


def _opponent_key(opponent: ResolvedOpponent) -> str:
    return f"{opponent.opponent_id}|{opponent.binding_kind}|{opponent.identity}"


def _proxy_request_id(group_id: str, sample_index: int, agent_instance_id: str) -> str:
    return "prid_" + digest([group_id, sample_index, agent_instance_id], length=24)


def _idempotency_key(
    evaluation_id: str, arm: str, held: HeldOutSeed, sample_index: int
) -> str:
    return "eval_" + digest([evaluation_id, arm, held.task_id, held.seed, sample_index], length=24)


def evaluate(
    resolver: EvaluationResolver,
    request: EvaluationRequest,
    *,
    session: ContainerSession,
    gateway: SamplerGateway,
    binder: PolicyBinder,
    receipts_dir: str | Path | None = None,
) -> EvaluationReceipt:
    """Run one paired evaluation and, when asked, persist its receipt."""

    receipt = PairedEvaluation(
        resolver, session=session, gateway=gateway, binder=binder
    ).run(request)
    if receipts_dir is not None:
        receipt.write(receipts_dir)
    return receipt


def seeds_from_pairs(pairs: Sequence[tuple[str, int]]) -> tuple[HeldOutSeed, ...]:
    """Build the held-out list both arms will run, in the order given."""

    return tuple(HeldOutSeed(task_id=task_id, seed=seed) for task_id, seed in pairs)


__all__ = [
    "ARMS",
    "ArmComparabilityError",
    "ArmResult",
    "AttemptFailedError",
    "AttemptRow",
    "BASELINE_ARM",
    "EVALUATION_RECEIPT_SCHEMA_VERSION",
    "EvaluationError",
    "EvaluationReceipt",
    "EvaluationRequest",
    "HeldOutSeed",
    "PairedEvaluation",
    "PairedRow",
    "PairedSummary",
    "PinTemplate",
    "RosterBindingError",
    "RosterSlot",
    "SamplerReferenceMismatchError",
    "TRAINED_ARM",
    "evaluate",
    "seeds_from_pairs",
]
