"""Paired evaluation: two arms, one held-out set, and nothing resolved by guess.

The suite holds two lines. A comparison must be a comparison -- the arms run
identical seeds through an identical roster against an identical pinned match
set, and the receipt carries both the selector asked for and the immutable id
it resolved to. And a refusal must happen *before* an attempt runs: a missing
artifact, a disagreeing digest, or a component in the wrong role ends the
evaluation with nothing submitted and no provider spend.

Everything is in process. The container half is the shared conformance fake
served on loopback; the sampler gateway and the policy binder are defined here,
because those two seams are what another stream owns.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import scenarios, serve
from fakes.container import ContainerClient, RunningContainer

from synth_optimizers.contracts.rl_identity import GroupPin, TaskSpec
from synth_optimizers.contracts.rl_records import (
    RendererProfile,
    RewardChannel,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
    digest,
)
from synth_optimizers.rl.catalog import (
    MUTABLE_SELECTOR_TOKENS,
    CheckpointArtifacts,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    SamplerWeightsRef,
    TrainingEvidence,
    TrainingStateRef,
)
from synth_optimizers.rl.evaluation import (
    BASELINE_ARM,
    TRAINED_ARM,
    ArmComparabilityError,
    EvaluationError,
    EvaluationRequest,
    HeldOutSeed,
    PairedEvaluation,
    PinTemplate,
    RosterBindingError,
    RosterSlot,
    SamplerReferenceMismatchError,
    evaluate,
)
from synth_optimizers.rl.policy_sets import (
    ComponentSaveAttempt,
    MatchSetRevision,
    OpponentBinding,
    PolicySetComponent,
    PolicySetPublisher,
    PolicySetRevision,
)
from synth_optimizers.rl.ports import AttemptFacts, PolicyRevision, SamplerOrigin
from synth_optimizers.rl.resolver import (
    ArtifactMissingError,
    DigestMismatchError,
    EvaluationResolver,
    MutableSelectorError,
    RoleMismatchError,
)

RENDERER: RendererProfile = scenarios.renderer_profile()
PACKED = ("pg_primary", "pg_second")
TASK_A = "row_0001"
TASK_B = "row_0002"


def sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


CONTRACT = sha("container_contract")


# --------------------------------------------------------------------------- #
# Catalog fixtures
# --------------------------------------------------------------------------- #


@dataclass
class Probe:
    """Mutable artifact probe, so a test can make the world disagree."""

    digests: dict[str, str]

    def exists(self, ref: str) -> bool:
        return ref in self.digests

    def digest_of(self, ref: str) -> str:
        return self.digests[ref]


def sampler_ref(checkpoint_id: str) -> SamplerWeightsRef:
    return SamplerWeightsRef(
        ref=f"provider://sampler/{checkpoint_id}", digest=sha(f"sampler:{checkpoint_id}")
    )


def state_ref(checkpoint_id: str) -> TrainingStateRef:
    return TrainingStateRef(
        ref=f"provider://state/{checkpoint_id}", digest=sha(f"state:{checkpoint_id}")
    )


def make_record(
    *,
    checkpoint_id: str,
    parameter_group_id: str = "pg_primary",
    policy_type_ids: tuple[str, ...] = ("type_primary",),
    policy_revision_id: str = "pg_primary@1",
    update_id: str = "update_0001",
    run_id: str = "run_a",
    publication_status: str = "staged",
    sampler: bool = True,
    training_state: bool = True,
    contract_hash: str = CONTRACT,
) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        update_id=update_id,
        train_call_ids=(f"provider_train_{checkpoint_id}",),
        parameter_group_id=parameter_group_id,
        policy_type_ids=policy_type_ids,
        policy_revision_id=policy_revision_id,
        base_model="vendor/base-model-a",
        artifacts=CheckpointArtifacts(
            sampler_weights=sampler_ref(checkpoint_id) if sampler else None,
            training_state=state_ref(checkpoint_id) if training_state else None,
        ),
        training_evidence=TrainingEvidence(
            groups=PACKED, examples=16, tokens=65536, provider_cost=0.75
        ),
        compatibility=CheckpointCompatibility(
            renderer_profile=RENDERER.profile_id,
            tokenizer=RENDERER.tokenizer_id,
            container_contract_hash=contract_hash,
        ),
        created_at="2026-09-02T00:00:00Z",
        publication_status=publication_status,
    )


def probe_for(*records: CheckpointRecord) -> Probe:
    digests: dict[str, str] = {}
    for record in records:
        for reference in (record.artifacts.sampler_weights, record.artifacts.training_state):
            if reference is not None:
                digests[reference.ref] = reference.digest
    return Probe(digests=digests)


@dataclass
class World:
    catalog: CheckpointCatalog
    publisher: PolicySetPublisher
    probe: Probe
    resolver: EvaluationResolver
    records: dict[str, CheckpointRecord]
    contract_hash: str


def build_world(tmp_path, *, contract_hash: str = CONTRACT) -> World:
    """A baseline checkpoint, one published two-component set, and its match set."""

    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    publisher = PolicySetPublisher(catalog, health_check=lambda request: True)
    records: dict[str, CheckpointRecord] = {}

    baseline = make_record(
        checkpoint_id="ckpt_baseline_primary",
        update_id="update_0000",
        policy_revision_id="pg_primary@0",
        publication_status="published",
        training_state=False,
        contract_hash=contract_hash,
    )
    catalog.register_baseline(baseline)
    records[baseline.checkpoint_id] = baseline

    baseline_second = make_record(
        checkpoint_id="ckpt_baseline_second",
        parameter_group_id="pg_second",
        policy_type_ids=("type_second",),
        policy_revision_id="pg_second@0",
        update_id="update_0000",
        publication_status="published",
        training_state=False,
        contract_hash=contract_hash,
    )
    catalog.register_checkpoint(baseline_second)
    catalog.record_publication(baseline_second.checkpoint_id, "published")
    records[baseline_second.checkpoint_id] = baseline_second

    baseline_set = PolicySetRevision(
        policy_set_revision_id="set-baseline",
        policy_set_id="team_set",
        run_id="run_a",
        update_id="update_0000",
        components=(
            PolicySetComponent(
                policy_type_id="type_primary",
                parameter_group_id="pg_primary",
                checkpoint_id="ckpt_baseline_primary",
                policy_revision_id="pg_primary@0",
            ),
            PolicySetComponent(
                policy_type_id="type_second",
                parameter_group_id="pg_second",
                checkpoint_id="ckpt_baseline_second",
                policy_revision_id="pg_second@0",
            ),
        ),
        created_at="2026-09-02T00:10:00Z",
    )
    publisher.publish(baseline_set)
    publisher.mark_loaded("set-baseline")
    publisher.mark_ready("set-baseline")

    trained_set = PolicySetRevision(
        policy_set_revision_id="set-trained",
        policy_set_id="team_set",
        run_id="run_a",
        update_id="update_0001",
        components=(
            PolicySetComponent(
                policy_type_id="type_primary",
                parameter_group_id="pg_primary",
                checkpoint_id="ckpt_primary_u1",
                policy_revision_id="pg_primary@1",
            ),
            PolicySetComponent(
                policy_type_id="type_second",
                parameter_group_id="pg_second",
                checkpoint_id="ckpt_second_u1",
                policy_revision_id="pg_second@1",
            ),
        ),
        created_at="2026-09-02T00:30:00Z",
    )
    components = [
        make_record(
            checkpoint_id=item.checkpoint_id,
            parameter_group_id=item.parameter_group_id,
            policy_type_ids=(item.policy_type_id,),
            policy_revision_id=item.policy_revision_id,
            contract_hash=contract_hash,
        )
        for item in trained_set.components
    ]
    publisher.publish_round(
        trained_set,
        tuple(
            ComponentSaveAttempt(
                parameter_group_id=record.parameter_group_id,
                record=record,
                packed_group_ids=PACKED,
            )
            for record in components
        ),
    )
    publisher.mark_loaded("set-trained")
    publisher.mark_ready("set-trained")
    for record in components:
        records[record.checkpoint_id] = record

    frozen = make_record(
        checkpoint_id="ckpt_frozen_opponent",
        parameter_group_id="pg_opponent",
        policy_type_ids=("type_opponent",),
        policy_revision_id="pg_opponent@7",
        publication_status="published",
        contract_hash=contract_hash,
    )
    catalog.register_checkpoint(frozen)
    records[frozen.checkpoint_id] = frozen

    matches = (("match-baseline", "set-baseline"), ("match-trained", "set-trained"))
    for revision_id, policy_set in matches:
        publisher.publish_match_set(
            MatchSetRevision(
                match_set_revision_id=revision_id,
                match_set_id="match_set",
                run_id="run_a",
                policy_set_revision_id=policy_set,
                opponents=(
                    OpponentBinding(
                        opponent_id="opponent_a",
                        binding_kind="pinned_checkpoint",
                        identity="ckpt_frozen_opponent",
                    ),
                ),
                created_at="2026-09-02T01:00:00Z",
            )
        )
        publisher.mark_loaded(revision_id)
        publisher.mark_ready(revision_id)

    catalog.put_alias("champion", "checkpoint", "ckpt_primary_u1")
    catalog.put_alias("champion-set", "policy_set", "set-trained")

    probe = probe_for(*records.values())
    resolver = EvaluationResolver(catalog, probe=probe)
    return World(
        catalog=catalog,
        publisher=publisher,
        probe=probe,
        resolver=resolver,
        records=records,
        contract_hash=contract_hash,
    )


@pytest.fixture()
def world(tmp_path) -> World:
    built = build_world(tmp_path)
    yield built
    built.catalog.close()


# --------------------------------------------------------------------------- #
# The two seams another stream owns, faked in process
# --------------------------------------------------------------------------- #


def revision_number(policy_revision_id: str) -> int:
    _, _, tail = policy_revision_id.partition("@")
    return int(tail or 0)


class FakeBinder:
    """Bridges the catalog to 'materialized' revisions. No provider is called."""

    def __init__(self, catalog: CheckpointCatalog, *, reference_drift: str | None = None) -> None:
        self._catalog = catalog
        self._drift = reference_drift
        self.resolved: list[str] = []

    def baseline(self, *, run_id: str, parameter_group_id: str) -> PolicyRevision:
        raise NotImplementedError("evaluation never mints a baseline")

    def train(self, **_kwargs: Any) -> Any:
        raise NotImplementedError("evaluation never trains")

    def publish(self, **_kwargs: Any) -> Any:
        raise NotImplementedError("evaluation never publishes")

    def resolve(self, selector: str) -> Mapping[str, PolicyRevision]:
        self.resolved.append(selector)
        return {
            record.parameter_group_id: self._revision(record, selector)
            for record in self._records(selector)
        }

    def _records(self, selector: str) -> tuple[CheckpointRecord, ...]:
        if self._catalog.has_checkpoint(selector):
            return (self._catalog.get_checkpoint(selector),)
        row = self._catalog.get_revision(selector)
        if row.revision_kind == "match_set":
            match = MatchSetRevision.from_payload(row.payload)
            row = self._catalog.get_revision(match.policy_set_revision_id)
        policy_set = PolicySetRevision.from_payload(row.payload)
        return tuple(
            self._catalog.get_checkpoint(item.checkpoint_id) for item in policy_set.components
        )

    def _revision(self, record: CheckpointRecord, selector: str) -> PolicyRevision:
        reference = record.sampler_weights.ref
        if self._drift is not None and record.checkpoint_id == self._drift:
            reference = "provider://sampler/somewhere-else"
        return PolicyRevision(
            revision=revision_number(record.policy_revision_id),
            revision_id=record.policy_revision_id,
            checkpoint_id=record.checkpoint_id,
            parameter_group_id=record.parameter_group_id,
            sampler_reference=reference,
            behavior_fingerprint=digest([RENDERER.fingerprint, record.checkpoint_id], length=32),
            policy_set_revision_id=selector if selector.startswith("set-") else None,
        )


class FakeGateway:
    """Owns the renderer. Origins are per attempt and never reused."""

    def __init__(self, profile: RendererProfile = RENDERER) -> None:
        self._profile = profile
        self._open: dict[str, tuple[SamplerOrigin, str]] = {}
        self.bound: list[tuple[str, str]] = []
        self.closed: list[str] = []
        self.facts: list[AttemptFacts] = []

    @property
    def renderer_profile(self) -> RendererProfile:
        return self._profile

    def bind(
        self,
        revision: PolicyRevision,
        *,
        pin: GroupPin,
        sample_index: int,
        proxy_request_id: str,
        attempt: AttemptFacts,
    ) -> SamplerOrigin:
        self.facts.append(attempt)
        existing = self._open.get(proxy_request_id)
        if existing is not None:
            if existing[1] != revision.revision_id:
                raise AssertionError("a route may never be rebound to a second revision")
            return existing[0]
        origin = SamplerOrigin(
            base_url=f"http://sampler.invalid/{proxy_request_id}",
            credential=f"cred::{proxy_request_id}",
            policy_revision=revision.revision,
            behavior_fingerprint=revision.behavior_fingerprint,
            proxy_request_id=proxy_request_id,
            wire_api=pin.wire_api,
            sampling_transport=pin.sampling_transport,
        )
        self._open[proxy_request_id] = (origin, revision.revision_id)
        self.bound.append((proxy_request_id, revision.checkpoint_id))
        _ = sample_index
        return origin

    def close(self, proxy_request_id: str) -> None:
        self._open.pop(proxy_request_id, None)
        self.closed.append(proxy_request_id)

    def episode(self, proxy_request_id: str) -> TrainableEpisode:
        raise NotImplementedError("the container seals the episode in this suite")


def segment(revision: int) -> TrainableSegment:
    return TrainableSegment(
        token_ids=(11, 12, 13),
        loss_mask=(0, 1, 1),
        behavior_logprobs=(-0.1, -0.2, -0.3),
        parameter_group_id="pg_primary",
        policy_revision=revision,
    )


class FakeSession:
    """An admitted container. Its reward moves with the policy, as a real one does."""

    def __init__(self, *, seeds: Mapping[str, int]) -> None:
        self._seeds = dict(seeds)
        self.submitted: list[dict[str, Any]] = []
        self.terminated: list[str] = []
        self._by_rollout: dict[str, dict[str, Any]] = {}
        self._by_key: dict[str, str] = {}

    @property
    def handshake_id(self) -> str:
        return "hs_fake"

    @property
    def agreement_digest(self) -> str:
        return sha("agreement")

    def tasks(self, *, split: str, task_ids: Sequence[str]) -> tuple[TaskSpec, ...]:
        return tuple(
            TaskSpec(
                task_id=task_id,
                split=split,
                seed=self._seeds[task_id],
                group_id="group_heldout",
                task_family="family_a",
                content_digest=sha(task_id),
            )
            for task_id in task_ids
            if task_id in self._seeds
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
        if idempotency_key in self._by_key:
            return self._by_key[idempotency_key]
        rollout_id = f"ro_{len(self.submitted):04d}"
        record = {
            "rollout_id": rollout_id,
            "task_id": task.task_id,
            "seed": task.seed,
            "group_id": pin.group_id,
            "policy_revision": origin.policy_revision,
            "policy_set_revision_id": pin.policy_set_revision_id,
            "match_set_revision_id": pin.match_set_revision_id,
            "sample_index": sample_index,
            "proxy_request_id": origin.proxy_request_id,
        }
        self.submitted.append(record)
        self._by_rollout[rollout_id] = record
        self._by_key[idempotency_key] = rollout_id
        return rollout_id

    def poll(self, rollout_id: str) -> Mapping[str, Any]:
        return {"rollout_id": rollout_id, "state": "scored", "terminal": False}

    def renew(self, rollout_id: str) -> Mapping[str, Any]:
        return {"rollout_id": rollout_id}

    def finalize(self, rollout_id: str) -> Mapping[str, Any]:
        return {"rollout_id": rollout_id, "state": "completed", "terminal": True}

    def terminate(self, rollout_id: str, *, reason: str) -> Any:
        self.terminated.append(f"{rollout_id}:{reason}")
        return None

    def evidence(self, rollout_id: str) -> tuple[TrainableEpisode, RewardRecord]:
        record = self._by_rollout[rollout_id]
        revision = int(record["policy_revision"])
        measure = round(0.4 + 0.2 * revision + 0.01 * int(record["seed"]), 6)
        trace_digest = digest([rollout_id, "trace"], length=32)
        episode = TrainableEpisode(
            rollout_id=rollout_id,
            task_id=str(record["task_id"]),
            seed=int(record["seed"]),
            policy_revision=revision,
            behavior_fingerprint=digest([rollout_id, "behavior"], length=32),
            segments=(segment(revision),),
            terminal_status="completed",
            trace_digest=trace_digest,
            usage={
                "calls": 1,
                "prompt_tokens": 10 + int(record["seed"]),
                "completion_tokens": 2,
                "provider_request_ids": [str(record["proxy_request_id"])],
            },
        )
        reward = RewardRecord(
            reward_id=f"reward_{rollout_id}",
            rollout_id=rollout_id,
            trace_digest=trace_digest,
            channels=(
                RewardChannel(channel_id="task_reward", team_id="team_solo", measure=measure),
            ),
            optimized_channel="task_reward",
            terminal_status="completed",
            evaluation_plan_id="plan_heldout",
        )
        return episode, reward


# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #


PIN = PinTemplate(
    run_id="run_a",
    algorithm_plan_hash="plan#alpha",
    wire_api="chat_completions",
    sampling_transport="text_in_text_out",
    policy_kind="lora",
    model_family="vendor/base-model-a",
    container_image_digest=sha("image"),
    container_contract_hash=CONTRACT,
    task_family="family_a",
)

SEEDS = (HeldOutSeed(task_id=TASK_A, seed=3), HeldOutSeed(task_id=TASK_B, seed=5))
SOLO_ROSTER = (RosterSlot(agent_instance_id="inst_a", parameter_group_id="pg_primary"),)
TEAM_ROSTER = (
    RosterSlot(
        agent_instance_id="inst_a", parameter_group_id="pg_primary", policy_type_id="type_primary"
    ),
    RosterSlot(
        agent_instance_id="inst_b", parameter_group_id="pg_second", policy_type_id="type_second"
    ),
)


def request_for(
    *,
    trained: str,
    baseline: str,
    roster: tuple[RosterSlot, ...] = SOLO_ROSTER,
    match_set: str | None = None,
    evaluation_id: str = "eval_0001",
    pin: PinTemplate = PIN,
) -> EvaluationRequest:
    return EvaluationRequest(
        evaluation_id=evaluation_id,
        baseline_selector=baseline,
        trained_selector=trained,
        seeds=SEEDS,
        roster=roster,
        pin=pin,
        match_set_selector=match_set,
    )


def session_for() -> FakeSession:
    return FakeSession(seeds={TASK_A: 3, TASK_B: 5})


def run_evaluation(world: World, request: EvaluationRequest, **kwargs: Any):
    session = kwargs.pop("session", None) or session_for()
    gateway = kwargs.pop("gateway", None) or FakeGateway()
    binder = kwargs.pop("binder", None) or FakeBinder(world.catalog, **kwargs)
    receipt = PairedEvaluation(
        world.resolver, session=session, gateway=gateway, binder=binder
    ).run(request)
    return receipt, session, gateway, binder


# --------------------------------------------------------------------------- #
# Paired evaluation
# --------------------------------------------------------------------------- #


def test_paired_arms_run_identical_seeds_and_produce_a_comparable_summary(world: World) -> None:
    receipt, session, gateway, _ = run_evaluation(
        world,
        request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
    )

    baseline_keys = [(row.task_id, row.seed) for row in receipt.baseline.attempts]
    trained_keys = [(row.task_id, row.seed) for row in receipt.trained.attempts]
    assert baseline_keys == trained_keys == [(TASK_A, 3), (TASK_B, 5)]

    summary = receipt.summary
    assert summary.pairs == 2
    assert summary.trained_mean > summary.baseline_mean
    assert summary.mean_delta == pytest.approx(0.2)
    assert (summary.wins, summary.losses, summary.ties) == (2, 0, 0)

    # Both arms ran through the same roster, and every origin was retired.
    assert len(session.submitted) == 4
    assert {row["group_id"] for row in session.submitted} == {
        "eval_0001::baseline",
        "eval_0001::trained",
    }
    assert sorted(gateway.closed) == sorted(item[0] for item in gateway.bound)


def test_concurrent_eval_preserves_order_and_caps_inflight(world: World) -> None:
    class DelayedSession(FakeSession):
        obligations = SimpleNamespace(max_concurrency=2)

        def __init__(self):
            super().__init__(seeds={TASK_A: 3, TASK_B: 5})
            self.active = set()
            self.peak = 0
            self.observations = {}
            self.finished = []

        def submit(self, *args, **kwargs):
            rid = super().submit(*args, **kwargs)
            self.active.add(rid)
            self.peak = max(self.peak, len(self.active))
            return rid

        def poll(self, rid):
            self.observations[rid] = self.observations.get(rid, 0) + 1
            if self._by_rollout[rid]["sample_index"] == 0 and self.observations[rid] == 1:
                return {"state": "running"}
            return super().poll(rid)

        def finalize(self, rid):
            self.active.remove(rid)
            self.finished.append(self._by_rollout[rid]["sample_index"])
            return super().finalize(rid)

    session = DelayedSession()
    request = replace(request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"), concurrency=8)
    receipt, _, gateway, _ = run_evaluation(world, request, session=session)
    assert session.peak == 2
    assert session.finished == [1, 0, 1, 0]
    assert [r.sample_index for r in receipt.baseline.attempts] == [0, 1]
    assert [r.sample_index for r in receipt.trained.attempts] == [0, 1]
    assert not session.active
    assert len(gateway.closed) == 4


def test_concurrent_eval_cleans_up_on_failure(world: World) -> None:
    class BrokenSession(FakeSession):
        obligations = SimpleNamespace(max_concurrency=2)

        def poll(self, rid):
            raise RuntimeError("transport failed")

    session = BrokenSession(seeds={TASK_A: 3, TASK_B: 5})
    gateway = FakeGateway()
    with pytest.raises(RuntimeError, match="transport failed"):
        run_evaluation(world, replace(request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"), concurrency=2), session=session, gateway=gateway)
    assert len(session.terminated) == 2
    assert len(gateway.closed) == 2


def test_evaluation_receipt_records_exact_injected_timing_and_throughput(world: World) -> None:
    utc_values = iter(("2026-09-04T12:00:00Z", "2026-09-04T12:00:08Z"))
    monotonic_values = iter((100.0, 108.0))
    receipt = PairedEvaluation(
        world.resolver,
        session=session_for(),
        gateway=FakeGateway(),
        binder=FakeBinder(world.catalog),
        clock=lambda: next(utc_values),
        monotonic_clock=lambda: next(monotonic_values),
    ).run(request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"))

    assert receipt.started_at == "2026-09-04T12:00:00Z"
    assert receipt.finished_at == receipt.created_at == "2026-09-04T12:00:08Z"
    assert receipt.duration_seconds == 8.0
    assert receipt.attempt_count == 4
    assert receipt.attempts_per_second == 0.5
    assert receipt.to_payload()["attempts_per_second"] == 0.5
    assert receipt.baseline.attempts[0].usage["provider_request_ids"]
    assert receipt.to_payload()["usage_totals"] == {
        "calls": 4,
        "prompt_tokens": 56,
        "completion_tokens": 8,
        "total_tokens": 64,
    }
    assert receipt.to_payload()["arms"][BASELINE_ARM]["usage_totals"] == {
        "calls": 2,
        "prompt_tokens": 28,
        "completion_tokens": 4,
        "total_tokens": 32,
    }


@pytest.mark.parametrize("finished", [100.0, 99.0])
def test_evaluation_receipt_has_null_rate_for_nonpositive_duration(
    world: World, finished: float
) -> None:
    monotonic_values = iter((100.0, finished))
    receipt = PairedEvaluation(
        world.resolver,
        session=session_for(),
        gateway=FakeGateway(),
        binder=FakeBinder(world.catalog),
        clock=lambda: "2026-09-04T12:00:00Z",
        monotonic_clock=lambda: next(monotonic_values),
    ).run(request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"))

    assert receipt.attempts_per_second is None


def test_every_origin_is_bound_with_the_task_and_seed_it_will_run(world: World) -> None:
    _, _, gateway, _ = run_evaluation(
        world,
        request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
    )
    # The gateway is told what the attempt is, not left to infer it from the pin.
    assert [(fact.task_id, fact.seed) for fact in gateway.facts] == [
        (TASK_A, 3),
        (TASK_B, 5),
        (TASK_A, 3),
        (TASK_B, 5),
    ]
    assert len({fact.rollout_id for fact in gateway.facts}) == 4


def test_awaiting_score_reaches_the_finalize_barrier(world: World) -> None:
    class AwaitingScoreSession(FakeSession):
        def __init__(self, *, seeds: Mapping[str, int]) -> None:
            super().__init__(seeds=seeds)
            self.finalized: list[str] = []

        def poll(self, rollout_id: str) -> Mapping[str, Any]:
            return {"rollout_id": rollout_id, "state": "awaiting_score", "terminal": False}

        def finalize(self, rollout_id: str) -> Mapping[str, Any]:
            self.finalized.append(rollout_id)
            return super().finalize(rollout_id)

    session = AwaitingScoreSession(seeds={TASK_A: 3, TASK_B: 5})
    receipt, _, _, _ = run_evaluation(
        world,
        request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
        session=session,
    )

    assert receipt.summary.pairs == 2
    assert session.finalized == ["ro_0000", "ro_0001", "ro_0002", "ro_0003"]


def test_receipt_carries_selector_resolution_refs_seeds_and_rewards(world: World) -> None:
    receipt, _, _, _ = run_evaluation(
        world,
        request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
    )
    payload = receipt.to_payload()

    assert payload["schema_version"] == "cispo.evaluation_receipt.v1"
    assert payload["seeds"] == [
        {"task_id": TASK_A, "seed": 3},
        {"task_id": TASK_B, "seed": 5},
    ]
    trained = payload["arms"][TRAINED_ARM]
    assert trained["requested_selector"] == "ckpt_primary_u1"
    assert trained["resolved_id"] == "ckpt_primary_u1"
    assert trained["resolution"]["resolved_checkpoint_ids"] == ["ckpt_primary_u1"]
    assert trained["loaded_sampler_references"] == ["provider://sampler/ckpt_primary_u1"]
    assert trained["catalogued_sampler_references"] == trained["loaded_sampler_references"]
    assert [row["reward"] for row in trained["attempts"]] == [
        pytest.approx(0.63),
        pytest.approx(0.65),
    ]
    assert payload["paired_summary"]["pairs"] == 2


def test_receipt_is_written_to_the_run_artifact_directory(world: World, tmp_path) -> None:
    receipt = evaluate(
        world.resolver,
        request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
        session=session_for(),
        gateway=FakeGateway(),
        binder=FakeBinder(world.catalog),
        receipts_dir=tmp_path / "receipts",
    )
    written = tmp_path / "receipts" / "eval_0001.evaluation.json"
    assert written.is_file()
    assert receipt.evaluation_id in written.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Resolution by each selector kind
# --------------------------------------------------------------------------- #


def test_resolution_by_checkpoint_id(world: World) -> None:
    receipt, _, _, _ = run_evaluation(
        world, request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary")
    )
    assert receipt.trained.resolution.resolved_kind == "checkpoint"
    assert receipt.selector_resolutions == (
        ("ckpt_baseline_primary", "ckpt_baseline_primary"),
        ("ckpt_primary_u1", "ckpt_primary_u1"),
    )


def test_resolution_by_policy_set_revision_binds_the_whole_team(world: World) -> None:
    receipt, session, gateway, _ = run_evaluation(
        world,
        request_for(trained="set-trained", baseline="set-baseline", roster=TEAM_ROSTER),
    )
    assert receipt.trained.resolution.resolved_kind == "policy_set"
    assert set(receipt.trained.resolution.checkpoint_ids) == {
        "ckpt_primary_u1",
        "ckpt_second_u1",
    }
    assert receipt.trained.loaded_sampler_references == (
        "provider://sampler/ckpt_primary_u1",
        "provider://sampler/ckpt_second_u1",
    )
    # Two roster slots bound per attempt, four attempts.
    assert len(gateway.bound) == 8
    assert len(session.submitted) == 4
    assert {row["policy_set_revision_id"] for row in session.submitted} == {
        "set-baseline",
        "set-trained",
    }


def test_resolution_by_match_set_revision_pins_the_opponent_for_both_arms(world: World) -> None:
    receipt, session, _, _ = run_evaluation(
        world,
        request_for(
            trained="match-trained",
            baseline="match-baseline",
            roster=TEAM_ROSTER,
            match_set="match-trained",
        ),
    )
    assert receipt.trained.resolution.resolved_kind == "match_set"
    assert receipt.match_set_revision_id == "match-trained"
    assert [opponent.identity for opponent in receipt.opponents] == ["ckpt_frozen_opponent"]
    assert {row["match_set_revision_id"] for row in session.submitted} == {"match-trained"}


def test_arms_pinned_to_different_match_sets_are_refused(world: World) -> None:
    session = session_for()
    with pytest.raises(ArmComparabilityError) as raised:
        run_evaluation(
            world,
            request_for(
                trained="match-trained", baseline="match-baseline", roster=TEAM_ROSTER
            ),
            session=session,
        )
    assert "match-baseline" in str(raised.value)
    assert session.submitted == []


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #


def test_alias_is_recorded_as_both_the_selector_and_the_immutable_id(world: World) -> None:
    receipt, _, _, _ = run_evaluation(
        world, request_for(trained="champion", baseline="ckpt_baseline_primary")
    )
    resolution = receipt.trained.resolution
    assert resolution.requested_selector == "champion"
    assert resolution.alias == "champion"
    assert resolution.resolved_id == "ckpt_primary_u1"

    payload = receipt.to_payload()["arms"][TRAINED_ARM]["resolution"]
    assert payload["requested_selector"] == "champion"
    assert payload["alias"] == "champion"
    assert payload["resolved_id"] == "ckpt_primary_u1"

    binding = next(
        item for item in receipt.bindings if item.evaluation_id.endswith(f"::{TRAINED_ARM}")
    )
    assert binding.requested_selector == "champion"
    assert binding.target_id == "ckpt_primary_u1"


def test_a_policy_set_alias_resolves_to_its_immutable_revision(world: World) -> None:
    receipt, _, _, _ = run_evaluation(
        world,
        request_for(trained="champion-set", baseline="set-baseline", roster=TEAM_ROSTER),
    )
    assert receipt.trained.resolution.alias == "champion-set"
    assert receipt.trained.resolved_id == "set-trained"


# --------------------------------------------------------------------------- #
# Refusals, every one of them before an attempt runs
# --------------------------------------------------------------------------- #


def test_digest_mismatch_is_refused_before_any_attempt(world: World) -> None:
    world.probe.digests["provider://sampler/ckpt_primary_u1"] = sha("someone_else")
    session = session_for()
    binder = FakeBinder(world.catalog)
    with pytest.raises(DigestMismatchError) as raised:
        run_evaluation(
            world,
            request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
            session=session,
            binder=binder,
        )
    assert "ckpt_primary_u1" in str(raised.value)
    assert session.submitted == []
    assert binder.resolved == []
    assert world.catalog.evaluations() == ()


def test_missing_artifact_is_refused_before_any_attempt(world: World) -> None:
    del world.probe.digests["provider://sampler/ckpt_second_u1"]
    session = session_for()
    binder = FakeBinder(world.catalog)
    with pytest.raises(ArtifactMissingError):
        run_evaluation(
            world,
            request_for(trained="set-trained", baseline="set-baseline", roster=TEAM_ROSTER),
            session=session,
            binder=binder,
        )
    assert session.submitted == []
    assert binder.resolved == []


def test_a_checkpoint_with_no_sampler_artifact_is_a_role_mismatch(world: World, tmp_path) -> None:
    resumable_only = make_record(
        checkpoint_id="ckpt_state_only",
        policy_revision_id="pg_primary@2",
        update_id="update_0002",
        publication_status="published",
        sampler=False,
    )
    world.catalog.register_checkpoint(resumable_only)
    world.probe.digests[resumable_only.artifacts.resumable.ref] = (
        resumable_only.artifacts.resumable.digest
    )
    session = session_for()
    with pytest.raises(RoleMismatchError) as raised:
        run_evaluation(
            world,
            request_for(trained="ckpt_state_only", baseline="ckpt_baseline_primary"),
            session=session,
        )
    assert "sampler_weights" in str(raised.value)
    assert session.submitted == []


def test_a_roster_slot_bound_to_the_wrong_policy_type_is_refused(world: World) -> None:
    session = session_for()
    roster = (
        RosterSlot(
            agent_instance_id="inst_a",
            parameter_group_id="pg_primary",
            policy_type_id="type_second",
        ),
    )
    with pytest.raises(RosterBindingError) as raised:
        run_evaluation(
            world,
            request_for(
                trained="ckpt_primary_u1", baseline="ckpt_baseline_primary", roster=roster
            ),
            session=session,
        )
    assert "type_second" in str(raised.value)
    assert session.submitted == []


def test_a_roster_slot_with_no_policy_in_the_resolution_is_refused(world: World) -> None:
    session = session_for()
    roster = (RosterSlot(agent_instance_id="inst_x", parameter_group_id="pg_absent"),)
    with pytest.raises(RosterBindingError):
        run_evaluation(
            world,
            request_for(
                trained="ckpt_primary_u1", baseline="ckpt_baseline_primary", roster=roster
            ),
            session=session,
        )
    assert session.submitted == []


def test_a_binder_that_loads_another_reference_is_an_evidence_failure(world: World) -> None:
    session = session_for()
    with pytest.raises(SamplerReferenceMismatchError) as raised:
        run_evaluation(
            world,
            request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
            session=session,
            binder=FakeBinder(world.catalog, reference_drift="ckpt_primary_u1"),
        )
    assert "somewhere-else" in str(raised.value)
    assert session.submitted == []


def test_a_seed_the_container_does_not_declare_is_refused(world: World) -> None:
    session = FakeSession(seeds={TASK_A: 3, TASK_B: 99})
    with pytest.raises(EvaluationError) as raised:
        run_evaluation(
            world,
            request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"),
            session=session,
        )
    assert "identical one" in str(raised.value)
    assert session.submitted == []


@pytest.mark.parametrize("token", sorted(MUTABLE_SELECTOR_TOKENS))
def test_no_path_reaches_a_mutable_selector(world: World, token: str) -> None:
    session = session_for()
    with pytest.raises(MutableSelectorError):
        run_evaluation(
            world,
            request_for(trained=token, baseline="ckpt_baseline_primary"),
            session=session,
        )
    with pytest.raises(MutableSelectorError):
        run_evaluation(
            world,
            request_for(trained="ckpt_primary_u1", baseline=token),
            session=session,
        )
    assert session.submitted == []


def test_a_resolved_receipt_never_names_a_mutable_pointer(world: World) -> None:
    receipt, _, _, _ = run_evaluation(
        world, request_for(trained="champion", baseline="ckpt_baseline_primary")
    )
    payload = receipt.to_payload()
    for arm in (BASELINE_ARM, TRAINED_ARM):
        resolution = payload["arms"][arm]["resolution"]
        assert resolution["resolved_id"].lower() not in MUTABLE_SELECTOR_TOKENS
        for reference in resolution["loaded_refs"]:
            assert "latest" not in reference


# --------------------------------------------------------------------------- #
# Append-only relation
# --------------------------------------------------------------------------- #


def test_evaluation_binding_is_appended_and_the_checkpoint_is_untouched(world: World) -> None:
    before = world.catalog.get_checkpoint("ckpt_primary_u1").record_digest
    run_evaluation(
        world,
        request_for(
            trained="ckpt_primary_u1", baseline="ckpt_baseline_primary", evaluation_id="eval_a"
        ),
    )
    run_evaluation(
        world,
        request_for(
            trained="ckpt_primary_u1", baseline="ckpt_baseline_primary", evaluation_id="eval_b"
        ),
    )

    assert world.catalog.get_checkpoint("ckpt_primary_u1").record_digest == before
    bindings = world.catalog.evaluations(checkpoint_id="ckpt_primary_u1")
    assert [binding.evaluation_id for binding in bindings] == [
        f"eval_a::{TRAINED_ARM}",
        f"eval_b::{TRAINED_ARM}",
    ]
    assert all(binding.target_id == "ckpt_primary_u1" for binding in bindings)
    assert all("mean_reward" in binding.metrics for binding in bindings)
    assert bindings[0].loaded_refs == ("provider://sampler/ckpt_primary_u1",)

    view = world.catalog.describe_checkpoint("ckpt_primary_u1")
    assert set(view.evaluation_ids) == {f"eval_a::{TRAINED_ARM}", f"eval_b::{TRAINED_ARM}"}
    # The metric index now reaches this checkpoint.
    ranked = world.catalog.metric_rows("mean_reward", target_kind="checkpoint")
    assert ("checkpoint", "ckpt_primary_u1", pytest.approx(0.64)) in ranked


def test_both_arms_are_recorded_as_relations(world: World) -> None:
    receipt, _, _, _ = run_evaluation(
        world,
        request_for(
            trained="ckpt_primary_u1", baseline="ckpt_baseline_primary", evaluation_id="eval_c"
        ),
    )
    assert [binding.evaluation_id for binding in receipt.bindings] == [
        f"eval_c::{BASELINE_ARM}",
        f"eval_c::{TRAINED_ARM}",
    ]
    trained = receipt.bindings[1]
    assert trained.metrics["paired_mean_delta"] == pytest.approx(0.2)
    baseline = world.catalog.evaluations(checkpoint_id="ckpt_baseline_primary")
    assert [binding.evaluation_id for binding in baseline] == [f"eval_c::{BASELINE_ARM}"]


# --------------------------------------------------------------------------- #
# Against the shared conformance container
# --------------------------------------------------------------------------- #


@dataclass
class ContainerBackedSession:
    """The conformance fake's declared routes, behind the session seam.

    Only routes the container advertised are called, and the reward comes from
    the container's own receipt rather than from anything computed here.
    """

    client: ContainerClient
    submitted: list[str] = field(default_factory=list)
    _bindings: dict[str, str] = field(default_factory=dict)

    @property
    def handshake_id(self) -> str:
        return str(self.client.handshake_id)

    @property
    def agreement_digest(self) -> str:
        return str(self.client.agreement_digest)

    def tasks(self, *, split: str, task_ids: Sequence[str]) -> tuple[TaskSpec, ...]:
        rows = self.client.taskset_tasks(task_ids)["rows"]
        return tuple(
            TaskSpec(
                task_id=str(row["task_id"]),
                split=split,
                seed=int(row["seed"]),
                group_id="group_heldout",
                task_family=str(row["task_family"]),
                content_digest=str(row["content_digest"]),
                topology_ref=row.get("topology_ref"),
            )
            for row in rows
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
        binding = self.client.bind_policy(
            kind="trainable",
            policy_revision=origin.policy_revision,
            transport=origin.sampling_transport,
            sampler_origin_url=origin.base_url,
        )
        payload = self.client.submit(
            task_id=task.task_id,
            idempotency_key=idempotency_key,
            policy_config_id=binding["config_id"],
            correlation={
                "run_id": pin.run_id,
                "group_id": pin.group_id,
                "sample_index": sample_index,
                "seed": task.seed,
                "policy_revision": origin.policy_revision,
            },
        )
        rollout_id = str(payload["rollout_id"])
        self.submitted.append(rollout_id)
        return rollout_id

    def poll(self, rollout_id: str) -> Mapping[str, Any]:
        return self.client.state(rollout_id)

    def renew(self, rollout_id: str) -> Mapping[str, Any]:
        return self.client.renew(rollout_id)

    def finalize(self, rollout_id: str) -> Mapping[str, Any]:
        return self.client.finalize(rollout_id)

    def terminate(self, rollout_id: str, *, reason: str) -> Any:
        return self.client.terminate(rollout_id, reason)

    def evidence(self, rollout_id: str) -> tuple[TrainableEpisode, RewardRecord]:
        from fakes.container import reward_record_from_payload, trainable_episode_from_payload

        trace = self.client.trace(rollout_id)
        status, reward_payload = self.client.reward(rollout_id)
        if status != 200 or reward_payload is None:
            raise EvaluationError(f"rollout {rollout_id} produced no reward; absent is not zero")
        episode = trainable_episode_from_payload((trace.get("episodes") or [])[0])
        return episode, reward_record_from_payload(reward_payload)


@pytest.fixture()
def container() -> RunningContainer:
    running = serve(scenarios.one_call_classification())
    yield running
    running.shutdown()


def test_paired_evaluation_drives_a_declared_route_container(tmp_path, container) -> None:
    client = container.client()
    exchanges = client.negotiate()
    assert exchanges[-1]["accepted"]
    home = tmp_path / "world"
    home.mkdir()
    built = build_world(home, contract_hash=container.config.contract_hash)
    try:
        session = ContainerBackedSession(client=client)
        rows = session.tasks(split="eval", task_ids=[TASK_A, TASK_B])
        pin = PinTemplate(
            run_id="run_a",
            algorithm_plan_hash="plan#alpha",
            wire_api=container.config.wire_api,
            sampling_transport=container.config.sampling_transport,
            policy_kind="lora",
            model_family="vendor/base-model-a",
            container_image_digest=container.config.image_digest,
            container_contract_hash=container.config.contract_hash,
            task_family=container.config.task_family,
            topology_id=container.config.topology.topology_id,
        )
        request = EvaluationRequest(
            evaluation_id="eval_container",
            baseline_selector="ckpt_baseline_primary",
            trained_selector="ckpt_primary_u1",
            seeds=tuple(HeldOutSeed(task_id=row.task_id, seed=row.seed) for row in rows),
            roster=SOLO_ROSTER,
            pin=pin,
            split="eval",
        )
        receipt = PairedEvaluation(
            built.resolver,
            session=session,
            gateway=FakeGateway(profile=container.config.renderer_profile),
            binder=FakeBinder(built.catalog),
        ).run(request)
    finally:
        built.catalog.close()

    assert len(session.submitted) == 4
    assert receipt.summary.pairs == 2
    assert receipt.handshake_id == client.handshake_id
    assert receipt.agreement_digest == client.agreement_digest
    # The container scores on the episode, not on the arm: the pair is a tie,
    # and a tie is a recorded result rather than a missing one.
    assert receipt.summary.mean_delta == pytest.approx(0.0)
    assert receipt.baseline.mean_reward == pytest.approx(receipt.trained.mean_reward)
    assert receipt.trained.loaded_sampler_references == ("provider://sampler/ckpt_primary_u1",)
