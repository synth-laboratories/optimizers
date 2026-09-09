"""Policy binder: provider first, catalog second, and nothing exists until it is catalogued."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from synth_optimizers.contracts.rl_records import RendererProfile
from synth_optimizers.providers.protocols import (
    ForwardRequest,
    ProviderCapabilities,
    ProviderCheckpoint,
    ProviderError,
    ProviderSession,
    ProviderUsage,
    SampleRequest,
    TrainingStepRequest,
    TrainingStepResult,
)
from synth_optimizers.rl.binder import (
    SAMPLER_KIND,
    TRAINING_STATE_KIND,
    BaselineRequiredError,
    BinderError,
    CatalogPolicyBinder,
    RevisionNumberError,
    revision_number_of,
)
from synth_optimizers.rl.catalog import BaselineMissingError, CheckpointCatalog
from synth_optimizers.rl.policy_sets import (
    HealthCheckRequest,
    PartialPublicationError,
    PolicySetPublisher,
)
from synth_optimizers.rl.ports import PolicyBinder, TrainOutcome
from synth_optimizers.rl.resolver import (
    EvaluationResolver,
    MappingArtifactProbe,
    MutableSelectorError,
    UnknownSelectorError,
)

BASE_MODEL = "vendor/base-model-a"
CONTRACT = "sha256:" + hashlib.sha256(b"container_contract").hexdigest()
GROUP_A = "pg_alpha"
GROUP_B = "pg_beta"
PACKED = ("rollout_group_1", "rollout_group_2", "rollout_group_3")


# --------------------------------------------------------------------- fakes


@dataclass
class FakeProvider:
    """A scripted ``TrainingProvider``. No network, no spend, no real Tinker."""

    artifacts: dict[str, str] = field(default_factory=dict)
    sessions: list[ProviderSession] = field(default_factory=list)
    save_calls: list[tuple[str, int, str, str]] = field(default_factory=list)
    train_calls: list[TrainingStepRequest] = field(default_factory=list)
    failing_sessions: set[str] = field(default_factory=set)
    step: int = 0
    restore_calls: list[ProviderCheckpoint] = field(default_factory=list)

    # ------------------------------------------------------------- sessions

    def create_session(
        self, model_id: str, *, rank: int, seed: int, request_id: str
    ) -> ProviderSession:
        session = ProviderSession(
            provider="fake",
            session_id=f"session_{len(self.sessions) + 1}",
            model_id=model_id,
            request_id=request_id,
        )
        self.sessions.append(session)
        return session

    def restore_session(
        self, checkpoint: ProviderCheckpoint, *, request_id: str
    ) -> ProviderSession:
        self.restore_calls.append(checkpoint)
        session = ProviderSession(
            provider="fake", session_id=f"restored_{len(self.sessions) + 1}",
            model_id=checkpoint.model_id or BASE_MODEL, request_id=request_id,
        )
        self.sessions.append(session)
        return session

    # ------------------------------------------------------------- training

    def train_step(
        self, session: ProviderSession, request: TrainingStepRequest
    ) -> TrainingStepResult:
        self.train_calls.append(request)
        self.step += 1
        return TrainingStepResult(
            request_id=request.request_id,
            step=self.step,
            metrics={"loss": 1.0 / self.step},
            usage=ProviderUsage(
                training_tokens=17 * len(request.data), cost_usd=0.25, cost_missing=False
            ),
        )

    def save_checkpoint(
        self, session: ProviderSession, *, step: int, kind: str, request_id: str
    ) -> ProviderCheckpoint:
        self.save_calls.append((session.session_id, step, kind, request_id))
        if session.session_id in self.failing_sessions:
            raise ProviderError("save_failed", f"scripted save failure for {session.session_id}")
        reference = f"provider://{session.session_id}/{kind}/{step}"
        digest = "sha256:" + hashlib.sha256(reference.encode()).hexdigest()
        self.artifacts[reference] = digest
        return ProviderCheckpoint(
            checkpoint_id=f"{session.session_id}-{kind}-{step}",
            provider_reference=reference,
            step=step,
            digest=digest,
            kind=kind,
            resume_token=None if kind == SAMPLER_KIND else f"resume:{session.session_id}:{step}",
        )

    # ------------------------------------------- unused provider surface

    def discover_capabilities(self, model_id: str) -> ProviderCapabilities:  # pragma: no cover
        return ProviderCapabilities(provider="fake", model_id=model_id, capabilities=frozenset())

    def resolve_model(self, model_id: str) -> str:  # pragma: no cover
        return model_id

    def sample(self, session: ProviderSession, request: SampleRequest) -> Any:  # pragma: no cover
        raise NotImplementedError

    def forward(self, session: ProviderSession, request: ForwardRequest) -> Any:  # pragma: no cover
        raise NotImplementedError

    def sample_checkpoint(
        self, checkpoint: ProviderCheckpoint, request: SampleRequest
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, session: ProviderSession) -> None:  # pragma: no cover
        return None

    def classify_error(self, error: BaseException) -> ProviderError:  # pragma: no cover
        return ProviderError("unknown", str(error))

    # ----------------------------------------------------------- test view

    def sampler_saves(self) -> list[tuple[str, int, str, str]]:
        return [call for call in self.save_calls if call[2] == SAMPLER_KIND]


def profile() -> RendererProfile:
    return RendererProfile(
        profile_id="renderers.stub.v1",
        package="renderers",
        package_version="0.1.11",
        config_digest="sha256:" + "cd" * 32,
        tokenizer_id=BASE_MODEL,
        tokenizer_digest="sha256:" + "ef" * 32,
        stop_token_ids=(200002, 199999),
    )


@dataclass
class Harness:
    provider: FakeProvider
    catalog: CheckpointCatalog
    publisher: PolicySetPublisher
    resolver: EvaluationResolver
    binder: CatalogPolicyBinder
    health_checks: list[HealthCheckRequest]


def build(tmp_path: Path, *, save_training_state: bool = True) -> Harness:
    provider = FakeProvider()
    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    publisher = PolicySetPublisher(catalog)
    resolver = EvaluationResolver(catalog, probe=MappingArtifactProbe(provider.artifacts))
    seen: list[HealthCheckRequest] = []

    def health_check(request: HealthCheckRequest) -> bool:
        seen.append(request)
        return request.sampler_weights.ref in provider.artifacts

    binder = CatalogPolicyBinder(
        provider,
        publisher,
        resolver,
        base_model=BASE_MODEL,
        model_family="family_a",
        renderer_profile=profile(),
        container_contract_hash=CONTRACT,
        policy_set_id="set_alpha",
        wire_api="chat_completions",
        sampling_transport="message_in_capture_out",
        loss_name="declared.loss.v1",
        policy_types={GROUP_A: ("type_alpha",), GROUP_B: ("type_beta",)},
        save_training_state=save_training_state,
        health_check=health_check,
    )
    return Harness(provider, catalog, publisher, resolver, binder, seen)


def batch(group_ids: Sequence[str] = PACKED) -> list[Mapping[str, Any]]:
    return [
        {"group_id": group_id, "token_ids": [1, 2, 3], "advantages": [0.5]}
        for group_id in group_ids
    ]


def train_round(harness: Harness, update: str, groups: Sequence[str]) -> dict[str, TrainOutcome]:
    return {
        group: harness.binder.train(
            parameter_group_id=group,
            batch=batch(),
            update_id=update,
            plan_hash="sha256:" + "aa" * 32,
        )
        for group in groups
    }


# ------------------------------------------------------------------ baseline


def test_binder_satisfies_the_policy_binder_port(tmp_path: Path) -> None:
    harness = build(tmp_path)
    assert isinstance(harness.binder, PolicyBinder)


def test_baseline_is_catalogued_before_any_attempt(tmp_path: Path) -> None:
    harness = build(tmp_path)
    with pytest.raises(BaselineMissingError):
        harness.catalog.assert_baseline_registered("run_a")
    revision = harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    assert revision.revision == 0
    assert revision.revision_id == f"{GROUP_A}@0"
    record = harness.catalog.get_checkpoint(revision.checkpoint_id)
    assert record.parent_checkpoint_id is None
    assert record.compatibility.renderer_profile == profile().profile_id
    assert harness.catalog.publication_status(revision.checkpoint_id) == "published"
    assert harness.catalog.assert_baseline_registered("run_a").checkpoint_id == (
        revision.checkpoint_id
    )
    assert harness.catalog.alias(f"baseline.{GROUP_A}:run_a") is not None
    assert revision.sampler_reference in harness.provider.artifacts


def test_baseline_is_idempotent_per_parameter_group(tmp_path: Path) -> None:
    harness = build(tmp_path)
    first = harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    second = harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    assert first == second
    assert len(harness.provider.sampler_saves()) == 1


def test_baseline_can_restore_verified_training_state_with_exact_parent_lineage(
    tmp_path: Path,
) -> None:
    source = build(tmp_path)
    source.binder.baseline(run_id="run_source", parameter_group_id=GROUP_A)
    outcome = train_round(source, "update_0001", (GROUP_A,))
    published = source.binder.publish(
        run_id="run_source", update_id="update_0001",
        parameter_groups=(GROUP_A,), outcome=outcome,
    )
    parent_id = published[GROUP_A].checkpoint_id

    resumed = CatalogPolicyBinder(
        source.provider, PolicySetPublisher(source.catalog), source.resolver,
        base_model=BASE_MODEL, model_family="family_a", renderer_profile=profile(),
        container_contract_hash=CONTRACT, policy_set_id="set_resumed",
        wire_api="chat_completions", sampling_transport="message_in_capture_out",
        loss_name="declared.loss.v1", policy_types={GROUP_A: ("type_alpha",)},
        save_training_state=True, resume_from_checkpoint=parent_id,
    )
    baseline = resumed.baseline(run_id="run_resumed", parameter_group_id=GROUP_A)
    repeated = resumed.baseline(run_id="run_resumed", parameter_group_id=GROUP_A)

    assert repeated == baseline
    assert len(source.provider.restore_calls) == 1
    assert source.provider.restore_calls[-1].checkpoint_id == parent_id
    assert source.provider.restore_calls[-1].kind == TRAINING_STATE_KIND
    assert source.provider.save_calls[-1][0].startswith("restored_")
    assert source.provider.save_calls[-1][1] == published[GROUP_A].revision
    assert source.catalog.get_checkpoint(baseline.checkpoint_id).parent_checkpoint_id == parent_id
    assert baseline.revision == published[GROUP_A].revision
    assert baseline.revision_id == published[GROUP_A].revision_id
    assert resumed.resolution_receipts()[-1]["role"] == TRAINING_STATE_KIND
    next_outcome = resumed.train(
        parameter_group_id=GROUP_A, batch=batch(), update_id="update_0002",
        plan_hash="sha256:" + "aa" * 32,
    )
    child = resumed.publish(
        run_id="run_resumed", update_id="update_0002",
        parameter_groups=(GROUP_A,), outcome={GROUP_A: next_outcome},
    )[GROUP_A]
    assert child.revision == published[GROUP_A].revision + 1
    assert source.catalog.get_checkpoint(child.checkpoint_id).parent_checkpoint_id == baseline.checkpoint_id
    with pytest.raises(BinderError, match="cannot be re-imported"):
        resumed.baseline(run_id="run_resumed", parameter_group_id=GROUP_A)


def test_a_binder_serves_one_run(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    with pytest.raises(BinderError):
        harness.binder.baseline(run_id="run_b", parameter_group_id=GROUP_B)


def test_each_parameter_group_gets_its_own_provider_session(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_B)
    assert len({session.session_id for session in harness.provider.sessions}) == 2


# ------------------------------------------------------------------ training


def test_training_before_a_baseline_is_refused(tmp_path: Path) -> None:
    harness = build(tmp_path)
    with pytest.raises(BaselineRequiredError):
        harness.binder.train(
            parameter_group_id=GROUP_A,
            batch=batch(),
            update_id="update_0001",
            plan_hash="sha256:" + "aa" * 32,
        )
    assert harness.provider.train_calls == []


def test_training_an_uncatalogued_group_is_refused(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    with pytest.raises(BaselineRequiredError):
        harness.binder.train(
            parameter_group_id=GROUP_B,
            batch=batch(),
            update_id="update_0001",
            plan_hash="sha256:" + "aa" * 32,
        )


def test_train_calls_the_provider_once_per_parameter_group(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_B)
    outcome = train_round(harness, "update_0001", (GROUP_A, GROUP_B))
    assert len(harness.provider.train_calls) == 2
    assert {request.loss_name for request in harness.provider.train_calls} == {"declared.loss.v1"}
    assert len({request.request_id for request in harness.provider.train_calls}) == 2
    sessions = {call.metadata["parameter_group_id"] for call in harness.provider.train_calls}
    assert sessions == {GROUP_A, GROUP_B}
    assert outcome[GROUP_A].examples == len(PACKED)
    assert outcome[GROUP_A].tokens == 17 * len(PACKED)
    assert outcome[GROUP_A].provider_cost == 0.25
    assert outcome[GROUP_A].metrics["packed_group_ids"] == list(PACKED)


def test_train_receipts_loss_weight_magnitude(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    outcome = harness.binder.train(
        parameter_group_id=GROUP_A,
        batch=[
            {"group_id": "g1", "token_ids": [1, 2], "loss_weight": 0.25, "loss_mask": [0, 1]},
            {"group_id": "g1", "token_ids": [1, 2], "loss_weight": -0.5, "loss_mask": [1, 1]},
        ],
        update_id="update_0001",
        plan_hash="sha256:" + "aa" * 32,
    )

    assert outcome.metrics["loss_weight_nonzero"] == 2
    assert outcome.metrics["loss_weight_l1"] == pytest.approx(0.75)
    assert outcome.metrics["loss_weight_token_mass"] == pytest.approx(1.25)
    assert outcome.metrics["loss_weight_l2_squared"] == pytest.approx(0.3125)
    assert outcome.metrics["loss_weight_min"] == pytest.approx(-0.5)
    assert outcome.metrics["loss_weight_max"] == pytest.approx(0.25)


def test_an_empty_batch_is_refused(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    with pytest.raises(BinderError):
        harness.binder.train(
            parameter_group_id=GROUP_A,
            batch=[],
            update_id="update_0001",
            plan_hash="sha256:" + "aa" * 32,
        )


# --------------------------------------------------------------- publication


def test_publish_saves_once_per_round_over_several_packed_groups(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_B)
    baseline_saves = len(harness.provider.sampler_saves())
    outcome = train_round(harness, "update_0001", (GROUP_A, GROUP_B))
    published = harness.binder.publish(
        run_id="run_a",
        update_id="update_0001",
        parameter_groups=(GROUP_A, GROUP_B),
        outcome=outcome,
    )
    round_saves = harness.provider.sampler_saves()[baseline_saves:]
    assert len(round_saves) == 2, "one sampler artifact per group per round, not per packed group"
    assert [call[2] for call in harness.provider.save_calls].count(TRAINING_STATE_KIND) == 2
    assert set(published) == {GROUP_A, GROUP_B}
    assert published[GROUP_A].revision == 1
    assert published[GROUP_A].policy_set_revision_id == "set_alpha@update_0001"
    assert published[GROUP_A].training_state_reference is not None
    record = harness.catalog.get_checkpoint(published[GROUP_A].checkpoint_id)
    assert record.training_evidence.groups == PACKED
    assert record.training_evidence.examples == len(PACKED)
    assert record.parent_checkpoint_id is not None
    assert harness.catalog.publication_status(record.checkpoint_id) == "published"
    saves = harness.catalog.saves_for_update("run_a", "update_0001")
    assert {group: len(ids) for group, ids in saves.items()} == {GROUP_A: 1, GROUP_B: 1}
    assert harness.catalog.active_revision_id("set_alpha") == "set_alpha@update_0001"
    assert {check.checkpoint_id for check in harness.health_checks} == set(
        revision.checkpoint_id for revision in published.values()
    )


def test_publish_is_atomic_and_a_one_sided_failure_leaves_the_prior_set_live(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_B)
    first = harness.binder.publish(
        run_id="run_a",
        update_id="update_0001",
        parameter_groups=(GROUP_A, GROUP_B),
        outcome=train_round(harness, "update_0001", (GROUP_A, GROUP_B)),
    )
    outcome = train_round(harness, "update_0002", (GROUP_A, GROUP_B))
    beta_session = harness.provider.sessions[1].session_id
    harness.provider.failing_sessions.add(beta_session)
    with pytest.raises(PartialPublicationError) as failure:
        harness.binder.publish(
            run_id="run_a",
            update_id="update_0002",
            parameter_groups=(GROUP_A, GROUP_B),
            outcome=outcome,
        )
    publication = failure.value.outcome
    assert publication.published is False
    assert publication.failed_parameter_groups == (GROUP_B,)
    assert publication.active_policy_set_revision_id == "set_alpha@update_0001"
    assert harness.catalog.active_revision_id("set_alpha") == "set_alpha@update_0001"
    (orphan,) = publication.orphaned_checkpoint_ids
    assert harness.catalog.publication_status(orphan) == "orphaned"
    assert harness.catalog.get_checkpoint(orphan).parameter_group_id == GROUP_A
    attempts = harness.catalog.save_attempts(run_id="run_a", update_id="update_0002")
    assert {attempt.parameter_group_id: attempt.outcome for attempt in attempts} == {
        GROUP_A: "succeeded",
        GROUP_B: "failed",
    }
    assert all(attempt.packed_group_ids == PACKED for attempt in attempts)
    # The prior revision is still what a rollout would bind, and the binder did
    # not advance either group past it.
    assert harness.binder.revision_for(GROUP_A) == first[GROUP_A]
    assert harness.binder.revision_for(GROUP_B) == first[GROUP_B]


def test_publishing_a_group_without_its_training_outcome_is_refused(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_B)
    outcome = train_round(harness, "update_0001", (GROUP_A,))
    with pytest.raises(BinderError):
        harness.binder.publish(
            run_id="run_a",
            update_id="update_0001",
            parameter_groups=(GROUP_A, GROUP_B),
            outcome=outcome,
        )


def test_publishing_before_a_baseline_is_refused(tmp_path: Path) -> None:
    harness = build(tmp_path)
    with pytest.raises(BaselineRequiredError):
        harness.binder.publish(
            run_id="run_a", update_id="update_0001", parameter_groups=(GROUP_A,), outcome={}
        )


def test_a_second_round_supersedes_the_first(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    first = harness.binder.publish(
        run_id="run_a",
        update_id="update_0001",
        parameter_groups=(GROUP_A,),
        outcome=train_round(harness, "update_0001", (GROUP_A,)),
    )
    second = harness.binder.publish(
        run_id="run_a",
        update_id="update_0002",
        parameter_groups=(GROUP_A,),
        outcome=train_round(harness, "update_0002", (GROUP_A,)),
    )
    assert second[GROUP_A].revision == 2
    assert harness.catalog.active_revision_id("set_alpha") == "set_alpha@update_0002"
    assert harness.catalog.publication_status(first[GROUP_A].checkpoint_id) == "superseded"
    assert harness.catalog.ancestry(second[GROUP_A].checkpoint_id)[0] == (
        first[GROUP_A].checkpoint_id
    )


# ---------------------------------------------------------------- resolution


def test_resolve_records_the_requested_selector_beside_the_immutable_id(
    tmp_path: Path,
) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_B)
    published = harness.binder.publish(
        run_id="run_a",
        update_id="update_0001",
        parameter_groups=(GROUP_A, GROUP_B),
        outcome=train_round(harness, "update_0001", (GROUP_A, GROUP_B)),
    )
    resolved = harness.binder.resolve("set_alpha@update_0001")
    assert set(resolved) == {GROUP_A, GROUP_B}
    assert resolved[GROUP_A].checkpoint_id == published[GROUP_A].checkpoint_id
    assert resolved[GROUP_A].revision == 1
    assert resolved[GROUP_A].sampler_reference == published[GROUP_A].sampler_reference
    assert resolved[GROUP_A].behavior_fingerprint == published[GROUP_A].behavior_fingerprint
    (binding,) = harness.catalog.evaluations(target_id="set_alpha@update_0001")
    assert binding.requested_selector == "set_alpha@update_0001"
    assert set(binding.resolved_checkpoint_ids) == {
        published[GROUP_A].checkpoint_id,
        published[GROUP_B].checkpoint_id,
    }
    (receipt,) = harness.binder.resolution_receipts()
    assert receipt["requested_selector"] == "set_alpha@update_0001"
    assert receipt["resolved_id"] == "set_alpha@update_0001"
    assert receipt["resolved_kind"] == "policy_set"


def test_resolve_records_an_alias_beside_what_it_resolved_to(tmp_path: Path) -> None:
    harness = build(tmp_path)
    baseline = harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    resolved = harness.binder.resolve(f"baseline.{GROUP_A}")
    assert resolved[GROUP_A].checkpoint_id == baseline.checkpoint_id
    assert resolved[GROUP_A].revision == 0
    (receipt,) = harness.binder.resolution_receipts()
    assert receipt["alias"] == f"baseline.{GROUP_A}"
    assert receipt["resolved_id"] == baseline.checkpoint_id
    assert receipt["requested_selector"] == f"baseline.{GROUP_A}"
    (binding,) = harness.catalog.evaluations(target_id=baseline.checkpoint_id)
    assert binding.requested_selector == f"baseline.{GROUP_A}"


def test_resolve_is_idempotent_for_one_resolution(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    first = harness.binder.resolve(f"baseline.{GROUP_A}")
    second = harness.binder.resolve(f"baseline.{GROUP_A}")
    assert first[GROUP_A].checkpoint_id == second[GROUP_A].checkpoint_id


@pytest.mark.parametrize("selector", ["latest", "newest", "current", "head", "tip"])
def test_no_path_falls_back_to_latest(tmp_path: Path, selector: str) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    with pytest.raises(MutableSelectorError):
        harness.binder.resolve(selector)
    assert harness.catalog.evaluations() == ()


def test_an_unregistered_selector_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    harness = build(tmp_path)
    harness.binder.baseline(run_id="run_a", parameter_group_id=GROUP_A)
    with pytest.raises(UnknownSelectorError):
        harness.binder.resolve("set_alpha@update_0009")


def test_a_revision_id_without_an_integer_revision_is_refused() -> None:
    assert revision_number_of("pg_alpha@7") == 7
    with pytest.raises(RevisionNumberError):
        revision_number_of("pg_alpha")
