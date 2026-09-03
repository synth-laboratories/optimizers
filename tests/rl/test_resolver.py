"""Resolver: immutable ids only, verified before an evaluation is allowed to start."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from synth_optimizers.rl.catalog import (
    CheckpointArtifacts,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    SamplerWeightsRef,
    TrainingEvidence,
    TrainingStateRef,
)
from synth_optimizers.rl.policy_sets import (
    ComponentSaveAttempt,
    MatchSetRevision,
    OpponentBinding,
    PolicySetComponent,
    PolicySetPublisher,
    PolicySetRevision,
)
from synth_optimizers.rl.resolver import (
    AmbiguousSelectorError,
    ArtifactMissingError,
    CompatibilityMismatchError,
    CompatibilityRequirement,
    DigestMismatchError,
    EvaluationResolver,
    MutableSelectorError,
    ResolutionScope,
    RetiredRevisionError,
    RevisionNotReadyError,
    RoleMismatchError,
    UnknownSelectorError,
    UnpublishedComponentError,
    selectors_are_immutable,
)

RENDERER = "renderer_alpha"
TOKENIZER = "tokenizer_alpha"
PACKED = ("group_1", "group_2")


def sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


CONTRACT = sha("container_contract")


@dataclass
class Probe:
    """Mutable artifact probe: the test can make the world disagree."""

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
    parameter_group_id: str = "pg_alpha",
    policy_type_ids: tuple[str, ...] = ("type_alpha",),
    policy_revision_id: str = "pg_alpha@1",
    update_id: str = "update_0001",
    run_id: str = "run_a",
    publication_status: str = "staged",
    sampler: bool = True,
    training_state: bool = True,
    renderer_profile: str = RENDERER,
    tokenizer: str = TOKENIZER,
    container_contract_hash: str = CONTRACT,
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
            renderer_profile=renderer_profile,
            tokenizer=tokenizer,
            container_contract_hash=container_contract_hash,
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
    baseline: CheckpointRecord
    policy_set: PolicySetRevision


def build_world(tmp_path, *, metric_directions: dict[str, str] | None = None) -> World:
    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    publisher = PolicySetPublisher(catalog, health_check=lambda request: True)
    baseline = make_record(
        checkpoint_id="ckpt_baseline",
        update_id="update_0000",
        policy_revision_id="pg_alpha@0",
        publication_status="published",
        training_state=False,
    )
    catalog.register_baseline(baseline)
    manifest = PolicySetRevision(
        policy_set_revision_id="team-set-1",
        policy_set_id="team_set",
        run_id="run_a",
        update_id="update_0001",
        components=(
            PolicySetComponent(
                policy_type_id="type_alpha",
                parameter_group_id="pg_alpha",
                checkpoint_id="ckpt_alpha_u1",
                policy_revision_id="pg_alpha@1",
            ),
            PolicySetComponent(
                policy_type_id="type_beta",
                parameter_group_id="pg_beta",
                checkpoint_id="ckpt_beta_u1",
                policy_revision_id="pg_beta@1",
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
        )
        for item in manifest.components
    ]
    publisher.publish_round(
        manifest,
        tuple(
            ComponentSaveAttempt(
                parameter_group_id=record.parameter_group_id,
                record=record,
                packed_group_ids=PACKED,
            )
            for record in components
        ),
    )
    publisher.mark_loaded("team-set-1")
    publisher.mark_ready("team-set-1")
    probe = probe_for(baseline, *components)
    resolver = EvaluationResolver(catalog, probe=probe, metric_directions=metric_directions or {})
    return World(
        catalog=catalog,
        publisher=publisher,
        probe=probe,
        resolver=resolver,
        baseline=baseline,
        policy_set=manifest,
    )


@pytest.fixture()
def world(tmp_path) -> World:
    built = build_world(tmp_path)
    yield built
    built.catalog.close()


def test_resolution_by_checkpoint_id(world: World) -> None:
    resolution = world.resolver.resolve_checkpoint("ckpt_baseline")
    assert resolution.resolved_kind == "checkpoint"
    assert resolution.resolved_id == "ckpt_baseline"
    assert resolution.requested_selector == "ckpt_baseline"
    assert resolution.alias is None
    assert resolution.checkpoint_ids == ("ckpt_baseline",)
    assert resolution.loaded_refs == ("provider://sampler/ckpt_baseline",)
    only = resolution.policies[0]
    assert only.artifact.role == "sampler_weights"
    assert only.artifact.digest == sha("sampler:ckpt_baseline")
    assert only.publication_status == "published"
    with pytest.raises(UnknownSelectorError):
        world.resolver.resolve_policy_set("ckpt_baseline")


def test_resolution_by_policy_set_revision_id_binds_the_whole_team(world: World) -> None:
    resolution = world.resolver.resolve_policy_set("team-set-1")
    assert resolution.resolved_kind == "policy_set"
    assert resolution.policy_set_revision_id == "team-set-1"
    assert set(resolution.checkpoint_ids) == {"ckpt_alpha_u1", "ckpt_beta_u1"}
    assert resolution.policy_for_group("pg_beta").policy_revision_id == "pg_beta@1"
    assert set(resolution.loaded_refs) == {
        "provider://sampler/ckpt_alpha_u1",
        "provider://sampler/ckpt_beta_u1",
    }
    with pytest.raises(UnknownSelectorError):
        resolution.policy_for_group("pg_absent")


def test_resolution_by_match_set_revision_id(world: World) -> None:
    frozen = make_record(
        checkpoint_id="ckpt_frozen_opponent",
        parameter_group_id="pg_opponent",
        policy_type_ids=("type_opponent",),
        policy_revision_id="pg_opponent@7",
        publication_status="published",
    )
    world.catalog.register_checkpoint(frozen)
    world.probe.digests[frozen.artifacts.sampler.ref] = frozen.artifacts.sampler.digest
    match = MatchSetRevision(
        match_set_revision_id="match-set-1",
        match_set_id="match_set",
        run_id="run_a",
        policy_set_revision_id="team-set-1",
        opponents=(
            OpponentBinding(
                opponent_id="opponent_a",
                binding_kind="pinned_checkpoint",
                identity="ckpt_frozen_opponent",
            ),
            OpponentBinding(
                opponent_id="opponent_b",
                binding_kind="external_model",
                identity="vendor/model-b@2026-01",
            ),
            OpponentBinding(
                opponent_id="opponent_c",
                binding_kind="scripted_baseline",
                identity="scripted_policy_v3",
            ),
        ),
        created_at="2026-09-02T02:00:00Z",
    )
    world.publisher.publish_match_set(match)
    world.publisher.mark_loaded("match-set-1")
    world.publisher.mark_ready("match-set-1")

    resolution = world.resolver.resolve_match_set("match-set-1")
    assert resolution.resolved_kind == "match_set"
    assert resolution.match_set_revision_id == "match-set-1"
    assert resolution.policy_set_revision_id == "team-set-1"
    assert set(resolution.checkpoint_ids) == {"ckpt_alpha_u1", "ckpt_beta_u1"}
    kinds = {opponent.opponent_id: opponent.binding_kind for opponent in resolution.opponents}
    assert kinds == {
        "opponent_a": "pinned_checkpoint",
        "opponent_b": "external_model",
        "opponent_c": "scripted_baseline",
    }
    pinned = next(item for item in resolution.opponents if item.opponent_id == "opponent_a")
    assert pinned.artifact is not None
    assert pinned.artifact.ref == "provider://sampler/ckpt_frozen_opponent"
    external = next(item for item in resolution.opponents if item.opponent_id == "opponent_b")
    assert external.artifact is None
    assert "provider://sampler/ckpt_frozen_opponent" in resolution.loaded_refs


def test_digest_mismatch_is_refused_before_an_evaluation_starts(world: World) -> None:
    world.probe.digests["provider://sampler/ckpt_beta_u1"] = sha("someone_else")
    with pytest.raises(DigestMismatchError) as raised:
        world.resolver.resolve_policy_set("team-set-1")
    assert "ckpt_beta_u1" in str(raised.value)
    # Atomic: the healthy component is not returned on its own.
    assert world.catalog.evaluations() == ()


def test_missing_artifact_is_an_evidence_failure(world: World) -> None:
    del world.probe.digests["provider://sampler/ckpt_alpha_u1"]
    with pytest.raises(ArtifactMissingError):
        world.resolver.resolve_policy_set("team-set-1")


def test_role_mismatch_is_refused(world: World) -> None:
    # The baseline carries a sampler artifact only: it is not resumable.
    with pytest.raises(RoleMismatchError) as raised:
        world.resolver.resolve_training_state("ckpt_baseline")
    assert "training_state" in str(raised.value)
    # The trained components are resumable, so the resume path resolves them.
    resumable = world.resolver.resolve_training_state("team-set-1")
    assert {policy.artifact.role for policy in resumable.policies} == {"training_state"}
    assert set(resumable.loaded_refs) == {
        "provider://state/ckpt_alpha_u1",
        "provider://state/ckpt_beta_u1",
    }
    sampler_only = make_record(
        checkpoint_id="ckpt_sampler_only",
        update_id="update_0002",
        policy_revision_id="pg_alpha@2",
        publication_status="published",
        training_state=False,
    )
    world.catalog.register_checkpoint(sampler_only)
    world.probe.digests[sampler_only.artifacts.sampler.ref] = sampler_only.artifacts.sampler.digest
    with pytest.raises(RoleMismatchError):
        world.resolver.resolve_training_state("ckpt_sampler_only")
    assert world.resolver.resolve_sampler("ckpt_sampler_only").policies[0].artifact.role == (
        "sampler_weights"
    )


def test_incompatible_renderer_or_tokenizer_is_refused(world: World) -> None:
    requirement = CompatibilityRequirement(
        renderer_profile=RENDERER, tokenizer=TOKENIZER, container_contract_hash=CONTRACT
    )
    assert world.resolver.resolve_policy_set("team-set-1", compatibility=requirement).policies
    for wrong in (
        CompatibilityRequirement(renderer_profile="renderer_other"),
        CompatibilityRequirement(tokenizer="tokenizer_other"),
        CompatibilityRequirement(container_contract_hash=sha("other_contract")),
    ):
        with pytest.raises(CompatibilityMismatchError):
            world.resolver.resolve_policy_set("team-set-1", compatibility=wrong)


def test_role_incompatible_component_in_a_set_is_refused(tmp_path) -> None:
    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    record = make_record(checkpoint_id="ckpt_alpha_u1", publication_status="published")
    catalog.register_checkpoint(record)
    catalog.put_revision(
        revision_id="team-set-broken",
        revision_kind="policy_set",
        family_id="team_set",
        payload=PolicySetRevision(
            policy_set_revision_id="team-set-broken",
            policy_set_id="team_set",
            run_id="run_a",
            update_id="update_0001",
            components=(
                PolicySetComponent(
                    policy_type_id="type_beta",
                    parameter_group_id="pg_alpha",
                    checkpoint_id="ckpt_alpha_u1",
                    policy_revision_id="pg_alpha@1",
                ),
            ),
            created_at="2026-09-02T00:00:00Z",
        ).to_payload(),
    )
    catalog.record_revision_transition("team-set-broken", "load")
    catalog.record_revision_transition("team-set-broken", "ready")
    resolver = EvaluationResolver(catalog, probe=probe_for(record))
    with pytest.raises(RoleMismatchError) as raised:
        resolver.resolve_policy_set("team-set-broken")
    assert "type_beta" in str(raised.value)
    catalog.close()


def test_staged_and_orphaned_components_are_not_evaluable(world: World) -> None:
    staged = make_record(
        checkpoint_id="ckpt_staged",
        update_id="update_0002",
        policy_revision_id="pg_alpha@2",
    )
    world.catalog.register_checkpoint(staged)
    world.probe.digests[staged.artifacts.sampler.ref] = staged.artifacts.sampler.digest
    with pytest.raises(UnpublishedComponentError):
        world.resolver.resolve_checkpoint("ckpt_staged")
    permissive = EvaluationResolver(world.catalog, probe=world.probe, allow_staged=True)
    assert permissive.resolve_checkpoint("ckpt_staged").resolved_id == "ckpt_staged"
    world.catalog.record_publication("ckpt_staged", "orphaned", reason="one_sided_publication")
    with pytest.raises(UnpublishedComponentError):
        world.resolver.resolve_checkpoint("ckpt_staged")
    with pytest.raises(UnpublishedComponentError):
        permissive.resolve_checkpoint("ckpt_staged")


def test_unready_and_retired_revisions_are_refused(tmp_path) -> None:
    built = build_world(tmp_path)
    second = PolicySetRevision(
        policy_set_revision_id="team-set-2",
        policy_set_id="team_set",
        run_id="run_a",
        update_id="update_0002",
        components=(
            PolicySetComponent(
                policy_type_id="type_alpha",
                parameter_group_id="pg_alpha",
                checkpoint_id="ckpt_alpha_u2",
                policy_revision_id="pg_alpha@2",
            ),
        ),
        created_at="2026-09-02T03:00:00Z",
    )
    record = make_record(
        checkpoint_id="ckpt_alpha_u2", update_id="update_0002", policy_revision_id="pg_alpha@2"
    )
    built.publisher.publish_round(
        second,
        (
            ComponentSaveAttempt(
                parameter_group_id="pg_alpha", record=record, packed_group_ids=PACKED
            ),
        ),
    )
    built.probe.digests[record.artifacts.sampler.ref] = record.artifacts.sampler.digest
    with pytest.raises(RevisionNotReadyError):
        built.resolver.resolve_policy_set("team-set-2")
    built.publisher.mark_loaded("team-set-2")
    built.publisher.mark_ready("team-set-2")
    assert built.resolver.resolve_policy_set("team-set-2").resolved_id == "team-set-2"
    built.publisher.retire("team-set-1", reason="superseded and drained")
    with pytest.raises(RetiredRevisionError):
        built.resolver.resolve_policy_set("team-set-1")
    built.catalog.close()


def test_alias_resolution_records_selector_and_immutable_id(tmp_path) -> None:
    built = build_world(tmp_path, metric_directions={"regret": "min"})
    resolver = built.resolver
    resolution = resolver.resolve("baseline")
    assert resolution.requested_selector == "baseline"
    assert resolution.alias == "baseline"
    assert resolution.resolved_id == "ckpt_baseline"
    receipt = resolution.to_receipt()
    assert receipt["requested_selector"] == "baseline"
    assert receipt["alias"] == "baseline"
    assert receipt["resolved_id"] == "ckpt_baseline"
    assert receipt["resolved_checkpoint_ids"] == ["ckpt_baseline"]
    assert receipt["loaded_refs"] == ["provider://sampler/ckpt_baseline"]

    scoped = resolver.resolve("baseline", scope=ResolutionScope(run_id="run_a"))
    assert scoped.resolved_id == "ckpt_baseline"

    latest = resolver.resolve(
        "latest-published", scope=ResolutionScope(parameter_group_id="pg_alpha")
    )
    assert latest.alias == "latest-published"
    assert latest.resolved_id == "ckpt_alpha_u1"
    with pytest.raises(AmbiguousSelectorError):
        resolver.resolve("latest-published")

    resolver.record_evaluation(
        "eval_baseline",
        resolver.resolve("ckpt_baseline"),
        metrics={"score": 0.30, "regret": 0.70},
    )
    resolver.record_evaluation(
        "eval_alpha",
        resolver.resolve("ckpt_alpha_u1"),
        metrics={"score": 0.62, "regret": 0.38},
    )
    best = resolver.resolve("best:score")
    assert best.alias == "best:score"
    assert best.resolved_id == "ckpt_alpha_u1"
    assert resolver.resolve("best:regret").resolved_id == "ckpt_alpha_u1"
    assert resolver.list_by_metric("score")[0] == ("checkpoint", "ckpt_alpha_u1", 0.62)
    assert resolver.list_by_metric("regret")[0] == ("checkpoint", "ckpt_alpha_u1", 0.38)
    scoped_best = resolver.resolve(
        "best:score", scope=ResolutionScope(parameter_group_id="pg_alpha")
    )
    assert scoped_best.resolved_id == "ckpt_alpha_u1"
    with pytest.raises(UnknownSelectorError):
        resolver.resolve("best:never_measured")
    with pytest.raises(UnknownSelectorError):
        resolver.resolve("best:")
    built.catalog.close()


def test_nothing_falls_back_to_latest(world: World) -> None:
    resolver = world.resolver
    for selector in ("latest", "LATEST", "newest", "current", "head", "tip"):
        with pytest.raises(MutableSelectorError):
            resolver.resolve(selector)
    for selector in ("", "   ", "ckpt_never_registered", "team-set-absent", "unregistered-alias"):
        with pytest.raises(UnknownSelectorError):
            resolver.resolve(selector)
    with pytest.raises(MutableSelectorError):
        selectors_are_immutable(("ckpt_baseline", "latest"))
    # An alias whose target vanished is a refusal, not a silent substitution.
    assert resolver.resolve("baseline").resolved_id == "ckpt_baseline"
    with pytest.raises(UnknownSelectorError):
        resolver.resolve("best:score")


def test_evaluation_binding_persists_selector_resolution_and_refs(world: World) -> None:
    resolution = world.resolver.resolve_policy_set("team-set-1")
    binding = world.resolver.record_evaluation("eval_team_1", resolution, metrics={"score": 0.55})
    assert binding.requested_selector == "team-set-1"
    assert binding.target_kind == "policy_set"
    assert binding.target_id == "team-set-1"
    assert set(binding.resolved_checkpoint_ids) == {"ckpt_alpha_u1", "ckpt_beta_u1"}
    assert set(binding.loaded_refs) == {
        "provider://sampler/ckpt_alpha_u1",
        "provider://sampler/ckpt_beta_u1",
    }
    stored = world.catalog.evaluations(target_id="team-set-1")
    assert [item.evaluation_id for item in stored] == ["eval_team_1"]
    assert world.catalog.describe_checkpoint("ckpt_alpha_u1").evaluation_ids == ("eval_team_1",)

    alias_resolution = world.resolver.resolve("baseline")
    alias_binding = world.resolver.record_evaluation(
        "eval_baseline", alias_resolution, metrics={"score": 0.31}
    )
    assert alias_binding.requested_selector == "baseline"
    assert alias_binding.resolved_checkpoint_ids == ("ckpt_baseline",)


def test_list_and_describe_by_each_declared_index(world: World) -> None:
    resolver = world.resolver
    resolver.record_evaluation(
        "eval_alpha", resolver.resolve("ckpt_alpha_u1"), metrics={"score": 0.6}
    )

    def ids(**kwargs: object) -> list[str]:
        return [view.checkpoint_id for view in resolver.list_checkpoints(**kwargs)]

    assert ids(run_id="run_a") == ["ckpt_baseline", "ckpt_alpha_u1", "ckpt_beta_u1"]
    assert ids(policy_type_id="type_beta") == ["ckpt_beta_u1"]
    assert ids(parameter_group_id="pg_alpha") == ["ckpt_baseline", "ckpt_alpha_u1"]
    assert ids(update_id="update_0001") == ["ckpt_alpha_u1", "ckpt_beta_u1"]
    assert ids(parent_checkpoint_id="ckpt_baseline") == []
    assert ids(publication_status="published") == [
        "ckpt_baseline",
        "ckpt_alpha_u1",
        "ckpt_beta_u1",
    ]
    assert ids(policy_set_revision_id="team-set-1") == ["ckpt_alpha_u1", "ckpt_beta_u1"]
    assert ids(train_call_id="provider_train_ckpt_beta_u1") == ["ckpt_beta_u1"]
    assert ids(evaluation_metric="score") == ["ckpt_alpha_u1"]

    described = resolver.describe("ckpt_alpha_u1")
    assert described["record_kind"] == "checkpoint"
    assert described["publication_status"] == "published"
    assert described["policy_set_revision_ids"] == ["team-set-1"]
    assert described["evaluation_ids"] == ["eval_alpha"]
    assert [attempt["outcome"] for attempt in described["save_attempts"]] == ["succeeded"]
    assert [edge["relation"] for edge in described["lineage_edges"]] == ["policy_set_component"]

    revision = resolver.describe("team-set-1")
    assert revision["record_kind"] == "policy_set"
    assert revision["is_active"] is True
    assert [item["transition"] for item in revision["transitions"]] == ["created", "load", "ready"]
    assert revision["active_attempts"] == []
    with pytest.raises(UnknownSelectorError):
        resolver.describe("ckpt_absent")
