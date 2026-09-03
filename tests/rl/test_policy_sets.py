"""Policy sets and match sets: atomic publication, readiness, retirement."""

from __future__ import annotations

import hashlib

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
    HealthCheckError,
    HealthCheckRequest,
    MatchSetRevision,
    MissingComponentError,
    OpponentBinding,
    PartialPublicationError,
    PolicySetComponent,
    PolicySetError,
    PolicySetPublisher,
    PolicySetRevision,
    ReadinessError,
    RetirementError,
)

RENDERER = "renderer_alpha"
TOKENIZER = "tokenizer_alpha"
PACKED = ("group_1", "group_2", "group_3")


def sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


CONTRACT = sha("container_contract")


def make_record(
    *,
    checkpoint_id: str,
    parameter_group_id: str = "pg_alpha",
    policy_type_ids: tuple[str, ...] = ("type_alpha",),
    policy_revision_id: str = "pg_alpha@1",
    update_id: str = "update_0001",
    run_id: str = "run_a",
    parent_checkpoint_id: str | None = None,
    publication_status: str = "staged",
    sampler: bool = True,
    groups: tuple[str, ...] = PACKED,
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
            sampler_weights=(
                SamplerWeightsRef(
                    ref=f"provider://sampler/{checkpoint_id}",
                    digest=sha(f"sampler:{checkpoint_id}"),
                )
                if sampler
                else None
            ),
            training_state=TrainingStateRef(
                ref=f"provider://state/{checkpoint_id}", digest=sha(f"state:{checkpoint_id}")
            ),
        ),
        training_evidence=TrainingEvidence(
            groups=groups, examples=24, tokens=98304, provider_cost=1.5
        ),
        compatibility=CheckpointCompatibility(
            renderer_profile=RENDERER, tokenizer=TOKENIZER, container_contract_hash=CONTRACT
        ),
        created_at="2026-09-02T00:00:00Z",
        parent_checkpoint_id=parent_checkpoint_id,
        publication_status=publication_status,
    )


def component(
    *, policy_type_id: str, parameter_group_id: str, checkpoint_id: str, revision: int
) -> PolicySetComponent:
    return PolicySetComponent(
        policy_type_id=policy_type_id,
        parameter_group_id=parameter_group_id,
        checkpoint_id=checkpoint_id,
        policy_revision_id=f"{parameter_group_id}@{revision}",
    )


def two_group_revision(*, revision: int) -> PolicySetRevision:
    return PolicySetRevision(
        policy_set_revision_id=f"team-set-{revision}",
        policy_set_id="team_set",
        run_id="run_a",
        update_id=f"update_000{revision}",
        components=(
            component(
                policy_type_id="type_alpha",
                parameter_group_id="pg_alpha",
                checkpoint_id=f"ckpt_alpha_u{revision}",
                revision=revision,
            ),
            component(
                policy_type_id="type_beta",
                parameter_group_id="pg_beta",
                checkpoint_id=f"ckpt_beta_u{revision}",
                revision=revision,
            ),
        ),
        created_at="2026-09-02T00:00:00Z",
    )


def attempts_for(
    revision: PolicySetRevision, *, failing: tuple[str, ...] = ()
) -> tuple[ComponentSaveAttempt, ...]:
    built: list[ComponentSaveAttempt] = []
    for item in revision.components:
        if item.parameter_group_id in failing:
            built.append(
                ComponentSaveAttempt(
                    parameter_group_id=item.parameter_group_id,
                    error="provider save_weights_for_sampler failed",
                    packed_group_ids=PACKED,
                )
            )
            continue
        built.append(
            ComponentSaveAttempt(
                parameter_group_id=item.parameter_group_id,
                record=make_record(
                    checkpoint_id=item.checkpoint_id,
                    parameter_group_id=item.parameter_group_id,
                    policy_type_ids=(item.policy_type_id,),
                    policy_revision_id=item.policy_revision_id,
                    update_id=revision.update_id,
                ),
                packed_group_ids=PACKED,
            )
        )
    return tuple(built)


@pytest.fixture()
def publisher(tmp_path) -> PolicySetPublisher:
    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    yield PolicySetPublisher(catalog, health_check=lambda request: True)
    catalog.close()


def publish_ready(publisher: PolicySetPublisher, *, revision: int) -> PolicySetRevision:
    manifest = two_group_revision(revision=revision)
    publisher.publish_round(manifest, attempts_for(manifest))
    publisher.mark_loaded(manifest.policy_set_revision_id)
    publisher.mark_ready(manifest.policy_set_revision_id)
    return manifest


def test_policy_set_revision_validates_its_manifest() -> None:
    with pytest.raises(PolicySetError):
        PolicySetRevision(
            policy_set_revision_id="team-set-1",
            policy_set_id="team_set",
            run_id="run_a",
            update_id="update_0001",
            components=(),
            created_at="2026-09-02T00:00:00Z",
        )
    duplicated = (
        component(
            policy_type_id="type_alpha",
            parameter_group_id="pg_alpha",
            checkpoint_id="ckpt_a",
            revision=1,
        ),
        component(
            policy_type_id="type_beta",
            parameter_group_id="pg_alpha",
            checkpoint_id="ckpt_b",
            revision=1,
        ),
    )
    with pytest.raises(PolicySetError):
        PolicySetRevision(
            policy_set_revision_id="team-set-1",
            policy_set_id="team_set",
            run_id="run_a",
            update_id="update_0001",
            components=duplicated,
            created_at="2026-09-02T00:00:00Z",
        )
    manifest = two_group_revision(revision=1)
    assert manifest.component_for_policy_type("type_beta").parameter_group_id == "pg_beta"
    assert manifest.component_for_group("pg_alpha").checkpoint_id == "ckpt_alpha_u1"
    with pytest.raises(PolicySetError):
        manifest.component_for_policy_type("type_absent")
    assert PolicySetRevision.from_payload(manifest.to_payload()) == manifest


def test_opponent_binding_refuses_mutable_identities() -> None:
    for identity in ("latest", "LATEST", "best:score", "current", "head"):
        with pytest.raises(PolicySetError):
            OpponentBinding(
                opponent_id="opponent_a", binding_kind="pinned_checkpoint", identity=identity
            )
    with pytest.raises(PolicySetError):
        OpponentBinding(opponent_id="opponent_a", binding_kind="whatever", identity="ckpt_x")
    external = OpponentBinding(
        opponent_id="opponent_b", binding_kind="external_model", identity="vendor/model-b@2026-01"
    )
    assert not external.is_pinned_checkpoint
    assert OpponentBinding.from_payload(external.to_payload()) == external


def test_atomic_publish_promotes_every_component(publisher: PolicySetPublisher) -> None:
    manifest = two_group_revision(revision=1)
    outcome = publisher.publish_round(manifest, attempts_for(manifest))
    catalog = publisher.catalog
    assert outcome.published
    assert outcome.policy_set_revision_id == "team-set-1"
    assert outcome.active_policy_set_revision_id == "team-set-1"
    assert set(outcome.published_checkpoint_ids) == set(manifest.checkpoint_ids)
    for checkpoint_id in manifest.checkpoint_ids:
        assert catalog.publication_status(checkpoint_id) == "published"
        assert catalog.memberships_of(checkpoint_id) == ("team-set-1",)
        edges = catalog.lineage_edges(
            child_checkpoint_id=checkpoint_id, relation="policy_set_component"
        )
        assert [edge.revision_id for edge in edges] == ["team-set-1"]
    assert catalog.policy_set_members("team-set-1") == tuple(sorted(manifest.checkpoint_ids))
    assert publisher.policy_set("team-set-1") == manifest


def test_publish_round_saves_once_per_group_not_once_per_packed_group(
    publisher: PolicySetPublisher,
) -> None:
    manifest = two_group_revision(revision=1)
    publisher.publish_round(manifest, attempts_for(manifest))
    catalog = publisher.catalog
    saves = catalog.saves_for_update("run_a", "update_0001")
    assert saves == {"pg_alpha": ("ckpt_alpha_u1",), "pg_beta": ("ckpt_beta_u1",)}
    assert len(catalog.save_attempts(update_id="update_0001", outcome="succeeded")) == 2
    for checkpoint_id in manifest.checkpoint_ids:
        assert catalog.get_checkpoint(checkpoint_id).training_evidence.groups == PACKED
    with pytest.raises(PolicySetError):
        publisher.publish_round(manifest, attempts_for(manifest)[:1])


def test_one_sided_failure_keeps_prior_set_active_and_retains_the_orphan(
    publisher: PolicySetPublisher,
) -> None:
    prior = publish_ready(publisher, revision=1)
    catalog = publisher.catalog
    second = two_group_revision(revision=2)
    with pytest.raises(PartialPublicationError) as raised:
        publisher.publish_round(second, attempts_for(second, failing=("pg_beta",)))
    outcome = raised.value.outcome
    assert outcome.policy_set_revision_id is None
    assert outcome.active_policy_set_revision_id == prior.policy_set_revision_id
    assert outcome.orphaned_checkpoint_ids == ("ckpt_alpha_u2",)
    assert outcome.failed_parameter_groups == ("pg_beta",)
    assert outcome.orphan_retention == "retain_for_run_receipt"

    # The orphan is retained, not lost, and it is not part of any policy set.
    assert catalog.has_checkpoint("ckpt_alpha_u2")
    assert catalog.publication_status("ckpt_alpha_u2") == "orphaned"
    assert catalog.memberships_of("ckpt_alpha_u2") == ()
    assert not catalog.has_revision("team-set-2")
    # The prior set is still the live one, and still ready.
    assert catalog.active_revision_id("team_set") == "team-set-1"
    assert publisher.is_ready("team-set-1")
    for checkpoint_id in prior.checkpoint_ids:
        assert catalog.publication_status(checkpoint_id) == "published"
    failures = catalog.save_attempts(update_id="update_0002", outcome="failed")
    assert [attempt.parameter_group_id for attempt in failures] == ["pg_beta"]
    assert failures[0].error == "provider save_weights_for_sampler failed"


def test_retry_after_an_orphaned_component_may_publish(publisher: PolicySetPublisher) -> None:
    publish_ready(publisher, revision=1)
    second = two_group_revision(revision=2)
    with pytest.raises(PartialPublicationError):
        publisher.publish_round(second, attempts_for(second, failing=("pg_beta",)))
    retry = PolicySetRevision(
        policy_set_revision_id="team-set-2-retry",
        policy_set_id="team_set",
        run_id="run_a",
        update_id="update_0002",
        components=(
            component(
                policy_type_id="type_alpha",
                parameter_group_id="pg_alpha",
                checkpoint_id="ckpt_alpha_u2_retry",
                revision=2,
            ),
            component(
                policy_type_id="type_beta",
                parameter_group_id="pg_beta",
                checkpoint_id="ckpt_beta_u2_retry",
                revision=2,
            ),
        ),
        created_at="2026-09-02T01:00:00Z",
    )
    outcome = publisher.publish_round(retry, attempts_for(retry))
    catalog = publisher.catalog
    assert outcome.policy_set_revision_id == "team-set-2-retry"
    assert catalog.active_revision_id("team_set") == "team-set-2-retry"
    assert set(outcome.superseded_checkpoint_ids) == {"ckpt_alpha_u1", "ckpt_beta_u1"}
    assert catalog.publication_status("ckpt_alpha_u1") == "superseded"
    assert catalog.publication_status("ckpt_alpha_u2") == "orphaned"


def test_publication_fails_closed_when_a_component_is_absent(
    publisher: PolicySetPublisher,
) -> None:
    manifest = two_group_revision(revision=1)
    publisher.stage_component(
        make_record(
            checkpoint_id="ckpt_alpha_u1",
            parameter_group_id="pg_alpha",
            policy_type_ids=("type_alpha",),
            policy_revision_id="pg_alpha@1",
        )
    )
    with pytest.raises(MissingComponentError) as raised:
        publisher.publish(manifest)
    assert "ckpt_beta_u1" in str(raised.value)
    catalog = publisher.catalog
    assert not catalog.has_revision("team-set-1")
    assert catalog.publication_status("ckpt_alpha_u1") == "staged"
    assert catalog.active_revision_id("team_set") is None


def test_publication_refuses_an_orphaned_or_inconsistent_component(
    publisher: PolicySetPublisher,
) -> None:
    manifest = two_group_revision(revision=1)
    for item in manifest.components:
        publisher.stage_component(
            make_record(
                checkpoint_id=item.checkpoint_id,
                parameter_group_id=item.parameter_group_id,
                policy_type_ids=(item.policy_type_id,),
                policy_revision_id=item.policy_revision_id,
            )
        )
    publisher.catalog.record_publication("ckpt_beta_u1", "orphaned", reason="operator_discard")
    with pytest.raises(PolicySetError):
        publisher.publish(manifest)
    assert not publisher.catalog.has_revision("team-set-1")

    mislabelled = PolicySetRevision(
        policy_set_revision_id="team-set-9",
        policy_set_id="team_set",
        run_id="run_a",
        update_id="update_0001",
        components=(
            component(
                policy_type_id="type_beta",
                parameter_group_id="pg_alpha",
                checkpoint_id="ckpt_alpha_u1",
                revision=1,
            ),
        ),
        created_at="2026-09-02T00:00:00Z",
    )
    with pytest.raises(PolicySetError):
        publisher.publish(mislabelled)


def test_match_set_pins_trainee_and_every_opponent(publisher: PolicySetPublisher) -> None:
    trainee = publish_ready(publisher, revision=1)
    catalog = publisher.catalog
    frozen = make_record(
        checkpoint_id="ckpt_frozen_opponent",
        parameter_group_id="pg_opponent",
        policy_type_ids=("type_opponent",),
        policy_revision_id="pg_opponent@7",
        publication_status="published",
    )
    catalog.register_checkpoint(frozen)
    match = MatchSetRevision(
        match_set_revision_id="match-set-1",
        match_set_id="match_set",
        run_id="run_a",
        policy_set_revision_id=trainee.policy_set_revision_id,
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
    publisher.publish_match_set(match)
    assert publisher.match_set("match-set-1") == match
    assert match.pinned_checkpoint_ids == ("ckpt_frozen_opponent",)
    trainee_edges = catalog.lineage_edges(revision_id="match-set-1", relation="match_set_trainee")
    assert {edge.child_checkpoint_id for edge in trainee_edges} == set(trainee.checkpoint_ids)
    opponent_edges = catalog.lineage_edges(revision_id="match-set-1", relation="match_set_opponent")
    assert [edge.child_checkpoint_id for edge in opponent_edges] == ["ckpt_frozen_opponent"]

    absent = MatchSetRevision(
        match_set_revision_id="match-set-2",
        match_set_id="match_set",
        run_id="run_a",
        policy_set_revision_id=trainee.policy_set_revision_id,
        opponents=(
            OpponentBinding(
                opponent_id="opponent_a",
                binding_kind="pinned_checkpoint",
                identity="ckpt_absent",
            ),
        ),
        created_at="2026-09-02T02:00:00Z",
    )
    with pytest.raises(MissingComponentError):
        publisher.publish_match_set(absent)
    assert not catalog.has_revision("match-set-2")

    with pytest.raises(MissingComponentError):
        publisher.publish_match_set(
            MatchSetRevision(
                match_set_revision_id="match-set-3",
                match_set_id="match_set",
                run_id="run_a",
                policy_set_revision_id="team-set-absent",
                opponents=(),
                created_at="2026-09-02T02:00:00Z",
            )
        )


def test_ready_requires_load_and_a_passing_health_check(tmp_path) -> None:
    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    seen: list[HealthCheckRequest] = []

    def health_check(request: HealthCheckRequest) -> bool:
        seen.append(request)
        return True

    publisher = PolicySetPublisher(catalog)
    manifest = two_group_revision(revision=1)
    publisher.publish_round(manifest, attempts_for(manifest))
    with pytest.raises(ReadinessError):
        publisher.mark_ready(manifest.policy_set_revision_id, health_check=health_check)
    publisher.mark_loaded(manifest.policy_set_revision_id)
    with pytest.raises(ReadinessError):
        publisher.mark_ready(manifest.policy_set_revision_id)
    outcomes = publisher.mark_ready(manifest.policy_set_revision_id, health_check=health_check)
    assert [outcome.healthy for outcome in outcomes] == [True, True]
    assert {request.checkpoint_id for request in seen} == set(manifest.checkpoint_ids)
    assert {request.sampler_weights.ref for request in seen} == {
        "provider://sampler/ckpt_alpha_u1",
        "provider://sampler/ckpt_beta_u1",
    }
    assert publisher.is_ready(manifest.policy_set_revision_id)
    transitions = [
        transition.transition
        for transition in catalog.revision_transitions(manifest.policy_set_revision_id)
    ]
    assert transitions == ["created", "load", "ready"]
    catalog.close()


def test_failed_health_check_blocks_ready_and_is_recorded(tmp_path) -> None:
    catalog = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    publisher = PolicySetPublisher(catalog)
    manifest = two_group_revision(revision=1)
    publisher.publish_round(manifest, attempts_for(manifest))
    publisher.mark_loaded(manifest.policy_set_revision_id)
    with pytest.raises(HealthCheckError):
        publisher.mark_ready(manifest.policy_set_revision_id, health_check=lambda request: False)
    assert not publisher.is_ready(manifest.policy_set_revision_id)
    transitions = [
        transition.transition
        for transition in catalog.revision_transitions(manifest.policy_set_revision_id)
    ]
    assert transitions == ["created", "load", "health_check_failed"]

    def raising(request: HealthCheckRequest) -> bool:
        raise TimeoutError("sampler did not answer")

    with pytest.raises(HealthCheckError) as raised:
        publisher.mark_ready(manifest.policy_set_revision_id, health_check=raising)
    assert "TimeoutError" in str(raised.value)
    with pytest.raises(ReadinessError):
        publisher.open_attempt(manifest.policy_set_revision_id, "attempt_1")
    catalog.close()


def test_retirement_refused_while_sampling_and_allowed_at_zero(
    publisher: PolicySetPublisher,
) -> None:
    manifest = publish_ready(publisher, revision=1)
    revision_id = manifest.policy_set_revision_id
    publisher.open_attempt(revision_id, "attempt_1")
    publisher.open_attempt(revision_id, "attempt_2")
    assert publisher.active_attempt_count(revision_id) == 2
    with pytest.raises(RetirementError) as raised:
        publisher.retire(revision_id)
    assert "attempt_1" in str(raised.value)
    publisher.close_attempt(revision_id, "attempt_1")
    assert publisher.active_attempt_count(revision_id) == 1
    with pytest.raises(RetirementError):
        publisher.retire(revision_id)
    with pytest.raises(PolicySetError):
        publisher.close_attempt(revision_id, "attempt_1")
    publisher.close_attempt(revision_id, "attempt_2")
    assert publisher.active_attempt_count(revision_id) == 0
    publisher.retire(revision_id, reason="round complete")
    assert publisher.is_retired(revision_id)
    assert not publisher.is_ready(revision_id)
    publisher.retire(revision_id)  # idempotent
    with pytest.raises(RetirementError):
        publisher.open_attempt(revision_id, "attempt_3")
    with pytest.raises(RetirementError):
        publisher.mark_loaded(revision_id)
    transitions = [
        transition.transition for transition in publisher.catalog.revision_transitions(revision_id)
    ]
    assert transitions == [
        "created",
        "load",
        "ready",
        "attempt_open",
        "attempt_open",
        "attempt_close",
        "attempt_close",
        "retire",
    ]


def test_attempt_may_not_bind_an_unready_revision(publisher: PolicySetPublisher) -> None:
    manifest = two_group_revision(revision=1)
    publisher.publish_round(manifest, attempts_for(manifest))
    with pytest.raises(ReadinessError):
        publisher.open_attempt(manifest.policy_set_revision_id, "attempt_1")
    assert publisher.active_attempt_count(manifest.policy_set_revision_id) == 0


def test_match_set_readiness_health_checks_opponents_too(publisher: PolicySetPublisher) -> None:
    trainee = publish_ready(publisher, revision=1)
    catalog = publisher.catalog
    catalog.register_checkpoint(
        make_record(
            checkpoint_id="ckpt_frozen_opponent",
            parameter_group_id="pg_opponent",
            policy_type_ids=("type_opponent",),
            policy_revision_id="pg_opponent@7",
            publication_status="published",
        )
    )
    match = MatchSetRevision(
        match_set_revision_id="match-set-1",
        match_set_id="match_set",
        run_id="run_a",
        policy_set_revision_id=trainee.policy_set_revision_id,
        opponents=(
            OpponentBinding(
                opponent_id="opponent_a",
                binding_kind="pinned_checkpoint",
                identity="ckpt_frozen_opponent",
            ),
            OpponentBinding(
                opponent_id="opponent_b",
                binding_kind="scripted_baseline",
                identity="scripted_policy_v3",
            ),
        ),
        created_at="2026-09-02T02:00:00Z",
    )
    publisher.publish_match_set(match)
    publisher.mark_loaded("match-set-1")
    checked: list[str] = []
    outcomes = publisher.mark_ready(
        "match-set-1",
        health_check=lambda request: checked.append(request.checkpoint_id) is None,
    )
    assert set(checked) == set(trainee.checkpoint_ids) | {"ckpt_frozen_opponent"}
    assert all(outcome.healthy for outcome in outcomes)
    publisher.open_attempt("match-set-1", "attempt_1")
    with pytest.raises(RetirementError):
        publisher.retire("match-set-1")
