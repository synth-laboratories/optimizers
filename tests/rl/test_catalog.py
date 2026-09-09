"""Catalog: immutable records, append-only relations, and typed artifact roles."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from synth_optimizers.rl.catalog import (
    ArtifactRoleError,
    BaselineMissingError,
    CatalogError,
    CheckpointArtifacts,
    CheckpointCatalog,
    CheckpointCompatibility,
    CheckpointRecord,
    DuplicateSaveError,
    EvaluationBinding,
    ImmutableRecordError,
    LineageEdge,
    LineageError,
    PublicationStatusError,
    SamplerWeightsRef,
    SaveAttempt,
    TrainingEvidence,
    TrainingStateRef,
    UnknownRecordError,
    checkpoint_id_for,
)

RENDERER = "renderer_alpha"
TOKENIZER = "tokenizer_alpha"


def sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


CONTRACT = sha("container_contract")


def make_record(
    *,
    checkpoint_id: str,
    run_id: str = "run_a",
    update_id: str = "update_0001",
    parameter_group_id: str = "pg_alpha",
    policy_type_ids: tuple[str, ...] = ("type_alpha",),
    policy_revision_id: str = "pg_alpha@1",
    parent_checkpoint_id: str | None = None,
    publication_status: str = "staged",
    sampler: bool = True,
    training_state: bool = True,
    groups: tuple[str, ...] = ("group_1",),
    renderer_profile: str = RENDERER,
    tokenizer: str = TOKENIZER,
    container_contract_hash: str = CONTRACT,
    train_call_ids: tuple[str, ...] = ("provider_train_1",),
    created_at: str = "2026-09-02T00:00:00Z",
) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        update_id=update_id,
        train_call_ids=train_call_ids,
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
            training_state=(
                TrainingStateRef(
                    ref=f"provider://state/{checkpoint_id}", digest=sha(f"state:{checkpoint_id}")
                )
                if training_state
                else None
            ),
        ),
        training_evidence=TrainingEvidence(
            groups=groups, examples=len(groups) * 8, tokens=len(groups) * 4096, provider_cost=0.25
        ),
        compatibility=CheckpointCompatibility(
            renderer_profile=renderer_profile,
            tokenizer=tokenizer,
            container_contract_hash=container_contract_hash,
        ),
        created_at=created_at,
        parent_checkpoint_id=parent_checkpoint_id,
        publication_status=publication_status,
    )


@pytest.fixture()
def catalog(tmp_path) -> CheckpointCatalog:
    instance = CheckpointCatalog(tmp_path / "catalog.sqlite3")
    yield instance
    instance.close()


def register_baseline(catalog: CheckpointCatalog, *, run_id: str = "run_a") -> CheckpointRecord:
    record = make_record(
        checkpoint_id="ckpt_baseline",
        run_id=run_id,
        update_id="update_0000",
        policy_revision_id="pg_alpha@0",
        publication_status="published",
        training_state=False,
        train_call_ids=(),
        groups=(),
    )
    return catalog.register_baseline(record)


def test_baseline_must_be_registered_before_rollout_admission(catalog: CheckpointCatalog) -> None:
    with pytest.raises(BaselineMissingError):
        catalog.assert_baseline_registered("run_a")
    baseline = register_baseline(catalog)
    admitted = catalog.assert_baseline_registered("run_a")
    assert admitted.checkpoint_id == baseline.checkpoint_id
    assert catalog.publication_status(baseline.checkpoint_id) == "published"
    assert catalog.alias("baseline").target_id == baseline.checkpoint_id


def test_checkpoint_payload_matches_the_declared_v1_shape(catalog: CheckpointCatalog) -> None:
    record = make_record(checkpoint_id="ckpt_one")
    payload = record.to_payload()
    assert list(payload) == [
        "schema_version",
        "checkpoint_id",
        "run_id",
        "update_id",
        "train_call_ids",
        "parameter_group_id",
        "policy_type_ids",
        "policy_revision_id",
        "parent_checkpoint_id",
        "base_model",
        "artifacts",
        "publication_status",
        "policy_set_revision_ids",
        "training_evidence",
        "compatibility",
        "created_at",
    ]
    assert payload["schema_version"] == "cispo.checkpoint.v1"
    assert set(payload["artifacts"]) == {"sampler_weights", "training_state"}
    assert set(payload["training_evidence"]) == {"groups", "examples", "tokens", "provider_cost"}
    assert set(payload["compatibility"]) == {
        "renderer_profile",
        "tokenizer",
        "container_contract_hash",
    }
    catalog.register_checkpoint(record)
    assert catalog.get_checkpoint("ckpt_one").to_payload() == payload


def test_sampler_and_training_state_refs_are_not_interchangeable() -> None:
    sampler = SamplerWeightsRef(ref="provider://sampler/x", digest=sha("s"))
    state = TrainingStateRef(ref="provider://state/x", digest=sha("t"))
    with pytest.raises(ArtifactRoleError):
        CheckpointArtifacts(sampler_weights=state)  # type: ignore[arg-type]
    with pytest.raises(ArtifactRoleError):
        CheckpointArtifacts(training_state=sampler)  # type: ignore[arg-type]
    with pytest.raises(ArtifactRoleError):
        CheckpointArtifacts(
            sampler_weights=SamplerWeightsRef(ref="provider://same", digest=sha("s")),
            training_state=TrainingStateRef(ref="provider://same", digest=sha("t")),
        )
    both = CheckpointArtifacts(sampler_weights=sampler, training_state=state)
    assert both.ref_for_role("sampler_weights") is sampler
    assert both.ref_for_role("training_state") is state
    sampler_only = CheckpointArtifacts(sampler_weights=sampler)
    with pytest.raises(ArtifactRoleError):
        sampler_only.resumable
    with pytest.raises(ArtifactRoleError):
        sampler_only.ref_for_role("training_state")
    with pytest.raises(ArtifactRoleError):
        both.ref_for_role("adapter_weights")
    with pytest.raises(CatalogError):
        CheckpointArtifacts()


def test_records_are_immutable_and_registration_is_idempotent(catalog: CheckpointCatalog) -> None:
    record = make_record(checkpoint_id="ckpt_one")
    catalog.register_checkpoint(record)
    catalog.register_checkpoint(record)
    assert len(catalog.list_checkpoints()) == 1
    mutated = make_record(checkpoint_id="ckpt_one", policy_revision_id="pg_alpha@2")
    with pytest.raises(ImmutableRecordError):
        catalog.register_checkpoint(mutated)
    with pytest.raises(FrozenInstanceError):
        record.policy_revision_id = "pg_alpha@9"  # type: ignore[misc]


def test_one_save_per_published_round_not_per_packed_group(catalog: CheckpointCatalog) -> None:
    register_baseline(catalog)
    packed = ("group_1", "group_2", "group_3")
    for group_id, policy_type in (("pg_alpha", "type_alpha"), ("pg_beta", "type_beta")):
        catalog.register_checkpoint(
            make_record(
                checkpoint_id=f"ckpt_{group_id}_u1",
                parameter_group_id=group_id,
                policy_type_ids=(policy_type,),
                policy_revision_id=f"{group_id}@1",
                parent_checkpoint_id="ckpt_baseline" if group_id == "pg_alpha" else None,
                groups=packed,
            )
        )
    saves = catalog.saves_for_update("run_a", "update_0001")
    assert saves == {"pg_alpha": ("ckpt_pg_alpha_u1",), "pg_beta": ("ckpt_pg_beta_u1",)}
    assert catalog.get_checkpoint("ckpt_pg_alpha_u1").training_evidence.groups == packed
    with pytest.raises(DuplicateSaveError):
        catalog.register_checkpoint(
            make_record(checkpoint_id="ckpt_pg_alpha_u1_again", groups=packed)
        )


def test_failed_save_attempt_is_recorded(catalog: CheckpointCatalog) -> None:
    catalog.record_save_attempt(
        SaveAttempt(
            run_id="run_a",
            update_id="update_0001",
            parameter_group_id="pg_beta",
            outcome="failed",
            error="provider save_state timed out",
            packed_group_ids=("group_1", "group_2"),
            provider_request_ids=("provider_train_9",),
        )
    )
    failures = catalog.save_attempts(run_id="run_a", outcome="failed")
    assert len(failures) == 1
    assert failures[0].error == "provider save_state timed out"
    assert failures[0].checkpoint_id is None
    assert catalog.saves_for_update("run_a", "update_0001") == {}
    with pytest.raises(CatalogError):
        SaveAttempt(
            run_id="run_a", update_id="u", parameter_group_id="pg", outcome="failed", error=None
        )
    with pytest.raises(UnknownRecordError):
        catalog.record_save_attempt(
            SaveAttempt(
                run_id="run_a",
                update_id="update_0001",
                parameter_group_id="pg_beta",
                outcome="succeeded",
                checkpoint_id="ckpt_never_registered",
            )
        )


def test_created_but_unpublished_component_stays_catalogued(catalog: CheckpointCatalog) -> None:
    catalog.register_checkpoint(make_record(checkpoint_id="ckpt_staged"))
    assert catalog.publication_status("ckpt_staged") == "staged"
    assert [
        view.checkpoint_id for view in catalog.list_checkpoints(publication_status="staged")
    ] == ["ckpt_staged"]
    catalog.record_publication("ckpt_staged", "orphaned", reason="one_sided_publication")
    assert catalog.publication_status("ckpt_staged") == "orphaned"
    assert catalog.has_checkpoint("ckpt_staged")
    history = [status for status, _at, _reason in catalog.publication_history("ckpt_staged")]
    assert history == ["staged", "orphaned"]


def test_publication_transitions_are_guarded(catalog: CheckpointCatalog) -> None:
    catalog.register_checkpoint(make_record(checkpoint_id="ckpt_one"))
    catalog.record_publication("ckpt_one", "published")
    with pytest.raises(PublicationStatusError):
        catalog.record_publication("ckpt_one", "staged")
    catalog.record_publication("ckpt_one", "superseded")
    with pytest.raises(PublicationStatusError):
        catalog.record_publication("ckpt_one", "published")
    with pytest.raises(PublicationStatusError):
        catalog.record_publication("ckpt_one", "not_a_status")
    with pytest.raises(PublicationStatusError):
        make_record(checkpoint_id="ckpt_two", publication_status="orphaned")


def test_catalog_recovers_after_process_interruption(tmp_path) -> None:
    path = tmp_path / "catalog.sqlite3"
    first = CheckpointCatalog(path)
    register_baseline(first)
    first.register_checkpoint(make_record(checkpoint_id="ckpt_staged"))
    first.record_save_attempt(
        SaveAttempt(
            run_id="run_a",
            update_id="update_0001",
            parameter_group_id="pg_beta",
            outcome="failed",
            error="interrupted",
        )
    )
    with pytest.raises(RuntimeError):
        with first.transaction():
            first.register_checkpoint(
                make_record(checkpoint_id="ckpt_never", parameter_group_id="pg_x")
            )
            raise RuntimeError("process killed mid-publication")
    del first  # no close(): the process simply went away

    recovered = CheckpointCatalog(path)
    assert recovered.assert_baseline_registered("run_a").checkpoint_id == "ckpt_baseline"
    assert recovered.publication_status("ckpt_staged") == "staged"
    assert not recovered.has_checkpoint("ckpt_never")
    assert [attempt.error for attempt in recovered.save_attempts(outcome="failed")] == [
        "interrupted"
    ]
    assert {view.checkpoint_id for view in recovered.list_checkpoints()} == {
        "ckpt_baseline",
        "ckpt_staged",
    }
    recovered.close()


def test_append_only_tables_refuse_update_and_delete(catalog: CheckpointCatalog) -> None:
    register_baseline(catalog)
    catalog.register_checkpoint(
        make_record(checkpoint_id="ckpt_one", parent_checkpoint_id="ckpt_baseline")
    )
    connection = catalog._conn
    for statement in (
        "UPDATE checkpoints SET base_model = 'x'",
        "DELETE FROM checkpoints",
        "UPDATE publication_events SET status = 'published'",
        "DELETE FROM lineage_edges",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement)


def test_lineage_edges_and_ancestry(catalog: CheckpointCatalog) -> None:
    register_baseline(catalog)
    catalog.register_checkpoint(
        make_record(checkpoint_id="ckpt_u1", parent_checkpoint_id="ckpt_baseline")
    )
    catalog.register_checkpoint(
        make_record(
            checkpoint_id="ckpt_u2",
            update_id="update_0002",
            policy_revision_id="pg_alpha@2",
            parent_checkpoint_id="ckpt_u1",
        )
    )
    assert catalog.ancestry("ckpt_u2") == ("ckpt_u1", "ckpt_baseline")
    parent_edges = catalog.lineage_edges(child_checkpoint_id="ckpt_u2", relation="parent")
    assert len(parent_edges) == 1
    assert parent_edges[0].parent_checkpoint_id == "ckpt_u1"
    assert parent_edges[0].train_call_ids == ("provider_train_1",)
    with pytest.raises(LineageError):
        catalog.register_checkpoint(
            make_record(checkpoint_id="ckpt_orphan_parent", parent_checkpoint_id="ckpt_absent")
        )
    with pytest.raises(LineageError):
        catalog.record_lineage_edge(
            LineageEdge(
                child_checkpoint_id="ckpt_absent",
                relation="policy_set_component",
                revision_id="set-1",
            )
        )


def test_evaluations_are_append_only_relations(catalog: CheckpointCatalog) -> None:
    register_baseline(catalog)
    before = catalog.get_checkpoint("ckpt_baseline").to_payload()
    catalog.record_evaluation(
        EvaluationBinding(
            evaluation_id="eval_1",
            target_kind="checkpoint",
            target_id="ckpt_baseline",
            requested_selector="baseline",
            resolved_checkpoint_ids=("ckpt_baseline",),
            loaded_refs=("provider://sampler/ckpt_baseline",),
            metrics={"score": 0.41},
        )
    )
    assert catalog.get_checkpoint("ckpt_baseline").to_payload() == before
    assert catalog.describe_checkpoint("ckpt_baseline").evaluation_ids == ("eval_1",)
    assert catalog.metric_rows("score") == (("checkpoint", "ckpt_baseline", 0.41),)
    with pytest.raises(UnknownRecordError):
        catalog.record_evaluation(
            EvaluationBinding(
                evaluation_id="eval_2",
                target_kind="checkpoint",
                target_id="ckpt_absent",
                requested_selector="ckpt_absent",
                resolved_checkpoint_ids=("ckpt_absent",),
            )
        )


def test_list_and_describe_by_each_declared_index(catalog: CheckpointCatalog) -> None:
    register_baseline(catalog)
    catalog.register_checkpoint(
        make_record(
            checkpoint_id="ckpt_alpha_u1",
            parent_checkpoint_id="ckpt_baseline",
            train_call_ids=("provider_train_a",),
        )
    )
    catalog.register_checkpoint(
        make_record(
            checkpoint_id="ckpt_beta_u1",
            parameter_group_id="pg_beta",
            policy_type_ids=("type_beta",),
            policy_revision_id="pg_beta@1",
            train_call_ids=("provider_train_b",),
        )
    )
    catalog.record_publication("ckpt_alpha_u1", "published")
    catalog.record_policy_set_membership("ckpt_alpha_u1", "team-set-1")
    catalog.record_evaluation(
        EvaluationBinding(
            evaluation_id="eval_1",
            target_kind="checkpoint",
            target_id="ckpt_alpha_u1",
            requested_selector="ckpt_alpha_u1",
            resolved_checkpoint_ids=("ckpt_alpha_u1",),
            metrics={"score": 0.7},
        )
    )

    def ids(**kwargs: object) -> list[str]:
        return [view.checkpoint_id for view in catalog.list_checkpoints(**kwargs)]

    assert ids(run_id="run_a") == ["ckpt_baseline", "ckpt_alpha_u1", "ckpt_beta_u1"]
    assert ids(update_id="update_0001") == ["ckpt_alpha_u1", "ckpt_beta_u1"]
    assert ids(parameter_group_id="pg_beta") == ["ckpt_beta_u1"]
    assert ids(policy_type_id="type_beta") == ["ckpt_beta_u1"]
    assert ids(parent_checkpoint_id="ckpt_baseline") == ["ckpt_alpha_u1"]
    assert ids(publication_status="staged") == ["ckpt_beta_u1"]
    assert ids(publication_status="published") == ["ckpt_baseline", "ckpt_alpha_u1"]
    assert ids(base_model="vendor/base-model-a") == [
        "ckpt_baseline",
        "ckpt_alpha_u1",
        "ckpt_beta_u1",
    ]
    assert ids(train_call_id="provider_train_b") == ["ckpt_beta_u1"]
    assert ids(policy_set_revision_id="team-set-1") == ["ckpt_alpha_u1"]
    assert ids(evaluation_metric="score") == ["ckpt_alpha_u1"]
    assert ids(evaluation_metric="unmeasured") == []

    described = catalog.describe_checkpoint("ckpt_alpha_u1")
    assert described.publication_status == "published"
    assert described.policy_set_revision_ids == ("team-set-1",)
    assert described.evaluation_ids == ("eval_1",)
    assert described.record.parent_checkpoint_id == "ckpt_baseline"


def test_deterministic_checkpoint_ids_are_stable() -> None:
    first = checkpoint_id_for(
        run_id="run_a",
        update_id="update_0004",
        parameter_group_id="pg_alpha",
        policy_revision_id="pg_alpha@4",
    )
    second = checkpoint_id_for(
        run_id="run_a",
        update_id="update_0004",
        parameter_group_id="pg_alpha",
        policy_revision_id="pg_alpha@4",
    )
    third = checkpoint_id_for(
        run_id="run_a",
        update_id="update_0005",
        parameter_group_id="pg_alpha",
        policy_revision_id="pg_alpha@5",
    )
    assert first == second != third
    assert first.startswith("ckpt_")
