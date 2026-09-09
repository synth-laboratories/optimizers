"""The plan dimension: vocabulary, hash stability, and illegal combinations."""

from __future__ import annotations

import dataclasses

import pytest

from synth_optimizers.rl.plan import (
    CONTEXT_VIEWS,
    CORRECTIONS,
    CREDITS,
    DEFAULT_GROUPS_PER_STEP,
    DEFAULT_MAX_STEPS_PER_ROUND,
    GROUPINGS,
    OBJECTIVES,
    ORIGINS,
    PRESETS,
    READINESS,
    REDUCERS,
    SCORER_ROLES,
    CreditEstimator,
    OffPolicyCorrection,
    PlanValidationError,
    PolicyObjective,
    RolloutStrategy,
    ScorerSpec,
    UpdateSchedule,
    expand,
    require_implemented,
    validate,
)

# Verbatim copy of ``tito_train.algorithm.PRESETS["cispo"].to_dict()``. If this
# ever diverges the two planes no longer reconcile by mapping.
TITO_CISPO_PAYLOAD = {
    "preset": "cispo",
    "rollout": {
        "origin": "task_reset",
        "cardinality": 4,
        "readiness": "group_complete",
        "grouping": "task",
    },
    "scorers": [{"role": "old_actor", "context_view": "actor", "update": "frozen"}],
    "credit": {"kind": "length_weighted_leave_one_out", "weights": {}},
    "objective": {
        "kind": "cispo",
        "variant": "cispo_minimax",
        "granularity": "token",
        "eps_low": 1.0,
        "eps_high": 4.0,
        "ratio_granularity": "token",
    },
    "correction": {"kind": "staleness_drop", "max_weight_staleness": 0, "enabled": False},
    "reducer": {"kind": "branch_aware_root_mean"},
    "schedule": {
        "weight_mode": "sync_pin",
        "policy_span_count": 1,
        "actor_epochs": 1,
        "publish_to": "new_pops_only",
    },
    "context_views": ["actor"],
    "auxiliary_learners": [],
}


def test_dimension_vocabularies_match_tito() -> None:
    assert ORIGINS == {"task_reset", "trace_pivot", "restored_env_state"}
    assert READINESS == {"group_complete", "each_rollout", "batch_window"}
    assert GROUPINGS == {"task", "pivot", "hierarchical", "none"}
    assert SCORER_ROLES == {"old_actor", "reference", "teacher", "critic", "reward_model"}
    assert CREDITS == {
        "group_mean",
        "leave_one_out",
        "length_weighted_leave_one_out",
        "length_weighted_leave_one_out_standardized",
        "gae",
        "skip_observation_gae",
        "teacher_logprob_gap",
        "raw_reward",
    }
    assert OBJECTIVES == {
        "cispo",
        "gspo",
        "ppo_clipped",
        "reinforce",
        "jsd_distillation",
        "sampled_distillation",
    }
    assert CORRECTIONS == {"none", "tis", "ice_pop", "sao_dis", "staleness_drop"}
    assert REDUCERS == {
        "token_mean",
        "sequence_mean",
        "root_rollout_mean",
        "fixed_token_denominator",
        "branch_aware_root_mean",
    }
    assert CONTEXT_VIEWS == {"actor", "teacher_privileged", "reference", "critic"}


def test_cispo_preset_shared_payload_is_byte_compatible_with_tito() -> None:
    assert PRESETS["cispo"].shared_dimension_payload() == TITO_CISPO_PAYLOAD


def test_every_preset_shares_tito_field_names() -> None:
    for name, plan in PRESETS.items():
        payload = plan.shared_dimension_payload()
        assert set(payload) == set(TITO_CISPO_PAYLOAD), name
        assert set(payload["schedule"]) == set(TITO_CISPO_PAYLOAD["schedule"]), name
        assert set(payload["credit"]) == set(TITO_CISPO_PAYLOAD["credit"]), name


def test_plan_hash_is_stable_across_equal_plans() -> None:
    first = expand({"preset": "cispo"})
    second = expand({"preset": "cispo"})
    assert first.plan_hash == second.plan_hash
    assert first.plan_hash == PRESETS["cispo"].plan_hash
    assert first.shared_dimension_hash == PRESETS["cispo"].shared_dimension_hash


@pytest.mark.parametrize(
    "overlay",
    [
        {"rollout": {"cardinality": 8}},
        {"credit": {"kind": "length_weighted_leave_one_out_standardized"}},
        {"credit": {"zero_advantage_atol": 1e-6}},
        {"credit": {"same_policy_reduction": "none"}},
        {"objective": {"eps_high": 3.0}},
        {"reducer": {"kind": "token_mean"}},
        {"schedule": {"groups_per_step": 4}},
        {"schedule": {"max_steps_per_round": 9}},
    ],
)
def test_plan_hash_moves_when_any_dimension_moves(overlay: dict[str, object]) -> None:
    base = PRESETS["cispo"]
    changed = expand({"preset": "cispo", **overlay})
    assert changed.plan_hash != base.plan_hash


def test_added_fields_do_not_move_the_shared_hash() -> None:
    """Packing and skipping are ours; they must not break cross-plane mapping."""

    changed = expand({"preset": "cispo", "schedule": {"groups_per_step": 7}})
    assert changed.shared_dimension_hash == PRESETS["cispo"].shared_dimension_hash
    assert changed.plan_hash != PRESETS["cispo"].plan_hash


def test_packing_and_skipping_are_plan_fields_not_constants() -> None:
    plan = PRESETS["cispo"]
    assert plan.groups_per_step == DEFAULT_GROUPS_PER_STEP == 3
    assert plan.max_steps_per_round == DEFAULT_MAX_STEPS_PER_ROUND == 15
    assert plan.credit.skip_zero_advantage is True
    assert plan.credit.zero_advantage_atol == 1e-8
    assert plan.correction.max_weight_staleness == 0
    wider = expand(
        {"preset": "cispo", "schedule": {"groups_per_step": 5, "max_steps_per_round": 30}}
    )
    assert (wider.groups_per_step, wider.max_steps_per_round) == (5, 30)


def test_plan_is_frozen_and_immutable() -> None:
    plan = PRESETS["cispo"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.preset = "gspo"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.objective.eps_high = 9.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.schedule.groups_per_step = 1  # type: ignore[misc]


def _cispo_like(**overrides: object) -> object:
    plan = PRESETS["cispo"]
    return dataclasses.replace(plan, **overrides)  # type: ignore[arg-type]


def test_cispo_with_sequence_ratios_is_illegal() -> None:
    plan = _cispo_like(
        objective=PolicyObjective(
            kind="cispo",
            variant="cispo_minimax",
            granularity="sequence",
            eps_low=1.0,
            eps_high=4.0,
            ratio_granularity="sequence",
        )
    )
    with pytest.raises(PlanValidationError, match="token-level"):
        validate(plan)  # type: ignore[arg-type]


def test_cispo_minimax_requires_one_sided_clip() -> None:
    with pytest.raises(PlanValidationError, match="eps_low >= 1"):
        expand({"preset": "cispo", "objective": {"eps_low": 0.2}})


def test_gspo_requires_sequence_ratios() -> None:
    with pytest.raises(PlanValidationError, match="sequence-level"):
        expand({"preset": "gspo", "objective": {"ratio_granularity": "token"}})


def test_gae_requires_a_critic() -> None:
    plan = _cispo_like(
        credit=CreditEstimator(kind="gae", skip_zero_advantage=False),
        scorers=(ScorerSpec(role="old_actor"),),
    )
    with pytest.raises(PlanValidationError, match="critic"):
        validate(plan)  # type: ignore[arg-type]


def test_group_relative_credit_needs_a_group() -> None:
    plan = _cispo_like(
        rollout=RolloutStrategy(cardinality=1, readiness="batch_window", grouping="none")
    )
    with pytest.raises(PlanValidationError, match="group-relative"):
        validate(plan)  # type: ignore[arg-type]


def test_group_relative_credit_needs_cardinality_two() -> None:
    with pytest.raises(PlanValidationError, match="cardinality >= 2"):
        expand({"preset": "cispo", "rollout": {"cardinality": 1}})


def test_zero_advantage_skipping_is_meaningless_without_group_variance() -> None:
    with pytest.raises(PlanValidationError, match="skip_zero_advantage"):
        expand({"preset": "ppo", "credit": {"skip_zero_advantage": True}})


def test_staleness_drop_only_with_async_lag() -> None:
    plan = _cispo_like(
        correction=OffPolicyCorrection(
            kind="staleness_drop", max_weight_staleness=2, enabled=True
        )
    )
    with pytest.raises(PlanValidationError, match="async_lag"):
        validate(plan)  # type: ignore[arg-type]


def test_async_lag_with_zero_staleness_is_a_sync_pin() -> None:
    plan = _cispo_like(schedule=UpdateSchedule(weight_mode="async_lag"))
    with pytest.raises(PlanValidationError, match="sync pin"):
        validate(plan)  # type: ignore[arg-type]


def test_policy_span_count_must_be_one() -> None:
    with pytest.raises(PlanValidationError, match="policy_span_count"):
        expand({"preset": "cispo", "schedule": {"policy_span_count": 2}})


@pytest.mark.parametrize(
    "overlay",
    [
        {"schedule": {"groups_per_step": 0}},
        {"schedule": {"max_steps_per_round": 0}},
        {"credit": {"zero_advantage_atol": -1.0}},
        {"credit": {"same_policy_reduction": "whatever"}},
        {"credit": {"kind": "not_a_credit"}},
        {"objective": {"variant": "cispo_sideways"}},
        {"correction": {"max_weight_staleness": -1}},
        {"reducer": {"kind": "mean"}},
        {"rollout": {"grouping": "sideways"}},
    ],
)
def test_illegal_values_are_rejected(overlay: dict[str, object]) -> None:
    with pytest.raises(PlanValidationError):
        expand({"preset": "cispo", **overlay})


def test_unknown_preset_and_unknown_keys_are_rejected() -> None:
    with pytest.raises(PlanValidationError, match="unknown preset"):
        expand({"preset": "not_a_preset"})
    with pytest.raises(PlanValidationError, match="unknown algorithm keys"):
        expand({"preset": "cispo", "objective_kind": "cispo"})
    with pytest.raises(PlanValidationError, match="unexpected keyword"):
        expand({"preset": "cispo", "objective": {"epsilon": 1.0}})


def test_credit_weights_accept_a_yaml_mapping() -> None:
    plan = expand({"preset": "cispo", "credit": {"weights": {"b": 2.0, "a": 1.0}}})
    assert plan.credit.weights == (("a", 1.0), ("b", 2.0))
    assert plan.shared_dimension_payload()["credit"]["weights"] == {"a": 1.0, "b": 2.0}


def test_unimplemented_presets_still_expand_and_hash() -> None:
    plan = expand({"preset": "multi_teacher_opd"})
    assert plan.plan_hash
    with pytest.raises(PlanValidationError, match="no table entry"):
        require_implemented(plan)
    require_implemented(expand({"preset": "cispo"}))
    require_implemented(expand({"preset": "gspo"}))
