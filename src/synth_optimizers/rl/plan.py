"""AlgorithmPlan: eight composed dimensions, hashed once, never a branch.

``preset = "cispo"`` is an expansion over these dimensions, recorded in the run
manifest and in every group pin. There is no ``if algorithm == ...`` anywhere in
this plane: a second algorithm is a row in :data:`PRESETS` plus a table entry in
``objective.py``/``credit.py``/``reducer.py``.

Field names and vocabulary values are deliberately those of the Tito data plane
(``tito_train.algorithm``) so the two planes reconcile by mapping rather than by
rewrite. :meth:`AlgorithmPlan.shared_dimension_payload` emits exactly Tito's
``to_dict()`` shape; the fields this plane adds (zero-advantage skipping, the
same-policy reduction, and packing) live outside that payload so the shared
hash of an overlapping plan stays byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..contracts.rl_records import digest

# --- Dimension vocabularies, adopted verbatim from tito_train.algorithm ------

ORIGINS = frozenset({"task_reset", "trace_pivot", "restored_env_state"})
READINESS = frozenset({"group_complete", "each_rollout", "batch_window"})
GROUPINGS = frozenset({"task", "pivot", "hierarchical", "none"})
SCORER_ROLES = frozenset({"old_actor", "reference", "teacher", "critic", "reward_model"})
CREDITS = frozenset(
    {
        "group_mean",
        "leave_one_out",
        "length_weighted_leave_one_out",
        "length_weighted_leave_one_out_standardized",
        "gae",
        "skip_observation_gae",
        "teacher_logprob_gap",
        "raw_reward",
    }
)
OBJECTIVES = frozenset(
    {
        "cispo",
        "gspo",
        "ppo_clipped",
        "reinforce",
        "jsd_distillation",
        "sampled_distillation",
    }
)
CORRECTIONS = frozenset({"none", "tis", "ice_pop", "sao_dis", "staleness_drop"})
REDUCERS = frozenset(
    {
        "token_mean",
        "sequence_mean",
        "root_rollout_mean",
        "fixed_token_denominator",
        "branch_aware_root_mean",
    }
)
CONTEXT_VIEWS = frozenset({"actor", "teacher_privileged", "reference", "critic"})

# --- Vocabularies this plane adds ------------------------------------------

#: Credit kinds whose advantage is only defined relative to a group. A group of
#: one has nothing to be relative to.
GROUP_RELATIVE_CREDITS = frozenset(
    {
        "group_mean",
        "leave_one_out",
        "length_weighted_leave_one_out",
        "length_weighted_leave_one_out_standardized",
    }
)

#: How agent instances that share one parameter group are combined. A role's
#: share of the update must not be an accident of its token count.
SAME_POLICY_REDUCTIONS = frozenset({"none", "token_weighted_mean", "episode_uniform"})

WEIGHT_MODES = frozenset({"sync_pin", "async_lag"})
SCORER_UPDATES = frozenset({"frozen", "online", "ema"})
GRANULARITIES = frozenset({"token", "sequence"})
PUBLISH_TARGETS = frozenset({"new_pops_only", "immediate", "staged_policy_set"})

OBJECTIVE_VARIANTS: Mapping[str, frozenset[str]] = {
    "cispo": frozenset({"cispo_minimax", "cispo_two_sided"}),
    "gspo": frozenset({"gspo"}),
    "ppo_clipped": frozenset({"ppo_clipped", "ppo_dual_clip"}),
    "reinforce": frozenset({"reinforce"}),
    "jsd_distillation": frozenset({"jsd_distillation"}),
    "sampled_distillation": frozenset({"sampled_distillation"}),
}

#: The operational floor the workspace requires. Both are plan fields, so a
#: plan may state otherwise; neither is an executor constant.
DEFAULT_GROUPS_PER_STEP = 3
DEFAULT_MAX_STEPS_PER_ROUND = 15


class PlanValidationError(ValueError):
    """Fail fast on an illegal dimension combination, naming the combination."""


@dataclass(frozen=True, slots=True)
class RolloutStrategy:
    origin: str = "task_reset"
    cardinality: int = 4
    readiness: str = "group_complete"
    grouping: str = "task"


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    role: str
    context_view: str = "actor"
    update: str = "frozen"


@dataclass(frozen=True, slots=True)
class CreditEstimator:
    """Reward-to-advantage, plus the plan fields the executor must not own."""

    kind: str = "length_weighted_leave_one_out"
    #: Tito types this ``dict[str, float]``; a frozen plan cannot hold a mutable
    #: mapping, so it is carried as sorted pairs and rendered back to a dict in
    #: the shared payload. Empty in every overlapping preset, so the shared
    #: hash is unaffected.
    weights: tuple[tuple[str, float], ...] = ()
    skip_zero_advantage: bool = True
    zero_advantage_atol: float = 1e-8
    same_policy_reduction: str = "token_weighted_mean"

    @property
    def weight_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.weights}


@dataclass(frozen=True, slots=True)
class PolicyObjective:
    kind: str = "cispo"
    variant: str = "cispo_minimax"
    granularity: str = "token"
    eps_low: float = 1.0
    eps_high: float = 4.0
    ratio_granularity: str = "token"


@dataclass(frozen=True, slots=True)
class OffPolicyCorrection:
    kind: str = "staleness_drop"
    max_weight_staleness: int = 0
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class LossReducer:
    kind: str = "branch_aware_root_mean"


@dataclass(frozen=True, slots=True)
class UpdateSchedule:
    weight_mode: str = "sync_pin"  # sync_pin | async_lag
    policy_span_count: int = 1
    actor_epochs: int = 1
    publish_to: str = "new_pops_only"
    #: Packing. Added by this plane: the receipt's operational floor is three
    #: groups per provider step and no more than fifteen steps per round.
    groups_per_step: int = DEFAULT_GROUPS_PER_STEP
    max_steps_per_round: int = DEFAULT_MAX_STEPS_PER_ROUND


@dataclass(frozen=True, slots=True)
class AlgorithmPlan:
    """Immutable, validated, hashed. The executor runs this and nothing else."""

    preset: str
    rollout: RolloutStrategy
    scorers: tuple[ScorerSpec, ...]
    credit: CreditEstimator
    objective: PolicyObjective
    correction: OffPolicyCorrection
    reducer: LossReducer
    schedule: UpdateSchedule
    context_views: tuple[str, ...] = ("actor",)
    auxiliary_learners: tuple[str, ...] = ()

    def shared_dimension_payload(self) -> dict[str, Any]:
        """Exactly ``tito_train.algorithm.AlgorithmPlan.to_dict()``.

        Nothing this plane added appears here, so two planes that agree on the
        overlapping vocabulary produce byte-identical payloads.
        """

        return {
            "preset": self.preset,
            "rollout": {
                "origin": self.rollout.origin,
                "cardinality": self.rollout.cardinality,
                "readiness": self.rollout.readiness,
                "grouping": self.rollout.grouping,
            },
            "scorers": [
                {
                    "role": scorer.role,
                    "context_view": scorer.context_view,
                    "update": scorer.update,
                }
                for scorer in self.scorers
            ],
            "credit": {"kind": self.credit.kind, "weights": self.credit.weight_map},
            "objective": {
                "kind": self.objective.kind,
                "variant": self.objective.variant,
                "granularity": self.objective.granularity,
                "eps_low": self.objective.eps_low,
                "eps_high": self.objective.eps_high,
                "ratio_granularity": self.objective.ratio_granularity,
            },
            "correction": {
                "kind": self.correction.kind,
                "max_weight_staleness": self.correction.max_weight_staleness,
                "enabled": self.correction.enabled,
            },
            "reducer": {"kind": self.reducer.kind},
            "schedule": {
                "weight_mode": self.schedule.weight_mode,
                "policy_span_count": self.schedule.policy_span_count,
                "actor_epochs": self.schedule.actor_epochs,
                "publish_to": self.schedule.publish_to,
            },
            "context_views": list(self.context_views),
            "auxiliary_learners": list(self.auxiliary_learners),
        }

    def to_dict(self) -> dict[str, Any]:
        """The full plan, including the fields this plane added."""

        payload = self.shared_dimension_payload()
        payload["credit"] = dict(payload["credit"])
        payload["credit"].update(
            {
                "skip_zero_advantage": self.credit.skip_zero_advantage,
                "zero_advantage_atol": self.credit.zero_advantage_atol,
                "same_policy_reduction": self.credit.same_policy_reduction,
            }
        )
        payload["schedule"] = dict(payload["schedule"])
        payload["schedule"].update(
            {
                "groups_per_step": self.schedule.groups_per_step,
                "max_steps_per_round": self.schedule.max_steps_per_round,
            }
        )
        return payload

    @property
    def plan_hash(self) -> str:
        """Part of group identity. Two members under different hashes do not mix."""

        return digest(self.to_dict(), length=32)

    @property
    def shared_dimension_hash(self) -> str:
        """The hash of the Tito-overlapping payload, for cross-plane mapping."""

        return digest(self.shared_dimension_payload(), length=32)

    @property
    def objective_event_name(self) -> str:
        return self.objective.variant

    @property
    def groups_per_step(self) -> int:
        return self.schedule.groups_per_step

    @property
    def max_steps_per_round(self) -> int:
        return self.schedule.max_steps_per_round


_CISPO_OBJECTIVE = PolicyObjective(
    kind="cispo", variant="cispo_minimax", granularity="token", eps_low=1.0, eps_high=4.0
)
_GSPO_OBJECTIVE = PolicyObjective(
    kind="gspo",
    variant="gspo",
    granularity="sequence",
    eps_low=0.2,
    eps_high=0.2,
    ratio_granularity="sequence",
)
_PPO_OBJECTIVE = PolicyObjective(
    kind="ppo_clipped",
    variant="ppo_clipped",
    granularity="token",
    eps_low=0.2,
    eps_high=0.2,
)


PRESETS: dict[str, AlgorithmPlan] = {
    "cispo": AlgorithmPlan(
        preset="cispo",
        rollout=RolloutStrategy(
            origin="task_reset", cardinality=4, readiness="group_complete", grouping="task"
        ),
        scorers=(ScorerSpec(role="old_actor"),),
        credit=CreditEstimator(kind="length_weighted_leave_one_out"),
        objective=_CISPO_OBJECTIVE,
        correction=OffPolicyCorrection(
            kind="staleness_drop", max_weight_staleness=0, enabled=False
        ),
        reducer=LossReducer(kind="branch_aware_root_mean"),
        schedule=UpdateSchedule(weight_mode="sync_pin", policy_span_count=1),
    ),
    # Same CISPO sized for a curve rather than a single proof step: credit is
    # standardized within the group and the group is wider so a tie is less
    # likely. `cispo` above stays byte-identical so its hashes reproduce.
    "cispo_climb": AlgorithmPlan(
        preset="cispo_climb",
        rollout=RolloutStrategy(
            origin="task_reset", cardinality=8, readiness="group_complete", grouping="task"
        ),
        scorers=(ScorerSpec(role="old_actor"),),
        credit=CreditEstimator(kind="length_weighted_leave_one_out_standardized"),
        objective=_CISPO_OBJECTIVE,
        correction=OffPolicyCorrection(
            kind="staleness_drop", max_weight_staleness=0, enabled=False
        ),
        reducer=LossReducer(kind="branch_aware_root_mean"),
        schedule=UpdateSchedule(weight_mode="sync_pin", policy_span_count=1),
    ),
    "gspo": AlgorithmPlan(
        preset="gspo",
        rollout=RolloutStrategy(cardinality=8),
        scorers=(ScorerSpec(role="old_actor"),),
        credit=CreditEstimator(kind="group_mean"),
        objective=_GSPO_OBJECTIVE,
        correction=OffPolicyCorrection(kind="none", enabled=False),
        reducer=LossReducer(kind="sequence_mean"),
        schedule=UpdateSchedule(),
    ),
    "ppo": AlgorithmPlan(
        preset="ppo",
        rollout=RolloutStrategy(cardinality=1, readiness="batch_window", grouping="none"),
        scorers=(ScorerSpec(role="old_actor"), ScorerSpec(role="critic", context_view="critic")),
        credit=CreditEstimator(kind="gae", skip_zero_advantage=False),
        objective=_PPO_OBJECTIVE,
        correction=OffPolicyCorrection(kind="none", enabled=False),
        reducer=LossReducer(kind="token_mean"),
        schedule=UpdateSchedule(actor_epochs=4),
    ),
    # Fixtures: these expand and hash without new plan fields.
    "sao": AlgorithmPlan(
        preset="sao",
        rollout=RolloutStrategy(cardinality=1, readiness="each_rollout", grouping="none"),
        scorers=(ScorerSpec(role="old_actor"), ScorerSpec(role="critic", context_view="critic")),
        credit=CreditEstimator(kind="skip_observation_gae", skip_zero_advantage=False),
        objective=_PPO_OBJECTIVE,
        correction=OffPolicyCorrection(kind="sao_dis", enabled=True),
        reducer=LossReducer(kind="token_mean"),
        schedule=UpdateSchedule(),
    ),
    "multi_teacher_opd": AlgorithmPlan(
        preset="multi_teacher_opd",
        rollout=RolloutStrategy(cardinality=1, readiness="each_rollout", grouping="none"),
        scorers=(
            ScorerSpec(role="old_actor"),
            ScorerSpec(role="teacher"),
            ScorerSpec(role="teacher"),
        ),
        credit=CreditEstimator(kind="teacher_logprob_gap", skip_zero_advantage=False),
        objective=PolicyObjective(kind="reinforce", variant="reinforce", granularity="token"),
        correction=OffPolicyCorrection(kind="ice_pop", enabled=True, max_weight_staleness=4),
        reducer=LossReducer(kind="token_mean"),
        schedule=UpdateSchedule(weight_mode="async_lag"),
    ),
    "opsd": AlgorithmPlan(
        preset="opsd",
        rollout=RolloutStrategy(cardinality=1, readiness="each_rollout", grouping="none"),
        scorers=(ScorerSpec(role="teacher", context_view="teacher_privileged", update="frozen"),),
        credit=CreditEstimator(kind="teacher_logprob_gap", skip_zero_advantage=False),
        objective=PolicyObjective(
            kind="jsd_distillation", variant="jsd_distillation", granularity="token"
        ),
        correction=OffPolicyCorrection(kind="none", enabled=False),
        reducer=LossReducer(kind="token_mean"),
        schedule=UpdateSchedule(),
        context_views=("actor", "teacher_privileged"),
    ),
    "pivot_gspo": AlgorithmPlan(
        preset="pivot_gspo",
        rollout=RolloutStrategy(
            origin="trace_pivot", cardinality=8, readiness="group_complete", grouping="pivot"
        ),
        scorers=(ScorerSpec(role="old_actor"),),
        credit=CreditEstimator(kind="leave_one_out"),
        objective=_GSPO_OBJECTIVE,
        correction=OffPolicyCorrection(kind="none", enabled=False),
        reducer=LossReducer(kind="sequence_mean"),
        schedule=UpdateSchedule(),
    ),
}

#: Presets whose credit, objective, and reducer dimensions all have a table
#: entry in this plane. The rest expand and hash; the executor refuses them.
IMPLEMENTED_PRESETS = frozenset({"cispo", "cispo_climb", "gspo", "pivot_gspo"})

_OVERLAY_KEYS = frozenset(
    {"rollout", "credit", "objective", "correction", "reducer", "schedule"}
)


def expand(config: Mapping[str, Any]) -> AlgorithmPlan:
    """An ``algorithm:`` config block -> one immutable, validated, hashed plan."""

    preset_name = str(config.get("preset") or "").strip()
    if preset_name not in PRESETS:
        raise PlanValidationError(f"unknown preset {preset_name!r}; known: {sorted(PRESETS)}")
    plan = PRESETS[preset_name]
    overlay = {key: value for key, value in config.items() if key != "preset"}
    unknown = set(overlay) - _OVERLAY_KEYS
    if unknown:
        raise PlanValidationError(f"unknown algorithm keys: {sorted(unknown)}")
    for key in ("rollout", "credit", "objective", "correction", "reducer", "schedule"):
        if key not in overlay:
            continue
        patch = overlay[key]
        if not isinstance(patch, Mapping):
            raise PlanValidationError(f"algorithm.{key} must be a mapping")
        current = getattr(plan, key)
        fields = dict(patch)
        if key == "credit" and isinstance(fields.get("weights"), Mapping):
            fields["weights"] = tuple(
                sorted((str(key), float(value)) for key, value in fields["weights"].items())
            )
        try:
            plan = replace(plan, **{key: replace(current, **fields)})
        except TypeError as error:  # unknown dimension field
            raise PlanValidationError(f"algorithm.{key}: {error}") from error
    validate(plan)
    return plan


def _validate_vocabularies(plan: AlgorithmPlan) -> None:
    if plan.rollout.origin not in ORIGINS:
        raise PlanValidationError(f"unknown rollout origin {plan.rollout.origin}")
    if plan.rollout.readiness not in READINESS:
        raise PlanValidationError(f"unknown readiness {plan.rollout.readiness}")
    if plan.rollout.grouping not in GROUPINGS:
        raise PlanValidationError(f"unknown grouping {plan.rollout.grouping}")
    if plan.rollout.cardinality < 1:
        raise PlanValidationError("cardinality must be >= 1")
    for scorer in plan.scorers:
        if scorer.role not in SCORER_ROLES:
            raise PlanValidationError(f"unknown scorer role {scorer.role}")
        if scorer.context_view not in CONTEXT_VIEWS:
            raise PlanValidationError(f"unknown context view {scorer.context_view}")
        if scorer.update not in SCORER_UPDATES:
            raise PlanValidationError(f"unknown scorer update {scorer.update}")
    if plan.credit.kind not in CREDITS:
        raise PlanValidationError(f"unknown credit estimator {plan.credit.kind}")
    if plan.credit.same_policy_reduction not in SAME_POLICY_REDUCTIONS:
        raise PlanValidationError(
            f"unknown same_policy_reduction {plan.credit.same_policy_reduction}"
        )
    if plan.credit.zero_advantage_atol < 0.0:
        raise PlanValidationError("zero_advantage_atol must be non-negative")
    if plan.objective.kind not in OBJECTIVES:
        raise PlanValidationError(f"unknown objective {plan.objective.kind}")
    if plan.objective.granularity not in GRANULARITIES:
        raise PlanValidationError(f"unknown objective granularity {plan.objective.granularity}")
    if plan.objective.ratio_granularity not in GRANULARITIES:
        raise PlanValidationError(
            f"unknown ratio granularity {plan.objective.ratio_granularity}"
        )
    if plan.objective.variant not in OBJECTIVE_VARIANTS[plan.objective.kind]:
        raise PlanValidationError(
            f"unknown {plan.objective.kind} variant {plan.objective.variant}"
        )
    if plan.objective.eps_low < 0.0 or plan.objective.eps_high < 0.0:
        raise PlanValidationError("clip bounds must be non-negative")
    if plan.correction.kind not in CORRECTIONS:
        raise PlanValidationError(f"unknown correction {plan.correction.kind}")
    if plan.correction.max_weight_staleness < 0:
        raise PlanValidationError("max_weight_staleness must be non-negative")
    if plan.reducer.kind not in REDUCERS:
        raise PlanValidationError(f"unknown reducer {plan.reducer.kind}")
    if plan.schedule.weight_mode not in WEIGHT_MODES:
        raise PlanValidationError(f"unknown weight_mode {plan.schedule.weight_mode}")
    if plan.schedule.publish_to not in PUBLISH_TARGETS:
        raise PlanValidationError(f"unknown publish target {plan.schedule.publish_to}")
    if plan.schedule.actor_epochs < 1:
        raise PlanValidationError("actor_epochs must be >= 1")
    for view in plan.context_views:
        if view not in CONTEXT_VIEWS:
            raise PlanValidationError(f"unknown context view {view}")


#: Cross-dimension rules that hold for every plan. Each entry is a predicate
#: that must be true and the message to raise when it is not.
_Rule = tuple[Callable[["AlgorithmPlan"], bool], str]

UNIVERSAL_RULES: tuple[_Rule, ...] = (
    (
        lambda plan: plan.schedule.policy_span_count == 1,
        "policy_span_count must be 1: a group may not straddle two published revisions",
    ),
    (lambda plan: plan.schedule.groups_per_step >= 1, "groups_per_step must be >= 1"),
    (lambda plan: plan.schedule.max_steps_per_round >= 1, "max_steps_per_round must be >= 1"),
    (
        lambda plan: plan.credit.kind in GROUP_RELATIVE_CREDITS
        or not plan.credit.skip_zero_advantage,
        "this credit estimator has no group variance to detect; skip_zero_advantage is "
        "meaningless for it",
    ),
    (
        lambda plan: plan.credit.kind not in GROUP_RELATIVE_CREDITS
        or plan.rollout.grouping != "none",
        "a group-relative credit estimator requires a grouping",
    ),
    (
        lambda plan: plan.credit.kind not in GROUP_RELATIVE_CREDITS
        or plan.rollout.cardinality >= 2,
        "a group-relative credit estimator requires cardinality >= 2",
    ),
)

#: Rules keyed by a dimension value, so a new algorithm adds rows rather than
#: an ``if`` in the validator. Nothing outside these tables compares a plan
#: field to an algorithm name.
OBJECTIVE_RULES: Mapping[str, tuple[_Rule, ...]] = {
    "cispo": (
        (
            lambda plan: plan.objective.granularity == "token"
            and plan.objective.ratio_granularity == "token",
            "CISPO is token-level and is incompatible with sequence importance ratios",
        ),
    ),
    "gspo": (
        (
            lambda plan: plan.objective.ratio_granularity == "sequence",
            "GSPO requires sequence-level importance ratios",
        ),
    ),
}

VARIANT_RULES: Mapping[str, tuple[_Rule, ...]] = {
    "cispo_minimax": (
        (
            lambda plan: plan.objective.eps_low >= 1.0,
            "cispo_minimax requires eps_low >= 1 (one-sided clip); use cispo_two_sided instead",
        ),
    ),
}

CREDIT_RULES: Mapping[str, tuple[_Rule, ...]] = {
    "gae": (
        (
            lambda plan: any(scorer.role == "critic" for scorer in plan.scorers),
            "GAE credit requires a critic scorer",
        ),
    ),
    "skip_observation_gae": (
        (
            lambda plan: any(scorer.role == "critic" for scorer in plan.scorers),
            "skip-observation GAE credit requires a critic scorer",
        ),
    ),
}

CORRECTION_RULES: Mapping[str, tuple[_Rule, ...]] = {
    "staleness_drop": (
        (
            lambda plan: not plan.correction.enabled
            or plan.schedule.weight_mode == "async_lag",
            "staleness_drop is only meaningful with weight_mode=async_lag",
        ),
    ),
}

WEIGHT_MODE_RULES: Mapping[str, tuple[_Rule, ...]] = {
    "async_lag": (
        (
            lambda plan: plan.correction.max_weight_staleness > 0,
            "async_lag with max_weight_staleness=0 is a sync pin; say so",
        ),
    ),
}


def validate(plan: AlgorithmPlan) -> None:
    """Every illegal combination, raised by name. Never returns a bool."""

    _validate_vocabularies(plan)
    tables = (
        UNIVERSAL_RULES,
        OBJECTIVE_RULES.get(plan.objective.kind, ()),
        VARIANT_RULES.get(plan.objective.variant, ()),
        CREDIT_RULES.get(plan.credit.kind, ()),
        CORRECTION_RULES.get(plan.correction.kind, ()),
        WEIGHT_MODE_RULES.get(plan.schedule.weight_mode, ()),
    )
    for rules in tables:
        for predicate, message in rules:
            if not predicate(plan):
                raise PlanValidationError(message)


def require_implemented(plan: AlgorithmPlan) -> None:
    if plan.preset not in IMPLEMENTED_PRESETS:
        raise PlanValidationError(
            f"preset {plan.preset!r} expands and hashes but has no table entry in this plane; "
            f"implemented: {sorted(IMPLEMENTED_PRESETS)}"
        )


for _name, _plan in PRESETS.items():
    if _name != _plan.preset:  # pragma: no cover - table integrity
        raise PlanValidationError(f"preset table key {_name!r} != plan preset {_plan.preset!r}")
    validate(_plan)
