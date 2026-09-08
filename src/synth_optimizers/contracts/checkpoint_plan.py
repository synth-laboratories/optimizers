"""Canonical checkpoint/evaluation schedules, independent of training length."""

from collections.abc import Mapping
from .training_schemas import SchemaError


def resolve_checkpoint_plan(config: Mapping) -> dict:
    training = config.get("training") or {}
    steps = training.get("steps", config.get("max_steps"))
    if steps is None:
        # Explicit adapter for historical flat requests, never applied to v2 plans.
        if "checkpoint_schedule" in config or "checkpoint_evaluation" in config:
            raise SchemaError("training.steps is required for a checkpoint plan")
        steps = max(config.get("checkpoint_steps") or [1])
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1_000_000:
        raise SchemaError("training.steps must be a positive bounded integer")
    schedule = config.get("checkpoint_schedule") or {}
    evaluation = config.get("checkpoint_evaluation") or {}
    if (
        config.get("container_url")
        or config.get("checkpoint_evaluation_policy")
        or config.get("evaluation_transport") == "tunnel"
        or (config.get("evaluation") or {}).get("transport") == "tunnel"
    ):
        raise SchemaError(
            "legacy container evaluation plan requires explicit supported authority migration"
        )
    mode = evaluation.get("mode", "builtin")
    if mode not in {"none", "builtin", "container", "both"}:
        raise SchemaError("unknown checkpoint evaluation mode")

    def cadence(field, default):
        value = training.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SchemaError(f"{field} must be a positive integer")
        return list(range(value, steps + 1, value))

    def validate(values, name):
        if not isinstance(values, list) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= steps for i in values
        ):
            raise SchemaError(f"{name} requires steps within training length")
        if sorted(set(values)) != values:
            raise SchemaError(f"{name} requires unique increasing steps")
        return values

    saved = validate(
        schedule.get(
            "save_steps", config.get("checkpoint_steps", cadence("checkpoint_every_steps", steps))
        ),
        "save_steps",
    )[:]
    if (
        "save_steps" in schedule
        and "checkpoint_steps" in config
        and saved != config["checkpoint_steps"]
    ):
        raise SchemaError("conflicting checkpoint schedules")
    if schedule.get("save_final", True) and steps not in saved:
        saved.append(steps)
    if not saved:
        raise SchemaError("at least one saved checkpoint is required")
    eval_schedule = evaluation.get("schedule") or {}
    evaluated = (
        []
        if mode == "none"
        else validate(
            eval_schedule.get(
                "steps",
                cadence("eval_every_steps", steps) if "eval_every_steps" in training else saved[:],
            ),
            "evaluation steps",
        )
    )
    if not set(evaluated) <= set(saved):
        raise SchemaError("evaluation steps require a saved sampler checkpoint")
    if mode == "builtin" and (
        not evaluated
        or not eval_schedule.get("baseline", True)
        or not eval_schedule.get("final", True)
    ):
        raise SchemaError(
            "built-in paired evaluation requires baseline, selection and final panels"
        )
    evaluators = evaluation.get("evaluators", [])
    if mode in {"container", "both"}:
        if not isinstance(evaluators, list) or not evaluators:
            raise SchemaError("container evaluation requires named frozen evaluator plans")
        ids = set()
        for evaluator in evaluators:
            for field in ("id", "recipe_id", "image_digest", "metric_ref", "reward_version", "units"):
                if not isinstance(evaluator.get(field), str) or not evaluator[field]:
                    raise SchemaError(f"evaluator requires {field}")
            if evaluator["id"] in ids or evaluator["id"] == "builtin":
                raise SchemaError("evaluator identities must be unique")
            ids.add(evaluator["id"])
            if evaluator.get("failure_policy", "block") not in {"block", "continue"}:
                raise SchemaError("unknown evaluator failure policy")
            selection_seeds, final_seeds = evaluator.get("selection_seeds"), evaluator.get("final_seeds")
            for panel in (selection_seeds, final_seeds):
                if not isinstance(panel, list) or not panel or len(panel)>1000 or any(type(seed) is not int for seed in panel) or len(set(panel)) != len(panel):
                    raise SchemaError("evaluation panels require distinct integer seeds")
            if set(selection_seeds) & set(final_seeds):
                raise SchemaError("selection and final panels must be disjoint")
        if not isinstance(config.get("evaluation_renderer_profile"), Mapping):
            raise SchemaError("container evaluation requires a pinned renderer profile")
    elif evaluators:
        raise SchemaError("configured container evaluators conflict with evaluation mode")
    selection = evaluation.get("selection", {"evaluator_id": "builtin" if mode in {"builtin", "both"} else (evaluators[0]["id"] if evaluators else "latest"),
                                              "direction": "maximize", "tie_break": "earliest_step"})
    allowed = {e["id"] for e in evaluators} | ({"builtin"} if mode in {"builtin", "both"} else {"latest"} if mode == "none" else set())
    if selection.get("evaluator_id") not in allowed or selection.get("direction", "maximize") not in {"maximize", "minimize"} or selection.get("tie_break", "earliest_step") not in {"earliest_step", "latest_step"}:
        raise SchemaError("unsupported checkpoint selection rule")
    return {
        "schema_version": "training.checkpoint_plan.v2",
        "steps": steps,
        "save_steps": saved,
        "evaluation_steps": evaluated,
        "mode": mode,
        "evaluators": evaluators,
        "selection": selection,
        "baseline": mode != "none" and eval_schedule.get("baseline", True),
        "final": mode != "none" and eval_schedule.get("final", True),
    }
