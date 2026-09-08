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
    if mode in {"container", "both"}:
        raise SchemaError(
            "container checkpoint evaluation authority is not installed in this runtime"
        )
    return {
        "schema_version": "training.checkpoint_plan.v2",
        "steps": steps,
        "save_steps": saved,
        "evaluation_steps": evaluated,
        "mode": mode,
        "baseline": mode != "none" and eval_schedule.get("baseline", True),
        "final": mode != "none" and eval_schedule.get("final", True),
    }
