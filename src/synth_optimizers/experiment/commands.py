"""`synth-optimizers experiment …`.

Five verbs, and every one of them recompiles the plan from the spec first. That
is what makes `resume` safe and `report` honest: if the recipe, the image, or
the staged candidate set has moved since the first dispatch, the digests
disagree and the command refuses instead of quietly comparing two different
measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import REGISTRY
from .models import ExperimentContractError
from .runner import ExperimentRunner
from .spec import ExperimentSpec, load_spec


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "experiment",
        help="Declarative, paired ablations over existing optimizer executors.",
    )
    commands = parser.add_subparsers(dest="experiment_command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--spec", required=True, help="Path to a synth.experiment.v1 TOML spec.")
        sub.add_argument(
            "--root",
            help="Where the plan, outcome log, and report live. Defaults to ./<experiment_id>.",
        )
        sub.add_argument("--json", action="store_true")

    factors = commands.add_parser(
        "factors", help="List what the executor is prepared to vary for this spec."
    )
    common(factors)

    plan = commands.add_parser("plan", help="Compile and freeze the trial matrix. Runs nothing.")
    common(plan)

    run = commands.add_parser("run", help="Dispatch pending trials in the planned order.")
    common(run)
    run.add_argument("--limit", type=int, help="Dispatch at most this many trials, then stop.")

    resume = commands.add_parser("resume", help="Continue a frozen plan. Refuses to start one.")
    common(resume)
    resume.add_argument("--limit", type=int)
    resume.add_argument(
        "--retry-rig-failures",
        action="store_true",
        help=(
            "Re-dispatch trials whose sealed failure was the rig's, not the arm's. "
            "The superseded row stays in the log and the report counts every retry."
        ),
    )

    aa = commands.add_parser(
        "aa", help="A/A preflight: run the baseline against itself to test isolation."
    )
    common(aa)
    aa.add_argument("--limit", type=int)

    report = commands.add_parser("report", help="Reduce the sealed outcome rows.")
    common(report)


def dispatch(args: argparse.Namespace) -> int:
    command = args.experiment_command
    spec = load_spec(args.spec)
    root = Path(args.root) if args.root else Path.cwd() / spec.experiment_id
    mode = "aa" if command == "aa" else "experiment"
    if mode == "aa" and not args.root:
        root = root / "aa"

    if command == "factors":
        return _factors(spec, args)

    runner = ExperimentRunner(spec, _adapter(spec), root, mode=mode)

    if command == "plan":
        plan = runner.prepare()
        return _emit(args, _plan_summary(plan), _render_plan(plan))
    if command == "resume" and not runner.plan_path.is_file():
        print(
            f"error: no frozen plan at {runner.plan_path}; use `experiment run` to start one",
            file=sys.stderr,
        )
        return 1
    if command in ("run", "resume", "aa"):
        summary = runner.run(
            limit=getattr(args, "limit", None),
            retry_rig_failures=getattr(args, "retry_rig_failures", False),
        )
        report = runner.report()
        payload = {"run": summary.to_json(), "report": report.to_json()}
        return _emit(args, payload, _render_report(report, summary.stopped_reason))
    if command == "report":
        report = runner.report()
        return _emit(args, report.to_json(), _render_report(report, None))
    raise SystemExit(f"unknown experiment command {command}")


def _adapter(spec: ExperimentSpec) -> Any:
    factory = REGISTRY.get(spec.executor)
    if factory is None:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise ExperimentContractError(
            f"no adapter for executor {spec.executor!r}; available: {known}"
        )
    return factory.from_spec(spec)


def _factors(spec: ExperimentSpec, args: argparse.Namespace) -> int:
    catalog = _adapter(spec).factor_catalog(spec)
    payload = catalog.to_json()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"{catalog.executor} / {catalog.base_ref}")
    for factor in catalog.factors:
        values = ", ".join(repr(value) for value in factor.values) if factor.values else "-"
        print(f"  {factor.path}  [{factor.kind}]  {values}")
        print(f"      {factor.description}")
    return 0


def _plan_summary(plan: Any) -> dict[str, Any]:
    return {
        "experiment_id": plan.experiment_id,
        "plan_digest": plan.plan_digest,
        "executor": plan.executor,
        "base_ref": plan.base_ref,
        "arms": [arm.to_json() for arm in plan.arms],
        "blocks": plan.blocks,
        "trials": len(plan.trials),
        "provenance": plan.provenance,
        "dispatch_order": [
            {"trial_id": trial.trial_id, "arm_id": trial.arm_id, "block_id": trial.block_id}
            for trial in sorted(plan.trials, key=lambda item: item.dispatch_index)
        ],
    }


def _render_plan(plan: Any) -> str:
    lines = [
        f"{plan.experiment_id}  ({plan.executor} / {plan.base_ref})",
        f"  plan digest   {plan.plan_digest}",
        f"  arms          {len(plan.arms)}",
    ]
    for arm in plan.arms:
        lines.append(f"    {arm.arm_id}  {arm.label}")
        lines.append(
            f"        subject {arm.subject.subject_kind}:{arm.subject.subject_id} "
            f"@ {arm.subject.subject_content_digest[:19]}"
        )
    lines.append(f"  blocks        {len(plan.blocks['ids'])} x {plan.blocks['replicates']}")
    lines.append(f"  trials        {len(plan.trials)}")
    # Each executor pins something different: an image digest, a config digest,
    # a wheel. Show what this one actually recorded rather than one executor's
    # vocabulary applied to every other.
    for key, value in sorted(plan.provenance.items()):
        if key.endswith("_digest") or key in ("image", "image_digest"):
            lines.append(f"  {key:<13} {value or 'UNPINNED'}")
    return "\n".join(lines)


def _render_report(report: Any, stopped: str | None) -> str:
    lines = []
    if stopped:
        lines.append(f"stopped: {stopped}")
    lines.append(
        f"{report.experiment_id}  [{report.mode}]  metric={report.primary_metric} "
        f"({report.direction})"
    )
    for arm in report.arms:
        mean = "-" if arm.mean is None else f"{arm.mean:.6g}"
        lines.append(
            f"  {arm.arm_id}  {arm.label}\n"
            f"      mean {mean}   completed {arm.completed_trials}/{arm.planned_trials}"
            f"   blocks {arm.blocks_completed}/{arm.blocks_expected}"
        )
        if arm.failure_classes:
            lines.append(f"      failures {arm.failure_classes}")
        if arm.missing_blocks:
            lines.append(f"      missing  {list(arm.missing_blocks)}")
    for comparison in report.comparisons:
        if comparison.mean_delta is None:
            lines.append(
                f"  {comparison.treatment_arm_id} vs {comparison.baseline_arm_id}: no paired blocks"
            )
            continue
        interval = (
            f"[{comparison.ci_low:.6g}, {comparison.ci_high:.6g}]"
            if comparison.ci_low is not None
            else "[n<2]"
        )
        lines.append(
            f"  {comparison.treatment_arm_id} vs {comparison.baseline_arm_id}: "
            f"delta {comparison.mean_delta:+.6g}  {comparison.confidence:.0%} CI {interval}  "
            f"p={comparison.p_value:.4g} ({comparison.p_method})  "
            f"n={comparison.blocks_paired}  W/L/T {comparison.wins}/{comparison.losses}/"
            f"{comparison.ties}"
        )
    verdict = "ALLOWED" if report.claim.allowed else "REFUSED"
    lines.append(f"  headline claim: {verdict}")
    for blocker in report.claim.blockers:
        lines.append(f"      blocked: {blocker}")
    for note in report.claim.notes:
        lines.append(f"      note:    {note}")
    return "\n".join(lines)


def _emit(args: argparse.Namespace, payload: dict[str, Any], text: str) -> int:
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-optimizers experiment")
    subcommands = parser.add_subparsers(dest="command", required=True)
    register(subcommands)
    args = parser.parse_args(["experiment", *(argv if argv is not None else sys.argv[1:])])
    try:
        return dispatch(args)
    except ExperimentContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
