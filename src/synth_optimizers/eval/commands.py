"""`synth-optimizers eval …`.

This is an app-internal launcher, not a general container CLI. It accepts a
recipe id and an app-owned manifest path; it never accepts an image, a command,
a mount, or an environment variable, so nothing an agent can say reaches a
container invocation through here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .executor import ContainerRuntimeError, OciTrialExecutor
from .home import EvalHome
from .models import EvalContractError
from .runner import request_cancel, run_worker
from .semaphore import TrialSemaphore
from .staging import CandidateSource, stage_candidate_set

DEFAULT_ENTRYPOINT = "policy:Policy"


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "eval",
        help="Local candidate evaluation against pinned target containers.",
    )
    commands = parser.add_subparsers(dest="eval_command", required=True)

    recipes = commands.add_parser("recipes", help="List the trusted local eval recipes.")
    recipes.add_argument("--home", help="App-owned eval home; defaults to the shipped catalog.")
    recipes.add_argument("--json", action="store_true")

    doctor = commands.add_parser("doctor", help="Preflight the local eval runtime.")
    doctor.add_argument("--home", required=True)
    doctor.add_argument("--json", action="store_true")

    pin = commands.add_parser("pin", help="Pin a catalog recipe's target image digest.")
    pin.add_argument("--home", required=True)
    pin.add_argument("--recipe", required=True)
    pin.add_argument("--digest", required=True)

    stage = commands.add_parser("stage", help="Stage policy sources into a candidate set.")
    stage.add_argument("--home", required=True)
    stage.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable. Copies PATH into an immutable content-addressed artifact.",
    )
    stage.add_argument(
        "--entrypoint",
        action="append",
        default=[],
        metavar="LABEL=ENTRYPOINT",
        help=f"Repeatable. Defaults to {DEFAULT_ENTRYPOINT}.",
    )
    stage.add_argument("--baseline", help="Label of the candidate to treat as the baseline.")
    stage.add_argument("--kind", default="python-code.v1")
    stage.add_argument("--json", action="store_true")

    worker = commands.add_parser("worker", help="Execute one sealed eval run.")
    worker.add_argument("--manifest", required=True, help="App-owned worker manifest path.")

    cancel = commands.add_parser("cancel", help="Ask a running worker to stop and seal.")
    cancel.add_argument("--home", required=True)
    cancel.add_argument("--run-id", required=True)


def dispatch(args: argparse.Namespace) -> int:
    command = args.eval_command
    if command == "recipes":
        return _recipes(args)
    if command == "doctor":
        return _doctor(args)
    if command == "pin":
        return _pin(args)
    if command == "stage":
        return _stage(args)
    if command == "worker":
        return run_worker(Path(args.manifest), stream=sys.stdout)
    if command == "cancel":
        found = request_cancel(Path(args.home), args.run_id)
        if not found:
            print(f"error: unknown eval run {args.run_id}", file=sys.stderr)
            return 1
        print(f"cancellation requested for {args.run_id}")
        return 0
    raise SystemExit(f"unknown eval command {command}")


def _recipes(args: argparse.Namespace) -> int:
    from .recipes import catalog

    recipes = EvalHome.open(args.home).catalog() if args.home else catalog()
    payload = {"recipes": [recipe.to_json() for recipe in recipes]}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for recipe in recipes:
        suffix = "" if recipe.available else f"  ({recipe.unavailable_reason})"
        print(f"{recipe.id}  [{'available' if recipe.available else 'unavailable'}]{suffix}")
        print(f"    {recipe.description}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    home = EvalHome.open(args.home)
    runtime_path = shutil.which(home.config.container_runtime)
    semaphore = TrialSemaphore(
        home.semaphore_dir,
        capacity=home.config.max_concurrent_trials,
        ttl_seconds=home.config.lease_ttl_seconds,
    )
    recipes = home.catalog()
    executor = OciTrialExecutor(home.config.container_runtime) if runtime_path else None
    recipe_readiness = _runtime_recipe_readiness(recipes, executor)
    payload = {
        "home": str(home.root),
        "containerRuntime": home.config.container_runtime,
        "containerRuntimePath": runtime_path,
        "containerRuntimeAvailable": runtime_path is not None,
        "maxConcurrentTrials": home.config.max_concurrent_trials,
        "semaphore": semaphore.snapshot(),
        "recipes": recipe_readiness,
        "ready": runtime_path is not None and any(item["available"] for item in recipe_readiness),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        state = "ready" if payload["ready"] else "not ready"
        print(f"eval home {home.root} is {state}")
        print(f"  runtime: {home.config.container_runtime} -> {runtime_path or 'MISSING'}")
        print(f"  semaphore: {payload['semaphore']}")
        for item in recipe_readiness:
            mark = "ok" if item["available"] else item["reason"]
            print(f"  {item['id']}: {mark}")
    return 0 if payload["ready"] else 1


def _runtime_recipe_readiness(recipes: Any, executor: Any | None) -> list[dict[str, Any]]:
    """Prove that each advertised target can run before a run is created.

    A syntactically valid digest is not execution readiness. The worker refuses
    missing and mismatched local images, so admission must apply that same
    authority and return the error while no run/trial records exist.
    """

    readiness: list[dict[str, Any]] = []
    for recipe in recipes:
        reason = recipe.unavailable_reason
        resolved_reference = None
        if reason is None and executor is None:
            reason = "container runtime is unavailable"
        if reason is None:
            try:
                resolved_reference = executor.resolve_reference(recipe.image, recipe.image_digest)
            except (ContainerRuntimeError, EvalContractError) as error:
                reason = str(error)
        readiness.append(
            {
                "id": recipe.id,
                "available": reason is None,
                "reason": reason,
                "image": recipe.image,
                "imageDigest": recipe.image_digest,
                "resolvedReference": resolved_reference,
            }
        )
    return readiness


def _pin(args: argparse.Namespace) -> int:
    home = EvalHome.open(args.home)
    home.write_pin(args.recipe, args.digest)
    print(f"pinned {args.recipe} to {args.digest}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    entrypoints = dict(_split_pair(item, field="--entrypoint") for item in args.entrypoint)
    sources = []
    for item in args.candidate:
        label, path = _split_pair(item, field="--candidate")
        sources.append(
            CandidateSource(
                label=label,
                path=Path(path),
                entrypoint=entrypoints.get(label, DEFAULT_ENTRYPOINT),
                kind=args.kind,
                is_baseline=label == args.baseline,
            )
        )
    if args.baseline and not any(source.is_baseline for source in sources):
        raise SystemExit(f"--baseline {args.baseline} is not one of the staged candidates")
    home = EvalHome.open(args.home)
    candidate_set = stage_candidate_set(home, sources)
    payload = {
        "candidateSetId": candidate_set.id,
        "candidateSetPath": str(home.candidates_dir / candidate_set.id / "candidate_set.json"),
        "digest": candidate_set.digest(),
        "candidates": [candidate.to_json() for candidate in candidate_set.candidates],
    }
    print(json.dumps(payload, indent=2) if args.json else payload["candidateSetId"])
    return 0


def _split_pair(raw: str, *, field: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"{field} expects LABEL=VALUE, got {raw!r}")
    label, value = raw.split("=", 1)
    if not label.strip() or not value.strip():
        raise SystemExit(f"{field} expects LABEL=VALUE, got {raw!r}")
    return label.strip(), value.strip()


def main(argv: list[str] | None = None) -> int:
    """Standalone entry so the worker can run as `python -m synth_optimizers.eval`."""

    parser = argparse.ArgumentParser(prog="synth-optimizers eval")
    subcommands = parser.add_subparsers(dest="command", required=True)
    register(subcommands)
    args = parser.parse_args(["eval", *(argv if argv is not None else sys.argv[1:])])
    try:
        return dispatch(args)
    except EvalContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
