"""``synth-optimizers rl …``: the command surface over the container-first plane.

Four families, all of them reading the same durable records the plane writes:
start a run from a config file, evaluate a selector, read the catalog, and work
the four lifecycle controls against a live run.

Two rules run through all of it. Every command that resolves a policy prints
the selector it was given *and* the immutable id that selector resolved to, so
a transcript never shows a number whose provenance has to be reconstructed. And
resolution is verification: a command that would load an artifact needs a
digest source, and refuses rather than trusting the catalog's own copy of what
it thinks is on the provider.

Nothing here names a task, a harness, an environment, or a provider.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts.rl_records import RecordError
from .catalog import PUBLICATION_STATUSES, CatalogError, CheckpointCatalog
from .config import ConfigError
from .evaluation import (
    EvaluationError,
    EvaluationRequest,
    HeldOutSeed,
    PairedEvaluation,
    PinTemplate,
    RosterSlot,
)
from .lifecycle import LifecycleError, RunLifecycle
from .ports import ContainerSession, PolicyBinder, PortError, SamplerGateway
from .resolver import (
    ArtifactMissingError,
    EvaluationResolver,
    MappingArtifactProbe,
    Resolution,
    ResolutionError,
    ResolutionScope,
)
from .store import JournalStore, RunIdentity, StoreError

#: Every error the plane raises that means "refused", not "crashed".
PLANE_ERRORS = (
    ResolutionError,
    EvaluationError,
    CatalogError,
    StoreError,
    LifecycleError,
    ConfigError,
    PortError,
)

LIFECYCLE_CONTROLS = ("pause", "drain", "resume", "stop")


@dataclass(frozen=True, slots=True)
class Plane:
    """The three ports one live run is driven through.

    :mod:`synth_optimizers.rl.plane` assembles these from a run configuration
    and is what a command reaches when no assembly is named.
    ``--plane MODULE:FACTORY`` overrides it: the factory is called with the
    parsed run configuration and returns one of these.
    """

    session: ContainerSession
    gateway: SamplerGateway
    binder: PolicyBinder
    clock: Any = None


class _RefusingProbe:
    """The probe you get when no digest source was supplied.

    Resolution verifies artifacts against the provider; the catalog's own
    record of a digest cannot verify itself. So rather than quietly skipping
    the check, every reference refuses and says what is missing.
    """

    def exists(self, ref: str) -> bool:
        raise ArtifactMissingError(self._refusal(ref))

    def digest_of(self, ref: str) -> str:
        raise ArtifactMissingError(self._refusal(ref))

    @staticmethod
    def _refusal(ref: str) -> str:
        return (
            f"cannot verify artifact {ref}: pass --artifact-digests with the digests "
            "observed at the provider; an unverified artifact is not an evaluable one"
        )


# --------------------------------------------------------------------- parser


def register(subcommands: argparse._SubParsersAction) -> None:
    """Add the ``rl`` family. Additive: no existing command changes."""

    parser = subcommands.add_parser(
        "rl",
        help="Container-first RL plane: runs, paired evaluation, catalog, lifecycle.",
    )
    # The family carries its own dispatcher, so the umbrella entrypoint routes it
    # without importing this module or growing a second copy of the routing table.
    parser.set_defaults(rl_dispatch=dispatch)
    commands = parser.add_subparsers(dest="rl_command", required=True)

    run = commands.add_parser("run", help="Execute a training run from a config file.")
    run.add_argument("--config", required=True, help="Path to a run config file.")
    run.add_argument("--receipts", help="Directory the run leaves its receipt in.")
    run.add_argument("--max-ticks", type=int, default=256)
    run.add_argument(
        "--plane",
        metavar="MODULE:FACTORY",
        help="Overrides the default assembly of the session, gateway, and binder.",
    )
    run.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the configuration, print its plan hash, and start nothing.",
    )
    run.add_argument("--json", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Paired baseline/trained evaluation of a selector."
    )
    evaluate.add_argument("--catalog", required=True, help="Checkpoint catalog path.")
    evaluate.add_argument(
        "--selector",
        required=True,
        help="What to evaluate: a checkpoint id, a policy-set or match-set revision, or an alias.",
    )
    evaluate.add_argument(
        "--baseline",
        help="The arm to compare against. Defaults to the run's registered baseline alias.",
    )
    evaluate.add_argument("--match-set", help="Pinned match-set revision both arms play.")
    evaluate.add_argument(
        "--evaluation-id", help="Names the receipt and the catalog relations it appends."
    )
    evaluate.add_argument(
        "--seed",
        action="append",
        default=[],
        metavar="TASK_ID=SEED",
        help="Repeatable. The held-out set; both arms run exactly this list.",
    )
    evaluate.add_argument(
        "--roster",
        action="append",
        default=[],
        metavar="INSTANCE=GROUP[:POLICY_TYPE]",
        help="Repeatable. The roster both arms bind.",
    )
    evaluate.add_argument("--split", default="heldout")
    evaluate.add_argument("--scope-run")
    evaluate.add_argument("--scope-parameter-group")
    evaluate.add_argument("--scope-policy-type")
    evaluate.add_argument("--metric", default="mean_reward")
    evaluate.add_argument("--reward-channel")
    evaluate.add_argument(
        "--artifact-digests",
        help="JSON object of provider reference -> observed digest.",
    )
    evaluate.add_argument("--pin", help="JSON file carrying the run-invariant group pin fields.")
    evaluate.add_argument("--config", help="Run config describing the container to evaluate in.")
    evaluate.add_argument(
        "--plane",
        metavar="MODULE:FACTORY",
        help="Overrides the default assembly of the session, gateway, and binder.",
    )
    evaluate.add_argument("--receipts-dir", help="Where to write the evaluation receipt.")
    evaluate.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve and verify both arms, print what they resolved to, and run nothing.",
    )
    evaluate.add_argument("--json", action="store_true")

    catalog = commands.add_parser("catalog", help="Read the append-only checkpoint catalog.")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)

    listing = catalog_commands.add_parser("list", help="List catalog entries by any index.")
    listing.add_argument("--catalog", required=True)
    listing.add_argument("--run", help="Index: producing run.")
    listing.add_argument("--update", help="Index: producing update.")
    listing.add_argument("--parameter-group", help="Index: parameter group.")
    listing.add_argument("--policy-type", help="Index: policy type.")
    listing.add_argument("--parent", help="Index: parent checkpoint.")
    listing.add_argument(
        "--status", choices=sorted(PUBLICATION_STATUSES), help="Index: publication status."
    )
    listing.add_argument("--policy-set", help="Index: policy-set revision membership.")
    listing.add_argument("--train-call", help="Index: provider train request.")
    listing.add_argument("--base-model", help="Index: base model.")
    listing.add_argument("--metric", help="Index: evaluation metric; ranks by that metric.")
    listing.add_argument(
        "--metric-direction",
        choices=("max", "min"),
        default="max",
        help="How --metric ranks. Defaults to maximizing.",
    )
    listing.add_argument("--limit", type=int)
    listing.add_argument("--json", action="store_true")

    describe = catalog_commands.add_parser(
        "describe", help="Describe one catalog entry, resolving a selector first."
    )
    describe.add_argument("--catalog", required=True)
    describe.add_argument("selector", help="Checkpoint id, revision id, or alias.")
    describe.add_argument("--artifact-digests", help="Required to resolve an alias.")
    describe.add_argument("--scope-run")
    describe.add_argument("--scope-parameter-group")
    describe.add_argument("--scope-policy-type")
    describe.add_argument("--json", action="store_true")

    receipt = commands.add_parser("receipt", help="Show a run's receipt.")
    receipt.add_argument("--journal", required=True, help="Queue/lifecycle journal path.")
    receipt.add_argument("--run-id", required=True)
    receipt.add_argument("--catalog", help="Checkpoint catalog, for the run's checkpoint rows.")
    receipt.add_argument("--json", action="store_true")

    for control in LIFECYCLE_CONTROLS:
        command = commands.add_parser(
            control, help=f"{control.capitalize()} a live run at its queue boundaries."
        )
        command.add_argument("--journal", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--reason", default="")
        command.add_argument("--json", action="store_true")
    commands.choices["drain"].add_argument(
        "--finish",
        action="store_true",
        help="Close a drain once in-flight work is done and complete groups are trained.",
    )
    commands.choices["resume"].add_argument(
        "--rehandshake",
        help="JSON file holding the run identity returned by re-handshaking the live container.",
    )


# ------------------------------------------------------------------- dispatch


def dispatch(args: argparse.Namespace) -> int:
    """Route one ``rl`` command. A refusal is exit 1 with a legible message."""

    command = args.rl_command
    handlers = {
        "run": _run,
        "evaluate": _evaluate,
        "catalog": _catalog,
        "receipt": _receipt,
    }
    handler = handlers.get(command)
    try:
        if handler is not None:
            return handler(args)
        if command in LIFECYCLE_CONTROLS:
            return _lifecycle(args)
    except PLANE_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except RecordError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    raise SystemExit(f"unknown rl command {command}")


# ------------------------------------------------------------------ commands


def _run(args: argparse.Namespace) -> int:
    """Start a training run. The loop is the executor's; the assembly is a seam."""

    from .config import load as load_run_config
    from .executor import ExecutionPlan, execute
    from .session import RunClock

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"cannot read {args.config}: no such config file")
    config = load_run_config(config_path)
    plan_hash = config.expanded_plan().plan_hash
    print(f"run {config.run_id}: plan={plan_hash} target={config.plan.target_train_updates}")
    if args.validate_only:
        print("configuration is valid; nothing was started (--validate-only)")
        return 0
    if not args.receipts:
        raise SystemExit("--receipts is required: a run that leaves no receipt is not a run")
    plane = _open_plane(args, config)
    # An assembly that opened a listener and a catalog closes them when the run
    # ends, however it ends. A plane that owns nothing declares no ``close``.
    release = getattr(plane, "close", None)
    try:
        report = execute(
            config,
            plane.session,
            plane.gateway,
            plane.binder,
            clock=plane.clock or RunClock(),
            plan=ExecutionPlan(receipts=Path(args.receipts), max_ticks=args.max_ticks),
        )
    finally:
        if callable(release):
            release()
    payload = {
        "run_id": report.run_id,
        "plan_hash": report.plan_hash,
        "stop_reason": report.stop_reason,
        "lifecycle_state": report.lifecycle_state,
        "sampled_groups": report.sampled_groups,
        "updates": len(report.updates),
        "trained_groups": list(report.trained_groups),
        "skipped_groups": list(report.skipped_groups),
        "stale_groups": list(report.stale_groups),
        "receipt_directory": str(report.receipt_directory),
        "final_revisions": {
            group: {
                "policy_revision_id": revision.revision_id,
                "checkpoint_id": revision.checkpoint_id,
                "sampler_reference": revision.sampler_reference,
            }
            for group, revision in report.final_revisions.items()
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(
        f"run {report.run_id} {report.stop_reason}: updates={len(report.updates)} "
        f"sampled_groups={report.sampled_groups} state={report.lifecycle_state}"
    )
    for group, revision in report.final_revisions.items():
        print(
            f"  {group}: selector={revision.revision_id} resolved={revision.checkpoint_id} "
            f"ref={revision.sampler_reference}"
        )
    print(f"receipts: {report.receipt_directory}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    catalog = CheckpointCatalog(args.catalog)
    try:
        resolver = _resolver(catalog, args)
        scope = _scope(args)
        baseline_selector = args.baseline or "baseline"
        trained = resolver.resolve(args.selector, scope=scope)
        baseline = resolver.resolve(baseline_selector, scope=scope)
        _print_resolution("trained", trained)
        _print_resolution("baseline", baseline)
        match_set = None
        if args.match_set:
            match_set = resolver.resolve_match_set(args.match_set, scope=scope)
            _print_resolution("match-set", match_set)
        if args.resolve_only:
            payload = {
                "resolve_only": True,
                "arms": {
                    "baseline": baseline.to_receipt(),
                    "trained": trained.to_receipt(),
                },
                "match_set": None if match_set is None else match_set.to_receipt(),
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("resolved and verified; no attempt was run (--resolve-only)")
            return 0
        request = _request(args, baseline_selector)
        plane = _open_plane(args, _optional_run_config(args))
        receipt = PairedEvaluation(
            resolver,
            session=plane.session,
            gateway=plane.gateway,
            binder=plane.binder,
        ).run(request)
        if args.receipts_dir:
            print(f"receipt: {receipt.write(args.receipts_dir)}")
        if args.json:
            print(json.dumps(receipt.to_payload(), indent=2, sort_keys=True))
        else:
            _print_summary(receipt)
        return 0
    finally:
        catalog.close()


def _catalog(args: argparse.Namespace) -> int:
    catalog = CheckpointCatalog(args.catalog)
    try:
        if args.catalog_command == "list":
            return _catalog_list(catalog, args)
        if args.catalog_command == "describe":
            return _catalog_describe(catalog, args)
    finally:
        catalog.close()
    raise SystemExit(f"unknown rl catalog command {args.catalog_command}")


def _catalog_list(catalog: CheckpointCatalog, args: argparse.Namespace) -> int:
    resolver = EvaluationResolver(
        catalog,
        probe=_RefusingProbe(),
        metric_directions={args.metric: args.metric_direction} if args.metric else {},
        require_ready=False,
    )
    views = resolver.list_checkpoints(
        run_id=args.run,
        update_id=args.update,
        parameter_group_id=args.parameter_group,
        policy_type_id=args.policy_type,
        parent_checkpoint_id=args.parent,
        publication_status=args.status,
        base_model=args.base_model,
        train_call_id=args.train_call,
        policy_set_revision_id=args.policy_set,
        evaluation_metric=args.metric,
        limit=args.limit,
    )
    ranking: dict[str, float] = {}
    if args.metric:
        ranking = {
            target_id: value
            for _kind, target_id, value in resolver.list_by_metric(
                args.metric, target_kind="checkpoint"
            )
        }
        views = tuple(
            sorted(
                views,
                key=lambda view: ranking.get(view.checkpoint_id, 0.0),
                reverse=args.metric_direction == "max",
            )
        )
    rows = [
        {
            "checkpoint_id": view.checkpoint_id,
            "run_id": view.record.run_id,
            "update_id": view.record.update_id,
            "parameter_group_id": view.record.parameter_group_id,
            "policy_type_ids": list(view.record.policy_type_ids),
            "parent_checkpoint_id": view.record.parent_checkpoint_id,
            "publication_status": view.publication_status,
            "policy_set_revision_ids": list(view.policy_set_revision_ids),
            "evaluation_ids": list(view.evaluation_ids),
            "metric": ranking.get(view.checkpoint_id),
        }
        for view in views
    ]
    if args.json:
        print(json.dumps({"checkpoints": rows}, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no checkpoint matches that index")
        return 0
    for row in rows:
        metric = "" if row["metric"] is None else f"  {args.metric}={row['metric']}"
        print(
            f"{row['checkpoint_id']}  [{row['publication_status']}]  "
            f"run={row['run_id']} update={row['update_id']} "
            f"group={row['parameter_group_id']} types={','.join(row['policy_type_ids'])}"
            f"{metric}"
        )
    return 0


def _catalog_describe(catalog: CheckpointCatalog, args: argparse.Namespace) -> int:
    resolver = _resolver(catalog, args)
    identifier = args.selector
    if not catalog.has_checkpoint(identifier) and catalog.revision_kind(identifier) is None:
        resolution = resolver.resolve(identifier, scope=_scope(args))
        _print_resolution("selector", resolution)
        identifier = resolution.resolved_id
    else:
        print(f"selector={identifier} resolved={identifier}")
    payload = resolver.describe(identifier)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    kind = payload.get("record_kind")
    print(f"{identifier}  [{kind}]")
    for key in sorted(payload):
        if key in {"record_kind", "schema_version"}:
            continue
        print(f"  {key}: {_compact(payload[key])}")
    return 0


def _receipt(args: argparse.Namespace) -> int:
    store = JournalStore(args.journal)
    try:
        identity = store.run_identity(args.run_id)
        events = store.lifecycle_events(args.run_id)
        payload: dict[str, Any] = {
            "run_id": args.run_id,
            "identity": dict(identity.to_payload()),
            "binding_digest": identity.binding_digest,
            "lifecycle_state": store.lifecycle_state(args.run_id),
            "lifecycle_transitions": [
                {
                    "cursor": row.cursor,
                    "control": row.subject,
                    "from_state": row.from_state,
                    "to_state": row.to_state,
                    "reason": row.reason,
                    "detail": dict(row.detail),
                }
                for row in events
            ],
        }
        if args.catalog:
            payload.update(_receipt_catalog_rows(args.catalog, args.run_id))
    finally:
        store.close()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print(f"run {args.run_id}  [{payload['lifecycle_state']}]")
    print(f"  binding digest: {payload['binding_digest']}")
    for row in payload["lifecycle_transitions"]:
        arrow = row["to_state"] or "refused"
        print(f"  {row['cursor']:>4}  {row['control']}: {row['from_state']} -> {arrow}")
    for checkpoint in payload.get("checkpoints", []):
        print(
            f"  checkpoint {checkpoint['checkpoint_id']}  [{checkpoint['publication_status']}]"
            f"  update={checkpoint['update_id']}"
        )
    for binding in payload.get("evaluations", []):
        print(
            f"  evaluation {binding['evaluation_id']}: selector="
            f"{binding['requested_selector']} resolved={binding['target_id']}"
        )
    return 0


def _receipt_catalog_rows(path: str, run_id: str) -> dict[str, Any]:
    catalog = CheckpointCatalog(path)
    try:
        views = catalog.list_checkpoints(run_id=run_id)
        evaluations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for view in views:
            for binding in catalog.evaluations(checkpoint_id=view.checkpoint_id):
                if binding.evaluation_id in seen:
                    continue
                seen.add(binding.evaluation_id)
                evaluations.append(binding.to_payload())
        return {
            "checkpoints": [view.to_payload() for view in views],
            "evaluations": evaluations,
        }
    finally:
        catalog.close()


def _lifecycle(args: argparse.Namespace) -> int:
    store = JournalStore(args.journal)
    try:
        lifecycle = RunLifecycle(
            store, args.run_id, rehandshake=_rehandshake_hook(args), terminate=None
        )
        control = args.rl_command
        if control == "pause":
            outcome: Any = lifecycle.pause(reason=args.reason)
        elif control == "drain":
            outcome = (
                lifecycle.finish_drain(reason=args.reason or "drain")
                if getattr(args, "finish", False)
                else lifecycle.drain(reason=args.reason)
            )
        elif control == "resume":
            outcome = lifecycle.resume()
        else:
            outcome = lifecycle.stop(reason=args.reason or "stop")
        state = lifecycle.state
        payload: dict[str, Any] = {
            "run_id": args.run_id,
            "control": control,
            "state": state,
        }
        if hasattr(outcome, "as_detail"):
            payload["report"] = dict(outcome.as_detail())
        if control == "drain" and not getattr(args, "finish", False):
            payload["outstanding"] = {
                name: list(ids) for name, ids in lifecycle.outstanding_drain_work().items()
            }
    finally:
        store.close()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print(f"run {args.run_id} {control}: state={payload['state']}")
    for name, ids in (payload.get("outstanding") or {}).items():
        if ids:
            print(f"  outstanding {name}: {len(ids)}")
    report = payload.get("report") or {}
    if report.get("cancelled_attempts"):
        print(f"  cancelled attempts: {len(report['cancelled_attempts'])}")
    if report.get("abandoned_groups"):
        print(f"  abandoned groups: {len(report['abandoned_groups'])}")
    return 0


# ------------------------------------------------------------------- helpers


def _resolver(catalog: CheckpointCatalog, args: argparse.Namespace) -> EvaluationResolver:
    digests = getattr(args, "artifact_digests", None)
    probe: Any = _RefusingProbe()
    if digests:
        probe = MappingArtifactProbe(digests=_digest_map(digests))
    metric = getattr(args, "metric", None)
    return EvaluationResolver(
        catalog,
        probe=probe,
        metric_directions={metric: getattr(args, "metric_direction", "max")} if metric else {},
    )


def _digest_map(path: str) -> dict[str, str]:
    payload = _json_file_object(path)
    bad = [key for key, value in payload.items() if not isinstance(value, str)]
    if bad:
        raise SystemExit(f"{path}: digest values must be strings; offending keys {sorted(bad)}")
    return {str(key): str(value) for key, value in payload.items()}


def _scope(args: argparse.Namespace) -> ResolutionScope:
    return ResolutionScope(
        run_id=getattr(args, "scope_run", None),
        parameter_group_id=getattr(args, "scope_parameter_group", None),
        policy_type_id=getattr(args, "scope_policy_type", None),
    )


def _request(args: argparse.Namespace, baseline_selector: str) -> EvaluationRequest:
    seeds = tuple(_seed(item) for item in args.seed)
    if not seeds:
        raise SystemExit("--seed TASK_ID=SEED is required: both arms run one held-out set")
    roster = tuple(_roster_slot(item) for item in args.roster)
    if not roster:
        raise SystemExit("--roster INSTANCE=GROUP[:POLICY_TYPE] is required")
    pin = _pin_template(args)
    evaluation_id = args.evaluation_id or f"eval_{args.selector}"
    return EvaluationRequest(
        evaluation_id=evaluation_id,
        baseline_selector=baseline_selector,
        trained_selector=args.selector,
        seeds=seeds,
        roster=roster,
        pin=pin,
        split=args.split,
        match_set_selector=args.match_set,
        scope=_scope(args),
        reward_channel=args.reward_channel,
        metric_name=args.metric,
    )


def _seed(item: str) -> HeldOutSeed:
    task_id, _, raw = str(item).partition("=")
    if not task_id or not raw:
        raise SystemExit(f"--seed expects TASK_ID=SEED, got {item!r}")
    try:
        return HeldOutSeed(task_id=task_id, seed=int(raw))
    except ValueError as error:
        raise SystemExit(f"--seed {item!r}: seed must be an integer") from error


def _roster_slot(item: str) -> RosterSlot:
    instance, _, rest = str(item).partition("=")
    if not instance or not rest:
        raise SystemExit(f"--roster expects INSTANCE=GROUP[:POLICY_TYPE], got {item!r}")
    group, _, policy_type = rest.partition(":")
    return RosterSlot(
        agent_instance_id=instance,
        parameter_group_id=group,
        policy_type_id=policy_type or None,
    )


def _pin_template(args: argparse.Namespace) -> PinTemplate:
    if not args.pin:
        raise SystemExit(
            "--pin is required: an evaluation attempt carries the same pinned identity "
            "fields a rollout does"
        )
    payload = _json_file_object(args.pin)
    required = (
        "run_id",
        "algorithm_plan_hash",
        "wire_api",
        "sampling_transport",
        "policy_kind",
        "model_family",
        "container_image_digest",
        "container_contract_hash",
        "task_family",
    )
    missing = [name for name in required if not str(payload.get(name) or "").strip()]
    if missing:
        raise SystemExit(f"{args.pin} is missing pin field(s): {missing}")
    return PinTemplate(
        run_id=str(payload["run_id"]),
        algorithm_plan_hash=str(payload["algorithm_plan_hash"]),
        wire_api=str(payload["wire_api"]),
        sampling_transport=str(payload["sampling_transport"]),
        policy_kind=str(payload["policy_kind"]),
        model_family=str(payload["model_family"]),
        container_image_digest=str(payload["container_image_digest"]),
        container_contract_hash=str(payload["container_contract_hash"]),
        task_family=str(payload["task_family"]),
        topology_id=payload.get("topology_id"),
    )


def _rehandshake_hook(args: argparse.Namespace) -> Any:
    """Resume re-verifies the agreement; it never assumes the binding held."""

    path = getattr(args, "rehandshake", None)
    if args.rl_command != "resume":
        return None
    if not path:
        raise SystemExit(
            "resume requires --rehandshake: the agreement must be re-verified against the "
            "live container before any work is re-admitted"
        )
    payload = _json_file_object(path)
    identity = RunIdentity.from_payload(payload)
    return lambda: identity


def _default_plane(config: Any) -> Plane:
    """No ``--plane``: assemble the real one from the parsed configuration.

    Every construction failure arrives here as a typed refusal naming what was
    missing -- a credential, a reachable container, a writable catalog path --
    and is re-raised as one legible line rather than a traceback.
    """

    from .plane import PlaneError, build_plane

    try:
        return build_plane(config)
    except PlaneError as error:
        raise SystemExit(
            f"cannot assemble the container plane from this configuration: {error}. "
            "Pass --plane MODULE:FACTORY to name an assembly of your own, or resolve "
            "offline with --resolve-only"
        ) from error


def _open_plane(args: argparse.Namespace, config: Any) -> Plane:
    """The live session, sampler gateway, and binder.

    ``--plane MODULE:FACTORY`` names an assembly and overrides everything. In
    its absence the default assembly is built from the run configuration, so a
    command that was handed one refuses only when there is no configuration to
    build from.
    """

    spec = getattr(args, "plane", None)
    if not spec:
        if config is not None:
            return _default_plane(config)
        raise SystemExit(
            "no container plane can be assembled without a run configuration: pass "
            "--config so the default plane has a container to build against, name one "
            "with --plane MODULE:FACTORY, or resolve offline with --resolve-only"
        )
    module_name, _, attribute = str(spec).partition(":")
    if not module_name or not attribute:
        raise SystemExit(f"--plane expects MODULE:FACTORY, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise SystemExit(f"--plane {spec}: {error}") from error
    factory = getattr(module, attribute, None)
    if factory is None:
        raise SystemExit(f"--plane {spec}: {module_name} declares no {attribute}")
    plane = factory(config=config)
    ports = ("session", "gateway", "binder")
    missing = [name for name in ports if getattr(plane, name, None) is None]
    if missing:
        raise SystemExit(f"--plane {spec} returned no {missing}")
    return plane


def _optional_run_config(args: argparse.Namespace) -> Any:
    """The run configuration an arm is evaluated inside, when one was named."""

    if not getattr(args, "config", None):
        return None
    from .config import load as load_run_config

    return load_run_config(Path(args.config))


def _print_resolution(label: str, resolution: Resolution) -> None:
    alias = f" alias={resolution.alias}" if resolution.alias else ""
    print(
        f"{label}: selector={resolution.requested_selector} "
        f"resolved={resolution.resolved_kind}:{resolution.resolved_id}{alias}"
    )
    for policy in resolution.policies:
        print(
            f"    {policy.parameter_group_id}: checkpoint={policy.checkpoint_id} "
            f"ref={policy.artifact.ref} digest={policy.artifact.digest}"
        )
    for opponent in resolution.opponents:
        print(
            f"    opponent {opponent.opponent_id}: {opponent.binding_kind}="
            f"{opponent.identity}"
        )


def _print_summary(receipt: Any) -> None:
    summary = receipt.summary
    for selector, resolved in receipt.selector_resolutions:
        print(f"evaluated: selector={selector} resolved={resolved}")
    print(
        f"pairs={summary.pairs} baseline={summary.baseline_mean:.6g} "
        f"trained={summary.trained_mean:.6g} delta={summary.mean_delta:+.6g} "
        f"wins={summary.wins} losses={summary.losses} ties={summary.ties}"
    )
    for binding in receipt.bindings:
        print(f"binding {binding.evaluation_id} -> {binding.target_kind}:{binding.target_id}")


def _compact(value: Any) -> str:
    if isinstance(value, (str, int, float)) or value is None:
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def _json_file_object(path: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry, for driving the plane without the umbrella parser."""

    parser = argparse.ArgumentParser(prog="synth-optimizers rl")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["rl", *(argv or [])])
    return dispatch(args)


__all__ = ["LIFECYCLE_CONTROLS", "PLANE_ERRORS", "Plane", "dispatch", "main", "register"]
