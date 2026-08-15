from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from . import __version__
from ._synth_optimizers import (
    SynthOptimizerError,
    events_compare,
    events_replay,
    gepa_compact_run_storage,
    gepa_delete_run_storage,
    gepa_inspect_run_storage,
    gepa_workspace_storage_health,
    gepa_serve,
)
from .gelo import (
    GeloMaterializeError,
    GeloMaterializer,
    GeloPreset,
    GeloPresetName,
)
from .hosted import (
    ContainerPoolTarget,
    HostedOptimizerClient,
    HostedOptimizerError,
    validate_online_reflexion_evidence_notes,
)
from .sft import SftConfig, SftPublicServiceClient, SftServiceError, serve_sft_service
from .tunnels import TunnelError, TunnelProvider
from .victorialogs import project_gepa_run_artifacts, project_gepa_run_started


def _duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    unit = value[-1]
    number = value[:-1] if unit.isalpha() else value
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }.get(unit, 1)
    try:
        return float(number) * multiplier
    except ValueError as exc:
        raise SystemExit(f"invalid --older-than duration: {value}") from exc


def _bytes_value(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    unit = text[-1].lower()
    number = text[:-1] if unit.isalpha() else text
    multiplier = {
        "b": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }.get(unit, 1)
    try:
        return int(float(number) * multiplier)
    except ValueError as exc:
        raise SystemExit(f"invalid byte value: {value}") from exc


def _run_manifest_status(run_dir: Path) -> str | None:
    manifest = run_dir / "result_manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {manifest}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{manifest} is not valid JSON: {exc}") from exc
    status = data.get("status") or data.get("final_status") or data.get("run_status")
    return str(status) if status else "terminal"


def _is_old_enough(run_dir: Path, older_than_seconds: float | None) -> bool:
    if older_than_seconds is None:
        return True
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        return False
    return time.time() - mtime >= older_than_seconds


def _resolve_gepa_run_dirs(
    runs: Sequence[str],
    roots: Sequence[str],
    all_terminal: bool,
    older_than: str | None,
    statuses: Sequence[str],
    *,
    all_root_children: bool = False,
) -> list[Path]:
    older_than_seconds = _duration_seconds(older_than)
    status_filter = set(statuses)
    resolved: list[Path] = []
    root_paths = [Path(root) for root in roots]
    for run in runs:
        path = Path(run)
        if not path.exists():
            for root in root_paths:
                candidate = root / run
                if candidate.exists():
                    path = candidate
                    break
        resolved.append(path)
    if all_terminal or all_root_children:
        if not root_paths:
            raise SystemExit("--root is required for bulk run discovery")
        for root in root_paths:
            for child in sorted(root.iterdir() if root.exists() else []):
                if not child.is_dir():
                    continue
                status = _run_manifest_status(child)
                if all_terminal and status is None:
                    continue
                if status_filter and status not in status_filter:
                    continue
                if not _is_old_enough(child, older_than_seconds):
                    continue
                resolved.append(child)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _print_storage_reports(reports: list[dict], json_output: bool) -> None:
    if json_output:
        print(json.dumps(reports, indent=2, sort_keys=True))
        return
    for report in reports:
        run_dir = report.get("run_dir")
        dry_run = report.get("dry_run")
        before = int(report.get("before_bytes") or report.get("bytes") or 0)
        after = int(report.get("after_bytes") or 0)
        estimated = int(report.get("estimated_reclaim_bytes") or report.get("bytes") or 0)
        mode = "dry-run" if dry_run else "applied"
        if "after_bytes" in report:
            print(f"{mode}: {run_dir} before={before} after={after} estimated_reclaim={estimated}")
        else:
            print(f"{mode}: {run_dir} bytes={before}")


def _format_bytes(value: object) -> str:
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        size = 0.0
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _print_run_storage_list(reports: list[dict], json_output: bool) -> None:
    if json_output:
        print(json.dumps(reports, indent=2, sort_keys=True))
        return
    print(f"{'run':32} {'status':14} {'size':>10} {'reclaim':>10} recommendation")
    for report in reports:
        recommendation = report.get("recommendation") or {}
        run_id = str(report.get("run_id") or Path(str(report.get("run_dir") or "")).name)
        status = str(report.get("terminal_status") or "unknown")
        if not bool(report.get("terminal")):
            status = f"{status}*"
        action = recommendation.get("action") or "none"
        profile = recommendation.get("profile")
        label = f"{action}:{profile}" if profile else action
        print(
            f"{run_id[:32]:32} {status[:14]:14} "
            f"{_format_bytes(report.get('bytes')):>10} "
            f"{_format_bytes(report.get('reclaimable_bytes')):>10} {label}"
        )
    print("* cleanup disabled until the run is terminal")


def _print_run_storage_detail(report: dict, json_output: bool, *, doctor: bool = False) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    run_id = report.get("run_id") or Path(str(report.get("run_dir") or "")).name
    recommendation = report.get("recommendation") or {}
    print(f"run: {run_id}")
    print(f"path: {report.get('run_dir')}")
    print(
        f"status: {report.get('terminal_status')} "
        f"({'terminal' if report.get('terminal') else 'not terminal'})"
    )
    print(f"size: {_format_bytes(report.get('bytes'))}")
    print(f"reclaimable: {_format_bytes(report.get('reclaimable_bytes'))}")
    print(
        "recommendation: "
        f"{recommendation.get('action') or 'none'}"
        f"{':' + recommendation.get('profile') if recommendation.get('profile') else ''}"
        f" — {recommendation.get('reason') or 'no recommendation'}"
    )
    artifacts = report.get("artifact_summary") or []
    if artifacts:
        print("\nartifacts:")
        for artifact in artifacts[:12]:
            print(f"  {_format_bytes(artifact.get('bytes')):>10}  {artifact.get('name')}")
    sqlite = report.get("sqlite") or []
    if sqlite:
        print("\nsqlite:")
        for db in sqlite:
            print(f"  {_format_bytes(db.get('bytes')):>10}  {db.get('path')}")
            for obj in (db.get("objects") or [])[:8]:
                print(f"    {_format_bytes(obj.get('bytes')):>8}  {obj.get('name')}")
            if db.get("error"):
                print(f"    dbstat unavailable: {db.get('error')}")
    top_files = report.get("top_files") or []
    if top_files:
        print("\ntop files:")
        for item in top_files[:12]:
            print(f"  {_format_bytes(item.get('bytes')):>10}  {item.get('relative_path')}")
    if doctor:
        print("\nnext command:")
        if report.get("terminal") and recommendation.get("action") == "compact":
            profile = recommendation.get("profile") or "compact"
            print(
                f"  synth-optimizers gepa runs compact {report.get('run_dir')} --profile {profile}"
            )
        elif report.get("terminal"):
            print("  no compaction needed; use gepa runs delete only if you want to remove the run")
        else:
            print("  no cleanup command; wait for terminal status or inspect the live run")


def _print_storage_health(report: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    summary = report.get("summary") or {}
    print(
        "storage: "
        f"{_format_bytes(summary.get('bytes'))} across "
        f"{summary.get('run_count', 0)} runs; "
        f"{_format_bytes(summary.get('stale_partial_bytes'))} stale partials; "
        f"{summary.get('alert_count', 0)} alert(s)"
    )
    alerts = report.get("alerts") or []
    if alerts:
        print("\nalerts:")
        for alert in alerts:
            target = alert.get("run_id") or alert.get("root") or alert.get("path") or "workspace"
            print(
                f"  {alert.get('kind')}: {_format_bytes(alert.get('bytes'))} "
                f">= {_format_bytes(alert.get('threshold_bytes'))}  {target}"
            )
    roots = report.get("roots") or []
    if roots:
        print("\nroots:")
        for root in roots:
            print(
                f"  {_format_bytes(root.get('bytes')):>10}  "
                f"runs={root.get('run_count', 0)} "
                f"partials={root.get('partial_count', 0)} "
                f"stale_partials={_format_bytes(root.get('stale_partial_bytes'))}  "
                f"{root.get('root')}"
            )
            for partial in (root.get("partials") or [])[:5]:
                stale = " stale" if partial.get("stale") else ""
                print(
                    f"    partial{stale} {_format_bytes(partial.get('bytes')):>10}  "
                    f"{partial.get('path')}"
                )


def _api_endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _hosted_client(args: argparse.Namespace) -> HostedOptimizerClient:
    return HostedOptimizerClient(
        backend_url=args.base_url,
        api_key=os.environ.get(args.api_key_env),
        timeout_seconds=args.timeout_seconds,
        register_usage=False if bool(getattr(args, "disable_usage_registration", False)) else None,
        usage_registration_surface="cli",
    )


def _container_pool_from_args(args: argparse.Namespace) -> ContainerPoolTarget | None:
    pool_id = getattr(args, "container_pool", None)
    task_id = getattr(args, "container_task_id", None)
    if task_id and not pool_id:
        raise SystemExit("--container-task-id requires --container-pool")
    if not pool_id:
        return None
    return ContainerPoolTarget(pool_id=pool_id, task_id=task_id)


def _tunnel_provider_from_args(args: argparse.Namespace) -> TunnelProvider:
    raw = getattr(args, "tunnel_provider", None) or TunnelProvider.SYNTH_TUNNEL.value
    try:
        return TunnelProvider(raw)
    except ValueError as exc:
        raise SystemExit(
            "--tunnel-provider must be auto, synth_tunnel, cloudflared, or ngrok"
        ) from exc


def _tunnel_ttl_seconds_from_args(args: argparse.Namespace) -> int:
    value = int(getattr(args, "tunnel_ttl_seconds", 86400) or 86400)
    if value < 60 or value > 86400:
        raise SystemExit("--tunnel-ttl-seconds must be between 60 and 86400")
    return value


def _close_tunnel_quietly(tunnel: Any | None) -> None:
    if tunnel is None:
        return
    try:
        tunnel.close()
    except Exception as exc:
        print(f"warning: tunnel close failed: {exc}", file=sys.stderr)


def _validate_container_args(args: argparse.Namespace) -> None:
    has_container_url = bool(getattr(args, "container_url", None))
    has_container_pool = bool(getattr(args, "container_pool", None))
    has_tunnel_url = bool(getattr(args, "tunnel_url", None))
    if has_container_url and has_container_pool:
        raise SystemExit("--container-url and --container-pool are mutually exclusive")
    if has_tunnel_url and (has_container_url or has_container_pool):
        raise SystemExit("--tunnel-url cannot be combined with --container-url or --container-pool")


def _require_tunnel_follow(args: argparse.Namespace) -> None:
    if getattr(args, "tunnel_url", None) and not bool(getattr(args, "follow", False)):
        raise SystemExit(
            "--tunnel-url requires --follow so the CLI keeps the tunnel open until "
            "the hosted run reaches a terminal status"
        )


def _add_gelo_jesterky_workflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--jesterky-workflow",
        dest="jesterky_workflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable go_ex.jesterky_workflow for this GELO run.",
    )
    parser.add_argument("--jesterky-workflow-spec", default=None)
    parser.add_argument("--jesterky-workflow-command", default=None)
    parser.add_argument("--jesterky-workflow-actor", choices=("fake", "codex"), default=None)
    parser.add_argument("--jesterky-workflow-model", default=None)
    parser.add_argument("--jesterky-workflow-concurrency", type=int, default=None)
    parser.add_argument("--jesterky-workflow-timeout-seconds", type=int, default=None)


def _gelo_preset_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for attr in (
        "proposer_rounds",
        "train_seed_count",
        "heldout_seed_count",
        "max_rollouts",
        "policy_model",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            overrides[attr] = value
    return overrides


def _apply_gelo_jesterky_workflow_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Merge CLI jesterky-workflow flags into a materialized GELO config."""
    enabled = getattr(args, "jesterky_workflow", None)
    if enabled is None and not any(
        getattr(args, attr, None) is not None
        for attr in (
            "jesterky_workflow_spec",
            "jesterky_workflow_command",
            "jesterky_workflow_actor",
            "jesterky_workflow_model",
            "jesterky_workflow_concurrency",
            "jesterky_workflow_timeout_seconds",
        )
    ):
        return config
    go_ex = dict(config.get("go_ex") or {})
    workflow = dict(go_ex.get("jesterky_workflow") or {})
    if enabled is not None:
        workflow["enabled"] = bool(enabled)
    for cli_attr, key in (
        ("jesterky_workflow_spec", "spec"),
        ("jesterky_workflow_command", "command"),
        ("jesterky_workflow_actor", "actor"),
        ("jesterky_workflow_model", "model"),
        ("jesterky_workflow_concurrency", "concurrency"),
        ("jesterky_workflow_timeout_seconds", "timeout_seconds"),
    ):
        value = getattr(args, cli_attr, None)
        if value is not None:
            workflow[key] = value
    if "fail_closed" not in workflow:
        workflow["fail_closed"] = True
    go_ex["jesterky_workflow"] = workflow
    out = dict(config)
    out["go_ex"] = go_ex
    return out


def _gelo_materialized_config(
    args: argparse.Namespace,
    *,
    container_tunnel: Any | None = None,
) -> dict[str, Any]:
    _validate_container_args(args)
    container_url = getattr(args, "container_url", None)
    if container_tunnel is None:
        container_url = container_url or getattr(args, "tunnel_url", None)
    container_pool = _container_pool_from_args(args)
    if getattr(args, "preset", None):
        try:
            config = GeloPreset.from_name(args.preset, **_gelo_preset_overrides(args)).materialize(
                container_url=container_url,
                container_pool=container_pool,
                container_tunnel=container_tunnel,
                run_id=getattr(args, "run_id", None),
            )
        except GeloMaterializeError as exc:
            raise SystemExit(str(exc)) from exc
        return _apply_gelo_jesterky_workflow_overrides(config, args)
    if getattr(args, "toml", None):
        try:
            config = GeloMaterializer.from_paths(
                args.toml,
                getattr(args, "overlay", None),
            ).materialize(
                container_url=container_url,
                container_pool=container_pool,
                container_tunnel=container_tunnel,
                run_id=getattr(args, "run_id", None),
            )
        except GeloMaterializeError as exc:
            raise SystemExit(str(exc)) from exc
        return _apply_gelo_jesterky_workflow_overrides(config, args)
    if getattr(args, "config", None):
        try:
            config = GeloMaterializer(_json_file_object(args.config)).materialize(
                container_url=container_url,
                container_pool=container_pool,
                container_tunnel=container_tunnel,
                run_id=getattr(args, "run_id", None),
            )
        except GeloMaterializeError as exc:
            raise SystemExit(str(exc)) from exc
        return _apply_gelo_jesterky_workflow_overrides(config, args)
    raise SystemExit("one of --config, --preset, or --toml is required")


def _startup_catalog_payload(catalog: Any) -> dict[str, Any]:
    return {
        "available_algorithms": [
            {
                "algorithm": entry.algorithm.value,
                "candidate_kinds": list(entry.candidate_kinds),
                "status": entry.status.value,
                "submit_supported": entry.submit_supported,
            }
            for entry in catalog.available_algorithms
        ],
        "submit_supported": [algorithm.value for algorithm in catalog.submit_supported],
        "org_id": catalog.org_id,
        "optimizers_beta_configured": catalog.optimizers_beta_configured,
        "billing_feature_ids": {
            algorithm: {
                "feature_id": config.feature_id,
                "env_override": config.env_override,
            }
            for algorithm, config in catalog.billing_feature_ids.items()
        },
        "billing_feature_ids_configured": dict(catalog.billing_feature_ids_configured),
        "online_reflexion_release_evidence": dict(catalog.online_reflexion_release_evidence),
    }


def _print_hosted_startup(catalog: Any, json_output: bool) -> None:
    payload = _startup_catalog_payload(catalog)
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(
        "startup "
        f"org_id={payload['org_id'] or '-'} "
        f"optimizers_beta_configured={payload['optimizers_beta_configured']}"
    )
    configured = payload["billing_feature_ids_configured"]
    print(
        "billing_feature_ids_configured "
        f"gepa={configured.get('gepa')} "
        f"go_ex={configured.get('go_ex')} "
        f"mapo={configured.get('mapo')} "
        f"online_reflexion={configured.get('online_reflexion')}"
    )
    for entry in payload["available_algorithms"]:
        candidates = ",".join(entry["candidate_kinds"]) or "-"
        print(
            "algorithm "
            f"{entry['algorithm']} "
            f"status={entry['status']} "
            f"submit_supported={entry['submit_supported']} "
            f"candidate_kinds={candidates}"
        )
    release_evidence = payload["online_reflexion_release_evidence"]
    if release_evidence:
        required_lanes = release_evidence.get("required_lanes")
        release_checks = release_evidence.get("release_gate_required_checks")
        standard_artifacts = release_evidence.get("standard_artifacts")
        lane_count = (
            len(required_lanes)
            if isinstance(required_lanes, Sequence) and not isinstance(required_lanes, str | bytes)
            else 0
        )
        check_count = (
            len(release_checks)
            if isinstance(release_checks, Sequence) and not isinstance(release_checks, str | bytes)
            else 0
        )
        artifact_count = len(standard_artifacts) if isinstance(standard_artifacts, Mapping) else 0
        print(
            "online_reflexion_release_evidence "
            f"schema={release_evidence.get('schema_version') or '-'} "
            f"release_gate={release_evidence.get('release_gate_key') or '-'} "
            f"lanes={lane_count} "
            f"release_checks={check_count} "
            f"standard_artifacts={artifact_count} "
            "public_copy_requires_owner_approval="
            f"{release_evidence.get('public_copy_requires_owner_approval')} "
            "effortbench_chinese_wall="
            f"{release_evidence.get('effortbench_cookbook_chinese_wall') or '-'}"
        )
    else:
        print("online_reflexion_release_evidence missing")


def _startup_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _online_reflexion_startup_preflight_failures(
    payload: Mapping[str, Any], args: argparse.Namespace
) -> list[str]:
    require_algorithm = bool(getattr(args, "require_online_reflexion", False))
    require_metadata = bool(getattr(args, "require_online_reflexion_release_metadata", False))
    if not require_algorithm and not require_metadata:
        return []

    failures: list[str] = []
    algorithms = _startup_sequence(payload.get("available_algorithms"))
    online_reflexion_available = any(
        isinstance(entry, Mapping)
        and entry.get("algorithm") == "online-reflexion"
        and entry.get("status") == "available"
        and entry.get("submit_supported") is True
        for entry in algorithms
    )
    if (require_algorithm or require_metadata) and not online_reflexion_available:
        failures.append("online-reflexion algorithm is not advertised as submit-supported")

    if not require_metadata:
        return failures

    release_evidence = payload.get("online_reflexion_release_evidence")
    if not isinstance(release_evidence, Mapping) or not release_evidence:
        failures.append("online_reflexion_release_evidence metadata is not advertised")
        return failures

    if release_evidence.get("schema_version") != "online_reflexion_release_evidence.v1":
        failures.append("online_reflexion_release_evidence schema_version is not v1")
    if release_evidence.get("release_gate_key") != "release_blog_growth":
        failures.append("online_reflexion release_gate_key is not release_blog_growth")

    lane_keys = {
        str(item.get("key"))
        for item in _startup_sequence(release_evidence.get("required_lanes"))
        if isinstance(item, Mapping) and item.get("key")
    }
    for key in (
        "craftax_rotated_121_125",
        "alfworld_6x6_x3",
        "ebr_first_scale_compare",
        "harvey_lab_pilot",
        "hosted_staging_smoke",
    ):
        if key not in lane_keys:
            failures.append(f"online_reflexion release lane missing: {key}")

    release_checks = _startup_sequence(release_evidence.get("release_gate_required_checks"))
    if not release_checks:
        failures.append("online_reflexion release_gate_required_checks is empty")

    standard_artifacts = release_evidence.get("standard_artifacts")
    if not isinstance(standard_artifacts, Mapping):
        failures.append("online_reflexion standard_artifacts is not advertised")
    else:
        for key in ("events", "exposures", "lever_effects", "summary"):
            if key not in standard_artifacts:
                failures.append(f"online_reflexion standard artifact missing: {key}")

    if release_evidence.get("public_copy_requires_owner_approval") is not True:
        failures.append("online_reflexion owner approval requirement is not advertised")
    if release_evidence.get("effortbench_cookbook_chinese_wall") != "grader_only":
        failures.append("online_reflexion EffortBench Chinese-wall marker is not grader_only")

    return failures


def _gelo_startup(args: argparse.Namespace) -> int:
    try:
        catalog = _hosted_client(args).startup()
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    _print_hosted_startup(catalog, args.json)
    failures = _online_reflexion_startup_preflight_failures(_startup_catalog_payload(catalog), args)
    if failures:
        for failure in failures:
            print(f"startup preflight failed: {failure}", file=sys.stderr)
        return 1
    return 0


def _submit_hosted_gepa(args: argparse.Namespace) -> int:
    try:
        config_text = Path(args.config).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {args.config}: {exc}") from exc
    tunnel: Any | None = None
    try:
        _validate_container_args(args)
        _require_tunnel_follow(args)
        client = _hosted_client(args)
        container_pool = _container_pool_from_args(args)
        if getattr(args, "tunnel_url", None):
            tunnel = client.open_tunnel(
                args.tunnel_url,
                provider=_tunnel_provider_from_args(args),
                requested_ttl_seconds=_tunnel_ttl_seconds_from_args(args),
                metadata={"optimizer": "gepa", "run_id": args.run_id or ""},
            )
            submit = client.submit_gepa_toml(
                config_text,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                container_tunnel=tunnel,
            )
        else:
            submit = client.submit_gepa_toml(
                config_text,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                container_pool=container_pool,
            )
        if args.json and not args.follow:
            print(json.dumps(dict(submit.raw), indent=2, sort_keys=True))
            return 0

        print(f"submitted run_id={submit.run_id} status={submit.status.value}")
        if submit.events_url:
            print(f"events: {_api_endpoint(args.base_url, submit.events_url)}")

        if args.follow:
            for event in client.events(submit.run_id):
                event_type = str(event.get("event_type") or "event")
                status = str(event.get("status") or "")
                print(f"event {event_type} status={status}")
                if status in {"succeeded", "failed", "cancelled"}:
                    break
            final_record = client.get_run(submit.run_id)
            if args.json:
                print(json.dumps(dict(final_record.raw), indent=2, sort_keys=True))
            else:
                print(f"final status={final_record.status.value}")
                if final_record.error:
                    print(f"error: {final_record.error}")
            if final_record.status.value == "failed":
                return 1
        return 0
    except (HostedOptimizerError, TunnelError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        _close_tunnel_quietly(tunnel)


def _json_file_object(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def _config_file_object(path: str) -> dict:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if p.suffix == ".toml":
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise SystemExit(f"{path} is not valid TOML: {exc}") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain an object")
    return data


def _sft_service_client(args: argparse.Namespace) -> SftPublicServiceClient:
    token = os.environ.get(args.service_token_env) if args.service_token_env else None
    return SftPublicServiceClient(args.service_url, token, timeout_seconds=args.timeout_seconds)


def _sft_validate(args: argparse.Namespace) -> int:
    try:
        config = SftConfig.from_toml(
            Path(args.config).read_text(encoding="utf-8"), run_id=args.run_id
        )
    except OSError as exc:
        raise SystemExit(f"cannot read {args.config}: {exc}") from exc
    except SftServiceError as exc:
        raise SystemExit(str(exc)) from exc
    payload = {
        "algorithm": "sft",
        "run_id": config.run_id,
        "backend": config.backend,
        "base_model": config.base_model,
        "checkpoint_steps": list(config.checkpoint_steps),
        "accelerator_slots": config.accelerator_slots,
    }
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if args.json
        else f"valid SFT config run_id={config.run_id} backend={config.backend}"
    )
    return 0


def _sft_submit(args: argparse.Namespace) -> int:
    try:
        config_toml = Path(args.config).read_text(encoding="utf-8")
        client = _sft_service_client(args)
        submitted = client.submit_toml(
            config_toml,
            run_id=args.run_id,
            idempotency_key=args.idempotency_key,
        )
        if args.json and not args.follow:
            print(json.dumps(submitted, indent=2, sort_keys=True))
            return 0
        run_id = str(submitted["run_id"])
        print(f"submitted run_id={run_id} status={submitted.get('status', 'queued')}")
        if not args.follow:
            return 0
        while True:
            record = client.get(run_id)
            status = str(record.get("status", "unknown"))
            print(f"status={status}")
            if status in {"succeeded", "failed", "cancelled"}:
                if args.json:
                    print(json.dumps(record, indent=2, sort_keys=True))
                return 1 if status == "failed" else 0
            time.sleep(args.poll_seconds)
    except (OSError, SftServiceError) as exc:
        raise SystemExit(str(exc)) from exc


def _sft_watch(args: argparse.Namespace) -> int:
    try:
        client = _sft_service_client(args)
        record = client.get(args.run_id)
        if args.events:
            page = client.optimizer_events(
                args.run_id, after_sequence=args.after_seq, limit=args.limit
            )
            record["events"] = page.get("events", [])
        print(
            json.dumps(record, indent=2, sort_keys=True)
            if args.json
            else f"run_id={args.run_id} status={record.get('status')}"
        )
        return 1 if record.get("status") == "failed" else 0
    except SftServiceError as exc:
        raise SystemExit(str(exc)) from exc


def _sft_cancel(args: argparse.Namespace) -> int:
    try:
        record = _sft_service_client(args).cancel(args.run_id)
    except SftServiceError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(record, indent=2, sort_keys=True)
        if args.json
        else f"run_id={args.run_id} status={record.get('status')}"
    )
    return 0


def _submit_hosted_gelo(args: argparse.Namespace) -> int:
    tunnel: Any | None = None
    try:
        _validate_container_args(args)
        _require_tunnel_follow(args)
        client = _hosted_client(args)
        container_pool = _container_pool_from_args(args)
        if args.tunnel_url:
            tunnel = client.open_tunnel(
                args.tunnel_url,
                provider=_tunnel_provider_from_args(args),
                requested_ttl_seconds=_tunnel_ttl_seconds_from_args(args),
                metadata={"optimizer": "gelo", "run_id": args.run_id or ""},
            )
            config = _gelo_materialized_config(args, container_tunnel=tunnel)
            submit = client.submit_gelo(
                config,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                billing_mode=args.billing_mode,
            )
        else:
            config = _gelo_materialized_config(args)
            submit = client.submit_gelo(
                config,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                container_pool=container_pool,
                billing_mode=args.billing_mode,
            )
        if args.json and not args.follow:
            print(json.dumps(dict(submit.raw), indent=2, sort_keys=True))
            return 0

        print(f"submitted run_id={submit.run_id} status={submit.status.value}")
        if submit.events_url:
            print(f"events: {_api_endpoint(args.base_url, submit.events_url)}")

        if args.follow:
            for event in client.events(submit.run_id):
                event_type = str(event.get("event_type") or "event")
                status = str(event.get("status") or "")
                print(f"event {event_type} status={status}")
                if status in {"succeeded", "failed", "cancelled"}:
                    break
            final_record = client.get_run(submit.run_id)
            if args.json:
                print(json.dumps(dict(final_record.raw), indent=2, sort_keys=True))
            else:
                print(f"final status={final_record.status.value}")
                if final_record.error:
                    print(f"error: {final_record.error}")
            if final_record.status.value == "failed":
                return 1
        return 0
    except (HostedOptimizerError, TunnelError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        _close_tunnel_quietly(tunnel)


def _submit_hosted_mapo(args: argparse.Namespace) -> int:
    tunnel: Any | None = None
    try:
        _validate_container_args(args)
        _require_tunnel_follow(args)
        client = _hosted_client(args)
        container_pool = _container_pool_from_args(args)
        if args.tunnel_url:
            tunnel = client.open_tunnel(
                args.tunnel_url,
                provider=_tunnel_provider_from_args(args),
                requested_ttl_seconds=_tunnel_ttl_seconds_from_args(args),
                metadata={"optimizer": "mapo", "run_id": args.run_id or ""},
            )
            submit = client.submit_mapo(
                _config_file_object(args.config),
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                container_tunnel=tunnel,
                billing_mode=args.billing_mode,
            )
        else:
            submit = client.submit_mapo(
                _config_file_object(args.config),
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                container_pool=container_pool,
                billing_mode=args.billing_mode,
            )
        if args.json and not args.follow:
            print(json.dumps(dict(submit.raw), indent=2, sort_keys=True))
            return 0

        print(f"submitted run_id={submit.run_id} status={submit.status.value}")
        if submit.events_url:
            print(f"events: {_api_endpoint(args.base_url, submit.events_url)}")

        if args.follow:
            for event in client.events(submit.run_id):
                event_type = str(event.get("event_type") or "event")
                status = str(event.get("status") or "")
                print(f"event {event_type} status={status}")
                if status in {"succeeded", "failed", "cancelled"}:
                    break
            final_record = client.get_run(submit.run_id)
            if args.json:
                print(json.dumps(dict(final_record.raw), indent=2, sort_keys=True))
            else:
                print(f"final status={final_record.status.value}")
                if final_record.error:
                    print(f"error: {final_record.error}")
            if final_record.status.value == "failed":
                return 1
        return 0
    except (HostedOptimizerError, TunnelError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        _close_tunnel_quietly(tunnel)


def _submit_hosted_reflexion(args: argparse.Namespace) -> int:
    try:
        client = _hosted_client(args)
        submit = client.submit_online_reflexion(
            _config_file_object(args.config),
            run_id=args.run_id,
            idempotency_key=args.idempotency_key,
            project_id=args.project_id,
            container_pool=_container_pool_from_args(args),
        )
        if args.json and not args.follow:
            print(json.dumps(dict(submit.raw), indent=2, sort_keys=True))
            return 0

        print(f"submitted run_id={submit.run_id} status={submit.status.value}")
        if submit.events_url:
            print(f"events: {_api_endpoint(args.base_url, submit.events_url)}")

        if args.follow:
            for event in client.events(submit.run_id):
                event_type = str(event.get("event_type") or "event")
                status = str(event.get("status") or "")
                print(f"event {event_type} status={status}")
                if status in {"succeeded", "failed", "cancelled"}:
                    break
            final_record = client.get_run(submit.run_id)
            if args.json:
                print(json.dumps(dict(final_record.raw), indent=2, sort_keys=True))
            else:
                print(f"final status={final_record.status.value}")
                if final_record.error:
                    print(f"error: {final_record.error}")
            if final_record.status.value == "failed":
                return 1
        return 0
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc


def _run_ids_from_args(args: argparse.Namespace) -> list[str] | None:
    values: list[str] = []
    for raw in getattr(args, "run_id", None) or []:
        values.extend(item.strip() for item in str(raw).split(",") if item.strip())
    for raw in getattr(args, "run_ids", None) or []:
        values.extend(item.strip() for item in str(raw).split(",") if item.strip())
    return values or None


def _print_json_payload(payload: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        return
    status = _text_field(payload.get("status"))
    schema = _text_field(payload.get("schema_version"))
    print(f"{schema} status={status}")
    counts = payload.get("counts")
    if isinstance(counts, Mapping):
        print(f"counts {_compact_json(counts)}")
    remaining = payload.get("remaining")
    if isinstance(remaining, Sequence) and not isinstance(remaining, str | bytes):
        for item in remaining:
            print(f"remaining: {item}")


def _write_json_payload(path: str | None, payload: Mapping[str, Any]) -> Path | None:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def _online_reflexion_receipt(args: argparse.Namespace) -> int:
    try:
        payload = _hosted_client(args).online_reflexion_receipt(
            args.run_id,
            exposure_limit=args.exposure_limit,
            outcome_limit=args.outcome_limit,
        )
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
    else:
        run = _as_mapping(payload.get("run"))
        receipt = _as_mapping(payload.get("receipt"))
        print(
            f"run_id={_text_field(run.get('run_id'), args.run_id)} "
            f"status={_text_field(run.get('status'))} "
            f"layer_id={_text_field(receipt.get('layer_id'))}"
        )
        print(
            f"artifacts={len(payload.get('artifacts') or [])} "
            f"exposures={len(payload.get('exposures') or [])} "
            f"outcomes={len(payload.get('outcomes') or [])}"
        )
    return 0


def _online_reflexion_audit(args: argparse.Namespace) -> int:
    try:
        client = _hosted_client(args)
        if getattr(args, "audit_run_id", None):
            payload = client.online_reflexion_receipt_audit(
                args.audit_run_id,
                strict=args.strict,
            )
        else:
            payload = client.online_reflexion_receipt_audits(
                run_ids=_run_ids_from_args(args),
                layer_id=args.layer_id,
                project_id=args.project_id,
                strict=args.strict,
                limit=args.limit,
            )
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    _print_json_payload(payload, json_output=args.json)
    return 0 if payload.get("status") == "pass" else 1


def _json_arg_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--evidence-notes is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--evidence-notes must be a JSON object")
    return payload


def _json_file_arg_object(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return _json_file_object(path)


def _online_reflexion_evidence_packet(args: argparse.Namespace) -> int:
    evidence_notes = _json_file_arg_object(args.evidence_notes_file)
    inline_notes = _json_arg_object(args.evidence_notes)
    if evidence_notes is not None and inline_notes is not None:
        raise SystemExit("provide --evidence-notes or --evidence-notes-file, not both")
    try:
        payload = _hosted_client(args).online_reflexion_evidence_packet(
            run_ids=_run_ids_from_args(args),
            layer_id=args.layer_id,
            project_id=args.project_id,
            evidence_notes=evidence_notes or inline_notes,
            blog_decision_owner=args.blog_decision_owner,
            blog_approved_by_owner=args.blog_approved_by_owner,
            include_receipt_summaries=not args.no_receipt_summaries,
            limit=args.limit,
        )
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    output_path = _write_json_payload(args.out, payload)
    _print_json_payload(payload, json_output=args.json)
    if output_path is not None and not args.json:
        print(f"evidence_packet_path: {output_path}")
    return 0 if payload.get("status") == "ready" else 1


def _online_reflexion_validate_evidence_notes(args: argparse.Namespace) -> int:
    evidence_notes = _json_file_arg_object(args.evidence_notes_file)
    inline_notes = _json_arg_object(args.evidence_notes)
    if evidence_notes is not None and inline_notes is not None:
        raise SystemExit("provide --evidence-notes or --evidence-notes-file, not both")
    if evidence_notes is None and inline_notes is None:
        raise SystemExit("provide --evidence-notes or --evidence-notes-file")
    payload = validate_online_reflexion_evidence_notes(evidence_notes or inline_notes or {})
    _print_json_payload(payload, json_output=args.json)
    return 0 if payload.get("status") == "pass" else 1


def _materialize_hosted_gelo(args: argparse.Namespace) -> int:
    config = _gelo_materialized_config(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(config, indent=2, sort_keys=True))
    else:
        print(f"wrote {out}")
    return 0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _slice_data(slice_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = slice_payload.get("data")
    return data if isinstance(data, Mapping) else slice_payload


def _text_field(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _json_line(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True, default=str))


def _print_gepa_watch_snapshot(*, record: Any, json_output: bool) -> None:
    if json_output:
        _json_line({"type": "snapshot", "run": dict(record.raw)})
        return
    created = _text_field(getattr(record, "created_at", None))
    updated = _text_field(getattr(record, "updated_at", None))
    finalize = _text_field(getattr(record, "finalize_state", None))
    print(
        f"run_id={record.run_id} status={record.status.value} "
        f"finalize_state={finalize} created_at={created} updated_at={updated}"
    )
    if record.error:
        print(f"error: {record.error}")


def _gepa_watch(args: argparse.Namespace) -> int:
    try:
        client = _hosted_client(args)
        record = client.get_run(args.run_id)
        _print_gepa_watch_snapshot(record=record, json_output=args.json)
        if args.once:
            return 0
        if args.algorithm_events:
            for event in client.algorithm_event_stream(
                args.run_id,
                after_seq=args.after_seq,
                limit=args.limit,
            ):
                if args.json:
                    _json_line({"type": "algorithm_event", "event": dict(event)})
                else:
                    event_type = _text_field(event.get("type"), "optimizer.algorithm.event")
                    seq = _text_field(event.get("sequence_number"))
                    item = _as_mapping(event.get("item"))
                    item_type = _text_field(item.get("type"))
                    item_id = _text_field(item.get("id"))
                    print(f"algorithm_event seq={seq} type={event_type} item={item_type}:{item_id}")
                if event.get("type") in {
                    "optimizer.run.completed",
                    "optimizer.run.failed",
                    "optimizer.run.cancelled",
                }:
                    break
            record = client.get_run(args.run_id)
            return 1 if record.status.value == "failed" else 0
        if args.events:
            for event in client.events(args.run_id):
                if args.json:
                    _json_line({"type": "event", "event": dict(event)})
                else:
                    event_type = _text_field(event.get("event_type"), "event")
                    status = _text_field(event.get("status"))
                    seq = _text_field(event.get("seq") or event.get("_seq"))
                    print(f"event seq={seq} event_type={event_type} status={status}")
                if str(event.get("status") or "") in {"succeeded", "failed", "cancelled"}:
                    break
            record = client.get_run(args.run_id)
            if args.json:
                _json_line({"type": "final", "run": dict(record.raw)})
            else:
                print(f"final status={record.status.value}")
                if record.error:
                    print(f"error: {record.error}")
            return 1 if record.status.value == "failed" else 0
        while record.status.value not in {"succeeded", "failed", "cancelled"}:
            time.sleep(max(0.1, args.poll_seconds))
            record = client.get_run(args.run_id)
            _print_gepa_watch_snapshot(record=record, json_output=args.json)
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    return 1 if record.status.value == "failed" else 0


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":"))


def _print_gelo_watch_snapshot(
    *,
    record: Any,
    state: Mapping[str, Any],
    slice_name: str | None,
    slice_payload: Mapping[str, Any] | None,
    json_output: bool,
) -> None:
    if json_output:
        payload: dict[str, Any] = {
            "type": "snapshot",
            "run": dict(record.raw),
            "state": dict(state),
        }
        if slice_name is not None and slice_payload is not None:
            payload["slice"] = {
                "name": slice_name,
                "payload": dict(slice_payload),
            }
        _json_line(payload)
        return

    phase = _text_field(state.get("phase"))
    tick = _text_field(state.get("tick_index"))
    event_seq = _text_field(state.get("event_seq_high_water"))
    finalize = _text_field(getattr(record, "finalize_state", None))
    print(
        f"run_id={record.run_id} status={record.status.value} "
        f"phase={phase} tick={tick} event_seq={event_seq} finalize_state={finalize}"
    )
    if record.error:
        print(f"error: {record.error}")
    if slice_name is not None and slice_payload is not None:
        _print_gelo_slice(slice_name, slice_payload)


def _print_gelo_slice(slice_name: str, slice_payload: Mapping[str, Any]) -> None:
    data = _slice_data(slice_payload)
    if slice_name == "board":
        summary = _as_mapping(data.get("summary"))
        print(
            "board "
            f"status={_text_field(summary.get('status'))} "
            f"phase={_text_field(summary.get('phase'))} "
            f"tick={_text_field(summary.get('tick_index'))} "
            f"promotions={_text_field(summary.get('promotion_count'))}"
        )
        themes = data.get("themes")
        if isinstance(themes, list):
            print(f"themes ({len(themes)})")
            for raw_theme in themes:
                theme = _as_mapping(raw_theme)
                print(
                    "  "
                    f"{_text_field(theme.get('theme_id'))} "
                    f"status={_text_field(theme.get('status'))} "
                    f"saturated={_text_field(theme.get('saturated'))} "
                    f"score={_text_field(theme.get('objective_score'))} "
                    f"candidates={_text_field(theme.get('candidate_count'))} "
                    f"name={_text_field(theme.get('name'))}"
                )
            return
    print(f"{slice_name} {_compact_json(data)}")


def _gelo_watch(args: argparse.Namespace) -> int:
    try:
        client = _hosted_client(args)
        record = client.get_run(args.run_id)
        state = client.get_state(args.run_id)
        slice_payload = client.get_state_slice(args.run_id, args.slice) if args.slice else None
        _print_gelo_watch_snapshot(
            record=record,
            state=state,
            slice_name=args.slice,
            slice_payload=slice_payload,
            json_output=args.json,
        )
        if args.once:
            return 0
        if args.algorithm_events:
            for event in client.algorithm_event_stream(
                args.run_id,
                after_seq=args.after_seq,
                limit=args.limit,
            ):
                if args.json:
                    _json_line({"type": "algorithm_event", "event": dict(event)})
                else:
                    event_type = _text_field(event.get("type"), "optimizer.algorithm.event")
                    seq = _text_field(event.get("sequence_number"))
                    item = _as_mapping(event.get("item"))
                    item_type = _text_field(item.get("type"))
                    item_id = _text_field(item.get("id"))
                    print(f"algorithm_event seq={seq} type={event_type} item={item_type}:{item_id}")
                if event.get("type") in {
                    "optimizer.run.completed",
                    "optimizer.run.failed",
                    "optimizer.run.cancelled",
                }:
                    break
            final_record = client.get_run(args.run_id)
            if final_record.status.value == "failed":
                return 1
            return 0
        if args.goex_events:
            for event in client.goex_event_stream(
                args.run_id,
                after_seq=args.after_seq,
                limit=args.limit,
            ):
                if args.json:
                    _json_line({"type": "goex_event", "event": dict(event)})
                else:
                    event_type = _text_field(event.get("event_type"), "goex.event")
                    seq = _text_field(event.get("_seq"))
                    phase = _text_field(event.get("phase"))
                    status = _text_field(event.get("status"))
                    print(
                        f"goex_event seq={seq} "
                        f"event_type={event_type} "
                        f"phase={phase} "
                        f"status={status}"
                    )
                if event.get("event_type") == "optimizer.events_unavailable":
                    payload = _as_mapping(event.get("payload"))
                    error = _text_field(payload.get("error"), "event stream unavailable")
                    raise HostedOptimizerError(error)
                if event.get("event_type") in {"goex.run_finished", "goex.run_failed"}:
                    break
            final_record = client.get_run(args.run_id)
            if final_record.status.value == "failed":
                return 1
            return 0
        while record.status.value not in {"succeeded", "failed", "cancelled"}:
            time.sleep(max(0.1, args.poll_seconds))
            record = client.get_run(args.run_id)
            state = client.get_state(args.run_id)
            slice_payload = client.get_state_slice(args.run_id, args.slice) if args.slice else None
            _print_gelo_watch_snapshot(
                record=record,
                state=state,
                slice_name=args.slice,
                slice_payload=slice_payload,
                json_output=args.json,
            )
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    return 1 if record.status.value == "failed" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-optimizers")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    gepa = subcommands.add_parser("gepa")
    gepa_subcommands = gepa.add_subparsers(dest="gepa_command", required=True)
    gepa_run = gepa_subcommands.add_parser("run")
    gepa_run.add_argument("--config", required=True)
    gepa_run.add_argument(
        "--proposer-execution-mode",
        choices=("local_process", "stdio", "websocket", "ws"),
        help=(
            "Override [proposer].execution_mode for Codex app-server runs. "
            "local_process/stdio use stdin/stdout JSON-RPC; websocket/ws use "
            "Codex app-server's experimental local WebSocket listener."
        ),
    )
    gepa_run.add_argument(
        "--proposer-model",
        help="Override [proposer].model for this run.",
    )
    gepa_run.add_argument(
        "--proposer-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        help="Override [proposer].reasoning_effort for this run.",
    )
    gepa_run.add_argument(
        "--proposer-service-tier",
        choices=("default", "fast"),
        help=(
            "Override the Codex app-server service tier for this run. "
            "fast uses Codex Fast mode and requires ChatGPT auth."
        ),
    )
    gepa_run.add_argument(
        "--proposer-auth-mode",
        choices=("auto", "api_key", "chatgpt", "host"),
        help="Override [proposer].auth_mode for this run.",
    )
    gepa_run.add_argument(
        "--proposer-codex-home",
        help="Override [proposer].codex_home for ChatGPT-authenticated Codex runs.",
    )
    gepa_run.add_argument(
        "--json",
        action="store_true",
        help="Print the full result JSON instead of the terminal progress view.",
    )
    gepa_run.add_argument(
        "--disable-usage-registration",
        action="store_true",
        help="Do not send the best-effort package usage registration event.",
    )
    gepa_submit = gepa_subcommands.add_parser("submit")
    gepa_submit.add_argument("--config", required=True)
    gepa_submit.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    gepa_submit.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    gepa_submit.add_argument("--run-id")
    gepa_submit.add_argument("--idempotency-key")
    gepa_submit.add_argument("--project-id")
    gepa_submit.add_argument("--tunnel-url")
    gepa_submit.add_argument(
        "--tunnel-provider",
        choices=[provider.value for provider in TunnelProvider],
        default=TunnelProvider.SYNTH_TUNNEL.value,
        help="Tunnel provider for --tunnel-url. Defaults to synth_tunnel.",
    )
    gepa_submit.add_argument(
        "--tunnel-ttl-seconds",
        type=int,
        default=86400,
        help="Lease TTL for --tunnel-url, in seconds. Defaults to 86400.",
    )
    gepa_submit.add_argument("--container-pool")
    gepa_submit.add_argument("--container-task-id")
    gepa_submit.add_argument("--timeout-seconds", type=float, default=120.0)
    gepa_submit.add_argument(
        "--disable-usage-registration",
        action="store_true",
        help="Do not send the best-effort package usage registration event.",
    )
    gepa_submit.add_argument("--follow", action="store_true")
    gepa_submit.add_argument("--json", action="store_true")

    gepa_watch = gepa_subcommands.add_parser("watch")
    gepa_watch.add_argument("run_id")
    gepa_watch.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    gepa_watch.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    gepa_watch.add_argument("--timeout-seconds", type=float, default=120.0)
    gepa_watch.add_argument(
        "--events",
        action="store_true",
        help="Tail lifecycle SSE events after the initial run snapshot.",
    )
    gepa_watch.add_argument(
        "--algorithm-events",
        action="store_true",
        help="Tail normalized optimizer algorithm events after the initial run snapshot.",
    )
    gepa_watch.add_argument("--after-seq", type=int, default=0)
    gepa_watch.add_argument("--limit", type=int, default=500)
    gepa_watch.add_argument("--poll-seconds", type=float, default=2.0)
    gepa_watch.add_argument("--once", action="store_true")
    gepa_watch.add_argument("--json", action="store_true")

    gelo = subcommands.add_parser("gelo")
    gelo_subcommands = gelo.add_subparsers(dest="gelo_command", required=True)
    gelo_startup = gelo_subcommands.add_parser("startup")
    gelo_startup.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    gelo_startup.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    gelo_startup.add_argument("--timeout-seconds", type=float, default=120.0)
    gelo_startup.add_argument("--json", action="store_true")

    gelo_watch = gelo_subcommands.add_parser("watch")
    gelo_watch.add_argument("run_id")
    gelo_watch.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    gelo_watch.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    gelo_watch.add_argument("--timeout-seconds", type=float, default=120.0)
    gelo_watch.add_argument(
        "--slice",
        choices=("agents", "board", "candidates", "data-engine", "frontier", "themes"),
        help="Fetch and print a GELO state slice with each watch snapshot.",
    )
    gelo_watch.add_argument(
        "--goex-events",
        action="store_true",
        help="Tail the GELO event SSE stream after the initial state snapshot.",
    )
    gelo_watch.add_argument(
        "--algorithm-events",
        action="store_true",
        help="Tail normalized optimizer algorithm events after the initial state snapshot.",
    )
    gelo_watch.add_argument("--after-seq", type=int, default=0)
    gelo_watch.add_argument("--limit", type=int, default=500)
    gelo_watch.add_argument("--poll-seconds", type=float, default=2.0)
    gelo_watch.add_argument("--once", action="store_true")
    gelo_watch.add_argument("--json", action="store_true")

    gelo_materialize = gelo_subcommands.add_parser("materialize")
    materialize_source = gelo_materialize.add_mutually_exclusive_group(required=True)
    materialize_source.add_argument("--preset", choices=[name.value for name in GeloPresetName])
    materialize_source.add_argument("--toml", help="Structured public GELO TOML or JSON config.")
    gelo_materialize.add_argument("--overlay", help="Structured TOML/JSON overlay.")
    gelo_materialize.add_argument("--container-url")
    gelo_materialize.add_argument("--container-pool")
    gelo_materialize.add_argument("--container-task-id")
    gelo_materialize.add_argument("--run-id")
    gelo_materialize.add_argument("--proposer-rounds", type=int)
    gelo_materialize.add_argument("--train-seed-count", type=int)
    gelo_materialize.add_argument("--heldout-seed-count", type=int)
    gelo_materialize.add_argument("--max-rollouts", type=int)
    gelo_materialize.add_argument("--policy-model")
    _add_gelo_jesterky_workflow_args(gelo_materialize)
    gelo_materialize.add_argument("-o", "--out", required=True)
    gelo_materialize.add_argument("--json", action="store_true")

    gelo_submit = gelo_subcommands.add_parser("submit")
    submit_source = gelo_submit.add_mutually_exclusive_group(required=True)
    submit_source.add_argument("--config", help="Path to hosted GELO config JSON.")
    submit_source.add_argument("--preset", choices=[name.value for name in GeloPresetName])
    submit_source.add_argument("--toml", help="Structured public GELO TOML or JSON config.")
    gelo_submit.add_argument("--overlay", help="Structured TOML/JSON overlay for --toml.")
    gelo_submit.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    gelo_submit.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    gelo_submit.add_argument("--run-id")
    gelo_submit.add_argument("--idempotency-key")
    gelo_submit.add_argument("--project-id")
    gelo_submit.add_argument("--container-url")
    gelo_submit.add_argument("--tunnel-url")
    gelo_submit.add_argument(
        "--tunnel-provider",
        choices=[provider.value for provider in TunnelProvider],
        default=TunnelProvider.SYNTH_TUNNEL.value,
        help="Tunnel provider for --tunnel-url. Defaults to synth_tunnel.",
    )
    gelo_submit.add_argument(
        "--tunnel-ttl-seconds",
        type=int,
        default=86400,
        help="Lease TTL for --tunnel-url, in seconds. Defaults to 86400.",
    )
    gelo_submit.add_argument("--container-pool")
    gelo_submit.add_argument("--container-task-id")
    gelo_submit.add_argument("--proposer-rounds", type=int)
    gelo_submit.add_argument("--train-seed-count", type=int)
    gelo_submit.add_argument("--heldout-seed-count", type=int)
    gelo_submit.add_argument("--max-rollouts", type=int)
    gelo_submit.add_argument("--policy-model")
    _add_gelo_jesterky_workflow_args(gelo_submit)
    gelo_submit.add_argument(
        "--billing-mode",
        choices=("promo", "paid"),
        default=None,
        help=(
            "GELO billing mode. Defaults to backend promo behavior; use paid to bypass "
            "launch-promo gates."
        ),
    )
    gelo_submit.add_argument("--timeout-seconds", type=float, default=120.0)
    gelo_submit.add_argument(
        "--disable-usage-registration",
        action="store_true",
        help="Do not send the best-effort package usage registration event.",
    )
    gelo_submit.add_argument("--follow", action="store_true")
    gelo_submit.add_argument("--json", action="store_true")

    gelo_console = gelo_subcommands.add_parser("console")
    gelo_console.add_argument("--title", default="GELO")
    gelo_console.add_argument("--host", default="127.0.0.1")
    gelo_console.add_argument("--port", type=int, default=8767)
    gelo_console.add_argument(
        "--docs",
        help="Override the docs directory (defaults to the bundled GELO docs).",
    )
    gelo_console.add_argument(
        "--docs-set", default="gelo", help="Bundled docs set to serve (default: gelo)."
    )

    sft = subcommands.add_parser("sft", help="Operate the public SFT control-plane service.")
    sft_subcommands = sft.add_subparsers(dest="sft_command", required=True)
    sft_validate = sft_subcommands.add_parser("validate")
    sft_validate.add_argument("--config", required=True)
    sft_validate.add_argument("--run-id")
    sft_validate.add_argument("--json", action="store_true")

    sft_service = sft_subcommands.add_parser("service")
    sft_service.add_argument("--db", default=".sft/service.sqlite")
    sft_service.add_argument("--bind", default="127.0.0.1:8878")
    sft_service.add_argument(
        "--service-token-env",
        default="SYNTH_OPTIMIZERS_SFT_SERVICE_TOKEN",
        help="Optional inbound bearer-token environment variable.",
    )

    for command_name in ("submit", "watch", "cancel"):
        command = sft_subcommands.add_parser(command_name)
        command.add_argument(
            "--service-url",
            default=os.environ.get("SYNTH_OPTIMIZERS_SFT_SERVICE_URL", "http://127.0.0.1:8878"),
        )
        command.add_argument("--service-token-env", default="SYNTH_OPTIMIZERS_SFT_SERVICE_TOKEN")
        command.add_argument("--timeout-seconds", type=float, default=300.0)
        command.add_argument("--json", action="store_true")
    sft_submit = sft_subcommands.choices["submit"]
    sft_submit.add_argument("--config", required=True)
    sft_submit.add_argument("--run-id")
    sft_submit.add_argument("--idempotency-key")
    sft_submit.add_argument("--follow", action="store_true")
    sft_submit.add_argument("--poll-seconds", type=float, default=1.0)
    sft_watch = sft_subcommands.choices["watch"]
    sft_watch.add_argument("run_id")
    sft_watch.add_argument("--events", action="store_true")
    sft_watch.add_argument("--after-seq", type=int, default=0)
    sft_watch.add_argument("--limit", type=int, default=500)
    sft_cancel = sft_subcommands.choices["cancel"]
    sft_cancel.add_argument("run_id")

    mapo = subcommands.add_parser("mapo")
    mapo_subcommands = mapo.add_subparsers(dest="mapo_command", required=True)
    mapo_startup = mapo_subcommands.add_parser("startup")
    mapo_startup.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    mapo_startup.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    mapo_startup.add_argument("--timeout-seconds", type=float, default=120.0)
    mapo_startup.add_argument("--json", action="store_true")

    mapo_submit = mapo_subcommands.add_parser("submit")
    mapo_submit.add_argument("--config", required=True, help="Path to MAPO TOML or JSON config.")
    mapo_submit.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    mapo_submit.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    mapo_submit.add_argument("--run-id")
    mapo_submit.add_argument("--idempotency-key")
    mapo_submit.add_argument("--project-id")
    mapo_submit.add_argument("--container-pool")
    mapo_submit.add_argument("--container-task-id")
    mapo_submit.add_argument("--tunnel-url")
    mapo_submit.add_argument(
        "--tunnel-provider",
        choices=[provider.value for provider in TunnelProvider],
        default=TunnelProvider.SYNTH_TUNNEL.value,
        help="Tunnel provider for --tunnel-url. Defaults to synth_tunnel.",
    )
    mapo_submit.add_argument(
        "--tunnel-ttl-seconds",
        type=int,
        default=86400,
        help="Lease TTL for --tunnel-url, in seconds. Defaults to 86400.",
    )
    mapo_submit.add_argument(
        "--billing-mode",
        choices=("promo", "paid"),
        default=None,
        help="MAPO billing mode. Defaults to backend behavior.",
    )
    mapo_submit.add_argument("--timeout-seconds", type=float, default=120.0)
    mapo_submit.add_argument(
        "--disable-usage-registration",
        action="store_true",
        help="Do not send the best-effort package usage registration event.",
    )
    mapo_submit.add_argument("--follow", action="store_true")
    mapo_submit.add_argument("--json", action="store_true")

    mapo_watch = mapo_subcommands.add_parser("watch")
    mapo_watch.add_argument("run_id")
    mapo_watch.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    mapo_watch.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    mapo_watch.add_argument("--timeout-seconds", type=float, default=120.0)
    mapo_watch.add_argument(
        "--events",
        action="store_true",
        help="Tail lifecycle SSE events after the initial run snapshot.",
    )
    mapo_watch.add_argument(
        "--algorithm-events",
        action="store_true",
        help="Tail normalized optimizer algorithm events after the initial run snapshot.",
    )
    mapo_watch.add_argument("--after-seq", type=int, default=0)
    mapo_watch.add_argument("--limit", type=int, default=500)
    mapo_watch.add_argument("--poll-seconds", type=float, default=2.0)
    mapo_watch.add_argument("--once", action="store_true")
    mapo_watch.add_argument("--json", action="store_true")

    reflexion = subcommands.add_parser("reflexion")
    reflexion_subcommands = reflexion.add_subparsers(dest="reflexion_command", required=True)
    reflexion_startup = reflexion_subcommands.add_parser("startup")
    reflexion_startup.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    reflexion_startup.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    reflexion_startup.add_argument("--timeout-seconds", type=float, default=120.0)
    reflexion_startup.add_argument(
        "--require-online-reflexion",
        action="store_true",
        help="Exit non-zero unless online-reflexion is advertised as submit-supported.",
    )
    reflexion_startup.add_argument(
        "--require-online-reflexion-release-metadata",
        action="store_true",
        help=(
            "Exit non-zero unless the startup catalog advertises the Online "
            "Reflexion release evidence schema and release_blog_growth gate."
        ),
    )
    reflexion_startup.add_argument("--json", action="store_true")

    reflexion_submit = reflexion_subcommands.add_parser("submit")
    reflexion_submit.add_argument(
        "--config",
        required=True,
        help="Path to hosted online Reflexion TOML or JSON config.",
    )
    reflexion_submit.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    reflexion_submit.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    reflexion_submit.add_argument("--run-id")
    reflexion_submit.add_argument("--idempotency-key")
    reflexion_submit.add_argument("--project-id")
    reflexion_submit.add_argument("--container-pool")
    reflexion_submit.add_argument("--container-task-id")
    reflexion_submit.add_argument("--timeout-seconds", type=float, default=120.0)
    reflexion_submit.add_argument(
        "--disable-usage-registration",
        action="store_true",
        help="Do not send the best-effort package usage registration event.",
    )
    reflexion_submit.add_argument("--follow", action="store_true")
    reflexion_submit.add_argument("--json", action="store_true")

    reflexion_watch = reflexion_subcommands.add_parser("watch")
    reflexion_watch.add_argument("run_id")
    reflexion_watch.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    reflexion_watch.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    reflexion_watch.add_argument("--timeout-seconds", type=float, default=120.0)
    reflexion_watch.add_argument(
        "--events",
        action="store_true",
        help="Tail lifecycle SSE events after the initial run snapshot.",
    )
    reflexion_watch.add_argument(
        "--algorithm-events",
        action="store_true",
        help="Tail normalized optimizer algorithm events after the initial run snapshot.",
    )
    reflexion_watch.add_argument("--after-seq", type=int, default=0)
    reflexion_watch.add_argument("--limit", type=int, default=500)
    reflexion_watch.add_argument("--poll-seconds", type=float, default=2.0)
    reflexion_watch.add_argument("--once", action="store_true")
    reflexion_watch.add_argument("--json", action="store_true")

    reflexion_receipt = reflexion_subcommands.add_parser("receipt")
    reflexion_receipt.add_argument("run_id")
    reflexion_receipt.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    reflexion_receipt.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    reflexion_receipt.add_argument("--timeout-seconds", type=float, default=120.0)
    reflexion_receipt.add_argument("--exposure-limit", type=int, default=500)
    reflexion_receipt.add_argument("--outcome-limit", type=int, default=500)
    reflexion_receipt.add_argument("--json", action="store_true")

    reflexion_audit = reflexion_subcommands.add_parser("audit")
    audit_selection = reflexion_audit.add_mutually_exclusive_group()
    audit_selection.add_argument("--audit-run-id")
    audit_selection.add_argument("--run-id", action="append")
    audit_selection.add_argument("--run-ids", action="append")
    reflexion_audit.add_argument("--layer-id")
    reflexion_audit.add_argument("--project-id")
    reflexion_audit.add_argument("--strict", action="store_true")
    reflexion_audit.add_argument("--limit", type=int, default=50)
    reflexion_audit.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    reflexion_audit.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    reflexion_audit.add_argument("--timeout-seconds", type=float, default=120.0)
    reflexion_audit.add_argument("--json", action="store_true")

    reflexion_packet = reflexion_subcommands.add_parser("evidence-packet")
    reflexion_packet.add_argument("--run-id", action="append")
    reflexion_packet.add_argument("--run-ids", action="append")
    reflexion_packet.add_argument("--layer-id")
    reflexion_packet.add_argument("--project-id")
    reflexion_packet.add_argument(
        "--evidence-notes",
        help=(
            "Structured JSON object keyed by required evidence lane plus "
            "release_blog_growth. Each entry must carry specific proof; bare "
            "true/status values do not complete release readiness."
        ),
    )
    reflexion_packet.add_argument(
        "--evidence-notes-file",
        help=(
            "Structured JSON file keyed by required evidence lane plus "
            "release_blog_growth. "
            "Use dev_examples/online_reflexion/evidence_notes_template.json "
            "as the fill-in shape."
        ),
    )
    reflexion_packet.add_argument("--blog-decision-owner", default="Josh")
    reflexion_packet.add_argument("--blog-approved-by-owner", action="store_true")
    reflexion_packet.add_argument("--no-receipt-summaries", action="store_true")
    reflexion_packet.add_argument("--limit", type=int, default=50)
    reflexion_packet.add_argument(
        "--base-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
        help="Synth API base URL. Defaults to SYNTH_BACKEND_URL or https://api.usesynth.ai.",
    )
    reflexion_packet.add_argument(
        "--api-key-env",
        default="SYNTH_API_KEY",
        help="Environment variable containing the Synth API key.",
    )
    reflexion_packet.add_argument("--timeout-seconds", type=float, default=120.0)
    reflexion_packet.add_argument(
        "--out",
        help="Optional path to write the assembled evidence packet JSON.",
    )
    reflexion_packet.add_argument("--json", action="store_true")

    reflexion_validate_notes = reflexion_subcommands.add_parser("validate-evidence-notes")
    reflexion_validate_notes.add_argument(
        "--evidence-notes",
        help=("Structured JSON object keyed by required evidence lane plus release_blog_growth."),
    )
    reflexion_validate_notes.add_argument(
        "--evidence-notes-file",
        help=(
            "Structured JSON file keyed by required evidence lane plus "
            "release_blog_growth. "
            "Use dev_examples/online_reflexion/evidence_notes_template.json "
            "as the fill-in shape."
        ),
    )
    reflexion_validate_notes.add_argument("--json", action="store_true")

    # The standing HTTP service is the public worker/workspace surface: queueing,
    # claiming, and lifecycle control happen over the /runs and /workspace routes.
    gepa_service = gepa_subcommands.add_parser("service")
    gepa_service.add_argument("--db", required=True)
    gepa_service.add_argument("--bind", default="127.0.0.1:8879")
    gepa_service.add_argument("--worker-id")
    gepa_service.add_argument("--lease-seconds", type=int, default=3600)
    gepa_service.add_argument("--workers", type=int, default=10)

    # The board is a local, read-only HTML projection of GEPA_HOME discovery,
    # explicit registry roots, and any live services it finds.
    gepa_board = gepa_subcommands.add_parser("board")
    gepa_board.add_argument(
        "roots",
        nargs="*",
        help="Additional registry roots to include alongside GEPA_HOME.",
    )
    gepa_board.add_argument(
        "--root",
        dest="extra_roots",
        action="append",
        default=[],
        help="Additional registry root to include; may be repeated.",
    )
    gepa_board.add_argument("--out", default="gepa_board.html", help="Static HTML output path.")
    gepa_board.add_argument("--title", default="GEPA Run Board")
    gepa_board.add_argument("--open", action="store_true", help="Open the board after start.")
    gepa_board.add_argument(
        "--serve",
        action="store_true",
        help="Serve a live board over SSE instead of writing a static file.",
    )
    gepa_board.add_argument("--host", default="127.0.0.1")
    gepa_board.add_argument("--port", type=int, default=8765)
    gepa_board.add_argument(
        "--service-url",
        help="Pin the board to one running `gepa service` (e.g. http://127.0.0.1:8899). "
        "When omitted, the board discovers services from GEPA_HOME.",
    )
    gepa_board.add_argument(
        "--interval", type=float, default=2.0, help="Live re-projection cadence (seconds)."
    )
    gepa_board.add_argument(
        "--stale-after",
        type=float,
        default=1200.0,
        help=argparse.SUPPRESS,
    )
    gepa_board.add_argument(
        "--json",
        action="store_true",
        help="Print the normalized board JSON to stdout instead of writing HTML.",
    )

    # The console serves the run board and the bundled GEPA docs behind one port
    # as two tabs (Dashboard + Docs).
    gepa_console = gepa_subcommands.add_parser("console")
    gepa_console.add_argument(
        "roots",
        nargs="*",
        help="Additional registry roots for the board, alongside GEPA_HOME.",
    )
    gepa_console.add_argument(
        "--root",
        dest="extra_roots",
        action="append",
        default=[],
        help="Additional registry root for the board; may be repeated.",
    )
    gepa_console.add_argument("--title", default="GEPA")
    gepa_console.add_argument("--host", default="127.0.0.1")
    gepa_console.add_argument("--port", type=int, default=8766)
    gepa_console.add_argument(
        "--service-url",
        help="Pin the board to one running `gepa service`; otherwise discover from GEPA_HOME.",
    )
    gepa_console.add_argument(
        "--interval", type=float, default=2.0, help="Live re-projection cadence (seconds)."
    )
    gepa_console.add_argument(
        "--docs",
        help="Override the docs directory (defaults to the bundled GEPA docs).",
    )
    gepa_console.add_argument(
        "--docs-set", default="gepa", help="Bundled docs set to serve (default: gepa)."
    )

    gepa_eval_stats = gepa_subcommands.add_parser("eval-stats")
    gepa_eval_stats.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or roots containing transitions.sqlite files.",
    )
    gepa_eval_stats.add_argument(
        "--no-write-json",
        action="store_true",
        help="Do not write per-run stats.json next to transitions.sqlite.",
    )
    gepa_eval_stats.add_argument(
        "--json",
        action="store_true",
        help="Print stats JSON instead of the table.",
    )

    gepa_runs = gepa_subcommands.add_parser("runs")
    gepa_runs_subcommands = gepa_runs.add_subparsers(dest="runs_command", required=True)

    gepa_runs_list = gepa_runs_subcommands.add_parser("list")
    gepa_runs_list.add_argument("runs", nargs="*", help="Run directories or run IDs.")
    gepa_runs_list.add_argument(
        "--root",
        action="append",
        default=[],
        help="Runs root used for run IDs and bulk scans; may be repeated.",
    )
    gepa_runs_list.add_argument(
        "--older-than",
        help="Only include bulk runs older than this duration, e.g. 7d, 12h, 30m.",
    )
    gepa_runs_list.add_argument(
        "--status",
        action="append",
        default=[],
        help="Terminal status to include for bulk scans; may be repeated.",
    )
    gepa_runs_list.add_argument("--json", action="store_true")

    gepa_runs_show = gepa_runs_subcommands.add_parser("show")
    gepa_runs_show.add_argument("runs", nargs="+", help="Run directories or run IDs.")
    gepa_runs_show.add_argument(
        "--root",
        action="append",
        default=[],
        help="Runs root used for run IDs; may be repeated.",
    )
    gepa_runs_show.add_argument("--json", action="store_true")

    gepa_runs_du = gepa_runs_subcommands.add_parser("du")
    gepa_runs_du.add_argument("runs", nargs="+", help="Run directories or run IDs.")
    gepa_runs_du.add_argument(
        "--root",
        action="append",
        default=[],
        help="Runs root used for run IDs; may be repeated.",
    )
    gepa_runs_du.add_argument("--json", action="store_true")

    gepa_runs_doctor = gepa_runs_subcommands.add_parser("doctor")
    gepa_runs_doctor.add_argument("runs", nargs="+", help="Run directories or run IDs.")
    gepa_runs_doctor.add_argument(
        "--root",
        action="append",
        default=[],
        help="Runs root used for run IDs; may be repeated.",
    )
    gepa_runs_doctor.add_argument("--json", action="store_true")

    gepa_runs_health = gepa_runs_subcommands.add_parser("health")
    gepa_runs_health.add_argument(
        "--root",
        action="append",
        required=True,
        help="Runs root to inspect; may be repeated.",
    )
    gepa_runs_health.add_argument("--run-warn-bytes", help="Per-run warning threshold, e.g. 5G.")
    gepa_runs_health.add_argument("--root-warn-bytes", help="Per-root warning threshold, e.g. 20G.")
    gepa_runs_health.add_argument(
        "--stale-partial-warn-bytes",
        help="Stale partial warning threshold, e.g. 2G.",
    )
    gepa_runs_health.add_argument(
        "--partial-stale-after",
        help="Partial artifact age before stale classification, e.g. 2h.",
    )
    gepa_runs_health.add_argument("--json", action="store_true")

    gepa_runs_compact = gepa_runs_subcommands.add_parser("compact")
    gepa_runs_compact.add_argument("runs", nargs="*", help="Run directories or run IDs.")
    gepa_runs_compact.add_argument(
        "--root",
        action="append",
        default=[],
        help="Runs root used for run IDs and bulk scans; may be repeated.",
    )
    gepa_runs_compact.add_argument(
        "--all-terminal",
        action="store_true",
        help="Compact all terminal-looking runs under --root.",
    )
    gepa_runs_compact.add_argument(
        "--older-than",
        help="Only include bulk runs older than this duration, e.g. 7d, 12h, 30m.",
    )
    gepa_runs_compact.add_argument(
        "--status",
        action="append",
        default=[],
        help="Terminal status to include for bulk scans; may be repeated.",
    )
    gepa_runs_compact.add_argument(
        "--profile",
        choices=("debug", "compact", "minimal"),
        default="compact",
    )
    gepa_runs_compact.add_argument("--yes", action="store_true", help="Apply the compaction.")
    gepa_runs_compact.add_argument("--json", action="store_true")

    gepa_runs_delete = gepa_runs_subcommands.add_parser("delete")
    gepa_runs_delete.add_argument("runs", nargs="*", help="Run directories or run IDs.")
    gepa_runs_delete.add_argument(
        "--root",
        action="append",
        default=[],
        help="Runs root used for run IDs and bulk scans; may be repeated.",
    )
    gepa_runs_delete.add_argument(
        "--all-terminal",
        action="store_true",
        help="Delete all terminal-looking runs under --root.",
    )
    gepa_runs_delete.add_argument(
        "--older-than",
        help="Only include bulk runs older than this duration, e.g. 7d, 12h, 30m.",
    )
    gepa_runs_delete.add_argument(
        "--status",
        action="append",
        default=[],
        help="Terminal status to include for bulk scans; may be repeated.",
    )
    gepa_runs_delete.add_argument("--yes", action="store_true", help="Apply the deletion.")
    gepa_runs_delete.add_argument("--json", action="store_true")

    events = subcommands.add_parser("events")
    events_subcommands = events.add_subparsers(dest="events_command", required=True)
    replay = events_subcommands.add_parser("replay")
    replay.add_argument("--events", required=True)
    compare = events_subcommands.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gepa" and args.gepa_command == "run":
        from .gepa import GepaRun, UsageRegistrationConfig

        old_terminal = os.environ.get("SYNTH_OPTIMIZERS_TERMINAL")
        old_proposer_execution_mode = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE")
        old_proposer_model = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_MODEL")
        old_proposer_reasoning_effort = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT")
        old_proposer_service_tier = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_SERVICE_TIER")
        old_proposer_auth_mode = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_AUTH_MODE")
        old_proposer_codex_home = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_CODEX_HOME")
        if not args.json:
            os.environ["SYNTH_OPTIMIZERS_TERMINAL"] = "1"
        if args.proposer_execution_mode:
            os.environ["SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE"] = args.proposer_execution_mode
        if args.proposer_model:
            os.environ["SYNTH_OPTIMIZERS_PROPOSER_MODEL"] = args.proposer_model
        if args.proposer_reasoning_effort:
            os.environ["SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT"] = (
                args.proposer_reasoning_effort
            )
        if args.proposer_service_tier:
            os.environ["SYNTH_OPTIMIZERS_PROPOSER_SERVICE_TIER"] = args.proposer_service_tier
        if args.proposer_auth_mode:
            os.environ["SYNTH_OPTIMIZERS_PROPOSER_AUTH_MODE"] = args.proposer_auth_mode
        if args.proposer_codex_home:
            os.environ["SYNTH_OPTIMIZERS_PROPOSER_CODEX_HOME"] = args.proposer_codex_home
        try:
            gepa_run = GepaRun.from_toml(args.config)
            if args.disable_usage_registration:
                gepa_run.config.usage_registration = UsageRegistrationConfig(enabled=False)
            project_gepa_run_started(
                run_id=gepa_run.config.run.run_id,
                config_path=args.config,
                output_dir=gepa_run.config.run.output_dir,
            )
            result = gepa_run.execute()
            project_gepa_run_artifacts(result.manifest_path)
        except SynthOptimizerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            if args.json:
                pass
            elif old_terminal is None:
                os.environ.pop("SYNTH_OPTIMIZERS_TERMINAL", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_TERMINAL"] = old_terminal
            if old_proposer_execution_mode is None:
                os.environ.pop("SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE"] = old_proposer_execution_mode
            if old_proposer_model is None:
                os.environ.pop("SYNTH_OPTIMIZERS_PROPOSER_MODEL", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_MODEL"] = old_proposer_model
            if old_proposer_reasoning_effort is None:
                os.environ.pop("SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT"] = (
                    old_proposer_reasoning_effort
                )
            if old_proposer_service_tier is None:
                os.environ.pop("SYNTH_OPTIMIZERS_PROPOSER_SERVICE_TIER", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_SERVICE_TIER"] = old_proposer_service_tier
            if old_proposer_auth_mode is None:
                os.environ.pop("SYNTH_OPTIMIZERS_PROPOSER_AUTH_MODE", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_AUTH_MODE"] = old_proposer_auth_mode
            if old_proposer_codex_home is None:
                os.environ.pop("SYNTH_OPTIMIZERS_PROPOSER_CODEX_HOME", None)
            else:
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_CODEX_HOME"] = old_proposer_codex_home
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print()
            print(f"artifacts: {Path(result.manifest_path).parent}")
        return 0
    if args.command == "gepa" and args.gepa_command == "submit":
        return _submit_hosted_gepa(args)
    if args.command == "gepa" and args.gepa_command == "watch":
        return _gepa_watch(args)
    if args.command == "gelo" and args.gelo_command == "startup":
        return _gelo_startup(args)
    if args.command == "gelo" and args.gelo_command == "watch":
        return _gelo_watch(args)
    if args.command == "gelo" and args.gelo_command == "materialize":
        return _materialize_hosted_gelo(args)
    if args.command == "gelo" and args.gelo_command == "submit":
        return _submit_hosted_gelo(args)
    if args.command == "gelo" and args.gelo_command == "console":
        from .board_server import AggregateSource
        from .docs_server import DocsSource, bundled_docs_root, serve_console

        docs_root = Path(args.docs) if args.docs else bundled_docs_root(args.docs_set)
        board = AggregateSource([], title=f"{args.title} — hosted")
        docs = DocsSource([docs_root], title=args.title)
        serve_console(board, docs, host=args.host, port=args.port)
        return 0
    if args.command == "sft" and args.sft_command == "validate":
        return _sft_validate(args)
    if args.command == "sft" and args.sft_command == "submit":
        return _sft_submit(args)
    if args.command == "sft" and args.sft_command == "watch":
        return _sft_watch(args)
    if args.command == "sft" and args.sft_command == "cancel":
        return _sft_cancel(args)
    if args.command == "sft" and args.sft_command == "service":
        token = os.environ.get(args.service_token_env) if args.service_token_env else None
        serve_sft_service(args.db, args.bind, service_token=token)
        return 0
    if args.command == "mapo" and args.mapo_command == "startup":
        return _gelo_startup(args)
    if args.command == "mapo" and args.mapo_command == "submit":
        return _submit_hosted_mapo(args)
    if args.command == "mapo" and args.mapo_command == "watch":
        return _gepa_watch(args)
    if args.command == "reflexion" and args.reflexion_command == "startup":
        return _gelo_startup(args)
    if args.command == "reflexion" and args.reflexion_command == "submit":
        return _submit_hosted_reflexion(args)
    if args.command == "reflexion" and args.reflexion_command == "watch":
        return _gepa_watch(args)
    if args.command == "reflexion" and args.reflexion_command == "receipt":
        return _online_reflexion_receipt(args)
    if args.command == "reflexion" and args.reflexion_command == "audit":
        return _online_reflexion_audit(args)
    if args.command == "reflexion" and args.reflexion_command == "evidence-packet":
        return _online_reflexion_evidence_packet(args)
    if args.command == "reflexion" and args.reflexion_command == "validate-evidence-notes":
        return _online_reflexion_validate_evidence_notes(args)
    if args.command == "gepa" and args.gepa_command == "service":
        gepa_serve(args.db, args.bind, args.worker_id, args.lease_seconds, args.workers)
        return 0
    if args.command == "gepa" and args.gepa_command == "board":
        roots = [*args.roots, *args.extra_roots]
        if args.json:
            from .board_server import board_snapshot

            print(
                json.dumps(
                    board_snapshot(
                        roots,
                        title=args.title,
                        service_url=args.service_url,
                        live_within_seconds=args.stale_after,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.serve:
            from .board_server import serve_board

            if args.open:
                import threading
                import subprocess

                url = f"http://{args.host}:{args.port}/"
                threading.Timer(0.5, lambda: subprocess.run(["open", "-a", "Safari", url])).start()
            serve_board(
                roots,
                host=args.host,
                port=args.port,
                title=args.title,
                interval=args.interval,
                service_url=args.service_url,
                live_within_seconds=args.stale_after,
            )
            return 0
        from .board_server import board_snapshot
        from .o11y import render_board_html

        data = board_snapshot(
            roots,
            title=args.title,
            service_url=args.service_url,
            live_within_seconds=args.stale_after,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render_board_html(data, title=args.title, service_url=args.service_url),
            encoding="utf-8",
        )
        summary = data["summary"]
        print(
            f"wrote {out} — {summary['total']} run(s): "
            f"{summary['running']} running, "
            f"{summary['succeeded']} succeeded, "
            f"{summary['failed']} failed"
        )
        if args.open:
            import subprocess

            subprocess.run(["open", "-a", "Safari", out.resolve().as_uri()])
        return 0
    if args.command == "gepa" and args.gepa_command == "console":
        from .board_server import AggregateSource
        from .docs_server import DocsSource, bundled_docs_root, serve_console

        roots = [*args.roots, *args.extra_roots]
        docs_root = Path(args.docs) if args.docs else bundled_docs_root(args.docs_set)
        board = AggregateSource(
            roots,
            title=f"{args.title} — runs",
            service_url=args.service_url,
        )
        docs = DocsSource([docs_root], title=args.title)
        serve_console(board, docs, host=args.host, port=args.port, interval=args.interval)
        return 0
    if args.command == "gepa" and args.gepa_command == "eval-stats":
        from .eval_stats import eval_stats_for_roots, render_eval_stats_table

        stats = eval_stats_for_roots(args.runs, write_json=not args.no_write_json)
        if args.json:
            print(json.dumps([row.to_dict() for row in stats], indent=2, sort_keys=True))
        else:
            print(render_eval_stats_table(stats))
        return 0
    if args.command == "gepa" and args.gepa_command == "runs":
        if args.runs_command == "health":
            try:
                report = gepa_workspace_storage_health(
                    args.root,
                    run_warn_bytes=_bytes_value(args.run_warn_bytes),
                    root_warn_bytes=_bytes_value(args.root_warn_bytes),
                    stale_partial_warn_bytes=_bytes_value(args.stale_partial_warn_bytes),
                    partial_stale_after_seconds=(
                        int(_duration_seconds(args.partial_stale_after))
                        if args.partial_stale_after
                        else None
                    ),
                )
            except SynthOptimizerError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            _print_storage_health(report, args.json)
            return 0
        run_dirs = _resolve_gepa_run_dirs(
            args.runs,
            args.root,
            getattr(args, "all_terminal", False),
            getattr(args, "older_than", None),
            getattr(args, "status", []),
            all_root_children=args.runs_command == "list" and not args.runs,
        )
        if not run_dirs:
            print("error: no runs matched", file=sys.stderr)
            return 1
        missing = [str(run_dir) for run_dir in run_dirs if not run_dir.is_dir()]
        if missing:
            print(f"error: run dir not found: {missing[0]}", file=sys.stderr)
            return 1
        try:
            if args.runs_command in {"list", "show", "du", "doctor"}:
                reports = [
                    gepa_inspect_run_storage(str(run_dir), run_id=run_dir.name)
                    for run_dir in run_dirs
                ]
                if args.runs_command == "list":
                    _print_run_storage_list(reports, args.json)
                else:
                    for index, report in enumerate(reports):
                        if index and not args.json:
                            print()
                        _print_run_storage_detail(
                            report,
                            args.json,
                            doctor=args.runs_command == "doctor",
                        )
                return 0
            dry_run = not args.yes
            reports = []
            for run_dir in run_dirs:
                inspection = gepa_inspect_run_storage(str(run_dir), run_id=run_dir.name)
                if not inspection.get("terminal"):
                    print(
                        "error: cleanup requires a terminal run: "
                        f"{run_dir} status={inspection.get('terminal_status')}",
                        file=sys.stderr,
                    )
                    return 1
                if args.runs_command == "compact":
                    reports.append(
                        gepa_compact_run_storage(
                            str(run_dir),
                            run_id=run_dir.name,
                            profile=args.profile,
                            dry_run=dry_run,
                        )
                    )
                elif args.runs_command == "delete":
                    reports.append(gepa_delete_run_storage(str(run_dir), dry_run=dry_run))
        except SynthOptimizerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_storage_reports(reports, args.json)
        if dry_run and not args.json:
            print("pass --yes to apply")
        return 0
    if args.command == "events" and args.events_command == "replay":
        print(events_replay(args.events), end="")
        return 0
    if args.command == "events" and args.events_command == "compare":
        events_compare(args.left, args.right)
        print("normalized event feeds match")
        return 0
    raise SystemExit(f"unsupported command: {args}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
