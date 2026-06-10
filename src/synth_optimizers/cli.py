from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from ._synth_optimizers import (
    SynthOptimizerError,
    events_compare,
    events_replay,
    gepa_compact_run_storage,
    gepa_delete_run_storage,
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
)


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
    if all_terminal:
        if not root_paths:
            raise SystemExit("--all-terminal requires at least one --root")
        for root in root_paths:
            for child in sorted(root.iterdir() if root.exists() else []):
                if not child.is_dir():
                    continue
                status = _run_manifest_status(child)
                if status is None:
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


def _api_endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _hosted_client(args: argparse.Namespace) -> HostedOptimizerClient:
    return HostedOptimizerClient(
        backend_url=args.base_url,
        api_key=os.environ.get(args.api_key_env),
        timeout_seconds=args.timeout_seconds,
    )


def _container_pool_from_args(args: argparse.Namespace) -> ContainerPoolTarget | None:
    pool_id = getattr(args, "container_pool", None)
    task_id = getattr(args, "container_task_id", None)
    if task_id and not pool_id:
        raise SystemExit("--container-task-id requires --container-pool")
    if not pool_id:
        return None
    return ContainerPoolTarget(pool_id=pool_id, task_id=task_id)


def _validate_container_args(args: argparse.Namespace) -> None:
    has_container_url = bool(getattr(args, "container_url", None))
    has_container_pool = bool(getattr(args, "container_pool", None))
    has_tunnel_url = bool(getattr(args, "tunnel_url", None))
    if has_container_url and has_container_pool:
        raise SystemExit("--container-url and --container-pool are mutually exclusive")
    if has_tunnel_url and (has_container_url or has_container_pool):
        raise SystemExit("--tunnel-url cannot be combined with --container-url or --container-pool")


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


def _gelo_materialized_config(args: argparse.Namespace) -> dict[str, Any]:
    _validate_container_args(args)
    container_url = getattr(args, "container_url", None) or getattr(args, "tunnel_url", None)
    container_pool = _container_pool_from_args(args)
    if getattr(args, "preset", None):
        try:
            return GeloPreset.from_name(args.preset, **_gelo_preset_overrides(args)).materialize(
                container_url=container_url,
                container_pool=container_pool,
                run_id=getattr(args, "run_id", None),
            )
        except GeloMaterializeError as exc:
            raise SystemExit(str(exc)) from exc
    if getattr(args, "toml", None):
        try:
            return GeloMaterializer.from_paths(args.toml, getattr(args, "overlay", None)).materialize(
                container_url=container_url,
                container_pool=container_pool,
                run_id=getattr(args, "run_id", None),
            )
        except GeloMaterializeError as exc:
            raise SystemExit(str(exc)) from exc
    if getattr(args, "config", None):
        try:
            return GeloMaterializer(_json_file_object(args.config)).materialize(
                container_url=container_url,
                container_pool=container_pool,
                run_id=getattr(args, "run_id", None),
            )
        except GeloMaterializeError as exc:
            raise SystemExit(str(exc)) from exc
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
        f"go_ex={configured.get('go_ex')}"
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


def _gelo_startup(args: argparse.Namespace) -> int:
    try:
        catalog = _hosted_client(args).startup()
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    _print_hosted_startup(catalog, args.json)
    return 0


def _submit_hosted_gepa(args: argparse.Namespace) -> int:
    try:
        config_text = Path(args.config).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {args.config}: {exc}") from exc
    try:
        client = _hosted_client(args)
        container_pool = _container_pool_from_args(args)
        if getattr(args, "tunnel_url", None) and container_pool is not None:
            raise SystemExit("--tunnel-url and --container-pool are mutually exclusive")
        if getattr(args, "tunnel_url", None):
            with client.open_synth_tunnel(
                args.tunnel_url,
                metadata={"optimizer": "gepa", "run_id": args.run_id or ""},
            ) as tunnel:
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
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json and not args.follow:
        print(json.dumps(dict(submit.raw), indent=2, sort_keys=True))
        return 0

    print(f"submitted run_id={submit.run_id} status={submit.status.value}")
    if submit.events_url:
        print(f"events: {_api_endpoint(args.base_url, submit.events_url)}")

    if args.follow:
        try:
            for event in client.events(submit.run_id):
                event_type = str(event.get("event_type") or "event")
                status = str(event.get("status") or "")
                print(f"event {event_type} status={status}")
                if status in {"succeeded", "failed", "cancelled"}:
                    break
            final_record = client.get_run(submit.run_id)
        except HostedOptimizerError as exc:
            raise SystemExit(str(exc)) from exc
        if args.json:
            print(json.dumps(dict(final_record.raw), indent=2, sort_keys=True))
        else:
            print(f"final status={final_record.status.value}")
            if final_record.error:
                print(f"error: {final_record.error}")
        if final_record.status.value == "failed":
            return 1
    return 0


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


def _submit_hosted_gelo(args: argparse.Namespace) -> int:
    try:
        client = _hosted_client(args)
        config = _gelo_materialized_config(args)
        container_pool = _container_pool_from_args(args)
        if args.tunnel_url:
            with client.open_synth_tunnel(
                args.tunnel_url,
                metadata={"optimizer": "gelo", "run_id": args.run_id or ""},
            ) as tunnel:
                submit = client.submit_gelo(
                    config,
                    run_id=args.run_id,
                    idempotency_key=args.idempotency_key,
                    project_id=args.project_id,
                    container_tunnel=tunnel,
                )
        else:
            submit = client.submit_gelo(
                config,
                run_id=args.run_id,
                idempotency_key=args.idempotency_key,
                project_id=args.project_id,
                container_pool=container_pool,
            )
    except HostedOptimizerError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json and not args.follow:
        print(json.dumps(dict(submit.raw), indent=2, sort_keys=True))
        return 0

    print(f"submitted run_id={submit.run_id} status={submit.status.value}")
    if submit.events_url:
        print(f"events: {_api_endpoint(args.base_url, submit.events_url)}")

    if args.follow:
        try:
            for event in client.events(submit.run_id):
                event_type = str(event.get("event_type") or "event")
                status = str(event.get("status") or "")
                print(f"event {event_type} status={status}")
                if status in {"succeeded", "failed", "cancelled"}:
                    break
            final_record = client.get_run(submit.run_id)
        except HostedOptimizerError as exc:
            raise SystemExit(str(exc)) from exc
        if args.json:
            print(json.dumps(dict(final_record.raw), indent=2, sort_keys=True))
        else:
            print(f"final status={final_record.status.value}")
            if final_record.error:
                print(f"error: {final_record.error}")
        if final_record.status.value == "failed":
            return 1
    return 0


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
    gepa_submit.add_argument("--container-pool")
    gepa_submit.add_argument("--container-task-id")
    gepa_submit.add_argument("--timeout-seconds", type=float, default=120.0)
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
    gelo_submit.add_argument("--container-pool")
    gelo_submit.add_argument("--container-task-id")
    gelo_submit.add_argument("--proposer-rounds", type=int)
    gelo_submit.add_argument("--train-seed-count", type=int)
    gelo_submit.add_argument("--heldout-seed-count", type=int)
    gelo_submit.add_argument("--max-rollouts", type=int)
    gelo_submit.add_argument("--policy-model")
    gelo_submit.add_argument("--timeout-seconds", type=float, default=120.0)
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
        from .gepa import GepaRun

        old_terminal = os.environ.get("SYNTH_OPTIMIZERS_TERMINAL")
        old_proposer_execution_mode = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE")
        old_proposer_model = os.environ.get("SYNTH_OPTIMIZERS_PROPOSER_MODEL")
        old_proposer_reasoning_effort = os.environ.get(
            "SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT"
        )
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
            result = GepaRun.from_toml(args.config).execute()
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
                os.environ["SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE"] = (
                    old_proposer_execution_mode
                )
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
        run_dirs = _resolve_gepa_run_dirs(
            args.runs,
            args.root,
            args.all_terminal,
            args.older_than,
            args.status,
        )
        if not run_dirs:
            print("error: no runs matched", file=sys.stderr)
            return 1
        missing = [str(run_dir) for run_dir in run_dirs if not run_dir.is_dir()]
        if missing:
            print(f"error: run dir not found: {missing[0]}", file=sys.stderr)
            return 1
        dry_run = not args.yes
        reports = []
        try:
            for run_dir in run_dirs:
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
