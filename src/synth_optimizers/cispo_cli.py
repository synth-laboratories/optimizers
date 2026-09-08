"""Standalone ``synth-optimizers-cispo`` CLI for the public CISPO control plane.

True CISPO only (``algorithm_id=cispo``, ``slime-reference`` / ``cispo.slime.v1``).
Not SFT, not go-ex, not generic IS. Do not add these commands to ``cli.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .cispo_service import (
    CispoPublicServiceClient,
    CispoServiceError,
    serve_cispo_service,
)

FOLLOW_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "completed"})
DEFAULT_BIND = "127.0.0.1:8880"
DEFAULT_URL = "http://127.0.0.1:8880"
TOKEN_ENV = "SYNTH_OPTIMIZERS_CISPO_SERVICE_TOKEN"
URL_ENV = "SYNTH_OPTIMIZERS_CISPO_SERVICE_URL"


def follow_is_terminal(status: str) -> bool:
    return str(status or "").strip() in FOLLOW_TERMINAL_STATUSES


def poll_follow(
    get_record: Callable[[], Mapping[str, Any]],
    *,
    poll_seconds: float,
    json_output: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> int:
    """Poll a run until a follow-terminal status. No live Tinker required."""

    while True:
        record = get_record()
        status = str(record.get("status", "unknown"))
        emit(f"status={status}")
        if follow_is_terminal(status):
            if json_output:
                emit(json.dumps(dict(record), indent=2, sort_keys=True))
            return 1 if status == "failed" else 0
        sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-optimizers-cispo")
    commands = parser.add_subparsers(dest="cispo_command", required=True)

    service = commands.add_parser("service", help="Serve the public CISPO control plane.")
    service.add_argument("--db", default=".cispo/service.sqlite")
    service.add_argument("--bind", default=DEFAULT_BIND)
    service.add_argument(
        "--service-token-env",
        default=TOKEN_ENV,
        help="Optional inbound bearer-token environment variable.",
    )
    service.add_argument(
        "--fixture",
        action="store_true",
        help="Use the unpaid fixture executor (same as SYNTH_OPTIMIZERS_CISPO_FIXTURE=1).",
    )

    for command_name in ("submit", "watch", "cancel", "pause", "resume"):
        command = commands.add_parser(command_name)
        command.add_argument(
            "--service-url",
            default=os.environ.get(URL_ENV, DEFAULT_URL),
        )
        command.add_argument("--service-token-env", default=TOKEN_ENV)
        command.add_argument("--timeout-seconds", type=float, default=300.0)
        command.add_argument("--json", action="store_true")
    submit = commands.choices["submit"]
    submit.add_argument("--config", required=True, help="Path to a cispo.request.v1 JSON file.")
    submit.add_argument("--run-id")
    submit.add_argument("--idempotency-key")
    submit.add_argument("--follow", action="store_true")
    submit.add_argument("--poll-seconds", type=float, default=1.0)
    watch = commands.choices["watch"]
    watch.add_argument("run_id")
    watch.add_argument("--events", action="store_true")
    watch.add_argument("--after-seq", type=int, default=0)
    watch.add_argument("--limit", type=int, default=500)
    for action in ("cancel", "pause", "resume"):
        commands.choices[action].add_argument("run_id")
    return parser


def dispatch(args: argparse.Namespace) -> int:
    command = args.cispo_command
    if command == "service":
        return cispo_service(args)
    if command == "submit":
        return cispo_submit(args)
    if command == "watch":
        return cispo_watch(args)
    if command in {"cancel", "pause", "resume"}:
        return cispo_cancel(args)
    raise SystemExit(f"unknown cispo command {command}")


def cispo_service_client(args: argparse.Namespace) -> CispoPublicServiceClient:
    token = os.environ.get(args.service_token_env) if args.service_token_env else None
    return CispoPublicServiceClient(args.service_url, token, timeout_seconds=args.timeout_seconds)


def cispo_service(args: argparse.Namespace) -> int:
    token = os.environ.get(args.service_token_env) if args.service_token_env else None
    serve_cispo_service(
        args.db,
        args.bind,
        service_token=token,
        fixture=bool(args.fixture),
    )
    return 0


def cispo_submit(args: argparse.Namespace) -> int:
    try:
        config_json = _json_file_object(args.config)
        client = cispo_service_client(args)
        submitted = client.submit(
            config_json,
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
        return poll_follow(
            lambda: client.get(run_id),
            poll_seconds=args.poll_seconds,
            json_output=args.json,
        )
    except (OSError, CispoServiceError) as exc:
        raise SystemExit(str(exc)) from exc


def cispo_watch(args: argparse.Namespace) -> int:
    try:
        client = cispo_service_client(args)
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
    except CispoServiceError as exc:
        raise SystemExit(str(exc)) from exc


def cispo_cancel(args: argparse.Namespace) -> int:
    try:
        record = getattr(cispo_service_client(args), args.cispo_command)(args.run_id)
    except CispoServiceError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(record, indent=2, sort_keys=True)
        if args.json
        else f"run_id={args.run_id} status={record.get('status')}"
    )
    return 0


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
    return dispatch(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
