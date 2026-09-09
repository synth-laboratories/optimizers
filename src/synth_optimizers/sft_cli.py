"""Public SFT CLI handlers extracted so ``cli.py`` can keep shrinking.

``sft submit --follow`` treats public-job ``completed`` as terminal, along with
``succeeded``, ``failed``, and ``cancelled``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .sft import SftConfig, SftPublicServiceClient, SftServiceError, serve_sft_service

FOLLOW_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "completed"})


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


def dispatch(args: argparse.Namespace) -> int:
    command = args.sft_command
    if command == "validate":
        return sft_validate(args)
    if command == "submit":
        return sft_submit(args)
    if command == "watch":
        return sft_watch(args)
    if command in {"cancel", "pause", "resume"}:
        return sft_cancel(args)
    if command == "service":
        return sft_service(args)
    raise SystemExit(f"unknown sft command {command}")


def sft_service_client(args: argparse.Namespace) -> SftPublicServiceClient:
    token = os.environ.get(args.service_token_env) if args.service_token_env else None
    return SftPublicServiceClient(args.service_url, token, timeout_seconds=args.timeout_seconds)


def sft_validate(args: argparse.Namespace) -> int:
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


def sft_submit(args: argparse.Namespace) -> int:
    try:
        config_toml = Path(args.config).read_text(encoding="utf-8")
        client = sft_service_client(args)
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
        return poll_follow(
            lambda: client.get(run_id),
            poll_seconds=args.poll_seconds,
            json_output=args.json,
        )
    except (OSError, SftServiceError) as exc:
        raise SystemExit(str(exc)) from exc


def sft_watch(args: argparse.Namespace) -> int:
    try:
        client = sft_service_client(args)
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


def sft_cancel(args: argparse.Namespace) -> int:
    try:
        record = getattr(sft_service_client(args), args.sft_command)(args.run_id)
    except SftServiceError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(record, indent=2, sort_keys=True)
        if args.json
        else f"run_id={args.run_id} status={record.get('status')}"
    )
    return 0


def sft_service(args: argparse.Namespace) -> int:
    token = os.environ.get(args.service_token_env) if args.service_token_env else None
    serve_sft_service(args.db, args.bind, service_token=token)
    return 0
