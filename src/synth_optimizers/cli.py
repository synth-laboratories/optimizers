from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from . import (
    GepaRun,
    events_compare,
    events_replay,
    gepa_serve,
    gepa_service_recover,
    gepa_service_run_next,
    gepa_service_tick,
    workspace_cancel_run_request,
    workspace_claim_optimizer_job,
    workspace_claim_next_optimizer_job,
    workspace_claim_next_run_request,
    workspace_complete_run_request,
    workspace_fail_run_request,
    workspace_heartbeat_run_request,
    workspace_heartbeat_optimizer_job,
    workspace_mark_optimizer_job_running,
    workspace_recover_expired_optimizer_jobs,
    workspace_recover_expired_run_requests,
    workspace_start_run_request,
    workspace_status,
    workspace_submit_run_request,
    SynthOptimizerError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-optimizers")
    subcommands = parser.add_subparsers(dest="command", required=True)

    gepa = subcommands.add_parser("gepa")
    gepa_subcommands = gepa.add_subparsers(dest="gepa_command", required=True)
    gepa_run = gepa_subcommands.add_parser("run")
    gepa_run.add_argument("--config", required=True)
    gepa_run.add_argument(
        "--json",
        action="store_true",
        help="Print the full result JSON instead of the terminal progress view.",
    )
    gepa_service = gepa_subcommands.add_parser("service")
    gepa_service.add_argument("--db", required=True)
    gepa_service.add_argument("--bind", default="127.0.0.1:8879")
    gepa_service.add_argument("--worker-id")
    gepa_service.add_argument("--lease-seconds", type=int, default=3600)
    gepa_run_next = gepa_subcommands.add_parser("run-next")
    gepa_run_next.add_argument("--db", required=True)
    gepa_run_next.add_argument("--worker-id", default="synth-gepa-worker")
    gepa_run_next.add_argument("--lease-seconds", type=int, default=3600)
    gepa_tick = gepa_subcommands.add_parser("tick")
    gepa_tick.add_argument("--db", required=True)
    gepa_tick.add_argument("--worker-id", default="synth-gepa-worker")
    gepa_tick.add_argument("--lease-seconds", type=int, default=3600)
    gepa_recover = gepa_subcommands.add_parser("recover")
    gepa_recover.add_argument("--db", required=True)

    events = subcommands.add_parser("events")
    events_subcommands = events.add_subparsers(dest="events_command", required=True)
    replay = events_subcommands.add_parser("replay")
    replay.add_argument("--events", required=True)
    compare = events_subcommands.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)

    workspace = subcommands.add_parser("workspace")
    workspace_subcommands = workspace.add_subparsers(dest="workspace_command", required=True)
    status = workspace_subcommands.add_parser("status")
    status.add_argument("--db", required=True)
    submit = workspace_subcommands.add_parser("submit")
    submit.add_argument("--db", required=True)
    submit.add_argument("--config", required=True)
    submit.add_argument("--priority", type=int, default=0)
    claim = workspace_subcommands.add_parser("claim")
    claim.add_argument("--db", required=True)
    claim.add_argument("--lease-id", required=True)
    claim.add_argument("--worker-id")
    claim.add_argument("--lease-seconds", type=int, default=3600)
    start = workspace_subcommands.add_parser("start")
    start.add_argument("--db", required=True)
    start.add_argument("--request-id", required=True)
    heartbeat = workspace_subcommands.add_parser("heartbeat")
    heartbeat.add_argument("--db", required=True)
    heartbeat.add_argument("--request-id", required=True)
    heartbeat.add_argument("--lease-id", required=True)
    heartbeat.add_argument("--lease-seconds", type=int, default=3600)
    complete = workspace_subcommands.add_parser("complete")
    complete.add_argument("--db", required=True)
    complete.add_argument("--request-id", required=True)
    fail = workspace_subcommands.add_parser("fail")
    fail.add_argument("--db", required=True)
    fail.add_argument("--request-id", required=True)
    fail.add_argument("--error", required=True)
    fail.add_argument("--reason-code")
    cancel = workspace_subcommands.add_parser("cancel")
    cancel.add_argument("--db", required=True)
    cancel.add_argument("--request-id", required=True)
    cancel.add_argument("--reason", default="cancelled")
    recover = workspace_subcommands.add_parser("recover")
    recover.add_argument("--db", required=True)
    job_claim = workspace_subcommands.add_parser("job-claim")
    job_claim.add_argument("--db", required=True)
    job_claim.add_argument("--run-id", required=True)
    job_claim.add_argument("--job-id")
    job_claim.add_argument("--lease-id", required=True)
    job_claim.add_argument("--worker-id")
    job_claim.add_argument("--lease-seconds", type=int, default=300)
    job_running = workspace_subcommands.add_parser("job-running")
    job_running.add_argument("--db", required=True)
    job_running.add_argument("--run-id", required=True)
    job_running.add_argument("--job-id", required=True)
    job_running.add_argument("--lease-id", required=True)
    job_running.add_argument("--lease-seconds", type=int, default=300)
    job_heartbeat = workspace_subcommands.add_parser("job-heartbeat")
    job_heartbeat.add_argument("--db", required=True)
    job_heartbeat.add_argument("--run-id", required=True)
    job_heartbeat.add_argument("--job-id", required=True)
    job_heartbeat.add_argument("--lease-id", required=True)
    job_heartbeat.add_argument("--lease-seconds", type=int, default=300)
    job_recover = workspace_subcommands.add_parser("job-recover")
    job_recover.add_argument("--db", required=True)
    job_recover.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gepa" and args.gepa_command == "run":
        old_terminal = os.environ.get("SYNTH_OPTIMIZERS_TERMINAL")
        if not args.json:
            os.environ["SYNTH_OPTIMIZERS_TERMINAL"] = "1"
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
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print()
            print(f"artifacts: {Path(result.manifest_path).parent}")
        return 0
    if args.command == "gepa" and args.gepa_command == "service":
        gepa_serve(args.db, args.bind, args.worker_id, args.lease_seconds)
        return 0
    if args.command == "gepa" and args.gepa_command == "run-next":
        print(
            json.dumps(
                gepa_service_run_next(args.db, args.worker_id, args.lease_seconds),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "gepa" and args.gepa_command == "tick":
        print(
            json.dumps(
                gepa_service_tick(args.db, args.worker_id, args.lease_seconds),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "gepa" and args.gepa_command == "recover":
        print(json.dumps(gepa_service_recover(args.db), indent=2, sort_keys=True))
        return 0
    if args.command == "events" and args.events_command == "replay":
        print(events_replay(args.events), end="")
        return 0
    if args.command == "events" and args.events_command == "compare":
        events_compare(args.left, args.right)
        print("normalized event feeds match")
        return 0
    if args.command == "workspace" and args.workspace_command == "status":
        print(json.dumps(workspace_status(args.db), indent=2, sort_keys=True))
        return 0
    if args.command == "workspace" and args.workspace_command == "submit":
        print(
            json.dumps(
                workspace_submit_run_request(args.db, args.config, args.priority),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "claim":
        print(
            json.dumps(
                workspace_claim_next_run_request(
                    args.db,
                    args.lease_id,
                    args.worker_id,
                    args.lease_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "start":
        print(
            json.dumps(
                workspace_start_run_request(args.db, args.request_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "heartbeat":
        print(
            json.dumps(
                workspace_heartbeat_run_request(
                    args.db,
                    args.request_id,
                    args.lease_id,
                    args.lease_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "complete":
        print(
            json.dumps(
                workspace_complete_run_request(args.db, args.request_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "fail":
        print(
            json.dumps(
                workspace_fail_run_request(
                    args.db,
                    args.request_id,
                    args.error,
                    args.reason_code,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "cancel":
        print(
            json.dumps(
                workspace_cancel_run_request(args.db, args.request_id, args.reason),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "recover":
        print(
            json.dumps(
                workspace_recover_expired_run_requests(args.db),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "job-claim":
        if args.job_id:
            print(
                json.dumps(
                    workspace_claim_optimizer_job(
                        args.db,
                        args.run_id,
                        args.job_id,
                        args.lease_id,
                        args.worker_id,
                        args.lease_seconds,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(
            json.dumps(
                workspace_claim_next_optimizer_job(
                    args.db,
                    args.run_id,
                    args.lease_id,
                    args.worker_id,
                    args.lease_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "job-running":
        print(
            json.dumps(
                workspace_mark_optimizer_job_running(
                    args.db,
                    args.run_id,
                    args.job_id,
                    args.lease_id,
                    args.lease_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "job-heartbeat":
        print(
            json.dumps(
                workspace_heartbeat_optimizer_job(
                    args.db,
                    args.run_id,
                    args.job_id,
                    args.lease_id,
                    args.lease_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "workspace" and args.workspace_command == "job-recover":
        print(
            json.dumps(
                workspace_recover_expired_optimizer_jobs(args.db, args.run_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise SystemExit(f"unsupported command: {args}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
