from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from . import (
    GepaRun,
    SynthOptimizerError,
    events_compare,
    events_replay,
    gepa_serve,
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
    # The standing HTTP service is the public worker/workspace surface: queueing,
    # claiming, and lifecycle control happen over the /runs and /workspace routes.
    gepa_service = gepa_subcommands.add_parser("service")
    gepa_service.add_argument("--db", required=True)
    gepa_service.add_argument("--bind", default="127.0.0.1:8879")
    gepa_service.add_argument("--worker-id")
    gepa_service.add_argument("--lease-seconds", type=int, default=3600)

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
