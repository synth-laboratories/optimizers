from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from . import (
    GepaRun,
    SynthOptimizerError,
    events_compare,
    events_replay,
    gepa_compact_run_storage,
    gepa_delete_run_storage,
    gepa_serve,
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
    except (OSError, json.JSONDecodeError):
        return "terminal"
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
