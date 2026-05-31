"""Run synth-optimizers and gepa-ai Banking77 GEPA backends in parallel (or sequential)."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from banking77_gepa_ai_dev import benchmark_prompt
from banking77_synth_gepa_dev import (
    BACKEND_SUMMARY_FILENAME,
    BERKELEY_BOOT_GRACE_SECONDS,
    COMPARE_PARALLEL_POLICY_CONCURRENCY,
    DEFAULT_PORT,
    DEV_ROOT,
    GEPA_COMPUTE,
    GepaDevCompute,
    GepaRunLogStream,
    HELDOUT_SEED_LIST,
    LOG_PREFIX_BERKELEY,
    LOG_PREFIX_SYNTH,
    TRAIN_SEEDS,
    BackendRunSummary,
)

SYNTH_SCRIPT = Path(__file__).resolve().parent / "banking77_synth_gepa_dev.py"
GEPA_AI_SCRIPT = Path(__file__).resolve().parent / "banking77_gepa_ai_dev.py"


@dataclass
class BackendLiveState:
    label: str
    budget: int
    status: str = "starting"
    phase: str = "—"
    candidates: int = 0
    rollouts: int = 0
    best_train: float | None = None
    best_heldout: float | None = None
    best_id: str = "—"
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=4))

    def note(self, line: str) -> None:
        compact = line.strip()
        if len(compact) > 76:
            compact = compact[:73] + "..."
        if compact:
            self.recent.append(compact)
        self.status = "running"

    def ingest_synth(self, line: str) -> None:
        self.note(line)
        if match := re.search(r"generation (\d+) proposer (started|finished)", line):
            self.phase = f"gen {match.group(1)} {match.group(2)}"
        elif match := re.search(r"rollout section done .* rollouts=(\d+)", line):
            self.rollouts = max(self.rollouts, int(match.group(1)))
        elif match := re.search(r"rollout \S+ rows=(\d+)", line):
            self.rollouts = max(self.rollouts, int(match.group(1)))
        elif match := re.search(r"seed (\S+) train=([\d.]+)", line):
            self.best_id = match.group(1)
            self.best_train = float(match.group(2))
        elif match := re.search(r"heldout (\S+) train=([\d.]+) heldout=([\d.]+)", line):
            self.best_id = match.group(1)
            self.best_train = float(match.group(2))
            self.best_heldout = float(match.group(3))
        elif match := re.search(r"candidate (\S+) minibatch=", line):
            self.candidates += 1
            self.best_id = match.group(1)
        elif match := re.search(r"(accepted|rejected) (\S+)", line):
            self.candidates += 1
            self.best_id = match.group(2)
        elif match := re.search(r"frontier \+(\S+)", line):
            self.best_id = match.group(1)
        elif "container ready" in line:
            self.phase = "container ready"

    def ingest_berkeley(self, line: str) -> None:
        self.note(line)
        if match := re.search(r"Iteration (\d+):", line):
            self.phase = f"iter {match.group(1)}"
        if match := re.search(r"New program candidate index: (\d+)", line):
            self.candidates = max(self.candidates, int(match.group(1)) + 1)
        if match := re.search(r"Best valset aggregate score so far: ([\d.]+)", line):
            self.best_heldout = float(match.group(1))
        if match := re.search(r"Best program as per aggregate score on valset: (\d+)", line):
            self.best_id = f"idx {match.group(1)}"
        if "Valset score for new program:" in line or "subsample score" in line.lower():
            self.rollouts += 1

    def apply_summary(self, summary: BackendRunSummary) -> None:
        self.status = "done"
        if summary.reported_train is not None:
            self.best_train = float(summary.reported_train)
        if summary.reported_heldout is not None:
            self.best_heldout = float(summary.reported_heldout)
        if summary.budget_used.startswith("metric_calls="):
            try:
                self.rollouts = int(summary.budget_used.split("=", 1)[1])
            except ValueError:
                pass
        if summary.run_id:
            self.phase = "complete"


class CompareLivePanel:
    """Fixed terminal panel with side-by-side synth / berkeley stats and recent events."""

    PANEL_WIDTH = 78

    def __init__(self, compute: GepaDevCompute) -> None:
        self.compute = compute
        self.synth = BackendLiveState(label=LOG_PREFIX_SYNTH, budget=compute.max_total_rollouts)
        self.berkeley = BackendLiveState(label=LOG_PREFIX_BERKELEY, budget=compute.max_metric_calls)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mode = "parallel"
        self._alt_screen = False

    def start(self, *, mode: str) -> None:
        self._mode = mode
        self._alt_screen = sys.stdout.isatty()
        if self._alt_screen:
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop, name="compare-live-panel", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.draw(final=True)
        if self._alt_screen:
            sys.stdout.write("\033[?1049l\033[?25h")
            sys.stdout.flush()

    def ingest(self, backend: str, line: str) -> None:
        with self._lock:
            if backend == "synth":
                self.synth.ingest_synth(line)
            else:
                self.berkeley.ingest_berkeley(line)

    def apply_summary(self, backend: str, summary: BackendRunSummary) -> None:
        with self._lock:
            if backend == "synth":
                self.synth.apply_summary(summary)
            else:
                self.berkeley.apply_summary(summary)

    def _refresh_loop(self) -> None:
        while not self._stop.wait(0.12):
            self.draw()

    @staticmethod
    def _pct(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{100.0 * value:.1f}%"

    def _render_side(self, state: BackendLiveState) -> list[str]:
        rollout_label = f"{state.rollouts}/{state.budget}"
        return [
            f"{state.label:<12} status={state.status:<10} phase={state.phase}",
            f"{'':12} candidates={state.candidates:<4} rollouts={rollout_label:<10} best={state.best_id}",
            f"{'':12} train={self._pct(state.best_train):<8} heldout={self._pct(state.best_heldout)}",
        ]

    def _render_recent(self, left: BackendLiveState, right: BackendLiveState) -> list[str]:
        left_events = list(left.recent) or ["—"]
        right_events = list(right.recent) or ["—"]
        rows = max(len(left_events), len(right_events), 1)
        lines: list[str] = []
        for idx in range(rows):
            left_text = left_events[idx] if idx < len(left_events) else ""
            right_text = right_events[idx] if idx < len(right_events) else ""
            lines.append(f"  {left_text[:36]:<36} │ {right_text[:36]}")
        return lines

    def _render(self) -> str:
        width = self.PANEL_WIDTH
        border = "═" * width
        synth_lines = self._render_side(self.synth)
        berkeley_lines = self._render_side(self.berkeley)
        stat_block = [
            f" {synth_lines[0]:<36} │ {berkeley_lines[0]}",
            f" {synth_lines[1]:<36} │ {berkeley_lines[1]}",
            f" {synth_lines[2]:<36} │ {berkeley_lines[2]}",
        ]
        recent = self._render_recent(self.synth, self.berkeley)
        lines = [
            border,
            " Banking77 GEPA compare".ljust(width),
            f" mode={self._mode}  budget={self.compute.max_total_rollouts} rollouts/metric_calls".ljust(
                width
            ),
            border,
            " SYNTH (synth-optimizers)".ljust(36) + " │ BERKELEY (gepa-ai)",
            " recent events".ljust(36) + " │ recent events",
            "─" * width,
            *stat_block,
            "─" * width,
            *recent,
            border,
        ]
        return "\n".join(lines)

    def draw(self, *, final: bool = False) -> None:
        with self._lock:
            body = self._render()
        if self._alt_screen:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(body)
            if not final:
                sys.stdout.write("\n refreshing...")
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif final:
            print(body)


def _pick_free_port(*, host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _should_skip_berkeley_verbose_line(line: str) -> bool:
    if line.startswith("Iteration "):
        return False
    if any(
        token in line.lower() for token in ("score", "candidate", "program", "error", "traceback")
    ):
        return False
    return len(line) > 120


def _wait_for_synth_boot(panel: CompareLivePanel | None, *, timeout_seconds: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if panel is not None:
            with panel._lock:
                ready = any("container ready" in event for event in panel.synth.recent)
            if ready:
                return
        time.sleep(0.25)
    time.sleep(BERKELEY_BOOT_GRACE_SECONDS)


def _stream_subprocess(
    cmd: list[str],
    *,
    prefix: str,
    backend: str,
    panel: CompareLivePanel | None = None,
    verbose: bool = False,
) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        if panel is not None and stripped:
            panel.ingest(backend, stripped)
        if verbose and stripped:
            if backend == "berkeley" and _should_skip_berkeley_verbose_line(stripped):
                continue
            print(f"{prefix} {stripped}", flush=True)
    return proc.wait()


def _backend_worker_cmd(
    *,
    worker_script: Path,
    worker: str,
    run_id: str,
    output_dir: Path,
    port: int,
    compare_parallel: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(worker_script),
        "--worker",
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
    ]
    if compare_parallel:
        cmd.append("--compare-parallel")
    if worker == "synth":
        cmd.extend(["--port", str(port)])
    return cmd


def _run_backend_subprocess(
    *,
    worker: str,
    run_id: str,
    output_dir: Path,
    port: int,
    panel: CompareLivePanel | None = None,
    verbose: bool = False,
    compare_parallel: bool = False,
) -> BackendRunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / BACKEND_SUMMARY_FILENAME
    backend = "synth" if worker == "synth" else "berkeley"
    prefix = LOG_PREFIX_SYNTH if worker == "synth" else LOG_PREFIX_BERKELEY
    rc = _stream_subprocess(
        _backend_worker_cmd(
            worker_script=GEPA_AI_SCRIPT if worker == "berkeley" else SYNTH_SCRIPT,
            worker=worker,
            run_id=run_id,
            output_dir=output_dir,
            port=port,
            compare_parallel=compare_parallel,
        ),
        prefix=prefix,
        backend=backend,
        panel=panel,
        verbose=verbose,
    )
    if rc != 0:
        raise SystemExit(
            f"{prefix} worker exited with status {rc} (run_id={run_id}, output_dir={output_dir})"
        )
    summary = _read_backend_summary(summary_path)
    if panel is not None:
        panel.apply_summary(backend, summary)
    return summary


def _read_backend_summary(path: Path) -> BackendRunSummary:
    payload = json.loads(path.read_text())
    return BackendRunSummary(**payload)


def _print_comparison(
    *,
    synth_summary: BackendRunSummary,
    gepa_ai_summary: BackendRunSummary,
) -> None:
    synth_scores = benchmark_prompt(synth_summary.best_prompt)
    gepa_ai_scores = benchmark_prompt(gepa_ai_summary.best_prompt)
    GepaRunLogStream().compare_backends(
        synth_summary=synth_summary,
        gepa_ai_summary=gepa_ai_summary,
        compute=GEPA_COMPUTE,
        synth_scores=synth_scores,
        gepa_ai_scores=gepa_ai_scores,
    )


def run_compare_dev(*, port: int, verbose: bool, parallel: bool) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    synth_run_id = f"banking77_dev_{stamp}_synth"
    gepa_ai_run_id = f"banking77_dev_{stamp}_gepa_ai"
    synth_output = DEV_ROOT / "runs" / synth_run_id
    gepa_ai_output = DEV_ROOT / "runs" / gepa_ai_run_id
    synth_port = _pick_free_port() if parallel else port
    compare_parallel = parallel
    mode = "parallel" if parallel else "sequential"
    panel = CompareLivePanel(GEPA_COMPUTE) if not verbose else None

    if panel is not None:
        panel.start(mode=mode)
    elif not verbose:
        print(
            f"Running synth-optimizers and gepa-ai in {mode} "
            f"(train 0..{TRAIN_SEEDS[-1]}, heldout 0..{HELDOUT_SEED_LIST[-1]}, "
            f"max_rollouts={GEPA_COMPUTE.max_total_rollouts}) ..."
        )
    if parallel and compare_parallel:
        print(
            f"Synth container port={synth_port}; "
            f"parallel policy concurrency={COMPARE_PARALLEL_POLICY_CONCURRENCY}"
        )

    try:
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                synth_future = pool.submit(
                    _run_backend_subprocess,
                    worker="synth",
                    run_id=synth_run_id,
                    output_dir=synth_output,
                    port=synth_port,
                    panel=panel,
                    verbose=verbose,
                    compare_parallel=compare_parallel,
                )
                _wait_for_synth_boot(panel)
                berkeley_future = pool.submit(
                    _run_backend_subprocess,
                    worker="berkeley",
                    run_id=gepa_ai_run_id,
                    output_dir=gepa_ai_output,
                    port=synth_port,
                    panel=panel,
                    verbose=verbose,
                    compare_parallel=compare_parallel,
                )
                synth_summary = synth_future.result()
                gepa_ai_summary = berkeley_future.result()
        else:
            synth_summary = _run_backend_subprocess(
                worker="synth",
                run_id=synth_run_id,
                output_dir=synth_output,
                port=port,
                panel=panel,
                verbose=verbose,
                compare_parallel=False,
            )
            gepa_ai_summary = _run_backend_subprocess(
                worker="berkeley",
                run_id=gepa_ai_run_id,
                output_dir=gepa_ai_output,
                port=port,
                panel=panel,
                verbose=verbose,
                compare_parallel=False,
            )
    finally:
        if panel is not None:
            panel.stop()

    _print_comparison(synth_summary=synth_summary, gepa_ai_summary=gepa_ai_summary)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking77 GEPA compare (synth vs gepa-ai)")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run backends one after another instead of in parallel.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Stream full [SYNTH]/[BERKELEY] logs instead of the live panel.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    raise SystemExit(
        run_compare_dev(
            port=args.port,
            verbose=args.verbose,
            parallel=not args.sequential,
        )
    )


if __name__ == "__main__":
    main()
