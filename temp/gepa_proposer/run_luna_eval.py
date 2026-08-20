#!/usr/bin/env python3
"""Manufacture a Banking77 GEPA fixture, then score luna-low vs luna-medium."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OPTIMIZERS = HERE.parents[1]
COOKBOOKS = OPTIMIZERS.parent / "synth-cookbooks-public" / "cookbooks" / "optimizers" / "gepa"
MATRIX = COOKBOOKS / "pipeline_matrix"
RUNS = COOKBOOKS / "runs"
ENV_FILE = Path("/Users/joshuapurtell/Documents/GitHub/synth-ai/.env")
ENGINE = OPTIMIZERS / ".venv" / "bin" / "synth-optimizers"

sys.path.insert(0, str(HERE))
from export_checkpoints import export  # noqa: E402
from gepa_proposer.scoring import score_episode  # noqa: E402


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run_with_retries(cmd: list[str], *, cwd: Path | None = None, attempts: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            run(cmd, cwd=cwd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"attempt {attempt}/{attempts} failed with exit {exc.returncode}", flush=True)
    assert last_error is not None
    raise last_error


def generate_manufacture_config(run_id: str, port: int) -> Path:
    run(
        [sys.executable, str(MATRIX / "run_matrix.py"), "--dry-run", "--family", "banking77", "--mode", "sync_serial"],
        cwd=MATRIX,
    )
    src = COOKBOOKS / "banking77_container" / ".pipeline_matrix" / "gepa.b77_sync_serial_pmx01.toml"
    dest = HERE / "generated" / f"gepa.{run_id}.toml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    text = text.replace("b77_sync_serial_pmx01", run_id)
    text = text.replace("http://127.0.0.1:8765", f"http://127.0.0.1:{port}")
    text = text.replace("--port\", \"8765\"", f"--port\", \"{port}\"")
    dest.write_text(text)
    return dest


def write_score_config(
    *,
    manufacture_config: Path,
    run_id: str,
    fixture_path: Path,
    effort: str,
    manufacture_port: int,
    port: int,
    max_generations: int,
    proposer_rounds: int,
) -> Path:
    dest = HERE / "generated" / f"gepa.{run_id}.toml"
    text = manufacture_config.read_text()
    old_id = None
    for line in text.splitlines():
        if line.startswith("run_id ="):
            old_id = line.split("=", 1)[1].strip().strip('"')
            break
    if not old_id:
        raise SystemExit(f"no run_id in {manufacture_config}")
    text = text.replace(old_id, run_id)
    text = text.replace(f"http://127.0.0.1:{manufacture_port}", f"http://127.0.0.1:{port}")
    text = text.replace(f'--port", "{manufacture_port}"', f'--port", "{port}"')
    lines = []
    for line in text.splitlines():
        if line.startswith("max_generations ="):
            lines.append(f"max_generations = {max_generations}")
        elif line.startswith("max_total_rollouts ="):
            lines.append("max_total_rollouts = 10000")
        elif line.startswith("max_train_rollouts ="):
            lines.append("max_train_rollouts = 8000")
        elif line.startswith("timeout_seconds ="):
            lines.append(line)
            lines.append("message_stall_timeout_seconds = 300")
        elif line.startswith("reasoning_effort ="):
            lines.append(f'reasoning_effort = "{effort}"')
        else:
            lines.append(line)
        if line.startswith("seed ="):
            lines.append(f'fixture_path = "{fixture_path}"')
    if not any(line.startswith("[gepa.episode]") for line in lines):
        lines.extend(
            [
                "",
                "[gepa.episode]",
                f"proposer_rounds = {proposer_rounds}",
                "skip_heldout = false",
            ]
        )
    dest.write_text("\n".join(lines) + "\n")
    return dest


def candidates_from_cursor(cursor: dict[str, Any]) -> list[dict[str, Any]]:
    rows = cursor.get("candidates") or []
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def load_final_cursor(run_dir: Path) -> dict[str, Any]:
    workspace = run_dir / "workspace.sqlite"
    import sqlite3

    conn = sqlite3.connect(workspace)
    row = conn.execute(
        """
        SELECT checkpoint_json FROM checkpoints
        WHERE checkpoint_kind = 'gepa_cursor'
        ORDER BY sequence_number DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise SystemExit(f"no gepa_cursor in {workspace}")
    record = json.loads(row[0])
    return record.get("snapshot") or {}


def score_arm(*, fixture_path: Path, run_dir: Path, effort: str) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text())
    pre_fork = candidates_from_cursor(fixture["cursor"])
    pre_ids = {str(row.get("candidate_id")) for row in pre_fork}
    final = load_final_cursor(run_dir)
    episode = [
        row
        for row in candidates_from_cursor(final)
        if str(row.get("candidate_id")) not in pre_ids
    ]
    scored = score_episode(pre_fork=pre_fork, episode_candidates=episode)
    scored.update(
        {
            "effort": effort,
            "run_dir": str(run_dir),
            "pre_fork_candidates": len(pre_fork),
            "episode_candidates": len(episode),
            "episode_candidate_ids": [row.get("candidate_id") for row in episode],
            "final_generation": final.get("generation"),
            "final_phase": final.get("phase"),
        }
    )
    return scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-manufacture", action="store_true")
    parser.add_argument("--manufacture-run-id")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--score-task", default="train:1")
    parser.add_argument(
        "--proposer-rounds",
        type=int,
        default=1,
        help="Proposer rounds after the forked checkpoint (delta-from-restart).",
    )
    args = parser.parse_args()

    load_env_file(ENV_FILE)
    os.environ["SYNTH_OPTIMIZERS_VL_PROJECT"] = "0"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    manufacture_id = args.manufacture_run_id or f"b77_gepa_eval_mfg_{stamp}"

    if not args.skip_build:
        run(
            [
                "uv",
                "run",
                "--with",
                "maturin",
                "maturin",
                "build",
                "--release",
                "--out",
                "target/wheels",
            ],
            cwd=OPTIMIZERS,
        )
        wheels = sorted((OPTIMIZERS / "target" / "wheels").glob("synth_optimizers-*.whl"))
        if not wheels:
            raise SystemExit("maturin build produced no wheel")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(OPTIMIZERS / ".venv" / "bin" / "python"),
                "--no-deps",
                "--reinstall",
                "--no-cache",
                str(wheels[-1]),
            ],
            cwd=OPTIMIZERS,
        )

    config = generate_manufacture_config(manufacture_id, args.port)
    run_dir = RUNS / manufacture_id
    if not args.skip_manufacture:
        if run_dir.exists():
            raise SystemExit(f"run dir already exists: {run_dir}")
        run(
            [
                str(ENGINE),
                "gepa",
                "run",
                "--config",
                str(config),
                "--disable-usage-registration",
            ],
            cwd=OPTIMIZERS,
        )

    report = export(run_dir / "workspace.sqlite", HERE / "fixtures")
    print(json.dumps(report, indent=2), flush=True)
    task_row = next(row for row in report["tasks"] if row["task_id"] == args.score_task)
    if task_row["status"] != "written":
        raise SystemExit(f"missing fixture for {args.score_task}: {report}")
    fixture_path = Path(task_row["path"])
    fixture = json.loads(fixture_path.read_text())
    imported_generation = int(fixture.get("generation") or 0)
    proposer_rounds = max(1, args.proposer_rounds)
    max_generations = imported_generation + proposer_rounds

    results = []
    for effort in ("low", "medium"):
        last_error: Exception | None = None
        scored = None
        for attempt in range(1, 4):
            score_id = (
                f"b77_gepa_eval_{args.score_task.replace(':', '')}_{effort}_{stamp}_a{attempt}"
            )
            score_config = write_score_config(
                manufacture_config=config,
                run_id=score_id,
                fixture_path=fixture_path,
                effort=effort,
                manufacture_port=args.port,
                port=args.port + (1 if effort == "low" else 2),
                max_generations=max_generations,
                proposer_rounds=proposer_rounds,
            )
            try:
                run(
                    [
                        str(ENGINE),
                        "gepa",
                        "run",
                        "--config",
                        str(score_config),
                        "--disable-usage-registration",
                        "--proposer-reasoning-effort",
                        effort,
                    ],
                    cwd=OPTIMIZERS,
                )
                scored = score_arm(
                    fixture_path=fixture_path, run_dir=RUNS / score_id, effort=effort
                )
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                print(
                    f"{effort} attempt {attempt}/3 failed with exit {exc.returncode}",
                    flush=True,
                )
        if scored is None:
            raise last_error if last_error else SystemExit(f"{effort} scoring produced no result")
        results.append(scored)

    comparison = {
        "manufacture_run_id": manufacture_id,
        "score_task": args.score_task,
        "fixture_id": fixture.get("fixture_id"),
        "imported_generation": imported_generation,
        "max_generations": max_generations,
        "proposer_rounds": proposer_rounds,
        "arms": results,
    }
    out = HERE / "generated" / f"luna_low_vs_med_{stamp}.json"
    out.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
