#!/usr/bin/env python3
"""Start inner Banking77 x2 + GEPA service + proposer, then luna low vs medium in parallel."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
OPTIMIZERS = HERE.parents[1]
COOKBOOKS = OPTIMIZERS.parent / "synth-cookbooks-public" / "cookbooks" / "optimizers" / "gepa"
ENV_FILE = Path("/Users/joshuapurtell/Documents/GitHub/synth-ai/.env")
ENGINE = OPTIMIZERS / ".venv" / "bin" / "synth-optimizers"
PROPOSER_PY = HERE / ".venv" / "bin" / "python"


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


def port_open(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def wait_http(url: str, *, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urlopen(Request(url, method="GET"), timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:
            last = exc
        time.sleep(0.5)
    raise SystemExit(f"timeout waiting for {url}: {last}")


def post_json(url: str, body: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    payload = json.dumps(body).encode()
    request = Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    with urlopen(Request(url, method="GET"), timeout=timeout) as response:
        return json.loads(response.read().decode())


def spawn(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    print("+", " ".join(cmd), flush=True)
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


BANKING77_RUNTIME_ENV = {
    "BANKING77_TRAIN_SAMPLE": "100",
    "BANKING77_TEST_SAMPLE": "200",
    "BANKING77_POLICY_CONCURRENCY": "128",
    "BANKING77_POLICY_TIMEOUT_SECONDS": "60",
}


def ensure_banking77(
    port: int, env: dict[str, str], logs: Path, *, allow_reuse: bool
) -> subprocess.Popen[str] | None:
    if port_open("127.0.0.1", port):
        # A leftover container is NOT interchangeable with a fresh one: it carries
        # whatever BANKING77_POLICY_* env it was started with. The 2026-08-19 22:42
        # luna-low arm died on `infra failure rate 0.28 > 0.25` purely because the
        # process squatting on 8877 had been started without
        # BANKING77_POLICY_TIMEOUT_SECONDS and so ran the default 20s, producing 54
        # provider 504s under 128-way heldout concurrency. Refuse by default.
        if not allow_reuse:
            raise SystemExit(
                f"port {port} is already serving; refusing to reuse an inner container "
                f"this driver did not start (its BANKING77_POLICY_* env is unknown). "
                f"Kill it, pick a free port, or pass --allow-reused-inner."
            )
        print(f"banking77 already on {port} (reuse forced; policy env unverified)", flush=True)
        return None
    child_env = dict(env)
    child_env.update(BANKING77_RUNTIME_ENV)
    return spawn(
        [
            "uv",
            "run",
            "--project",
            "banking77_container",
            "python",
            "banking77_container/synth_service_app.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=COOKBOOKS,
        env=child_env,
        log_path=logs / f"banking77_{port}.log",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--task-id", default="train:1")
    parser.add_argument("--proposer-rounds", type=int, default=1)
    parser.add_argument("--proposer-port", type=int, default=8879)
    parser.add_argument("--service-port", type=int, default=8088)
    parser.add_argument("--inner-ports", default="8765,8766")
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            'JSON list of proposer arms, e.g. \'[{"label":"luna-low","provider":"openai",'
            '"model":"gpt-5.6-luna","reasoning_effort":"low"},'
            '{"label":"nemotron","provider":"openrouter",'
            '"model":"nvidia/nemotron-3.5-lightning"}]\'. '
            "Defaults to luna low vs luna medium. Needs one inner port per arm."
        ),
    )
    parser.add_argument("--allow-reused-inner", action="store_true")
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    parser.add_argument("--max-spend-usd", type=float, default=15.0)
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Independent episodes per arm (SCOPE pass is N>=3).",
    )
    parser.add_argument(
        "--ascope",
        action="store_true",
        help="Enable v0.7 operator surfaces on each episode (scratchpad, hypotheses, MQ inbox, reward extras, schema repair).",
    )
    args = parser.parse_args()

    load_env_file(ENV_FILE)
    # Logs only. Does not disable usage/cost accounting.
    os.environ["SYNTH_OPTIMIZERS_VL_PROJECT"] = "0"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    logs = HERE / "generated" / f"parallel_{stamp}"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)

    if not args.skip_build:
        subprocess.run(
            ["uv", "run", "--with", "maturin", "maturin", "build", "--release", "--out", "target/wheels"],
            cwd=OPTIMIZERS,
            check=True,
        )
        wheels = sorted((OPTIMIZERS / "target" / "wheels").glob("synth_optimizers-*.whl"))
        if not wheels:
            raise SystemExit("maturin build produced no wheel")
        subprocess.run(
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
            check=True,
        )

    if args.arms:
        arms = json.loads(args.arms)
        if not isinstance(arms, list) or not arms:
            raise SystemExit("--arms must be a non-empty JSON list")
    else:
        arms = [
            {
                "label": f"luna-{effort}",
                "provider": "openai",
                "model": "gpt-5.6-luna",
                "reasoning_effort": effort,
            }
            for effort in ("low", "medium")
        ]

    inner_ports = [int(item) for item in args.inner_ports.split(",") if item.strip()]
    if len(inner_ports) < len(arms):
        # GEPA exclusive-locks one container_url per run; sharing one inner across
        # parallel arms is a container_exclusive_conflict, not a slowdown.
        raise SystemExit(
            f"{len(arms)} arms need {len(arms)} inner ports; got {len(inner_ports)} "
            f"({args.inner_ports}). Pass --inner-ports with one port per arm."
        )
    children: list[subprocess.Popen[str]] = []
    try:
        for port in inner_ports:
            child = ensure_banking77(
                port, env, logs, allow_reuse=args.allow_reused_inner
            )
            if child is not None:
                children.append(child)
        for port in inner_ports:
            wait_http(f"http://127.0.0.1:{port}/health")

        env["SYNTH_OPTIMIZERS_DISK_BUDGET_SOFT_LIMIT_GB"] = os.environ.get(
            "SYNTH_OPTIMIZERS_DISK_BUDGET_SOFT_LIMIT_GB", "40"
        )
        env["SYNTH_OPTIMIZERS_DISK_BUDGET_HARD_LIMIT_GB"] = os.environ.get(
            "SYNTH_OPTIMIZERS_DISK_BUDGET_HARD_LIMIT_GB", "60"
        )
        env["SYNTH_OPTIMIZERS_DISK_BUDGET_PATH"] = str(logs / "gepa-runs")
        (logs / "gepa-runs").mkdir(parents=True, exist_ok=True)

        service_db = Path("/tmp") / f"gepa-banking77-{stamp}.sqlite"
        if not port_open("127.0.0.1", args.service_port):
            children.append(
                spawn(
                    [
                        str(ENGINE),
                        "gepa",
                        "service",
                        "--db",
                        str(service_db),
                        "--bind",
                        f"127.0.0.1:{args.service_port}",
                        "--workers",
                        "4",
                    ],
                    cwd=OPTIMIZERS,
                    env=env,
                    log_path=logs / "gepa_service.log",
                )
            )
        wait_http(f"http://127.0.0.1:{args.service_port}/health")

        proposer_env = dict(env)
        proposer_env["GEPA_SERVICE_URL"] = f"http://127.0.0.1:{args.service_port}"
        proposer_env["BANKING77_URLS"] = ",".join(f"http://127.0.0.1:{port}" for port in inner_ports)
        proposer_env["BANKING77_URL"] = f"http://127.0.0.1:{inner_ports[0]}"
        proposer_env["GEPA_PROPOSER_STATE_DIR"] = str(logs / "proposer-state")
        proposer_env["BANKING77_POLICY_TIMEOUT_SECONDS"] = "60"
        proposer_env["GEPA_PROPOSER_AUTH_MODE"] = "chatgpt"
        proposer_env["GEPA_PROPOSER_WAIT_HEADROOM_SECONDS"] = "90"
        proposer_env["SYNTH_OPTIMIZERS_VL_PROJECT"] = "0"
        proposer_env["GEPA_PROPOSER_OUTPUT_DIR"] = str(logs / "gepa-runs")
        if not port_open("127.0.0.1", args.proposer_port):
            children.append(
                spawn(
                    [
                        str(PROPOSER_PY if PROPOSER_PY.exists() else sys.executable),
                        "-m",
                        "gepa_proposer.app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.proposer_port),
                    ],
                    cwd=HERE,
                    env=proposer_env,
                    log_path=logs / "gepa_proposer.log",
                )
            )
        wait_http(f"http://127.0.0.1:{args.proposer_port}/health")

        base = f"http://127.0.0.1:{args.proposer_port}"
        episode = {
            "proposer_rounds": args.proposer_rounds,
            "skip_heldout": False,
            "max_wall_seconds": args.max_wall_seconds,
            "max_spend_usd": args.max_spend_usd,
        }
        if args.ascope:
            episode["schema_repair_rounds"] = 1
            episode["operator"] = {
                "scratchpad": {"enabled": True},
                "hypotheses": {"enabled": True},
                "manderqueue": {"enabled": True, "fail_closed": False},
                "control": {"pause": True, "restart": True, "branch": True},
                "levers": {"prompt": True, "code": True, "harness": True},
                "reward": {
                    "exploration_reduce": "mean",
                    "missing": "zero",
                    "confidence": True,
                    "time": True,
                    "cost": True,
                    "milestones": True,
                    "rubrics": True,
                },
            }

        all_records: list[dict[str, Any]] = []
        for replicate in range(args.replicates):
            started = []
            for arm in arms:
                label = arm.get("label") or f"{arm.get('model')}:{arm.get('reasoning_effort')}"
                policy = {k: v for k, v in arm.items() if k != "label"}
                started.append(
                    post_json(
                        f"{base}/rollout",
                        {
                            "task_id": args.task_id,
                            "submission_mode": "async",
                            "policy": policy,
                            "episode": episode,
                        },
                    )
                )
                print(
                    json.dumps(
                        {"started": label, "replicate": replicate, **started[-1]}
                    ),
                    flush=True,
                )

            deadline = time.time() + float(args.max_wall_seconds) + 180.0
            records = []
            while time.time() < deadline:
                records = [get_json(f"{base}/rollouts/{row['rollout_id']}") for row in started]
                statuses = [row.get("status") for row in records]
                print(
                    {
                        "replicate": replicate,
                        "poll": statuses,
                        "rewards": [row.get("reward") for row in records],
                    },
                    flush=True,
                )
                if all(status in {"completed", "failed", "cancelled"} for status in statuses):
                    break
                time.sleep(10)
            all_records.extend(records)

        comparison = {
            "task_id": args.task_id,
            "proposer_rounds": args.proposer_rounds,
            "replicates": args.replicates,
            "ascope": args.ascope,
            "stamp": stamp,
            "arms": [
                {
                    "label": arms[index % len(arms)].get("label"),
                    "replicate": index // len(arms),
                    "model": record.get("arm", {}).get("model"),
                    "provider": record.get("arm", {}).get("provider"),
                    "effort": record.get("arm", {}).get("reasoning_effort"),
                    "status": record.get("status"),
                    "reward": record.get("reward"),
                    "objective_scores": record.get("objective_scores"),
                    "optimizer_run_id": record.get("optimizer_run_id"),
                    "inner_url": record.get("inner_url"),
                    "error": record.get("error"),
                    "optional_terms": (record.get("reward_details") or {}).get(
                        "optional_terms"
                    ),
                    "episode_cost_usd": (record.get("reward_details") or {}).get(
                        "episode_cost_usd"
                    ),
                    "cost_pricing": (record.get("reward_details") or {}).get(
                        "cost_pricing"
                    ),
                    "exploration_reduce": (record.get("reward_details") or {}).get(
                        "exploration_reduce"
                    ),
                    "reward_details": {
                        "train_exploration": (record.get("reward_details") or {}).get(
                            "train_exploration"
                        ),
                        "train_exploitation": (record.get("reward_details") or {}).get(
                            "train_exploitation"
                        ),
                        "eval_uplift": (record.get("reward_details") or {}).get("eval_uplift"),
                        "heldout_evaluated": (record.get("reward_details") or {}).get(
                            "heldout_evaluated"
                        ),
                        "episode_candidate_ids": (record.get("reward_details") or {}).get(
                            "episode_candidate_ids"
                        ),
                    },
                }
                for index, record in enumerate(all_records)
            ],
        }
        out = logs / "luna_low_vs_med.json"
        out.write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2), flush=True)
        print(f"wrote {out}", flush=True)
        if any(row.get("status") != "completed" for row in all_records):
            return 1
        return 0
    finally:
        for child in children:
            child.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
