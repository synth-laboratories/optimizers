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

from gepa_proposer.ascope_harvest import harvest_episode_dir
from gepa_proposer.episode import build_run_request
from gepa_proposer.fixtures import by_task_id

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


def pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def jesterky_command() -> str | None:
    candidates = [
        Path("/Users/joshuapurtell/Documents/GitHub/jesterky/target/release/jesterky"),
        Path.home() / ".cargo" / "bin" / "jesterky",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


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
        raw = response.read().decode()
        return json.loads(raw) if raw else {}


def prove_operator_control(
    *,
    service_url: str,
    records: list[dict[str, Any]],
    episode: dict[str, Any],
) -> dict[str, Any]:
    """Fork + pause a completed GEPA episode so ascope control is live-proven."""
    parent = next(
        (
            row
            for row in records
            if row.get("status") == "completed" and row.get("optimizer_run_id")
        ),
        None,
    )
    if parent is None:
        return {"ok": False, "error": "no completed parent run to fork"}
    spec = by_task_id(str(parent.get("task_id") or "train:1"))
    body = build_run_request(
        spec=spec,
        cursor=spec["cursor"],
        arm=parent.get("arm") or {},
        episode={**episode, "proposer_rounds": 1, "skip_heldout": True, "jesterky": False, "jesterky_bulk": False},
        container_url=str(parent.get("inner_url") or ""),
        run_id=f"fork-{str(parent.get('optimizer_run_id'))[-8:]}",
    )
    body.pop("fixture", None)
    body["fork_from"] = {"run_id": parent["optimizer_run_id"]}
    try:
        created = post_json(f"{service_url.rstrip('/')}/runs", body, timeout=60.0)
    except Exception as exc:
        return {"ok": False, "error": f"fork_from failed: {exc}"}
    child = str(created.get("run_id") or created.get("id") or "")
    if not child:
        return {"ok": False, "error": "fork created no run_id", "created": created}
    try:
        paused = post_json(
            f"{service_url.rstrip('/')}/runs/{child}/pause",
            {"timeout_seconds": 120},
            timeout=60.0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "parent": parent["optimizer_run_id"],
            "child": child,
            "created_status": created.get("status"),
            "error": f"pause failed: {exc}",
        }
    return {
        "ok": str(paused.get("status") or "") == "paused",
        "parent": parent["optimizer_run_id"],
        "child": child,
        "created_status": created.get("status"),
        "paused_status": paused.get("status"),
    }


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


def inner_family(task_id: str) -> str:
    if task_id.startswith("tau2:"):
        return "tau2"
    return "banking77"


def ensure_tau2(
    port: int, env: dict[str, str], logs: Path, *, allow_reuse: bool
) -> subprocess.Popen[str] | None:
    if port_open("127.0.0.1", port):
        if not allow_reuse:
            raise SystemExit(
                f"port {port} is already serving; refusing to reuse an inner container "
                f"this driver did not start. Kill it, pick a free port, or pass --allow-reused-inner."
            )
        print(f"tau2 already on {port} (reuse forced)", flush=True)
        return None
    python = HERE / "tau2_container" / ".venv" / "bin" / "python"
    if not python.is_file():
        raise SystemExit(f"missing {python}; run uv sync in temp/gepa_proposer/tau2_container")
    return spawn(
        [
            str(python),
            "synth_service_app.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=HERE / "tau2_container",
        env=env,
        log_path=logs / f"tau2_{port}.log",
    )


def ensure_inner(
    family: str, port: int, env: dict[str, str], logs: Path, *, allow_reuse: bool
) -> subprocess.Popen[str] | None:
    if family == "tau2":
        return ensure_tau2(port, env, logs, allow_reuse=allow_reuse)
    return ensure_banking77(port, env, logs, allow_reuse=allow_reuse)


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
    parser.add_argument(
        "--serial-arms",
        action="store_true",
        help="Run arms one at a time (all replicates of arm A, then arm B). Avoids OpenRouter 429 while luna is proposing.",
    )
    parser.add_argument(
        "--pipeline-mode",
        default="sync_serial",
        help="GEPA pipeline mode: sync_serial, flash_evolve/combee, or async_pipelined.",
    )
    args = parser.parse_args()

    load_env_file(ENV_FILE)
    # Logs only. Does not disable usage/cost accounting.
    os.environ["SYNTH_OPTIMIZERS_VL_PROJECT"] = "0"
    # Seconds-only stamps let independently launched replays share proposer
    # state and overwrite luna_low_vs_med.json.  The PID also protects the
    # (unlikely) case where two processes format the same microsecond.
    stamp = f'{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")}_{os.getpid()}'
    logs = HERE / "generated" / f"parallel_{stamp}"
    logs.mkdir(parents=True, exist_ok=False)
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
    family = inner_family(args.task_id)
    needed = 1 if args.serial_arms else len(arms)
    if len(inner_ports) < needed:
        raise SystemExit(
            f"{'serial arms need 1 inner port' if args.serial_arms else f'{len(arms)} parallel arms need {len(arms)} inner ports'}; "
            f"got {len(inner_ports)} ({args.inner_ports})."
        )
    children: list[subprocess.Popen[str]] = []
    try:
        for port in inner_ports:
            child = ensure_inner(
                family, port, env, logs, allow_reuse=args.allow_reused_inner
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
        jesterky_bin = jesterky_command() if args.ascope else None
        if jesterky_bin:
            env["STACK_JESTERKY_COMMAND"] = jesterky_bin

        service_db = Path("/tmp") / f"gepa-{family}-{stamp}.sqlite"
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
        inner_urls = ",".join(f"http://127.0.0.1:{port}" for port in inner_ports)
        if family == "tau2":
            proposer_env["TAU2_URLS"] = inner_urls
            proposer_env["TAU2_URL"] = f"http://127.0.0.1:{inner_ports[0]}"
        else:
            proposer_env["BANKING77_URLS"] = inner_urls
            proposer_env["BANKING77_URL"] = f"http://127.0.0.1:{inner_ports[0]}"
            proposer_env["BANKING77_POLICY_TIMEOUT_SECONDS"] = "60"
        proposer_env["GEPA_PROPOSER_STATE_DIR"] = str(logs / "proposer-state")
        proposer_env["GEPA_PROPOSER_AUTH_MODE"] = "chatgpt"
        proposer_env["GEPA_PROPOSER_WAIT_HEADROOM_SECONDS"] = "90"
        proposer_env["GEPA_PROPOSER_STALL_SECONDS"] = os.environ.get(
            "GEPA_PROPOSER_STALL_SECONDS"
        ) or "300"
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
            "pipeline_mode": args.pipeline_mode,
        }
        if args.ascope:
            mq_port = pick_free_port()
            children.append(
                spawn(
                    [
                        str(PROPOSER_PY if PROPOSER_PY.exists() else sys.executable),
                        "-m",
                        "gepa_proposer.mq_stub",
                        "--port",
                        str(mq_port),
                        "--thread-id",
                        "gepa-ascope",
                    ],
                    cwd=HERE,
                    env=env,
                    log_path=logs / "manderqueue_stub.log",
                )
            )
            wait_http(f"http://127.0.0.1:{mq_port}/health")
            episode["schema_repair_rounds"] = 1
            episode["jesterky"] = True
            # Cap 6 traces (engine default). Bulk annotates the whole minibatch
            # with gpt-5.5-high and blocks the proposer for the 600s workflow budget.
            episode["jesterky_bulk"] = False
            if jesterky_bin:
                episode["jesterky_workflow"] = {
                    "enabled": True,
                    "bulk": False,
                    "fail_closed": False,
                    "command": jesterky_bin,
                    "spec": "/Users/joshuapurtell/Documents/GitHub/jesterky/examples/gepa_trace_annotate.json",
                    # gpt-5.5-high on 6 traces previously hit the 600s fail-open
                    # and 429'd the OpenRouter arm. Luna is the eval proposer.
                    "model": "gpt-5.6-luna",
                }
            episode["operator"] = {
                "scratchpad": {"enabled": True},
                "hypotheses": {"enabled": True},
                "manderqueue": {
                    "enabled": True,
                    "fail_closed": False,
                    "base_url": f"http://127.0.0.1:{mq_port}",
                    "thread_id": "gepa-ascope",
                },
                "control": {"pause": True, "restart": True, "branch": True},
                "levers": {"prompt": True, "code": True, "harness": True},
                "mcp_agent": {
                    "enabled": True,
                    "command": "npx -y @modelcontextprotocol/server-filesystem .",
                    "server": "workspace_fs",
                },
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

        def wait_started(started: list[dict[str, Any]], *, replicate: int) -> list[dict[str, Any]]:
            deadline = time.time() + float(args.max_wall_seconds) + 180.0
            records: list[dict[str, Any]] = []
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
            return records

        def start_arm(arm: dict[str, Any], *, replicate: int) -> dict[str, Any]:
            label = arm.get("label") or f"{arm.get('model')}:{arm.get('reasoning_effort')}"
            policy = {k: v for k, v in arm.items() if k != "label"}
            started = post_json(
                f"{base}/rollout",
                {
                    "task_id": args.task_id,
                    "submission_mode": "async",
                    "policy": policy,
                    "episode": episode,
                },
            )
            print(json.dumps({"started": label, "replicate": replicate, **started}), flush=True)
            started["_label"] = label
            started["_replicate"] = replicate
            return started

        if args.serial_arms:
            for arm_index, arm in enumerate(arms):
                for replicate in range(args.replicates):
                    started = start_arm(arm, replicate=replicate)
                    records = wait_started([started], replicate=replicate)
                    for record in records:
                        record["_label"] = started["_label"]
                        record["_replicate"] = replicate
                        record["_arm_index"] = arm_index
                    all_records.extend(records)
        else:
            for replicate in range(args.replicates):
                started = [start_arm(arm, replicate=replicate) for arm in arms]
                records = wait_started(started, replicate=replicate)
                for index, record in enumerate(records):
                    record["_label"] = started[index]["_label"]
                    record["_replicate"] = replicate
                    record["_arm_index"] = index
                all_records.extend(records)

        control = {"ok": False, "error": "ascope off"}
        if args.ascope:
            control = prove_operator_control(
                service_url=f"http://127.0.0.1:{args.service_port}",
                records=all_records,
                episode=episode,
            )
            print(json.dumps({"operator_control": control}), flush=True)

        comparison = {
            "task_id": args.task_id,
            "proposer_rounds": args.proposer_rounds,
            "replicates": args.replicates,
            "ascope": args.ascope,
            "operator_control": control,
            "stamp": stamp,
            "arms": [
                {
                    "label": record.get("_label") or arms[index % len(arms)].get("label"),
                    "replicate": record.get("_replicate", index // len(arms)),
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
                    "ascope": record.get("ascope")
                    or harvest_episode_dir(
                        record.get("output_dir")
                        or (logs / "gepa-runs" / str(record.get("run_id") or ""))
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
