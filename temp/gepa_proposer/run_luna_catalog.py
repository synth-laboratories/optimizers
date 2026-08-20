#!/usr/bin/env python3
"""Run every reachable catalog seed: luna low vs medium, $1 / 5 minutes."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
OPTIMIZERS = HERE.parents[1]
COOKBOOKS = OPTIMIZERS.parent / "synth-cookbooks-public" / "cookbooks" / "optimizers" / "gepa"
ENV_FILE = Path("/Users/joshuapurtell/Documents/GitHub/synth-ai/.env")
ENGINE = OPTIMIZERS / ".venv" / "bin" / "synth-optimizers"
PROPOSER_PY = HERE / ".venv" / "bin" / "python"
OFFICEQA_CSV_CANDIDATES = [
    Path(os.environ.get("OFFICEQA_CSV") or ""),
    HERE / "officeqa_container" / "data" / "officeqa_full.csv",
]

FAMILIES: dict[str, dict[str, Any]] = {
    "banking77": {
        "task_ids": ["train:0", "train:1", "train:2", "train:3", "train:4", "train:5"],
        "pool_env": "BANKING77_URLS",
        "ports": [8877, 8765],
    },
    "healthbench": {
        "task_ids": [
            "healthbench:0",
            "healthbench:1",
            "healthbench:2",
            "healthbench:3",
            "healthbench:4",
        ],
        "pool_env": "HEALTHBENCH_URLS",
        "ports": [8114, 8115],
    },
    "crafter": {
        "task_ids": ["crafter:0", "crafter:1", "crafter:2", "crafter:3", "crafter:4"],
        "pool_env": "CRAFTER_URLS",
        "ports": [8768, 8767],
    },
    "tau2": {
        "task_ids": ["tau2:0", "tau2:1", "tau2:2"],
        "pool_env": "TAU2_URLS",
        "ports": [8774, 8775],
    },
    "minigrid": {
        "task_ids": ["minigrid:0", "minigrid:1", "minigrid:2"],
        "pool_env": "MINIGRID_URLS",
        "ports": [8769, 8770],
    },
    "officeqa": {
        "task_ids": ["officeqa:0"],
        "pool_env": "OFFICEQA_URLS",
        "ports": [8120],
    },
}


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
    raise TimeoutError(f"timeout waiting for {url}: {last}")


def post_json(url: str, body: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    payload = json.dumps(body).encode()
    request = Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    last: Exception | None = None
    for attempt in range(8):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            last = exc
            if exc.code in {409, 429, 503} and attempt < 7:
                time.sleep(2.0 * (attempt + 1))
                request = Request(
                    url, data=payload, method="POST", headers={"Content-Type": "application/json"}
                )
                continue
            raise RuntimeError(f"POST {url} -> {exc.code}: {detail[:800]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
            request = Request(
                url, data=payload, method="POST", headers={"Content-Type": "application/json"}
            )
    raise RuntimeError(f"POST {url} failed: {last}")


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


def officeqa_csv() -> Path | None:
    for path in OFFICEQA_CSV_CANDIDATES:
        if path and path.is_file():
            return path
    return None


def ensure_port(
    *,
    port: int,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    logs: Path,
    name: str,
    health_timeout: float = 90.0,
) -> tuple[subprocess.Popen[str] | None, bool]:
    if port_open("127.0.0.1", port):
        print(f"{name} already on {port}", flush=True)
        return None, True
    child = spawn(cmd, cwd=cwd, env=env, log_path=logs / f"{name}_{port}.log")
    try:
        wait_http(f"http://127.0.0.1:{port}/health", timeout=health_timeout)
        return child, True
    except Exception as exc:
        print(f"{name} failed on {port}: {exc}", flush=True)
        child.terminate()
        return None, False


def arm_summary(record: dict[str, Any]) -> dict[str, Any]:
    details = record.get("reward_details") or {}
    return {
        "effort": (record.get("arm") or {}).get("reasoning_effort"),
        "status": record.get("status"),
        "success_status": record.get("success_status"),
        "reward": record.get("reward"),
        "objective_scores": record.get("objective_scores"),
        "optimizer_run_id": record.get("optimizer_run_id"),
        "optimizer_status": record.get("optimizer_status"),
        "inner_url": record.get("inner_url"),
        "error": record.get("error"),
        "train_exploration": details.get("train_exploration"),
        "train_exploitation": details.get("train_exploitation"),
        "eval_uplift": details.get("eval_uplift"),
        "heldout_evaluated": details.get("heldout_evaluated"),
        "episode_candidate_ids": details.get("episode_candidate_ids"),
    }


def wait_rollouts(
    base: str,
    started: list[dict[str, Any]],
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout
    records: list[dict[str, Any]] = []
    while time.time() < deadline:
        records = [get_json(f"{base}/rollouts/{row['rollout_id']}") for row in started]
        statuses = [row.get("status") for row in records]
        print(
            {
                "poll": [
                    {
                        "task_id": row.get("task_id"),
                        "effort": (row.get("arm") or {}).get("reasoning_effort"),
                        "status": row.get("status"),
                        "reward": row.get("reward"),
                    }
                    for row in records
                ]
            },
            flush=True,
        )
        if all(status in {"completed", "failed", "cancelled"} for status in statuses):
            return records
        time.sleep(15)
    return records or [get_json(f"{base}/rollouts/{row['rollout_id']}") for row in started]


def start_arm(base: str, task_id: str, effort: str, episode: dict[str, Any]) -> dict[str, Any]:
    body = {
        "task_id": task_id,
        "submission_mode": "async",
        "candidate": {},
        "policy": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "reasoning_effort": effort,
        },
        "episode": episode,
    }
    started = post_json(f"{base}/rollout", body)
    print(json.dumps({"started": effort, "task_id": task_id, **started}), flush=True)
    return started


def run_task(
    *,
    base: str,
    task_id: str,
    n_urls: int,
    episode: dict[str, Any],
    wait_timeout: float,
) -> dict[str, Any]:
    efforts = ("low", "medium")
    if n_urls >= 2:
        started = [start_arm(base, task_id, effort, episode) for effort in efforts]
        records = wait_rollouts(base, started, timeout=wait_timeout)
    else:
        records = []
        for effort in efforts:
            started = [start_arm(base, task_id, effort, episode)]
            records.extend(wait_rollouts(base, started, timeout=wait_timeout))
    by_effort = {}
    for record in records:
        effort = str((record.get("arm") or {}).get("reasoning_effort") or "")
        by_effort[effort] = arm_summary(record)
    return {"task_id": task_id, "arms": by_effort}


def fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], skipped: list[dict[str, str]]) -> None:
    print("\n=== luna low vs medium ===", flush=True)
    header = (
        f"{'task':18} {'low_status':12} {'low_R':8} {'low_expl':9} {'low_explt':9} {'low_upl':8} "
        f"{'med_status':12} {'med_R':8} {'med_expl':9} {'med_explt':9} {'med_upl':8} {'heldout'}"
    )
    print(header, flush=True)
    for row in rows:
        low = (row.get("arms") or {}).get("low") or {}
        med = (row.get("arms") or {}).get("medium") or {}
        held = f"L={low.get('heldout_evaluated')} M={med.get('heldout_evaluated')}"
        print(
            f"{row.get('task_id', ''):18} "
            f"{str(low.get('status') or 'missing'):12} {fmt_num(low.get('reward')):8} "
            f"{fmt_num(low.get('train_exploration')):9} {fmt_num(low.get('train_exploitation')):9} "
            f"{fmt_num(low.get('eval_uplift')):8} "
            f"{str(med.get('status') or 'missing'):12} {fmt_num(med.get('reward')):8} "
            f"{fmt_num(med.get('train_exploration')):9} {fmt_num(med.get('train_exploitation')):9} "
            f"{fmt_num(med.get('eval_uplift')):8} {held}",
            flush=True,
        )
        for effort, arm in (("low", low), ("medium", med)):
            if arm.get("error"):
                print(f"  {effort} error: {arm['error']}", flush=True)
    if skipped:
        print("\nskipped:", flush=True)
        for item in skipped:
            print(f"  {item['task_id']}: {item['reason']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--proposer-port", type=int, default=8879)
    parser.add_argument("--service-port", type=int, default=8088)
    parser.add_argument("--max-spend-usd", type=float, default=1.0)
    parser.add_argument("--max-wall-seconds", type=int, default=300)
    parser.add_argument("--proposer-rounds", type=int, default=1)
    args = parser.parse_args()

    load_env_file(ENV_FILE)
    os.environ["SYNTH_OPTIMIZERS_VL_PROJECT"] = "0"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    logs = HERE / "generated" / f"catalog_{stamp}"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    episode = {
        "proposer_rounds": args.proposer_rounds,
        "skip_heldout": False,
        "max_wall_seconds": args.max_wall_seconds,
        "max_spend_usd": args.max_spend_usd,
    }
    wait_timeout = float(args.max_wall_seconds) + 120.0
    skipped: list[dict[str, str]] = []
    reachable: dict[str, list[str]] = {}

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

    children: list[subprocess.Popen[str]] = []
    try:
        # Banking77: reuse leftover 8877; start 8765 as the second exclusive URL.
        for port in FAMILIES["banking77"]["ports"]:
            child_env = dict(env)
            child_env["BANKING77_TRAIN_SAMPLE"] = "100"
            child_env["BANKING77_TEST_SAMPLE"] = "200"
            child_env["BANKING77_POLICY_CONCURRENCY"] = "128"
            child_env["BANKING77_POLICY_TIMEOUT_SECONDS"] = "60"
            child, ok = ensure_port(
                port=port,
                cmd=[
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
                logs=logs,
                name="banking77",
                health_timeout=90,
            )
            if child is not None:
                children.append(child)
            if ok:
                reachable.setdefault("banking77", []).append(f"http://127.0.0.1:{port}")

        # HealthBench: reuse 8114; start a second isolated storage root on 8115.
        for port in FAMILIES["healthbench"]["ports"]:
            storage = logs / f"healthbench_{port}"
            child, ok = ensure_port(
                port=port,
                cmd=[
                    "uv",
                    "run",
                    "--project",
                    "healthbench_groq",
                    "python",
                    "healthbench_groq/synth_service_app.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--storage-root",
                    str(storage),
                ],
                cwd=COOKBOOKS,
                env=env,
                logs=logs,
                name="healthbench",
                health_timeout=180,
            )
            if child is not None:
                children.append(child)
            if ok:
                reachable.setdefault("healthbench", []).append(f"http://127.0.0.1:{port}")

        # Crafter: OpenAI policy from the fixture overlay, not the cookbook Gemini default.
        for port in FAMILIES["crafter"]["ports"]:
            child_env = dict(env)
            child_env["CRAFTER_MAX_TURNS"] = "20"
            child_env["CRAFTER_MIN_BATCH"] = "1"
            child_env["CRAFTER_MAX_BATCH"] = "5"
            child, ok = ensure_port(
                port=port,
                cmd=[
                    "uv",
                    "run",
                    "--project",
                    "crafter_container",
                    "python",
                    "crafter_container/synth_service_app.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=COOKBOOKS,
                env=child_env,
                logs=logs,
                name="crafter",
                health_timeout=180,
            )
            if child is not None:
                children.append(child)
            if ok:
                reachable.setdefault("crafter", []).append(f"http://127.0.0.1:{port}")

        tau2_py = HERE / "tau2_container" / ".venv" / "bin" / "python"
        for port in FAMILIES["tau2"]["ports"]:
            child, ok = ensure_port(
                port=port,
                cmd=[
                    str(tau2_py if tau2_py.exists() else sys.executable),
                    "synth_service_app.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=HERE / "tau2_container",
                env=env,
                logs=logs,
                name="tau2",
                health_timeout=60,
            )
            if child is not None:
                children.append(child)
            if ok:
                reachable.setdefault("tau2", []).append(f"http://127.0.0.1:{port}")

        minigrid_py = HERE / "minigrid_container" / ".venv" / "bin" / "python"
        for port in FAMILIES["minigrid"]["ports"]:
            child_env = dict(env)
            child_env["MINIGRID_ENV_ID"] = "MiniGrid-Empty-5x5-v0"
            child, ok = ensure_port(
                port=port,
                cmd=[
                    str(minigrid_py if minigrid_py.exists() else sys.executable),
                    "synth_service_app.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=HERE / "minigrid_container",
                env=child_env,
                logs=logs,
                name="minigrid",
                health_timeout=60,
            )
            if child is not None:
                children.append(child)
            if ok:
                reachable.setdefault("minigrid", []).append(f"http://127.0.0.1:{port}")

        if officeqa_csv() is None:
            skipped.append(
                {
                    "task_id": "officeqa:0",
                    "reason": "OFFICEQA_CSV missing; inner /rollout is 503 without the gated CSV",
                }
            )

        service_db = Path("/tmp") / f"gepa-catalog-{stamp}.sqlite"
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
        proposer_env["GEPA_PROPOSER_STATE_DIR"] = str(logs / "proposer-state")
        proposer_env["GEPA_PROPOSER_AUTH_MODE"] = "chatgpt"
        proposer_env["GEPA_PROPOSER_WAIT_HEADROOM_SECONDS"] = "90"
        proposer_env["SYNTH_OPTIMIZERS_VL_PROJECT"] = "0"
        for family, urls in reachable.items():
            pool_env = str(FAMILIES[family]["pool_env"])
            proposer_env[pool_env] = ",".join(urls)
            fallback = pool_env[:-1] if pool_env.endswith("S") else pool_env
            proposer_env[fallback] = urls[0]
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
        print({"reachable": reachable, "skipped": skipped, "logs": str(logs)}, flush=True)

        base = f"http://127.0.0.1:{args.proposer_port}"
        results: list[dict[str, Any]] = []

        def run_family(family: str) -> list[dict[str, Any]]:
            urls = reachable.get(family) or []
            if not urls:
                return [
                    {
                        "task_id": task_id,
                        "error": f"no inner URL for {family}",
                        "arms": {},
                    }
                    for task_id in FAMILIES[family]["task_ids"]
                ]
            family_rows = []
            for task_id in FAMILIES[family]["task_ids"]:
                print(f"=== {family} {task_id} ({len(urls)} inner URL(s)) ===", flush=True)
                try:
                    family_rows.append(
                        run_task(
                            base=base,
                            task_id=task_id,
                            n_urls=len(urls),
                            episode=episode,
                            wait_timeout=wait_timeout,
                        )
                    )
                except Exception as exc:
                    family_rows.append(
                        {
                            "task_id": task_id,
                            "error": str(exc),
                            "arms": {},
                        }
                    )
            return family_rows

        families_to_run = [name for name in FAMILIES if name != "officeqa" and reachable.get(name)]
        missing = [name for name in FAMILIES if name != "officeqa" and not reachable.get(name)]
        for name in missing:
            for task_id in FAMILIES[name]["task_ids"]:
                skipped.append({"task_id": task_id, "reason": f"{name} inner did not come up"})

        with ThreadPoolExecutor(max_workers=min(2, max(1, len(families_to_run)))) as pool:
            futs = {pool.submit(run_family, name): name for name in families_to_run}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    results.extend(fut.result())
                except Exception as exc:
                    for task_id in FAMILIES[name]["task_ids"]:
                        results.append({"task_id": task_id, "error": str(exc), "arms": {}})

        order = [task_id for spec in FAMILIES.values() for task_id in spec["task_ids"]]
        results.sort(key=lambda row: order.index(row["task_id"]) if row.get("task_id") in order else 99)
        comparison = {
            "stamp": stamp,
            "limits": episode,
            "reachable": reachable,
            "skipped": skipped,
            "tasks": results,
        }
        out = logs / "luna_catalog_low_vs_med.json"
        out.write_text(json.dumps(comparison, indent=2) + "\n")
        print_table(results, skipped)
        print(f"wrote {out}", flush=True)
        failed = False
        for row in results:
            arms = row.get("arms") or {}
            if row.get("error"):
                failed = True
            for effort in ("low", "medium"):
                if (arms.get(effort) or {}).get("status") != "completed":
                    failed = True
        return 1 if failed else 0
    finally:
        for child in children:
            child.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
