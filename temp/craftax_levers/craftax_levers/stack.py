"""Start env + policy service + GEPA orchestrator. Policy is a real subprocess so restart_policy is safe."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import uvicorn
from fastapi import FastAPI

from craftax_levers.orchestrator_app import create_app as create_orch
from craftax_levers.seeds import SEED_HARNESS, SEED_POLICY

Mode = Literal["code", "react"]
ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, timeout: float = 8.0) -> dict:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=0.3)
            if response.status_code < 500:
                return response.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.05)
    raise RuntimeError(f"server at {url} did not become healthy: {last}")


def _wait_down(url: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=0.15)
        except Exception:  # noqa: BLE001
            return
        time.sleep(0.05)


def _spawn(module: str, args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", module, *args],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _serve(app, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server.install_signal_handlers = False
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http(f"http://{host}:{port}/health")
    return server


class PolicyProcess:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        mode: Mode,
        script_path: Path | None,
        env_url: str,
    ) -> None:
        self.host = host
        self.port = port
        self.mode = mode
        self.script_path = script_path
        self.env_url = env_url
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> dict:
        extra = {"CRAFTAX_ENV_URL": self.env_url}
        args = ["--host", self.host, "--port", str(self.port)]
        if self.mode == "react":
            module = "craftax_levers.policy_react_app"
            if self.script_path is not None:
                args.extend(["--script-path", str(self.script_path)])
                extra["CRAFTAX_REACT_SCRIPT"] = str(self.script_path)
        else:
            module = "craftax_levers.policy_code_app"
        self.proc = _spawn(module, args, extra)
        return _wait_http(f"{self.url}/health")

    def stop(self) -> None:
        _stop_proc(self.proc)
        self.proc = None
        _wait_down(f"{self.url}/health")

    def restart(self) -> dict:
        started = time.perf_counter()
        old_pid = None
        try:
            old_pid = httpx.get(f"{self.url}/health", timeout=1.0).json().get("pid")
        except Exception:  # noqa: BLE001
            pass
        self.stop()
        health = self.start()
        return {
            "restart_ok": health.get("status") == "ok" and bool(health.get("compile_ok", True)),
            "compile_ok": bool(health.get("compile_ok", health.get("status") == "ok")),
            "old_pid": old_pid,
            "new_pid": health.get("pid"),
            "restart_ms": (time.perf_counter() - started) * 1000.0,
            "entrypoint": health.get("entrypoint"),
        }


def _control_app(policy: PolicyProcess, env_url: str) -> FastAPI:
    app = FastAPI(title="craftax-supervisor")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "role": "supervisor"}

    @app.post("/restart_policy")
    def restart_policy() -> dict:
        env_before = httpx.get(f"{env_url}/health", timeout=2.0).json()
        result = policy.restart()
        env_after = httpx.get(f"{env_url}/health", timeout=2.0).json()
        result["env_untouched"] = env_before.get("pid") == env_after.get("pid")
        result["env_pid"] = env_after.get("pid")
        return result

    return app


@dataclass
class Stack:
    mode: Mode
    env_url: str
    policy_url: str
    orch_url: str
    control_url: str | None
    script_path: Path | None
    servers: list[uvicorn.Server] = field(default_factory=list)
    procs: list[subprocess.Popen] = field(default_factory=list)
    policy: PolicyProcess | None = None
    tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def stop(self) -> None:
        for server in self.servers:
            server.should_exit = True
        if self.policy is not None:
            self.policy.stop()
        for proc in self.procs:
            _stop_proc(proc)
        self.procs.clear()
        if self.tmpdir is not None:
            self.tmpdir.cleanup()
            self.tmpdir = None


def start_stack(
    mode: Mode = "code",
    *,
    host: str = "127.0.0.1",
    env_port: int | None = None,
    policy_port: int | None = None,
    orch_port: int | None = None,
    control_port: int | None = None,
) -> Stack:
    env_port = env_port or _free_port()
    policy_port = policy_port or _free_port()
    orch_port = orch_port or _free_port()
    control_port = control_port or _free_port()
    env_url = f"http://{host}:{env_port}"
    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    script_path: Path | None = None
    if mode == "react":
        tmpdir = tempfile.TemporaryDirectory(prefix="craftax_react_")
        script_path = Path(tmpdir.name) / "react_loop.py"
        script_path.write_text(SEED_HARNESS, encoding="utf-8")
    elif mode == "code":
        tmpdir = tempfile.TemporaryDirectory(prefix="craftax_code_")
        script_path = Path(tmpdir.name) / "policy.py"
        script_path.write_text(SEED_POLICY, encoding="utf-8")

    env_proc = _spawn("craftax_levers.env_app", ["--host", host, "--port", str(env_port)])
    _wait_http(f"{env_url}/health")

    policy = PolicyProcess(
        host=host,
        port=policy_port,
        mode=mode,
        script_path=script_path,
        env_url=env_url,
    )
    policy.start()

    control_url = f"http://{host}:{control_port}"
    control = _serve(_control_app(policy, env_url), host, control_port)
    orch = _serve(
        create_orch(
            mode,
            env_url,
            policy.url,
            control_url=control_url,
            script_path=str(script_path) if script_path else None,
        ),
        host,
        orch_port,
    )
    return Stack(
        mode=mode,
        env_url=env_url,
        policy_url=policy.url,
        orch_url=f"http://{host}:{orch_port}",
        control_url=control_url,
        script_path=script_path,
        servers=[control, orch],
        procs=[env_proc],
        policy=policy,
        tmpdir=tmpdir,
    )


def _run_forever(mode: Mode, args: argparse.Namespace) -> None:
    stack = start_stack(
        mode,
        host=args.host,
        env_port=args.env_port,
        policy_port=args.policy_port,
        orch_port=args.orch_port,
    )
    print(f"mode={mode}")
    print(f"env      {stack.env_url}")
    print(f"policy   {stack.policy_url}")
    print(f"control  {stack.control_url}  POST /restart_policy")
    print(f"gepa     {stack.orch_url}")
    print(f"program  {stack.orch_url}/program")
    print(f"rollout  POST {stack.orch_url}/rollout")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stack.stop()


def main_code() -> None:
    parser = argparse.ArgumentParser(description="Run Craftax code-policy lever stack")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=19101)
    parser.add_argument("--policy-port", type=int, default=19102)
    parser.add_argument("--orch-port", type=int, default=19100)
    _run_forever("code", parser.parse_args())


def main_react() -> None:
    parser = argparse.ArgumentParser(description="Run Craftax ReAct policy-service lever stack")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=19201)
    parser.add_argument("--policy-port", type=int, default=19202)
    parser.add_argument("--orch-port", type=int, default=19200)
    _run_forever("react", parser.parse_args())
