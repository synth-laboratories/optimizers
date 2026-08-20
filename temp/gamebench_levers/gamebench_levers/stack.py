"""Boot env + policy service + GEPA orchestrator for one (game, mode).

The policy is a real subprocess so `harness_restart.v1` can kill and respawn it
while the env process keeps running. The env is also its own process because each
GameBench task dir ships a distinct `gold_python` package.
"""

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
from typing import Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI

from gamebench_levers.orchestrator_app import create_app as create_orch
from gamebench_levers.seeds import harness_seed

Mode = Literal["code", "harness"]
ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=0.5)
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


def _spawn(
    module: str,
    args: list[str],
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    # Keep stderr: a policy process that dies at boot is undiagnosable otherwise.
    sink = open(log_path, "ab") if log_path is not None else subprocess.DEVNULL  # noqa: SIM115
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", module, *args],
        cwd=str(ROOT), env=env,
        stdout=sink, stderr=sink,
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


def _serve(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server.install_signal_handlers = False
    threading.Thread(target=server.run, daemon=True).start()
    _wait_http(f"http://{host}:{port}/health")
    return server


class PolicyProcess:
    def __init__(
        self, *, game: str, mode: Mode, host: str, port: int,
        script_path: Path | None, log_path: Path | None = None,
    ) -> None:
        self.game, self.mode, self.host, self.port = game, mode, host, port
        self.script_path = script_path
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def tail(self, limit: int = 1200) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-limit:]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> dict[str, Any]:
        args = ["--game", self.game, "--host", self.host, "--port", str(self.port)]
        extra: dict[str, str] = {}
        if self.mode == "harness":
            module = "gamebench_levers.policy_harness_app"
            if self.script_path is not None:
                args += ["--script-path", str(self.script_path)]
                extra["GAMEBENCH_HARNESS_SCRIPT"] = str(self.script_path)
        else:
            module = "gamebench_levers.policy_code_app"
        self.proc = _spawn(module, args, extra, self.log_path)
        return _wait_http(f"{self.url}/health")

    def stop(self) -> None:
        _stop_proc(self.proc)
        self.proc = None
        _wait_down(f"{self.url}/health")

    def restart(self) -> dict[str, Any]:
        started = time.perf_counter()
        old_pid = None
        try:
            old_pid = httpx.get(f"{self.url}/health", timeout=1.0).json().get("pid")
        except Exception:  # noqa: BLE001
            pass
        self.stop()
        try:
            health = self.start()
        except RuntimeError as exc:
            # The new process never came up. Report it as a failed apply; the caller
            # rolls the tree back and restarts the parent.
            return {
                "restart_ok": False, "compile_ok": False,
                "error": f"{exc}", "stderr": self.tail(),
                "old_pid": old_pid, "new_pid": None,
                "restart_ms": (time.perf_counter() - started) * 1000.0,
            }
        return {
            "restart_ok": health.get("status") == "ok" and bool(health.get("compile_ok", True)),
            "compile_ok": bool(health.get("compile_ok", health.get("status") == "ok")),
            "old_pid": old_pid, "new_pid": health.get("pid"),
            "restart_ms": (time.perf_counter() - started) * 1000.0,
            "public_skills": health.get("public_skills"),
            "compile_error": health.get("compile_error"),
            "stderr": "" if health.get("compile_ok", True) else self.tail(800),
        }


class PolicyPool:
    """One policy worker per candidate (`apply_isolation: per_candidate_worker`).

    With a single worker, GEPA interleaving rollouts across candidates forces a
    process restart on *every* switch -- register-once/run-many degrades back into
    restart-per-rollout. Keyed workers make a switch a routing decision instead.

    Workers are capped and evicted least-recently-used; an evicted candidate is
    respawned from its stored source on next use, so eviction costs latency, never
    correctness.
    """

    def __init__(self, *, game: str, mode: Mode, host: str, root: Path, max_workers: int = 4) -> None:
        self.game, self.mode, self.host, self.root = game, mode, host, root
        self.max_workers = max(1, int(max_workers))
        self.workers: dict[str, PolicyProcess] = {}
        self.sources: dict[str, str] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.restarts = 0
        self.reuses = 0
        self.evictions = 0

    def _touch(self, candidate_id: str) -> None:
        if candidate_id in self.order:
            self.order.remove(candidate_id)
        self.order.append(candidate_id)

    def _evict_if_needed(self, keep: str) -> None:
        while len(self.workers) > self.max_workers:
            victim = next((c for c in self.order if c != keep), None)
            if victim is None:
                return
            self.order.remove(victim)
            worker = self.workers.pop(victim, None)
            if worker is not None:
                worker.stop()
                self.evictions += 1

    def _spawn(self, candidate_id: str, source: str) -> PolicyProcess:
        slug = "".join(ch if ch.isalnum() else "_" for ch in candidate_id)[:40]
        script = self.root / f"harness_{slug}.py"
        script.write_text(source, encoding="utf-8")
        worker = PolicyProcess(
            game=self.game, mode=self.mode, host=self.host, port=_free_port(),
            script_path=script, log_path=self.root / f"policy_{slug}.log",
        )
        worker.start()
        self.restarts += 1
        return worker

    def ensure(self, candidate_id: str, source: str | None) -> dict[str, Any]:
        """Return a healthy worker serving `source` for this candidate."""
        with self.lock:
            text = source if source is not None else self.sources.get(candidate_id)
            if text is None:
                return {"ok": False, "error": f"no source known for {candidate_id}"}
            live = self.workers.get(candidate_id)
            if live is not None and self.sources.get(candidate_id) == text and live.proc and live.proc.poll() is None:
                self._touch(candidate_id)
                self.reuses += 1
                health = httpx.get(f"{live.url}/health", timeout=10.0).json()
                return {
                    "ok": bool(health.get("compile_ok", True)), "reused": True,
                    "url": live.url, "pid": health.get("pid"),
                    "compile_ok": health.get("compile_ok"), "compile_error": health.get("compile_error"),
                }
            if live is not None:
                live.stop()
                self.workers.pop(candidate_id, None)
            try:
                worker = self._spawn(candidate_id, text)
            except RuntimeError as exc:
                return {"ok": False, "reused": False, "error": str(exc)}
            self.workers[candidate_id] = worker
            self.sources[candidate_id] = text
            self._touch(candidate_id)
            self._evict_if_needed(candidate_id)
            health = httpx.get(f"{worker.url}/health", timeout=10.0).json()
            return {
                "ok": bool(health.get("compile_ok", True)), "reused": False,
                "url": worker.url, "pid": health.get("pid"),
                "compile_ok": health.get("compile_ok"), "compile_error": health.get("compile_error"),
                "public_skills": health.get("public_skills"),
            }

    def drop(self, candidate_id: str) -> None:
        with self.lock:
            worker = self.workers.pop(candidate_id, None)
            self.sources.pop(candidate_id, None)
            if candidate_id in self.order:
                self.order.remove(candidate_id)
            if worker is not None:
                worker.stop()

    def stats(self) -> dict[str, Any]:
        return {
            "workers": {cid: w.url for cid, w in self.workers.items()},
            "max_workers": self.max_workers,
            "spawns": self.restarts, "reuses": self.reuses, "evictions": self.evictions,
        }

    def stop(self) -> None:
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()
        self.order.clear()


def _control_app(policy: PolicyProcess, env_url: str, pool: PolicyPool | None = None) -> FastAPI:
    app = FastAPI(title="gamebench-supervisor")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "role": "supervisor"}

    @app.get("/workers")
    def list_workers() -> dict[str, Any]:
        return pool.stats() if pool is not None else {"workers": {}, "max_workers": 0}

    @app.post("/workers")
    def ensure_worker(body: dict[str, Any]) -> dict[str, Any]:
        if pool is None:
            return {"ok": False, "error": "pool disabled"}
        env_before = httpx.get(f"{env_url}/health", timeout=5.0).json()
        result = pool.ensure(str(body.get("candidate_id") or ""), body.get("source"))
        env_after = httpx.get(f"{env_url}/health", timeout=5.0).json()
        result["env_untouched"] = env_before.get("pid") == env_after.get("pid")
        result["env_pid"] = env_after.get("pid")
        return result

    @app.delete("/workers/{candidate_id}")
    def drop_worker(candidate_id: str) -> dict[str, Any]:
        if pool is not None:
            pool.drop(candidate_id)
        return {"ok": True}

    @app.post("/restart_policy")
    def restart_policy() -> dict[str, Any]:
        env_before = httpx.get(f"{env_url}/health", timeout=5.0).json()
        try:
            result = policy.restart()
        except Exception as exc:  # noqa: BLE001
            return {"restart_ok": False, "compile_ok": False, "error": str(exc), "stderr": policy.tail()}
        env_after = httpx.get(f"{env_url}/health", timeout=5.0).json()
        result["env_untouched"] = env_before.get("pid") == env_after.get("pid")
        result["env_pid"] = env_after.get("pid")
        return result

    return app


@dataclass
class Stack:
    game: str
    mode: Mode
    env_url: str
    policy_url: str
    orch_url: str
    control_url: str | None
    spec: dict[str, Any]
    servers: list[uvicorn.Server] = field(default_factory=list)
    procs: list[subprocess.Popen] = field(default_factory=list)
    policy: PolicyProcess | None = None
    pool: PolicyPool | None = None
    isolation: str = "serial_restart"
    tmpdir: Any = None

    def stop(self) -> None:
        for server in self.servers:
            server.should_exit = True
        if self.pool is not None:
            self.pool.stop()
        if self.policy is not None:
            self.policy.stop()
        for proc in self.procs:
            _stop_proc(proc)
        self.procs.clear()
        if self.tmpdir is not None:
            self.tmpdir.cleanup()
            self.tmpdir = None


def start_stack(
    game: str,
    mode: Mode = "code",
    *,
    host: str = "127.0.0.1",
    env_port: int | None = None,
    policy_port: int | None = None,
    orch_port: int | None = None,
    control_port: int | None = None,
    max_steps: int | None = None,
    isolation: str = "per_candidate_worker",
    max_workers: int = 4,
) -> Stack:
    env_port = env_port or _free_port()
    policy_port = policy_port or _free_port()
    orch_port = orch_port or _free_port()
    control_port = control_port or _free_port()
    env_url = f"http://{host}:{env_port}"

    tmpdir = tempfile.TemporaryDirectory(prefix=f"gb_{game}_{mode}_")
    script_path: Path | None = None
    if mode == "harness":
        script_path = Path(tmpdir.name) / "harness.py"
        script_path.write_text(harness_seed(game), encoding="utf-8")

    env_proc = _spawn("gamebench_levers.env_app", ["--game", game, "--host", host, "--port", str(env_port)])
    _wait_http(f"{env_url}/health")
    spec = httpx.get(f"{env_url}/spec", timeout=10.0).json()

    policy = PolicyProcess(
        game=game, mode=mode, host=host, port=policy_port,
        script_path=script_path, log_path=Path(tmpdir.name) / "policy.log",
    )
    policy.start()

    # Per-candidate workers only matter for the harness lane: the code lane reloads
    # in-process, which costs an import rather than a process.
    pool: PolicyPool | None = None
    if mode == "harness" and isolation == "per_candidate_worker":
        pool = PolicyPool(
            game=game, mode=mode, host=host,
            root=Path(tmpdir.name), max_workers=max_workers,
        )
        # the already-running process serves the seed
        pool.workers["seed"] = policy
        pool.sources["seed"] = harness_seed(game)
        pool.order.append("seed")

    control_url = f"http://{host}:{control_port}"
    control = _serve(_control_app(policy, env_url, pool), host, control_port)
    orch = _serve(
        create_orch(
            game, mode, env_url, policy.url,
            control_url=control_url,
            script_path=str(script_path) if script_path else None,
            train_seeds=tuple(spec.get("train_seeds") or ()),
            heldout_seeds=tuple(spec.get("heldout_seeds") or ()),
            max_steps=max_steps,
            isolation="per_candidate_worker" if pool is not None else "serial_restart",
        ),
        host, orch_port,
    )
    return Stack(
        game=game, mode=mode, env_url=env_url, policy_url=policy.url,
        orch_url=f"http://{host}:{orch_port}", control_url=control_url, spec=spec,
        servers=[control, orch], procs=[env_proc], policy=policy, pool=pool,
        isolation="per_candidate_worker" if pool is not None else "serial_restart",
        tmpdir=tmpdir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GameBench lever stack")
    parser.add_argument("--game", required=True)
    parser.add_argument("--mode", default="code", choices=["code", "harness"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--isolation", default="per_candidate_worker",
                        choices=["per_candidate_worker", "serial_restart"])
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    stack = start_stack(
        args.game, args.mode, host=args.host, max_steps=args.max_steps,
        isolation=args.isolation, max_workers=args.max_workers,
    )
    print(f"game={stack.game} mode={stack.mode}")
    print(f"env      {stack.env_url}")
    print(f"policy   {stack.policy_url}")
    print(f"control  {stack.control_url}  POST /workers | POST /restart_policy")
    print(f"isolation {stack.isolation}")
    print(f"gepa     {stack.orch_url}")
    print(f"program  {stack.orch_url}/program")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stack.stop()


if __name__ == "__main__":
    main()
