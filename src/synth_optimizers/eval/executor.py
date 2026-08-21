"""Launching one trial in the recipe-pinned target container.

The executor is the only place in `eval` that starts a process, and it will
only ever start the image a trusted recipe pinned by digest. It mounts the
candidate read-only, gives the container a bounded writable `/output`, denies
the network by default, and inherits no credentials: a container gets a policy,
a trial description, and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import EvalContractError, TrialLimits


class ContainerRuntimeError(RuntimeError):
    """The OCI runtime, not the evaluated policy, is what went wrong."""


@dataclass(frozen=True, slots=True)
class TrialRunRequest:
    trial_id: str
    image_reference: str
    input_dir: Path
    policy_dir: Path
    output_dir: Path
    limits: TrialLimits
    network: str
    secrets: Mapping[str, str] = field(default_factory=dict)
    extra_hosts: tuple[str, ...] | list[str] = field(default_factory=tuple)
    workshop_proxy: bool = False


@dataclass(frozen=True, slots=True)
class TrialExecution:
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    started_at: float
    finished_at: float
    stderr_tail: str


class TrialExecutor(Protocol):
    def run(
        self,
        request: TrialRunRequest,
        *,
        on_event: Callable[[dict[str, Any]], None],
        should_cancel: Callable[[], bool],
        heartbeat: Callable[[], None],
    ) -> TrialExecution: ...


class OciTrialExecutor:
    """Runs pinned OCI images through `docker` or `podman`."""

    def __init__(self, runtime: str = "docker") -> None:
        if runtime not in {"docker", "podman"}:
            raise EvalContractError("container runtime must be docker or podman")
        self.runtime_name = runtime
        binary = shutil.which(runtime)
        if binary is None:
            raise ContainerRuntimeError(
                f"{runtime} is not on PATH; install it or change container_runtime "
                f"in the eval home's runtime.toml"
            )
        self.binary = binary

    def _env(self) -> dict[str, str]:
        """A minimal env for the CLI itself. Nothing here reaches the container."""

        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
        for name in ("HOME", "DOCKER_HOST", "DOCKER_CONFIG", "XDG_RUNTIME_DIR"):
            value = os.environ.get(name)
            if value:
                env[name] = value
        return env

    def image_digests(self, image: str) -> tuple[str | None, tuple[str, ...]]:
        """Return the local image id and any repository digests it carries."""

        completed = subprocess.run(  # noqa: S603 - fixed binary, fixed argv
            [self.binary, "image", "inspect", "--format", "{{json .}}", image],
            capture_output=True,
            text=True,
            env=self._env(),
            check=False,
        )
        if completed.returncode != 0:
            return None, ()
        payload = json.loads(completed.stdout or "{}")
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        repo_digests = tuple(
            entry.split("@", 1)[1] for entry in payload.get("RepoDigests", []) or [] if "@" in entry
        )
        return payload.get("Id"), repo_digests

    def resolve_reference(self, image: str, digest: str) -> str:
        """Verify the pin and return the reference that runs exactly it.

        A published target is addressed by repository digest. A locally built
        one has no repository digest yet, so its image id is the pin. Either
        way the runner never launches a bare tag, which could be re-pointed
        between two trials of the same run.
        """

        image_id, repo_digests = self.image_digests(image)
        if image_id is None:
            raise ContainerRuntimeError(
                f"target image {image} is not present locally; pull or build it first"
            )
        if digest in repo_digests:
            return f"{image}@{digest}"
        if digest == image_id:
            return digest
        raise ContainerRuntimeError(
            f"target image {image} resolves to {image_id}, which does not match the "
            f"pinned digest {digest}"
        )

    def run(
        self,
        request: TrialRunRequest,
        *,
        on_event: Callable[[dict[str, Any]], None],
        should_cancel: Callable[[], bool],
        heartbeat: Callable[[], None],
    ) -> TrialExecution:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        # Truncating a trial id collides: two trials of the same candidate
        # differ only in their tail. Keep a readable prefix, then a digest of
        # the whole id so the name is unique as well as short.
        fingerprint = hashlib.sha256(request.trial_id.encode("utf-8")).hexdigest()[:12]
        container = f"synth-eval-{request.trial_id[:40]}-{fingerprint}"
        argv = [
            self.binary,
            "run",
            "--rm",
            "--name",
            container,
            "--network",
            "none" if request.network == "none" else "bridge",
            "--cpus",
            str(request.limits.cpus),
            "--memory",
            f"{request.limits.memory_mb}m",
            "--pids-limit",
            "512",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=bind,source={request.input_dir},target=/input,readonly",
            "--mount",
            f"type=bind,source={request.policy_dir},target=/input/policy,readonly",
            "--mount",
            f"type=bind,source={request.output_dir},target=/output",
        ]
        for name, value in request.secrets.items():
            if request.workshop_proxy and name == "WORKSHOP_CAPABILITY":
                continue
            argv.extend(["--env", f"{name}={value}"])
        for mapping in request.extra_hosts:
            argv.extend(["--add-host", str(mapping)])
        argv.append(request.image_reference)

        stderr_path = request.output_dir / "container.stderr.log"
        started = time.time()
        with stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(  # noqa: S603 - fixed binary, recipe-pinned argv
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle,
                env=self._env(),
            )
            stop_tailing = threading.Event()
            tail = threading.Thread(
                target=_tail_events,
                args=(request.output_dir / "events.jsonl", on_event, stop_tailing),
                daemon=True,
            )
            tail.start()
            timed_out = False
            cancelled = False
            deadline = started + request.limits.timeout_seconds
            try:
                while True:
                    try:
                        process.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    heartbeat()
                    if should_cancel():
                        cancelled = True
                        self._kill(container, process)
                        break
                    if time.time() > deadline:
                        timed_out = True
                        self._kill(container, process)
                        break
            finally:
                stop_tailing.set()
                tail.join(timeout=2.0)
        finished = time.time()
        return TrialExecution(
            exit_code=process.returncode,
            timed_out=timed_out,
            cancelled=cancelled,
            started_at=started,
            finished_at=finished,
            stderr_tail=_tail_text(stderr_path),
        )

    def _kill(self, container: str, process: subprocess.Popen[bytes]) -> None:
        subprocess.run(  # noqa: S603 - fixed binary, fixed argv
            [self.binary, "kill", container],
            capture_output=True,
            env=self._env(),
            check=False,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _tail_events(
    path: Path, on_event: Callable[[dict[str, Any]], None], stop: threading.Event
) -> None:
    """Follow the container's optional live event stream while it runs."""

    offset = 0
    while True:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for line in handle:
                        if not line.endswith("\n"):
                            break
                        offset += len(line.encode("utf-8"))
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict):
                            on_event(payload)
            except OSError:
                pass
        if stop.wait(0.4):
            # Drain whatever landed between the last read and the container exit.
            if not path.is_file():
                return
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        on_event(payload)
            return


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")
