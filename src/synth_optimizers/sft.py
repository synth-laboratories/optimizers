"""Public SFT control plane executed in-process by the Tinker SFT executor.

The public service owns SFT's stable API, canonical run identity, validation,
and replay-facing endpoints. Training runs locally against the shared Tinker
adapter. Historical Optimizers-beta remains a reference implementation only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .contracts.training_schemas import TERMINAL_STATES
from .recipes.banking77 import fixture_examples
from .runtime import JobStore, JobStoreError, after_sequence_from, wants_live_stream, write_sse
from .sft_executor import SftExecutor, TinkerSftExecutor


SFT_ALGORITHM_ID = "sft"


class SftServiceError(ValueError):
    """An invalid public SFT request or an unavailable executor."""


@dataclass(frozen=True, slots=True)
class SftArtifact:
    """An artifact streamed through the public SFT service."""

    body: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class SftConfig:
    run_id: str
    base_model: str
    backend: str
    checkpoint_steps: tuple[int, ...]
    accelerator_slots: int
    config_json: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        run_id: str | None = None,
    ) -> "SftConfig":
        data = _json_object(value, context="SFT config")
        resolved_run_id = _non_empty_text(run_id or data.get("run_id"), field="run_id")
        base_model = _non_empty_text(
            data.get("base_model") or data.get("model_id") or "openai/gpt-oss-20b",
            field="base_model",
        )
        backend = _non_empty_text(data.get("backend", "tinker"), field="backend")
        if backend not in {"fixture", "tinker"}:
            raise SftServiceError("backend must be fixture or tinker")
        slots = data.get("accelerator_slots", 1)
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise SftServiceError("accelerator_slots must be a positive integer")
        from .contracts.checkpoint_plan import resolve_checkpoint_plan
        from .contracts.training_schemas import SchemaError
        try:
            plan = resolve_checkpoint_plan(data)
        except SchemaError as exc:
            raise SftServiceError(str(exc)) from exc
        raw_steps = plan["save_steps"]
        data["training"] = {**(data.get("training") or {}), "steps": plan["steps"]}
        if backend == "tinker" and not _has_training_data(data):
            raise SftServiceError("Tinker SFT requires training_file_id, training_jsonl, examples, or dataset")
        data["run_id"] = resolved_run_id
        data["base_model"] = base_model
        data["model_id"] = base_model
        data["backend"] = backend
        data["accelerator_slots"] = slots
        if "checkpoint_schedule" not in data:
            data["checkpoint_steps"] = raw_steps
        return cls(
            run_id=resolved_run_id,
            base_model=base_model,
            backend=backend,
            checkpoint_steps=tuple(raw_steps),
            accelerator_slots=slots,
            config_json=data,
        )

    @classmethod
    def from_toml(cls, text: str, *, run_id: str | None = None) -> "SftConfig":
        import tomllib

        try:
            value = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise SftServiceError(f"invalid SFT TOML: {exc}") from exc
        return cls.from_mapping(value, run_id=run_id)


class SftPublicServiceClient:
    """Client for the public local SFT service, suitable for CLI and Workshop."""

    def __init__(
        self, base_url: str, token: str | None = None, *, timeout_seconds: float = 300.0
    ) -> None:
        self.base_url = _non_empty_text(base_url, field="SFT service URL").rstrip("/")
        self.token = _optional_text(token)
        self.timeout_seconds = timeout_seconds

    def submit_toml(
        self,
        config_toml: str,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/runs",
            {
                "algorithm": SFT_ALGORITHM_ID,
                "config_toml": config_toml,
                **({"run_id": run_id} if run_id else {}),
                **({"idempotency_key": idempotency_key} if idempotency_key else {}),
            },
        )

    def get(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/runs/{run_id}")

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{run_id}/cancel", {})

    def pause(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{run_id}/pause", {})

    def resume(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{run_id}/resume", {})

    def estimate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/runs/estimate", {"algorithm": SFT_ALGORITHM_ID, "config_json": dict(config)})

    def optimizer_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"after_sequence": max(0, after_sequence), "limit": max(1, min(5_000, limit))}
        )
        return self._request("GET", f"/v1/runs/{run_id}/optimizer-events?{query}")

    def optimizer_event_stream(
        self, run_id: str, *, after_sequence: int = 0
    ) -> Iterator[dict[str, Any]]:
        query = urllib.parse.urlencode({"after_sequence": max(0, after_sequence)})
        path = f"/v1/runs/{run_id}/optimizer-events/stream?{query}"
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "close",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", method="GET", headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for event in _iter_sse_events(response):
                    yield event
                    if str(event.get("phase") or "") in TERMINAL_STATES:
                        return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SftServiceError(
                f"public SFT service GET {path} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SftServiceError(f"public SFT service GET {path} failed: {exc}") from exc

    def artifact(self, run_id: str, name: str) -> SftArtifact:
        request = urllib.request.Request(
            f"{self.base_url}/v1/runs/{urllib.parse.quote(run_id, safe='')}/artifacts/"
            f"{urllib.parse.quote(name, safe='')}",
            headers={
                "Accept": "application/octet-stream",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return SftArtifact(
                    body=response.read(),
                    content_type=response.headers.get_content_type(),
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SftServiceError(
                f"public SFT artifact {run_id}/{name} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SftServiceError(f"public SFT artifact {run_id}/{name} failed: {exc}") from exc

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SftServiceError(
                f"public SFT service {method} {path} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SftServiceError(f"public SFT service {method} {path} failed: {exc}") from exc
        try:
            decoded = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise SftServiceError(f"public SFT service returned invalid JSON: {exc}") from exc
        return _json_object(decoded, context="public SFT response")


class SftService:
    """Durable public SFT façade with one canonical run ID per submission."""

    def __init__(
        self,
        database_path: str | Path,
        executor: SftExecutor | None = None,
        *,
        fixture: bool = False,
        background: bool = False,
    ) -> None:
        if executor is None:
            self.store = JobStore(database_path)
            self.executor = TinkerSftExecutor.local(
                self.store, fixture=fixture or _use_fixture_executor()
            )
        else:
            self.executor = executor
            store = getattr(executor, "store", None)
            self.store = store if isinstance(store, JobStore) else JobStore(database_path)
        if background:
            self.executor.sync = False

    @classmethod
    def from_env(cls, database_path: str | Path) -> "SftService":
        return cls(database_path, background=True)

    @classmethod
    def from_fixture(cls, database_path: str | Path) -> "SftService":
        return cls(database_path, fixture=True)

    def estimate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        validated = SftConfig.from_mapping(config, run_id=str(config.get("run_id") or "sft_estimate"))
        return self.executor.estimate(_executor_config(validated))

    def submit(
        self,
        config: Mapping[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        requested_run_id = run_id or idempotency_key or _optional_text(config.get("run_id"))
        canonical_run_id = requested_run_id or _fresh_run_id()
        validated = SftConfig.from_mapping(config, run_id=canonical_run_id)
        payload = _executor_config(validated)
        result = self.executor.submit(
            payload,
            job_id=canonical_run_id,
            idempotency_key_override=idempotency_key,
        )
        return self._submit_response(canonical_run_id, str(result.get("status") or "queued"))

    def submit_toml(
        self,
        config_toml: str,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        requested_run_id = run_id or idempotency_key
        config = SftConfig.from_toml(config_toml, run_id=requested_run_id)
        return self.submit(
            config.config_json, run_id=config.run_id, idempotency_key=idempotency_key
        )

    def get(self, run_id: str) -> dict[str, Any]:
        return self._public_run(self.executor.status(run_id))

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._public_run(self.executor.cancel(run_id))

    def pause(self, run_id: str) -> dict[str, Any]:
        return self._public_run(self.executor.pause(run_id))

    def resume(self, run_id: str) -> dict[str, Any]:
        return self._public_run(self.executor.resume(run_id))

    def optimizer_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        from .runtime.workshop import optimizer_event_page

        return optimizer_event_page(
            self.store, run_id, after_sequence=after_sequence, limit=limit
        )

    def state_batch(self, run_id: str, slices: str) -> dict[str, Any]:
        from .runtime.workshop import state_batch

        return state_batch(self.store, run_id, slices, algorithm_id=SFT_ALGORITHM_ID)

    def artifact(self, run_id: str, name: str) -> SftArtifact:
        body, content_type, _digest = self.store.artifact(run_id, name)
        return SftArtifact(body=body, content_type=content_type)

    def _public_run(self, remote: Mapping[str, Any]) -> dict[str, Any]:
        run_id = str(remote.get("run_id") or remote.get("job_id"))
        status = str(remote.get("status") or "queued")
        response = self._submit_response(run_id, status)
        if remote.get("error"):
            response["error"] = remote["error"]
        result = remote.get("result")
        if isinstance(result, Mapping):
            public_result = {
                field: result[field]
                for field in ("best_candidate", "cost_usd", "usage")
                if field in result
            }
            if public_result:
                response["result"] = public_result
        return response

    @staticmethod
    def _submit_response(run_id: str, status: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "algorithm": SFT_ALGORITHM_ID,
            "status": status,
            "events_url": f"/v1/runs/{run_id}/optimizer-events",
            "events_stream_url": f"/v1/runs/{run_id}/optimizer-events/stream",
            "status_url": f"/v1/runs/{run_id}",
            "artifact_base_url": f"/v1/runs/{run_id}/artifacts",
        }


def create_sft_http_server(
    bind: tuple[str, int],
    service: SftService,
    *,
    service_token: str | None = None,
) -> ThreadingHTTPServer:
    token = _optional_text(service_token)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._dispatch()

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch()

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _dispatch(self) -> None:
            try:
                if token and self.headers.get("Authorization") != f"Bearer {token}":
                    self._write(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                    return
                parsed = urllib.parse.urlsplit(self.path)
                parts = [part for part in parsed.path.split("/") if part]
                query = urllib.parse.parse_qs(parsed.query)
                if self.command == "GET" and parsed.path == "/health":
                    self._write(HTTPStatus.OK, {"status": "ok", "algorithm": SFT_ALGORITHM_ID})
                elif self.command == "POST" and parts == ["v1", "runs", "estimate"]:
                    payload = self._body()
                    self._write(HTTPStatus.OK, service.estimate(_mapping(payload.get("config_json"), context="config_json")))
                elif self.command == "POST" and parts == ["v1", "runs"]:
                    payload = self._body()
                    if payload.get("algorithm", SFT_ALGORITHM_ID) != SFT_ALGORITHM_ID:
                        raise SftServiceError("public SFT service accepts algorithm=sft only")
                    run = (
                        service.submit_toml(
                            _non_empty_text(payload.get("config_toml"), field="config_toml"),
                            run_id=_optional_text(payload.get("run_id")),
                            idempotency_key=_optional_text(payload.get("idempotency_key")),
                        )
                        if payload.get("config_toml") is not None
                        else service.submit(
                            _mapping(payload.get("config_json"), context="config_json"),
                            run_id=_optional_text(payload.get("run_id")),
                            idempotency_key=_optional_text(payload.get("idempotency_key")),
                        )
                    )
                    self._write(HTTPStatus.OK, run)
                elif len(parts) >= 3 and parts[:2] == ["v1", "runs"]:
                    run_id = parts[2]
                    if self.command == "GET" and len(parts) == 3:
                        self._write(HTTPStatus.OK, service.get(run_id))
                    elif self.command == "POST" and parts[3:] == ["cancel"]:
                        self._write(HTTPStatus.OK, service.cancel(run_id))
                    elif self.command == "POST" and parts[3:] == ["pause"]:
                        self._write(HTTPStatus.OK, service.pause(run_id))
                    elif self.command == "POST" and parts[3:] == ["resume"]:
                        self._write(HTTPStatus.OK, service.resume(run_id))
                    elif self.command == "GET" and parts[3:] in (
                        ["optimizer-events"],
                        ["optimizer-events", "stream"],
                    ):
                        try:
                            service.store.require(run_id)
                        except JobStoreError as exc:
                            self._write(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                            return
                        if wants_live_stream(parsed.path, query):
                            write_sse(
                                self,
                                service.store,
                                run_id,
                                after_sequence=after_sequence_from(query),
                            )
                            return
                        self._write(
                            HTTPStatus.OK,
                            service.optimizer_events(
                                run_id,
                                after_sequence=after_sequence_from(query),
                                limit=_query_int(query, "limit", default=500),
                            ),
                        )
                    elif self.command == "GET" and parts[3:] == ["state", "batch"]:
                        self._write(
                            HTTPStatus.OK,
                            service.state_batch(run_id, ",".join(query.get("slices", []))),
                        )
                    elif self.command == "GET" and len(parts) == 5 and parts[3] == "artifacts":
                        self._write_artifact(service.artifact(run_id, parts[4]))
                    else:
                        self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                else:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except SftServiceError as exc:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except JobStoreError as exc:
                self._write(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - final HTTP boundary
                self._write(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                value = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                raise SftServiceError(f"invalid JSON request body: {exc}") from exc
            return _json_object(value, context="SFT service request")

        def _write(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
            body = json.dumps(value, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_artifact(self, artifact: SftArtifact) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", artifact.content_type)
            self.send_header("Content-Length", str(len(artifact.body)))
            self.end_headers()
            self.wfile.write(artifact.body)

    return ThreadingHTTPServer(bind, Handler)


def serve_sft_service(
    database_path: str | Path,
    bind: str,
    *,
    service_token: str | None = None,
) -> None:
    host, port = _parse_bind(bind)
    server = create_sft_http_server(
        (host, port),
        SftService.from_env(database_path),
        service_token=service_token or os.environ.get("SYNTH_OPTIMIZERS_SFT_SERVICE_TOKEN"),
    )
    server.serve_forever()


def _executor_config(config: SftConfig) -> dict[str, Any]:
    payload = dict(config.config_json)
    if not _has_training_data(payload):
        payload["examples"] = fixture_examples()
        payload.setdefault(
            "dataset",
            {
                "examples": payload["examples"],
                "train_indexes": [0, 1, 2, 3],
                "calibration_indexes": [4],
                "heldout_indexes": [5],
            },
        )
    payload.setdefault("training", {})
    payload["training"].setdefault("steps", max(config.checkpoint_steps))
    payload["training"].setdefault("checkpoint_every_steps", min(config.checkpoint_steps))
    payload["training"].setdefault("batch_size", 1)
    return payload


def _has_training_data(data: Mapping[str, Any]) -> bool:
    dataset = data.get("dataset")
    return bool(
        _optional_text(data.get("training_file_id"))
        or _optional_text(data.get("training_jsonl"))
        or isinstance(data.get("examples"), list)
        and data.get("examples")
        or isinstance(dataset, Mapping)
        and (dataset.get("examples") or dataset.get("recipe_id"))
    )


def _use_fixture_executor() -> bool:
    return os.environ.get("SYNTH_OPTIMIZERS_SFT_FIXTURE", "").strip() == "1"


def _fresh_run_id() -> str:
    import uuid

    return f"sft_{uuid.uuid4().hex}"


def _parse_bind(bind: str) -> tuple[str, int]:
    host, separator, raw_port = bind.rpartition(":")
    if not separator or not host:
        raise SftServiceError("bind must be HOST:PORT")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise SftServiceError("bind port must be an integer") from exc
    if not 0 < port < 65536:
        raise SftServiceError("bind port must be in 1..65535")
    return host, port


def _iter_sse_events(response: Any) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    event_name = ""
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            event = _sse_event(data_lines, event_name)
            data_lines = []
            event_name = ""
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("id:"):
            continue
    event = _sse_event(data_lines, event_name)
    if event is not None:
        yield event


def _sse_event(data_lines: list[str], event_name: str) -> dict[str, Any] | None:
    if not data_lines:
        return None
    if event_name and event_name != "optimizer":
        return None
    payload = "\n".join(data_lines)
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SftServiceError(f"public SFT SSE returned invalid JSON: {exc}") from exc
    return _json_object(decoded, context="public SFT SSE event")


def _query_int(query: Mapping[str, list[str]], key: str, *, default: int) -> int:
    try:
        return int(query.get(key, [str(default)])[0])
    except ValueError:
        return default


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SftServiceError(f"{context} must be an object")
    encoded = json.dumps(value)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise SftServiceError(f"{context} must be an object")
    return decoded


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SftServiceError(f"{context} must be an object")
    return value


def _non_empty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SftServiceError(f"{field} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
