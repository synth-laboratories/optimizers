"""Public SFT control plane backed by an internal Optimizers-beta executor.

The public service owns SFT's stable API, canonical run identity, validation, and
replay-facing endpoints.  The beta service is deliberately an executor: it receives
only validated jobs and is not a Workshop-facing control plane.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol


SFT_ALGORITHM_ID = "sft"


class SftServiceError(ValueError):
    """An invalid public SFT request or an unavailable executor."""


@dataclass(frozen=True, slots=True)
class SftArtifact:
    """An artifact streamed through the public SFT service."""

    body: bytes
    content_type: str


class SftExecutor(Protocol):
    def request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]: ...


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
            data.get("base_model", "openai/gpt-oss-20b"), field="base_model"
        )
        backend = _non_empty_text(data.get("backend", "tinker"), field="backend")
        if backend not in {"fixture", "tinker"}:
            raise SftServiceError("backend must be fixture or tinker")
        slots = data.get("accelerator_slots", 1)
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise SftServiceError("accelerator_slots must be a positive integer")
        raw_steps = data.get("checkpoint_steps", [10, 20])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise SftServiceError("checkpoint_steps must be a non-empty list")
        if any(
            not isinstance(step, int) or isinstance(step, bool) or step < 1 for step in raw_steps
        ):
            raise SftServiceError("checkpoint_steps must contain positive integers")
        if sorted(raw_steps) != raw_steps or len(set(raw_steps)) != len(raw_steps):
            raise SftServiceError("checkpoint_steps must be strictly increasing")
        if backend == "tinker" and not (
            _optional_text(data.get("training_file_id"))
            or _optional_text(data.get("training_jsonl"))
        ):
            raise SftServiceError("Tinker SFT requires training_file_id or training_jsonl")
        data["run_id"] = resolved_run_id
        data["base_model"] = base_model
        data["backend"] = backend
        data["accelerator_slots"] = slots
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
        try:
            value = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise SftServiceError(f"invalid SFT TOML: {exc}") from exc
        return cls.from_mapping(value, run_id=run_id)


class BetaSftExecutorClient:
    """Authenticated internal client for the Optimizers-beta SFT executor."""

    def __init__(self, base_url: str, token: str, *, timeout_seconds: float = 300.0) -> None:
        self.base_url = _non_empty_text(base_url, field="beta base URL").rstrip("/")
        self.token = _non_empty_text(token, field="beta service token")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "BetaSftExecutorClient":
        return cls(
            os.environ.get("SYNTH_OPTIMIZERS_BETA_URL")
            or os.environ.get("OPTIMIZERS_BETA_URL")
            or "http://127.0.0.1:8879",
            os.environ.get("OPTIMIZERS_BETA_SERVICE_TOKEN", ""),
        )

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SftServiceError(
                f"beta SFT executor {method} {path} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SftServiceError(f"beta SFT executor {method} {path} failed: {exc}") from exc
        try:
            decoded = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise SftServiceError(f"beta SFT executor returned invalid JSON: {exc}") from exc
        return _json_object(decoded, context="beta SFT response")

    def artifact(self, run_id: str, name: str) -> SftArtifact:
        request = urllib.request.Request(
            f"{self.base_url}/v1/runs/{urllib.parse.quote(run_id, safe='')}/artifacts/"
            f"{urllib.parse.quote(name, safe='')}",
            headers={"Authorization": f"Bearer {self.token}"},
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
                f"beta SFT artifact {run_id}/{name} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SftServiceError(f"beta SFT artifact {run_id}/{name} failed: {exc}") from exc


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

    def optimizer_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"after_sequence": max(0, after_sequence), "limit": max(1, min(5_000, limit))}
        )
        return self._request("GET", f"/v1/runs/{run_id}/optimizer-events?{query}")

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

    def __init__(self, database_path: str | Path, executor: SftExecutor) -> None:
        database = Path(database_path)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(database)
        self.executor = executor
        self._db = sqlite3.connect(self.database_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sft_public_runs (
                run_id TEXT PRIMARY KEY,
                beta_run_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT
            )
            """
        )
        self._db.commit()

    @classmethod
    def from_env(cls, database_path: str | Path) -> "SftService":
        return cls(database_path, BetaSftExecutorClient.from_env())

    def submit(
        self,
        config: Mapping[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            requested_run_id = run_id or idempotency_key or _optional_text(config.get("run_id"))
            canonical_run_id = requested_run_id or f"sft_{uuid.uuid4().hex}"
            validated = SftConfig.from_mapping(config, run_id=canonical_run_id)
            existing = self._lookup(canonical_run_id)
            if existing is not None:
                if existing["config_json"] != _compact_json(validated.config_json):
                    raise SftServiceError(
                        f"idempotency key {canonical_run_id!r} was already submitted with a different SFT config"
                    )
                return self._submit_response(existing["run_id"], existing["status"])

            response = self.executor.request(
                "POST",
                "/v1/runs",
                {
                    "algorithm": SFT_ALGORITHM_ID,
                    "idempotency_key": canonical_run_id,
                    "config_json": validated.config_json,
                },
            )
            beta_run_id = _non_empty_text(response.get("run_id"), field="beta run_id")
            status = _non_empty_text(response.get("status", "queued"), field="beta status")
            now = _now()
            self._db.execute(
                """
                INSERT INTO sft_public_runs(run_id, beta_run_id, config_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_run_id,
                    beta_run_id,
                    _compact_json(validated.config_json),
                    status,
                    now,
                    now,
                ),
            )
            self._db.commit()
            return self._submit_response(canonical_run_id, status)

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
        with self._lock:
            record = self._require(run_id)
            remote = self.executor.request("GET", f"/v1/runs/{record['beta_run_id']}")
            status = _non_empty_text(remote.get("status", record["status"]), field="beta status")
            error = _optional_text(remote.get("error"))
            self._update_status(run_id, status, error)
            return self._public_run(record, remote, status, error)

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(run_id)
            remote = self.executor.request("POST", f"/v1/runs/{record['beta_run_id']}/cancel", {})
            status = _non_empty_text(remote.get("status", "cancelled"), field="beta status")
            self._update_status(run_id, status, _optional_text(remote.get("error")))
            return self.get(run_id)

    def infer_openai(self, family: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Proxy Chat Completions / Responses sampling to the beta executor.

        Catalog inference is not a public-run mutation: identity stays on the
        checkpoint envelope, and Workshop never sees the Tinker sampler.
        """
        if family not in {"chat", "responses"}:
            raise SftServiceError("family must be chat or responses")
        path = (
            "/v1/checkpoints/infer/chat/completions"
            if family == "chat"
            else "/v1/checkpoints/infer/responses"
        )
        return self.executor.request("POST", path, dict(payload))

    def optimizer_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require(run_id)
            query = urllib.parse.urlencode(
                {"after_sequence": max(0, after_sequence), "limit": max(1, min(5_000, limit))}
            )
            remote = self.executor.request(
                "GET", f"/v1/runs/{record['beta_run_id']}/optimizer-events?{query}"
            )
            remote["run_id"] = run_id
            return remote

    def state_batch(self, run_id: str, slices: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(run_id)
            encoded = urllib.parse.urlencode({"slices": slices})
            remote = self.executor.request(
                "GET", f"/v1/runs/{record['beta_run_id']}/state/batch?{encoded}"
            )
            remote["run_id"] = run_id
            return remote

    def artifact(self, run_id: str, name: str) -> SftArtifact:
        with self._lock:
            record = self._require(run_id)
            artifact = getattr(self.executor, "artifact", None)
            if not callable(artifact):
                raise SftServiceError("public SFT artifact proxy is unavailable")
            return artifact(record["beta_run_id"], name)

    def _lookup(self, run_id: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM sft_public_runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def _require(self, run_id: str) -> sqlite3.Row:
        record = self._lookup(run_id)
        if record is None:
            raise SftServiceError(f"unknown public SFT run {run_id!r}")
        return record

    def _update_status(self, run_id: str, status: str, error: str | None) -> None:
        self._db.execute(
            "UPDATE sft_public_runs SET status = ?, updated_at = ?, error = ? WHERE run_id = ?",
            (status, _now(), error, run_id),
        )
        self._db.commit()

    @staticmethod
    def _public_run(
        record: sqlite3.Row,
        remote: Mapping[str, Any],
        status: str,
        error: str | None,
    ) -> dict[str, Any]:
        response = SftService._submit_response(record["run_id"], status)
        for field in ("created_at", "updated_at"):
            if value := _optional_text(remote.get(field)):
                response[field] = value
        if error:
            response["error"] = error
        if isinstance(remote.get("cancellation_requested"), bool):
            response["cancellation_requested"] = remote["cancellation_requested"]
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
                    elif self.command == "GET" and parts[3:] == ["optimizer-events"]:
                        self._write(
                            HTTPStatus.OK,
                            service.optimizer_events(
                                run_id,
                                after_sequence=_query_int(query, "after_sequence", default=0),
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
                elif self.command == "POST" and parts in (
                    ["v1", "checkpoints", "infer", "chat", "completions"],
                    ["v1", "checkpoints", "infer", "responses"],
                ):
                    payload = self._body()
                    family = "chat" if parts[-1] == "completions" else "responses"
                    body = payload.get("body") if isinstance(payload.get("body"), Mapping) else payload
                    streamed = _streaming(payload)
                    if streamed and isinstance(body, dict):
                        forwarded = dict(payload)
                        inner = dict(body)
                        inner.pop("stream", None)
                        if "body" in forwarded:
                            forwarded["body"] = inner
                        else:
                            forwarded = inner
                        payload = forwarded
                    result = service.infer_openai(family, payload)
                    if streamed:
                        self._write_sse(_openai_family_sse(family, result))
                        return
                    self._write(HTTPStatus.OK, result)
                else:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except SftServiceError as exc:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
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

        def _write_sse(self, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
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


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _non_empty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SftServiceError(f"{field} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _streaming(payload: Mapping[str, Any]) -> bool:
    if payload.get("stream") is True:
        return True
    body = payload.get("body")
    return isinstance(body, Mapping) and body.get("stream") is True


def _openai_family_sse(family: str, payload: Mapping[str, Any]) -> bytes:
    """Wrap a completed Tinker sample as family-native SSE."""

    if family == "responses":
        text = ""
        output = payload.get("output")
        if isinstance(output, list) and output:
            content = output[0].get("content") if isinstance(output[0], Mapping) else None
            if isinstance(content, list) and content and isinstance(content[0], Mapping):
                text = str(content[0].get("text") or "")
        response_id = str(payload.get("id") or "resp_hosted")
        created = payload.get("created_at") or 0
        model = str(payload.get("model") or "hosted-tinker-checkpoint")
        message_id = f"msg_{response_id}"
        skeleton = {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": model,
            "status": "in_progress",
        }
        events = [
            ("response.created", {"type": "response.created", "response": skeleton}),
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": message_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
            ),
        ]
        if text:
            events.append(
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    },
                )
            )
        events.append(
            (
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
            )
        )
        events.append(
            (
                "response.completed",
                {"type": "response.completed", "response": dict(payload)},
            )
        )
        return "".join(
            f"event: {name}\ndata: {json.dumps(body, separators=(',', ':'))}\n\n"
            for name, body in events
        ).encode("utf-8")

    text = ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            text = str(message.get("content") or "")
    completion_id = str(payload.get("id") or "chatcmpl-hosted")
    created = payload.get("created") or 0
    model = str(payload.get("model") or "hosted-tinker-checkpoint")
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    ]
    if text:
        chunks.append(
            {**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
        )
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    body = "".join(f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks)
    return f"{body}data: [DONE]\n\n".encode("utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
