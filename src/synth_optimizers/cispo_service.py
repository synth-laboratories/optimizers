"""Public CISPO control plane executed in-process by the Tinker CISPO executor.

The sqlite journal is the record. SSE is a mirror. HTTP submit returns a run
id immediately so a client can tail events while the job runs. This service
accepts true CISPO only (``algorithm_id="cispo"``, ``slime-reference`` /
``cispo.slime.v1``). Standalone SFT belongs on ``SftService``.
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

from .cispo_executor import TinkerCispoExecutor
from .contracts.training_schemas import (
    CISPO_ALGORITHM_ID,
    SchemaError,
    TERMINAL_STATES,
    validate_cispo_request,
)
from .recipes.banking77 import fixture_examples
from .runtime import JobStore, JobStoreError, after_sequence_from, wants_live_stream, write_sse
from .rl.experiment import CoordinationError


class CispoServiceError(ValueError):
    """An invalid public CISPO request or an unavailable executor."""


@dataclass(frozen=True, slots=True)
class CispoArtifact:
    """An artifact streamed through the public CISPO service."""

    body: bytes
    content_type: str


class CispoPublicServiceClient:
    """Client for the public local CISPO service, suitable for CLI and Workshop."""

    def __init__(
        self, base_url: str, token: str | None = None, *, timeout_seconds: float = 300.0
    ) -> None:
        self.base_url = _non_empty_text(base_url, field="CISPO service URL").rstrip("/")
        self.token = _optional_text(token)
        self.timeout_seconds = timeout_seconds

    def submit(
        self,
        config_json: Mapping[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/runs",
            {
                "algorithm": CISPO_ALGORITHM_ID,
                "config_json": dict(config_json),
                **({"run_id": run_id} if run_id else {}),
                **({"idempotency_key": idempotency_key} if idempotency_key else {}),
            },
        )

    def get(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}")

    def pause(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/pause", {})

    def resume(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/resume", {})

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/cancel", {})

    def optimizer_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"after_sequence": max(0, after_sequence), "limit": max(1, min(5_000, limit))}
        )
        return self._request(
            "GET", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/optimizer-events?{query}"
        )

    def optimizer_event_stream(
        self, run_id: str, *, after_sequence: int = 0
    ) -> Iterator[dict[str, Any]]:
        query = urllib.parse.urlencode({"after_sequence": max(0, after_sequence)})
        path = f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/optimizer-events/stream?{query}"
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
            raise CispoServiceError(
                f"public CISPO event stream {run_id} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CispoServiceError(f"public CISPO event stream {run_id} failed: {exc}") from exc

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
            raise CispoServiceError(
                f"public CISPO service {method} {path} failed: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CispoServiceError(f"public CISPO service {method} {path} failed: {exc}") from exc
        try:
            decoded = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise CispoServiceError(f"public CISPO service returned invalid JSON: {exc}") from exc
        return _json_object(decoded, context="public CISPO response")


class CispoService:
    """Durable public CISPO façade with one canonical run ID per submission."""

    def __init__(
        self,
        database_path: str | Path,
        executor: TinkerCispoExecutor | None = None,
        *,
        fixture: bool = False,
        background: bool = False,
        experiments: Any | None = None,
    ) -> None:
        self.experiments = experiments
        if self.experiments is None and os.environ.get('SYNTH_OPTIMIZERS_RL_EXPERIMENT_PREVIEW') == '1':
            from .rl.experiment_service import ExperimentService
            self.experiments = ExperimentService(str(database_path) + '.experiments')
        self.background = background
        if executor is not None:
            self.store = executor.store
            self.executor = executor
        else:
            use_fixture = fixture or _use_fixture_executor()
            self.store = JobStore(database_path)
            self.executor = TinkerCispoExecutor.local(
                self.store, fixture=use_fixture, validate_cispo=use_fixture
            )
        if background:
            self.executor.sync = False

    @classmethod
    def from_env(cls, database_path: str | Path) -> "CispoService":
        return cls(database_path, background=True)

    @classmethod
    def from_fixture(cls, database_path: str | Path) -> "CispoService":
        return cls(database_path, fixture=True)

    def submit(
        self,
        config_json: Mapping[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if config_json.get('schema_version') == 'rl.experiment.v1':
            if self.experiments is None:
                raise CispoServiceError('container experiment preview is not enabled')
            identity = config_json.get('experiment_id')
            if any(value is not None and value != identity for value in (run_id, idempotency_key)):
                raise CispoServiceError('experiment, run and idempotency identities must agree')
            try:
                self.store.require(str(identity))
            except JobStoreError:
                pass
            else:
                raise CispoServiceError('run identity already belongs to the legacy CISPO runtime')
            return self.experiments.submit(config_json, start=self.background)
        payload = _executor_config(config_json)
        algorithm = str(payload.get("algorithm_id") or payload.get("algorithm") or "").strip()
        if algorithm != CISPO_ALGORITHM_ID:
            raise CispoServiceError("public CISPO service accepts algorithm=cispo only")
        try:
            validate_cispo_request(payload)
        except SchemaError as exc:
            raise CispoServiceError(str(exc)) from exc
        requested_run_id = run_id or idempotency_key or _optional_text(payload.get("run_id"))
        canonical_run_id = requested_run_id or _fresh_run_id()
        if self.is_experiment(canonical_run_id):
            raise CispoServiceError('run identity already belongs to a container experiment')
        result = self.executor.submit(
            payload,
            job_id=canonical_run_id,
            idempotency_key_override=idempotency_key,
        )
        return self._submit_response(canonical_run_id, str(result.get("status") or "queued"))

    def get(self, run_id: str) -> dict[str, Any]:
        if self.is_experiment(run_id):
            return self.experiments.get(run_id)
        return self._public_run(self.executor.status(run_id))

    def cancel(self, run_id: str) -> dict[str, Any]:
        if self.is_experiment(run_id):
            return self.experiments.control(run_id, 'stop')
        return self._public_run(self.executor.cancel(run_id))

    def is_experiment(self, run_id: str) -> bool:
        return self.experiments is not None and self.experiments.contains(run_id)

    def experiment_control(self, run_id, action):
        if not self.is_experiment(run_id):
            if action in {'pause', 'resume'}:
                return self._public_run(getattr(self.executor, action)(run_id))
            raise CispoServiceError('run does not support experiment controls')
        return self.experiments.control(run_id, action)

    def optimizer_events(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        from .runtime.workshop import optimizer_event_page

        if self.is_experiment(run_id):
            return self.experiments.events(run_id, after_sequence, limit)

        return optimizer_event_page(
            self.store, run_id, after_sequence=after_sequence, limit=limit
        )

    def state_batch(self, run_id: str, slices: str) -> dict[str, Any]:
        from .runtime.workshop import state_batch
        if self.is_experiment(run_id):
            summary = self.experiments.get(run_id)
            payload = {'run_id': run_id, 'summary': summary}
            for name in filter(None, (s.strip() for s in slices.split(','))):
                if name == 'summary':
                    continue
                if name in {'candidates', 'checkpoints'}:
                    items = self.experiments.checkpoints(run_id)['checkpoints']
                elif name == 'evaluations':
                    items = [p['result'] for p in summary['phases'] if p['state'] == 'completed'
                             and p['phase']['kind'] in {'validation', 'final'}]
                else:
                    raise CispoServiceError('unsupported experiment state slice')
                payload[name] = {'items': items}
            return payload
        return state_batch(self.store, run_id, slices, algorithm_id=CISPO_ALGORITHM_ID)

    def artifact(self, run_id: str, name: str) -> CispoArtifact:
        body, content_type, _digest = self.store.artifact(run_id, name)
        return CispoArtifact(body=body, content_type=content_type)

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
            "algorithm": CISPO_ALGORITHM_ID,
            "status": status,
            "events_url": f"/v1/runs/{run_id}/optimizer-events",
            "events_stream_url": f"/v1/runs/{run_id}/optimizer-events/stream",
            "status_url": f"/v1/runs/{run_id}",
            "artifact_base_url": f"/v1/runs/{run_id}/artifacts",
        }


def create_cispo_http_server(
    bind: tuple[str, int],
    service: CispoService,
    *,
    service_token: str | None = None,
) -> ThreadingHTTPServer:
    token = _optional_text(service_token)
    if service.experiments is not None and token is None:
        raise CispoServiceError('container experiment HTTP service requires a bearer token')

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
                parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
                query = urllib.parse.parse_qs(parsed.query)
                if self.command == "GET" and parsed.path == "/health":
                    self._write(HTTPStatus.OK, {"status": "ok", "algorithm": CISPO_ALGORITHM_ID})
                elif self.command == 'GET' and parts == ['v1', 'capabilities']:
                    self._write(HTTPStatus.OK, {'schema_version': 'cispo_service_capabilities.v1',
                        'container_experiments': service.experiments is not None,
                        'experiment_schema': 'rl.experiment.v1', 'experiment_events': 'cursor_polling',
                        'experiment_control_boundary': 'phase_drain', 'release_stage': 'preview'})
                elif self.command == "POST" and parts == ["v1", "runs"]:
                    payload = self._body()
                    if payload.get("algorithm", CISPO_ALGORITHM_ID) != CISPO_ALGORITHM_ID:
                        raise CispoServiceError("public CISPO service accepts algorithm=cispo only")
                    run = service.submit(
                        _mapping(payload.get("config_json"), context="config_json"),
                        run_id=_optional_text(payload.get("run_id")),
                        idempotency_key=_optional_text(payload.get("idempotency_key")),
                    )
                    self._write(HTTPStatus.OK, run)
                elif len(parts) >= 3 and parts[:2] == ["v1", "runs"]:
                    run_id = parts[2]
                    if self.command == "GET" and len(parts) == 3:
                        self._write(HTTPStatus.OK, service.get(run_id))
                    elif self.command == "POST" and parts[3:] == ["cancel"]:
                        self._write(HTTPStatus.OK, service.cancel(run_id))
                    elif self.command == 'POST' and len(parts) == 4 and parts[3] in {'start', 'pause', 'resume', 'stop', 'recover'}:
                        self._write(HTTPStatus.OK, service.experiment_control(run_id, parts[3]))
                    elif self.command == 'GET' and parts[3:] == ['checkpoints'] and service.is_experiment(run_id):
                        self._write(HTTPStatus.OK, service.experiments.checkpoints(run_id))
                    elif self.command == 'GET' and parts[3:] == ['evaluations'] and service.is_experiment(run_id):
                        self._write(HTTPStatus.OK, service.experiments.evaluations(run_id))
                    elif self.command == 'POST' and len(parts) == 6 and parts[3] == 'checkpoints' and parts[5] == 'verify' and service.is_experiment(run_id):
                        self._write(HTTPStatus.OK, service.experiments.verify_checkpoint(run_id, parts[4]))
                    elif self.command == "GET" and (
                        parts[3:] == ["optimizer-events"]
                        or parts[3:] == ["optimizer-events", "stream"]
                    ):
                        if wants_live_stream(parsed.path, query):
                            if service.is_experiment(run_id):
                                raise CispoServiceError('experiment events use cursor polling; SSE is not advertised')
                            service.store.require(run_id)
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
                else:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except CispoServiceError as exc:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except JobStoreError as exc:
                self._write(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except CoordinationError:
                self._write(HTTPStatus.CONFLICT, {'error': 'experiment_state_conflict', 'reconciliation_required': True})
            except ValueError:
                self._write(HTTPStatus.BAD_REQUEST, {'error': 'invalid_experiment_request'})
            except Exception:  # pragma: no cover - final HTTP boundary
                self._write(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_service_error"})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                value = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                raise CispoServiceError(f"invalid JSON request body: {exc}") from exc
            return _json_object(value, context="CISPO service request")

        def _write(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
            body = json.dumps(value, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_artifact(self, artifact: CispoArtifact) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", artifact.content_type)
            self.send_header("Content-Length", str(len(artifact.body)))
            self.end_headers()
            self.wfile.write(artifact.body)

    return ThreadingHTTPServer(bind, Handler)


def serve_cispo_service(
    database_path: str | Path,
    bind: str,
    *,
    service_token: str | None = None,
    fixture: bool = False,
) -> None:
    host, port = _parse_bind(bind)
    server = create_cispo_http_server(
        (host, port),
        cispo_service_for_serve(database_path, fixture=fixture),
        service_token=service_token or os.environ.get("SYNTH_OPTIMIZERS_CISPO_SERVICE_TOKEN"),
    )
    server.serve_forever()


def cispo_service_for_serve(
    database_path: str | Path, *, fixture: bool = False
) -> CispoService:
    """Construct the background CISPO service used by ``serve_cispo_service``.

    Honors ``SYNTH_OPTIMIZERS_CISPO_FIXTURE=1`` the same way SFT honors
    ``SYNTH_OPTIMIZERS_SFT_FIXTURE``. Tests can call this instead of serving forever.
    """

    return CispoService(
        database_path, fixture=fixture or _use_fixture_executor(), background=True
    )


def _use_fixture_executor() -> bool:
    return os.environ.get("SYNTH_OPTIMIZERS_CISPO_FIXTURE", "").strip() == "1"


def _executor_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    dataset = payload.get("dataset")
    has_examples = isinstance(payload.get("examples"), list) and bool(payload.get("examples"))
    has_dataset_examples = isinstance(dataset, Mapping) and bool(dataset.get("examples"))
    has_dataset_source = isinstance(dataset, Mapping) and bool(dataset.get("split_strategy"))
    if not has_examples and not has_dataset_examples and not has_dataset_source:
        examples = fixture_examples()
        payload["examples"] = examples
        merged_dataset = dict(dataset) if isinstance(dataset, Mapping) else {}
        merged_dataset.setdefault("examples", examples)
        merged_dataset.setdefault("train_indexes", [0, 1, 2, 3])
        merged_dataset.setdefault("calibration_indexes", [4])
        merged_dataset.setdefault("heldout_indexes", [5])
        payload["dataset"] = merged_dataset
    return payload


def _iter_sse_events(response: Any) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
            continue
        if line or not data_lines:
            continue
        payload_text = "\n".join(data_lines)
        data_lines.clear()
        yield _json_object(json.loads(payload_text), context="CISPO SSE event")
    if data_lines:
        yield _json_object(json.loads("\n".join(data_lines)), context="CISPO SSE event")


def _fresh_run_id() -> str:
    import uuid

    return f"cispo_{uuid.uuid4().hex}"


def _parse_bind(bind: str) -> tuple[str, int]:
    host, separator, raw_port = bind.rpartition(":")
    if not separator or not host:
        raise CispoServiceError("bind must be HOST:PORT")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise CispoServiceError("bind port must be an integer") from exc
    if not 0 < port < 65536:
        raise CispoServiceError("bind port must be in 1..65535")
    return host, port


def _query_int(query: Mapping[str, list[str]], key: str, *, default: int) -> int:
    try:
        return int(query.get(key, [str(default)])[0])
    except ValueError:
        return default


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CispoServiceError(f"{context} must be an object")
    encoded = json.dumps(value)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise CispoServiceError(f"{context} must be an object")
    return decoded


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CispoServiceError(f"{context} must be an object")
    return value


def _non_empty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CispoServiceError(f"{field} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
