from __future__ import annotations

import base64
import http.client
import json
import os
import secrets
import socket
import ssl
import struct
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class HostedOptimizerError(RuntimeError):
    pass


class HostedOptimizerAuthError(HostedOptimizerError):
    pass


class HostedOptimizerHTTPError(HostedOptimizerError):
    def __init__(self, status_code: int, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class HostedGepaConfig(Protocol):
    def to_toml_dict(self) -> Mapping[str, Any]: ...


_HOP_BY_HOP_HEADERS = {
    "connection",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


@dataclass(slots=True)
class HostedOptimizerClient:
    backend_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.backend_url = (
            self.backend_url
            or os.getenv("SYNTH_BACKEND_URL")
            or os.getenv("SYNTH_API_BASE_URL")
            or "https://api.usesynth.ai"
        ).rstrip("/")
        self.api_key = self.api_key if self.api_key is not None else os.getenv("SYNTH_API_KEY")

    def startup(
        self,
        *,
        client: str | None = None,
        version: str | None = None,
        algorithms: list[str] | None = None,
        runtime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        record: bool = False,
    ) -> dict[str, Any]:
        catalog = self._request_json("GET", "/api/v1/optimizers/startup")
        if record:
            self.record_startup(
                client=client,
                version=version,
                algorithms=algorithms,
                runtime=runtime,
                metadata=metadata,
            )
        return catalog

    def record_startup(
        self,
        *,
        client: str | None = None,
        version: str | None = None,
        algorithms: list[str] | None = None,
        runtime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/optimizers/startup",
            {
                "client": client,
                "version": version,
                "algorithms": list(algorithms or []),
                "runtime": runtime,
                "metadata": dict(metadata or {}),
            },
        )

    def submit_gepa(
        self,
        config: HostedGepaConfig | Mapping[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        container_pool: Mapping[str, Any] | None = None,
        container_tunnel: SynthTunnelLease | None = None,
    ) -> dict[str, Any]:
        config_json = _gepa_config_json(config)
        if container_tunnel is not None:
            _apply_container_tunnel(config_json, container_tunnel)
        return self._request_json(
            "POST",
            "/api/v1/optimizers/runs",
            {
                "algorithm": "gepa",
                "run_id": run_id,
                "idempotency_key": idempotency_key,
                "project_id": project_id,
                "config_json": config_json,
                "container_pool": dict(container_pool) if container_pool is not None else None,
            },
        )

    def open_synth_tunnel(
        self,
        local_base_url: str,
        *,
        client_instance_id: str | None = None,
        ttl_seconds: int = 3600,
        max_inflight: int = 32,
        metadata: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        start: bool = True,
        attach_timeout_seconds: float = 15.0,
    ) -> SynthTunnelLease:
        local_target = _local_target_from_url(local_base_url)
        lease = self._request_json(
            "POST",
            "/api/v1/synthtunnel/leases",
            {
                "client_instance_id": client_instance_id
                or f"synth-optimizers-sdk-{uuid.uuid4().hex[:12]}",
                "local_target": local_target,
                "requested_ttl_seconds": ttl_seconds,
                "metadata": dict(metadata or {}),
                "capabilities": {
                    "max_inflight": max_inflight,
                    **dict(capabilities or {}),
                },
            },
        )
        tunnel = SynthTunnelLease(
            client=self,
            lease=lease,
            local_base_url=local_base_url,
            max_workers=max(1, min(max_inflight, 128)),
        )
        if start:
            try:
                tunnel.start(timeout_seconds=attach_timeout_seconds)
            except Exception:
                tunnel.close()
                raise
        return tunnel

    def close_synth_tunnel(self, lease_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"/api/v1/synthtunnel/leases/{lease_id}")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/api/v1/optimizers/runs/{run_id}")

    def wait_for_run(
        self,
        run_id: str,
        *,
        poll_interval_seconds: float = 10.0,
        timeout_seconds: float = 1200.0,
        terminal_statuses: set[str] | None = None,
    ) -> dict[str, Any]:
        terminal = terminal_statuses or {"succeeded", "failed", "cancelled"}
        deadline = time.monotonic() + timeout_seconds
        while True:
            payload = self.get_run(run_id)
            status = str(payload.get("status") or "").strip().lower()
            if status in terminal:
                return payload
            if time.monotonic() >= deadline:
                raise HostedOptimizerError(
                    f"optimizer run {run_id!r} did not finish within {timeout_seconds:.1f}s"
                )
            time.sleep(max(0.1, poll_interval_seconds))

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/api/v1/optimizers/runs/{run_id}/cancel")

    def get_artifact(self, run_id: str, artifact_name: str) -> bytes:
        return self._request_bytes(
            "GET",
            f"/api/v1/optimizers/runs/{run_id}/artifacts/{artifact_name}",
        )

    def events(self, run_id: str) -> Iterator[str]:
        request = self._build_request(
            "GET",
            f"/api/v1/optimizers/runs/{run_id}/events",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line:
                        yield line
        except HTTPError as exc:
            payload = exc.read()
            raise HostedOptimizerHTTPError(
                exc.code,
                payload.decode("utf-8", errors="replace") or str(exc),
            ) from exc
        except URLError as exc:
            raise HostedOptimizerError(f"GET events for {run_id} failed: {exc}") from exc

    def _request_json(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._request_bytes(method, path, body)
        if not payload:
            return {}
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise HostedOptimizerError(f"{method} {path} did not return a JSON object")
        return decoded

    def _request_bytes(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> bytes:
        request = self._build_request(method, path, body)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            payload = exc.read()
            parsed: Any = None
            message = payload.decode("utf-8", errors="replace") or str(exc)
            try:
                parsed = json.loads(payload.decode("utf-8"))
                if isinstance(parsed, dict):
                    message = str(parsed.get("detail") or parsed.get("error") or message)
            except json.JSONDecodeError:
                pass
            raise HostedOptimizerHTTPError(exc.code, message, parsed) from exc
        except URLError as exc:
            raise HostedOptimizerError(f"{method} {path} failed: {exc}") from exc

    def _build_request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Request:
        api_key = self._api_key_for_request()
        data = None
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(_drop_none(body)).encode("utf-8")
        return Request(
            urljoin(f"{self.backend_url}/", path.lstrip("/")),
            data=data,
            headers=headers,
            method=method,
        )

    def _api_key_for_request(self) -> str | None:
        api_key = (self.api_key or "").strip()
        if api_key:
            return api_key
        if not _is_local_backend_url(str(self.backend_url or "")):
            raise HostedOptimizerAuthError(
                "SYNTH_API_KEY is required for hosted optimizer backend calls"
            )
        return None


@dataclass(slots=True)
class SynthTunnelLease:
    client: HostedOptimizerClient
    lease: Mapping[str, Any]
    local_base_url: str
    max_workers: int = 32
    _agent: _SynthTunnelAgent | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def lease_id(self) -> str:
        return str(self.lease["lease_id"])

    @property
    def public_url(self) -> str:
        return str(self.lease["public_url"]).rstrip("/")

    @property
    def worker_token(self) -> str:
        return str(self.lease["worker_token"])

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.worker_token}"}

    def container_config(self) -> dict[str, Any]:
        return {"url": self.public_url, "headers": self.headers}

    def start(self, *, timeout_seconds: float = 15.0) -> SynthTunnelLease:
        if self._agent is not None:
            return self
        agent = _SynthTunnelAgent(
            lease=self.lease,
            local_base_url=self.local_base_url,
            max_workers=self.max_workers,
        )
        agent.start(timeout_seconds=timeout_seconds)
        self._agent = agent
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._agent is not None:
            self._agent.close()
            self._agent = None
        try:
            self.client.close_synth_tunnel(self.lease_id)
        except HostedOptimizerHTTPError as exc:
            if exc.status_code != 404:
                raise

    def __enter__(self) -> SynthTunnelLease:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class _SynthTunnelAgent:
    def __init__(
        self,
        *,
        lease: Mapping[str, Any],
        local_base_url: str,
        max_workers: int,
    ) -> None:
        self._lease = lease
        self._local_base_url = local_base_url.rstrip("/")
        self._lease_id = str(lease["lease_id"])
        agent_connect = lease.get("agent_connect")
        if not isinstance(agent_connect, Mapping):
            raise HostedOptimizerError("synth tunnel lease is missing agent_connect")
        self._agent_url = str(agent_connect["url"])
        self._agent_token = str(agent_connect["agent_token"])
        self._stop = threading.Event()
        self._attached = threading.Event()
        self._send_lock = threading.Lock()
        self._requests_lock = threading.Lock()
        self._requests: dict[str, dict[str, Any]] = {}
        self._bodies: dict[str, bytearray] = {}
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._socket: socket.socket | ssl.SSLSocket | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"synth-tunnel-{self._lease_id[:8]}",
            daemon=True,
        )
        self._startup_error: BaseException | None = None

    def start(self, *, timeout_seconds: float) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._attached.wait(timeout=0.05):
                return
            if self._startup_error is not None:
                self.close()
                raise HostedOptimizerError(
                    f"synth tunnel agent failed to attach: {self._startup_error}"
                ) from self._startup_error
            if not self._thread.is_alive():
                break
        self.close()
        if self._startup_error is not None:
            raise HostedOptimizerError(
                f"synth tunnel agent failed to attach: {self._startup_error}"
            ) from self._startup_error
        raise HostedOptimizerError(
            f"synth tunnel agent did not attach within {timeout_seconds:.1f}s"
        )

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            with _WebSocketConnection(self._agent_url, self._agent_token) as websocket:
                self._socket = websocket.socket
                self._send_json(
                    websocket,
                    {
                        "type": "ATTACH",
                        "leases": [
                            {
                                "lease_id": self._lease_id,
                                "local_base_url": self._local_base_url,
                            }
                        ],
                    },
                )
                while not self._stop.is_set():
                    payload = websocket.receive_json()
                    if payload is None:
                        return
                    self._handle_message(websocket, payload)
        except Exception as exc:
            if not self._stop.is_set():
                self._startup_error = exc

    def _handle_message(self, websocket: _WebSocketConnection, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")
        if msg_type == "ATTACH_ACK":
            accepted = payload.get("accepted_leases")
            if isinstance(accepted, list) and self._lease_id in {str(item) for item in accepted}:
                self._attached.set()
            else:
                self._startup_error = HostedOptimizerError(
                    f"synth tunnel lease {self._lease_id!r} was not accepted"
                )
            return
        rid = payload.get("rid")
        if rid is None:
            return
        rid_text = str(rid)
        if msg_type == "REQ_HEADERS":
            with self._requests_lock:
                self._requests[rid_text] = payload
                self._bodies[rid_text] = bytearray()
            return
        if msg_type == "REQ_BODY":
            chunk = payload.get("chunk_b64")
            if isinstance(chunk, str) and chunk:
                with self._requests_lock:
                    body = self._bodies.setdefault(rid_text, bytearray())
                    body.extend(base64.b64decode(chunk.encode("ascii")))
            return
        if msg_type != "REQ_END":
            return
        with self._requests_lock:
            request_payload = self._requests.pop(rid_text, None)
            body = bytes(self._bodies.pop(rid_text, bytearray()))
        if request_payload is None:
            return
        self._executor.submit(self._forward_request, websocket, rid_text, request_payload, body)

    def _forward_request(
        self,
        websocket: _WebSocketConnection,
        rid: str,
        request_payload: Mapping[str, Any],
        body: bytes,
    ) -> None:
        try:
            response = _forward_local_request(
                self._local_base_url,
                method=str(request_payload.get("method") or "GET"),
                path=str(request_payload.get("path") or "/"),
                query=str(request_payload.get("query") or ""),
                headers=_headers_list_to_dict(request_payload.get("headers")),
                body=body,
            )
            self._send_json(
                websocket,
                {
                    "type": "RESP_HEADERS",
                    "lease_id": self._lease_id,
                    "rid": rid,
                    "status": response["status"],
                    "headers": response["headers"],
                },
            )
            content = response["body"]
            if content:
                self._send_json(
                    websocket,
                    {
                        "type": "RESP_BODY",
                        "lease_id": self._lease_id,
                        "rid": rid,
                        "chunk_b64": base64.b64encode(content).decode("ascii"),
                    },
                )
            self._send_json(
                websocket,
                {"type": "RESP_END", "lease_id": self._lease_id, "rid": rid},
            )
        except Exception as exc:
            self._send_json(
                websocket,
                {
                    "type": "RESP_ERROR",
                    "lease_id": self._lease_id,
                    "rid": rid,
                    "code": "LOCAL_FORWARD_ERROR",
                    "message": str(exc),
                },
            )

    def _send_json(self, websocket: _WebSocketConnection, payload: Mapping[str, Any]) -> None:
        with self._send_lock:
            websocket.send_json(payload)


class _WebSocketConnection:
    def __init__(self, url: str, bearer_token: str) -> None:
        self._url = url
        self._bearer_token = bearer_token
        self.socket: socket.socket | ssl.SSLSocket | None = None

    def __enter__(self) -> _WebSocketConnection:
        self.socket = self._connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.socket is None:
            return
        try:
            self.socket.close()
        except OSError:
            pass
        self.socket = None

    def send_json(self, payload: Mapping[str, Any]) -> None:
        text = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        self._write_frame(0x1, text)

    def receive_json(self) -> dict[str, Any] | None:
        while True:
            opcode, payload = self._read_frame()
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._write_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode != 0x1:
                continue
            decoded = json.loads(payload.decode("utf-8"))
            if isinstance(decoded, dict):
                return decoded

    def _connect(self) -> socket.socket | ssl.SSLSocket:
        parsed = urlparse(self._url)
        if parsed.scheme not in {"ws", "wss"}:
            raise HostedOptimizerError(
                f"unsupported synth tunnel agent URL scheme: {parsed.scheme}"
            )
        host = parsed.hostname
        if not host:
            raise HostedOptimizerError("synth tunnel agent URL is missing a host")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        raw_socket = socket.create_connection((host, port), timeout=30.0)
        sock: socket.socket | ssl.SSLSocket
        if parsed.scheme == "wss":
            sock = ssl.create_default_context().wrap_socket(raw_socket, server_hostname=host)
        else:
            sock = raw_socket
        sock.settimeout(None)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        host_header = host if parsed.port is None else f"{host}:{port}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Authorization: Bearer {self._bearer_token}\r\n"
            "\r\n"
        ).encode("ascii")
        sock.sendall(request)
        response = _read_http_headers(sock)
        if not response.startswith(b"HTTP/1.1 101") and not response.startswith(b"HTTP/1.0 101"):
            sock.close()
            raise HostedOptimizerError(
                f"synth tunnel websocket handshake failed: {response[:120]!r}"
            )
        return sock

    def _read_frame(self) -> tuple[int, bytes]:
        header = self._read_exact(2)
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def _write_frame(self, opcode: int, payload: bytes) -> None:
        sock = self.socket
        if sock is None:
            raise HostedOptimizerError("synth tunnel websocket is not connected")
        mask = secrets.token_bytes(4)
        header = bytes([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header += bytes([0x80 | length])
        elif length <= 0xFFFF:
            header += bytes([0x80 | 126]) + struct.pack("!H", length)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", length)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        sock.sendall(header + mask + masked)

    def _read_exact(self, length: int) -> bytes:
        sock = self.socket
        if sock is None:
            raise HostedOptimizerError("synth tunnel websocket is not connected")
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise HostedOptimizerError("synth tunnel websocket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def _read_http_headers(sock: socket.socket | ssl.SSLSocket) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(1)
        if not chunk:
            break
        chunks.append(chunk)
        if b"".join(chunks).endswith(b"\r\n\r\n"):
            break
        if len(chunks) > 16384:
            raise HostedOptimizerError("synth tunnel websocket handshake response exceeded 16 KiB")
    return b"".join(chunks)


def _forward_local_request(
    local_base_url: str,
    *,
    method: str,
    path: str,
    query: str,
    headers: Mapping[str, str],
    body: bytes,
) -> dict[str, Any]:
    parsed = urlparse(local_base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"}:
        raise HostedOptimizerError(f"unsupported local tunnel URL scheme: {parsed.scheme!r}")
    request_path = path if path.startswith("/") else f"/{path}"
    if parsed.path and parsed.path != "/":
        request_path = f"{parsed.path.rstrip('/')}{request_path}"
    if query:
        request_path = f"{request_path}?{query}"
    connection_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_cls(parsed.netloc, timeout=1200.0)
    try:
        connection.request(method.upper(), request_path, body=body, headers=dict(headers))
        response = connection.getresponse()
        response_body = response.read()
        return {
            "status": response.status,
            "headers": _response_headers_to_list(response.getheaders()),
            "body": response_body,
        }
    finally:
        connection.close()


def _headers_list_to_dict(raw_headers: object) -> dict[str, str]:
    if not isinstance(raw_headers, list):
        return {}
    headers: dict[str, str] = {}
    for item in raw_headers:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        key = str(item[0])
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        headers[key] = str(item[1])
    return headers


def _response_headers_to_list(raw_headers: list[tuple[str, str]]) -> list[list[str]]:
    result: list[list[str]] = []
    for key, value in raw_headers:
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        result.append([key, value])
    return result


def _local_target_from_url(local_base_url: str) -> dict[str, Any]:
    parsed = urlsplit(local_base_url)
    if parsed.scheme not in {"http", "https"}:
        raise HostedOptimizerError(f"unsupported local container URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise HostedOptimizerError(
            "SynthTunnel local targets must be loopback: 127.0.0.1, localhost, or ::1"
        )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return {"host": host, "port": port}


def _is_local_backend_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _gepa_config_json(config: HostedGepaConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    to_toml_dict = getattr(config, "to_toml_dict", None)
    if not callable(to_toml_dict):
        raise HostedOptimizerError("GEPA hosted config must be a mapping or expose to_toml_dict()")
    payload = to_toml_dict()
    if not isinstance(payload, Mapping):
        raise HostedOptimizerError("GEPA hosted config to_toml_dict() must return a mapping")
    return dict(payload)


def _apply_container_tunnel(config: dict[str, Any], tunnel: SynthTunnelLease) -> None:
    container = config.get("container")
    if not isinstance(container, dict):
        container = {}
        config["container"] = container
    existing_headers = container.get("headers")
    headers = dict(existing_headers) if isinstance(existing_headers, Mapping) else {}
    headers.update(tunnel.headers)
    container["url"] = tunnel.public_url
    container["headers"] = headers


__all__ = [
    "HostedGepaConfig",
    "HostedOptimizerAuthError",
    "HostedOptimizerClient",
    "HostedOptimizerError",
    "HostedOptimizerHTTPError",
    "SynthTunnelLease",
]
