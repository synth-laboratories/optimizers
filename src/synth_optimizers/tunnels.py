from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse, urlunparse


class TunnelError(RuntimeError):
    pass


class TunnelProvider(StrEnum):
    AUTO = "auto"
    SYNTH_TUNNEL = "synth_tunnel"
    CLOUDFLARED = "cloudflared"
    NGROK = "ngrok"


_CLIENT_INSTANCE_ID_ENV = "SYNTH_OPTIMIZERS_TUNNEL_CLIENT_INSTANCE_ID"
_STATE_DIR_ENV = "SYNTH_OPTIMIZERS_STATE_DIR"
_CLIENT_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_LOCAL_ONLY_AUTH_HEADERS = {"authorization", "x-api-key", "x-api-keys"}
_CLOUDFLARED_READY_TIMEOUT_SECONDS = 180.0
_TUNNEL_USER_AGENT = "synth-optimizers tunnel-client"


@dataclass(frozen=True, slots=True)
class ContainerDirectTarget:
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    auth_bearer_env: str | None = None
    startup_timeout_seconds: int | None = None

    def to_config_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": self.url}
        if self.headers:
            payload["headers"] = dict(self.headers)
        if self.auth_bearer_env:
            payload["auth_bearer_env"] = self.auth_bearer_env
        if self.startup_timeout_seconds is not None:
            payload["startup_timeout_seconds"] = self.startup_timeout_seconds
        return payload


@dataclass(frozen=True, slots=True)
class TunnelLocalTarget:
    base_url: str
    host: str
    port: int
    scheme: str

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass(slots=True, kw_only=True)
class TunnelLease:
    provider: TunnelProvider
    public_url: str
    lease_id: str | None = None
    expires_at: str | None = None
    connector_mode: str | None = None
    diagnostics_hint: str | None = None
    agent_connect_required: bool = False

    def container_config(self) -> ContainerDirectTarget:
        return ContainerDirectTarget(url=_required_text(self.public_url, "tunnel public_url"))

    def preflight(self) -> None:
        return None

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        _wait_for_http_ok(_join_health_url(self.public_url), timeout_seconds=timeout_seconds)

    def close(self) -> None:
        return None

    def __enter__(self) -> "TunnelLease":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(slots=True, kw_only=True)
class SynthTunnelLease(TunnelLease):
    worker_token: str = field(repr=False)
    local_target: TunnelLocalTarget
    client: Any | None = field(default=None, repr=False)
    route_token: str | None = None
    agent_connect: Mapping[str, Any] | None = None
    agent: "_SynthTunnelAgent | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.provider = TunnelProvider.SYNTH_TUNNEL
        self.agent_connect_required = True
        self.connector_mode = self.connector_mode or "synth_tunnel_agent"

    def refresh_worker_token(self) -> str:
        if self.client is None:
            raise TunnelError("SynthTunnelLease has no client for token refresh")
        payload = self.client._json_request(
            "POST", f"/api/v1/synthtunnel/leases/{self.lease_id}/token:refresh"
        )
        self.worker_token = str(payload.get("worker_token") or "")
        if not self.worker_token:
            raise TunnelError("SynthTunnel token refresh returned no worker_token")
        return self.worker_token

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        if not self.worker_token:
            raise TunnelError("SynthTunnel worker token is required for readiness checks")
        _wait_for_http_ok(_join_health_url(self.local_target.base_url), timeout_seconds=10.0)
        if self.agent is None:
            self.agent = _SynthTunnelAgent(
                lease_id=_required_text(self.lease_id, "SynthTunnel lease_id"),
                local_target=self.local_target,
                agent_connect=_required_mapping(
                    self.agent_connect,
                    "SynthTunnel lease agent_connect",
                ),
            )
            self.agent.start(timeout_seconds=min(30.0, max(1.0, timeout_seconds)))
        _wait_for_http_ok(
            _join_health_url(self.public_url),
            headers={"Authorization": f"Bearer {self.worker_token}"},
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        if self.agent is not None:
            self.agent.stop()
            self.agent = None
        if self.client is not None and self.lease_id:
            self.client._json_request(
                "DELETE",
                f"/api/v1/synthtunnel/leases/{self.lease_id}",
                context="Synth tunnel lease close response",
                allow_empty=True,
            )


@dataclass(slots=True, kw_only=True)
class ManagedTunnelLease(TunnelLease):
    client: Any | None = field(default=None, repr=False)
    local_target: TunnelLocalTarget
    tunnel_token: str = field(default="", repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    heartbeat: "_HeartbeatLoop | None" = field(default=None, repr=False)

    def preflight(self) -> None:
        _wait_for_http_ok(_join_health_url(self.local_target.base_url), timeout_seconds=10.0)

    def _start_heartbeat(
        self,
        *,
        connected_to_edge: bool,
        gateway_ready: bool,
        local_ready: bool,
        last_error: str | None = None,
    ) -> None:
        if self.client is None or not self.lease_id:
            return
        if self.heartbeat is not None:
            self.heartbeat.stop()
        self.heartbeat = _HeartbeatLoop(
            client=self.client,
            lease_id=self.lease_id,
            connected_to_edge=connected_to_edge,
            gateway_ready=gateway_ready,
            local_ready=local_ready,
            last_error=last_error,
        )
        self.heartbeat.start()

    def close(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.stop()
            self.heartbeat = None
        try:
            if self.client is not None and self.lease_id:
                try:
                    self.client._json_request(
                        "POST",
                        f"/v1/tunnels/lease/{self.lease_id}/release",
                        context="Tunnel lease release response",
                        allow_empty=True,
                    )
                except Exception:
                    return
        finally:
            _terminate_process(self.process)
            self.process = None

@dataclass(slots=True, kw_only=True)
class CloudflaredTunnelLease(ManagedTunnelLease):
    gateway: "_GatewayServer | None" = field(default=None, repr=False)

    def container_config(self) -> ContainerDirectTarget:
        return ContainerDirectTarget(
            url=_required_text(self.public_url, "tunnel public_url"),
            headers=_cloudflared_request_headers(),
        )

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        self.preflight()
        if self.gateway is None:
            self.gateway = _GatewayServer(
                route_prefix=_route_prefix_from_public_url(self.public_url),
                local_target=self.local_target,
                gateway_port=_gateway_port_from_mode(self.connector_mode),
            )
            self.gateway.start()
        binary = _required_binary("cloudflared", self.provider)
        if self.process is None:
            self.process = _start_cloudflared(binary, self.tunnel_token)
        _ensure_process_running(self.process, "cloudflared")
        self._start_heartbeat(
            connected_to_edge=True,
            gateway_ready=True,
            local_ready=True,
        )
        ready_timeout = max(timeout_seconds, _CLOUDFLARED_READY_TIMEOUT_SECONDS)
        _wait_for_public_dns(
            _required_hostname(self.public_url, "cloudflared public_url"),
            timeout_seconds=min(ready_timeout, 90.0),
        )
        _wait_for_http_ok(
            _join_health_url(self.public_url),
            headers=_cloudflared_request_headers(),
            timeout_seconds=ready_timeout,
        )

    def close(self) -> None:
        ManagedTunnelLease.close(self)
        if self.gateway is not None:
            self.gateway.stop()
            self.gateway = None


@dataclass(slots=True, kw_only=True)
class NgrokTunnelLease(ManagedTunnelLease):
    ngrok_api_url: str | None = None

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        self.preflight()
        binary = _required_binary("ngrok", self.provider)
        if self.process is None:
            api_port = _free_local_port()
            self.ngrok_api_url = f"http://127.0.0.1:{api_port}"
            self.process = _start_ngrok(
                binary=binary,
                authtoken=self.tunnel_token,
                local_base_url=self.local_target.base_url,
                api_addr=f"127.0.0.1:{api_port}",
            )
        self.public_url = _discover_ngrok_public_url(
            self.process,
            self.ngrok_api_url or "http://127.0.0.1:4040",
            timeout_seconds=timeout_seconds,
        )
        self._start_heartbeat(
            connected_to_edge=True,
            gateway_ready=True,
            local_ready=True,
        )
        _wait_for_http_ok(_join_health_url(self.public_url), timeout_seconds=timeout_seconds)


def create_tunnel_lease(
    client: Any,
    local_base_url: str,
    *,
    provider: TunnelProvider | str = TunnelProvider.AUTO,
    requested_ttl_seconds: int = 3600,
    metadata: Mapping[str, Any] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    wait_ready: bool = True,
) -> TunnelLease:
    resolved_provider = _resolve_provider(provider)
    target = parse_local_target(local_base_url)
    if resolved_provider is TunnelProvider.SYNTH_TUNNEL:
        lease = _create_synth_tunnel_lease(
            client,
            target,
            requested_ttl_seconds=requested_ttl_seconds,
            metadata=metadata,
            capabilities=capabilities,
        )
        if wait_ready:
            try:
                lease.wait_ready()
            except Exception:
                _close_after_ready_failure(lease)
                raise
        return lease
    _require_managed_localhost_target(target, provider=resolved_provider)
    if resolved_provider is TunnelProvider.CLOUDFLARED:
        _required_binary("cloudflared", resolved_provider)
    if resolved_provider is TunnelProvider.NGROK:
        _required_binary("ngrok", resolved_provider)
    lease = _create_managed_tunnel_lease(
        client,
        target,
        provider=resolved_provider,
        requested_ttl_seconds=requested_ttl_seconds,
        metadata=metadata,
    )
    if wait_ready:
        try:
            lease.wait_ready()
        except Exception:
            _close_after_ready_failure(lease)
            raise
    return lease


def parse_local_target(local_base_url: str) -> TunnelLocalTarget:
    raw = _required_text(local_base_url, "local_base_url")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise TunnelError("local_base_url must use http or https")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )
    return TunnelLocalTarget(base_url=base_url, host=host, port=port, scheme=parsed.scheme)


def tunnel_provider_value(provider: Any) -> str:
    if isinstance(provider, TunnelProvider):
        return provider.value
    return str(provider or "").strip().lower()


def _resolve_provider(provider: TunnelProvider | str) -> TunnelProvider:
    raw = tunnel_provider_value(provider)
    if raw in {"", TunnelProvider.AUTO.value}:
        return TunnelProvider.SYNTH_TUNNEL
    try:
        resolved = TunnelProvider(raw)
    except ValueError as exc:
        raise TunnelError(
            "tunnel provider must be auto, synth_tunnel, cloudflared, or ngrok"
        ) from exc
    if resolved is TunnelProvider.AUTO:
        return TunnelProvider.SYNTH_TUNNEL
    return resolved


def _create_synth_tunnel_lease(
    client: Any,
    target: TunnelLocalTarget,
    *,
    requested_ttl_seconds: int,
    metadata: Mapping[str, Any] | None,
    capabilities: Mapping[str, Any] | None,
) -> SynthTunnelLease:
    payload = {
        "client_instance_id": _stable_client_instance_id(TunnelProvider.SYNTH_TUNNEL),
        "local_target": {"host": target.host, "port": target.port},
        "requested_ttl_seconds": requested_ttl_seconds,
        "metadata": dict(metadata or {}),
        "capabilities": dict(capabilities or {}),
    }
    response = client._json_request("POST", "/api/v1/synthtunnel/leases", payload)
    context = "Synth tunnel lease response"
    return SynthTunnelLease(
        provider=TunnelProvider.SYNTH_TUNNEL,
        lease_id=_response_text(response, "lease_id", context),
        public_url=_response_text(response, "public_url", context),
        worker_token=_response_text(response, "worker_token", context),
        local_target=target,
        client=client,
        route_token=_optional_text(response.get("route_token")),
        agent_connect=_mapping_or_none(response.get("agent_connect")),
        expires_at=_optional_text(response.get("expires_at")),
        connector_mode="synth_tunnel_agent",
        diagnostics_hint=_optional_text(response.get("diagnostics_hint")),
    )


def _create_managed_tunnel_lease(
    client: Any,
    target: TunnelLocalTarget,
    *,
    provider: TunnelProvider,
    requested_ttl_seconds: int,
    metadata: Mapping[str, Any] | None,
) -> ManagedTunnelLease:
    payload: dict[str, Any] = {
        "client_instance_id": _stable_client_instance_id(provider),
        "local_host": target.host,
        "local_port": target.port,
        "provider_preference": provider.value,
        "requested_ttl_seconds": requested_ttl_seconds,
        "reuse_connector": True,
    }
    app_name = _optional_text((metadata or {}).get("optimizer"))
    if app_name:
        payload["app_name"] = app_name
    response = client._json_request("POST", "/v1/tunnels/lease", payload)
    context = "managed tunnel lease response"
    lease_kwargs: dict[str, Any] = {
        "provider": provider,
        "lease_id": _response_text(response, "lease_id", context),
        "public_url": _optional_text(response.get("public_url")) or "",
        "expires_at": _optional_text(response.get("expires_at")),
        "connector_mode": _optional_text(response.get("connector_mode")),
        "diagnostics_hint": _optional_text(response.get("diagnostics_hint")),
        "agent_connect_required": False,
        "client": client,
        "local_target": target,
        "tunnel_token": _response_text(response, "tunnel_token", context),
    }
    if provider is TunnelProvider.CLOUDFLARED:
        if not lease_kwargs["public_url"]:
            raise TunnelError("cloudflared tunnel lease response did not include public_url")
        gateway_port = response.get("gateway_port")
        lease_kwargs["connector_mode"] = (
            f"cloudflared_tunnel_token:gateway_port={int(gateway_port)}"
            if gateway_port is not None
            else "cloudflared_tunnel_token"
        )
        return CloudflaredTunnelLease(**lease_kwargs)
    return NgrokTunnelLease(**lease_kwargs)


def _require_managed_localhost_target(
    target: TunnelLocalTarget, *, provider: TunnelProvider
) -> None:
    if target.host not in {"127.0.0.1", "localhost"}:
        raise TunnelError(
            f"{provider.value} managed leases require a localhost target; "
            f"got {target.host!r}"
        )


def _close_after_ready_failure(lease: TunnelLease) -> None:
    try:
        lease.close()
    except Exception:
        return


def _stable_client_instance_id(provider: TunnelProvider) -> str:
    env_value = _optional_text(os.environ.get(_CLIENT_INSTANCE_ID_ENV))
    base = _normalize_client_id(env_value) if env_value else _load_or_create_base_client_id()
    provider_suffix = provider.value.replace("_", "-")
    return _normalize_client_id(f"{base}-{provider_suffix}")


def _load_or_create_base_client_id() -> str:
    path = _client_id_path()
    try:
        existing = _optional_text(path.read_text(encoding="utf-8"))
    except OSError:
        existing = None
    if existing:
        return _normalize_client_id(existing)

    generated = f"synth-optimizers-{uuid.uuid4().hex[:24]}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return _normalize_client_id(generated)
    except OSError:
        return _fallback_base_client_id()


def _client_id_path() -> Path:
    raw_state_dir = _optional_text(os.environ.get(_STATE_DIR_ENV))
    if raw_state_dir:
        state_root = Path(raw_state_dir).expanduser()
    else:
        xdg_state = _optional_text(os.environ.get("XDG_STATE_HOME"))
        state_root = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local" / "state"
    return state_root / "synth-optimizers" / "tunnel-client-id"


def _fallback_base_client_id() -> str:
    raw = f"{socket.gethostname()}:{Path.home()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return _normalize_client_id(f"synth-optimizers-{digest}")


def _normalize_client_id(value: str) -> str:
    normalized = _CLIENT_ID_SAFE_RE.sub("-", value.strip()).strip("-._:")
    if len(normalized) < 8:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        normalized = f"synth-optimizers-{digest}"
    return normalized[:128]


class _HeartbeatLoop:
    def __init__(
        self,
        *,
        client: Any,
        lease_id: str,
        connected_to_edge: bool,
        gateway_ready: bool,
        local_ready: bool,
        last_error: str | None,
    ) -> None:
        self._client = client
        self._lease_id = lease_id
        self._payload = {
            "connected_to_edge": connected_to_edge,
            "gateway_ready": gateway_ready,
            "local_ready": local_ready,
            "last_error": last_error,
        }
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="synth-tunnel-heartbeat")
        self._thread.daemon = True

    def start(self) -> None:
        self._send_once()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(30.0):
            self._send_once()

    def _send_once(self) -> None:
        try:
            self._client._json_request(
                "POST",
                f"/v1/tunnels/lease/{self._lease_id}/heartbeat",
                self._payload,
                context="Tunnel lease heartbeat response",
            )
        except Exception:
            return


@dataclass(slots=True)
class _SynthTunnelRequest:
    method: str
    path: str
    query: str
    headers: list[tuple[str, str]]
    deadline_ms: int
    body: bytearray = field(default_factory=bytearray)


class _SynthTunnelAgent:
    def __init__(
        self,
        *,
        lease_id: str,
        local_target: TunnelLocalTarget,
        agent_connect: Mapping[str, Any],
    ) -> None:
        transport = _required_text(agent_connect.get("transport"), "SynthTunnel transport")
        if transport != "ws":
            raise TunnelError(f"unsupported SynthTunnel agent transport {transport!r}")
        self._lease_id = lease_id
        self._local_target = local_target
        self._url = _required_text(agent_connect.get("url"), "SynthTunnel agent url")
        self._agent_token = _required_text(
            agent_connect.get("agent_token"),
            "SynthTunnel agent token",
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._requests_lock = threading.Lock()
        self._requests: dict[str, _SynthTunnelRequest] = {}
        self._thread: threading.Thread | None = None
        self._ws: Any | None = None
        self._startup_error: str | None = None

    def start(self, *, timeout_seconds: float) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="synth-tunnel-agent",
            daemon=True,
        )
        self._thread.start()
        if self._ready.wait(timeout=max(1.0, timeout_seconds)):
            return
        self.stop()
        detail = self._startup_error or "agent did not attach before the readiness deadline"
        raise TunnelError(f"SynthTunnel agent attach failed: {detail}")

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._ws = None
        with self._requests_lock:
            self._requests.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            ws = None
            try:
                ws = _connect_websocket(
                    self._url,
                    headers={"Authorization": f"Bearer {self._agent_token}"},
                )
                self._ws = ws
                self._send_frame({"type": "ATTACH", "leases": [{"lease_id": self._lease_id}]})
                while not self._stop.is_set():
                    raw = ws.recv()
                    if raw in (None, b"", ""):
                        raise TunnelError("SynthTunnel websocket closed")
                    payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                    if isinstance(payload, Mapping):
                        self._handle_frame(payload)
            except Exception as exc:
                if not self._ready.is_set():
                    self._startup_error = f"{self._url}: {exc}"
                if self._stop.wait(1.0):
                    break
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                if self._ws is ws:
                    self._ws = None
                with self._requests_lock:
                    self._requests.clear()

    def _handle_frame(self, payload: Mapping[str, Any]) -> None:
        msg_type = str(payload.get("type") or "")
        if msg_type == "ATTACH_ACK":
            accepted = payload.get("accepted_leases") or []
            if self._lease_id not in {str(item) for item in accepted}:
                rejected = ", ".join(str(item) for item in payload.get("rejected_leases") or [])
                raise TunnelError(
                    "SynthTunnel agent attach was rejected"
                    + (f": {rejected}" if rejected else "")
                )
            self._ready.set()
            return

        rid = str(payload.get("rid") or "")
        if not rid:
            return
        if msg_type == "REQ_HEADERS":
            request = _SynthTunnelRequest(
                method=str(payload.get("method") or "GET").upper(),
                path=_request_path(payload.get("path")),
                query=str(payload.get("query") or ""),
                headers=_header_pairs(payload.get("headers")),
                deadline_ms=max(1000, int(payload.get("deadline_ms") or 120000)),
            )
            with self._requests_lock:
                self._requests[rid] = request
            return
        if msg_type == "REQ_BODY":
            chunk = _decode_bytes(str(payload.get("chunk_b64") or ""))
            with self._requests_lock:
                request = self._requests.get(rid)
                if request is not None:
                    request.body.extend(chunk)
            return
        if msg_type == "REQ_END":
            with self._requests_lock:
                request = self._requests.pop(rid, None)
            if request is not None:
                thread = threading.Thread(
                    target=self._serve_request,
                    args=(rid, request),
                    name="synth-tunnel-request",
                    daemon=True,
                )
                thread.start()

    def _serve_request(self, rid: str, request: _SynthTunnelRequest) -> None:
        timeout = max(1.0, request.deadline_ms / 1000.0)
        upstream_url = _local_upstream_url(self._local_target, request.path, request.query)
        headers = {
            key: value
            for key, value in request.headers
            if key.strip().lower() not in _HOP_BY_HOP_HEADERS | _LOCAL_ONLY_AUTH_HEADERS
        }
        try:
            upstream_request = urllib.request.Request(
                upstream_url,
                data=bytes(request.body) if request.body else None,
                headers=headers,
                method=request.method,
            )
            with urllib.request.urlopen(upstream_request, timeout=timeout) as response:
                self._send_response(rid, response.status, response.headers, response)
        except urllib.error.HTTPError as exc:
            self._send_response(rid, exc.code, exc.headers, exc)
        except Exception as exc:
            self._send_frame(
                {
                    "type": "RESP_ERROR",
                    "lease_id": self._lease_id,
                    "rid": rid,
                    "code": "LOCAL_REQUEST_FAILED",
                    "message": str(exc),
                }
            )

    def _send_response(
        self,
        rid: str,
        status_code: int,
        headers: Mapping[str, Any],
        response: Any,
    ) -> None:
        header_list = [
            [str(key), str(value)]
            for key, value in headers.items()
            if key.lower() not in {"connection", "content-length", "transfer-encoding"}
        ]
        self._send_frame(
            {
                "type": "RESP_HEADERS",
                "lease_id": self._lease_id,
                "rid": rid,
                "status": int(status_code),
                "headers": header_list,
            }
        )
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            self._send_frame(
                {
                    "type": "RESP_BODY",
                    "lease_id": self._lease_id,
                    "rid": rid,
                    "chunk_b64": _encode_bytes(chunk),
                    "eof": False,
                }
            )
        self._send_frame({"type": "RESP_END", "lease_id": self._lease_id, "rid": rid})

    def _send_frame(self, payload: Mapping[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise TunnelError("SynthTunnel websocket is not connected")
        with self._send_lock:
            ws.send(json.dumps(dict(payload)))


class _GatewayServer:
    def __init__(
        self,
        *,
        route_prefix: str,
        local_target: TunnelLocalTarget,
        gateway_port: int,
    ) -> None:
        self._route_prefix = route_prefix.rstrip("/")
        self._local_target = local_target
        self._gateway_port = gateway_port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _gateway_handler(self._route_prefix, self._local_target)
        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self._gateway_port), handler)
        except OSError as exc:
            raise TunnelError(
                f"cannot start tunnel gateway on 127.0.0.1:{self._gateway_port}: {exc}"
            ) from exc
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="synth-cloudflared-gateway",
        )
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _gateway_handler(route_prefix: str, target: TunnelLocalTarget) -> type[BaseHTTPRequestHandler]:
    class GatewayHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_GET(self) -> None:
            self._proxy()

        def do_POST(self) -> None:
            self._proxy()

        def do_PUT(self) -> None:
            self._proxy()

        def do_PATCH(self) -> None:
            self._proxy()

        def do_DELETE(self) -> None:
            self._proxy()

        def do_OPTIONS(self) -> None:
            self._proxy()

        def do_HEAD(self) -> None:
            self._proxy()

        def _proxy(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != route_prefix and not parsed.path.startswith(route_prefix + "/"):
                self.send_error(404)
                return
            upstream_path = parsed.path[len(route_prefix) :] or "/"
            upstream_url = urljoin(target.base_url.rstrip("/") + "/", upstream_path.lstrip("/"))
            if parsed.query:
                upstream_url = f"{upstream_url}?{parsed.query}"
            body = self.rfile.read(int(self.headers.get("content-length") or "0"))
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower()
                not in {
                    "connection",
                    "content-length",
                    "host",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailer",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                }
            }
            request = urllib.request.Request(
                upstream_url,
                data=body if body else None,
                headers=headers,
                method=self.command,
            )
            try:
                with urllib.request.urlopen(request, timeout=120.0) as response:
                    self._send_upstream_response(response.status, response.headers, response.read())
            except urllib.error.HTTPError as exc:
                self._send_upstream_response(exc.code, exc.headers, exc.read())
            except urllib.error.URLError as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(502)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

        def _send_upstream_response(
            self,
            status_code: int,
            headers: Mapping[str, Any],
            body: bytes,
        ) -> None:
            self.send_response(status_code)
            for key, value in headers.items():
                if key.lower() in {"connection", "content-length", "transfer-encoding"}:
                    continue
                self.send_header(key, str(value))
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

    return GatewayHandler


def _required_binary(name: str, provider: TunnelProvider) -> str:
    binary = shutil.which(name)
    if binary:
        return binary
    raise TunnelError(
        f"{provider.value} tunnel provider requires the {name!r} binary in PATH"
    )


def _start_cloudflared(binary: str, tunnel_token: str) -> subprocess.Popen[str]:
    if not tunnel_token.strip():
        raise TunnelError("cloudflared tunnel lease response did not include tunnel_token")
    # cloudflared does not read `--token` from stdin (a bare `-` is taken as the
    # literal token value). Pass the token via `--token-file`, which is the
    # documented way to supply it without exposing it in the process argv. The
    # file is created 0600 and removed once cloudflared has consumed it.
    token_fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="synth-cloudflared-", delete=False
    )
    try:
        os.chmod(token_fd.name, 0o600)
        token_fd.write(tunnel_token)
        token_fd.flush()
    finally:
        token_fd.close()
    process = subprocess.Popen(
        [binary, "tunnel", "run", "--token-file", token_fd.name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    time.sleep(1.0)
    try:
        _ensure_process_running(process, "cloudflared")
    finally:
        # cloudflared reads the token file at startup; safe to remove after the
        # liveness check so the secret does not linger on disk.
        try:
            os.unlink(token_fd.name)
        except OSError:
            pass
    return process


def _connect_websocket(url: str, *, headers: Mapping[str, str]) -> Any:
    try:
        import websocket
    except ImportError as exc:
        raise TunnelError(
            "synth_tunnel provider requires the 'websocket-client' package"
        ) from exc
    ws = websocket.WebSocket()
    header_lines = [f"{key}: {value}" for key, value in headers.items()]
    try:
        ws.connect(url, header=header_lines, timeout=10)
    except Exception:
        try:
            ws.close()
        except Exception:
            pass
        raise
    return ws


def _start_ngrok(
    *,
    binary: str,
    authtoken: str,
    local_base_url: str,
    api_addr: str,
) -> subprocess.Popen[str]:
    if not authtoken.strip():
        raise TunnelError("ngrok tunnel lease response did not include tunnel_token")
    env = dict(os.environ)
    env["NGROK_AUTHTOKEN"] = authtoken
    # ngrok v3 removed the `--web-addr` CLI flag; the inspection web/API address
    # is now configured under `agent.web_addr` in a config file. We write a
    # minimal v3 config so each lease gets an isolated inspection API port (the
    # authtoken still comes from NGROK_AUTHTOKEN in the env). The `http` command
    # forwards to a host:port target, not a full URL.
    config_fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="synth-ngrok-", delete=False
    )
    try:
        config_fd.write(f'version: "3"\nagent:\n  web_addr: {api_addr}\n')
        config_fd.flush()
    finally:
        config_fd.close()
    forward_target = local_base_url.split("://", 1)[-1].rstrip("/")
    process = subprocess.Popen(
        [
            binary,
            "http",
            "--log=stdout",
            "--log-format=json",
            "--config",
            config_fd.name,
            forward_target,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    time.sleep(1.0)
    _ensure_process_running(process, "ngrok")
    return process


def _discover_ngrok_public_url(
    process: subprocess.Popen[str],
    api_url: str,
    *,
    timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_error = "ngrok public URL was not reported"
    while time.monotonic() < deadline:
        _ensure_process_running(process, "ngrok")
        try:
            with urllib.request.urlopen(f"{api_url}/api/tunnels", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            tunnels = payload.get("tunnels") if isinstance(payload, Mapping) else None
            if isinstance(tunnels, list):
                urls = [
                    str(item.get("public_url") or "")
                    for item in tunnels
                    if isinstance(item, Mapping)
                ]
                for url in urls:
                    if url.startswith("https://"):
                        return url.rstrip("/")
                for url in urls:
                    if url.startswith("http://"):
                        return url.rstrip("/")
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TunnelError(f"timed out waiting for ngrok public URL: {last_error}")


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _ensure_process_running(process: subprocess.Popen[str] | None, name: str) -> None:
    if process is None:
        raise TunnelError(f"{name} process was not started")
    exit_code = process.poll()
    if exit_code is not None:
        raise TunnelError(f"{name} process exited early with status {exit_code}")


def _wait_for_http_ok(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            if 200 <= exc.code < 300:
                return
            last_error = f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TunnelError(f"timed out waiting for {url}: {last_error}")


def _wait_for_public_dns(hostname: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "no DNS answer"
    while time.monotonic() < deadline:
        try:
            if _cloudflare_dns_has_address(hostname):
                return
            last_error = "no A/AAAA answer"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TunnelError(f"timed out waiting for DNS for {hostname}: {last_error}")


def _cloudflare_dns_has_address(hostname: str) -> bool:
    for query_type in ("A", "AAAA"):
        query = urlencode({"name": hostname, "type": query_type})
        request = urllib.request.Request(
            f"https://cloudflare-dns.com/dns-query?{query}",
            headers={"accept": "application/dns-json"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if int(payload.get("Status") or 0) != 0:
            continue
        answers = payload.get("Answer")
        if isinstance(answers, list) and answers:
            return True
    return False


def _cloudflared_request_headers() -> dict[str, str]:
    return {"User-Agent": _TUNNEL_USER_AGENT}


def _join_health_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "health")


def _required_hostname(url: str, context: str) -> str:
    hostname = urlparse(url).hostname
    if not hostname:
        raise TunnelError(f"{context} did not include a hostname")
    return hostname


def _local_upstream_url(target: TunnelLocalTarget, path: str, query: str) -> str:
    upstream_url = urljoin(target.base_url.rstrip("/") + "/", path.lstrip("/"))
    if query:
        return f"{upstream_url}?{query}"
    return upstream_url


def _request_path(value: Any) -> str:
    path = str(value or "/").strip() or "/"
    return path if path.startswith("/") else f"/{path}"


def _header_pairs(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    headers: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) < 2:
            continue
        name = str(item[0])
        if not name.strip():
            continue
        headers.append((name, str(item[1])))
    return headers


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: str) -> bytes:
    if not data:
        return b""
    return base64.b64decode(data.encode("ascii"))


def _route_prefix_from_public_url(public_url: str) -> str:
    parsed = urlparse(public_url)
    path = parsed.path.rstrip("/")
    if not path:
        raise TunnelError("cloudflared public_url did not include a route prefix")
    return path


def _gateway_port_from_mode(connector_mode: str | None) -> int:
    if not connector_mode:
        return 8016
    marker = "gateway_port="
    if marker not in connector_mode:
        return 8016
    raw = connector_mode.split(marker, 1)[1].split(":", 1)[0].strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise TunnelError(f"invalid cloudflared gateway port {raw!r}") from exc


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _response_text(payload: Mapping[str, Any], field_name: str, context: str) -> str:
    return _required_text(payload.get(field_name), f"{context} {field_name}")


def _required_text(value: Any, context: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TunnelError(f"{context} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _required_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TunnelError(f"{context} is required")
