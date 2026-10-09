"""Shared connector job leases; physical attachment remains owned by its service.

See backend notes/specifications/tanha/current/systems/tunnels/v2_catalog.md.
"""

from __future__ import annotations

import secrets
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from .tunnel_custody import CallerLeaseCustody
from .tunnel_custody_journal import FileLeaseCustodyJournal, TunnelCustodyInDoubt
from .tunnels import (
    SynthTunnelLease,
    TunnelError,
    TunnelProvider,
    _join_health_url,
    _wait_for_http_ok,
    parse_local_target,
)


class SharedSynthTunnelLease(SynthTunnelLease):
    """A job authorization handle with no right to stop the shared connector."""

    def __init__(
        self,
        *,
        connector: UUID,
        lease: UUID,
        public_url: str,
        worker_token: str,
        local_url: str,
        client,
        expires_at: str,
        custody: CallerLeaseCustody,
    ) -> None:
        super().__init__(
            provider=TunnelProvider.SYNTH_TUNNEL,
            public_url=public_url,
            lease_id=str(lease),
            worker_token=worker_token,
            local_target=parse_local_target(local_url),
            client=client,
            expires_at=expires_at,
            owns_lease=True,
        )
        self.connector = connector
        self.connector_mode = "shared_connector_v2"
        self.agent_connect_required = False
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        self._custody = custody
        self.receipt_path = custody.journal.path

    def hosted_auth_refresh(self) -> dict:
        """Only this job lease is delegated to the enrolled hosted executor."""
        descriptor = {
            "provider": "synth_tunnel_v2",
            "connector_id": str(self.connector),
            "lease_id": self.lease_id,
            "refresh_interval_seconds": 30,
        }
        offer = self._custody.receipt.get("offer_request")
        if offer:
            descriptor.update(offer_id=offer["offer"], job_id=offer["job_id"])
        return descriptor

    def prepare_handoff(self, job_id: str | None = None) -> dict:
        with self._credentials_lock:
            if self._closed:
                raise TunnelError("shared tunnel lease is closed")
            return self._custody.prepare_handoff(job_id)

    def handoff_status(self) -> dict:
        with self._credentials_lock:
            return self._custody.handoff_status()

    def submit_once(self, payload: dict) -> dict:
        with self._credentials_lock:
            if self._closed:
                raise TunnelError("shared tunnel lease is closed")
            return self._custody.submit_once(payload)

    def submission_status(self) -> dict:
        with self._credentials_lock:
            return self._custody.submission_status()

    def refresh_worker_token(self) -> str:
        with self._credentials_lock:
            if self._closed:
                raise TunnelError("shared tunnel lease is closed")
            payload = self.client._json_request(
                "POST",
                f"/api/v1/synthtunnel/v2/connectors/{self.connector}/leases/{self.lease_id}/forward-capability",
                {"ttl_seconds": 300},
            )
            token = str(payload.get("token") or "")
            if not token:
                raise TunnelError("shared lease capability response missing token")
            self.worker_token = token
            return token

    def heartbeat_once(self) -> str:
        return self.refresh_worker_token()

    def send_heartbeat(self) -> str:
        """Refresh only v2 authority; never call the legacy attach/heartbeat path."""
        return self.refresh_worker_token()

    def wait_ready(self, timeout_seconds: float = 60) -> None:
        _wait_for_http_ok(
            _join_health_url(self.public_url),
            headers={"Authorization": "Bearer " + self.worker_token},
            timeout_seconds=timeout_seconds,
        )
        with self._credentials_lock:
            if self._closed:
                raise TunnelError("shared tunnel lease is closed")
            if self._refresh_thread is None:
                self._refresh_thread = threading.Thread(
                    target=self._refresh, name="shared-tunnel-capability-refresh", daemon=True
                )
                self._refresh_thread.start()

    def _refresh(self) -> None:
        import logging

        while not self._stop_refresh.wait(30):
            try:
                self.refresh_worker_token()
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "shared tunnel capability refresh failed error_kind=%s", type(error).__name__
                )
                self._stop_refresh.set()
                return

    def close(self) -> None:
        with self._credentials_lock:
            if self._closed:
                return
            self._stop_refresh.set()
            self._custody.close()
            self._closed = True
        # No agent.stop(), registration removal, or shared connector shutdown.


def borrow_shared_synth_tunnel(
    client,
    *,
    connector: UUID,
    routes: tuple[UUID, ...],
    route_name: str,
    gateway_url: str,
    local_url: str,
    requested_ttl_seconds: int = 3600,
    custody_directory: Path | str | None = None,
) -> SharedSynthTunnelLease:
    """Acquire a distinct job lease on registered routes without owning their lifetime."""
    import re
    from urllib.parse import urlparse

    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", route_name):
        raise TunnelError("invalid shared tunnel route name")
    gateway = urlparse(gateway_url)
    if (
        gateway.scheme not in {"http", "https"}
        or not gateway.hostname
        or gateway.username
        or gateway.password
        or gateway.query
        or gateway.fragment
    ):
        raise TunnelError("invalid shared tunnel gateway")
    if type(requested_ttl_seconds) is not int or not 1 <= requested_ttl_seconds <= 86400:
        raise TunnelError("invalid shared tunnel lease duration")
    if (
        connector.int == 0
        or not 1 <= len(routes) <= 64
        or len(set(routes)) != len(routes)
        or any(route.int == 0 for route in routes)
    ):
        raise TunnelError("invalid shared tunnel identities")
    parse_local_target(local_url)
    status = client._json_request("GET", f"/api/v1/synthtunnel/v2/connectors/{connector}")
    lease = uuid4()
    expiry = datetime.now(UTC) + timedelta(seconds=requested_ttl_seconds)
    request = {
        "command_id": str(uuid4()),
        "expected_revision": status["revision"],
        "deadline_at": (datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
        "lease_id": str(lease),
        "route_token": "rt_" + secrets.token_urlsafe(24),
        "routes": [str(route) for route in routes],
        "expires_at": expiry.isoformat(),
        "owner_binding": str(uuid4()),
        "owner_kind": "job",
        "max_inflight": 64,
    }
    journal = FileLeaseCustodyJournal(
        Path(custody_directory or Path.cwd() / ".synth/tunnel-leases"), lease
    )
    receipt = journal.save(
        {
            "connector": str(connector),
            "phase": "grant_in_doubt",
            "grant_request": request,
            "public_url": f"{gateway_url.rstrip('/')}/v2/leases/{lease}/routes/{route_name}",
            "local_url": local_url,
        }
    )
    custody = CallerLeaseCustody(client, journal, receipt)
    try:
        client._json_request("POST", custody.connector_path + "/leases", request)
    except Exception as error:
        raise TunnelCustodyInDoubt(journal.path, "grant") from error
    custody.settle_grant()
    capability = client._json_request(
        "POST",
        f"/api/v1/synthtunnel/v2/connectors/{connector}/leases/{lease}/forward-capability",
        {"ttl_seconds": 300},
    )
    return SharedSynthTunnelLease(
        connector=connector,
        lease=lease,
        public_url=f"{gateway_url.rstrip('/')}/v2/leases/{lease}/routes/{route_name}",
        worker_token=capability["token"],
        local_url=local_url,
        client=client,
        expires_at=expiry.isoformat(),
        custody=custody,
    )


def recover_shared_synth_tunnel(client, receipt_path: Path | str) -> SharedSynthTunnelLease:
    """Read retained custody and current authority; never repeat a control effect."""
    path = Path(receipt_path)
    journal = FileLeaseCustodyJournal(path.parent, UUID(path.stem))
    if path.name != journal.path.name:
        raise TunnelError("invalid custody receipt filename")
    custody = CallerLeaseCustody(client, journal, journal.read())
    if custody.receipt["phase"] in {"closed", "detached", "close_in_doubt"}:
        custody.close()
        raise TunnelError("shared tunnel handle is closed")
    current = custody.settle_grant()
    if custody.receipt.get("offer_request"):
        custody.handoff_status()
    capability = client._json_request(
        "POST", custody.lease_path + "/forward-capability", {"ttl_seconds": 300}
    )
    return SharedSynthTunnelLease(
        connector=UUID(custody.receipt["connector"]),
        lease=journal.lease,
        public_url=custody.receipt["public_url"],
        local_url=custody.receipt["local_url"],
        worker_token=capability["token"],
        expires_at=current["expires_at"],
        client=client,
        custody=custody,
    )
