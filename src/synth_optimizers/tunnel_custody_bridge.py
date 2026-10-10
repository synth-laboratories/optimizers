"""Private bounded stdio adapter for Workshop; all authority stays in the API.

Credentials arrive on stdin, never argv. Receipts remain private and fsynced.
No action starts/stops a connector, origin, actor service or optimizer process.
"""
from __future__ import annotations

import json
import re
import sys
import time
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5

from .hosted import _with_tunnel_container
from .tunnel_custody_transport import CustodyApiClient
from .shared_tunnels import borrow_shared_synth_tunnel, recover_shared_synth_tunnel
from .tunnel_custody import CallerLeaseCustody
from .tunnel_custody_journal import FileLeaseCustodyJournal, TunnelCustodyInDoubt, MAX_RECEIPTS
from .tunnels import TunnelError

MAX_MESSAGE_BYTES = 262144
MAX_CLOSE_SECONDS = 10


def dispatch(request: dict, *, client=None) -> dict:
    if not isinstance(request, dict) or type(request.get("version")) is not int or request.get("version") != 1:
        raise TunnelError("invalid custody bridge version")
    action = request.get("action")
    if action not in {"submit", "close_all"}:
        raise TunnelError("invalid custody bridge action")
    allowed = {"version", "action", "backend_url", "api_key", "directory", "binding", "payload"}
    if set(request) - allowed:
        raise TunnelError("unknown custody bridge field")
    directory = Path(request["directory"])
    if not directory.is_absolute():
        raise TunnelError("custody directory must be absolute")
    client = client or CustodyApiClient(
        backend_url=request["backend_url"], api_key=request["api_key"],
        timeout_seconds=2 if action == "close_all" else 30,
    )
    if action == "close_all":
        return close_all(client, directory)
    binding = request["binding"]
    keys = {"connector_id", "route_ids", "route_name", "gateway_url", "local_url", "requested_ttl_seconds"}
    if not isinstance(binding, dict) or set(binding) - keys:
        raise TunnelError("invalid shared connector binding")
    payload = dict(request["payload"])
    job = payload.get("run_id")
    if not isinstance(job, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job):
        raise TunnelError("custody submit requires a stable job identity")
    if payload.get("algorithm") not in {"gepa", "go-ex"}:
        raise TunnelError("shared custody requires qualified parent/child mapping for this algorithm")
    if set(payload) - {"algorithm", "run_id", "idempotency_key", "project_id", "config_json", "config_toml"}:
        raise TunnelError("unknown shared custody submission field")
    if payload.get("idempotency_key", job) != job:
        raise TunnelError("custody submission identity mismatch")
    if "config_json" in payload and "config_toml" in payload:
        raise TunnelError("ambiguous optimizer config")
    config = payload.get("config_json")
    if config is None:
        config = tomllib.loads(payload.pop("config_toml"))
    if not isinstance(config, dict):
        raise TunnelError("optimizer config must be an object")
    config = {**config, "run": dict(config.get("run") or {})}
    if config["run"].get("run_id", job) != job:
        raise TunnelError("optimizer config job identity mismatch")
    config["run"]["run_id"] = job
    connector = UUID(binding["connector_id"])
    routes = tuple(UUID(value) for value in binding["route_ids"])
    identity = uuid5(NAMESPACE_URL, f"synth-workshop-custody:{directory}:{connector}:{job}")
    receipt_path = directory / f"{identity}.json"
    if receipt_path.exists():
        handle = recover_shared_synth_tunnel(client, receipt_path)
        if handle._custody.receipt.get("job_hint") not in {None, job}:
            raise TunnelError("custody retained job identity mismatch")
        expected_url = f"{binding['gateway_url'].rstrip('/')}/v2/leases/{identity}/routes/{binding['route_name']}"
        if handle.public_url != expected_url or handle._custody.receipt["local_url"] != binding["local_url"]:
            raise TunnelError("custody origin binding mismatch")
        if set(handle._custody.receipt["grant_request"]["routes"]) != set(map(str, routes)):
            raise TunnelError("custody route binding mismatch")
    else:
        handle = borrow_shared_synth_tunnel(
            client, connector=connector, routes=routes, route_name=binding["route_name"],
            gateway_url=binding["gateway_url"], local_url=binding["local_url"],
            requested_ttl_seconds=binding.get("requested_ttl_seconds", 3600),
            custody_directory=directory, lease_identity=identity, job_identity=job,
        )
    handle.prepare_handoff(job)
    payload["idempotency_key"] = job
    payload["config_json"] = _with_tunnel_container(config, handle)
    handle.submit_once(payload)
    # ACK is not custody acceptance. Keep the caller handle in the receipt;
    # application exit will query canonical acceptance before releasing it.
    return {"version": 1, "receipt": str(identity), "response": handle._custody.receipt["submission"]["response"]}


def close_all(client, directory: Path) -> dict:
    if not directory.exists():
        return {"version": 1, "closed": 0, "unresolved": 0}
    paths = list(islice(directory.glob("*.json"), MAX_RECEIPTS + 1))
    if len(paths) > MAX_RECEIPTS:
        raise TunnelError("custody directory capacity exceeded")
    deadline = time.monotonic() + MAX_CLOSE_SECONDS
    connector_locks = {}
    registry_lock = threading.Lock()

    def close(path):
        if time.monotonic() >= deadline:
            return False
        try:
            identity = UUID(path.stem)
            journal = FileLeaseCustodyJournal(directory, identity)
            receipt = journal.read()
            with registry_lock:
                lock = connector_locks.setdefault(receipt["connector"], threading.Lock())
            # Catalog commands share a connector revision. Serialize only this
            # invocation's closes on each connector; unrelated connectors proceed.
            if not lock.acquire(timeout=max(0, deadline - time.monotonic())):
                return False
            try:
                if time.monotonic() >= deadline:
                    return False
                custody = CallerLeaseCustody(client, journal, receipt)
                custody.close()  # accepted -> local detach; caller-owned -> exact revoke
                return True
            finally:
                lock.release()
        except Exception:
            return False  # retain unresolved receipts; never infer termination

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="tunnel-custody-close") as pool:
        results = list(pool.map(close, paths))
    return {"version": 1, "closed": sum(results), "unresolved": len(results) - sum(results)}


def main() -> int:
    try:
        encoded = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise TunnelError("custody request byte capacity exceeded")
        result = dispatch(json.loads(encoded))
        output = json.dumps(result, allow_nan=False).encode()
        if len(output) > MAX_MESSAGE_BYTES:
            raise TunnelError("custody response byte capacity exceeded")
        sys.stdout.buffer.write(output + b"\n")
        return 0
    except TunnelCustodyInDoubt as error:
        result = {"version": 1, "error": "custody_in_doubt", "phase": error.phase, "receipt": error.receipt_path.stem}
    except Exception as error:
        # Exceptions can contain upstream authority or user config. Only the
        # bounded error class crosses this diagnostic channel.
        result = {"version": 1, "error": "custody_refused", "kind": type(error).__name__}
    sys.stdout.write(json.dumps(result) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
