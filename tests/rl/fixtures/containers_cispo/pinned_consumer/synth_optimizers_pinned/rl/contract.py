"""Declared container contract: routes are advertised, never guessed.

A container publishes a versioned block at ``metadata.optimizer_contracts.cispo``
in its ``/metadata`` response. This module parses that block, refuses it unless
every mandatory route is present and absolute, resolves route templates, and
hashes the result so a group pin can name the exact contract it ran under.

The executor calls only declared routes. A container may rename any of them; it
may not omit a mandatory one and it may not expect a path to be inferred.

Transport is deliberately behind an interface: the queue engine and the batch
assembler depend on ``ContainerClient``, not on HTTP. One concrete urllib
implementation lives here, with the bearer/header and transient-retry behavior
the Rust GEPA client already uses.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.rl_records import digest

CISPO_CONTRACT_VERSION = "synth_optimizers.cispo.v1"
CONTRACT_METADATA_KEY = "cispo"

MAX_RESPONSE_BYTES = 8_388_608

# Every declared key, the placeholders it must carry, and the method the
# executor will use. A route with an unexpected placeholder is a rejection: the
# executor has no value to substitute for it.
ROUTE_PARAMETERS: dict[str, tuple[str, ...]] = {
    "health_route": (),
    "capabilities_route": (),
    "handshake_route": (),
    "taskset_route": (),
    "taskset_tasks_route": (),
    "topology_route": ("topology_id",),
    "policy_bind_route": (),
    "policy_set_bind_route": (),
    "rollout_route": (),
    "rollout_state_route": ("rollout_id",),
    "rollout_events_route": ("rollout_id",),
    "rollout_renew_route": ("rollout_id",),
    "rollout_finalize_route": ("rollout_id",),
    "rollout_terminate_route": ("rollout_id",),
    "trace_route": ("rollout_id",),
    "artifacts_route": ("rollout_id",),
    "reward_route": (),
}

MANDATORY_ROUTES: tuple[str, ...] = tuple(ROUTE_PARAMETERS)

ROUTE_METHODS: dict[str, str] = {
    "health_route": "GET",
    "capabilities_route": "GET",
    "handshake_route": "POST",
    "taskset_route": "GET",
    "taskset_tasks_route": "POST",
    "topology_route": "GET",
    "policy_bind_route": "POST",
    "policy_set_bind_route": "POST",
    "rollout_route": "POST",
    "rollout_state_route": "GET",
    "rollout_events_route": "GET",
    "rollout_renew_route": "POST",
    "rollout_finalize_route": "POST",
    "rollout_terminate_route": "POST",
    "trace_route": "GET",
    "artifacts_route": "GET",
    "reward_route": "GET",
}

ROUTE_PLACEHOLDERS: frozenset[str] = frozenset({"rollout_id", "topology_id"})


class ContractError(ValueError):
    """A container's declared contract is absent, stale, or malformed."""


class RouteError(ContractError):
    """A declared route cannot be resolved for this call."""


class TransportError(RuntimeError):
    """The container could not be reached. Never a data fallback."""


class ContainerStatusError(TransportError):
    """The container answered with a non-success status. Its reply, verbatim."""

    def __init__(self, path: str, status: int, body: str) -> None:
        super().__init__(f"container {path} returned status {status}: {body[:1000]}")
        self.path = path
        self.status = status
        self.body = body[:1000]


class ContainerAuthError(ContractError):
    """Bearer configuration named an environment variable that is not set."""


def _placeholders(route: str) -> tuple[str, ...]:
    found: list[str] = []
    rest = route
    while "{" in rest:
        head, _, rest = rest.partition("{")
        del head
        name, closer, rest = rest.partition("}")
        if not closer:
            raise ContractError(f"route {route!r} has an unterminated placeholder")
        found.append(name)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class RouteTable:
    """The declared route map, validated, with template substitution."""

    routes: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in MANDATORY_ROUTES:
            route = self.routes.get(name)
            if not isinstance(route, str) or not route.strip():
                raise ContractError(
                    f"metadata.optimizer_contracts.cispo.{name} is required "
                    "and may not be omitted"
                )
            route = route.strip()
            if not route.startswith("/"):
                raise ContractError(
                    f"metadata.optimizer_contracts.cispo.{name} must be an absolute "
                    f"route, got {route!r}"
                )
            declared = _placeholders(route)
            unknown = tuple(item for item in declared if item not in ROUTE_PLACEHOLDERS)
            if unknown:
                raise ContractError(
                    f"metadata.optimizer_contracts.cispo.{name} declares placeholders "
                    f"the executor cannot substitute: {unknown}"
                )
            expected = ROUTE_PARAMETERS[name]
            missing = tuple(item for item in expected if item not in declared)
            if missing:
                raise ContractError(
                    f"metadata.optimizer_contracts.cispo.{name} must address "
                    f"{missing} in its path"
                )

    @property
    def declared(self) -> Mapping[str, str]:
        return {name: str(self.routes[name]).strip() for name in MANDATORY_ROUTES}

    def method(self, name: str) -> str:
        if name not in ROUTE_METHODS:
            raise RouteError(f"unknown route {name!r}")
        return ROUTE_METHODS[name]

    def route(self, name: str) -> str:
        if name not in ROUTE_PARAMETERS:
            raise RouteError(f"unknown route {name!r}")
        return str(self.routes[name]).strip()

    def resolve(
        self,
        name: str,
        *,
        rollout_id: str | None = None,
        topology_id: str | None = None,
    ) -> str:
        """Substitute the declared template. A leftover placeholder is an error."""

        route = self.route(name)
        values = {"rollout_id": rollout_id, "topology_id": topology_id}
        for placeholder in ROUTE_PARAMETERS[name]:
            value = values.get(placeholder)
            if value is None or not str(value).strip():
                raise RouteError(f"route {name!r} needs {placeholder!r} to resolve")
            route = route.replace(
                "{" + placeholder + "}",
                urllib.parse.quote(str(value).strip(), safe=""),
            )
        if "{" in route:
            raise RouteError(f"route {name!r} still has an unresolved placeholder: {route!r}")
        return route


def _select_contract_block(contracts: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find the block that declares this contract, by version rather than key.

    A container may already publish something under ``cispo``: the predecessor
    training block is read there by a live lane, and overwriting it would point
    that lane at these routes. So the key is a convention and the version is
    the identity — the executor accepts the declaration wherever it is
    advertised, provided it says what it is.
    """

    preferred = contracts.get(CONTRACT_METADATA_KEY)
    if isinstance(preferred, Mapping) and _declares_this_contract(preferred):
        return preferred
    for key in sorted(str(name) for name in contracts):
        block = contracts.get(key)
        if isinstance(block, Mapping) and _declares_this_contract(block):
            return block
    return None


def _declares_this_contract(block: Mapping[str, Any]) -> bool:
    return str(block.get("version") or "").strip() == CISPO_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class ContainerContract:
    """A parsed, validated, hashed contract advertisement."""

    version: str
    route_table: RouteTable
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def contract_hash(self) -> str:
        """Stable over the version and the declared route map, nothing else."""

        return "sha256:" + digest(
            {
                "version": self.version,
                "routes": dict(sorted(self.route_table.declared.items())),
            }
        )

    def resolve(
        self,
        name: str,
        *,
        rollout_id: str | None = None,
        topology_id: str | None = None,
    ) -> str:
        return self.route_table.resolve(name, rollout_id=rollout_id, topology_id=topology_id)

    @classmethod
    def from_block(cls, block: Any) -> "ContainerContract":
        if not isinstance(block, Mapping):
            raise ContractError(
                "metadata.optimizer_contracts.cispo must be an object"
            )
        version = block.get("version")
        if not isinstance(version, str) or version.strip() != CISPO_CONTRACT_VERSION:
            raise ContractError(
                "container does not advertise metadata.optimizer_contracts.cispo."
                f"version={CISPO_CONTRACT_VERSION}, got {version!r}"
            )
        routes = {
            name: value
            for name, value in block.items()
            if name.endswith("_route") and isinstance(value, str)
        }
        extra = {
            name: value
            for name, value in block.items()
            if name != "version" and name not in routes
        }
        return cls(version=version.strip(), route_table=RouteTable(routes=routes), extra=extra)

    @classmethod
    def from_metadata(cls, payload: Any) -> "ContainerContract":
        """Parse a whole ``/metadata`` document."""

        if not isinstance(payload, Mapping):
            raise ContractError("container /metadata must be an object")
        metadata = payload.get("metadata", payload)
        if not isinstance(metadata, Mapping):
            raise ContractError("container /metadata.metadata must be an object")
        contracts = metadata.get("optimizer_contracts")
        if not isinstance(contracts, Mapping):
            raise ContractError(
                "container metadata must advertise metadata.optimizer_contracts"
            )
        block = _select_contract_block(contracts)
        if block is None:
            raise ContractError(
                "container metadata advertises no block declaring "
                f"version={CISPO_CONTRACT_VERSION!r}; looked at "
                f"metadata.optimizer_contracts.{CONTRACT_METADATA_KEY} and every "
                f"sibling key. Found: {sorted(str(key) for key in contracts)}"
            )
        return cls.from_block(block)


class ContainerClient(ABC):
    """One method per declared route, so callers never speak HTTP.

    Streams that consume a container depend on this interface. A fake, a
    recorded transcript, and the urllib implementation are interchangeable.
    """

    @property
    @abstractmethod
    def contract(self) -> ContainerContract:
        """The contract this client was built against."""

    @abstractmethod
    def health(self) -> Mapping[str, Any]:
        """Liveness, container version, image digest."""

    @abstractmethod
    def metadata(self) -> Mapping[str, Any]:
        """The raw advertisement, for receipt persistence."""

    @abstractmethod
    def capabilities(self) -> Mapping[str, Any]:
        """The hashed capability document."""

    @abstractmethod
    def handshake(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post the requirement document, receive per-clause verdicts."""

    @abstractmethod
    def taskset(self) -> Mapping[str, Any]:
        """Taskset id, version, declared splits."""

    @abstractmethod
    def taskset_tasks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """One duplicate-free row per requested id, each naming its topology."""

    @abstractmethod
    def topology(self, topology_id: str) -> Mapping[str, Any]:
        """Full instance roster, teams, channels, turn and actuation model."""

    @abstractmethod
    def bind_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind one instance's sampler without embedding credentials."""

    @abstractmethod
    def bind_policy_set(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """One atomic binding for every instance in a joint episode."""

    @abstractmethod
    def submit_rollout(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Idempotent asynchronous submission."""

    @abstractmethod
    def rollout_state(self, rollout_id: str) -> Mapping[str, Any]:
        """State, lease expiry, per-instance liveness."""

    @abstractmethod
    def rollout_events(self, rollout_id: str, *, cursor: str | None = None) -> Mapping[str, Any]:
        """Ordered events with a monotone resumable cursor."""

    @abstractmethod
    def renew_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """New lease expiry."""

    @abstractmethod
    def finalize_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Horizon-clipped snapshot plus the quiescence attestation."""

    @abstractmethod
    def terminate_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Terminal cancellation, exactly once."""

    @abstractmethod
    def trace(self, rollout_id: str) -> Mapping[str, Any]:
        """Sealed evidence inline, or a reference plus digest."""

    @abstractmethod
    def artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        """Artifact inventory with digests and fetch handles."""

    @abstractmethod
    def reward(
        self,
        rollout_id: str,
        *,
        request: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Receipt bound to the rollout id and the sealed trace digest."""


@dataclass(frozen=True, slots=True)
class HttpReply:
    """A status and a body. A status is a real reply, never a retry trigger."""

    status: int
    body: bytes


Sender = Callable[[urllib.request.Request, float], HttpReply]


def _urllib_send(request: urllib.request.Request, timeout: float, *, max_response_bytes: int = MAX_RESPONSE_BYTES) -> HttpReply:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HttpReply(status=int(response.status), body=response.read(max_response_bytes + 1))
    except urllib.error.HTTPError as exc:  # a real server reply
        return HttpReply(status=int(exc.code), body=exc.read(max_response_bytes + 1))


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Transient transport failures only. A 4xx/5xx status is never retried."""

    max_attempts: int = 4
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ContractError("retry policy needs at least one attempt")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ContractError("retry backoff must be non-negative")


class UrllibContainerClient(ContainerClient):
    """Declared-route client over urllib, with bearer and header support."""

    def __init__(
        self,
        base_url: str,
        contract: ContainerContract,
        *,
        headers: Mapping[str, str] | None = None,
        auth_bearer_env: str | None = None,
        timeout_seconds: float = 30.0,
        retry: RetryPolicy | None = None,
        sender: Sender | None = None,
        sleep: Callable[[float], None] | None = None,
        environ: Mapping[str, str] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ContractError(f"container base url must be http(s), got {base_url!r}")
        self._base_url = base_url.rstrip("/")
        self._contract = contract
        self._headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self._auth_bearer_env = (auth_bearer_env or "").strip() or None
        self._timeout_seconds = float(timeout_seconds)
        self._retry = retry or RetryPolicy()
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 67_108_864:
            raise ContractError('response byte limit must be an integer in 1..67108864')
        self._max_response_bytes = max_response_bytes
        self._sender: Sender = sender or (lambda request, timeout: _urllib_send(request, timeout, max_response_bytes=max_response_bytes))
        self._sleep = sleep or time.sleep
        self._environ = environ if environ is not None else os.environ

    @property
    def contract(self) -> ContainerContract:
        return self._contract

    def _request_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", **self._headers}
        lowered = {name.lower() for name in headers}
        if self._auth_bearer_env and "authorization" not in lowered:
            token = (self._environ.get(self._auth_bearer_env) or "").strip()
            if not token:
                raise ContainerAuthError(
                    f"auth_bearer_env references missing environment variable "
                    f"{self._auth_bearer_env!r}"
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _send(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        headers = self._request_headers()
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        backoff = self._retry.initial_backoff_seconds
        last: Exception | None = None
        for attempt_index in range(self._retry.max_attempts):
            request = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                reply = self._sender(request, self._timeout_seconds)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # Transport-level failure under concurrent container load. The
                # identical request is re-issued; nothing is degraded or faked.
                last = exc
                if attempt_index + 1 == self._retry.max_attempts:
                    raise TransportError(f"container {path} unreachable: {exc}") from exc
                self._sleep(backoff)
                backoff = min(backoff * 2, self._retry.max_backoff_seconds)
                continue
            return self._decode(path, reply, max_response_bytes=self._max_response_bytes)
        raise TransportError(f"container {path} unreachable: {last}")

    @staticmethod
    def _decode(path: str, reply: HttpReply, *, max_response_bytes: int = MAX_RESPONSE_BYTES) -> Mapping[str, Any]:
        if reply.status < 200 or reply.status >= 300:
            raise ContainerStatusError(path, reply.status, reply.body.decode("utf-8", "replace"))
        if len(reply.body) > max_response_bytes:
            raise TransportError(f"container {path} response exceeded {max_response_bytes} bytes")
        text = reply.body.decode("utf-8", "replace").strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TransportError(f"container {path} returned invalid json: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise TransportError(f"container {path} returned a non-object body")
        return decoded

    def health(self) -> Mapping[str, Any]:
        return self._send("GET", self._contract.resolve("health_route"))

    def metadata(self) -> Mapping[str, Any]:
        return self._send("GET", "/metadata")

    def capabilities(self) -> Mapping[str, Any]:
        return self._send("GET", self._contract.resolve("capabilities_route"))

    def handshake(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST", self._contract.resolve("handshake_route"), payload=request
        )

    def taskset(self) -> Mapping[str, Any]:
        return self._send("GET", self._contract.resolve("taskset_route"))

    def taskset_tasks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST", self._contract.resolve("taskset_tasks_route"), payload=request
        )

    def topology(self, topology_id: str) -> Mapping[str, Any]:
        return self._send(
            "GET", self._contract.resolve("topology_route", topology_id=topology_id)
        )

    def bind_policy(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST", self._contract.resolve("policy_bind_route"), payload=request
        )

    def bind_policy_set(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST", self._contract.resolve("policy_set_bind_route"), payload=request
        )

    def submit_rollout(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send("POST", self._contract.resolve("rollout_route"), payload=request)

    def rollout_state(self, rollout_id: str) -> Mapping[str, Any]:
        return self._send(
            "GET", self._contract.resolve("rollout_state_route", rollout_id=rollout_id)
        )

    def rollout_events(self, rollout_id: str, *, cursor: str | None = None) -> Mapping[str, Any]:
        return self._send(
            "GET",
            self._contract.resolve("rollout_events_route", rollout_id=rollout_id),
            query={"cursor": cursor} if cursor else None,
        )

    def renew_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST",
            self._contract.resolve("rollout_renew_route", rollout_id=rollout_id),
            payload=request,
        )

    def finalize_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST",
            self._contract.resolve("rollout_finalize_route", rollout_id=rollout_id),
            payload=request,
        )

    def terminate_rollout(self, rollout_id: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._send(
            "POST",
            self._contract.resolve("rollout_terminate_route", rollout_id=rollout_id),
            payload=request,
        )

    def trace(self, rollout_id: str) -> Mapping[str, Any]:
        return self._send("GET", self._contract.resolve("trace_route", rollout_id=rollout_id))

    def artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        return self._send(
            "GET", self._contract.resolve("artifacts_route", rollout_id=rollout_id)
        )

    def reward(
        self,
        rollout_id: str,
        *,
        request: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        path = self._contract.resolve("reward_route")
        if request is not None:
            return self._send("POST", path, payload=request)
        return self._send("GET", path, query={"rollout_id": rollout_id})


def preflight_contract(metadata: Any, *, expected_routes: Sequence[str] = ()) -> ContainerContract:
    """Parse the advertisement and, optionally, assert extra declared routes."""

    contract = ContainerContract.from_metadata(metadata)
    for name in expected_routes:
        if name not in contract.route_table.routes:
            raise ContractError(f"container does not declare route {name!r}")
    return contract
