"""Stream 2: the declared contract, its route table, and the urllib client."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from synth_optimizers.rl.contract import (
    CISPO_CONTRACT_VERSION,
    MANDATORY_ROUTES,
    ContainerAuthError,
    ContainerContract,
    ContainerStatusError,
    ContractError,
    HttpReply,
    RetryPolicy,
    RouteError,
    TransportError,
    UrllibContainerClient,
    preflight_contract,
)

DECLARED_ROUTES: dict[str, str] = {
    "health_route": "/health",
    "capabilities_route": "/training/capabilities",
    "handshake_route": "/training/handshake",
    "taskset_route": "/taskset",
    "taskset_tasks_route": "/taskset/tasks",
    "topology_route": "/topologies/{topology_id}",
    "policy_bind_route": "/policy-configs",
    "policy_set_bind_route": "/policy-sets",
    "rollout_route": "/rollout",
    "rollout_state_route": "/rollouts/{rollout_id}",
    "rollout_events_route": "/rollouts/{rollout_id}/events",
    "rollout_renew_route": "/rollouts/{rollout_id}/renew",
    "rollout_finalize_route": "/rollouts/{rollout_id}/finalize",
    "rollout_terminate_route": "/rollouts/{rollout_id}/terminate",
    "trace_route": "/rollouts/{rollout_id}/trace",
    "artifacts_route": "/rollouts/{rollout_id}/artifacts",
    "reward_route": "/reward",
}


def metadata(**overrides: object) -> dict[str, object]:
    block: dict[str, object] = {"version": CISPO_CONTRACT_VERSION, **DECLARED_ROUTES}
    block.update(overrides)
    return {"metadata": {"optimizer_contracts": {"cispo": block}}}


class RecordingSender:
    """A local fake transport. No socket is opened by these tests."""

    def __init__(self, replies: list[object] | None = None) -> None:
        self.replies = replies or []
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> HttpReply:
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else HttpReply(200, b"{}")
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, HttpReply)
        return reply

    @property
    def paths(self) -> list[str]:
        return [item.full_url for item in self.requests]


def client(
    *,
    sender: RecordingSender,
    sleeps: list[float] | None = None,
    headers: dict[str, str] | None = None,
    auth_bearer_env: str | None = None,
    environ: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
) -> UrllibContainerClient:
    return UrllibContainerClient(
        "http://127.0.0.1:8080",
        ContainerContract.from_metadata(metadata()),
        headers=headers,
        auth_bearer_env=auth_bearer_env,
        environ=environ or {},
        retry=retry,
        sender=sender,
        sleep=(sleeps if sleeps is None else sleeps.append),
    )


def test_full_advertisement_parses_and_declares_every_mandatory_route() -> None:
    contract = preflight_contract(metadata())
    assert contract.version == CISPO_CONTRACT_VERSION
    assert set(contract.route_table.declared) == set(MANDATORY_ROUTES)
    assert len(MANDATORY_ROUTES) == 17


@pytest.mark.parametrize("route_name", MANDATORY_ROUTES)
def test_each_mandatory_route_missing_is_refused(route_name: str) -> None:
    block = dict(DECLARED_ROUTES)
    del block[route_name]
    payload = {
        "metadata": {
            "optimizer_contracts": {"cispo": {"version": CISPO_CONTRACT_VERSION, **block}}
        }
    }
    with pytest.raises(ContractError) as excinfo:
        ContainerContract.from_metadata(payload)
    assert route_name in str(excinfo.value)


@pytest.mark.parametrize("route_name", MANDATORY_ROUTES)
def test_each_relative_route_is_refused(route_name: str) -> None:
    relative = DECLARED_ROUTES[route_name].lstrip("/")
    with pytest.raises(ContractError) as excinfo:
        ContainerContract.from_metadata(metadata(**{route_name: relative}))
    assert "absolute route" in str(excinfo.value)


def test_wrong_contract_version_is_refused() -> None:
    with pytest.raises(ContractError) as excinfo:
        ContainerContract.from_metadata(metadata(version="synth_optimizers.cispo.v0"))
    assert CISPO_CONTRACT_VERSION in str(excinfo.value)


def test_absent_cispo_block_is_refused() -> None:
    with pytest.raises(ContractError):
        ContainerContract.from_metadata({"metadata": {"optimizer_contracts": {"gepa": {}}}})
    with pytest.raises(ContractError):
        ContainerContract.from_metadata({"metadata": {}})


def test_route_without_its_required_placeholder_is_refused() -> None:
    with pytest.raises(ContractError) as excinfo:
        ContainerContract.from_metadata(metadata(trace_route="/rollouts/trace"))
    assert "rollout_id" in str(excinfo.value)


def test_route_with_an_unsubstitutable_placeholder_is_refused() -> None:
    with pytest.raises(ContractError) as excinfo:
        ContainerContract.from_metadata(metadata(reward_route="/reward/{tenant_id}"))
    assert "tenant_id" in str(excinfo.value)


def test_route_resolution_substitutes_and_quotes() -> None:
    contract = ContainerContract.from_metadata(metadata())
    assert contract.resolve("trace_route", rollout_id="ro/1") == "/rollouts/ro%2F1/trace"
    assert contract.resolve("topology_route", topology_id="t-1") == "/topologies/t-1"
    assert contract.resolve("reward_route") == "/reward"


def test_route_resolution_without_its_parameter_is_an_error() -> None:
    contract = ContainerContract.from_metadata(metadata())
    with pytest.raises(RouteError):
        contract.resolve("trace_route")
    with pytest.raises(RouteError):
        contract.resolve("not_a_route")


def test_contract_hash_is_stable_and_route_sensitive() -> None:
    first = ContainerContract.from_metadata(metadata())
    again = ContainerContract.from_metadata(metadata())
    renamed = ContainerContract.from_metadata(metadata(reward_route="/rewards"))
    assert first.contract_hash == again.contract_hash
    assert first.contract_hash.startswith("sha256:")
    assert first.contract_hash != renamed.contract_hash


def test_preflight_can_assert_an_extra_declared_route() -> None:
    with pytest.raises(ContractError):
        preflight_contract(metadata(), expected_routes=("frames_route",))


def test_client_calls_only_declared_routes() -> None:
    sender = RecordingSender([HttpReply(200, b'{"ok": true}') for _ in range(9)])
    connection = client(sender=sender)
    connection.health()
    connection.capabilities()
    connection.handshake({"schema_version": "cispo.handshake.v1"})
    connection.rollout_state("ro-1")
    connection.rollout_events("ro-1", cursor="7")
    connection.renew_rollout("ro-1", {"lease": 1})
    connection.trace("ro-1")
    connection.artifacts("ro-1")
    connection.reward("ro-1")
    assert sender.paths == [
        "http://127.0.0.1:8080/health",
        "http://127.0.0.1:8080/training/capabilities",
        "http://127.0.0.1:8080/training/handshake",
        "http://127.0.0.1:8080/rollouts/ro-1",
        "http://127.0.0.1:8080/rollouts/ro-1/events?cursor=7",
        "http://127.0.0.1:8080/rollouts/ro-1/renew",
        "http://127.0.0.1:8080/rollouts/ro-1/trace",
        "http://127.0.0.1:8080/rollouts/ro-1/artifacts",
        "http://127.0.0.1:8080/reward?rollout_id=ro-1",
    ]
    assert [item.method for item in sender.requests] == [
        "GET",
        "GET",
        "POST",
        "GET",
        "GET",
        "POST",
        "GET",
        "GET",
        "GET",
    ]


def test_client_sends_bearer_from_environment_and_custom_headers() -> None:
    sender = RecordingSender([HttpReply(200, b"{}")])
    connection = client(
        sender=sender,
        headers={"X-Run": "run-1"},
        auth_bearer_env="CONTAINER_TOKEN",
        environ={"CONTAINER_TOKEN": "  secret  "},
    )
    connection.health()
    request = sender.requests[0]
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("X-run") == "run-1"


def test_client_does_not_override_an_explicit_authorization_header() -> None:
    sender = RecordingSender([HttpReply(200, b"{}")])
    connection = client(
        sender=sender,
        headers={"Authorization": "Bearer explicit"},
        auth_bearer_env="CONTAINER_TOKEN",
        environ={"CONTAINER_TOKEN": "secret"},
    )
    connection.health()
    assert sender.requests[0].get_header("Authorization") == "Bearer explicit"


def test_client_refuses_a_bearer_env_that_is_not_set() -> None:
    sender = RecordingSender([HttpReply(200, b"{}")])
    connection = client(sender=sender, auth_bearer_env="CONTAINER_TOKEN", environ={})
    with pytest.raises(ContainerAuthError):
        connection.health()
    assert sender.requests == []


def test_transient_transport_failure_is_retried_then_succeeds() -> None:
    sleeps: list[float] = []
    sender = RecordingSender(
        [
            urllib.error.URLError("connection reset"),
            TimeoutError("timed out"),
            HttpReply(200, b'{"status": "ok"}'),
        ]
    )
    connection = client(sender=sender, sleeps=sleeps)
    assert connection.health() == {"status": "ok"}
    assert len(sender.requests) == 3
    assert sleeps == [0.25, 0.5]


def test_transient_failure_exhausts_attempts_and_raises_transport_error() -> None:
    sleeps: list[float] = []
    sender = RecordingSender([urllib.error.URLError("down") for _ in range(4)])
    connection = client(sender=sender, sleeps=sleeps)
    with pytest.raises(TransportError):
        connection.health()
    assert len(sender.requests) == 4
    assert sleeps == [0.25, 0.5, 1.0]


def test_error_status_is_a_real_reply_and_is_never_retried() -> None:
    sleeps: list[float] = []
    sender = RecordingSender([HttpReply(503, b"pool exhausted")])
    connection = client(sender=sender, sleeps=sleeps)
    with pytest.raises(ContainerStatusError) as excinfo:
        connection.submit_rollout({"idempotency_key": "k"})
    assert excinfo.value.status == 503
    assert len(sender.requests) == 1
    assert sleeps == []


def test_non_object_body_is_a_transport_error() -> None:
    sender = RecordingSender([HttpReply(200, b"[1, 2, 3]")])
    connection = client(sender=sender)
    with pytest.raises(TransportError):
        connection.taskset()


def test_client_refuses_a_non_http_base_url() -> None:
    with pytest.raises(ContractError):
        UrllibContainerClient("file:///tmp", ContainerContract.from_metadata(metadata()))


def test_response_limit_is_explicit_and_enforced():
    contract=ContainerContract.from_metadata(metadata())
    for limit in (0, True, 67_108_865):
        with pytest.raises(ContractError):
            UrllibContainerClient('http://localhost',contract,max_response_bytes=limit)
    sender=RecordingSender([HttpReply(200,b'{"ok":1}')])
    small=UrllibContainerClient('http://localhost',contract,sender=sender,max_response_bytes=7)
    with pytest.raises(TransportError,match='exceeded 7'):
        small.taskset()
    exact=UrllibContainerClient('http://localhost',contract,
        sender=RecordingSender([HttpReply(200,b'{"ok":1}')]),max_response_bytes=8)
    assert exact.taskset()=={'ok':1}


def test_urllib_sender_reads_one_extra_byte_for_overflow_detection(monkeypatch):
    from synth_optimizers.rl.contract import _urllib_send
    sizes=[]
    class Response:
        status=200
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,n): sizes.append(n); return b'{}'
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Response())
    assert _urllib_send(urllib.request.Request('http://localhost'),1,max_response_bytes=12).body==b'{}'
    assert sizes==[13]


def test_the_contract_is_found_by_version_not_by_key_name() -> None:
    """A container may already publish something under ``cispo``.

    The predecessor training block lives at that key on at least one shipped
    image, and a live lane reads it for its own routes. Overwriting it to
    satisfy this executor would point that lane at these routes, so the image
    advertises beside it — and the executor has to find the declaration by
    what it says it is.
    """

    block: dict[str, object] = {"version": CISPO_CONTRACT_VERSION, **DECLARED_ROUTES}
    predecessor = {"version": "training.rollout.v1", "rollout_route": "/training/rollouts"}

    # Advertised under the conventional key.
    canonical = ContainerContract.from_metadata(
        {"metadata": {"optimizer_contracts": {"cispo": block}}}
    )
    assert canonical.version == CISPO_CONTRACT_VERSION

    # Advertised beside a predecessor that already holds the key.
    beside = ContainerContract.from_metadata(
        {"metadata": {"optimizer_contracts": {"cispo": predecessor, "cispo_v1": block}}}
    )
    assert beside.route_table.routes == canonical.route_table.routes

    # The conventional key still wins when it is the real one.
    both = ContainerContract.from_metadata(
        {"metadata": {"optimizer_contracts": {"cispo": block, "cispo_v1": predecessor}}}
    )
    assert both.route_table.routes == canonical.route_table.routes

    # Nothing that declares this contract anywhere is a refusal that says so.
    with pytest.raises(ContractError, match="advertises no block declaring"):
        ContainerContract.from_metadata(
            {"metadata": {"optimizer_contracts": {"cispo": predecessor}}}
        )
