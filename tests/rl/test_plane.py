"""The plane assembly: one configuration in, three wired ports out.

Every test here stands a conformance fake up in process and points a real
``cispo.container.v1`` document at it, so the client, the contract, the
catalog, the gateway, its listener, the binder and the session are all the
production ones. Only two things are doubles: the training provider, which is
never constructed for real because a real one costs money, and -- where a
routing decision is under test -- the address probe, because the answer would
otherwise depend on the machine the suite runs on.

Three lines are held here. Every construction failure is a typed refusal that
names the thing that was missing rather than a traceback from three layers
down. Everything opened is closed, on the failure path as well as the success
path. And the origin the container is handed is an address *the container*
could dial, which is not the same fact as the address the listener bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakes import scenarios
from fakes.container import RunningContainer, serve
from plane_harness import config_text

from synth_optimizers.contracts.rl_records import SamplingProfile
from synth_optimizers.rl import config as config_module
from synth_optimizers.rl import plane as plane_module
from synth_optimizers.rl.cli import main as rl_main
from synth_optimizers.rl.gateway import GatewayError, GatewayServer
from synth_optimizers.rl.plane import (
    SAMPLER_ORIGIN_ENV,
    CatalogPathError,
    ContainerUnreachableError,
    OriginPlan,
    Plane,
    ProviderArtifactProbe,
    ProviderCredentialError,
    RendererUnavailableError,
    SamplerOriginError,
    UnsupportedProviderError,
    build_container_client,
    build_plane,
    build_provider,
    open_plane,
    plan_origin,
)
from synth_optimizers.rl.resolver import ArtifactMissingError

#: A port nothing serves. Refused immediately rather than after a timeout.
DEAD_URL = "http://127.0.0.1:1/"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class StubProvider:
    """The provider surface the plane touches while it is being assembled.

    A real provider is constructed only when a run actually executes, so
    nothing here connects, tokenizes for real, or spends.
    """

    def __init__(self) -> None:
        self.artifacts: dict[str, str] = {}
        self.sampled: list[Any] = []

    def tokenize_chat(
        self, messages: Any, *, add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        return {"prompt_token_ids": (11, 12, 13), "stop_token_ids": (99,)}

    def decode_tokens(self, token_ids: Any) -> str:
        return "".join(str(int(token) % 10) for token in token_ids)

    def sample_checkpoint(self, checkpoint: Any, request: Any) -> Any:
        self.sampled.append(request)
        raise AssertionError("assembly must not sample")


class RecordingServer(GatewayServer):
    """The real listener, remembering every instance the assembly made."""

    made: list["RecordingServer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        RecordingServer.made.append(self)


def is_closed(server: GatewayServer) -> bool:
    """``base_url`` is the listener's public liveness: closed means refused."""

    try:
        server.base_url
    except GatewayError:
        return True
    return False


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def container() -> RunningContainer:
    running = serve(scenarios.multi_turn_environment_reward())
    try:
        yield running
    finally:
        running.shutdown()


def write_config(
    running: RunningContainer,
    tmp_path: Path,
    *,
    url: str | None = None,
    provider: str = "fake",
) -> Any:
    """A full document aimed at this fake, with durable state under ``tmp_path``."""

    text = config_text(running.config, url or running.base_url)
    text = text.replace(
        'catalog = "checkpoints.sqlite3"', f'catalog = "{tmp_path / "checkpoints.sqlite3"}"'
    )
    text = text.replace('directory = "runs"', f'directory = "{tmp_path / "runs"}"')
    text = text.replace('provider = "fake"', f'provider = "{provider}"')
    return config_module.loads(text)


def sampling_for(running: RunningContainer) -> SamplingProfile:
    """The sampling identity this fake stamps its behavior fingerprint from."""

    return SamplingProfile(temperature=1.0, top_p=1.0, seed=running.config.seed)


def assemble(running: RunningContainer, tmp_path: Path, **options: Any) -> Plane:
    options.setdefault("provider", StubProvider())
    options.setdefault("environ", {})
    options.setdefault("sampling", sampling_for(running))
    config = options.pop("config", None) or write_config(running, tmp_path)
    return build_plane(config, **options)


# --------------------------------------------------------------------------- #
# Construction from a full configuration
# --------------------------------------------------------------------------- #


def test_the_plane_assembles_every_port_from_one_configuration(
    container: RunningContainer, tmp_path: Path
) -> None:
    with assemble(container, tmp_path) as plane:
        assert plane.session.handshake_id
        assert plane.session.agreement_digest
        assert plane.gateway.renderer_profile == container.config.renderer_profile
        assert plane.binder.catalog is plane.catalog
        assert plane.clock is not None
        assert Path(plane.catalog.path) == tmp_path / "checkpoints.sqlite3"
        assert plane.artifact_directory == tmp_path / "runs"
        assert plane.client.contract.contract_hash.startswith("sha256:")


def test_the_assembled_plane_satisfies_the_shape_the_commands_read(
    container: RunningContainer, tmp_path: Path
) -> None:
    with assemble(container, tmp_path) as plane:
        for port in ("session", "gateway", "binder", "clock"):
            assert getattr(plane, port, None) is not None


def test_the_receipt_payload_names_what_was_assembled(
    container: RunningContainer, tmp_path: Path
) -> None:
    with assemble(container, tmp_path) as plane:
        payload = plane.to_payload()
    assert payload["schema_version"] == "cispo.plane.v1"
    assert payload["handshake_id"]
    assert payload["origin_base_url"].startswith("http://")
    assert payload["container_contract_hash"].startswith("sha256:")


def test_the_client_is_built_from_the_connection_and_the_declared_contract(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path)

    client = build_container_client(config, environ={})

    assert client.contract.contract_hash.startswith("sha256:")
    assert client.health()


def test_resolution_verifies_against_the_provider_not_the_catalog() -> None:
    """The catalog's own copy of a digest cannot verify the catalog."""

    provider = StubProvider()
    provider.artifacts["weights://a"] = "sha256:" + "ab" * 32
    probe = ProviderArtifactProbe(provider)

    assert probe.exists("weights://a")
    assert probe.digest_of("weights://a").startswith("sha256:")
    with pytest.raises(ArtifactMissingError, match="weights://b"):
        probe.digest_of("weights://b")


def test_a_provider_that_reports_no_digests_refuses_rather_than_waving_through() -> None:
    probe = ProviderArtifactProbe(StubProvider())

    with pytest.raises(ArtifactMissingError) as raised:
        probe.exists("weights://a")

    assert "weights://a" in str(raised.value)


# --------------------------------------------------------------------------- #
# Typed construction failures
# --------------------------------------------------------------------------- #


def test_a_missing_credential_names_the_variable_it_is_read_from(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path, provider="tinker")

    with pytest.raises(ProviderCredentialError) as raised:
        build_plane(config, environ={}, sampling=sampling_for(container))

    assert "TINKER_API_KEY" in str(raised.value)


def test_a_credential_is_read_from_the_environment_and_never_from_the_config(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path, provider="tinker")

    provider = build_provider(config, environ={"TINKER_API_KEY": "  key-from-env  "})

    assert provider.credentials.api_key == "key-from-env"
    assert "key-from-env" not in str(config.redacted_payload())


def test_a_provider_with_no_assembly_here_is_refused_by_name(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path, provider="fake")

    with pytest.raises(UnsupportedProviderError) as raised:
        build_plane(config, environ={}, sampling=sampling_for(container))

    assert "fake" in str(raised.value)
    assert "--plane" in str(raised.value)


def test_an_unreachable_container_is_named_rather_than_traced(tmp_path: Path) -> None:
    running = serve(scenarios.multi_turn_environment_reward())
    try:
        config = write_config(running, tmp_path, url=DEAD_URL)
    finally:
        running.shutdown()

    with pytest.raises(ContainerUnreachableError) as raised:
        build_plane(config, provider=StubProvider(), environ={})

    message = str(raised.value)
    assert DEAD_URL.rstrip("/") in message
    assert "unreachable" in message


def test_an_unwritable_catalog_path_is_refused_before_a_provider_exists(
    container: RunningContainer, tmp_path: Path
) -> None:
    closed = tmp_path / "closed"
    closed.mkdir()
    config = write_config(container, closed)
    closed.chmod(0o500)
    provider = StubProvider()
    try:
        with pytest.raises(CatalogPathError) as raised:
            build_plane(config, provider=provider, environ={})
    finally:
        closed.chmod(0o700)

    assert str(closed) in str(raised.value)
    assert "not writable" in str(raised.value)
    assert provider.sampled == []


def test_a_provider_with_no_renderer_surface_is_refused(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path)

    with pytest.raises(RendererUnavailableError) as raised:
        build_plane(config, provider=object(), environ={}, sampling=sampling_for(container))

    assert "tokenize_chat" in str(raised.value)


# --------------------------------------------------------------------------- #
# The origin the container is handed
# --------------------------------------------------------------------------- #


def test_a_loopback_container_is_told_a_loopback_origin() -> None:
    plan = plan_origin("http://127.0.0.1:8080", environ={})

    assert plan.bind_host == "127.0.0.1"
    assert plan.advertised_host == "127.0.0.1"


def test_a_remote_container_is_told_the_address_the_route_leaves_this_host_at() -> None:
    plan = plan_origin(
        "http://198.51.100.7:8080", environ={}, address_of=lambda _host, _port: "203.0.113.4"
    )

    assert plan.advertised_host == "203.0.113.4"
    assert plan.bind_host == "0.0.0.0"
    assert plan.base_url(4100) == "http://203.0.113.4:4100"


def test_a_remote_container_with_no_route_back_is_refused_rather_than_given_loopback() -> None:
    with pytest.raises(SamplerOriginError) as raised:
        plan_origin(
            "http://198.51.100.7:8080", environ={}, address_of=lambda _host, _port: "127.0.0.1"
        )

    assert SAMPLER_ORIGIN_ENV in str(raised.value)


def test_an_undeterminable_route_is_refused_rather_than_guessed() -> None:
    with pytest.raises(SamplerOriginError):
        plan_origin("http://198.51.100.7:8080", environ={}, address_of=lambda _h, _p: "")


def test_the_origin_override_names_the_address_the_container_dials() -> None:
    plan = plan_origin(
        "http://198.51.100.7:8080", environ={SAMPLER_ORIGIN_ENV: "gateway.internal:4300"}
    )

    assert plan.base_url(1) == "http://gateway.internal:4300"
    assert plan.fixed_port == 4300


def test_a_malformed_origin_override_is_refused() -> None:
    with pytest.raises(SamplerOriginError):
        plan_origin("http://127.0.0.1:8080", environ={SAMPLER_ORIGIN_ENV: "ftp:///nowhere"})


def test_the_gateway_advertises_the_reachable_origin_not_the_bound_loopback(
    container: RunningContainer, tmp_path: Path
) -> None:
    """The listener binds locally; the container is told where *it* can dial."""

    origin = OriginPlan(bind_host="127.0.0.1", advertised_host="203.0.113.9")

    with assemble(container, tmp_path, origin=origin) as plane:
        port = plane.server.base_url.rsplit(":", 1)[-1]
        assert plane.origin_base_url == f"http://203.0.113.9:{port}"
        assert plane.gateway.origin_root == plane.origin_base_url
        assert plane.gateway.origin_root != plane.server.base_url
        assert "127.0.0.1" not in plane.gateway.origin_root


# --------------------------------------------------------------------------- #
# Everything opened is closed
# --------------------------------------------------------------------------- #


def test_the_context_manager_closes_the_gateway_server_on_success(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path)
    with open_plane(
        config,
        provider=StubProvider(),
        environ={},
        sampling=sampling_for(container),
    ) as plane:
        assert not is_closed(plane.server)
        server = plane.server

    assert is_closed(server)


def test_the_context_manager_closes_the_gateway_server_when_the_body_fails(
    container: RunningContainer, tmp_path: Path
) -> None:
    config = write_config(container, tmp_path)
    server: GatewayServer | None = None

    with pytest.raises(RuntimeError, match="the body failed"):
        with open_plane(
            config,
            provider=StubProvider(),
            environ={},
            sampling=sampling_for(container),
        ) as plane:
            server = plane.server
            raise RuntimeError("the body failed")

    assert server is not None
    assert is_closed(server)


def test_a_failed_later_step_closes_the_listener_an_earlier_step_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup refuses after the listener exists; nothing is left listening."""

    running = serve(scenarios.rejected_mandatory_clause())
    monkeypatch.setattr(plane_module, "GatewayServer", RecordingServer)
    RecordingServer.made.clear()
    try:
        config = write_config(running, tmp_path)
        with pytest.raises(Exception) as raised:
            build_plane(
                config,
                provider=StubProvider(),
                environ={},
                sampling=sampling_for(running),
            )
    finally:
        running.shutdown()

    assert not isinstance(raised.value, AssertionError)
    assert RecordingServer.made, "the assembly never reached the listener"
    assert is_closed(RecordingServer.made[-1])


# --------------------------------------------------------------------------- #
# The command surface reaches the default assembly
# --------------------------------------------------------------------------- #


class PlaneBuilt(RuntimeError):
    """Raised by the recorder instead of assembling anything live."""


def test_run_without_a_plane_argument_builds_the_default_plane(
    container: RunningContainer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def recorder(config: Any, **_options: Any) -> Plane:
        seen.append(config)
        raise PlaneBuilt("the default assembly was reached")

    monkeypatch.setattr(plane_module, "build_plane", recorder)
    path = tmp_path / "run.toml"
    path.write_text(_config_text(container, tmp_path), encoding="utf-8")

    with pytest.raises(PlaneBuilt):
        rl_main(
            ["run", "--config", str(path), "--receipts", str(tmp_path / "receipts")]
        )

    assert [item.run_id for item in seen] == ["run_test"]


def test_run_validate_only_needs_no_plane_argument(
    container: RunningContainer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "run.toml"
    path.write_text(_config_text(container, tmp_path), encoding="utf-8")

    assert rl_main(["run", "--config", str(path), "--validate-only"]) == 0

    output = capsys.readouterr().out
    assert "plan=" in output
    assert "nothing was started" in output
    assert "--plane" not in output


def test_run_reports_a_refusal_from_the_default_assembly_legibly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    running = serve(scenarios.multi_turn_environment_reward())
    try:
        text = _config_text(running, tmp_path).replace(running.base_url, DEAD_URL.rstrip("/"))
    finally:
        running.shutdown()
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        rl_main(["run", "--config", str(path), "--receipts", str(tmp_path / "receipts")])

    message = str(raised.value)
    assert "unreachable" in message
    assert "--plane" in message
    assert "Traceback" not in message
    capsys.readouterr()


def _config_text(running: RunningContainer, tmp_path: Path) -> str:
    text = config_text(running.config, running.base_url)
    text = text.replace(
        'catalog = "checkpoints.sqlite3"', f'catalog = "{tmp_path / "checkpoints.sqlite3"}"'
    )
    return text.replace('directory = "runs"', f'directory = "{tmp_path / "runs"}"')


def test_the_renderer_check_touches_tokens_not_declarations() -> None:
    """Startup's profile equality cannot fail, so it proves nothing.

    The bound profile is the container's declared profile, so asserting they
    match is a tautology. Rendering the same canary on both sides and comparing
    the digest is the check that can actually catch a disagreement.
    """

    from synth_optimizers.contracts.rl_records import (
        CANARY_MESSAGES,
        RendererProfile,
        canary_digest,
    )
    from synth_optimizers.rl.plane import (
        RendererDisagreementError,
        build_renderer,
        verify_renderer_agreement,
    )

    class StubProvider:
        def __init__(self, tokens: tuple[int, ...]) -> None:
            self.tokens = tokens

        def tokenize_chat(self, rows, *, add_generation_prompt: bool = True):
            assert len(rows) == len(CANARY_MESSAGES)
            return {"prompt_token_ids": list(self.tokens)}

        def decode_tokens(self, token_ids):
            return "".join(str(token) for token in token_ids)

    agreed = canary_digest((11, 12, 13))
    profile_ok = RendererProfile(
        profile_id="renderers.stub.v1",
        package="renderers",
        package_version="0.1.11",
        config_digest="sha256:cfg",
        tokenizer_id="vendor/policy-20b",
        tokenizer_digest="sha256:tok",
        stop_token_ids=(2,),
        canary_digest=agreed,
    )
    renderer = build_renderer(StubProvider((11, 12, 13)), profile_ok, wire_api="chat_completions")
    assert verify_renderer_agreement(renderer, profile_ok).proven

    # Identical declared profile, different tokens: caught only by the canary.
    with pytest.raises(RendererDisagreementError, match="tokenize differently"):
        build_renderer(StubProvider((11, 12, 99)), profile_ok, wire_api="chat_completions")

    # A container that declares no canary runs, and the run says so.
    bare = RendererProfile(
        profile_id="renderers.stub.v1",
        package="renderers",
        package_version="0.1.11",
        config_digest="sha256:cfg",
        tokenizer_id="vendor/policy-20b",
        tokenizer_digest="sha256:tok",
        stop_token_ids=(2,),
    )
    agreement = verify_renderer_agreement(
        build_renderer(StubProvider((1,)), bare, wire_api="chat_completions"), bare
    )
    assert not agreement.proven
    assert agreement.to_payload()["agreement_proven"] is False
