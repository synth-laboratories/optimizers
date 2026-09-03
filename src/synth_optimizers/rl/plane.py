"""The real plane, assembled from one validated configuration.

Every part of the container-first plane exists on its own -- a declared-route
client, a checkpoint catalog, a training provider, a sampler gateway, a policy
binder, an admitted session -- and until now nothing put them together. This
module is that assembly, and it is the default the ``rl`` commands reach when
no ``--plane MODULE:FACTORY`` names one.

The construction order is the one the startup sequence requires, because each
step is the input to the next and because a later step must never be able to
spend money that an earlier refusal should have prevented:

1. **The container client**, from ``[container]``: the base URL, the declared
   headers, and the bearer token named by ``auth_bearer_env`` and read from the
   environment. Building it fetches ``/metadata`` once, so an unreachable
   container is refused here rather than three layers down.
2. **The catalog and its stores**, from ``[artifacts]``: the checkpoint catalog
   at the configured path, the atomic policy-set publisher over it, and the
   shared resolver. A path that cannot be written is refused before a provider
   session could ever have been created.
3. **The training provider**, from ``[model]``. The credential is read from the
   environment, never from the configuration file: ``[model]`` has no field for
   a secret and this module never invents one.
4. **The renderer and the sampler gateway**, with its loopback listener stood
   up and its origin root pointed at an address *the container* can dial. The
   renderer profile is the one the container declares; nothing here chooses a
   renderer, because a second renderer entering the run would put two parties
   in disagreement about what a token means.
5. **The binder**, over the provider and the catalog.
6. **The session**, over the client and the gateway's declared profile.

Every construction failure is one of the typed errors below, each naming the
thing that was missing: a credential, a reachable container, a writable catalog
path, an origin the container can resolve. None of them is a traceback.

Nothing here names a task, a harness, an environment, or an algorithm.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts.rl_records import CANARY_MESSAGES, RecordError, RendererProfile, SamplingProfile
from .binder import CatalogPolicyBinder
from .capabilities import CapabilityDocument
from .catalog import CheckpointCatalog
from .config import ConfigError, RunConfig
from .contract import (
    MAX_RESPONSE_BYTES,
    ContainerClient,
    ContainerContract,
    ContractError,
    UrllibContainerClient,
)
from .gateway import (
    UNBOUNDED_BUDGET,
    WIRE_RESPONSES,
    GatewayServer,
    RenderedPrompt,
    SamplerGatewayService,
    project_responses_items,
)
from .policy_sets import PolicySetPublisher
from .ports import PortError
from .resolver import ArtifactMissingError, EvaluationResolver
from .session import ContractContainerSession, RunClock, start_session

PLANE_SCHEMA_VERSION = "cispo.plane.v1"

#: The route every container serves before its contract is known. It is the one
#: path that cannot be declared, because the declaration is what it carries.
METADATA_ROUTE = "/metadata"

#: Names that mean "this process", and therefore mean nothing to a container
#: that is not sharing this network namespace.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"", "0.0.0.0", "127.0.0.1", "::1", "localhost"})

#: Overrides the address the container is told to dial the gateway at. Set it
#: when the optimizer's route back from the container is not the address this
#: process would pick on its own -- a published port, a gateway name, a proxy.
SAMPLER_ORIGIN_ENV = "SYNTH_OPTIMIZERS_SAMPLER_ORIGIN"

#: ``[model] provider`` -> the environment variable its credential is read
#: from. A provider absent from this table cannot be constructed here.
PROVIDER_CREDENTIAL_ENV: Mapping[str, str] = {"tinker": "TINKER_API_KEY"}

#: The optional per-provider endpoint override, read from the same environment.
PROVIDER_BASE_URL_ENV: Mapping[str, str] = {"tinker": "TINKER_BASE_URL"}


# --------------------------------------------------------------------------- #
# Typed refusals
# --------------------------------------------------------------------------- #


class PlaneError(PortError):
    """A plane could not be assembled. Always names what was missing."""


class ContainerUnreachableError(PlaneError):
    """The container did not answer, or answered with something unusable."""


class ContainerAuthError(PlaneError):
    """The container's bearer configuration names an unset environment variable."""


class CatalogPathError(PlaneError):
    """The configured catalog path cannot be created, opened, or written."""


class ProviderCredentialError(PlaneError):
    """The provider's credential environment variable is unset or empty."""


class UnsupportedProviderError(PlaneError):
    """``[model] provider`` names a provider this assembly cannot construct."""


class RendererUnavailableError(PlaneError):
    """The provider exposes no renderer surface, so no party could render."""


class RendererDisagreementError(PlaneError):
    """A local renderer and the container tokenize the same canary differently."""


class SamplerOriginError(PlaneError):
    """No origin base URL could be chosen that the container could reach."""


# --------------------------------------------------------------------------- #
# The container client
# --------------------------------------------------------------------------- #


def _connection_headers(config: RunConfig, environ: Mapping[str, str]) -> Mapping[str, str]:
    """Declared headers plus the bearer named by ``auth_bearer_env``."""

    try:
        return config.container.resolved_headers(environ)
    except ConfigError as error:
        raise ContainerAuthError(str(error)) from error


def fetch_metadata(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """One unretried ``GET /metadata``. A container that is not there says so now.

    The declared-route client needs a contract to be built, and the contract is
    what ``/metadata`` carries, so this one call cannot go through the client.
    It is deliberately a single attempt: startup is not the place to spend
    seconds of backoff discovering that nothing is listening.
    """

    target = url.rstrip("/") + METADATA_ROUTE
    request = urllib.request.Request(
        target, method="GET", headers={"Accept": "application/json", **dict(headers)}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            status = int(response.status)
            body = response.read(MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as error:
        raise ContainerUnreachableError(
            f"container at {target} answered status {error.code}; it advertises no "
            "contract this run can be assembled against"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ContainerUnreachableError(
            f"container at {target} is unreachable: {error}. Start the container, or "
            "point [container] url at one that is running"
        ) from error
    if status < 200 or status >= 300:
        raise ContainerUnreachableError(
            f"container at {target} answered status {status}, not its advertisement"
        )
    try:
        decoded = json.loads(body.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as error:
        raise ContainerUnreachableError(
            f"container at {target} returned a body that is not JSON: {error}"
        ) from error
    if not isinstance(decoded, Mapping):
        raise ContainerUnreachableError(f"container at {target} returned a non-object body")
    return decoded


def build_container_client(
    config: RunConfig, *, environ: Mapping[str, str] | None = None
) -> UrllibContainerClient:
    """The declared-route client, from ``[container]`` and the environment."""

    source = os.environ if environ is None else environ
    headers = _connection_headers(config, source)
    connection = config.container
    metadata = fetch_metadata(
        connection.url, headers=headers, timeout_seconds=connection.timeout_seconds
    )
    try:
        contract = ContainerContract.from_metadata(metadata)
    except ContractError as error:
        raise ContainerUnreachableError(
            f"container at {connection.url} does not advertise a usable contract: {error}"
        ) from error
    try:
        return UrllibContainerClient(
            connection.url,
            contract,
            headers=connection.headers,
            auth_bearer_env=connection.auth_bearer_env,
            timeout_seconds=connection.timeout_seconds,
            environ=source,
        )
    except ContractError as error:
        raise ContainerUnreachableError(
            f"container at {connection.url} cannot be addressed: {error}"
        ) from error


# --------------------------------------------------------------------------- #
# The catalog and its stores
# --------------------------------------------------------------------------- #


def open_catalog(config: RunConfig) -> CheckpointCatalog:
    """The checkpoint catalog at ``[artifacts] catalog``, or a named refusal."""

    path = Path(config.artifacts.catalog).expanduser()
    directory = path.parent if str(path.parent) else Path(".")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CatalogPathError(
            f"catalog directory {directory} cannot be created: {error}"
        ) from error
    if not os.access(directory, os.W_OK):
        raise CatalogPathError(
            f"catalog directory {directory} is not writable; a run whose checkpoints "
            "cannot be catalogued has no way to prove a revision exists"
        )
    if path.exists() and not os.access(path, os.W_OK):
        raise CatalogPathError(f"catalog file {path} is not writable")
    try:
        return CheckpointCatalog(path)
    except (sqlite3.Error, OSError) as error:
        raise CatalogPathError(f"catalog {path} cannot be opened: {error}") from error


def prepare_artifact_directory(config: RunConfig) -> Path:
    """``[artifacts] directory``, created up front rather than at first write."""

    directory = Path(config.artifacts.directory).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CatalogPathError(
            f"artifact directory {directory} cannot be created: {error}"
        ) from error
    if not os.access(directory, os.W_OK):
        raise CatalogPathError(f"artifact directory {directory} is not writable")
    return directory


@dataclass(frozen=True, slots=True)
class ProviderArtifactProbe:
    """Artifact existence and digest as the *provider* reported them.

    Resolution is verification, and the catalog's own copy of a digest cannot
    verify itself. Every reference a run resolves was minted by this provider's
    ``save_checkpoint``, so the provider's own record of what it returned is
    the only local source that is not circular. A provider that keeps no such
    record refuses every reference by name rather than waving one through.
    """

    provider: Any

    def _observed(self) -> Mapping[str, str]:
        artifacts = getattr(self.provider, "artifacts", None)
        return artifacts if isinstance(artifacts, Mapping) else {}

    def exists(self, ref: str) -> bool:
        observed = self._observed()
        if not observed:
            raise ArtifactMissingError(self._refusal(ref))
        return ref in observed

    def digest_of(self, ref: str) -> str:
        observed = self._observed()
        if not observed:
            raise ArtifactMissingError(self._refusal(ref))
        try:
            return observed[ref]
        except KeyError as error:
            raise ArtifactMissingError(
                f"artifact {ref} was never observed at this provider"
            ) from error

    @staticmethod
    def _refusal(ref: str) -> str:
        return (
            f"cannot verify artifact {ref}: this provider reports no artifact digests, "
            "so nothing here can confirm the reference exists. Resolve with an explicit "
            "digest source instead of trusting the catalog's own copy"
        )


# --------------------------------------------------------------------------- #
# The training provider
# --------------------------------------------------------------------------- #


def build_provider(config: RunConfig, *, environ: Mapping[str, str] | None = None) -> Any:
    """The training provider named by ``[model] provider``.

    The credential is read from the environment. ``[model]`` carries no secret
    field and this function never reads one from the configuration document: a
    key in a config file is a key in a receipt, a diff, and a bug report.
    """

    source = os.environ if environ is None else environ
    name = str(config.model.provider or "").strip().lower()
    variable = PROVIDER_CREDENTIAL_ENV.get(name)
    if variable is None:
        raise UnsupportedProviderError(
            f"[model] provider={config.model.provider!r} has no assembly here; "
            f"this plane constructs {sorted(PROVIDER_CREDENTIAL_ENV)}. Pass "
            "--plane MODULE:FACTORY to name an assembly of your own"
        )
    credential = str(source.get(variable, "") or "").strip()
    if not credential:
        raise ProviderCredentialError(
            f"[model] provider={name!r} needs its credential in ${variable}, and that "
            "variable is unset or empty; a credential is never read from the config file"
        )
    base_url = str(source.get(PROVIDER_BASE_URL_ENV.get(name, ""), "") or "").strip() or None
    from ..providers.tinker.client import TinkerAdapter, TinkerCredentials

    return TinkerAdapter(TinkerCredentials(api_key=credential, base_url=base_url))


# --------------------------------------------------------------------------- #
# The renderer and the origin the container dials
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProviderRenderer:
    """The one renderer in the run, over the provider's own tokenizer.

    The profile is not chosen here: it is the profile the container declared,
    and the session asserts equality against it during startup. A provider
    binds its tokenizer lazily -- there is none until a training client exists
    -- so this renderer speaks to the provider's tokenizer surface rather than
    holding a tokenizer object that could not have been built yet.
    """

    provider: Any
    profile: RendererProfile
    wire_api: str

    @property
    def wire_apis(self) -> tuple[str, ...]:
        return (self.wire_api,)

    def render(self, rows: Sequence[Mapping[str, Any]]) -> RenderedPrompt:
        source = (
            project_responses_items(rows)
            if self.wire_api == WIRE_RESPONSES
            else [dict(row) for row in rows]
        )
        rendered = self.provider.tokenize_chat(source, add_generation_prompt=True)
        token_ids = tuple(int(token) for token in rendered["prompt_token_ids"])
        if not token_ids:
            raise RendererUnavailableError("the provider's renderer produced no prompt tokens")
        return RenderedPrompt(
            token_ids=token_ids,
            stop_token_ids=tuple(int(token) for token in rendered.get("stop_token_ids") or ()),
        )

    @property
    def bridges(self) -> bool:
        return callable(getattr(self.provider, "bridge_chat", None))

    def bridge(
        self,
        previous_prompt_token_ids: Sequence[int],
        previous_generation_token_ids: Sequence[int],
        new_rows: Sequence[Mapping[str, Any]],
    ) -> RenderedPrompt | None:
        """Extend the previous turn's ids by the turns this call added.

        The new turns pass through the same declared projection the full render
        uses, so a bridged Responses prompt is the Responses prompt. ``None``
        means the renderer would not vouch for the extension -- a thinking
        retention policy that drops history at a user boundary is the usual
        reason -- and the gateway forks a branch instead of pretending.
        """

        bridge = getattr(self.provider, "bridge_chat", None)
        if not callable(bridge) or not new_rows:
            return None
        source = (
            project_responses_items(new_rows)
            if self.wire_api == WIRE_RESPONSES
            else [dict(row) for row in new_rows]
        )
        bridged = bridge(
            list(previous_prompt_token_ids), list(previous_generation_token_ids), source
        )
        if not bridged:
            return None
        token_ids = tuple(int(token) for token in bridged.get("prompt_token_ids") or ())
        if not token_ids:
            return None
        return RenderedPrompt(
            token_ids=token_ids,
            stop_token_ids=tuple(int(token) for token in bridged.get("stop_token_ids") or ()),
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.provider.decode_tokens(list(token_ids)))


@dataclass(frozen=True, slots=True)
class RendererAgreement:
    """Whether the local renderer was proven to agree, and on what."""

    profile_id: str
    proven: bool
    digest: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "agreement_proven": self.proven,
            "canary_digest": self.digest,
        }


def build_renderer(provider: Any, profile: RendererProfile, *, wire_api: str) -> ProviderRenderer:
    """Refuse a provider with no renderer surface before anything is bound."""

    missing = [
        name for name in ("tokenize_chat", "decode_tokens") if not callable(
            getattr(provider, name, None)
        )
    ]
    if missing:
        raise RendererUnavailableError(
            f"the training provider exposes no {missing}; exactly one party renders in "
            "this run and it has to be the one that samples"
        )
    renderer = ProviderRenderer(provider=provider, profile=profile, wire_api=wire_api)
    verify_renderer_agreement(renderer, profile)
    return renderer


def verify_renderer_agreement(
    renderer: ProviderRenderer, profile: RendererProfile
) -> RendererAgreement:
    """Make the renderer check touch tokens rather than declarations.

    Startup asserts the bound profile equals the container's declared profile,
    but the bound profile *is* the declared one, so that assertion cannot fail
    and proves nothing. The only check that touches what goes into training is
    rendering the same canary on both sides and comparing the digest.
    """

    if not profile.agreement_proven:
        return RendererAgreement(profile_id=profile.profile_id, proven=False)
    try:
        rendered = renderer.render(list(CANARY_MESSAGES))
    except Exception as exc:  # noqa: BLE001 - any failure here is a refusal
        raise RendererUnavailableError(
            f"the provider's renderer could not render the agreement canary: {exc}"
        ) from exc
    try:
        profile.assert_renders_like(rendered.token_ids)
    except RecordError as exc:
        raise RendererDisagreementError(str(exc)) from exc
    return RendererAgreement(
        profile_id=profile.profile_id,
        proven=True,
        digest=profile.canary_digest,
    )


@dataclass(frozen=True, slots=True)
class OriginPlan:
    """Where the gateway listens, and the address the container is told to dial.

    These are two different facts. A gateway bound to loopback is invisible to
    a container that is not in this network namespace, and a container told to
    dial ``127.0.0.1`` dials itself.
    """

    bind_host: str
    advertised_host: str
    fixed_port: int | None = None
    reason: str = ""

    def base_url(self, port: int) -> str:
        return f"http://{self.advertised_host}:{self.fixed_port or port}"


def _is_loopback(host: str) -> bool:
    return str(host or "").strip().lower() in LOOPBACK_HOSTS


def local_address_toward(host: str, port: int) -> str:
    """The local address this host would use to reach ``host``.

    A datagram socket that is *connected* sends nothing; it only makes the
    kernel pick a route and bind a source address. That source address is
    exactly what a container at ``host`` would see, which is what has to go in
    the origin.
    """

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.5)
            probe.connect((host, port or 80))
            return str(probe.getsockname()[0])
    except OSError:
        return ""


def plan_origin(
    container_url: str,
    *,
    environ: Mapping[str, str] | None = None,
    address_of: Callable[[str, int], str] = local_address_toward,
) -> OriginPlan:
    """Choose an origin the container can actually reach, or refuse by name."""

    source = os.environ if environ is None else environ
    override = str(source.get(SAMPLER_ORIGIN_ENV, "") or "").strip()
    if override:
        return _override_origin(override)
    parsed = urllib.parse.urlparse(container_url)
    host = parsed.hostname or ""
    if _is_loopback(host):
        # The container answers on this host's loopback, so this host's
        # loopback is precisely the address it can dial back on.
        return OriginPlan(
            bind_host="127.0.0.1",
            advertised_host="127.0.0.1",
            reason="the container is addressed on this host's loopback",
        )
    routable = address_of(host, int(parsed.port or 0))
    if not routable or _is_loopback(routable):
        raise SamplerOriginError(
            f"the container at {container_url} is not on this host, and no address this "
            f"host is reachable at from there could be determined; set ${SAMPLER_ORIGIN_ENV} "
            "to the base URL the container reaches this process at"
        )
    return OriginPlan(
        bind_host="0.0.0.0",  # noqa: S104 - a remote container has to be able to connect
        advertised_host=routable,
        reason=f"the route toward {host} leaves this host at {routable}",
    )


def _override_origin(override: str) -> OriginPlan:
    """``http://host:port``, ``host:port`` or ``host`` -- all mean one address."""

    candidate = override if "://" in override else f"http://{override}"
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SamplerOriginError(
            f"${SAMPLER_ORIGIN_ENV}={override!r} is not an http(s) base URL or a host[:port]"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise SamplerOriginError(
            f"${SAMPLER_ORIGIN_ENV}={override!r} carries a port that is not a number"
        ) from error
    bind = "127.0.0.1" if _is_loopback(parsed.hostname) else "0.0.0.0"  # noqa: S104
    return OriginPlan(
        bind_host=bind,
        advertised_host=parsed.hostname,
        fixed_port=port,
        reason=f"${SAMPLER_ORIGIN_ENV} names this address",
    )


# --------------------------------------------------------------------------- #
# The assembled plane
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Plane:
    """One live plane: the three ports, the clock, and what has to be closed.

    ``session``, ``gateway``, ``binder`` and ``clock`` are the shape the ``rl``
    commands read. Everything else is here so a caller can close what was
    opened and so a receipt can name what was assembled.
    """

    session: ContractContainerSession
    gateway: SamplerGatewayService
    binder: CatalogPolicyBinder
    clock: RunClock
    client: ContainerClient
    catalog: CheckpointCatalog
    provider: Any
    server: GatewayServer
    origin_base_url: str
    artifact_directory: Path

    def to_payload(self) -> dict[str, Any]:
        """What was assembled, safe to write into a receipt."""

        return {
            "schema_version": PLANE_SCHEMA_VERSION,
            "origin_base_url": self.origin_base_url,
            "catalog": self.catalog.path,
            "artifact_directory": str(self.artifact_directory),
            "container_contract_hash": self.client.contract.contract_hash,
            "renderer_profile_fingerprint": self.gateway.renderer_profile.fingerprint,
            "handshake_id": self.session.handshake_id,
            "agreement_digest": self.session.agreement_digest,
        }

    def close(self) -> None:
        """Release everything, in the reverse of the order it was acquired."""

        self.server.close()
        self.catalog.close()

    def __enter__(self) -> "Plane":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_plane(
    config: RunConfig,
    *,
    clock: RunClock | None = None,
    environ: Mapping[str, str] | None = None,
    client: ContainerClient | None = None,
    provider: Any | None = None,
    origin: OriginPlan | None = None,
    sampling: SamplingProfile | None = None,
) -> Plane:
    """Assemble the live plane this configuration describes.

    The keyword seams exist so a test can stand the assembly up against an
    in-process container and a stubbed provider; left alone, every one of them
    is built from the configuration and the environment. A failure at any step
    closes whatever the earlier steps opened before it propagates.
    """

    run_clock = clock or RunClock()
    sampling_profile = sampling or SamplingProfile()
    with ExitStack() as stack:
        # 1. The container client, and with it the declared contract.
        container = client if client is not None else build_container_client(
            config, environ=environ
        )

        # 2. The catalog and the stores over it.
        catalog = open_catalog(config)
        stack.callback(catalog.close)
        artifact_directory = prepare_artifact_directory(config)
        publisher = PolicySetPublisher(catalog)

        # 3. The training provider, credential from the environment.
        training_provider = (
            provider if provider is not None else build_provider(config, environ=environ)
        )
        resolver = EvaluationResolver(catalog, probe=ProviderArtifactProbe(training_provider))

        # 4. The renderer the container declared, the gateway, its listener,
        #    and the origin root the container will be handed.
        document = _capability_document(container)
        renderer = build_renderer(
            training_provider, document.renderer_profile, wire_api=config.model.wire_api
        )
        gateway = SamplerGatewayService(
            renderer,
            training_provider,
            prompt_budget=UNBOUNDED_BUDGET,
            credential_salt=secrets.token_hex(16),
            now=run_clock.utc,
        )
        origin_plan = origin or plan_origin(config.container.url, environ=environ)
        server = GatewayServer(
            gateway, host=origin_plan.bind_host, port=origin_plan.fixed_port or 0
        )
        server.start()
        stack.callback(server.close)
        origin_base_url = origin_plan.base_url(_port_of(server))
        # ``start`` points the root at the listener's own address; the container
        # is told the address it can reach, which is not always the same one.
        gateway.set_origin_root(origin_base_url)

        # 5. The binder, over the provider and the catalog.
        binder = _build_binder(
            config,
            provider=training_provider,
            publisher=publisher,
            resolver=resolver,
            document=document,
            contract_hash=container.contract.contract_hash,
            sampling=sampling_profile,
        )

        # 6. The session: health, capabilities, tasks, handshake, probe.
        session = start_session(
            container,
            config,
            renderer_profile=gateway.renderer_profile,
            clock=run_clock,
            sampling=sampling_profile,
        )
        stack.pop_all()
    return Plane(
        session=session,
        gateway=gateway,
        binder=binder,
        clock=run_clock,
        client=container,
        catalog=catalog,
        provider=training_provider,
        server=server,
        origin_base_url=origin_base_url,
        artifact_directory=artifact_directory,
    )


@contextmanager
def open_plane(config: RunConfig, **options: Any) -> Iterator[Plane]:
    """``build_plane`` as a context manager: closed on exit, failure included."""

    plane = build_plane(config, **options)
    try:
        yield plane
    finally:
        plane.close()


# --------------------------------------------------------------------------- #
# Private construction helpers
# --------------------------------------------------------------------------- #


def _capability_document(client: ContainerClient) -> CapabilityDocument:
    """The container's own declaration: its renderer profile and its roster."""

    try:
        payload = client.capabilities()
    except Exception as error:  # noqa: BLE001 - every transport failure is one refusal
        raise ContainerUnreachableError(
            f"container did not serve its capability document: {error}"
        ) from error
    inner = payload.get("capabilities")
    document = inner if isinstance(inner, Mapping) else payload
    return CapabilityDocument.from_payload(document)


def _port_of(server: GatewayServer) -> int:
    """The port the listener actually took, whether or not one was asked for."""

    return int(urllib.parse.urlparse(server.base_url).port or 0)


def _policy_types(
    config: RunConfig, document: CapabilityDocument
) -> dict[str, tuple[str, ...]]:
    """``parameter group -> policy types``, from the container's own roster.

    The container declares ``policy_type -> parameter_group``; the binder needs
    the inverse. ``[topology] policy_types`` is read in the container's
    direction and layered on top, so an operator adding a mapping writes it the
    same way the container does.
    """

    declared = dict(document.topology.parameter_groups)
    declared.update(dict(config.topology.policy_types))
    inverted: dict[str, list[str]] = {}
    for policy_type, group in declared.items():
        inverted.setdefault(str(group), []).append(str(policy_type))
    return {group: tuple(sorted(types)) for group, types in inverted.items()}


#: What the plan calls an objective and what a provider calls the loss that
#: implements it are different names for different things: the plan names a
#: family and a variant, the provider names the one implementation it ships. A
#: plan whose objective no provider here implements is refused by name, rather
#: than sent as a string the provider rejects three layers down after the
#: rollouts are already paid for.
PROVIDER_LOSS_NAMES: Mapping[str, str] = {"cispo": "cispo.slime.v1"}


def provider_loss_name(plan: Any) -> str:
    """The provider's loss for this plan's objective, or a named refusal."""

    kind = str(plan.objective.kind)
    loss = PROVIDER_LOSS_NAMES.get(kind)
    if loss is None:
        raise UnsupportedProviderError(
            f"objective {kind!r} (variant {plan.objective.variant!r}) has no loss "
            f"implemented by this provider; it ships {sorted(PROVIDER_LOSS_NAMES)}"
        )
    return loss


def _build_binder(
    config: RunConfig,
    *,
    provider: Any,
    publisher: PolicySetPublisher,
    resolver: EvaluationResolver,
    document: CapabilityDocument,
    contract_hash: str,
    sampling: SamplingProfile,
) -> CatalogPolicyBinder:
    plan = config.expanded_plan()
    return CatalogPolicyBinder(
        provider,
        publisher,
        resolver,
        base_model=config.model.id,
        model_family=config.model.family,
        renderer_profile=document.renderer_profile,
        container_contract_hash=contract_hash,
        policy_set_id=f"{config.run_id}::policy_set",
        wire_api=config.model.wire_api,
        sampling_transport=config.model.sampling_transport,
        loss_name=provider_loss_name(plan),
        policy_types=_policy_types(config, document),
        sampling=sampling,
        rank=config.model.rank,
        save_training_state=config.artifacts.retain_training_state,
    )


__all__ = [
    "LOOPBACK_HOSTS",
    "METADATA_ROUTE",
    "PLANE_SCHEMA_VERSION",
    "PROVIDER_BASE_URL_ENV",
    "PROVIDER_CREDENTIAL_ENV",
    "SAMPLER_ORIGIN_ENV",
    "CatalogPathError",
    "ContainerAuthError",
    "ContainerUnreachableError",
    "OriginPlan",
    "Plane",
    "PlaneError",
    "ProviderArtifactProbe",
    "ProviderCredentialError",
    "ProviderRenderer",
    "RendererUnavailableError",
    "SamplerOriginError",
    "UnsupportedProviderError",
    "build_container_client",
    "build_plane",
    "build_provider",
    "build_renderer",
    "fetch_metadata",
    "local_address_toward",
    "open_catalog",
    "open_plane",
    "plan_origin",
    "prepare_artifact_directory",
]
