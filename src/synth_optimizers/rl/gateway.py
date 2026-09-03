"""Session-scoped sampler origins that own the renderer and the token capture.

The container's harness makes an ordinary model call against a bound origin
whose path carries the per-attempt identity. This module renders the messages,
samples through the provider, and records one immutable ``InferenceCall`` per
proxied call with the exact prompt and generation token ids, the per-token
behavior logprobs from that same sampling call, the sampled mask, the renderer
profile fingerprint, the pinned revision, the finish reason, and the original
wire objects. Containers therefore need no tokenizer, and no second renderer
can enter the run.

Generalized from the working TBLite gateway, which already had immutable
per-route revision binding and a compaction counter. What it lacked, and what
is here: session-scoped origins keyed by attempt rather than by phase, a wire
choice carried on the pin, strict-prefix stitching with branch provenance, a
declared prompt-budget policy, and refusal of sampling evidence that cannot
prove a real logprob came back.

Nothing here names a task, a harness, an environment, or an algorithm.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, runtime_checkable

from ..contracts.rl_identity import GroupPin
from ..contracts.rl_records import (
    LOGPROB_SENTINEL,
    SAMPLING_TRANSPORTS,
    WIRE_APIS,
    CompactionProvenance,
    EvidenceError,
    InferenceCall,
    RendererProfile,
    TrainableEpisode,
    TrainableSegment,
    assert_strict_prefix,
    digest,
)
from ..providers.protocols import ProviderCheckpoint, SampleRequest, SampleResult
from ..providers.tinker.prime import parse_completion, tokenize_with_renderer
from .ports import AttemptFacts, PolicyRevision, PortError, SamplerOrigin

GATEWAY_SCHEMA_VERSION = "cispo.sampler_gateway.v1"

WIRE_CHAT_COMPLETIONS = "chat_completions"
WIRE_RESPONSES = "responses"

TRANSPORT_MESSAGE_IN = "message_in_capture_out"
TRANSPORT_TOKENS_IN = "tokens_in_tokens_out"

ATTEMPT_PATH_SEGMENT = "attempts"
WIRE_PATH_SUFFIX: Mapping[str, str] = {
    WIRE_CHAT_COMPLETIONS: "chat/completions",
    WIRE_RESPONSES: "responses",
}

PROMPT_BUDGET_POLICIES = frozenset({"refuse", "truncate", "compact"})
TRUNCATE_RULE = "prompt_budget.truncate.drop_oldest.v1"
COMPACT_RULE = "prompt_budget.compact.drop_oldest_with_marker.v1"
# The marker stands in the prompt for what was elided; the rule that elided it
# is recorded on the span, not spelled out to the model.
COMPACT_MARKER = "[{count} earlier turns elided]"
CONTAINER_REWRITE_RULE = "container.declared_history_rewrite.v1"
RESPONSES_PROJECTION = "responses.items_to_renderer_rows.v1"

# Turn k+1 normally extends turn k. When it cannot, the reason is named here
# rather than left to a re-render nobody recorded: the renderer refused to
# extend its own prior turn, the renderer re-closed that turn under a different
# token, or the container itself dropped turns out of the history it resent.
BRIDGE_DECLINED_RULE = "renderer.bridge_declined.full_rerender.v1"
BRIDGE_RECLOSE_RULE = "renderer.turn_close_rewrite.v1"
HISTORY_DROP_RULE = "container.detected_history_drop.v1"

# The provider names its own stop conditions; the record names three. An
# unmapped reason is refused rather than pooled into "stop".
_FINISH_REASONS: Mapping[str, str] = {
    "stop": "stop_token",
    "stop_token": "stop_token",
    "stop_sequence": "stop_token",
    "end_turn": "stop_token",
    "eos": "stop_token",
    "length": "length_cap",
    "length_cap": "length_cap",
    "max_tokens": "length_cap",
    "abort": "container_abort",
    "aborted": "container_abort",
    "cancelled": "container_abort",
    "container_abort": "container_abort",
}


class GatewayError(PortError):
    """The sampler gateway refused. Never degrade one of these to a reward."""


class UnknownOriginError(GatewayError):
    """No route was ever bound for this attempt id."""


class ClosedOriginError(GatewayError):
    """The origin was retired. Calls against it are refused, not replayed."""


class RouteRebindError(GatewayError):
    """A bound route is immutable: one attempt, one revision, one wire."""


class WireError(GatewayError):
    """The wire payload was malformed, or spoke a wire this route is not."""


class RendererMismatchError(GatewayError):
    """A second renderer tried to enter the run. Exactly one party renders."""


class RendererBridgeError(GatewayError):
    """The renderer offers no path from one sampled turn to the next.

    Without one, turn two could only be built by re-tokenizing turn one's text,
    which is the detokenize-then-retokenize the contract prohibits. The gateway
    names the renderer and refuses rather than doing it quietly.
    """


class PromptBudgetError(GatewayError):
    """The rendered prompt exceeded its budget under the declared policy."""


class AttemptFactsError(GatewayError):
    """The attempt facts a training episode needs were never declared."""


class SamplerEvidenceError(EvidenceError):
    """Sampling came back without evidence that can be trained on."""


class HistoryDivergenceError(EvidenceError):
    """The resent turn list neither extends the previous one nor drops from it.

    Removing turns is a compaction and forks a branch; inventing a turn that was
    never in the history, or editing one that was, is a rewrite the container has
    to declare. Either way the gateway never retokenizes new text onto old ids.
    """


# --------------------------------------------------------------- rendering


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """What the renderer produced for one turn, and nothing derived from text."""

    token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...] = ()


@runtime_checkable
class GatewayRenderer(Protocol):
    """The one party that turns wire turns into token ids for this run."""

    @property
    def profile(self) -> RendererProfile: ...

    @property
    def wire_apis(self) -> tuple[str, ...]: ...

    @property
    def bridges(self) -> bool:
        """Whether this renderer can extend a sampled turn without re-rendering."""
        ...

    def render(self, rows: Sequence[Mapping[str, Any]]) -> RenderedPrompt: ...

    def bridge(
        self,
        previous_prompt_token_ids: Sequence[int],
        previous_generation_token_ids: Sequence[int],
        new_rows: Sequence[Mapping[str, Any]],
    ) -> RenderedPrompt | None:
        """Extend ``previous_prompt + previous_generation`` by ``new_rows``.

        The sampled tokens are carried through verbatim; only the turns the
        container added this time are rendered. ``None`` means the renderer
        will not vouch for the extension -- a thinking-retention policy that
        drops history at a user boundary, a prior turn with no recoverable
        close -- and the caller must fork a branch rather than pretend.
        """
        ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


def _prime_bridge(
    renderer: Any,
    previous_prompt_token_ids: Sequence[int],
    previous_generation_token_ids: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
) -> RenderedPrompt | None:
    """The ``renderers`` package's own bridge, or nothing.

    ``bridge_to_next_turn`` exists precisely so the next prompt is the previous
    prompt-plus-generation with the new turns appended. It returns ``None``
    whenever it cannot prove that contract holds, and so does this.
    """

    bridge = getattr(renderer, "bridge_to_next_turn", None)
    if not callable(bridge) or not rows:
        return None
    rendered = bridge(
        [int(token) for token in previous_prompt_token_ids],
        [int(token) for token in previous_generation_token_ids],
        [dict(row) for row in rows],
    )
    if rendered is None:
        return None
    token_ids = tuple(int(token) for token in getattr(rendered, "token_ids", ()) or ())
    if not token_ids:
        return None
    return RenderedPrompt(
        token_ids=token_ids,
        stop_token_ids=tuple(int(token) for token in renderer.get_stop_token_ids()),
    )


@dataclass(frozen=True, slots=True)
class PrimeChatRenderer:
    """The Prime ``renderers`` wrapper, serving the chat-completions wire."""

    renderer: Any
    profile: RendererProfile

    @property
    def wire_apis(self) -> tuple[str, ...]:
        return (WIRE_CHAT_COMPLETIONS,)

    @property
    def bridges(self) -> bool:
        return callable(getattr(self.renderer, "bridge_to_next_turn", None))

    def bridge(
        self,
        previous_prompt_token_ids: Sequence[int],
        previous_generation_token_ids: Sequence[int],
        new_rows: Sequence[Mapping[str, Any]],
    ) -> RenderedPrompt | None:
        return _prime_bridge(
            self.renderer,
            previous_prompt_token_ids,
            previous_generation_token_ids,
            [dict(row) for row in new_rows],
        )

    def render(self, rows: Sequence[Mapping[str, Any]]) -> RenderedPrompt:
        rendered = tokenize_with_renderer(
            self.renderer, [dict(row) for row in rows], add_generation_prompt=True
        )
        token_ids = tuple(int(token) for token in rendered["prompt_token_ids"])
        if not token_ids:
            raise RendererMismatchError("renderer produced no prompt tokens")
        return RenderedPrompt(
            token_ids=token_ids,
            stop_token_ids=tuple(int(token) for token in rendered.get("stop_token_ids") or ()),
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return parse_completion(self.renderer, list(token_ids))


@dataclass(frozen=True, slots=True)
class PrimeResponsesRenderer:
    """The same renderer serving the Responses wire under a declared projection.

    The Prime package speaks chat rows, so Responses input items reach it
    through :data:`RESPONSES_PROJECTION`. That projection is part of this
    renderer's identity -- its profile digest folds the projection id in -- so a
    Responses span and a chat-completions span never share a behavior
    fingerprint and can never be pooled into one group.
    """

    renderer: Any
    profile: RendererProfile
    projection: str = RESPONSES_PROJECTION

    @classmethod
    def over(cls, renderer: Any, profile: RendererProfile) -> "PrimeResponsesRenderer":
        folded = replace(
            profile,
            profile_id=f"{profile.profile_id}+{RESPONSES_PROJECTION}",
            config_digest=digest(
                {"config": profile.config_digest, "projection": RESPONSES_PROJECTION}, length=32
            ),
        )
        return cls(renderer=renderer, profile=folded)

    @property
    def wire_apis(self) -> tuple[str, ...]:
        return (WIRE_RESPONSES,)

    @property
    def bridges(self) -> bool:
        return callable(getattr(self.renderer, "bridge_to_next_turn", None))

    def bridge(
        self,
        previous_prompt_token_ids: Sequence[int],
        previous_generation_token_ids: Sequence[int],
        new_rows: Sequence[Mapping[str, Any]],
    ) -> RenderedPrompt | None:
        # The new items reach the renderer through the same declared projection
        # the full render uses, so a bridged Responses prompt and a re-rendered
        # one are the same tokens under the same folded profile identity.
        if not new_rows:
            return None
        return _prime_bridge(
            self.renderer,
            previous_prompt_token_ids,
            previous_generation_token_ids,
            project_responses_items(new_rows),
        )

    def render(self, rows: Sequence[Mapping[str, Any]]) -> RenderedPrompt:
        projected = project_responses_items(rows)
        rendered = tokenize_with_renderer(
            self.renderer, projected, add_generation_prompt=True
        )
        token_ids = tuple(int(token) for token in rendered["prompt_token_ids"])
        if not token_ids:
            raise RendererMismatchError("renderer produced no prompt tokens")
        return RenderedPrompt(
            token_ids=token_ids,
            stop_token_ids=tuple(int(token) for token in rendered.get("stop_token_ids") or ()),
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return parse_completion(self.renderer, list(token_ids))


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, Mapping):
        text = part.get("text")
        if isinstance(text, str):
            return text
    raise WireError(f"responses content part {part!r} carries no text")


def project_responses_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Declared projection of Responses input items onto renderer rows.

    Unknown item types are refused rather than dropped: a silently skipped item
    is a prompt the trainer cannot reconstruct.
    """

    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise WireError(f"responses input item {item!r} is not an object")
        kind = str(item.get("type") or "message")
        if kind == "function_call":
            rows.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"name": item.get("name"), "arguments": item.get("arguments")},
                        sort_keys=True,
                    ),
                }
            )
            continue
        if kind == "function_call_output":
            rows.append({"role": "tool", "content": str(item.get("output", ""))})
            continue
        if kind not in {"message", "input_text", "output_text"}:
            raise WireError(f"unsupported responses input item type {kind!r}")
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            raise WireError("responses message item requires a role")
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, Sequence):
            text = "".join(_part_text(part) for part in content)
        else:
            raise WireError("responses message item requires string or list content")
        rows.append({"role": role.strip(), "content": text})
    if not rows:
        raise WireError("responses request carries no input items")
    return rows


# ------------------------------------------------------- conversation shape


TurnIdentity = tuple[str, str]


def row_identity(row: Mapping[str, Any]) -> TurnIdentity:
    """A wire-agnostic identity for one turn, for comparing two message lists.

    The container knows nothing about tokens, so the only thing it can be held
    to across turns is the turns themselves. Whitespace is stripped because a
    harness echoes back the text it was handed, and every harness strips it.
    """

    role = str(row.get("role") or "").strip()
    content = row.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        text = "".join(_part_text(part) for part in content)
    elif content is None:
        text = ""
    else:
        text = str(content)
    structured = row.get("tool_calls") or row.get("tool_call_id")
    if structured is not None:
        text = f"{text}\x00{json.dumps(structured, sort_keys=True, default=str)}"
    return role, text.strip()


def removals_between(
    history: Sequence[TurnIdentity], rows: Sequence[TurnIdentity]
) -> tuple[int, ...] | None:
    """Which history turns ``rows`` dropped, or ``None`` if it did not just drop.

    A compaction removes turns; it does not invent them. ``rows`` therefore has
    to retain the last turn of the history -- the assistant turn the gateway
    itself sampled -- and everything it keeps before that has to appear in the
    history, in order. Anything else is an edit, and an edit is not detectable
    as a removal, so it is refused rather than guessed at.
    """

    if not history:
        return None
    boundary = -1
    for index in range(len(rows) - 1, -1, -1):
        if rows[index] == history[-1]:
            boundary = index
            break
    if boundary < 0:
        return None
    removed: list[int] = []
    cursor = 0
    for row in rows[: boundary + 1]:
        while cursor < len(history) and history[cursor] != row:
            removed.append(cursor)
            cursor += 1
        if cursor >= len(history):
            return None
        cursor += 1
    if cursor != len(history):
        return None
    return tuple(removed)


# ------------------------------------------------------------ wire parsing


@dataclass(frozen=True, slots=True)
class WireRequest:
    """One proxied call as the container sent it, before any rendering."""

    wire_api: str
    rows: tuple[Mapping[str, Any], ...]
    max_tokens: int
    temperature: float
    seed: int | None
    prompt_token_ids: tuple[int, ...]
    declared_rewrite: Mapping[str, Any] | None
    declared_renderer_fingerprint: str
    raw: Mapping[str, Any]


def _int_field(payload: Mapping[str, Any], name: str, default: int) -> int:
    value = payload.get(name, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireError(f"{name} must be a number")
    return int(value)


def parse_wire_request(wire_api: str, payload: Mapping[str, Any]) -> WireRequest:
    """Parse a payload as the wire the route is pinned to, and no other."""

    if not isinstance(payload, Mapping):
        raise WireError("request body must be a JSON object")
    if wire_api == WIRE_CHAT_COMPLETIONS:
        items = payload.get("messages")
        budget_field = "max_tokens"
    elif wire_api == WIRE_RESPONSES:
        items = payload.get("input")
        budget_field = "max_output_tokens"
    else:
        raise WireError(f"unknown wire_api {wire_api!r}")
    prompt_token_ids = tuple(int(token) for token in payload.get("prompt_token_ids") or ())
    rows: tuple[Mapping[str, Any], ...] = ()
    if items is not None:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise WireError(f"{wire_api} request turns must be a list")
        for item in items:
            if not isinstance(item, Mapping):
                raise WireError(f"{wire_api} request carries a non-object turn")
        rows = tuple(dict(item) for item in items)
    if not rows and not prompt_token_ids:
        raise WireError(f"{wire_api} request carries neither turns nor prompt token ids")
    temperature = payload.get("temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise WireError("temperature must be a number")
    seed_value = payload.get("seed")
    if seed_value is not None and (
        isinstance(seed_value, bool) or not isinstance(seed_value, (int, float))
    ):
        raise WireError("seed must be a number")
    rewrite = payload.get("synth_history_rewrite")
    if rewrite is not None and not isinstance(rewrite, Mapping):
        raise WireError("synth_history_rewrite must be an object")
    return WireRequest(
        wire_api=wire_api,
        rows=rows,
        max_tokens=max(1, _int_field(payload, budget_field, 512)),
        temperature=float(temperature),
        seed=None if seed_value is None else int(seed_value),
        prompt_token_ids=prompt_token_ids,
        declared_rewrite=None if rewrite is None else dict(rewrite),
        declared_renderer_fingerprint=str(payload.get("renderer_profile_fingerprint") or ""),
        raw=dict(payload),
    )


# ---------------------------------------------------------- prompt budgets


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """The declared behavior for an overlong rendered prompt.

    ``refuse`` fails the attempt, ``truncate`` drops the oldest droppable turns,
    ``compact`` drops them and leaves a marker turn in their place. Every
    outcome, refusals included, is recorded on the route.
    """

    max_prompt_tokens: int
    policy: str = "refuse"
    keep_head_rows: int = 1
    keep_tail_rows: int = 2
    reserve_completion_tokens: bool = True
    marker_role: str = "system"

    def __post_init__(self) -> None:
        if self.policy not in PROMPT_BUDGET_POLICIES:
            raise PromptBudgetError(
                f"unknown prompt budget policy {self.policy!r}; "
                f"expected one of {sorted(PROMPT_BUDGET_POLICIES)}"
            )
        if self.max_prompt_tokens < 1:
            raise PromptBudgetError("max_prompt_tokens must be positive")
        if self.keep_head_rows < 0 or self.keep_tail_rows < 1:
            raise PromptBudgetError("a budget must keep the newest turn and a non-negative head")

    @property
    def rewrites(self) -> bool:
        return self.policy != "refuse"

    @property
    def rule(self) -> str:
        return TRUNCATE_RULE if self.policy == "truncate" else COMPACT_RULE


@dataclass(frozen=True, slots=True)
class BudgetEvent:
    """One budget decision, kept whether it rewrote, passed, or refused."""

    call_index: int
    policy: str
    rule: str
    prompt_tokens_before: int
    prompt_tokens_after: int
    removed_row_indices: tuple[int, ...] = ()
    refused: bool = False


UNBOUNDED_BUDGET = PromptBudget(max_prompt_tokens=2**31 - 1, policy="refuse")


# ------------------------------------------------------------- the stitch


@dataclass(frozen=True, slots=True)
class _Stitch:
    """How this turn relates to the last one: a token splice, or a fork.

    ``prompt`` is the spliced sequence when the renderer bridged the sampled
    turn forward. ``rule`` names why it could not, in which case the caller
    re-renders and forks a branch under that rule. Both empty means there was
    no previous turn to stitch to.
    """

    prompt: tuple[int, ...] | None = None
    rule: str | None = None
    removed: tuple[int, ...] = ()


# --------------------------------------------------------------- the route


@dataclass(slots=True)
class _Route:
    origin: SamplerOrigin
    revision: PolicyRevision
    pin: GroupPin
    sample_index: int
    checkpoint: ProviderCheckpoint
    facts: AttemptFacts | None = None
    closed: bool = False
    branch_id: str = "root"
    fork_count: int = 0
    calls: list[InferenceCall] = field(default_factory=list)
    budget_events: list[BudgetEvent] = field(default_factory=list)
    # The conversation as the container last sent it, with the assistant turn
    # the gateway sampled appended. The next call is measured against this: it
    # says where the container's new turns begin, and whether it dropped any.
    history: tuple[TurnIdentity, ...] = ()
    # One attempt's calls are a sequence and are serialized against each other;
    # two attempts are not, so the route map lock is never held across sampling.
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def rollout_id(self) -> str:
        return self.facts.rollout_id if self.facts is not None else self.origin.proxy_request_id


@runtime_checkable
class SamplerBackend(Protocol):
    """The provider surface the gateway needs. ``TrainingProvider`` satisfies it."""

    def sample_checkpoint(
        self, checkpoint: ProviderCheckpoint, request: SampleRequest
    ) -> SampleResult: ...


def _assert_sampling_evidence(result: SampleResult, *, context: str) -> None:
    """Refuse sampling that cannot prove a real per-token logprob came back."""

    tokens = tuple(result.token_ids)
    logprobs = tuple(result.logprobs)
    if not tokens:
        raise SamplerEvidenceError(f"{context}: sampling returned no tokens")
    if len(logprobs) != len(tokens):
        raise SamplerEvidenceError(
            f"{context}: {len(logprobs)} logprobs for {len(tokens)} generated tokens"
        )
    for index, value in enumerate(logprobs):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SamplerEvidenceError(f"{context}: logprob {index} is not a number")
        if math.isnan(value) or math.isinf(value):
            raise SamplerEvidenceError(f"{context}: logprob {index} is not finite")
        if float(value) == LOGPROB_SENTINEL:
            raise SamplerEvidenceError(
                f"{context}: logprob {index} is the provider sentinel {LOGPROB_SENTINEL}; "
                "its presence can never prove a real logprob was returned"
            )
    if all(float(value) == 0.0 for value in logprobs):
        raise SamplerEvidenceError(f"{context}: logprobs are identically zero across the span")


def _finish_reason(raw: str) -> str:
    mapped = _FINISH_REASONS.get(str(raw).strip().lower())
    if mapped is None:
        raise SamplerEvidenceError(
            f"provider finish reason {raw!r} maps to no declared reason; a length-truncated "
            "tail must not be pooled with a stopped one"
        )
    return mapped


class SamplerGatewayService:
    """A :class:`~synth_optimizers.rl.ports.SamplerGateway` over one renderer.

    One instance per run. Routes are session-scoped: the attempt id lives in
    the origin path, so stitching is a URL parse and a leaked credential cannot
    cross rollouts.
    """

    def __init__(
        self,
        renderer: GatewayRenderer,
        sampler: SamplerBackend,
        *,
        origin_root: str = "http://127.0.0.1",
        prompt_budget: PromptBudget = UNBOUNDED_BUDGET,
        credential_salt: str = "",
        origin_ttl_seconds: int = 0,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._renderer = renderer
        self._sampler = sampler
        self._origin_root = origin_root.rstrip("/")
        self._budget = prompt_budget
        self._salt = credential_salt
        self._ttl = int(origin_ttl_seconds)
        self._now = now
        self._routes: dict[str, _Route] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ identity

    @property
    def renderer_profile(self) -> RendererProfile:
        return self._renderer.profile

    @property
    def origin_root(self) -> str:
        return self._origin_root

    def set_origin_root(self, root: str) -> None:
        """Point origins at a real listener. Refused once a route exists."""

        with self._lock:
            if self._routes:
                raise RouteRebindError("origins are already bound; the root cannot move")
            self._origin_root = root.rstrip("/")

    # ------------------------------------------------------------- binding

    def bind(
        self,
        revision: PolicyRevision,
        *,
        pin: GroupPin,
        sample_index: int,
        proxy_request_id: str,
        attempt: AttemptFacts | None = None,
    ) -> SamplerOrigin:
        if not isinstance(proxy_request_id, str) or not proxy_request_id.strip():
            raise RouteRebindError("a proxy_request_id is required")
        key = proxy_request_id.strip()
        if "/" in key:
            raise RouteRebindError(f"proxy_request_id {key!r} may not contain a path separator")
        with self._lock:
            existing = self._routes.get(key)
            if existing is not None:
                if (
                    existing.revision != revision
                    or existing.pin.pin_digest != pin.pin_digest
                    or existing.sample_index != sample_index
                ):
                    raise RouteRebindError(
                        f"route {key} is bound to revision {existing.revision.revision_id!r}; "
                        "a bound route is immutable for the life of the attempt"
                    )
                return existing.origin
            self._assert_bindable(revision, pin)
            origin = SamplerOrigin(
                base_url=f"{self._origin_root}/v1/{ATTEMPT_PATH_SEGMENT}/{key}",
                credential=self._credential(revision, pin, sample_index, key),
                policy_revision=revision.revision,
                behavior_fingerprint=revision.behavior_fingerprint,
                proxy_request_id=key,
                wire_api=pin.wire_api,
                sampling_transport=pin.sampling_transport,
                expires_at=self._expiry(),
            )
            self._routes[key] = _Route(
                origin=origin,
                revision=revision,
                pin=pin,
                sample_index=sample_index,
                checkpoint=_checkpoint_for(revision),
            )
        # After the route exists, never before: declaring facts against an
        # unregistered route raises, and every dispatch passes facts, so doing
        # this first meant no attempt could be bound at all.
        if attempt is not None:
            self.declare_attempt(
                key,
                rollout_id=attempt.rollout_id,
                task_id=attempt.task_id,
                seed=attempt.seed,
                terminal_status=attempt.terminal_status,
                provisional=True,
            )
        return origin

    def close(self, proxy_request_id: str) -> None:
        with self._lock:
            route = self._routes.get(str(proxy_request_id).strip())
            if route is None:
                raise UnknownOriginError(f"no origin was bound for attempt {proxy_request_id!r}")
            route.closed = True

    def declare_attempt(
        self,
        proxy_request_id: str,
        *,
        rollout_id: str,
        task_id: str,
        seed: int,
        terminal_status: str = "completed",
        provisional: bool = False,
    ) -> AttemptFacts:
        """Bind the attempt facts ``bind`` does not carry: task id and seed.

        ``SamplerGateway.bind`` receives a group pin and a sample index, and a
        ``TrainableEpisode`` requires a task id and a seed, so those facts have
        to arrive by a second door.

        One of those facts cannot be known at bind time. The origin is what
        gets submitted, so the container has no rollout id to give until it has
        accepted the attempt -- and a container that serves submission
        synchronously runs the whole episode inside that call, so by the time
        its rollout id comes back the calls are already recorded. The executor
        therefore binds with its own attempt id, marked provisional, and this
        replaces it once the container names its own. Everything else about an
        attempt stays fixed from the first call: only the provisional rollout id
        may be settled, and only when the task and seed still agree.
        """

        route = self._locked_route(proxy_request_id)
        with route.lock:
            held = route.facts
            settling = (
                held is not None
                and held.provisional
                and held.task_id == str(task_id).strip()
                and held.seed == int(seed)
            )
            if route.calls and not settling:
                raise AttemptFactsError(
                    f"attempt {proxy_request_id} already recorded calls; its facts are fixed"
                )
            if route.calls and settling:
                # Re-stamp what is already recorded, so no call is left naming
                # the placeholder the container never knew about.
                route.calls = [
                    replace(call, rollout_id=str(rollout_id).strip()) for call in route.calls
                ]
            facts = AttemptFacts(
                rollout_id=str(rollout_id).strip(),
                task_id=str(task_id).strip(),
                seed=int(seed),
                terminal_status=str(terminal_status).strip(),
                provisional=provisional,
            )
            if not facts.rollout_id or not facts.task_id:
                raise AttemptFactsError("an attempt declares both a rollout id and a task id")
            route.facts = facts
            return facts

    # -------------------------------------------------------------- serving

    def handle(
        self,
        proxy_request_id: str,
        payload: Mapping[str, Any],
        *,
        credential: str | None = None,
        wire_api: str | None = None,
    ) -> Mapping[str, Any]:
        """Proxy one model call and record its immutable evidence."""

        with self._lock:
            route = self._require_open_route(proxy_request_id)
            if credential is not None and credential != route.origin.credential:
                raise UnknownOriginError(
                    f"attempt {proxy_request_id} was presented a credential it was not issued"
                )
            if wire_api is not None and wire_api != route.origin.wire_api:
                raise WireError(
                    f"attempt {proxy_request_id} is pinned to {route.origin.wire_api}; "
                    f"a {wire_api} call against it is a different dataset"
                )
        with route.lock:
            if route.closed:
                raise ClosedOriginError(
                    f"attempt {proxy_request_id} was retired; its origin no longer samples"
                )
            request = parse_wire_request(route.origin.wire_api, payload)
            prompt, rows, provenance = self._prompt_for(route, request)
            completion_budget = max(
                1, min(request.max_tokens, self._budget.max_prompt_tokens - len(prompt))
            )
            sample_request = SampleRequest(
                request_id=digest(
                    {
                        "attempt": route.origin.proxy_request_id,
                        "call_index": len(route.calls),
                        "prompt": list(prompt),
                    },
                    length=32,
                ),
                prompt_token_ids=prompt,
                max_tokens=completion_budget,
                temperature=request.temperature,
                seed=request.seed,
                checkpoint_id=route.revision.checkpoint_id,
            )
            result = self._sampler.sample_checkpoint(route.checkpoint, sample_request)
            _assert_sampling_evidence(
                result, context=f"attempt {route.origin.proxy_request_id}"
            )
            finish_reason = _finish_reason(result.finish_reason)
            text = result.text or self._renderer.decode(result.token_ids)
            body = self._wire_response(route, request, result, text, finish_reason, prompt)
            call = self._record_call(
                route,
                request=request,
                rows=rows,
                prompt=prompt,
                result=result,
                finish_reason=finish_reason,
                provenance=provenance,
                body=body,
            )
            self._remember(route, request, text)
            return {**body, "synth_capture": _capture(call, self.renderer_profile)}

    def calls(self, proxy_request_id: str) -> tuple[InferenceCall, ...]:
        route = self._locked_route(proxy_request_id)
        with route.lock:
            return tuple(route.calls)

    def budget_events(self, proxy_request_id: str) -> tuple[BudgetEvent, ...]:
        route = self._locked_route(proxy_request_id)
        with route.lock:
            return tuple(route.budget_events)

    def origin(self, proxy_request_id: str) -> SamplerOrigin:
        with self._lock:
            return self._require_route(proxy_request_id).origin

    # ------------------------------------------------------------ evidence

    def episode(self, proxy_request_id: str) -> TrainableEpisode:
        """The captured evidence for one attempt, validated or refused."""

        route = self._locked_route(proxy_request_id)
        with route.lock:
            if route.facts is None:
                raise AttemptFactsError(
                    f"attempt {proxy_request_id} never declared its task id and seed; "
                    "a training episode cannot be identified without them"
                )
            if not route.calls:
                raise EvidenceError(f"attempt {proxy_request_id} proxied no model calls")
            segments: list[TrainableSegment] = []
            for call in route.calls:
                call.validate_for_training()
                prompt_length = len(call.prompt_token_ids)
                segments.append(
                    TrainableSegment(
                        token_ids=call.full_sequence,
                        loss_mask=call.loss_mask,
                        behavior_logprobs=(0.0,) * prompt_length + call.generation_logprobs,
                        branch_id=call.branch_id,
                        parameter_group_id=call.parameter_group_id,
                        call_ids=(call.call_id,),
                        author_kind=call.author_kind,
                        policy_revision=call.policy_revision,
                        policy_set_revision_id=call.policy_set_revision_id,
                    )
                )
            episode = TrainableEpisode(
                rollout_id=route.facts.rollout_id,
                task_id=route.facts.task_id,
                seed=route.facts.seed,
                policy_revision=route.revision.revision,
                behavior_fingerprint=route.revision.behavior_fingerprint,
                segments=tuple(segments),
                terminal_status=route.facts.terminal_status,
                usage={
                    "calls": len(route.calls),
                    "prompt_tokens": sum(len(c.prompt_token_ids) for c in route.calls),
                    "generation_tokens": sum(len(c.generation_token_ids) for c in route.calls),
                },
                policy_set_revision_id=route.revision.policy_set_revision_id,
                root_rollout_id=route.facts.rollout_id,
                trace_digest=digest(
                    {
                        "schema_version": GATEWAY_SCHEMA_VERSION,
                        "attempt": route.origin.proxy_request_id,
                        "renderer": self.renderer_profile.fingerprint,
                        "calls": [
                            {
                                "call_id": call.call_id,
                                "branch_id": call.branch_id,
                                "sequence": list(call.full_sequence),
                                "logprobs": list(call.generation_logprobs),
                            }
                            for call in route.calls
                        ],
                    },
                    length=64,
                ),
            )
            episode.validate()
            return episode

    # -------------------------------------------------------------- private

    def _assert_bindable(self, revision: PolicyRevision, pin: GroupPin) -> None:
        if pin.wire_api not in WIRE_APIS:
            raise WireError(f"unknown wire_api {pin.wire_api!r}")
        if pin.sampling_transport not in SAMPLING_TRANSPORTS:
            raise WireError(f"unknown sampling_transport {pin.sampling_transport!r}")
        if pin.wire_api not in self._renderer.wire_apis:
            raise RendererMismatchError(
                f"renderer {self.renderer_profile.profile_id} serves "
                f"{list(self._renderer.wire_apis)}, not {pin.wire_api!r}"
            )
        if pin.policy_revision != revision.revision:
            raise RouteRebindError(
                f"group pin names revision {pin.policy_revision} and the binding carries "
                f"{revision.revision}"
            )
        if pin.behavior_fingerprint != revision.behavior_fingerprint:
            raise RouteRebindError(
                "group pin behavior fingerprint disagrees with the revision being bound"
            )
        if pin.policy_revision_id not in (None, revision.revision_id):
            raise RouteRebindError(
                f"group pin names policy revision {pin.policy_revision_id!r}, "
                f"binding carries {revision.revision_id!r}"
            )
        if (
            pin.policy_set_revision_id is not None
            and revision.policy_set_revision_id is not None
            and pin.policy_set_revision_id != revision.policy_set_revision_id
        ):
            raise RouteRebindError("group pin and revision name different policy sets")
        if not revision.sampler_reference.strip():
            raise RouteRebindError(
                f"revision {revision.revision_id} carries no sampler artifact reference"
            )
        if pin.sampling_transport == TRANSPORT_TOKENS_IN and self._budget.rewrites:
            raise PromptBudgetError(
                "a tokens-in transport gives the gateway no turns to rewrite; declare a "
                "refusing prompt budget or let the row-owning side compact"
            )

    def _credential(
        self, revision: PolicyRevision, pin: GroupPin, sample_index: int, key: str
    ) -> str:
        return "cispo-" + digest(
            {
                "salt": self._salt,
                "attempt": key,
                "group_id": pin.group_id,
                "run_id": pin.run_id,
                "sample_index": sample_index,
                "wire_api": pin.wire_api,
                "sampling_transport": pin.sampling_transport,
                "policy_kind": pin.policy_kind,
                "policy_revision": revision.revision,
                "policy_revision_id": revision.revision_id,
            },
            length=48,
        )

    def _expiry(self) -> str:
        if self._ttl <= 0:
            return ""
        moment = self._now() + timedelta(seconds=self._ttl)
        return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _locked_route(self, proxy_request_id: str) -> _Route:
        with self._lock:
            return self._require_route(proxy_request_id)

    def _require_route(self, proxy_request_id: str) -> _Route:
        route = self._routes.get(str(proxy_request_id).strip())
        if route is None:
            raise UnknownOriginError(f"no origin was bound for attempt {proxy_request_id!r}")
        return route

    def _require_open_route(self, proxy_request_id: str) -> _Route:
        route = self._require_route(proxy_request_id)
        if route.closed:
            raise ClosedOriginError(
                f"attempt {proxy_request_id} was retired; its origin no longer samples"
            )
        return route

    def _prompt_for(
        self, route: _Route, request: WireRequest
    ) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...], CompactionProvenance | None]:
        """Render the turn, apply the declared budget, and say what it cost."""

        if route.origin.sampling_transport == TRANSPORT_TOKENS_IN:
            return self._tokens_in_prompt(route, request)
        if not request.rows:
            raise WireError("a message-in call carries no turns")
        rows = request.rows
        # Turn two is turn one's prompt-plus-generation with the new turns
        # appended, spliced from the ids the gateway already holds. Rendering
        # the whole list again would tokenize the sampled turn from its text.
        stitch = self._stitch(route, request)
        prompt = (
            stitch.prompt if stitch.prompt is not None else self._renderer.render(rows).token_ids
        )
        cap = self._effective_cap(request)
        removed: tuple[int, ...] = ()
        before = len(prompt)
        if len(prompt) > cap:
            if self._budget.policy == "refuse":
                route.budget_events.append(
                    BudgetEvent(
                        call_index=len(route.calls),
                        policy="refuse",
                        rule="prompt_budget.refuse.v1",
                        prompt_tokens_before=before,
                        prompt_tokens_after=before,
                        refused=True,
                    )
                )
                raise PromptBudgetError(
                    f"rendered prompt is {before} tokens against a budget of {cap}; "
                    "the declared policy is to refuse the attempt"
                )
            rows, prompt, removed = self._shrink(rows, cap)
        route.budget_events.append(
            BudgetEvent(
                call_index=len(route.calls),
                policy=self._budget.policy,
                rule=self._budget.rule if removed else "prompt_budget.within_budget.v1",
                prompt_tokens_before=before,
                prompt_tokens_after=len(prompt),
                removed_row_indices=removed,
            )
        )
        declared = self._declared_rewrite(route, request, prompt)
        if removed:
            return (
                prompt,
                rows,
                self._provenance(route, prompt, self._budget.rule, removed),
            )
        if declared is not None:
            return prompt, rows, declared
        if stitch.rule is not None:
            return prompt, rows, self._provenance(route, prompt, stitch.rule, stitch.removed)
        return prompt, rows, None

    def _stitch(self, route: _Route, request: WireRequest) -> _Stitch:
        """Splice this turn onto the last one, or name why it cannot be spliced.

        The container hands over a whole message list and knows nothing about
        tokens, so the gateway is the party that has to tell an extension from
        a rewrite. An extension bridges; a removal forks under its own rule; an
        edit is neither, and is refused rather than retokenized onto old ids.
        """

        previous = route.calls[-1] if route.calls else None
        if previous is None or request.declared_rewrite is not None:
            return _Stitch()
        rows = tuple(row_identity(row) for row in self._identity_rows(route, request.rows))
        history = route.history
        if not history:
            return _Stitch(rule=BRIDGE_DECLINED_RULE)
        if len(rows) < len(history) or rows[: len(history)] != history:
            removed = removals_between(history, rows)
            if removed is None:
                raise HistoryDivergenceError(
                    f"attempt {route.origin.proxy_request_id} resent a history that neither "
                    f"extends nor drops turns from the {len(history)} it was sampled against; "
                    "an edited or invented turn must be declared in synth_history_rewrite"
                )
            return _Stitch(rule=HISTORY_DROP_RULE, removed=removed)
        added = request.rows[len(history) :]
        if not getattr(self._renderer, "bridges", False):
            raise RendererBridgeError(
                f"renderer {self.renderer_profile.profile_id} offers no bridge from one turn "
                "to the next, so turn two could only be built by re-tokenizing turn one's "
                "text; bind a renderer that can extend a sampled turn"
            )
        bridged = self._renderer.bridge(
            previous.prompt_token_ids, previous.generation_token_ids, added
        )
        if bridged is None:
            return _Stitch(rule=BRIDGE_DECLINED_RULE)
        anchor = previous.full_sequence
        if bridged.token_ids[: len(anchor)] != anchor:
            # The renderer re-closed the prior turn under a different token.
            # That is a real rewrite of sampled context, so it forks.
            return _Stitch(rule=BRIDGE_RECLOSE_RULE)
        return _Stitch(prompt=bridged.token_ids)

    def _identity_rows(
        self, route: _Route, rows: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]:
        """The renderer rows these wire turns stand for, one for one."""

        if route.origin.wire_api == WIRE_RESPONSES:
            return project_responses_items(rows)
        return rows

    def _remember(self, route: _Route, request: WireRequest, reply: str) -> None:
        """Record the conversation the next turn will be measured against."""

        if route.origin.sampling_transport == TRANSPORT_TOKENS_IN or not request.rows:
            route.history = ()
            return
        route.history = tuple(
            row_identity(row) for row in self._identity_rows(route, request.rows)
        ) + (("assistant", reply.strip()),)

    def _tokens_in_prompt(
        self, route: _Route, request: WireRequest
    ) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...], CompactionProvenance | None]:
        if not request.prompt_token_ids:
            raise WireError(
                "a tokens-in transport must send prompt_token_ids; text is not authoritative"
            )
        fingerprint = request.declared_renderer_fingerprint
        if fingerprint and fingerprint != self.renderer_profile.fingerprint:
            raise RendererMismatchError(
                f"tokens-in call declares renderer {fingerprint!r}, the run renders with "
                f"{self.renderer_profile.fingerprint!r}"
            )
        prompt = request.prompt_token_ids
        cap = self._effective_cap(request)
        refused = len(prompt) > cap
        route.budget_events.append(
            BudgetEvent(
                call_index=len(route.calls),
                policy="refuse",
                rule="prompt_budget.refuse.v1"
                if refused
                else "prompt_budget.within_budget.v1",
                prompt_tokens_before=len(prompt),
                prompt_tokens_after=len(prompt),
                refused=refused,
            )
        )
        if refused:
            raise PromptBudgetError(
                f"tokens-in prompt is {len(prompt)} tokens against a budget of {cap}; "
                "the gateway may not rewrite turns it was never sent"
            )
        return prompt, request.rows, self._declared_rewrite(route, request, prompt)

    def _effective_cap(self, request: WireRequest) -> int:
        cap = self._budget.max_prompt_tokens
        if self._budget.reserve_completion_tokens:
            cap -= request.max_tokens
        if cap < 1:
            raise PromptBudgetError(
                f"a completion budget of {request.max_tokens} leaves no room under a prompt "
                f"budget of {self._budget.max_prompt_tokens}"
            )
        return cap

    def _shrink(
        self, rows: tuple[Mapping[str, Any], ...], cap: int
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[int, ...], tuple[int, ...]]:
        head = self._budget.keep_head_rows
        tail = self._budget.keep_tail_rows
        removed: list[int] = []
        cursor = head
        current = rows
        while True:
            rendered = self._renderer.render(current)
            if len(rendered.token_ids) <= cap:
                return current, rendered.token_ids, tuple(removed)
            if cursor >= len(rows) - tail:
                raise PromptBudgetError(
                    f"prompt is {len(rendered.token_ids)} tokens and the budget of {cap} cannot "
                    f"be met while keeping {head} leading and {tail} trailing turns"
                )
            removed.append(cursor)
            cursor += 1
            kept = [row for index, row in enumerate(rows) if index not in set(removed)]
            if self._budget.policy == "compact":
                kept.insert(
                    min(head, len(kept)),
                    {
                        "role": self._budget.marker_role,
                        "content": COMPACT_MARKER.format(count=len(removed)),
                    },
                )
            current = tuple(kept)

    def _declared_rewrite(
        self, route: _Route, request: WireRequest, prompt: tuple[int, ...]
    ) -> CompactionProvenance | None:
        if request.declared_rewrite is None:
            return None
        rewrite = request.declared_rewrite
        rule = str(rewrite.get("rule") or CONTAINER_REWRITE_RULE)
        indices = tuple(int(value) for value in rewrite.get("removed_message_indices") or ())
        return self._provenance(
            route,
            prompt,
            rule,
            indices,
            authored_by_policy=bool(rewrite.get("authored_by_policy", False)),
        )

    def _provenance(
        self,
        route: _Route,
        prompt: tuple[int, ...],
        rule: str,
        removed: Sequence[int],
        *,
        authored_by_policy: bool = False,
    ) -> CompactionProvenance:
        previous = route.calls[-1] if route.calls else None
        if previous is None:
            divergence = 0
        else:
            sequence = previous.full_sequence
            divergence = next(
                (i for i, (a, b) in enumerate(zip(sequence, prompt, strict=False)) if a != b),
                min(len(sequence), len(prompt)),
            )
        return CompactionProvenance(
            rule=rule,
            divergence_index=divergence,
            removed_message_indices=tuple(int(index) for index in removed),
            authored_by_policy=authored_by_policy,
        )

    def _record_call(
        self,
        route: _Route,
        *,
        request: WireRequest,
        rows: tuple[Mapping[str, Any], ...],
        prompt: tuple[int, ...],
        result: SampleResult,
        finish_reason: str,
        provenance: CompactionProvenance | None,
        body: Mapping[str, Any],
    ) -> InferenceCall:
        previous = route.calls[-1] if route.calls else None
        generation = tuple(int(token) for token in result.token_ids)
        branch_id = route.branch_id
        parent_branch: str | None = None
        # A rewrite severs nothing when there is no prior generation to sever,
        # so the first turn of an attempt stays on the root branch.
        compaction = provenance if previous is not None else None
        if compaction is not None:
            branch_id = f"{route.branch_id}.{route.fork_count + 1}"
            parent_branch = route.branch_id
        call = InferenceCall(
            call_id=digest(
                {
                    "attempt": route.origin.proxy_request_id,
                    "index": len(route.calls),
                    "prompt": list(prompt),
                    "generation": list(generation),
                },
                length=32,
            ),
            proxy_request_id=route.origin.proxy_request_id,
            rollout_id=route.rollout_id,
            group_id=route.pin.group_id,
            sample_index=route.sample_index,
            behavior_fingerprint=route.revision.behavior_fingerprint,
            policy_revision=route.revision.revision,
            wire_api=route.origin.wire_api,
            sampling_transport=route.origin.sampling_transport,
            token_capture_provenance="engine_meta",
            prompt_token_ids=prompt,
            generation_token_ids=generation,
            generation_logprobs=tuple(float(value) for value in result.logprobs),
            sampled_mask=(1,) * len(generation),
            finish_reason=finish_reason,
            stop_token_ids=self.renderer_profile.stop_token_ids,
            author_kind="policy",
            renderer_profile_fingerprint=self.renderer_profile.fingerprint,
            branch_id=branch_id,
            parent_branch_id=parent_branch,
            compaction=compaction,
            parameter_group_id=route.revision.parameter_group_id,
            policy_set_revision_id=route.revision.policy_set_revision_id,
            wire_request=dict(request.raw),
            wire_response=dict(body),
            usage={
                "prompt_tokens": len(prompt),
                "completion_tokens": len(generation),
                "total_tokens": len(prompt) + len(generation),
                "rendered_turns": len(rows),
            },
            created_at=self._now().isoformat(timespec="microseconds").replace("+00:00", "Z"),
        )
        if previous is not None:
            assert_strict_prefix(previous, call)
        route.calls.append(call)
        if parent_branch is not None:
            route.branch_id = branch_id
            route.fork_count += 1
        return call

    def _wire_response(
        self,
        route: _Route,
        request: WireRequest,
        result: SampleResult,
        text: str,
        finish_reason: str,
        prompt: tuple[int, ...],
    ) -> Mapping[str, Any]:
        generated = len(tuple(result.token_ids))
        usage = {
            "prompt_tokens": len(prompt),
            "completion_tokens": generated,
            "total_tokens": len(prompt) + generated,
        }
        identifier = f"{route.origin.proxy_request_id}-{len(route.calls)}"
        if route.origin.wire_api == WIRE_CHAT_COMPLETIONS:
            return {
                "id": f"chatcmpl-{identifier}",
                "object": "chat.completion",
                "model": route.revision.revision_id,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if finish_reason == "length_cap" else "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": usage,
            }
        return {
            "id": f"resp-{identifier}",
            "object": "response",
            "model": route.revision.revision_id,
            "status": "incomplete" if finish_reason == "length_cap" else "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }


def _checkpoint_for(revision: PolicyRevision) -> ProviderCheckpoint:
    sampler_digest = str(revision.metadata.get("sampler_digest") or "")
    return ProviderCheckpoint(
        checkpoint_id=revision.checkpoint_id,
        provider_reference=revision.sampler_reference,
        step=revision.revision,
        digest=sampler_digest or "sha256:" + digest({"ref": revision.sampler_reference}),
        kind="sampler_weights",
    )


def _capture(call: InferenceCall, profile: RendererProfile) -> Mapping[str, Any]:
    """The token evidence the container echoes back, beside the wire object."""

    return {
        "schema_version": GATEWAY_SCHEMA_VERSION,
        "call_id": call.call_id,
        "proxy_request_id": call.proxy_request_id,
        "prompt_token_ids": list(call.prompt_token_ids),
        "generation_token_ids": list(call.generation_token_ids),
        "generation_logprobs": list(call.generation_logprobs),
        "sampled_mask": list(call.sampled_mask),
        "stop_token_ids": list(call.stop_token_ids),
        "finish_reason": call.finish_reason,
        "renderer_profile_fingerprint": profile.fingerprint,
        "renderer_profile_id": profile.profile_id,
        "behavior_fingerprint": call.behavior_fingerprint,
        "policy_revision": call.policy_revision,
        "wire_api": call.wire_api,
        "sampling_transport": call.sampling_transport,
        "token_capture_provenance": call.token_capture_provenance,
        "branch_id": call.branch_id,
        "parent_branch_id": call.parent_branch_id,
        "compaction": None
        if call.compaction is None
        else {
            "rule": call.compaction.rule,
            "divergence_index": call.compaction.divergence_index,
            "removed_message_indices": list(call.compaction.removed_message_indices),
            "authored_by_policy": call.compaction.authored_by_policy,
        },
    }


# ------------------------------------------------------------- http surface


class GatewayServer:
    """Loopback listener that turns the origin path back into an attempt id."""

    def __init__(
        self, gateway: SamplerGatewayService, *, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        self._gateway = gateway
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise GatewayError("gateway server is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "GatewayServer":
        owner = self._gateway

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
                status, body = _dispatch(owner, self)
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._gateway.set_origin_root(self.base_url)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "GatewayServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()


_STATUS_FOR: tuple[tuple[type[BaseException], int], ...] = (
    (UnknownOriginError, 404),
    (ClosedOriginError, 409),
    (RouteRebindError, 409),
    (PromptBudgetError, 413),
    (RendererBridgeError, 501),
    (RendererMismatchError, 409),
    (WireError, 400),
    (SamplerEvidenceError, 502),
    (EvidenceError, 502),
    (GatewayError, 500),
)


def _status_for(error: BaseException) -> int:
    for kind, status in _STATUS_FOR:
        if isinstance(error, kind):
            return status
    return 500


def _dispatch(
    gateway: SamplerGatewayService, handler: BaseHTTPRequestHandler
) -> tuple[int, Mapping[str, Any]]:
    try:
        attempt, wire_api = _parse_path(handler.path)
        size = int(handler.headers.get("content-length", "0") or 0)
        payload = json.loads(handler.rfile.read(size) or b"{}")
        credential = str(handler.headers.get("authorization", "") or "")
        if credential.lower().startswith("bearer "):
            credential = credential[7:].strip()
        body = gateway.handle(
            attempt, payload, credential=credential or None, wire_api=wire_api
        )
        return 200, body
    except Exception as error:  # noqa: BLE001 - every refusal is a typed status
        return _status_for(error), {
            "error": {"type": type(error).__name__, "message": str(error)}
        }


def _parse_path(path: str) -> tuple[str, str]:
    parts = [part for part in path.split("?", 1)[0].strip("/").split("/") if part]
    if len(parts) < 3 or ATTEMPT_PATH_SEGMENT not in parts:
        raise UnknownOriginError(f"path {path!r} carries no attempt identity")
    index = parts.index(ATTEMPT_PATH_SEGMENT)
    if index + 1 >= len(parts):
        raise UnknownOriginError(f"path {path!r} carries no attempt identity")
    attempt = parts[index + 1]
    suffix = "/".join(parts[index + 2 :])
    for wire_api, expected in WIRE_PATH_SUFFIX.items():
        if suffix == expected:
            return attempt, wire_api
    raise WireError(f"path {path!r} names no known wire surface")


__all__ = [
    "ATTEMPT_PATH_SEGMENT",
    "AttemptFacts",
    "AttemptFactsError",
    "BRIDGE_DECLINED_RULE",
    "BRIDGE_RECLOSE_RULE",
    "BudgetEvent",
    "COMPACT_MARKER",
    "COMPACT_RULE",
    "CONTAINER_REWRITE_RULE",
    "ClosedOriginError",
    "GATEWAY_SCHEMA_VERSION",
    "GatewayError",
    "GatewayRenderer",
    "GatewayServer",
    "HISTORY_DROP_RULE",
    "HistoryDivergenceError",
    "PROMPT_BUDGET_POLICIES",
    "PrimeChatRenderer",
    "PrimeResponsesRenderer",
    "PromptBudget",
    "PromptBudgetError",
    "RESPONSES_PROJECTION",
    "RenderedPrompt",
    "RendererBridgeError",
    "RendererMismatchError",
    "RouteRebindError",
    "SamplerBackend",
    "SamplerEvidenceError",
    "SamplerGatewayService",
    "TRANSPORT_MESSAGE_IN",
    "TRANSPORT_TOKENS_IN",
    "TRUNCATE_RULE",
    "UNBOUNDED_BUDGET",
    "UnknownOriginError",
    "WIRE_CHAT_COMPLETIONS",
    "WIRE_RESPONSES",
    "WireError",
    "WireRequest",
    "parse_wire_request",
    "project_responses_items",
    "removals_between",
    "row_identity",
]
