"""Tinker credentials and the shared adapter used by SFT and CISPO."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .capabilities import discover_tinker_capabilities
from .checkpoints import load_checkpoint, save_checkpoint
from .errors import classify_tinker_error
from .models import resolve_tinker_model
from .receipts import usage_from_metrics
from .sampling import sample_tokens
from .tokenize import fallback_tokenize
from .training import forward_logprobs, run_training_step
from ..protocols import (
    CISPO_REQUIRED_CAPABILITIES,
    ForwardRequest,
    ForwardResult,
    ProviderCapabilities,
    ProviderCheckpoint,
    ProviderError,
    ProviderSession,
    SampleRequest,
    SampleResult,
    TrainingStepRequest,
    TrainingStepResult,
    UnsupportedCapability,
)


@dataclass(frozen=True, slots=True)
class TinkerCredentials:
    api_key: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "TinkerCredentials":
        api_key = os.environ.get("TINKER_API_KEY", "").strip()
        if not api_key:
            raise ProviderError("tinker_credentials_missing", "TINKER_API_KEY is required")
        base_url = os.environ.get("TINKER_BASE_URL", "").strip() or None
        return cls(api_key=api_key, base_url=base_url)


class TinkerAdapter:
    """One Tinker client for SFT, CISPO, and future weight-update lanes."""

    def __init__(
        self,
        credentials: TinkerCredentials,
        *,
        transport: Any | None = None,
        user_metadata: Mapping[str, str] | None = None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.credentials = credentials
        self._transport = transport
        # Preserve the adapter's historical attribution for existing direct
        # SFT/CISPO callers. RL assembly supplies its own run-scoped metadata.
        self.user_metadata = dict(
            user_metadata
            if user_metadata is not None
            else {"project": "synth-optimizers", "task": "sft-cispo"}
        )
        self.max_attempts = max(1, max_attempts)
        self._sleep = sleep
        self._sessions: dict[str, Any] = {}
        self._completed_requests: dict[str, Any] = {}
        self._request_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._cancelled: set[str] = set()

    def discover_capabilities(self, model_id: str) -> ProviderCapabilities:
        resolved = self.resolve_model(model_id)
        return discover_tinker_capabilities(self._client(), resolved)

    def resolve_model(self, model_id: str) -> str:
        return resolve_tinker_model(model_id)

    def prepare_renderer(self, model_id: str) -> None:
        prepare = getattr(self._client(), 'prepare_renderer', None)
        if callable(prepare):
            prepare(self.resolve_model(model_id))

    def renderer_profile(self, model_id: str) -> dict[str, Any]:
        return self._client().renderer_profile(self.resolve_model(model_id))

    def require_cispo(self, model_id: str) -> ProviderCapabilities:
        capabilities = self.discover_capabilities(model_id)
        try:
            capabilities.require(CISPO_REQUIRED_CAPABILITIES)
        except UnsupportedCapability:
            raise ProviderError(
                "unsupported",
                "CISPO requires cispo.slime.v1 and the grouped-rollout/logprob capabilities",
            ) from None
        if capabilities.validated.get("cispo.slime.v1") is not True:
            raise ProviderError(
                "unsupported",
                "cispo.slime.v1 is not validated on this provider",
            )
        return capabilities

    def create_session(
        self, model_id: str, *, rank: int, seed: int, request_id: str
    ) -> ProviderSession:
        self._claim(request_id, "create_session", model_id, rank, seed)
        cached = self._completed_requests.get(request_id)
        if isinstance(cached, ProviderSession):
            return cached
        resolved = self.resolve_model(model_id)
        session = self._retry(
            request_id,
            lambda: _create_training_session(self._client(), resolved, rank, seed, request_id),
        )
        self._sessions[session.session_id] = session
        self._completed_requests[request_id] = session
        return session

    def restore_session(
        self, checkpoint: ProviderCheckpoint, *, request_id: str
    ) -> ProviderSession:
        if checkpoint.kind not in {"training", "training_state"} or not checkpoint.resume_token:
            raise ProviderError(
                "checkpoint_not_resumable",
                f"checkpoint kind {checkpoint.kind!r} is not resumable training state",
            )
        self._claim(
            request_id,
            "restore_session",
            checkpoint.checkpoint_id,
            checkpoint.resume_token,
            checkpoint.digest,
        )
        cached = self._completed_requests.get(request_id)
        if isinstance(cached, ProviderSession):
            return cached
        session = self._retry(
            request_id,
            lambda: load_checkpoint(self._client(), checkpoint, request_id=request_id),
        )
        self._sessions[session.session_id] = session
        self._completed_requests[request_id] = session
        return session

    def sample(self, session: ProviderSession, request: SampleRequest) -> SampleResult:
        self._ensure_active(session)
        self._claim(request.request_id, "sample", session.session_id, request)
        cached = self._completed_requests.get(request.request_id)
        if isinstance(cached, SampleResult):
            return cached
        result = self._retry(
            request.request_id,
            lambda: sample_tokens(self._client(), session, request),
        )
        self._completed_requests[request.request_id] = result
        return result

    def forward(self, session: ProviderSession, request: ForwardRequest) -> ForwardResult:
        self._ensure_active(session)
        self._claim(request.request_id, "forward", session.session_id, request)
        cached = self._completed_requests.get(request.request_id)
        if isinstance(cached, ForwardResult):
            return cached
        result = self._retry(
            request.request_id,
            lambda: forward_logprobs(self._client(), session, request),
        )
        self._completed_requests[request.request_id] = result
        return result

    def train_step(
        self, session: ProviderSession, request: TrainingStepRequest
    ) -> TrainingStepResult:
        self._ensure_active(session)
        self._claim(request.request_id, "train_step", session.session_id, request)
        cached = self._completed_requests.get(request.request_id)
        if isinstance(cached, TrainingStepResult):
            return cached
        result = self._retry(
            request.request_id,
            lambda: run_training_step(self._client(), session, request),
        )
        self._completed_requests[request.request_id] = result
        return result

    def save_checkpoint(
        self, session: ProviderSession, *, step: int, kind: str, request_id: str
    ) -> ProviderCheckpoint:
        self._ensure_active(session)
        self._claim(request_id, "save_checkpoint", session.session_id, step, kind)
        cached = self._completed_requests.get(request_id)
        if isinstance(cached, ProviderCheckpoint):
            return cached
        checkpoint = self._retry(
            request_id,
            lambda: save_checkpoint(self._client(), session, step=step, kind=kind, request_id=request_id),
        )
        self._completed_requests[request_id] = checkpoint
        return checkpoint

    def sample_checkpoint(
        self, checkpoint: ProviderCheckpoint, request: SampleRequest
    ) -> SampleResult:
        self._claim(
            request.request_id,
            "sample_checkpoint",
            checkpoint.checkpoint_id,
            checkpoint.provider_reference,
            checkpoint.digest,
            request,
        )
        cached = self._completed_requests.get(request.request_id)
        if isinstance(cached, SampleResult):
            return cached
        result = self._retry(
            request.request_id,
            lambda: sample_tokens(self._client(), checkpoint, request),
        )
        self._completed_requests[request.request_id] = result
        return result

    def cancel(self, session: ProviderSession) -> None:
        self._cancelled.add(session.session_id)
        cancel = getattr(self._client(), "cancel", None)
        if callable(cancel):
            cancel(session.session_id)

    def tokenize_chat(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        tokenizer = getattr(self._client(), "tokenize_chat", None)
        if callable(tokenizer):
            return tokenizer(messages, add_generation_prompt=add_generation_prompt)
        return fallback_tokenize(messages, add_generation_prompt=add_generation_prompt)

    def bridge_chat(
        self,
        previous_prompt_token_ids: Sequence[int],
        previous_completion_token_ids: Sequence[int],
        messages: Sequence[Mapping[str, str]],
    ) -> dict[str, Any] | None:
        """The renderer's own turn-to-turn bridge, when this client has one.

        ``None`` means no extension was proven, not that one was refused: the
        caller forks a branch and records why rather than splicing on faith.
        """

        bridge = getattr(self._client(), "bridge_chat", None)
        if not callable(bridge):
            return None
        return bridge(previous_prompt_token_ids, previous_completion_token_ids, messages)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        decoder = getattr(self._client(), "decode", None)
        if callable(decoder):
            return str(decoder(token_ids))
        return "".join(chr(32 + (int(token) % 95)) for token in token_ids)

    def describe_artifact(self, reference: str) -> Mapping[str, Any]:
        inspector = getattr(self._client(), 'describe_artifact', None)
        if not callable(inspector):
            raise ProviderError('artifact_inspection_unsupported', 'provider transport has no artifact inspection')
        return inspector(reference)

    def classify_error(self, error: BaseException) -> ProviderError:
        return classify_tinker_error(error)

    def _claim(self, request_id: str, *fingerprint: Any) -> None:
        """Bind an idempotency key to exactly one operation and resource."""

        claimed = tuple(fingerprint)
        previous = self._request_fingerprints.get(request_id)
        if previous is not None and previous != claimed:
            raise ProviderError(
                "idempotency_conflict",
                f"request id {request_id!r} was reused for a different Tinker operation",
            )
        self._request_fingerprints[request_id] = claimed

    def receipt_from_usage(
        self,
        request_id: str,
        usage: Mapping[str, Any],
        *,
        algorithm_id: str,
        implementation_version: str,
    ) -> Any:
        return usage_from_metrics(
            request_id,
            usage,
            algorithm_id=algorithm_id,
            implementation_version=implementation_version,
        )

    def _client(self) -> Any:
        if self._transport is not None:
            return self._transport
        from .sdk import TinkerSdkTransport

        self._transport = TinkerSdkTransport.connect(
            self.credentials.api_key,
            base_url=self.credentials.base_url,
            user_metadata=self.user_metadata,
        )
        return self._transport

    def _ensure_active(self, session: ProviderSession) -> None:
        if session.session_id in self._cancelled:
            raise ProviderError("cancelled", "training session was cancelled")

    def _retry(self, request_id: str, operation: Callable[[], Any]) -> Any:
        last_error: ProviderError | None = None
        for attempt in range(self.max_attempts):
            try:
                return operation()
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt + 1 >= self.max_attempts:
                    raise
                self._sleep(min(2**attempt, 8))
            except Exception as exc:
                mapped = classify_tinker_error(exc)
                mapped.request_id = request_id
                last_error = mapped
                if not mapped.retryable or attempt + 1 >= self.max_attempts:
                    raise mapped from exc
                self._sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


def new_request_id(*parts: str) -> str:
    material = "|".join(parts) if parts else uuid.uuid4().hex
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"tinkerreq_{digest}"


def _create_training_session(
    client: Any, model_id: str, rank: int, seed: int, request_id: str
) -> ProviderSession:
    create = getattr(client, "create_lora_training_client", None)
    if not callable(create):
        raise ProviderError("tinker_session_unsupported", "training client factory is missing")
    handle = create(base_model=model_id, rank=rank, seed=seed)
    session_id = str(getattr(handle, "session_id", request_id))
    if hasattr(client, "register_session"):
        client.register_session(session_id, handle)
    return ProviderSession(
        provider="tinker",
        session_id=session_id,
        model_id=model_id,
        request_id=request_id,
    )
