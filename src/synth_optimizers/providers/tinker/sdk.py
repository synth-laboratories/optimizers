"""Live Tinker SDK transport. FakeTinkerProvider stays the unpaid test double."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..protocols import (
    CAPABILITY_CISPO_SLIME_V1,
    ForwardRequest,
    ProviderCheckpoint,
    ProviderError,
    ProviderSession,
    SampleRequest,
    TrainingStepRequest,
)
from .capabilities import KNOWN_CAPABILITIES
from .models import resolve_tinker_model
from .prime import create_prime_renderer, parse_completion
from .tokenize import extract_final_label, tokenize_live
from .validation import default_receipt_path, is_cispo_validated


def apply_macos_tls() -> None:
    """Use system certificates when the SDK's bundled roots omit them."""

    try:
        import pyqwest
        import tinker._base_client as base_client
        from pyqwest.httpx import AsyncPyqwestTransport

        base_client._default_pyqwest_transport = lambda: AsyncPyqwestTransport(
            transport=pyqwest.HTTPTransport(tls_include_system_certs=True)
        )
    except (ImportError, AttributeError, TypeError):
        return


class TinkerSdkTransport:
    """Presents the FakeTinkerProvider method surface over ``tinker.ServiceClient``."""

    def __init__(
        self,
        service: Any,
        *,
        tinker_module: Any,
        validation_receipt: Any | None = None,
    ) -> None:
        self._service = service
        self._tinker = tinker_module
        self.validation_receipt = validation_receipt
        self.sessions: dict[str, dict[str, Any]] = {}
        self._samplers: dict[str, Any] = {}
        self._tokenizer: Any | None = None
        self._renderer: Any | None = None
        self.cancelled: set[str] = set()

    @classmethod
    def connect(cls, api_key: str, *, base_url: str | None = None) -> "TinkerSdkTransport":
        apply_macos_tls()
        try:
            import tinker
        except ImportError as exc:
            raise ProviderError("tinker_sdk_missing", "the tinker package is not installed") from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        service = tinker.ServiceClient(
            user_metadata={"project": "synth-optimizers", "task": "sft-cispo"},
            **kwargs,
        )
        return cls(service, tinker_module=tinker, validation_receipt=default_receipt_path())

    def capabilities(self, model_id: str) -> dict[str, Any]:
        resolved = resolve_tinker_model(model_id)
        return {
            "capabilities": sorted(KNOWN_CAPABILITIES),
            "validated": {
                CAPABILITY_CISPO_SLIME_V1: is_cispo_validated(self.validation_receipt, resolved)
            },
            "model_id": resolved,
        }

    def create_lora_training_client(self, base_model: str, rank: int, seed: int) -> Any:
        trainer = self._service.create_lora_training_client(
            base_model=base_model, rank=rank, seed=seed
        )
        session_id = str(getattr(trainer, "model_id", f"tinker-{len(self.sessions) + 1}"))
        self.sessions[session_id] = {
            "training": trainer,
            "model_id": base_model,
            "rank": rank,
            "seed": seed,
            "step": 0,
        }
        self._bind_tokenizer(trainer, base_model)
        return type("Handle", (), {"session_id": session_id})()

    def register_session(self, session_id: str, handle: Any) -> None:
        self.sessions.setdefault(session_id, {"handle": handle, "step": 0})

    def tokenize_chat(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        return tokenize_live(
            self._renderer,
            self._tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        if self._tokenizer is None:
            return "".join(chr(32 + (int(token) % 95)) for token in token_ids)
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=False))

    def sample(self, handle: Any, request: SampleRequest) -> dict[str, Any]:
        sampler = self._sampler_for(handle)
        stop_ids = list(self._renderer.get_stop_token_ids()) if self._renderer is not None else None
        result = sampler.sample(
            prompt=self._tinker.ModelInput.from_ints(list(request.prompt_token_ids)),
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                seed=request.seed,
                stop=stop_ids or None,
            ),
        ).result()
        sequence = result.sequences[0]
        tokens = [int(token) for token in sequence.tokens]
        logprobs = [float(value) for value in (sequence.logprobs or [0.0] * len(tokens))]
        raw = self.decode(tokens)
        parsed = parse_completion(self._renderer, tokens) if self._renderer is not None else ""
        text = extract_final_label(parsed or raw) or raw
        return {
            "token_ids": tokens,
            "logprobs": logprobs,
            "text": text,
            "finish_reason": str(getattr(sequence, "stop_reason", "stop")),
            "usage": {
                "input_tokens": len(request.prompt_token_ids),
                "output_tokens": len(tokens),
            },
        }

    def forward(self, session: ProviderSession, request: ForwardRequest) -> dict[str, Any]:
        trainer = self._trainer(session.session_id)
        data = [
            self._ce_datum(tokens, mask)
            for tokens, mask in zip(request.token_ids, request.response_masks, strict=True)
        ]
        output = trainer.forward(data, loss_fn="cross_entropy").result()
        rows = tuple(_logprob_row(item, tokens) for item, tokens in zip(output.loss_fn_outputs, request.token_ids))
        return {
            "logprobs": rows,
            "usage": {"training_tokens": sum(sum(1 for flag in mask if flag) for mask in request.response_masks)},
        }

    def train_step(self, session: ProviderSession, request: TrainingStepRequest) -> dict[str, Any]:
        if request.loss_name == "importance_sampling":
            raise ProviderError("unsupported", "generic importance sampling is not cispo.slime.v1")
        trainer = self._trainer(session.session_id)
        loss_fn, config = _tinker_loss(request)
        data = [_train_datum(self._tinker, item, loss_fn) for item in request.data]
        output = trainer.forward_backward(data, loss_fn=loss_fn, loss_fn_config=config).result()
        learning_rate = float((request.metadata or {}).get("learning_rate") or 2e-5)
        trainer.optim_step(self._tinker.AdamParams(learning_rate=learning_rate)).result()
        state = self.sessions.setdefault(session.session_id, {"step": 0})
        state["step"] = int(state.get("step", 0)) + 1
        self._samplers.pop(session.session_id, None)
        metrics = {str(key): float(value) for key, value in dict(getattr(output, "metrics", {}) or {}).items()}
        return {
            "step": state["step"],
            "metrics": metrics or {"loss": 0.0},
            "usage": {"training_tokens": sum(_token_count(item) for item in request.data)},
        }

    def save_checkpoint(
        self, session_id: str, *, step: int, kind: str, request_id: str
    ) -> dict[str, Any]:
        trainer = self._trainer(session_id)
        name = tinker_checkpoint_name(kind, request_id)
        if kind == "training":
            path = str(trainer.save_state(name, ttl_seconds=30 * 86400).result().path)
        else:
            path = str(trainer.save_weights_for_sampler(name, ttl_seconds=30 * 86400).result().path)
            self._samplers[session_id] = self._service.create_sampling_client(model_path=path)
        digest = "sha256:" + hashlib.sha256(path.encode("utf-8")).hexdigest()
        return {
            "checkpoint_id": f"{kind}-{step}-{digest[-12:]}",
            "provider_reference": path,
            "step": step,
            "digest": digest,
            "resume_token": path,
        }

    def load_checkpoint(self, checkpoint: ProviderCheckpoint, *, request_id: str) -> dict[str, Any]:
        path = checkpoint.resume_token or checkpoint.provider_reference
        trainer = self._service.create_training_client_from_state(path)
        session_id = str(getattr(trainer, "model_id", request_id))
        self.sessions[session_id] = {
            "training": trainer,
            "model_id": checkpoint.kind,
            "step": checkpoint.step,
        }
        self._bind_tokenizer(trainer, str(getattr(trainer, "model_id", checkpoint.kind)))
        return {"session_id": session_id, "model_id": str(self.sessions[session_id].get("model_id") or "")}

    def cancel(self, session_id: str) -> None:
        self.cancelled.add(session_id)

    def _bind_tokenizer(self, trainer: Any, model_id: str) -> None:
        self._tokenizer = trainer.get_tokenizer()
        self._renderer = create_prime_renderer(self._tokenizer, model_id=model_id)

    def _trainer(self, session_id: str) -> Any:
        state = self.sessions.get(session_id)
        if not state or "training" not in state:
            raise ProviderError("tinker_session_missing", f"no training client for {session_id}")
        return state["training"]

    def _sampler_for(self, handle: Any) -> Any:
        if isinstance(handle, ProviderCheckpoint):
            return self._service.create_sampling_client(model_path=handle.provider_reference)
        session_id = getattr(handle, "session_id", None) or (
            handle.session_id if isinstance(handle, ProviderSession) else None
        )
        if session_id in self._samplers:
            return self._samplers[session_id]
        if session_id is None:
            raise ProviderError("sample_unsupported", "sampling handle is missing a session")
        state = self.sessions.get(session_id) or {}
        step = int(state.get("step", 0))
        # Tinker requires every persisted sampler name to be unique. Training
        # invalidates the cached sampler after each optimizer step, so a
        # constant ``<session>-live`` name collides as soon as the next
        # on-policy group asks for fresh weights. Scope the implicit live
        # sampler to the model step while keeping retries idempotent.
        saved = self.save_checkpoint(
            session_id,
            step=step,
            kind="inference",
            request_id=f"{session_id}-live-{step}",
        )
        sampler = self._service.create_sampling_client(model_path=saved["provider_reference"])
        self._samplers[session_id] = sampler
        return sampler

    def _ce_datum(self, tokens: Sequence[int], mask: Sequence[bool]) -> Any:
        ids = list(tokens)
        if len(ids) < 2:
            ids = ids + [0]
        weights = [1.0 if flag else 0.0 for flag in list(mask)[: len(ids) - 1]]
        weights.extend([0.0] * max(0, len(ids) - 1 - len(weights)))
        return self._tinker.Datum(
            model_input=self._tinker.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={
                "target_tokens": self._tinker.TensorData(
                    data=ids[1:], dtype="int64", shape=[len(ids) - 1]
                ),
                "weights": self._tinker.TensorData(
                    data=weights, dtype="float32", shape=[len(weights)]
                ),
            },
        )


_TINKER_CHECKPOINT_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def tinker_checkpoint_name(kind: str, request_id: str) -> str:
    """Tinker rejects colons and other punctuation in `save_weights_for_sampler` names."""

    raw = f"optimizers-{kind}-{request_id}"
    cleaned = _TINKER_CHECKPOINT_NAME.sub("-", raw).strip("-._") or "optimizers-ckpt"
    if not (cleaned[0].isalnum() or cleaned[0] == "_"):
        cleaned = f"ckpt-{cleaned}"
    return cleaned[:180]


def _tinker_loss(request: TrainingStepRequest) -> tuple[str, dict[str, float] | None]:
    if request.loss_name == "cross_entropy":
        return "cross_entropy", None
    if request.loss_name == "cispo.slime.v1":
        metadata = request.metadata or {}
        eps_clip = float(metadata.get("eps_clip") or 1.0)
        eps_clip_high = float(metadata.get("eps_clip_high") or 4.0)
        return "cispo", {
            "clip_low_threshold": max(0.0, 1.0 - eps_clip),
            "clip_high_threshold": 1.0 + eps_clip_high,
        }
    raise ProviderError("unsupported", f"loss {request.loss_name} is not cispo.slime.v1 or cross_entropy")


def _train_datum(tinker_module: Any, item: Mapping[str, Any], loss_fn: str) -> Any:
    if loss_fn == "cross_entropy":
        ids = list(item.get("input_ids") or item.get("token_ids") or ())
        targets = list(item.get("target_tokens") or ids[1:])
        weights = list(item.get("weights") or [1.0] * len(targets))
        model_input = list(item.get("input_ids") or ids[:-1] or ids)
        return tinker_module.Datum(
            model_input=tinker_module.ModelInput.from_ints(list(model_input)),
            loss_fn_inputs={
                "target_tokens": tinker_module.TensorData(
                    data=targets, dtype="int64", shape=[len(targets)]
                ),
                "weights": tinker_module.TensorData(
                    data=weights, dtype="float32", shape=[len(weights)]
                ),
            },
        )
    full_sequence = bool(item.get("loss_mask"))
    prompt = list(item.get("prompt_token_ids") or ())
    completion = list(item.get("token_ids") or ())
    ids = completion if full_sequence else (prompt + completion if prompt else completion)
    if len(ids) < 2:
        raise ProviderError("cispo_tokens_missing", "CISPO datum needs at least two tokens")
    prompt_len = len(prompt) if prompt else 0
    supplied_mask = list(item.get("loss_mask") or ())
    shifted = (
        [bool(value) for value in supplied_mask[1:len(ids)]]
        if full_sequence else [index >= prompt_len for index in range(1, len(ids))]
    )
    if len(shifted) != len(ids) - 1:
        raise ProviderError("cispo_mask_alignment", "CISPO loss mask must align with full token sequence")
    trained = sum(shifted)
    behavior = list(item.get("behavior_logprobs") or ())
    advantage = item.get("advantages")
    if isinstance(advantage, Sequence) and not isinstance(advantage, (str, bytes)):
        scalar = float(advantage[0]) if advantage else 0.0
    else:
        scalar = float(advantage or 0.0)
    logprobs, advantages = [], []
    if full_sequence and len(behavior) != len(ids):
        raise ProviderError("cispo_logprob_alignment", "full-sequence behavior logprobs must align with tokens")
    completion_logprobs = iter(behavior)
    for position, enabled in enumerate(shifted, start=1):
        logprobs.append(
            float(behavior[position]) if full_sequence and enabled
            else (next(completion_logprobs, 0.0) if enabled else 0.0)
        )
        advantages.append(scalar / trained if enabled and trained else 0.0)
    return tinker_module.Datum(
        model_input=tinker_module.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker_module.TensorData(
                data=ids[1:], dtype="int64", shape=[len(ids) - 1]
            ),
            "logprobs": tinker_module.TensorData(
                data=logprobs, dtype="float32", shape=[len(logprobs)]
            ),
            "advantages": tinker_module.TensorData(
                data=advantages, dtype="float32", shape=[len(advantages)]
            ),
        },
    )


def _token_count(item: Mapping[str, Any]) -> int:
    tokens = item.get("target_tokens") or item.get("token_ids") or item.get("input_ids") or ()
    return max(1, len(tokens))


def _logprob_row(output: Any, tokens: Sequence[int]) -> tuple[float, ...]:
    payload = output
    if isinstance(output, Mapping):
        payload = output.get("logprobs") or output.get("element_logprobs") or output
    data = getattr(payload, "data", payload)
    if isinstance(data, Mapping):
        data = data.get("data") or []
    values = [float(value) for value in (data or [])]
    if not values:
        values = [0.0] * max(1, len(tokens))
    return tuple(values)
