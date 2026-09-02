"""Scripted Tinker transport used by contract and executor tests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..protocols import (
    CAPABILITY_CISPO_SLIME_V1,
    CISPO_REQUIRED_CAPABILITIES,
    ForwardRequest,
    ProviderCheckpoint,
    ProviderError,
    ProviderSession,
    SampleRequest,
    TrainingStepRequest,
)
from .models import resolve_tinker_model
from .tokenize import fallback_tokenize


def _digest(material: str) -> str:
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class FakeTinkerProvider:
    model_id: str = "openai/gpt-oss-20b"
    offered_capabilities: set[str] = field(default_factory=lambda: set(CISPO_REQUIRED_CAPABILITIES))
    validate_cispo: bool = False
    sample_text: str | Callable[[SampleRequest], str] = "other"
    sample_logprob: float = -0.2
    current_logprob: float = -0.25
    train_loss: float = 1.0
    fail_once: str | None = None
    cancelled: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)
    sessions: dict[str, Any] = field(default_factory=dict)
    paid_requests: set[str] = field(default_factory=set)

    def capabilities(self, model_id: str) -> dict[str, Any]:
        return {
            "capabilities": sorted(self.offered_capabilities),
            "validated": {CAPABILITY_CISPO_SLIME_V1: self.validate_cispo},
            "model_id": resolve_tinker_model(model_id),
        }

    def create_lora_training_client(self, base_model: str, rank: int, seed: int) -> Any:
        session_id = f"session_{len(self.sessions) + 1}"
        handle = type("Handle", (), {"session_id": session_id, "rank": rank, "seed": seed})()
        self.sessions[session_id] = {"model_id": base_model, "rank": rank, "seed": seed, "step": 0}
        return handle

    def register_session(self, session_id: str, handle: Any) -> None:
        self.sessions.setdefault(session_id, {"handle": handle, "step": 0})

    def tokenize_chat(
        self, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        return fallback_tokenize(messages, add_generation_prompt=add_generation_prompt)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(32 + (int(token) % 95)) for token in token_ids)

    def sample(self, handle: Any, request: SampleRequest) -> dict[str, Any]:
        self._record("sample", request.request_id)
        text = self.sample_text(request) if callable(self.sample_text) else str(self.sample_text)
        tokens = tuple(ord(ch) % 97 for ch in text) or (1,)
        return {
            "token_ids": tokens,
            "logprobs": (self.sample_logprob,) * len(tokens),
            "text": text,
            "finish_reason": "stop",
            "usage": {"input_tokens": len(request.prompt_token_ids), "output_tokens": len(tokens)},
        }

    def forward(self, session: ProviderSession, request: ForwardRequest) -> dict[str, Any]:
        self._record("forward", request.request_id)
        rows = []
        for tokens, mask in zip(request.token_ids, request.response_masks, strict=True):
            rows.append(tuple(self.current_logprob if flag else 0.0 for flag in mask[: len(tokens)]))
        return {
            "logprobs": rows,
            "usage": {"training_tokens": sum(sum(1 for flag in mask if flag) for mask in request.response_masks)},
        }

    def train_step(self, session: ProviderSession, request: TrainingStepRequest) -> dict[str, Any]:
        self._record("train", request.request_id)
        if request.loss_name == "importance_sampling":
            raise ProviderError("unsupported", "generic importance sampling is not cispo.slime.v1")
        state = self.sessions.setdefault(session.session_id, {"step": 0})
        state["step"] = int(state.get("step", 0)) + 1
        self.train_loss = max(0.05, self.train_loss * 0.7)
        return {
            "step": state["step"],
            "metrics": {"loss": self.train_loss, "mean_tokens": float(len(request.data))},
            "usage": {"training_tokens": max(1, len(request.data))},
        }

    def save_checkpoint(
        self, session_id: str, *, step: int, kind: str, request_id: str
    ) -> dict[str, Any]:
        self._record("checkpoint", request_id)
        material = f"{session_id}:{step}:{kind}"
        return {
            "checkpoint_id": f"ckpt_{step}_{kind}",
            "provider_reference": f"tinker://{session_id}/{kind}/{step}",
            "step": step,
            "digest": _digest(material),
            "resume_token": f"resume:{session_id}:{step}",
        }

    def load_checkpoint(self, checkpoint: ProviderCheckpoint, *, request_id: str) -> dict[str, Any]:
        self._record("restore", request_id)
        session_id = checkpoint.resume_token or checkpoint.checkpoint_id
        self.sessions.setdefault(session_id, {"step": checkpoint.step, "model_id": self.model_id})
        return {"session_id": session_id, "model_id": self.model_id}

    def cancel(self, session_id: str) -> None:
        self.cancelled.add(session_id)

    def _record(self, kind: str, request_id: str) -> None:
        if request_id in self.paid_requests:
            self.calls.append((kind, request_id))
            return
        if self.fail_once == kind:
            self.fail_once = None
            self.calls.append((kind, request_id))
            raise ProviderError("timeout", "scripted timeout", retryable=True)
        self.paid_requests.add(request_id)
        self.calls.append((kind, request_id))
