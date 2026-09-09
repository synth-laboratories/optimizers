from __future__ import annotations

from typing import Any

from ..protocols import ProviderCheckpoint, ProviderError, ProviderSession


def save_checkpoint(
    client: Any,
    session: ProviderSession,
    *,
    step: int,
    kind: str,
    request_id: str,
) -> ProviderCheckpoint:
    saver = getattr(client, "save_checkpoint", None)
    if not callable(saver):
        raise ProviderError("checkpoint_save_unsupported", "Tinker checkpoint save is unavailable")
    payload = saver(session.session_id, step=step, kind=kind, request_id=request_id)
    return ProviderCheckpoint(
        checkpoint_id=str(payload["checkpoint_id"]),
        provider_reference=str(payload["provider_reference"]),
        step=int(payload.get("step", step)),
        digest=str(payload["digest"]),
        kind=kind,
        resume_token=payload.get("resume_token"),
        model_id=str(payload.get("model_id") or session.model_id),
    )


def load_checkpoint(
    client: Any,
    checkpoint: ProviderCheckpoint,
    *,
    request_id: str,
) -> ProviderSession:
    loader = getattr(client, "load_checkpoint", None)
    if not callable(loader):
        raise ProviderError("checkpoint_load_unsupported", "Tinker checkpoint restore is unavailable")
    payload = loader(checkpoint, request_id=request_id)
    return ProviderSession(
        provider="tinker",
        session_id=str(payload["session_id"]),
        model_id=str(payload["model_id"]),
        request_id=request_id,
    )
