from __future__ import annotations

from typing import Any

from .models import CANONICAL_GPT_OSS_20B, resolve_tinker_model
from ..protocols import (
    CAPABILITY_CHECKPOINT_SAMPLE,
    CAPABILITY_CISPO_SLIME_V1,
    CAPABILITY_IMPORTANCE_WEIGHTS,
    CAPABILITY_ROLLOUT_GROUPED,
    CAPABILITY_SFT_TRAIN,
    CAPABILITY_TRAJECTORY_LOGPROBS,
    ProviderCapabilities,
)


KNOWN_CAPABILITIES = frozenset(
    {
        CAPABILITY_SFT_TRAIN,
        CAPABILITY_CHECKPOINT_SAMPLE,
        CAPABILITY_ROLLOUT_GROUPED,
        CAPABILITY_TRAJECTORY_LOGPROBS,
        CAPABILITY_IMPORTANCE_WEIGHTS,
        CAPABILITY_CISPO_SLIME_V1,
    }
)


def discover_tinker_capabilities(client: Any, model_id: str) -> ProviderCapabilities:
    resolved = resolve_tinker_model(model_id)
    advertised = getattr(client, "capabilities", None)
    names: set[str] = set()
    validated: dict[str, bool] = {}
    if callable(advertised):
        payload = advertised(resolved) or {}
        raw_names = payload.get("capabilities", ())
        names = {str(name) for name in raw_names}
        raw_validated = payload.get("validated", {})
        if isinstance(raw_validated, dict):
            validated = {str(key): bool(value) for key, value in raw_validated.items()}
    elif advertised is None:
        names = set(KNOWN_CAPABILITIES)
        validated = {CAPABILITY_CISPO_SLIME_V1: False}
    return ProviderCapabilities(
        provider="tinker",
        model_id=resolved or CANONICAL_GPT_OSS_20B,
        capabilities=frozenset(names & KNOWN_CAPABILITIES),
        validated=validated,
        maximums={"sequence_cap": 131072, "batch_size": 64, "rank": 4096},
        spend_free=True,
    )
