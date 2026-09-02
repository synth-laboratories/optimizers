from __future__ import annotations

from ..protocols import ProviderError

CANONICAL_GPT_OSS_20B = "openai/gpt-oss-20b"
ALIASES = {
    "gpt-oss-20b": CANONICAL_GPT_OSS_20B,
    "openai/gpt-oss-20b": CANONICAL_GPT_OSS_20B,
}


def resolve_tinker_model(model_id: str) -> str:
    resolved = ALIASES.get(str(model_id).strip(), str(model_id).strip())
    if not resolved:
        raise ProviderError("model_id_required", "a Tinker model id is required")
    return resolved
