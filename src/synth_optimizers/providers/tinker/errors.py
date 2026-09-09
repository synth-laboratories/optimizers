from __future__ import annotations

from ..protocols import ProviderError


RETRYABLE_MARKERS = (
    "timeout",
    "temporarily unavailable",
    "rate limit",
    "429",
    "503",
    "connection reset",
    "try again",
)


def classify_tinker_error(error: BaseException) -> ProviderError:
    if isinstance(error, ProviderError):
        return error
    message = str(error)
    lowered = message.lower()
    retryable = any(marker in lowered for marker in RETRYABLE_MARKERS)
    code = "tinker_retryable" if retryable else "tinker_fatal"
    return ProviderError(code, message, retryable=retryable)
