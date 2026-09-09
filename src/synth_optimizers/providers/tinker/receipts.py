from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocols import ProviderUsage, UsageReceipt


def usage_from_metrics(
    request_id: str,
    metrics: Mapping[str, Any],
    *,
    algorithm_id: str,
    implementation_version: str,
) -> UsageReceipt:
    cost = metrics.get("cost_usd")
    usage = ProviderUsage(
        input_tokens=int(metrics.get("input_tokens", 0)),
        output_tokens=int(metrics.get("output_tokens", 0)),
        training_tokens=int(metrics.get("training_tokens", 0)),
        cost_usd=None if cost is None else float(cost),
        cost_missing=cost is None,
        counters={
            key: float(value)
            for key, value in metrics.items()
            if key not in {"input_tokens", "output_tokens", "training_tokens", "cost_usd"}
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        },
    )
    return UsageReceipt(
        request_id=request_id,
        provider="tinker",
        usage=usage,
        algorithm_id=algorithm_id,
        implementation_version=implementation_version,
    )
