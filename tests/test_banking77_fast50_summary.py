"""Token accounting must not silently exhaust streamed receipt inputs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/e2e"))

from summarize_banking77_fast50 import estimate, usage_sum  # noqa: E402


def test_usage_sum_consumes_generator_once():
    rows = ({"calls": 1, "prompt_tokens": 10, "completion_tokens": 2} for _ in range(3))
    assert usage_sum(rows) == {"calls": 3, "prompt_tokens": 30, "completion_tokens": 6}


def test_estimate_keeps_cache_scenarios_separate():
    result = estimate({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, 1_000_000)
    assert abs(result["all_prefill_cached_usd"] - .882) < 1e-12
    assert abs(result["no_prefill_cached_usd"] - 1.026) < 1e-12
