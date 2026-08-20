from __future__ import annotations

from typing import Any, Mapping

# Official / published USD per 1M tokens. A cost_usd of 0.0 with nonzero tokens
# is unpriced, not free — apply this catalog instead of treating the zero as spend.

_RATES: dict[str, tuple[float, float, float]] = {
    # model: (input, cached_input, output)
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    "x-ai/grok-4.3": (1.25, 0.20, 2.50),
    "grok-4.3": (1.25, 0.20, 2.50),
    "nvidia/nemotron-3.5-lightning": (0.08, 0.04, 0.20),
    "nemotron-3.5-lightning": (0.08, 0.04, 0.20),
    "deepseek-chat": (0.27, 0.27, 1.10),
    "deepseek-reasoner": (0.27, 0.27, 1.10),
}


def normalize_model_id(model: str | None) -> str:
    text = str(model or "").strip()
    for prefix in ("openai/", "openrouter/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
    return text.strip().lower()


def rates_for(model: str | None, provider: str | None = None) -> tuple[float, float, float] | None:
    provider_l = str(provider or "").strip().lower()
    model_id = normalize_model_id(model)
    if provider_l == "deepseek" or "deepseek" in model_id:
        return (0.27, 0.27, 1.10)
    if model_id in _RATES:
        return _RATES[model_id]
    if "/" in model_id:
        suffix = model_id.rsplit("/", 1)[-1]
        if suffix in _RATES:
            return _RATES[suffix]
    return None


def _as_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return None
    return value


def token_counts(blob: Mapping[str, Any] | None) -> tuple[int, int, int]:
    if not isinstance(blob, Mapping):
        return (0, 0, 0)
    prompt = 0
    completion = 0
    cached = 0
    for key in ("prompt_tokens", "input_tokens"):
        value = _as_float(blob.get(key))
        if value is not None:
            prompt = max(prompt, int(value))
    for key in ("completion_tokens", "output_tokens"):
        value = _as_float(blob.get(key))
        if value is not None:
            completion = max(completion, int(value))
    for key in ("cached_prompt_tokens", "cached_tokens"):
        value = _as_float(blob.get(key))
        if value is not None:
            cached = max(cached, int(value))
    for nested_key in ("prompt_tokens_details", "input_tokens_details"):
        details = blob.get(nested_key)
        if isinstance(details, Mapping):
            value = _as_float(details.get("cached_tokens"))
            if value is not None:
                cached = max(cached, int(value))
    if prompt == 0 and completion == 0:
        total = _as_float(blob.get("total_tokens"))
        if total is not None:
            prompt = int(total)
    return (prompt, completion, min(cached, prompt))


def billed_cost_usd(blob: Mapping[str, Any] | None) -> float | None:
    """Return a real billed USD amount, or None when the field is unpriced."""
    if not isinstance(blob, Mapping):
        return None
    prompt, completion, _cached = token_counts(blob)
    for key in ("cost_usd", "spend_usd", "cost"):
        value = _as_float(blob.get(key))
        if value is None:
            continue
        if value > 0.0:
            return value
        if value == 0.0 and prompt == 0 and completion == 0:
            return 0.0
    return None


def price_usage_usd(
    blob: Mapping[str, Any] | None,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[float | None, str]:
    billed = billed_cost_usd(blob)
    if billed is not None:
        source = "provider_billed" if billed > 0.0 else "no_tokens"
        return billed, source
    prompt, completion, cached = token_counts(blob)
    if prompt == 0 and completion == 0:
        return 0.0, "no_tokens"
    model_id = normalize_model_id(model) or normalize_model_id(
        blob.get("model") if isinstance(blob, Mapping) else None
    )
    provider_id = provider or (
        str(blob.get("provider") or "") if isinstance(blob, Mapping) else ""
    )
    rates = rates_for(model_id, provider_id)
    if rates is None:
        return None, "unpriced"
    input_rate, cached_rate, output_rate = rates
    billable_prompt = max(0, prompt - cached)
    cost = (
        billable_prompt * input_rate / 1_000_000.0
        + cached * cached_rate / 1_000_000.0
        + completion * output_rate / 1_000_000.0
    )
    return cost, f"static_price:{model_id or 'unknown'}"


def _context_models(context: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    ctx = dict(context or {})
    arm = ctx.get("arm") if isinstance(ctx.get("arm"), dict) else {}
    downstream = ctx.get("downstream") if isinstance(ctx.get("downstream"), dict) else {}
    policy = downstream.get("policy") if isinstance(downstream.get("policy"), dict) else {}
    episode = ctx.get("episode") if isinstance(ctx.get("episode"), dict) else {}
    episode_policy = episode.get("policy") if isinstance(episode.get("policy"), dict) else {}
    proposer = (
        str(arm.get("model") or episode.get("proposer_model") or "").strip() or None
    )
    policy_model = (
        str(policy.get("model") or episode_policy.get("model") or "").strip() or None
    )
    return policy_model, proposer


def _row_model(row: Mapping[str, Any], policy_model: str | None, proposer_model: str | None) -> str | None:
    explicit = str(row.get("model") or "").strip()
    if explicit:
        return explicit
    boundary = str(row.get("boundary") or "").lower()
    if "proposer" in boundary:
        return proposer_model
    return policy_model


def _price_ledger(
    rows: list[Any],
    *,
    policy_model: str | None,
    proposer_model: str | None,
) -> tuple[float, int]:
    total = 0.0
    unpriced = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else row
        model = _row_model(row, policy_model, proposer_model)
        provider = str(row.get("provider") or "").strip() or None
        cost, source = price_usage_usd(usage, model=model, provider=provider)
        if cost is None:
            prompt, completion, _cached = token_counts(usage)
            if prompt or completion:
                unpriced += 1
            continue
        total += cost
    return total, unpriced


def episode_cost_usd(
    context: Mapping[str, Any] | None,
    *,
    missing: str = "zero",
) -> dict[str, Any]:
    """Priced spend for this episode: post − pre, never treating 0+tokens as free.

    Prefers usage_ledger rows (policy vs proposer models). Falls back to billed
    flattened usage, then to catalog-priced token totals when a single model is known.
    """
    ctx = dict(context or {})
    pre = ctx.get("pre_cursor") if isinstance(ctx.get("pre_cursor"), dict) else {}
    post = ctx.get("post_cursor") if isinstance(ctx.get("post_cursor"), dict) else {}
    finished = (
        ctx.get("optimizer_finished")
        if isinstance(ctx.get("optimizer_finished"), dict)
        else {}
    )
    policy_model, proposer_model = _context_models(ctx)
    pre_ledger = pre.get("usage_ledger") if isinstance(pre.get("usage_ledger"), list) else None
    post_ledger = post.get("usage_ledger") if isinstance(post.get("usage_ledger"), list) else None
    source = "usage_ledger"
    unpriced = 0
    if isinstance(pre_ledger, list) or isinstance(post_ledger, list):
        post_total, post_unpriced = _price_ledger(
            list(post_ledger or []),
            policy_model=policy_model,
            proposer_model=proposer_model,
        )
        pre_total, pre_unpriced = _price_ledger(
            list(pre_ledger or []),
            policy_model=policy_model,
            proposer_model=proposer_model,
        )
        unpriced = post_unpriced + pre_unpriced
        pre_ids = {
            str(row.get("usage_ledger_id"))
            for row in (pre_ledger or [])
            if isinstance(row, dict) and row.get("usage_ledger_id")
        }
        new_rows = [
            row
            for row in (post_ledger or [])
            if isinstance(row, dict)
            and row.get("usage_ledger_id")
            and str(row.get("usage_ledger_id")) not in pre_ids
        ]
        if pre_ids and new_rows:
            delta, new_unpriced = _price_ledger(
                new_rows,
                policy_model=policy_model,
                proposer_model=proposer_model,
            )
            unpriced = new_unpriced
            source = "usage_ledger_delta"
        elif post_total + 1e-12 < pre_total:
            # Imported fixture usage is not a prefix of this cursor (reset / fork
            # that did not keep archive totals). Post is the episode spend.
            delta = post_total
            unpriced = post_unpriced
            source = "usage_ledger_post"
        else:
            delta = max(0.0, post_total - pre_total)
    else:
        source = "flattened_usage"
        post_usage = (
            (finished.get("usage") if isinstance(finished.get("usage"), dict) else None)
            or (post.get("usage") if isinstance(post.get("usage"), dict) else post)
        )
        pre_usage = pre.get("usage") if isinstance(pre.get("usage"), dict) else pre
        post_billed = billed_cost_usd(post_usage)
        pre_billed = billed_cost_usd(pre_usage) or 0.0
        if post_billed is not None:
            delta = max(0.0, post_billed - pre_billed)
            source = "provider_billed"
        else:
            prompt, completion, _cached = token_counts(post_usage)
            mixed = bool(policy_model and proposer_model and policy_model != proposer_model)
            if mixed and (prompt or completion):
                unpriced += 1
                delta = 0.0
                source = "unpriced_mixed_models"
            else:
                model = proposer_model or policy_model
                post_cost, post_source = price_usage_usd(
                    post_usage, model=model, provider=None
                )
                pre_cost, _pre_source = price_usage_usd(
                    pre_usage, model=model, provider=None
                )
                if post_cost is None:
                    if prompt or completion:
                        unpriced += 1
                    delta = 0.0
                    source = post_source
                else:
                    delta = max(0.0, post_cost - (pre_cost or 0.0))
                    source = post_source
    if unpriced and missing == "fail":
        raise ValueError(
            "cost evidence unpriced (tokens present, cost_usd=0, no catalog rate) "
            "and combine.missing='fail'"
        )
    return {
        "episode_cost_usd": delta,
        "cost_source": source,
        "unpriced_rows": unpriced,
        "policy_model": policy_model,
        "proposer_model": proposer_model,
    }
