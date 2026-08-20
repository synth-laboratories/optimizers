"""PRBench GEPA container (real ScaleAI/PRBench data, OpenAI policy + OpenAI rubric judge).

Speaks the public synth-optimizers GEPA contract:
  GET  /health
  GET  /metadata
  GET  /task_info
  GET  /program
  GET  /taskset          POST /taskset/tasks
  GET  /dataset          POST /dataset/rows
  POST /rollout          POST /rollouts

Each rollout: run the candidate `system_prompt` against one real PRBench task
through an OpenAI chat-completions policy, then grade the generated response
against that task's expert rubric with an OpenAI judge. Reward is the PRBench
paper score (Appendix D.1): signed weighted sum of satisfied criteria over the
sum of positive weights, clipped to [0, 1].

Chat Completions only, on both the policy and judge legs. The Responses API
branch is deliberately absent: it needs `max_output_tokens` rather than
`max_tokens`, needs reasoning suppression, and needs `status != "completed"`
treated as an infra failure — three separate ways to silently score an empty
generation as a bad answer.

Required env:
  OPENAI_API_KEY          — policy (byok) and judge.
  OPENROUTER_API_KEY      — only if rollout.policy.provider=openrouter.
  PRBENCH_GRADER_MODEL    — judge model, default gpt-4.1-mini.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

import tasks as prbench

GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

try:
    from openai import AsyncOpenAI
except Exception as _openai_err:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = _openai_err
else:
    _OPENAI_IMPORT_ERROR = None

TASK_ID = "prbench.professional_reasoning"
DATASET_ID = "prbench_scaleai_public"

GRADER_MODEL = os.environ.get("PRBENCH_GRADER_MODEL", "gpt-4.1-mini")
GRADER_BASE_URL = (os.environ.get("PRBENCH_GRADER_BASE_URL") or "").strip() or None
GRADER_BATCH = int(os.environ.get("PRBENCH_GRADER_BATCH", "12"))
GRADER_TIMEOUT_SECONDS = float(os.environ.get("PRBENCH_GRADER_TIMEOUT_SECONDS", "180"))
POLICY_TIMEOUT_SECONDS = float(os.environ.get("PRBENCH_POLICY_TIMEOUT_SECONDS", "180"))
RETRIES = int(os.environ.get("PRBENCH_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("PRBENCH_RETRY_BACKOFF_SECONDS", "1.0"))
DEFAULT_POLICY_MAX_TOKENS = int(os.environ.get("PRBENCH_POLICY_MAX_TOKENS", "2048"))
MAX_RESPONSE_CHARS_FOR_JUDGE = int(os.environ.get("PRBENCH_MAX_RESPONSE_CHARS", "24000"))

_clients: dict[tuple[str, str], Any] = {}
_RAW_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_key",
}


# --- policy plumbing ----------------------------------------------------------


def _find_raw_credential_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            normalized = str(raw_key).strip().lower().replace("-", "_")
            if normalized in _RAW_CREDENTIAL_KEYS or normalized.endswith("_api_key"):
                return str(raw_key)
            nested = _find_raw_credential_key(raw_value)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_raw_credential_key(item)
            if nested is not None:
                return nested
    return None


def _normalize_policy_enum(value: Any, default: str) -> str:
    return str(value or "").strip().lower().replace("-", "_") or default


def _strip_openai_endpoint_suffix(url: str) -> str:
    normalized = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _require_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    provider = str(
        policy.get("provider") or os.environ.get("PRBENCH_POLICY_PROVIDER") or "openai"
    ).strip()
    model = str(
        policy.get("model") or os.environ.get("PRBENCH_POLICY_MODEL") or "gpt-4.1-nano"
    ).strip()
    raw_key = _find_raw_credential_key(policy.get("config", {}))
    if raw_key is not None:
        raise HTTPException(
            status_code=422,
            detail=f"rollout.policy.config must not carry raw credential field {raw_key!r}.",
        )
    if not provider or not model:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.provider and rollout.policy.model are required.",
        )
    api_family = _normalize_policy_enum(policy.get("api_family"), "chat_completions")
    if api_family != "chat_completions":
        raise HTTPException(
            status_code=422,
            detail=(
                f"{TASK_ID} supports rollout.policy.api_family='chat_completions' only; "
                f"got {api_family!r}."
            ),
        )
    credential_mode = _normalize_policy_enum(policy.get("credential_mode"), "byok")
    if credential_mode in {"proxy_only", "proxy"} and not (
        str(policy.get("inference_url") or "").strip()
        or str(policy.get("base_url") or "").strip()
    ):
        credential_mode = "byok"
    if credential_mode not in {"byok", "proxy"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported rollout.policy.credential_mode: {credential_mode!r}",
        )
    raw_base_url = (
        str(policy.get("inference_url") or "").strip()
        if credential_mode == "proxy"
        else str(policy.get("base_url") or "").strip()
    )
    if credential_mode == "proxy" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.inference_url is required when credential_mode=proxy.",
        )
    if provider.lower() == "openrouter" and credential_mode == "byok" and not raw_base_url:
        raise HTTPException(
            status_code=422,
            detail="rollout.policy.base_url is required for provider=openrouter.",
        )
    max_tokens = policy.get("max_tokens")
    if max_tokens is None:
        max_tokens = DEFAULT_POLICY_MAX_TOKENS
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="rollout.policy.max_tokens must be an integer when set."
        ) from exc
    if max_tokens <= 0:
        raise HTTPException(
            status_code=422, detail="rollout.policy.max_tokens must be positive when set."
        )
    temperature = policy.get("temperature")
    return {
        "provider": provider,
        "model": model,
        "base_url": _strip_openai_endpoint_suffix(raw_base_url) if raw_base_url else None,
        "credential_mode": credential_mode,
        "max_tokens": max_tokens,
        "temperature": 0.0 if temperature is None else float(temperature),
    }


def _policy_api_key(policy: dict[str, Any]) -> str:
    if policy["credential_mode"] == "proxy":
        return "proxy"
    env_name = (
        "OPENROUTER_API_KEY" if policy["provider"].lower() == "openrouter" else "OPENAI_API_KEY"
    )
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    raise HTTPException(
        status_code=503,
        detail=f"{env_name} is not set; credential_mode=byok requires a container env credential.",
    )


def _client(api_key: str, base_url: str | None) -> Any:
    if AsyncOpenAI is None:
        raise HTTPException(
            status_code=503,
            detail=f"openai package not installed; see pyproject.toml. {_OPENAI_IMPORT_ERROR!r}",
        )
    key = (api_key[-8:], str(base_url or ""))
    client = _clients.get(key)
    if client is None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncOpenAI(**kwargs)
        _clients[key] = client
    return client


def _grader_client() -> Any:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if not value:
        raise HTTPException(
            status_code=503, detail="OPENAI_API_KEY is not set; the PRBench rubric judge requires it."
        )
    return _client(value, GRADER_BASE_URL)


def _is_timeout(error: Exception) -> bool:
    name = type(error).__name__.lower()
    return isinstance(error, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name


def _retry_delay(attempt: int) -> float:
    return min(RETRY_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)), 8.0)


async def _chat(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
    json_object: bool = False,
    leg: str = "policy",
) -> tuple[str, dict[str, int]]:
    """Chat-completions call with bounded retries.

    Any exhausted failure is raised as a non-2xx: a policy or judge call that
    never produced text is an infra failure, not a wrong answer, and must never
    be scored as a zero.
    """
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        request["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(**request), timeout=timeout
            )
        except Exception as error:  # noqa: BLE001 — retried, then surfaced as 5xx
            last_error = error
            if attempt >= RETRIES:
                break
            await asyncio.sleep(_retry_delay(attempt))
            continue

        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        text = ((getattr(choice, "message", None).content if choice else None) or "").strip()
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        usage_obj = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }
        if not text:
            # Empty completion (length cap, content filter, upstream hiccup) is an
            # infra failure. Grading "" against the rubric would score 0 and look
            # like a bad prompt.
            last_error = RuntimeError(f"empty completion (finish_reason={finish_reason!r})")
            if attempt >= RETRIES:
                break
            await asyncio.sleep(_retry_delay(attempt))
            continue
        return text, usage

    status_code = 504 if last_error is not None and _is_timeout(last_error) else 502
    raise HTTPException(
        status_code=status_code,
        detail=(
            f"PRBench {leg} model {model!r} failed via Chat Completions after "
            f"{RETRIES} attempts: {type(last_error).__name__}: {last_error}"
        ),
    )


# --- policy + judge -----------------------------------------------------------


def _policy_messages(rec: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in rec.get("context") or []:
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or "")
        if content:
            messages.append({"role": role, "content": content})
    user = rec["prompt"]
    if rec.get("reference_text"):
        user = f"{user}\n\n<reference_material>\n{rec['reference_text']}\n</reference_material>"
    messages.append({"role": "user", "content": user})
    return messages


_JUDGE_SYSTEM = (
    "You are an expert grader for PRBench, a benchmark of high-stakes professional "
    "reasoning in finance and law. You are given a professional's question, the "
    "assistant response under evaluation, and a list of binary rubric criteria "
    "written by a domain expert (JD, CFA, or 6+ years of practice).\n\n"
    "For each criterion, decide whether the response satisfies it. A criterion is "
    "satisfied only if the response actually contains or demonstrates what the "
    "criterion describes; do not give credit for near-misses, implications the "
    "reader would have to supply, or content the response merely gestures at. Some "
    "criteria describe undesirable properties (errors, omissions, overstatements). "
    "For those, 'met' means the response exhibits the undesirable property.\n\n"
    'Reply with strict JSON only: {"judgements": [{"criterion_id": "<id>", '
    '"met": true|false}, ...]} with exactly one entry per criterion given to you.'
)


def _judge_user_message(rec: dict[str, Any], response_text: str, batch: list[dict[str, Any]]) -> str:
    context_block = ""
    if rec.get("context"):
        rendered = "\n\n".join(
            f"[{t['role']}]\n{t['content']}" for t in rec["context"] if t.get("content")
        )
        context_block = f"<prior_conversation>\n{rendered}\n</prior_conversation>\n\n"
    reference_block = ""
    if rec.get("reference_text"):
        reference_block = (
            f"<reference_material>\n{rec['reference_text']}\n</reference_material>\n\n"
        )
    criteria_lines = []
    for c in batch:
        polarity = "UNDESIRABLE" if c["weight"] < 0 else "REQUIRED"
        extra = f"\n  note: {c['description']}" if c["description"] else ""
        criteria_lines.append(
            f"- criterion_id: {c['criterion_id']}\n"
            f"  polarity: {polarity} ({c['weight_class']})\n"
            f"  category: {c['category']}\n"
            f"  criterion: {c['title']}{extra}"
        )
    truncated = response_text
    if len(truncated) > MAX_RESPONSE_CHARS_FOR_JUDGE:
        truncated = truncated[:MAX_RESPONSE_CHARS_FOR_JUDGE] + "\n[...response truncated...]"
    return (
        f"<domain>{rec['field']} / {rec['topic']}</domain>\n\n"
        f"{context_block}{reference_block}"
        f"<question>\n{rec['prompt']}\n</question>\n\n"
        f"<response_under_evaluation>\n{truncated}\n</response_under_evaluation>\n\n"
        f"<rubric_criteria>\n" + "\n".join(criteria_lines) + "\n</rubric_criteria>\n\n"
        f"Return JSON with exactly {len(batch)} judgements, one per criterion_id above."
    )


def _parse_judgements(text: str, batch: list[dict[str, Any]]) -> dict[str, bool]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    items = parsed.get("judgements") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError(f"judge returned no judgements list: {type(items).__name__}")
    valid = {c["criterion_id"] for c in batch}
    out: dict[str, bool] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("criterion_id") or "").strip()
        if cid in valid:
            out[cid] = bool(item.get("met"))
    if not out:
        raise ValueError("judge returned no criterion_id matching this batch")
    return out


async def _grade(rec: dict[str, Any], response_text: str) -> tuple[dict[str, bool], dict[str, int], int]:
    client = _grader_client()
    criteria = rec["criteria"]
    batches = [criteria[i : i + GRADER_BATCH] for i in range(0, len(criteria), GRADER_BATCH)]

    async def one(batch: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, int]]:
        last_error: Exception | None = None
        for attempt in range(1, RETRIES + 1):
            text, usage = await _chat(
                client,
                model=GRADER_MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": _judge_user_message(rec, response_text, batch)},
                ],
                max_tokens=max(512, 64 * len(batch)),
                temperature=0.0,
                timeout=GRADER_TIMEOUT_SECONDS,
                json_object=True,
                leg="judge",
            )
            try:
                return _parse_judgements(text, batch), usage
            except Exception as error:  # noqa: BLE001 — unparseable judge output is infra
                last_error = error
                if attempt < RETRIES:
                    await asyncio.sleep(_retry_delay(attempt))
        raise HTTPException(
            status_code=502,
            detail=(
                f"PRBench judge {GRADER_MODEL!r} returned unparseable output after "
                f"{RETRIES} attempts: {type(last_error).__name__}: {last_error}"
            ),
        )

    results = await asyncio.gather(*(one(b) for b in batches))
    met: dict[str, bool] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for judgements, batch_usage in results:
        met.update(judgements)
        for k in usage:
            usage[k] += batch_usage.get(k, 0)
    return met, usage, len(batches)


# --- request parsing ----------------------------------------------------------


def _example_row(payload: dict[str, Any]) -> dict[str, Any]:
    """GEPA sends the *program id* as top-level task_id and the dataset row under `task`.

    Parsing the row out of `payload.task` is mandatory. Treating the top-level
    task_id as the row id makes every rollout hit the same example.
    """
    task = payload.get("task")
    if isinstance(task, dict):
        example = task.get("example") if isinstance(task.get("example"), dict) else task
        if isinstance(example, dict) and (
            example.get("example_id") or example.get("task_id") or example.get("prbench_task")
        ):
            return example
    row = payload.get("dataset_row")
    if isinstance(row, dict) and row:
        return row
    return {}


def _resolve_record(payload: dict[str, Any]) -> dict[str, Any]:
    example = _example_row(payload)
    rec = prbench.lookup(
        example.get("example_id"),
        example.get("task_id"),
        example.get("prbench_task"),
    )
    if rec is None:
        # Only fall back to the top-level task_id if it names a real row; a GEPA
        # program id must not silently resolve to train:0.
        rec = prbench.lookup(payload.get("example_id"), payload.get("task_id"))
    if rec is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "could not resolve a PRBench row: expected payload.task to carry "
                "example_id/task_id (e.g. 'train:3') or prbench_task; got "
                f"example keys {sorted(example.keys())!r}, top-level task_id "
                f"{payload.get('task_id')!r}."
            ),
        )
    return rec


# --- FastAPI ------------------------------------------------------------------

app = FastAPI(title="prbench-gepa-container")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@app.get("/health")
async def health() -> dict[str, Any]:
    error = prbench.load_error()
    stats = prbench.dataset_stats()
    return {
        "status": "ok" if error is None else "degraded",
        "error": error,
        "dataset": stats,
        "grader_model": GRADER_MODEL,
        "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    stats = prbench.dataset_stats()
    return {
        "runtime": {
            "runtime_id": "prbench_gepa_live",
            "name": "PRBench GEPA (real ScaleAI/PRBench tasks, OpenAI policy, OpenAI rubric judge)",
            "description": (
                "Prompt-optimizer container serving real expert-authored PRBench "
                "finance/law tasks and grading generations against the released "
                "expert rubrics with an OpenAI judge."
            ),
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking"],
            "metadata": {
                "trace_schema": "prompt_calls.llm_request.messages.v1",
                "policy_ready": True,
            },
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "taskset_route": "/taskset",
                    "taskset_tasks_route": "/taskset/tasks",
                    "dataset_route": "/dataset",
                    "dataset_rows_route": "/dataset/rows",
                    "rollout_route": "/rollout",
                    "overlay": "prompt_overlay.v1",
                    "candidate_fields": ["system_prompt"],
                }
            },
            "benchmark": {
                "name": "PRBench",
                "provenance": "REAL — unmodified upstream release, no synthetic or hand-authored tasks",
                "source": "https://huggingface.co/datasets/ScaleAI/PRBench",
                "paper": "arXiv:2511.11562",
                "code": "https://github.com/scaleapi/PRBench",
                "license": "cc-by-4.0",
                "subsets_served": stats["loaded_subsets"],
                "subsets_available_upstream": ["finance", "finance_hard", "legal", "legal_hard"],
                "rows_served": {
                    "train": stats["train_rows"],
                    "heldout": stats["heldout_rows"],
                    "sampled_from_pool": stats["candidate_pool_size"],
                },
                "expert_criteria_in_served_split": stats["criteria_in_served_split"],
                "row_filter": f"turns <= {stats['max_turns']} (long conversations excluded for cost)",
                "scorer": (
                    f"OpenAI LLM judge ({GRADER_MODEL}) over the released expert rubric; "
                    "reward = PRBench Appendix D.1 score = signed weighted sum of satisfied "
                    "criteria / sum of positive weights, clipped to [0,1]"
                ),
                "scorer_caveat": (
                    "The judge prompt is this container's own HealthBench-style implementation, "
                    "not the upstream scaleapi/PRBench judge template; absolute rewards are not "
                    "directly comparable to the published leaderboard."
                ),
                "rubric_visibility": "rubric criteria are withheld from dataset rows and the optimizer",
            },
        },
    }


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    stats = prbench.dataset_stats()
    return {
        "task": {
            "task_id": TASK_ID,
            "name": "PRBench professional reasoning (finance + law)",
            "description": (
                "Optimize a system prompt for an OpenAI model answering real "
                "expert-authored finance and law questions. Each rollout generates "
                "one response and grades it against that task's expert rubric."
            ),
            "objective": "Maximize the weighted fraction of expert rubric criteria satisfied.",
            "domain": "open-ended high-stakes professional reasoning, rubric-graded",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "visible_splits": ["train", "test"],
            "default_split": "train",
            "row_count": stats["train_rows"] + stats["heldout_rows"],
            "row_semantics": (
                "Each row is one PRBench conversation. The row carries the final "
                "user prompt plus prior turns; the rubric stays server-side."
            ),
        },
        "prompt_program": {
            "mutable_modules": ["system_prompt"],
            "candidate_field": "system_prompt",
            "output_contract": "Free-form professional prose. No required output format.",
        },
        "evaluation": {
            "primary_metric": "outcome_reward",
            "success_status": "succeeded for every graded generation, including low scores",
            "rollout_trace_contains": [
                "weighted_numerator",
                "weighted_denominator",
                "positive_criteria_met",
                "detrimental_criteria_triggered",
            ],
        },
        "proposal_guidance": {
            "premises": [
                "Rubrics are written by JDs, CFAs and 6+ year practitioners; they reward "
                "specific, correct, actionable professional content.",
                "Criteria span accuracy, procedural correctness, handling uncertainty, "
                "risk/regulatory disclosure, practical utility and instruction following.",
                "Rubrics also carry negatively weighted criteria that penalize "
                "overstatement, unsupported quantitative claims and missing caveats.",
                "Rubrics have 5-30 criteria; broad but grounded coverage scores better "
                "than a short answer that nails one point.",
            ],
            "constraints": [
                "Do not instruct the model to guess at or restate rubric criteria; the "
                "rubric is never shown to the policy.",
                "Do not encourage fabricated citations, statutes, or figures — those "
                "trip critically-detrimental criteria.",
                "Keep the system prompt domain-general: the split mixes finance and law.",
            ],
            "high_leverage_heuristics": [
                "Ask for jurisdiction/standard identification before analysis.",
                "Ask for explicit uncertainty and assumption statements rather than "
                "confident single answers.",
                "Ask for actionable next steps and risk/regulatory caveats.",
                "Ask for structure that covers multiple angles without padding.",
            ],
            "anti_patterns": [
                "Generic helpful-assistant persona text.",
                "Instructions to be brief — brevity loses coverage of many criteria.",
                "Instructions to invent precise numbers or case citations.",
            ],
        },
        "metadata": {
            "policy_model_source": "rollout.policy.model",
            "grader_model": GRADER_MODEL,
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/program")
async def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "prbench_system_prompt_gepa",
        "modules": [
            {
                "module_id": "system_prompt",
                "role": "system",
                "content": prbench.DEFAULT_SYSTEM_PROMPT,
                "mutable": True,
                "candidate_field": "system_prompt",
                "template_variables": [],
                "metadata": {"benchmark": "PRBench"},
            }
        ],
        "target_modules": [
            {
                "module_id": "system_prompt",
                "candidate_field": "system_prompt",
                "objective": "outcome_reward",
            }
        ],
        "seed_candidate": {"system_prompt": prbench.DEFAULT_SYSTEM_PROMPT},
        "rollout_overlay_schema": {"candidate_fields": ["system_prompt"]},
        "metadata": {
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "overlay": "prompt_overlay.v1",
            "trace_schema": "prompt_calls.llm_request.messages.v1",
        },
    }


@app.get("/taskset")
async def taskset() -> dict[str, Any]:
    stats = prbench.dataset_stats()
    return {
        "taskset_id": "prbench:finance_law_hard",
        "splits": {"train": stats["train_rows"], "heldout": stats["heldout_rows"]},
        "source": DATASET_ID,
    }


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    raw_ids = payload.get("task_ids") or []
    if not raw_ids:
        return {"tasks": prbench.rows_for(split)}
    out = []
    for raw in raw_ids:
        rec = prbench.lookup(raw)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"unknown PRBench task id {raw!r}")
        out.append(prbench.public_row(rec))
    return {"tasks": out}


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    stats = prbench.dataset_stats()
    return {
        "dataset_id": DATASET_ID,
        "splits": {"train": stats["train_rows"], "heldout": stats["heldout_rows"]},
        "source": "https://huggingface.co/datasets/ScaleAI/PRBench",
    }


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return {"rows": prbench.rows_for(str(payload.get("split") or "train"))}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    load_error = prbench.load_error()
    if load_error:
        raise HTTPException(status_code=503, detail=load_error)

    policy = _require_policy(payload)
    rec = _resolve_record(payload)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    system_prompt = str(candidate.get("system_prompt") or "").strip() or prbench.DEFAULT_SYSTEM_PROMPT

    client = _client(_policy_api_key(policy), policy["base_url"])
    started = time.time()
    response_text, policy_usage = await _chat(
        client,
        model=policy["model"],
        messages=_policy_messages(rec, system_prompt),
        max_tokens=policy["max_tokens"],
        temperature=policy["temperature"],
        timeout=POLICY_TIMEOUT_SECONDS,
        leg="policy",
    )
    met, judge_usage, judge_calls = await _grade(rec, response_text)
    scored = prbench.score_from_judgements(rec, met)
    reward = float(scored["reward"])

    usage = {
        "prompt_tokens": policy_usage["prompt_tokens"] + judge_usage["prompt_tokens"],
        "completion_tokens": policy_usage["completion_tokens"] + judge_usage["completion_tokens"],
        "total_tokens": policy_usage["total_tokens"] + judge_usage["total_tokens"],
        "cost_usd": 0.0,
    }
    rollout_id = str(payload.get("rollout_id") or f"prbench_{uuid.uuid4().hex[:12]}")
    now = _now()
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        # A low rubric score is a solved rollout with a bad answer, never an
        # infra failure. Only genuine call failures leave via HTTPException.
        "success_status": "succeeded",
        "task_id": rec["task_id"],
        "reward": reward,
        "reward_info": {
            "outcome_reward": reward,
            "metrics": {
                "prbench_score": reward,
                "raw_score": scored["raw_score"],
                "weighted_numerator": scored["weighted_numerator"],
                "weighted_denominator": scored["weighted_denominator"],
                "criteria_total": scored["criteria_total"],
                "positive_criteria": scored["positive_criteria"],
                "positive_criteria_met": scored["positive_criteria_met"],
                "unweighted_positive_fraction": scored["unweighted_positive_fraction"],
                "detrimental_criteria_triggered": scored["detrimental_criteria_triggered"],
                "judge_calls": judge_calls,
                "subset": rec["subset"],
                "field": rec["field"],
            },
        },
        "summary": {
            "outcome_reward": reward,
            "example_id": rec["example_id"],
            "split": rec["split"],
            "prbench_task": rec["prbench_task"],
            "subset": rec["subset"],
            "topic": rec["topic"],
            "prompt_preview": rec["prompt"][:240],
            "response_chars": len(response_text),
            "response_preview": response_text[:400],
            "criteria_total": scored["criteria_total"],
            "positive_criteria_met": scored["positive_criteria_met"],
            "detrimental_criteria_triggered": scored["detrimental_criteria_triggered"],
            "elapsed_seconds": round(time.time() - started, 2),
        },
        "trace": {
            "policy_model": policy["model"],
            "grader_model": GRADER_MODEL,
            "per_criterion": scored["per_criterion"],
        },
        "usage": usage,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
