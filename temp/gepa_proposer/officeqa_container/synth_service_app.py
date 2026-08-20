"""OfficeQA inner http_task for GEPA. Oracle-text v0: question (+ source files if mounted)."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

from reward import score_answer

TASK_ID = "officeqa.grounded_reasoning"
DEFAULT_SYSTEM = (
    "You answer questions about U.S. Treasury documents. Use only the supplied "
    "source text. Be exact on figures, years, and units. Return only the final "
    "answer string, with no explanation."
)
DEFAULT_USER = "Question:\n{question}\n\nSource documents:\n{context}\n\nFinal answer:"
GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

_questions: list[dict[str, Any]] = []
_async_rollouts: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _csv_path() -> Path | None:
    raw = os.environ.get("OFFICEQA_CSV") or ""
    if raw:
        path = Path(raw)
        return path if path.is_file() else None
    default = Path(__file__).resolve().parent / "data" / "officeqa_full.csv"
    return default if default.is_file() else None


def _corpus_dir() -> Path | None:
    raw = os.environ.get("OFFICEQA_CORPUS_DIR") or ""
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def _load_questions() -> list[dict[str, Any]]:
    path = _csv_path()
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "uid": str(raw.get("uid") or f"q{len(rows)}"),
                    "question": str(raw.get("question") or ""),
                    "answer": str(raw.get("answer") or ""),
                    "source_files": str(raw.get("source_files") or ""),
                    "difficulty": str(raw.get("difficulty") or "hard").lower(),
                }
            )
    return [row for row in rows if row["question"] and row["answer"]]


def _split_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _questions:
        return [], []
    easy = [row for row in _questions if row["difficulty"] == "easy"]
    hard = [row for row in _questions if row["difficulty"] != "easy"]
    if easy and hard:
        return easy, hard
    n = len(_questions)
    cut = max(1, int(n * 0.6))
    return _questions[:cut], _questions[cut:] or _questions[-1:]


def _row_for(split: str, seed: int) -> dict[str, Any]:
    train, heldout = _split_rows()
    pool = train if split == "train" else heldout
    if not pool:
        raise HTTPException(
            status_code=503,
            detail=(
                "OfficeQA dataset is not mounted. Request access to "
                "huggingface.co/datasets/databricks/officeqa and set OFFICEQA_CSV."
            ),
        )
    row = pool[seed % len(pool)]
    return {
        "task_id": f"{split}:{seed}",
        "split": split,
        "seed": seed,
        "uid": row["uid"],
        "question": row["question"],
        "answer": row["answer"],
        "source_files": row["source_files"],
        "difficulty": row["difficulty"],
    }


def _context_for(row: dict[str, Any]) -> str:
    corpus = _corpus_dir()
    names = [part.strip() for part in str(row.get("source_files") or "").split(";") if part.strip()]
    if not corpus or not names:
        return "(corpus not mounted; answer from the question text only if you can, otherwise return Unable to determine)"
    chunks: list[str] = []
    for name in names:
        path = corpus / name
        if not path.is_file():
            path = corpus / Path(name).name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"# {path.name}\n{text[:80000]}")
    return "\n\n".join(chunks) if chunks else "(source files not found in OFFICEQA_CORPUS_DIR)"


def _policy(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    return {
        "model": raw.get("model") or os.environ.get("OFFICEQA_POLICY_MODEL") or "gpt-4.1",
        "base_url": (raw.get("base_url") or os.environ.get("OFFICEQA_POLICY_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        "api_key": os.environ.get(str(raw.get("env_var") or os.environ.get("OFFICEQA_POLICY_CREDENTIAL_ENV") or "OPENAI_API_KEY")),
        "max_tokens": int(raw.get("max_tokens") or os.environ.get("OFFICEQA_POLICY_MAX_TOKENS") or 256),
    }


def _overlay(payload: dict[str, Any]) -> str:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    return str(candidate.get("system_prompt") or DEFAULT_SYSTEM)


async def _complete(payload: dict[str, Any]) -> dict[str, Any]:
    split = str(payload.get("split") or "train")
    seed = int(payload.get("seed") or 0)
    task_id = str(payload.get("task_id") or f"{split}:{seed}")
    if ":" in task_id:
        maybe_split, _, maybe_seed = task_id.partition(":")
        if maybe_split in {"train", "heldout", "validation"} and maybe_seed.isdigit():
            split, seed = maybe_split, int(maybe_seed)
            if split == "validation":
                split = "heldout"
    row = _row_for(split, seed)
    policy = _policy(payload)
    if not policy["api_key"]:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is required for OfficeQA rollouts")
    client = AsyncOpenAI(api_key=policy["api_key"], base_url=policy["base_url"])
    user = DEFAULT_USER.format(question=row["question"], context=_context_for(row))
    response = await client.chat.completions.create(
        model=policy["model"],
        messages=[
            {"role": "system", "content": _overlay(payload)},
            {"role": "user", "content": user},
        ],
        max_tokens=policy["max_tokens"],
        temperature=0,
    )
    predicted = (response.choices[0].message.content or "").strip() if response.choices else ""
    usage = {
        "input_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
    }
    tolerance = float(os.environ.get("OFFICEQA_TOLERANCE") or 0.0)
    reward = float(score_answer(row["answer"], predicted, tolerance=tolerance))
    now = _now()
    rollout_id = str(payload.get("rollout_id") or f"officeqa_{uuid.uuid4().hex[:12]}")
    return {
        "rollout_id": rollout_id,
        "status": "completed",
        "success_status": "succeeded",
        "task_id": task_id,
        "seed": seed,
        "reward": reward,
        "reward_info": {"outcome_reward": reward, "metrics": {"officeqa_exact": reward}},
        "summary": {"outcome_reward": reward, "prediction": predicted, "uid": row["uid"]},
        "usage": usage,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
    }


app = FastAPI(title="officeqa-gepa-container")


@app.on_event("startup")
def _startup() -> None:
    global _questions
    _questions = _load_questions()


@app.get("/health")
def health() -> dict[str, Any]:
    train, heldout = _split_rows()
    return {
        "status": "ok",
        "dataset_mounted": bool(_questions),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
    }


@app.get("/metadata")
@app.get("/info")
def metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": "officeqa_gepa",
            "name": "OfficeQA GEPA inner task",
            "description": "Databricks OfficeQA grounded reasoning over U.S. Treasury Bulletins.",
        },
        "capabilities": {
            "contract_version": "container_contract.v1",
            "rollout_modes": ["blocking", "async"],
        },
        "metadata": {
            "optimizer_contracts": {
                "gepa": {
                    "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                    "program_route": "/program",
                    "taskset_route": "/taskset",
                    "taskset_tasks_route": "/taskset/tasks",
                    "rollout_route": "/rollout",
                }
            },
            "benchmark": {
                "name": "OfficeQA",
                "url": "https://www.databricks.com/blog/introducing-officeqa-benchmark-end-to-end-grounded-reasoning",
                "dataset": "databricks/officeqa",
                "scorer": "github.com/databricks/officeqa reward.py",
            },
        },
    }


@app.get("/program")
def program() -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "officeqa_system_prompt",
        "modules": [
            {
                "module_id": "system_prompt",
                "role": "system",
                "content": DEFAULT_SYSTEM,
                "mutable": True,
                "candidate_field": "system_prompt",
                "template_variables": [],
            }
        ],
        "target_modules": [{"module_id": "system_prompt", "candidate_field": "system_prompt", "objective": "outcome_reward"}],
        "seed_candidate": {"system_prompt": DEFAULT_SYSTEM},
        "rollout_overlay_schema": {"candidate_fields": ["system_prompt"]},
    }


@app.get("/taskset")
def taskset() -> dict[str, Any]:
    train, heldout = _split_rows()
    train_n = len(train) or int(os.environ.get("OFFICEQA_TRAIN_N") or 24)
    heldout_n = len(heldout) or int(os.environ.get("OFFICEQA_HELDOUT_N") or 16)
    return {"taskset_id": "officeqa:full", "splits": {"train": train_n, "heldout": heldout_n}}


@app.post("/taskset/tasks")
async def taskset_tasks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    split = str(payload.get("split") or "train")
    raw_ids = payload.get("task_ids") or []
    tasks = []
    for raw in raw_ids:
        task_id = str(raw)
        split_name, _, seed_s = task_id.partition(":")
        seed = int(seed_s) if seed_s.isdigit() else 0
        if _questions:
            row = _row_for(split_name if split_name in {"train", "heldout"} else split, seed)
            tasks.append(row)
        else:
            tasks.append({"task_id": task_id, "split": split, "seed": seed})
    return {"tasks": tasks}


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    mode = str(payload.get("submission_mode") or "sync").strip().lower()
    if mode == "async":
        rollout_id = str(payload.get("rollout_id") or f"officeqa_{uuid.uuid4().hex[:12]}")
        queued = {
            "rollout_id": rollout_id,
            "status": "running",
            "success_status": "running",
            "task_id": payload.get("task_id"),
            "created_at": _now(),
            "updated_at": _now(),
        }
        async with _lock:
            _async_rollouts[rollout_id] = queued

        async def worker() -> None:
            try:
                finished = await _complete({**payload, "rollout_id": rollout_id})
            except HTTPException as exc:
                finished = {
                    **queued,
                    "status": "failed",
                    "success_status": "failed",
                    "error": exc.detail,
                    "updated_at": _now(),
                }
            async with _lock:
                _async_rollouts[rollout_id] = finished

        asyncio.create_task(worker())
        return queued
    return await _complete(payload)


@app.get("/rollouts/{rollout_id}")
@app.get("/rollouts/{rollout_id}/state")
async def rollout_get(rollout_id: str) -> dict[str, Any]:
    async with _lock:
        record = _async_rollouts.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail="rollout not found")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
