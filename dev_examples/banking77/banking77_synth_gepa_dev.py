"""Banking77 task, HTTP container, and synth-optimizers GEPA dev run."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import uvicorn
from openai import AsyncOpenAI
from synth_containers import Container
from synth_optimizers import (
    BudgetConfig,
    CacheConfig,
    GepaBudgetConfig,
    GepaConfig,
    GepaPipeline,
    OptimizerRun,
    ProposerConfig,
    RunSettings,
    TasksetSelection,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*leaked semaphore objects.*")
logging.getLogger("datasets").setLevel(logging.ERROR)

GEPA_CONTRACT_VERSION = "synth_optimizers.gepa.v2"


# --- 1. Task -----------------------------------------------------------------


class Banking77ClassificationTask:
    TASK_ID = "banking77.intent_classification"
    CLASSIFIER_FIELD = "banking77_classifier"
    PROGRAM_ID = "banking77_single_stage_gepa"
    DATASET_ID = "banking77_dev_rows"
    RUNTIME_ID = "banking77_gepa_dev"

    SEED_CLASSIFIER_PROMPT = (
        "Classify the customer banking query into exactly one Banking77 intent. "
        "Return exactly one label from the allowed list, preserving spelling, "
        "underscores, capitalization, and punctuation. Return only the label."
    )

    PROPOSER_HINTS = {
        "task_output_space": "finite_intent_label",
        "proposal_goal": (
            "Use rollout wins and losses to refine label-boundary rules in the system prompt."
        ),
    }

    def __init__(
        self,
        *,
        train_sample: int | None = None,
        test_sample: int | None = None,
        train_shuffle_seed: int = 1009,
        test_shuffle_seed: int = 2003,
        policy_model: str | None = None,
        policy_timeout_seconds: float = 20.0,
        policy_concurrency: int | None = None,
    ) -> None:
        self.train_sample = train_sample or int(os.environ.get("BANKING77_TRAIN_SAMPLE", "24"))
        self.test_sample = test_sample or int(os.environ.get("BANKING77_TEST_SAMPLE", "8"))
        self.train_shuffle_seed = train_shuffle_seed
        self.test_shuffle_seed = test_shuffle_seed
        self.policy_model = policy_model or os.environ.get("BANKING77_POLICY_MODEL", "gpt-4.1-nano")
        self.policy_timeout_seconds = policy_timeout_seconds
        self.policy_concurrency = policy_concurrency or int(
            os.environ.get("BANKING77_POLICY_CONCURRENCY", "16")
        )
        self.rollout_timeout_seconds = self.policy_timeout_seconds + 5.0

        self._data_cache: tuple[list[str], list[dict[str, Any]]] | None = None
        self._label_lookup: dict[str, str] | None = None
        self._openai: AsyncOpenAI | None = None
        self._policy_sem: asyncio.Semaphore | None = None

    def labels_and_rows(self) -> tuple[list[str], list[dict[str, Any]]]:
        if self._data_cache is None:
            self._data_cache = self._load_rows()
        return self._data_cache

    def task_for_id(self, *, split: str, task_id: str) -> dict[str, Any]:
        rows = self.labels_and_rows()[1]
        split_rows = [row for row in rows if row["split"] == split] or list(rows)
        match = next((row for row in split_rows if str(row["task_id"]) == task_id), None)
        if match is None:
            raise ValueError(f"unknown Banking77 task_id {task_id!r} for split {split!r}")
        return dict(match)

    def normalize_label(self, raw: str) -> str:
        labels = self.labels_and_rows()[0]
        lookup = self._canonical_label_lookup()
        candidate = raw.strip().strip("`'\"").splitlines()[0].strip()
        if candidate in labels:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
        for label in labels:
            if label.lower() in candidate.lower():
                return label
        return candidate or "<no_label>"

    async def predict_label(self, text: str, *, system_prompt: str) -> tuple[str, dict[str, int]]:
        labels = self.labels_and_rows()[0]
        user_content = (
            f"Customer query:\n{text}\n\n"
            f"Return exactly one Banking77 label from this list:\n"
            + "\n".join(f"- {label}" for label in labels)
        )
        async with self._policy_semaphore():
            response = await asyncio.wait_for(
                self._openai_client().chat.completions.create(
                    model=self.policy_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=16,
                ),
                timeout=self.policy_timeout_seconds,
            )
        raw = (response.choices[0].message.content or "").strip()
        usage = response.usage
        token_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return self.normalize_label(raw), token_usage

    async def run_rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = payload.get("task")
        if not isinstance(row, dict):
            row = self.task_for_id(
                split=str(payload.get("split") or "train"),
                task_id=str(payload.get("task_id") or ""),
            )

        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        system_prompt = str(candidate.get(self.CLASSIFIER_FIELD) or self.SEED_CLASSIFIER_PROMPT)
        prediction, usage = await self.predict_label(
            str(row.get("text") or ""),
            system_prompt=system_prompt,
        )

        expected = str(row.get("label") or "")
        reward = 1.0 if prediction == expected else 0.0
        now = self._utc_now()
        rollout_id = str(payload.get("rollout_id") or f"rollout_{uuid.uuid4().hex[:12]}")

        return {
            "rollout_id": rollout_id,
            "status": "completed",
            "success_status": "succeeded" if reward > 0 else "failed",
            "task_id": str(row.get("task_id") or ""),
            "reward_info": {
                "outcome_reward": reward,
                "event_rewards": [reward],
                "details": {
                    "prediction": prediction,
                    "expected": expected,
                    "policy_model": self.policy_model,
                },
            },
            "summary": {"outcome_reward": reward, "prediction": prediction, "expected": expected},
            "usage": {**usage, "cost_usd": 0.0},
            "trace": {
                "event_history": [
                    {"type": "input", "text": row.get("text")},
                    {"type": "prediction", "label": prediction},
                ],
                "metadata": {"label": expected},
            },
            "metadata": {"candidate": candidate},
            "created_at": now,
            "updated_at": now,
        }

    def container_metadata(self) -> dict[str, Any]:
        return {
            "runtime": {
                "runtime_id": self.RUNTIME_ID,
                "name": "Banking77 GEPA dev container",
            },
            "capabilities": {
                "contract_version": "container_contract.v1",
                "rollout_modes": ["blocking"],
            },
            "metadata": {
                "optimizer_contracts": {
                    "gepa": {
                        "version": GEPA_CONTRACT_VERSION,
                        "program_route": "/program",
                        "taskset_route": "/taskset",
                        "taskset_tasks_route": "/taskset/tasks",
                        "rollout_route": "/rollout",
                    }
                }
            },
        }

    def task_info_payload(self) -> dict[str, Any]:
        labels, rows = self.labels_and_rows()
        return {
            "task": {
                "task_id": self.TASK_ID,
                "name": "Banking77 intent classification",
                "description": "Classify a customer banking question into one Banking77 label.",
            },
            "output_space": {
                "kind": "finite_intent_label",
                "label_count": len(labels),
                "labels": labels,
                "contract": "Return exactly one canonical label from the allowed list.",
            },
            "taskset": {
                "taskset_id": self.DATASET_ID,
                "visible_splits": ["train", "test"],
                "default_split": "train",
                "row_count": len(rows),
                "sampling": self.sampling_metadata(),
            },
            "proposer_hints": self.PROPOSER_HINTS,
        }

    def program_payload(self) -> dict[str, Any]:
        field = self.CLASSIFIER_FIELD
        return {
            "version": "prompt_program.v1",
            "program_id": self.PROGRAM_ID,
            "modules": [
                {
                    "module_id": field,
                    "role": "system",
                    "content": self.SEED_CLASSIFIER_PROMPT,
                    "mutable": True,
                    "candidate_field": field,
                    "template_variables": [],
                }
            ],
            "target_modules": [
                {
                    "module_id": field,
                    "candidate_field": field,
                    "objective": "classification_accuracy",
                }
            ],
            "seed_candidate": {field: self.SEED_CLASSIFIER_PROMPT},
            "rollout_overlay_schema": {"candidate_fields": [field]},
        }

    def taskset_payload(self) -> dict[str, Any]:
        labels, rows = self.labels_and_rows()
        return {
            "taskset_id": self.DATASET_ID,
            "splits": {
                "train": sum(1 for row in rows if row["split"] == "train"),
                "test": sum(1 for row in rows if row["split"] == "test"),
            },
            "sampling": self.sampling_metadata(),
            "labels": labels,
        }

    def taskset_tasks_payload(self, *, split: str, task_ids: list[str]) -> dict[str, Any]:
        return {
            "tasks": [self.task_for_id(split=split, task_id=task_id) for task_id in task_ids],
        }

    def sampling_metadata(self) -> dict[str, Any]:
        return {
            "train_sample": self.train_sample,
            "test_sample": self.test_sample,
            "train_shuffle_seed": self.train_shuffle_seed,
            "test_shuffle_seed": self.test_shuffle_seed,
            "method": "balanced_random_per_label",
        }

    def format_user_prompt(self, text: str) -> str:
        labels = self.labels_and_rows()[0]
        return (
            f"Customer query:\n{text}\n\n"
            f"Return exactly one Banking77 label from this list:\n"
            + "\n".join(f"- {label}" for label in labels)
        )

    async def score_classifier_prompt(
        self,
        prompt: str,
        *,
        split: str,
        task_ids: list[str],
    ) -> float:
        if not task_ids:
            return 0.0
        correct = 0
        for task_id in task_ids:
            row = self.task_for_id(split=split, task_id=task_id)
            prediction, _ = await self.predict_label(
                str(row.get("text") or ""), system_prompt=prompt
            )
            if prediction == str(row.get("label") or ""):
                correct += 1
        return correct / len(task_ids)

    def score_classifier_prompt_sync(
        self,
        prompt: str,
        *,
        split: str,
        task_ids: list[str],
    ) -> float:
        return asyncio.run(self.score_classifier_prompt(prompt, split=split, task_ids=task_ids))

    def _load_rows(self) -> tuple[list[str], list[dict[str, Any]]]:
        from datasets import load_dataset

        dataset = load_dataset("PolyAI/banking77")
        label_names: list[str] = list(dataset["train"].features["label"].names)

        def sample_split(
            split_name: str, sample_size: int, shuffle_seed: int
        ) -> list[dict[str, Any]]:
            split = dataset[split_name]
            by_label: dict[int, list[int]] = {idx: [] for idx in range(len(label_names))}
            for source_index, example in enumerate(split):
                by_label[int(example["label"])].append(source_index)

            rng = random.Random(shuffle_seed)
            for indices in by_label.values():
                rng.shuffle(indices)
            label_order = list(by_label)
            rng.shuffle(label_order)

            picked: list[int] = []
            while len(picked) < min(sample_size, len(split)):
                progressed = False
                for label_idx in label_order:
                    if by_label[label_idx]:
                        picked.append(by_label[label_idx].pop())
                        progressed = True
                        if len(picked) >= sample_size:
                            break
                if not progressed:
                    break

            rng.shuffle(picked)
            rows: list[dict[str, Any]] = []
            for seed, source_index in enumerate(picked):
                example = split[source_index]
                rows.append(
                    {
                        "task_id": f"{split_name}:{seed}",
                        "rng_seed": seed,
                        "split": split_name,
                        "text": str(example["text"]),
                        "label": label_names[int(example["label"])],
                        "example_id": f"{split_name}:{seed}",
                    }
                )
            return rows

        rows = sample_split("train", self.train_sample, self.train_shuffle_seed)
        rows.extend(sample_split("test", self.test_sample, self.test_shuffle_seed))
        return label_names, rows

    def _canonical_label_lookup(self) -> dict[str, str]:
        if self._label_lookup is None:
            self._label_lookup = {label.lower(): label for label in self.labels_and_rows()[0]}
        return self._label_lookup

    def _openai_client(self) -> AsyncOpenAI:
        if self._openai is not None:
            return self._openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._openai = AsyncOpenAI(api_key=api_key, timeout=self.policy_timeout_seconds)
        return self._openai

    def _policy_semaphore(self) -> asyncio.Semaphore:
        if self._policy_sem is None:
            self._policy_sem = asyncio.Semaphore(max(1, self.policy_concurrency))
        return self._policy_sem

    @staticmethod
    def _utc_now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


task = Banking77ClassificationTask()


# --- 2. Container (wraps task) ---------------------------------------------

container = Container(
    "banking77-gepa-dev",
    runtime_id=Banking77ClassificationTask.RUNTIME_ID,
    description="Banking77 intent-classification prompt optimization container.",
    policy_ready=True,
)


@container.task_info
async def task_info() -> dict[str, Any]:
    return task.task_info_payload()


@container.program
async def program() -> dict[str, Any]:
    return task.program_payload()


@container.taskset
async def taskset() -> dict[str, Any]:
    return task.taskset_payload()


@container.taskset_tasks
async def taskset_tasks(payload: dict[str, Any]) -> dict[str, Any]:
    split = str(payload.get("split") or "train")
    task_ids = [str(task_id) for task_id in payload.get("task_ids") or []]
    return task.taskset_tasks_payload(split=split, task_ids=task_ids)


@container.rollout
async def rollout(payload: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.wait_for(task.run_rollout(payload), timeout=task.rollout_timeout_seconds)


app = container.fastapi(title="banking77-gepa-dev")


def serve_container(*, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


# --- 3. GEPA init ------------------------------------------------------------

DEV_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8765
HELDOUT_SEEDS = min(4, task.test_sample)
TRAIN_SEEDS = list(range(task.train_sample))
HELDOUT_SEED_LIST = list(range(HELDOUT_SEEDS))


@dataclass(frozen=True)
class GepaDevCompute:
    """Shared smoke budget for synth-optimizers and gepa-ai runs."""

    max_generations: int = 2
    proposals_per_generation: int = 4
    minibatch_size: int = 8
    max_total_rollouts: int = 240
    max_cost_usd: float = 0.0
    proposer_model: str = "gpt-5.4-nano"

    @property
    def max_metric_calls(self) -> int:
        return self.max_total_rollouts


GEPA_COMPUTE = GepaDevCompute()

CONTAINER_SCRIPT_PATH = Path(__file__).resolve()


@dataclass(frozen=True)
class PromptBenchmarkScores:
    train: float
    heldout: float


@dataclass(frozen=True)
class BackendRunSummary:
    backend: str
    run_id: str
    best_prompt: str
    reported_train: float | None
    reported_heldout: float | None
    budget_used: str
    manifest_path: str | None = None
    run_dir: str | None = None
    cost_usd: float | None = None


LOG_PREFIX_SYNTH = "[SYNTH]"
LOG_PREFIX_BERKELEY = "[BERKELEY]"
BACKEND_SUMMARY_FILENAME = "backend_summary.json"
COMPARE_PARALLEL_POLICY_CONCURRENCY = 4
BERKELEY_BOOT_GRACE_SECONDS = 8.0


# --- Synth GEPA run ---------------------------------------------------------


def _write_gepa_toml(
    *,
    port: int,
    run_id: str,
    output_dir: Path,
    compute: GepaDevCompute = GEPA_COMPUTE,
    policy_concurrency: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{run_id}_cache.sqlite"
    train_ids = ", ".join(json.dumps(f"train:{task_id}") for task_id in range(task.train_sample))
    heldout_ids = ", ".join(json.dumps(f"test:{task_id}") for task_id in range(HELDOUT_SEEDS))
    field = Banking77ClassificationTask.CLASSIFIER_FIELD
    concurrency = policy_concurrency if policy_concurrency is not None else task.policy_concurrency
    config_path = DEV_ROOT / "gepa.toml"
    config_path.write_text(
        f"""[run]
run_id = "{run_id}"
output_dir = "{output_dir.as_posix()}"
seed = 0

[container]
url = "http://127.0.0.1:{port}"
command = [
  "/usr/bin/env",
  "BANKING77_TRAIN_SAMPLE={task.train_sample}",
  "BANKING77_TEST_SAMPLE={task.test_sample}",
  "BANKING77_POLICY_CONCURRENCY={concurrency}",
  "BANKING77_POLICY_MODEL={task.policy_model}",
  "HF_HUB_DISABLE_PROGRESS_BARS=1",
  "HF_DATASETS_OFFLINE=1",
  {json.dumps(sys.executable)},
  {json.dumps(str(CONTAINER_SCRIPT_PATH))},
  "--serve",
  "--host", "127.0.0.1",
  "--port", "{port}",
]
startup_timeout_seconds = 60

[taskset]
train_split = "train"
heldout_split = "test"
train_ids = [{train_ids}]
heldout_ids = [{heldout_ids}]

[candidate]
target_modules = [{json.dumps(field)}]

[seed_candidate]
{field} = {json.dumps(Banking77ClassificationTask.SEED_CLASSIFIER_PROMPT)}

[policy]
provider = "openai"
model = "{task.policy_model}"
api_key_env = "OPENAI_API_KEY"

[proposer]
backend = "codex_app_server"
execution_mode = "local_process"
timeout_seconds = 900
model = "{compute.proposer_model}"
reasoning_effort = "medium"
auth_mode = "api_key"
copy_host_auth = false
api_key_env = "OPENAI_API_KEY"
sandbox_mode = "workspace-write"
approval_policy = "never"

[gepa]
max_generations = {compute.max_generations}
proposals_per_generation = {compute.proposals_per_generation}
minibatch_size = {compute.minibatch_size}
max_total_rollouts = {compute.max_total_rollouts}
max_cost_usd = {compute.max_cost_usd}

[cache]
mode = "readwrite"
path = "{cache_path.as_posix()}"
namespace = "{run_id}"
"""
    )
    return config_path


class GepaRunLogStream:
    """Streams GEPA dev run output and toggles live optimizer terminal events."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        verbose: bool = False,
        prefix: str = "",
    ) -> None:
        self._stream = stream or sys.stdout
        self.verbose = verbose
        self.prefix = prefix
        self._saved_terminal: str | None = None
        self._saved_terminal_detail: str | None = None

    def __enter__(self) -> GepaRunLogStream:
        self._saved_terminal = os.environ.get("SYNTH_OPTIMIZERS_TERMINAL")
        self._saved_terminal_detail = os.environ.get("SYNTH_OPTIMIZERS_TERMINAL_DETAIL")
        os.environ["SYNTH_OPTIMIZERS_TERMINAL"] = "1"
        if self.verbose:
            os.environ["SYNTH_OPTIMIZERS_TERMINAL_DETAIL"] = "debug"
        elif "SYNTH_OPTIMIZERS_TERMINAL_DETAIL" in os.environ:
            del os.environ["SYNTH_OPTIMIZERS_TERMINAL_DETAIL"]
        return self

    def __exit__(self, *_args: object) -> None:
        if self._saved_terminal is None:
            os.environ.pop("SYNTH_OPTIMIZERS_TERMINAL", None)
        else:
            os.environ["SYNTH_OPTIMIZERS_TERMINAL"] = self._saved_terminal
        if self._saved_terminal_detail is None:
            os.environ.pop("SYNTH_OPTIMIZERS_TERMINAL_DETAIL", None)
        else:
            os.environ["SYNTH_OPTIMIZERS_TERMINAL_DETAIL"] = self._saved_terminal_detail

    def write(self, line: str = "") -> None:
        if not line:
            print(file=self._stream)
            return
        if self.prefix:
            print(f"{self.prefix} {line}", file=self._stream)
        else:
            print(line, file=self._stream)

    @staticmethod
    def format_reward(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{100.0 * float(value):.1f}%"
        return "?"

    def run_started(self, *, run_id: str, train_last: int, heldout_last: int) -> None:
        self.write(f"Running {run_id} (train 0..{train_last}, heldout 0..{heldout_last}) ...")

    def run_complete(self, result: object, *, classifier_field: str) -> None:
        best = getattr(result, "best_candidate", None) or {}
        payload = best.get("payload") if isinstance(best.get("payload"), dict) else {}
        prompt = str(payload.get(classifier_field) or "")
        if len(prompt) > 120:
            prompt = prompt[:120] + "..."

        self.write()
        self.write("GEPA run complete")
        self.write(
            f"  best     {best.get('candidate_id', '?')}"
            f"  train={self.format_reward(best.get('train_reward'))}"
            f"  heldout={self.format_reward(best.get('heldout_reward'))}"
        )
        if prompt:
            self.write(f"  prompt   {prompt}")
        self.write(f"  manifest {getattr(result, 'manifest_path', '')}")
        self.write(f"  cost_usd {float(getattr(result, 'cost_usd', 0.0) or 0.0):.4f}")

    def backend_started(self, *, backend: str, run_id: str) -> None:
        self.write()
        self.write(f"=== {backend} ===")
        self.run_started(
            run_id=run_id,
            train_last=TRAIN_SEEDS[-1],
            heldout_last=HELDOUT_SEED_LIST[-1],
        )

    def gepa_ai_complete(self, result: object, *, classifier_field: str) -> None:
        best = getattr(result, "best_candidate", None) or {}
        prompt = ""
        if isinstance(best, dict):
            prompt = str(best.get(classifier_field) or "")
        elif isinstance(best, str):
            prompt = best
        if len(prompt) > 120:
            prompt = prompt[:120] + "..."

        val_scores = getattr(result, "val_aggregate_scores", None) or []
        best_idx = getattr(result, "best_idx", 0)
        val_score = val_scores[best_idx] if val_scores else None

        self.write()
        self.write("gepa-ai run complete")
        self.write(
            f"  best     idx={best_idx}"
            f"  val={self.format_reward(val_score)}"
            f"  metric_calls={getattr(result, 'total_metric_calls', '?')}"
        )
        if prompt:
            self.write(f"  prompt   {prompt}")
        run_dir = getattr(result, "run_dir", "")
        if run_dir:
            self.write(f"  run_dir  {run_dir}")

    def compare_backends(
        self,
        *,
        synth_summary: BackendRunSummary | dict[str, Any],
        gepa_ai_summary: BackendRunSummary | dict[str, Any],
        compute: GepaDevCompute,
        synth_scores: PromptBenchmarkScores,
        gepa_ai_scores: PromptBenchmarkScores,
    ) -> None:
        gepa_val = (
            gepa_ai_summary.get("reported_heldout")
            if isinstance(gepa_ai_summary, dict)
            else gepa_ai_summary.reported_heldout
        )
        synth_train = (
            synth_summary.get("reported_train")
            if isinstance(synth_summary, dict)
            else synth_summary.reported_train
        )
        synth_heldout = (
            synth_summary.get("reported_heldout")
            if isinstance(synth_summary, dict)
            else synth_summary.reported_heldout
        )
        synth_budget = (
            synth_summary.get("budget_used", "?")
            if isinstance(synth_summary, dict)
            else synth_summary.budget_used
        )
        gepa_budget = (
            gepa_ai_summary.get("budget_used", "?")
            if isinstance(gepa_ai_summary, dict)
            else gepa_ai_summary.budget_used
        )
        synth_manifest = (
            synth_summary.get("manifest_path", "?")
            if isinstance(synth_summary, dict)
            else synth_summary.manifest_path
        )
        gepa_run_dir = (
            gepa_ai_summary.get("run_dir", "?")
            if isinstance(gepa_ai_summary, dict)
            else gepa_ai_summary.run_dir
        )
        rows = [
            (
                "synth-optimizers",
                synth_train,
                synth_heldout,
                synth_scores.train,
                synth_scores.heldout,
                synth_budget,
            ),
            (
                "gepa-ai",
                "—",
                gepa_val,
                gepa_ai_scores.train,
                gepa_ai_scores.heldout,
                gepa_budget,
            ),
        ]
        headers = (
            "backend",
            "reported_train",
            "reported_heldout",
            "rescore_train",
            "rescore_heldout",
            "budget_used",
        )

        def cell_text(cell: object) -> str:
            if cell is None:
                return "?"
            if isinstance(cell, (int, float)):
                return self.format_reward(cell)
            return str(cell)

        widths = [
            max(len(cell_text(row[col_idx])) for row in rows + [headers])
            for col_idx in range(len(headers))
        ]

        def fmt_row(cells: tuple[object, ...]) -> str:
            rendered = [cell_text(cells[0]).ljust(widths[0])]
            for idx, cell in enumerate(cells[1:], start=1):
                rendered.append(cell_text(cell).rjust(widths[idx]))
            return "  " + "  ".join(rendered)

        self.write()
        self.write("Banking77 GEPA comparison")
        self.write(
            f"  budget  max_generations={compute.max_generations}"
            f"  proposals_per_generation={compute.proposals_per_generation}"
            f"  minibatch_size={compute.minibatch_size}"
            f"  max_rollouts={compute.max_total_rollouts}"
        )

        self.write(fmt_row(headers))
        self.write(fmt_row(tuple("-" * widths[i] for i in range(len(headers)))))
        for row in rows:
            self.write(fmt_row(row))

        heldout_delta = (gepa_ai_scores.heldout - synth_scores.heldout) * 100.0
        self.write(f"  heldout delta (rescore)  gepa-ai - synth = {heldout_delta:+.1f} pp")
        self.write(f"  synth manifest           {synth_manifest or '?'}")
        self.write(f"  gepa-ai run_dir          {gepa_run_dir or '?'}")


def run_synth_gepa_dev(
    *,
    port: int,
    run_id: str,
    output_dir: Path,
    log: GepaRunLogStream | None = None,
    compute: GepaDevCompute = GEPA_COMPUTE,
    policy_concurrency: int | None = None,
) -> object:
    effective_policy_concurrency = policy_concurrency or task.policy_concurrency
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{run_id}_cache.sqlite"
    if log is not None:
        log.backend_started(backend="synth-optimizers GEPA", run_id=run_id)
    with container.serve(port=port, startup_timeout_seconds=60) as handle:
        config = GepaConfig(
            container=handle.connection(),
            run=RunSettings(run_id=run_id, output_dir=output_dir, seed=0),
            taskset=TasksetSelection(
                train_split="train",
                heldout_split="test",
                train_ids=[f"train:{task_id}" for task_id in TRAIN_SEEDS],
                heldout_ids=[f"test:{task_id}" for task_id in HELDOUT_SEED_LIST],
            ),
            program=None,
            objectives=None,
            policy=None,
            proposer=ProposerConfig(
                model=compute.proposer_model,
                reasoning_effort="medium",
                auth_mode="api_key",
                api_key_env="OPENAI_API_KEY",
                copy_host_auth=False,
                timeout_seconds=900,
                sandbox_mode="workspace-write",
                approval_policy="never",
            ),
            budgets=GepaBudgetConfig(
                max_generations=compute.max_generations,
                proposals_per_generation=compute.proposals_per_generation,
                minibatch_size=compute.minibatch_size,
                max_total_rollouts=compute.max_total_rollouts,
            ),
            budget=BudgetConfig(max_cost_usd=compute.max_cost_usd),
            pipeline=GepaPipeline(
                rollout_transport="async",
                candidate_concurrency=compute.proposals_per_generation,
                rollout_concurrency=effective_policy_concurrency,
            ),
            cache=CacheConfig(
                mode="readwrite",
                path=cache_path,
                namespace=run_id,
            ),
        )
        result = OptimizerRun(config).execute()
    if log is not None:
        log.run_complete(result, classifier_field=Banking77ClassificationTask.CLASSIFIER_FIELD)
    return result


def summarize_synth_result(
    *, result: object, run_id: str, compute: GepaDevCompute
) -> BackendRunSummary:
    best = getattr(result, "best_candidate", None) or {}
    payload = best.get("payload") if isinstance(best.get("payload"), dict) else {}
    field = Banking77ClassificationTask.CLASSIFIER_FIELD
    return BackendRunSummary(
        backend="synth-optimizers",
        run_id=run_id,
        best_prompt=str(payload.get(field) or Banking77ClassificationTask.SEED_CLASSIFIER_PROMPT),
        reported_train=best.get("train_reward"),
        reported_heldout=best.get("heldout_reward"),
        budget_used=f"rollouts<={compute.max_total_rollouts}",
        manifest_path=str(getattr(result, "manifest_path", "") or ""),
        cost_usd=float(getattr(result, "cost_usd", 0.0) or 0.0),
    )


def _write_backend_summary(path: Path, summary: BackendRunSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True))


def _read_backend_summary(path: Path) -> BackendRunSummary:
    payload = json.loads(path.read_text())
    return BackendRunSummary(**payload)


def _compare_policy_concurrency(*, compare_parallel: bool) -> int | None:
    if not compare_parallel:
        return None
    return COMPARE_PARALLEL_POLICY_CONCURRENCY


def run_worker_cli(
    *,
    run_id: str,
    output_dir: Path,
    port: int,
    compare_parallel: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / BACKEND_SUMMARY_FILENAME
    policy_concurrency = _compare_policy_concurrency(compare_parallel=compare_parallel)
    os.environ["SYNTH_OPTIMIZERS_TERMINAL"] = "1"
    with GepaRunLogStream(prefix=LOG_PREFIX_SYNTH) as log:
        result = run_synth_gepa_dev(
            port=port,
            run_id=run_id,
            output_dir=output_dir,
            log=log,
            policy_concurrency=policy_concurrency,
        )
    summary = summarize_synth_result(result=result, run_id=run_id, compute=GEPA_COMPUTE)
    _write_backend_summary(summary_path, summary)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking77 synth-optimizers GEPA dev")
    parser.add_argument("--serve", action="store_true", help="Run the HTTP container only.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream full GEPA logs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--compare-parallel", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.run_id or not args.output_dir:
            raise SystemExit("--worker requires --run-id and --output-dir")
        raise SystemExit(
            run_worker_cli(
                run_id=args.run_id,
                output_dir=Path(args.output_dir),
                port=args.port,
                compare_parallel=args.compare_parallel,
            )
        )
    if args.serve:
        serve_container(host=args.host, port=args.port)
        return
    run_id = f"banking77_dev_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_synth"
    with GepaRunLogStream(verbose=args.verbose, prefix=LOG_PREFIX_SYNTH):
        run_synth_gepa_dev(
            port=args.port,
            run_id=run_id,
            output_dir=DEV_ROOT / "runs" / run_id,
        )


if __name__ == "__main__":
    main()
