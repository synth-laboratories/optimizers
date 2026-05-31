"""Banking77 task + gepa-ai GEPA dev run (in-process adapter, same prompts/taskset)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from banking77_synth_gepa_dev import (
    BACKEND_SUMMARY_FILENAME,
    Banking77ClassificationTask,
    COMPARE_PARALLEL_POLICY_CONCURRENCY,
    DEFAULT_PORT,
    DEV_ROOT,
    GEPA_COMPUTE,
    GepaDevCompute,
    HELDOUT_SEEDS,
    HELDOUT_SEED_LIST,
    LOG_PREFIX_BERKELEY,
    TRAIN_SEEDS,
    BackendRunSummary,
    PromptBenchmarkScores,
    serve_container,
    task,
)


class _PrefixedGepaLogger:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def log(self, message: str) -> None:
        for line in message.splitlines() or [message]:
            if line:
                print(f"{self._prefix} {line}", flush=True)


class Banking77GepaAiEvaluator:
    def __init__(self, banking_task: Banking77ClassificationTask) -> None:
        self._task = banking_task

    def __call__(self, data: dict[str, str], response: str) -> object:
        from gepa.adapters.default_adapter.default_adapter import EvaluationResult

        expected = str(data["answer"])
        prediction = self._task.normalize_label(response)
        score = 1.0 if prediction == expected else 0.0
        if score > 0:
            feedback = f"Correct label '{expected}'."
        else:
            feedback = (
                f"Predicted '{prediction}' but expected '{expected}'. "
                "Return exactly one canonical Banking77 label."
            )
        return EvaluationResult(score=score, feedback=feedback)


def _gepa_ai_dataset(*, split: str) -> list[dict[str, str]]:
    rows = [row for row in task.labels_and_rows()[1] if row["split"] == split]
    return [
        {
            "input": task.format_user_prompt(str(row["text"])),
            "additional_context": {"example_id": str(row["example_id"])},
            "answer": str(row["label"]),
        }
        for row in sorted(rows, key=lambda row: str(row["task_id"]))
    ]


def _openai_litellm_model(model: str) -> str:
    return model if model.startswith("openai/") else f"openai/{model}"


def _best_prompt_from_gepa_ai(result: object, *, classifier_field: str) -> str:
    best = getattr(result, "best_candidate", None) or {}
    if isinstance(best, dict):
        return str(best.get(classifier_field) or "")
    if isinstance(best, str):
        return best
    return ""


def benchmark_prompt(prompt: str) -> PromptBenchmarkScores:
    train = task.score_classifier_prompt_sync(
        prompt,
        split="train",
        task_ids=[f"train:{task_id}" for task_id in TRAIN_SEEDS],
    )
    heldout = task.score_classifier_prompt_sync(
        prompt,
        split="test",
        task_ids=[f"test:{task_id}" for task_id in HELDOUT_SEED_LIST],
    )
    return PromptBenchmarkScores(train=train, heldout=heldout)


def run_gepa_ai_core(
    *,
    run_id: str,
    output_dir: Path,
    compute: GepaDevCompute = GEPA_COMPUTE,
    log_prefix: str = "",
    policy_concurrency: int | None = None,
) -> object:
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
    from gepa.logging.logger import StdOutLogger

    output_dir.mkdir(parents=True, exist_ok=True)
    field = Banking77ClassificationTask.CLASSIFIER_FIELD
    workers = policy_concurrency if policy_concurrency is not None else task.policy_concurrency
    if log_prefix:
        logger: Any = _PrefixedGepaLogger(log_prefix)
    else:
        logger = StdOutLogger()
    result = gepa.optimize(
        seed_candidate={field: Banking77ClassificationTask.SEED_CLASSIFIER_PROMPT},
        trainset=_gepa_ai_dataset(split="train"),
        valset=_gepa_ai_dataset(split="test")[:HELDOUT_SEEDS],
        adapter=DefaultAdapter(
            model=_openai_litellm_model(task.policy_model),
            evaluator=Banking77GepaAiEvaluator(task),
            max_litellm_workers=max(1, workers),
        ),
        reflection_lm=_openai_litellm_model(compute.proposer_model),
        reflection_minibatch_size=compute.minibatch_size,
        max_metric_calls=compute.max_metric_calls,
        run_dir=str(output_dir),
        seed=0,
        display_progress_bar=False,
        logger=logger,
    )
    return result


def summarize_gepa_ai_result(
    *, result: object, run_id: str, compute: GepaDevCompute
) -> BackendRunSummary:
    field = Banking77ClassificationTask.CLASSIFIER_FIELD
    val_scores = getattr(result, "val_aggregate_scores", None) or []
    best_idx = getattr(result, "best_idx", 0)
    val_score = val_scores[best_idx] if val_scores else None
    return BackendRunSummary(
        backend="gepa-ai",
        run_id=run_id,
        best_prompt=_best_prompt_from_gepa_ai(result, classifier_field=field),
        reported_train=None,
        reported_heldout=val_score,
        budget_used=f"metric_calls={getattr(result, 'total_metric_calls', compute.max_metric_calls)}",
        run_dir=str(getattr(result, "run_dir", "") or ""),
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


def _format_reward(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{100.0 * float(value):.1f}%"
    return "?"


def run_worker_cli(
    *,
    run_id: str,
    output_dir: Path,
    compare_parallel: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / BACKEND_SUMMARY_FILENAME
    policy_concurrency = _compare_policy_concurrency(compare_parallel=compare_parallel)
    print(
        f"{LOG_PREFIX_BERKELEY} Running {run_id} (train 0..{TRAIN_SEEDS[-1]}, heldout 0..{HELDOUT_SEED_LIST[-1]}) ..."
    )
    result = run_gepa_ai_core(
        run_id=run_id,
        output_dir=output_dir,
        log_prefix=LOG_PREFIX_BERKELEY,
        policy_concurrency=policy_concurrency,
    )
    summary = summarize_gepa_ai_result(result=result, run_id=run_id, compute=GEPA_COMPUTE)
    best = summary.best_prompt
    if len(best) > 120:
        best = best[:120] + "..."
    print()
    print(f"{LOG_PREFIX_BERKELEY} gepa-ai run complete")
    print(
        f"{LOG_PREFIX_BERKELEY}   best     val={_format_reward(summary.reported_heldout)}"
        f"  budget={summary.budget_used}"
    )
    if best:
        print(f"{LOG_PREFIX_BERKELEY}   prompt   {best}")
    if summary.run_dir:
        print(f"{LOG_PREFIX_BERKELEY}   run_dir  {summary.run_dir}")
    _write_backend_summary(summary_path, summary)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking77 gepa-ai GEPA dev")
    parser.add_argument("--serve", action="store_true", help="Run the HTTP container only.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream full gepa-ai logs.")
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
                compare_parallel=args.compare_parallel,
            )
        )
    if args.serve:
        serve_container(host=args.host, port=args.port)
        return
    run_id = f"banking77_dev_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_gepa_ai"
    prefix = LOG_PREFIX_BERKELEY if args.verbose else ""
    print(f"{LOG_PREFIX_BERKELEY} Running {run_id} ...")
    result = run_gepa_ai_core(
        run_id=run_id,
        output_dir=DEV_ROOT / "runs" / run_id,
        log_prefix=prefix,
    )
    summary = summarize_gepa_ai_result(result=result, run_id=run_id, compute=GEPA_COMPUTE)
    print(
        f"{LOG_PREFIX_BERKELEY} done  val={_format_reward(summary.reported_heldout)}  {summary.budget_used}"
    )


if __name__ == "__main__":
    main()
