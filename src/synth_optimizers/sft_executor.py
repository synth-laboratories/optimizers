"""In-repository Tinker SFT executor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .contracts.checkpoint_plan import resolve_checkpoint_plan
from .contracts.training_schemas import (
    SFT_ALGORITHM_ID,
    SFT_IMPLEMENTATION_VERSION,
    TERMINAL_STATES,
    validate_dataset_manifest,
)
from .providers.protocols import (
    SFT_REQUIRED_CAPABILITIES,
    ProviderError,
    ProviderSession,
    TrainingProvider,
    TrainingStepRequest,
    UnsupportedCapability,
)
from .providers.tinker.client import TinkerAdapter, TinkerCredentials, new_request_id
from .providers.tinker.fake import FakeTinkerProvider
from .runtime import RUNNER_VERSION, JobStore, JobStoreError, TrainingJob, digest_payload, idempotency_key
from .sft_dataset import (
    DatasetError,
    Example,
    SplitDataset,
    split_dataset_from_config,
)
from .training_eval import (
    encode_example,
    eval_max_tokens,
    evaluate_checkpoint,
    paired_uplift,
    public_evaluation,
    system_prompt_from,
)


class SftExecutor(Protocol):
    def estimate(self, config: Mapping[str, Any]) -> dict[str, Any]: ...
    def submit(self, config: Mapping[str, Any], *, job_id: str | None = None) -> dict[str, Any]: ...
    def status(self, job_id: str) -> dict[str, Any]: ...
    def cancel(self, job_id: str) -> dict[str, Any]: ...
    def resume(self, job_id: str) -> dict[str, Any]: ...


class TinkerSftExecutor:
    def __init__(
        self,
        store: JobStore,
        provider: TrainingProvider,
        *,
        owner: str = "sft-worker",
        sync: bool = True,
    ) -> None:
        self.store = store
        self.provider = provider
        self.owner = owner
        self.sync = sync

    @classmethod
    def local(cls, store: JobStore, *, fixture: bool = False) -> "TinkerSftExecutor":
        if fixture:
            return cls(store, TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=FakeTinkerProvider()))
        return cls(store, TinkerAdapter(TinkerCredentials.from_env()))

    def estimate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        resolve_checkpoint_plan(config)
        dataset = self._dataset(config)
        steps = _positive_int(config.get("training", {}).get("steps") or config.get("max_steps") or 2, "steps")
        batch_size = _positive_int(config.get("training", {}).get("batch_size") or 1, "batch_size")
        prompt = system_prompt_from(config)
        tokens = sum(
            encode_example(self.provider, example, system_prompt=prompt)["n_tokens"]
            for example in dataset.train[:batch_size]
        )
        return {
            "algorithm_id": SFT_ALGORITHM_ID,
            "implementation_version": SFT_IMPLEMENTATION_VERSION,
            "model_id": self.provider.resolve_model(str(config.get("base_model") or config.get("model_id") or "")),
            "train_examples": len(dataset.train),
            "estimated_training_tokens": tokens * steps,
            "cost_usd": None,
            "cost_missing": True,
        }

    def submit(
        self,
        config: Mapping[str, Any],
        *,
        job_id: str | None = None,
        idempotency_key_override: str | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare(
            config,
            job_id=job_id,
            idempotency_key_override=idempotency_key_override,
        )
        if prepared.state in TERMINAL_STATES or prepared.state == "running":
            return self.status(prepared.job_id)
        if self.sync:
            return self._run(prepared.job_id)
        from .runtime import start_job_worker

        start_job_worker(prepared.job_id, lambda: self._run(prepared.job_id))
        return self.status(prepared.job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        job = self.store.require(job_id)
        events = self.store.events(job_id, after_sequence=0, limit=5_000)
        return _public_status(job, events)

    def cancel(self, job_id: str) -> dict[str, Any]:
        self.store.request_cancel(job_id)
        return self.status(job_id)

    def resume(self, job_id: str) -> dict[str, Any]:
        job = self.store.require(job_id)
        if job.state in TERMINAL_STATES:
            return self.status(job_id)
        if self.sync:
            return self._run(job_id)
        from .runtime import start_job_worker
        start_job_worker(job_id, lambda: self._run(job_id))
        return self.status(job_id)

    def _prepare(
        self,
        config: Mapping[str, Any],
        *,
        job_id: str | None,
        idempotency_key_override: str | None = None,
    ) -> TrainingJob:
        plan = resolve_checkpoint_plan(config)
        dataset = self._dataset(config)
        validate_dataset_manifest(dataset.manifest)
        model_id = self.provider.resolve_model(
            str(config.get("base_model") or config.get("model_id") or "openai/gpt-oss-20b")
        )
        training = dict(config.get("training") or {})
        training["steps"] = plan["steps"]
        training.setdefault("batch_size", 1)
        training.setdefault("learning_rate", 2e-5)
        training.setdefault("checkpoint_every_steps", int((config.get("checkpoint_steps") or [training["steps"]])[0]))
        training.setdefault("eval_every_steps", training["checkpoint_every_steps"])
        seed = int(config.get("seed") or 0)
        generated_key = idempotency_key(
            algorithm_id=SFT_ALGORITHM_ID,
            implementation_version=SFT_IMPLEMENTATION_VERSION,
            provider="tinker",
            model_id=model_id,
            dataset_digest=str(dataset.manifest["digest"]),
            split_manifest_digest=digest_payload(dataset.manifest["split_digests"]),
            renderer_version=dataset.renderer_version,
            training_config={"training": training, "checkpoint_plan": plan,
                             "evaluation": config.get("evaluation", {}),
                             "system_prompt": system_prompt_from(config)},
            reward_version="sft.cross_entropy.v1",
            seed=seed,
            runner_version=str(config.get("runner_version") or RUNNER_VERSION),
            repeat_index=int(config.get("repeat_index") or 0),
        )
        key = idempotency_key_override or generated_key
        snapshot = {
            **dict(config),
            "base_model": model_id,
            "model_id": model_id,
            "backend": "tinker",
            "training": training,
            "dataset_manifest": dataset.manifest,
            "resolved_checkpoint_plan": plan,
            "seed": seed,
        }
        job = self.store.persist_prepared(
            algorithm_id=SFT_ALGORITHM_ID,
            implementation_version=SFT_IMPLEMENTATION_VERSION,
            provider="tinker",
            model_id=model_id,
            idempotency_key=key,
            config=snapshot,
            job_id=job_id or str(config.get("run_id") or ""),
        )
        return job

    def _dataset(self, config: Mapping[str, Any]) -> SplitDataset:
        return split_dataset_from_config(config)

    def _run(self, job_id: str) -> dict[str, Any]:
        from .runtime.worker import execute_owned
        try:
            result = execute_owned(self.store, job_id, self._execute)
            return result or self.status(job_id)
        except JobStoreError:
            return self.status(job_id)

    def _execute(self, job: TrainingJob, owner: str) -> dict[str, Any]:
        from copy import copy
        from .runtime.worker import AdmissionProvider
        executor = copy(self)
        from .runtime.operations import DurableProvider, UncertainOperation
        executor.provider = AdmissionProvider(
            DurableProvider(self.provider, self.store, job.job_id, owner), self.store, job.job_id)
        try:
            executor.provider.provider.recovery_checkpoint()
            result = executor._execute_body(job, owner)
            if self.store.require(job.job_id).state == "stop_requested":
                self.store.transition(job.job_id, "cancelled")
                return self.status(job.job_id)
            return result
        except UncertainOperation as exc:
            self.store.transition(job.job_id, "blocked_uncertain", error=str(exc))
            return self.status(job.job_id)
        except ProviderError as exc:
            return executor._fail(job.job_id, str(exc))

    def _execute_body(self, job: TrainingJob, owner: str) -> dict[str, Any]:
        job_id = job.job_id
        config = json.loads(job.config_json)
        if job.resume_token and not job.resume_token.startswith("{"):
            from .runtime.operations import UncertainOperation
            raise UncertainOperation("legacy resume has no verified operation journal; explicit migration required")
        dataset = self._dataset(config)
        if config.get("dataset_manifest") != dataset.manifest:
            from .runtime.operations import UncertainOperation
            raise UncertainOperation("dataset identity changed; exact resume refused")
        self.store.append_event_once(
            job_id,
            "sft.dataset.validated",
            {"manifest": dataset.manifest, "labels": list(dataset.labels)},
            phase="prepared",
        )
        try:
            capabilities = self.provider.discover_capabilities(job.model_id)
            capabilities.require(SFT_REQUIRED_CAPABILITIES if config["resolved_checkpoint_plan"]["mode"] != "none"
                                 else frozenset({"sft.train"}))
        except UnsupportedCapability as exc:
            return self._fail(job_id, str(exc))
        session = self.provider.create_session(
            job.model_id,
            rank=int(config.get("rank") or 8),
            seed=int(config.get("seed") or 0),
            request_id=new_request_id(job_id, "session"),
        )
        self.store.append_event_once(
            job_id,
            "sft.training.started",
            {"model_id": job.model_id, "session_id": session.session_id},
            phase="running",
        )
        training = config["training"]
        prompt = system_prompt_from(config)
        max_tokens = eval_max_tokens(config)
        start_step = 1  # Confirmed provider operations replay from durable results.
        checkpoints: list[dict[str, Any]] = []
        plan = config["resolved_checkpoint_plan"]
        try:
            baseline, baseline_record = {}, {}
            if plan["mode"] != "none":
                baseline_checkpoint = self.provider.save_checkpoint(
                    session,
                    step=0,
                    kind="inference",
                    request_id=new_request_id(job_id, "baseline", "inference"),
                )
                baseline_record = {
                    "checkpoint_id": baseline_checkpoint.checkpoint_id,
                    "provider_reference": baseline_checkpoint.provider_reference,
                    "digest": baseline_checkpoint.digest,
                    "step": 0,
                }
                baseline = self._evaluate(
                    job_id,
                    baseline_record,
                    dataset.calibration,
                    phase="selection",
                    candidate="base",
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                self.store.append_event_once(
                    job_id,
                    "sft.baseline_eval.completed",
                    {**baseline_record, **public_evaluation(baseline), "role": "selection"},
                    phase="running",
                )
            for step in range(start_step, int(training["steps"]) + 1):
                if self.store.cancellation_requested(job_id):
                    return self._fail(job_id, "training cancellation requested")
                self.store.heartbeat(job_id, owner)
                batch = _batch(dataset.train, step, int(training["batch_size"]))
                data = [
                    encode_example(self.provider, example, system_prompt=prompt) for example in batch
                ]
                result = self.provider.train_step(
                    session,
                    TrainingStepRequest(
                        request_id=new_request_id(job_id, "train", str(step)),
                        loss_name="cross_entropy",
                        data=tuple(data),
                        metadata={"learning_rate": float(training.get("learning_rate") or 2e-5)},
                    ),
                )
                self.store.append_event_once(
                    job_id,
                    "sft.step.metrics",
                    {"step": step, "metrics": dict(result.metrics), "tokens": sum(item["n_tokens"] for item in data)},
                    phase="running",
                )
                self._receipt(job_id, result.request_id, result.usage)
                if step in plan["save_steps"]:
                    checkpoint = self._checkpoint_and_eval(
                        job_id,
                        session,
                        dataset,
                        step,
                        checkpoints,
                        baseline=baseline,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                    self.store.set_resume_token(job_id, json.dumps({"step": step, "training_provider_reference": checkpoint["training_provider_reference"]}))
            if plan["mode"] == "none":
                promoted = checkpoints[-1]
                heldout = {"status": "not_configured", "accuracy": None, "evaluated": False}
                self.store.append_event_once(job_id, "sft.checkpoint.selected", promoted, phase="materializing")
            else:
                promoted = max((item for item in checkpoints if "calibration_accuracy" in item), key=lambda item: item["calibration_accuracy"])
                self.store.append_event_once(
                    job_id,
                    "sft.checkpoint.promoted",
                    promoted,
                    phase="evaluating",
                )
                self.store.transition(job_id, "evaluating")
                heldout_base = self._evaluate(
                    job_id,
                    baseline_record,
                    dataset.heldout,
                    phase="heldout",
                    candidate="base",
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                heldout_trained = self._evaluate(
                    job_id,
                    promoted,
                    dataset.heldout,
                    phase="heldout",
                    candidate="selected",
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                heldout_uplift = self._paired(config, heldout_base, heldout_trained)
                heldout = {
                    **public_evaluation(heldout_trained),
                    "role": "heldout",
                    "heldout_locked": True,
                    "baseline": public_evaluation(heldout_base),
                    "trained": public_evaluation(heldout_trained),
                    "paired_uplift": heldout_uplift,
                }
                self.store.append_event_once(
                    job_id,
                    "sft.heldout_eval.completed",
                    heldout,
                    phase="evaluating",
                )
            self.store.transition(job_id, "materializing")
            bundle = {
                "schema_version": "policy_bundle.v1",
                "algorithm_id": SFT_ALGORITHM_ID,
                "implementation_version": SFT_IMPLEMENTATION_VERSION,
                "model_id": job.model_id,
                "checkpoint_id": promoted["checkpoint_id"],
                "provider_reference": promoted["provider_reference"],
                "heldout": heldout,
            }
            digest = self.store.put_artifact(
                job_id, "policy_bundle.json", json.dumps(bundle, sort_keys=True).encode(), content_type="application/json"
            )
            self.store.append_event_once(
                job_id,
                "sft.model.materialized",
                {"digest": digest, "checkpoint_id": promoted["checkpoint_id"]},
                phase="materializing",
            )
            self.store.append_event_once(
                job_id,
                "sft.completed",
                {"selected_checkpoint_id": promoted["checkpoint_id"], "heldout_accuracy": heldout["accuracy"]},
                phase="completed",
            )
            self.store.transition(job_id, "completed")
        except ProviderError as exc:
            from .runtime.operations import UncertainOperation
            if isinstance(exc, UncertainOperation):
                raise
            return self._fail(job_id, str(exc))
        return self.status(job_id)

    def _checkpoint_and_eval(
        self,
        job_id: str,
        session: ProviderSession,
        dataset: SplitDataset,
        step: int,
        checkpoints: list[dict[str, Any]],
        *,
        baseline: Mapping[str, Any],
        prompt: str | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        training = self.provider.save_checkpoint(
            session, step=step, kind="training", request_id=new_request_id(job_id, "ckpt", str(step), "training")
        )
        inference = self.provider.save_checkpoint(
            session, step=step, kind="inference", request_id=new_request_id(job_id, "ckpt", str(step), "inference")
        )
        record = {
            "checkpoint_id": inference.checkpoint_id,
            "training_checkpoint_id": training.checkpoint_id,
            "provider_reference": inference.provider_reference,
            "training_provider_reference": training.provider_reference,
            "training_digest": training.digest,
            "digest": inference.digest,
            "resume_token": training.resume_token,
            "step": step,
        }
        self.store.append_event_once(job_id, "sft.checkpoint.created", record, phase="evaluating")
        plan = json.loads(self.store.require(job_id).config_json)["resolved_checkpoint_plan"]
        if step not in plan["evaluation_steps"]:
            record["evaluation_status"] = "not_configured" if plan["mode"] == "none" else "not_scheduled"
            checkpoints.append(record)
            return record
        evaluation = self._evaluate(
            job_id,
            record,
            dataset.calibration,
            phase="selection",
            candidate=f"checkpoint:{step}",
            prompt=prompt,
            max_tokens=max_tokens,
        )
        payload = {
            **record,
            **public_evaluation(evaluation),
            "calibration_accuracy": evaluation["accuracy"],
            "role": "selection",
            "paired_uplift": self._paired(
                json.loads(self.store.require(job_id).config_json), baseline, evaluation
            ),
        }
        self.store.append_event_once(job_id, "sft.checkpoint_eval.completed", payload, phase="evaluating")
        checkpoints.append(payload)
        return payload

    def _evaluate(
        self,
        job_id: str,
        checkpoint: Mapping[str, Any],
        examples: Sequence[Example],
        *,
        phase: str,
        candidate: str,
        prompt: str | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        def stream(record: Mapping[str, Any]) -> None:
            self.store.append_event_once(
                job_id,
                "sft.evaluation.example.completed",
                {
                    **record,
                    "evaluation_id": f"{phase}:{candidate}",
                    "role": phase,
                    "phase": phase,
                    "candidate": candidate,
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "step": checkpoint.get("step", 0),
                    "score": record["cumulative_accuracy"],
                    "sample_count": record["completed"],
                    "metric": "accuracy",
                    "status": "running" if record["completed"] != record["total"] else "completed",
                },
                phase="evaluating" if phase == "heldout" else "running",
            )

        return evaluate_checkpoint(
            self.provider,
            checkpoint,
            examples,
            system_prompt=prompt,
            max_tokens=max_tokens,
            on_example=stream,
            on_usage=lambda request_id, usage: self._receipt(job_id, request_id, usage),
        )

    @staticmethod
    def _paired(
        config: Mapping[str, Any], baseline: Mapping[str, Any], challenger: Mapping[str, Any]
    ) -> dict[str, Any]:
        evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), Mapping) else {}
        return paired_uplift(
            baseline,
            challenger,
            confidence=float(evaluation.get("confidence") or 0.95),
            bootstrap_resamples=int(evaluation.get("bootstrap_resamples") or 4_000),
            seed=int(config.get("seed") or 20260907),
            minimum_claim_uplift=float(evaluation.get("minimum_claim_uplift") or 0.01),
            minimum_paired_examples=int(evaluation.get("minimum_paired_examples") or 100),
        )

    def _receipt(self, job_id: str, request_id: str, usage: Any) -> None:
        self.store.put_receipt(
            job_id,
            request_id,
            {
                "schema_version": "training.usage_receipt.v1",
                "provider": "tinker",
                "request_id": request_id,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "training_tokens": usage.training_tokens,
                "cost_usd": usage.cost_usd,
                "cost_missing": usage.cost_missing,
                "algorithm_id": SFT_ALGORITHM_ID,
                "implementation_version": SFT_IMPLEMENTATION_VERSION,
            },
        )

    def _fail(self, job_id: str, reason: str) -> dict[str, Any]:
        if self.store.cancellation_requested(job_id):
            self.store.transition(job_id, "cancelled")
            return self.status(job_id)
        self.store.append_event_once(job_id, "sft.failed", {"reason": reason}, phase="failed")
        self.store.transition(job_id, "failed", error=reason)
        return self.status(job_id)


def _batch(examples: Sequence[Example], step: int, size: int) -> list[Example]:
    start = ((step - 1) * size) % len(examples)
    return [examples[(start + offset) % len(examples)] for offset in range(size)]


def _resume_step(job: TrainingJob) -> int:
    if not job.resume_token:
        return 0
    try:
        progress = json.loads(job.resume_token)
        step = progress["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("invalid progress")
        return step
    except (ValueError, TypeError, KeyError) as exc:
        raise JobStoreError("legacy opaque resume token has no verified progress") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetError(f"{field} must be a positive integer")
    return value


def _public_status(job: TrainingJob, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": job.job_id,
        "job_id": job.job_id,
        "algorithm": job.algorithm_id,
        "status": job.state if job.state != "completed" else "completed",
        "error": job.error,
        "events": list(events),
        "events_url": f"/v1/runs/{job.job_id}/optimizer-events",
        "events_stream_url": f"/v1/runs/{job.job_id}/optimizer-events/stream",
        "status_url": f"/v1/runs/{job.job_id}",
        "artifact_base_url": f"/v1/runs/{job.job_id}/artifacts",
    }
