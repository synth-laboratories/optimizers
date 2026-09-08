"""Closed-loop ``cispo.slime.v1`` executor on the shared Tinker adapter."""

from __future__ import annotations

import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from typing import Any

from .cispo import (
    ALGORITHM_ID,
    IMPLEMENTATION_VERSION,
    CispoConfig,
    CispoError,
    group_advantages,
    importance_ratios,
    is_zero_advantage_group,
    objective,
    ppo_kl_from_ratio,
)
from .contracts.training_schemas import CISPO_IMPLEMENTATION, TERMINAL_STATES, validate_cispo_request
from .providers.protocols import (
    CISPO_REQUIRED_CAPABILITIES,
    ForwardRequest,
    ProviderCheckpoint,
    ProviderError,
    ProviderSession,
    SampleRequest,
    TrainingProvider,
    TrainingStepRequest,
    UnsupportedCapability,
)
from .providers.tinker.client import TinkerAdapter, TinkerCredentials, new_request_id
from .providers.tinker.fake import FakeTinkerProvider
from .providers.tinker.tokenize import extract_final_label
from .runtime import RUNNER_VERSION, JobStore, JobStoreError, TrainingJob, digest_payload, idempotency_key
from .sft_dataset import Example, SplitDataset, split_dataset_from_config
from .sft_executor import _public_status
from .training_eval import (
    encode_example,
    eval_max_tokens,
    evaluate_checkpoint,
    paired_uplift,
    public_evaluation,
    sample_parallelism,
    system_prompt_from,
)


class TinkerCispoExecutor:
    def __init__(
        self,
        store: JobStore,
        provider: TrainingProvider,
        *,
        owner: str = "cispo-worker",
        sync: bool = True,
        allow_unvalidated_canary: bool = False,
    ) -> None:
        self.store = store
        self.provider = self._require_adapter(provider)
        self.owner = owner
        self.sync = sync
        self.allow_unvalidated_canary = allow_unvalidated_canary

    @staticmethod
    def _require_adapter(provider: TrainingProvider) -> TrainingProvider:
        return provider

    @classmethod
    def local(
        cls,
        store: JobStore,
        *,
        fixture: bool = False,
        validate_cispo: bool = False,
        allow_unvalidated_canary: bool = False,
    ) -> "TinkerCispoExecutor":
        if fixture:
            transport = FakeTinkerProvider(validate_cispo=validate_cispo)
            return cls(
                store,
                TinkerAdapter(TinkerCredentials(api_key="fixture"), transport=transport),
                allow_unvalidated_canary=allow_unvalidated_canary,
            )
        return cls(
            store,
            TinkerAdapter(TinkerCredentials.from_env()),
            allow_unvalidated_canary=allow_unvalidated_canary,
        )

    def estimate(self, config: Mapping[str, Any]) -> dict[str, Any]:
        request = _cispo_request(config)
        dataset = _dataset(config)
        training = request.training
        groups = int(training.get("prompts_per_update") or 1)
        group_size = int(training.get("group_size") or 2)
        updates = int(training.get("updates") or 1)
        return {
            "algorithm_id": ALGORITHM_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "model_id": request.model_id,
            "estimated_rollouts": groups * group_size * updates,
            "train_examples": len(dataset.train),
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
        return _public_status(self.store.require(job_id), self.store.status_events(job_id))

    def cancel(self, job_id: str) -> dict[str, Any]:
        self.store.request_cancel(job_id)
        return self.status(job_id)

    def pause(self, job_id: str) -> dict[str, Any]:
        self.store.request_pause(job_id)
        return self.status(job_id)

    def resume(self, job_id: str) -> dict[str, Any]:
        job = self.store.resume_prepared(job_id)
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
        request = _cispo_request(config)
        from .runtime.training_budget import resolve_budget
        budget = resolve_budget(config, self.provider)
        dataset = _dataset(config)
        model_id = self.provider.resolve_model(request.model_id)
        generated_key = idempotency_key(
            algorithm_id=ALGORITHM_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            provider="tinker",
            model_id=model_id,
            dataset_digest=str(dataset.manifest["digest"]),
            split_manifest_digest=digest_payload(dataset.manifest["split_digests"]),
            renderer_version=request.renderer_version,
            training_config=dict(request.training),
            reward_version=str(request.reward.get("version") or "banking77.exact_label.v1"),
            seed=request.seed,
            runner_version=request.runner_version or RUNNER_VERSION,
            repeat_index=request.repeat_index,
        )
        key = idempotency_key_override or generated_key
        snapshot = {
            **dict(config),
            "algorithm_id": ALGORITHM_ID,
            "implementation": CISPO_IMPLEMENTATION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "model_id": model_id,
            "dataset_manifest": dataset.manifest,
            "budget": budget,
        }
        return self.store.persist_prepared(
            algorithm_id=ALGORITHM_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            provider="tinker",
            model_id=model_id,
            idempotency_key=key,
            config=snapshot,
            job_id=job_id or str(config.get("run_id") or ""),
        )

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
            raise UncertainOperation("legacy resume requires explicit verified migration")
        request = _cispo_request(config)
        dataset = _dataset(config)
        if config.get("dataset_manifest") != dataset.manifest:
            from .runtime.operations import UncertainOperation
            raise UncertainOperation("dataset identity changed; exact resume refused")
        slime = CispoConfig(
            eps_clip=float(request.training.get("eps_clip", 1.0)),
            eps_clip_high=float(request.training.get("eps_clip_high", 4.0)),
        )
        canary = bool(self.allow_unvalidated_canary or config.get("allow_unvalidated_canary"))
        try:
            slime.validate()
            capabilities = self.provider.discover_capabilities(job.model_id)
            capabilities.require(CISPO_REQUIRED_CAPABILITIES)
            if capabilities.validated.get("cispo.slime.v1") is not True and not canary:
                raise ProviderError("unsupported", "cispo.slime.v1 is not validated on this provider")
        except (UnsupportedCapability, ProviderError, CispoError) as exc:
            return self._fail(job_id, str(exc))
        if canary:
            self.store.append_event_once(
                job_id,
                "cispo.canary.started",
                {"model_id": job.model_id, "validated": False},
                phase="running",
            )
        session = self._session(job_id, job.model_id, config, request)
        clip_low = max(0.0, 1.0 - slime.eps_clip)
        clip_high = 1.0 + slime.eps_clip_high
        self.store.append_event_once(
            job_id,
            "cispo.clip.identity",
            {
                "identity": "cispo.slime.v1",
                "clip": {
                    "clip_low": clip_low,
                    "clip_high": clip_high,
                    "eps_clip": slime.eps_clip,
                    "eps_clip_high": slime.eps_clip_high,
                },
            },
            phase="running",
        )
        updates = int(request.training.get("updates") or 1)
        group_size = int(request.training.get("group_size") or 2)
        prompts_per_update = int(request.training.get("prompts_per_update") or 1)
        start = 1  # Replay confirmed operations from the durable journal.
        checkpoints: list[dict[str, Any]] = []
        try:
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
                candidate="parent",
                config=config,
            )
            self.store.append_event_once(
                job_id,
                "cispo.baseline_eval.completed",
                {**baseline_record, **public_evaluation(baseline), "role": "selection"},
                phase="running",
            )
            for update in range(start, updates + 1):
                if self.store.cancellation_requested(job_id):
                    return self._fail(job_id, "training cancellation requested")
                self.store.heartbeat(job_id, owner)
                prompts = _batch(dataset.train, update, prompts_per_update)
                groups, metrics = self._rollout_update(
                    job_id, session, prompts, update, group_size, request, slime, config
                )
                zero_groups = sum(1 for group in groups if group["zero_advantage"])
                if zero_groups == len(groups):
                    self.store.append_event_once(
                        job_id,
                        "cispo.update.completed",
                        {"update": update, "skipped": True, "reason": "zero_advantage", **metrics},
                        phase="running",
                    )
                else:
                    trainable = [group for group in groups if not group["zero_advantage"]]
                    self._train(job_id, session, trainable, update, slime, config)
                    self.store.append_event_once(
                        job_id,
                        "cispo.update.completed",
                        {"update": update, "skipped": False, **metrics},
                        phase="running",
                    )
                if update == updates or update % int(request.training.get("checkpoint_every_updates") or updates) == 0:
                    checkpoints.append(
                        self._checkpoint_and_eval(
                            job_id, session, dataset, update, config, baseline=baseline
                        )
                    )
                    self.store.set_resume_token(job_id, json.dumps({"step": update, "training_provider_reference": checkpoints[-1]["training_provider_reference"]}))
                    if self.store.require(job_id).state == "pause_requested":
                        self.store.transition(job_id, "paused")
                        return self.status(job_id)
            promoted = max(checkpoints, key=lambda item: item["calibration_accuracy"]) if checkpoints else None
            if promoted is None:
                return self._fail(job_id, "CISPO produced no checkpoint")
            self.store.append_event_once(job_id, "cispo.checkpoint.promoted", promoted, phase="evaluating")
            self.store.transition(job_id, "evaluating")
            heldout_base = self._evaluate(
                job_id,
                baseline_record,
                dataset.heldout,
                phase="heldout",
                candidate="parent",
                config=config,
            )
            heldout_trained = self._evaluate(
                job_id,
                promoted,
                dataset.heldout,
                phase="heldout",
                candidate="selected",
                config=config,
            )
            heldout = {
                **public_evaluation(heldout_trained),
                "role": "heldout",
                "heldout_locked": True,
                "baseline": public_evaluation(heldout_base),
                "trained": public_evaluation(heldout_trained),
                "paired_uplift": self._paired(config, heldout_base, heldout_trained),
            }
            self.store.append_event_once(job_id, "cispo.heldout_eval.completed", heldout, phase="evaluating")
            self.store.transition(job_id, "materializing")
            bundle = {
                "schema_version": "policy_bundle.v1",
                "algorithm_id": ALGORITHM_ID,
                "implementation_version": IMPLEMENTATION_VERSION,
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
                "cispo.model.materialized",
                {"digest": digest, "checkpoint_id": promoted["checkpoint_id"]},
                phase="materializing",
            )
            self.store.append_event_once(
                job_id,
                "cispo.completed",
                {"selected_checkpoint_id": promoted["checkpoint_id"], "heldout_accuracy": heldout["accuracy"]},
                phase="completed",
            )
            self.store.transition(job_id, "completed")
        except (ProviderError, CispoError) as exc:
            from .runtime.operations import UncertainOperation
            if isinstance(exc, UncertainOperation):
                raise
            if getattr(exc, "code", "") in {"experiment_budget_exhausted", "reservation_exceeded", "pricing_reconciliation_required"}:
                self.store.transition(job_id, "blocked_budget", error=str(exc))
                return self.status(job_id)
            if getattr(exc, "code", "") == "evaluation_blocked":
                self.store.transition(job_id, "blocked_evaluation", error=str(exc))
                return self.status(job_id)
            return self._fail(job_id, str(exc))
        return self.status(job_id)

    def _session(self, job_id: str, model_id: str, config: Mapping[str, Any], request: Any) -> ProviderSession:
        parent = config.get("parent_checkpoint")
        if isinstance(parent, Mapping) and parent.get("provider_reference"):
            return self.provider.restore_session(
                ProviderCheckpoint(
                    checkpoint_id=str(parent.get("checkpoint_id") or "parent"),
                    provider_reference=str(parent["provider_reference"]),
                    step=int(parent.get("step") or 0),
                    digest=str(parent.get("digest") or "sha256:" + "0" * 64),
                    kind=str(parent.get("kind") or "training"),
                    resume_token=str(parent.get("resume_token") or parent["provider_reference"]),
                    model_id=str(parent.get("model_id") or model_id),
                ),
                request_id=new_request_id(job_id, "restore"),
            )
        return self.provider.create_session(
            model_id,
            rank=int(config.get("rank") or 8),
            seed=request.seed,
            request_id=new_request_id(job_id, "session"),
        )

    def _rollout_update(
        self,
        job_id: str,
        session: ProviderSession,
        prompts: Sequence[Example],
        update: int,
        group_size: int,
        request: Any,
        slime: CispoConfig,
        config: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        all_rewards: list[float] = []
        zero = 0
        prompt = system_prompt_from(config)
        for prompt_index, example in enumerate(prompts):
            trajectories = []
            tokenized = encode_example(
                self.provider, example, system_prompt=prompt, add_generation_prompt=True
            )
            prompt_ids = tuple(tokenized.get("prompt_token_ids") or (1, 2, 3))
            requests = [
                SampleRequest(
                    request_id=new_request_id(
                        job_id, "roll", str(update), str(prompt_index), str(member)
                    ),
                    prompt_token_ids=prompt_ids,
                    max_tokens=int(request.training.get("max_sample_tokens") or 24),
                    temperature=float(request.training.get("temperature") or 1.0),
                    seed=request.seed + update * 1000 + prompt_index * 10 + member,
                )
                for member in range(group_size)
            ]
            with ThreadPoolExecutor(max_workers=sample_parallelism()) as executor:
                futures = [
                    executor.submit(self.provider.sample, session, sample_request)
                    for sample_request in requests
                ]
                sampled_group = [future.result() for future in futures]
            for sampled in sampled_group:
                predicted = extract_final_label(sampled.text)
                label = extract_final_label(example.label or "")
                reward = 1.0 if predicted == label else 0.0
                trajectories.append(
                    {
                        "text": sampled.text,
                        "reward": reward,
                        "token_ids": sampled.token_ids,
                        "prompt_token_ids": prompt_ids,
                        "behavior_logprobs": sampled.logprobs,
                        "label": label,
                    }
                )
            rewards = [item["reward"] for item in trajectories]
            advantages = group_advantages(rewards, normalize=bool(request.training.get("normalize_group_rewards", True)))
            zero_adv = is_zero_advantage_group(advantages)
            group = {
                "group_id": f"{update}:{prompt_index}",
                "iteration": update,
                "prompt_digest": digest_payload(example.example_id),
                "rewards": rewards,
                "advantages": list(advantages),
                "zero_advantage": zero_adv,
                "trajectories": trajectories,
                "label": example.label,
            }
            self.store.append_event_once(
                job_id,
                "cispo.rollout_group.completed",
                {
                    "group_id": group["group_id"],
                    "iteration": update,
                    "rewards": rewards,
                    "reward_mean": statistics.fmean(rewards),
                    "reward_range": max(rewards) - min(rewards),
                    "reward_variance": statistics.pvariance(rewards),
                    "label": example.label,
                },
                phase="running",
            )
            self.store.append_event_once(
                job_id,
                "cispo.group_advantage.computed",
                {"group_id": group["group_id"], "advantages": list(advantages), "zero_advantage": zero_adv},
                phase="running",
            )
            if zero_adv:
                zero += 1
                self.store.append_event_once(
                    job_id,
                    "cispo.zero_advantage.detected",
                    {"group_id": group["group_id"], "rewards": rewards},
                    phase="running",
                )
            groups.append(group)
            all_rewards.extend(rewards)
        metrics = {
            "group_count": len(groups),
            "zero_advantage_groups": zero,
            "zero_advantage_rate": zero / max(1, len(groups)),
            "reward_mean": statistics.fmean(all_rewards) if all_rewards else 0.0,
            "reward_range": (max(all_rewards) - min(all_rewards)) if all_rewards else 0.0,
            "reward_variance": statistics.pvariance(all_rewards) if len(all_rewards) > 1 else 0.0,
        }
        return groups, metrics

    def _train(
        self,
        job_id: str,
        session: ProviderSession,
        groups: Sequence[Mapping[str, Any]],
        update: int,
        slime: CispoConfig,
        config: Mapping[str, Any],
    ) -> None:
        token_rows: list[tuple[int, ...]] = []
        masks: list[tuple[bool, ...]] = []
        behavior_rows: list[tuple[float, ...]] = []
        advantage_rows: list[list[float]] = []
        train_rows: list[dict[str, Any]] = []
        for group in groups:
            for trajectory, advantage in zip(group["trajectories"], group["advantages"], strict=True):
                tokens = tuple(int(token) for token in trajectory["token_ids"])
                mask = tuple(True for _ in tokens)
                token_rows.append(tokens)
                masks.append(mask)
                behavior_rows.append(tuple(float(value) for value in trajectory["behavior_logprobs"]))
                advantage_rows.append([float(advantage)] * len(tokens))
                train_rows.append(
                    {
                        "token_ids": tokens,
                        "prompt_token_ids": tuple(int(token) for token in trajectory.get("prompt_token_ids") or ()),
                        "behavior_logprobs": trajectory["behavior_logprobs"],
                        "advantages": [float(advantage)] * len(tokens),
                    }
                )
        forward = self.provider.forward(
            session,
            ForwardRequest(
                request_id=new_request_id(job_id, "forward", str(update)),
                token_ids=tuple(token_rows),
                response_masks=tuple(masks),
            ),
        )
        all_ratios: list[float] = []
        clipped_tokens = 0
        selected_tokens = 0
        for current, behavior, advantages, mask in zip(
            forward.logprobs, behavior_rows, advantage_rows, masks, strict=True
        ):
            usable = min(len(current), len(behavior), len(advantages), len(mask))
            ratios = importance_ratios(current[:usable], behavior[:usable])
            kls = [ppo_kl_from_ratio(ratio) for ratio in ratios]
            measured = objective(kls, current[:usable], advantages[:usable], mask[:usable], slime)
            all_ratios.extend(ratios)
            clipped_tokens += measured.clipped_token_count
            selected_tokens += measured.selected_token_count
        self.store.append_event_once(
            job_id,
            "cispo.importance_ratio.measured",
            {
                "update": update,
                "mean_ratio": statistics.fmean(all_ratios) if all_ratios else 1.0,
                "ratio_min": min(all_ratios) if all_ratios else 1.0,
                "ratio_max": max(all_ratios) if all_ratios else 1.0,
                "clipped_token_fraction": clipped_tokens / max(1, selected_tokens),
                "effective_tokens": selected_tokens,
                "kl_proxy": statistics.fmean(abs(math.log(max(ratio, 1e-8))) for ratio in all_ratios)
                if all_ratios
                else 0.0,
            },
            phase="running",
        )
        result = self.provider.train_step(
            session,
            TrainingStepRequest(
                request_id=new_request_id(job_id, "cispo-train", str(update)),
                loss_name="cispo.slime.v1",
                data=tuple(train_rows),
                metadata={
                    "implementation": CISPO_IMPLEMENTATION,
                    "implementation_version": IMPLEMENTATION_VERSION,
                    "eps_clip": slime.eps_clip,
                    "eps_clip_high": slime.eps_clip_high,
                    "learning_rate": float((config.get("training") or {}).get("learning_rate") or 5e-6),
                },
            ),
        )
        if result.metrics.get("update_norm") is not None:
            update_norm = float(result.metrics["update_norm"])
        else:
            update_norm = float(result.metrics.get("loss", 0.0))
        self.store.put_receipt(
            job_id,
            result.request_id,
            {
                "schema_version": "training.usage_receipt.v1",
                "provider": "tinker",
                "request_id": result.request_id,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "training_tokens": result.usage.training_tokens,
                "cost_usd": result.usage.cost_usd,
                "cost_missing": result.usage.cost_missing,
                "algorithm_id": ALGORITHM_ID,
                "implementation_version": IMPLEMENTATION_VERSION,
                "update_norm": update_norm,
            },
        )

    def _checkpoint_and_eval(
        self,
        job_id: str,
        session: ProviderSession,
        dataset: SplitDataset,
        update: int,
        config: Mapping[str, Any],
        *,
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        training = self.provider.save_checkpoint(
            session, step=update, kind="training", request_id=new_request_id(job_id, "ckpt", str(update), "training")
        )
        inference = self.provider.save_checkpoint(
            session, step=update, kind="inference", request_id=new_request_id(job_id, "ckpt", str(update), "inference")
        )
        record = {
            "checkpoint_id": inference.checkpoint_id,
            "training_checkpoint_id": training.checkpoint_id,
            "provider_reference": inference.provider_reference,
            "training_provider_reference": training.provider_reference,
            "training_digest": training.digest,
            "digest": inference.digest,
            "resume_token": training.resume_token,
            "step": update,
        }
        self.store.append_event_once(job_id, "cispo.checkpoint.created", record, phase="evaluating")
        evaluation = self._evaluate(
            job_id,
            record,
            dataset.calibration,
            phase="selection",
            candidate=f"checkpoint:{update}",
            config=config,
        )
        payload = {
            **record,
            **public_evaluation(evaluation),
            "calibration_accuracy": evaluation["accuracy"],
            "role": "selection",
            "paired_uplift": self._paired(config, baseline, evaluation),
        }
        self.store.append_event_once(job_id, "cispo.checkpoint_eval.completed", payload, phase="evaluating")
        return payload

    def _evaluate(
        self,
        job_id: str,
        checkpoint: Mapping[str, Any],
        examples: Sequence[Example],
        *,
        phase: str,
        candidate: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        def stream(record: Mapping[str, Any]) -> None:
            self.store.append_event_once(
                job_id,
                "cispo.evaluation.example.completed",
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
            system_prompt=system_prompt_from(config),
            max_tokens=eval_max_tokens(config),
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
        from dataclasses import asdict
        self.store.put_receipt(job_id, request_id, {
            **asdict(usage), "request_id": request_id, "provider": "tinker",
            "schema_version": "training.usage_receipt.v1", "algorithm_id": ALGORITHM_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
        })

    def _fail(self, job_id: str, reason: str) -> dict[str, Any]:
        if self.store.cancellation_requested(job_id):
            self.store.transition(job_id, "cancelled")
            return self.status(job_id)
        self.store.append_event_once(job_id, "cispo.failed", {"reason": reason}, phase="failed")
        self.store.transition(job_id, "failed", error=reason)
        return self.status(job_id)


def _cispo_request(config: Mapping[str, Any]) -> Any:
    payload = {
        "schema_version": "cispo.request.v1",
        "algorithm_id": config.get("algorithm_id", ALGORITHM_ID),
        "implementation": config.get("implementation", CISPO_IMPLEMENTATION),
        "implementation_version": config.get("implementation_version", IMPLEMENTATION_VERSION),
        "provider": config.get("provider", "tinker"),
        "model_id": config.get("model_id") or config.get("base_model") or "openai/gpt-oss-20b",
        "dataset": config.get("dataset") or {"examples": config.get("examples") or []},
        "training": config.get("training") or {},
        "reward": config.get("reward") or {"version": "banking77.exact_label.v1"},
        "evaluation": config.get("evaluation") or {},
        "seed": config.get("seed", 0),
        "repeat_index": config.get("repeat_index", 0),
        "renderer_version": config.get("renderer_version", "chat.v1"),
        "runner_version": config.get("runner_version", RUNNER_VERSION),
        "mode": config.get("mode", "canonical"),
    }
    return validate_cispo_request(payload)


def _dataset(config: Mapping[str, Any]) -> SplitDataset:
    return split_dataset_from_config(config)


def _batch(examples: Sequence[Example], step: int, size: int) -> list[Example]:
    start = ((step - 1) * size) % len(examples)
    return [examples[(start + offset) % len(examples)] for offset in range(size)]


def _resume_update(job: TrainingJob) -> int:
    if not job.resume_token:
        return 0
    from .sft_executor import _resume_step
    return _resume_step(job)
