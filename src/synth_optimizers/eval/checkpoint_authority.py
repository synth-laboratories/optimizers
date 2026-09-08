"""Checkpoint child jobs executed by the existing local eval authority.

The request mapping points at real EvalRunner run directories and manifests.
There is no second rollout scheduler, synthetic score, or shadow eval ID.
"""
from dataclasses import replace
from pathlib import Path
import json
import math
import sqlite3
import threading
import uuid

from .checkpoint_gateway import CheckpointGateway
from .home import EvalHome
from .models import EvalContractError, write_json, digest_of
from .runner import EvalRunner, WorkerManifest
from .staging import CandidateSource, stage_candidate_set


class CheckpointRunner(EvalRunner):
    def __init__(self, *args, gateway, model_id, should_stop, prices, **kwargs):
        self.gateway, self.model_id, self.should_stop = gateway, model_id, should_stop
        self.prices = prices
        self._trial_local = threading.local()
        super().__init__(*args, **kwargs)

    def _secrets(self):
        token = getattr(self._trial_local, "credential", "")
        return {name: token for name in self.recipe.secrets}

    def _write_trial_manifest(self, key, candidate):
        origin = self.gateway.bind(key.trial_id, key.seed)
        self._trial_local.credential = origin.credential
        path = super()._write_trial_manifest(key, candidate)
        value = json.loads((path / "trial.json").read_text())
        if len(self.recipe.secrets) != 1:
            raise EvalContractError("checkpoint target must declare exactly one sampler credential")
        value["policy_snapshot_id"] = self.gateway.checkpoint["checkpoint_id"]
        value["models"] = [{"id": self.model_id, "route": origin.base_url + "/chat/completions",
                             "secret": self.recipe.secrets[0], "efforts": [],
                             "usd_per_1m_input": float(self.prices["input_usd_per_million"]),
                             "usd_per_1m_output": float(self.prices["output_usd_per_million"]),
                             "usd_per_1m_cached_input": float(self.prices["input_usd_per_million"]),
                             "price_source": "parent_training_budget", "price_as_of": "parent_spec"}]
        write_json(path / "trial.json", value)
        return path

    def _run_trial(self, key):
        if self.should_stop():
            self.cancel._event.set()
        marker = self._trial_dir(key) / "checkpoint_dispatch.json"
        existing = self._existing_record(key)
        if existing is None:
            if marker.exists():
                raise EvalContractError("checkpoint trial outcome uncertain; reconcile retained container work")
            write_json(marker, {"trial_id": key.trial_id, "checkpoint": self.gateway.checkpoint})
        return super()._run_trial(key)

    def _record(self, key, **kwargs):
        container = kwargs.get("container")
        if container is not None:
            binding = self.gateway.evidence(key.trial_id)
            path = self._trial_dir(key) / "output" / "checkpoint_binding.json"
            write_json(path, binding)
            served = bool(binding["calls"])
            kwargs["container"] = replace(container,
                gates=(*container.gates, ("exact_checkpoint", served)),
                artifacts=(*container.artifacts, {"role": "checkpoint_binding", "path": "checkpoint_binding.json"}))
            if not served:
                kwargs["error"] = "target produced no verified checkpoint sampler calls"
        return super()._record(key, **kwargs)

    def _run_stage(self, stage, candidate_ids, seeds):
        # Preserve bounded admission on failure/cancel rather than eagerly submitting every trial.
        from concurrent.futures import ThreadPoolExecutor
        keys = list(self._trial_keys(stage, candidate_ids, seeds))
        records = []
        with ThreadPoolExecutor(max_workers=self._parallelism) as pool:
            for offset in range(0, len(keys), self._parallelism):
                if self.should_stop():
                    self.cancel._event.set()
                    break
                futures = [pool.submit(self._run_trial, key) for key in keys[offset:offset+self._parallelism]]
                error = None
                for future in futures:
                    try:
                        records.append(future.result())
                    except Exception as exc:
                        error = error or exc
                if error is not None:
                    raise error
        return records


class CheckpointEvaluationAuthority:
    def __init__(self, home, *, executor=None):
        self.home = EvalHome.open(home)
        self.executor = executor
        self.database = self.home.root / "checkpoint_requests.sqlite"
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, job_id TEXT NOT NULL, status TEXT NOT NULL)")

    def validate(self, evaluator):
        recipe = self.home.recipe(evaluator["recipe_id"])
        if not recipe.available or recipe.image_digest != evaluator["image_digest"]:
            raise EvalContractError("checkpoint evaluation requires the registered exact image digest")
        if recipe.policy_kind != "tinker-sampler.v1":
            raise EvalContractError("target does not advertise immutable hosted checkpoint sampling")
        selection, final = evaluator["selection_seeds"], evaluator["final_seeds"]
        if not selection or not final or set(selection) & set(final):
            raise EvalContractError("selection and final panels must be nonempty and disjoint")
        declared = set(recipe.screening_seeds) | set(recipe.confirmation_seeds)
        if not set(selection + final) <= declared:
            raise EvalContractError("checkpoint panel is outside the registered recipe")
        if evaluator["metric_ref"] not in {metric.id for metric in recipe.target.metrics}:
            raise EvalContractError("unknown checkpoint reward metric")
        # Admission must fail before creating a paid training session if the
        # registered image has disappeared or its mutable tag changed.
        from .executor import OciTrialExecutor
        executor = self.executor if self.executor is not None else OciTrialExecutor(self.home.config.container_runtime)
        resolve = getattr(executor, "resolve_reference", None)
        if resolve is not None:
            resolve(recipe.image, recipe.image_digest)
        return recipe

    def lookup(self, request_id):
        with sqlite3.connect(self.database) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
        return dict(row) if row else None

    def evaluate(self, request_id, checkpoint, evaluator, *, provider, model_id, parent_run_id,
                 role, on_event, should_stop, renderer_profile):
        recipe = self.validate(evaluator)
        seeds = evaluator["final_seeds"] if role == "final" else evaluator["selection_seeds"]
        identity = {"checkpoint": checkpoint, "evaluator": evaluator, "role": role,
                    "parent_run_id": parent_run_id, "renderer_profile": renderer_profile}
        with sqlite3.connect(self.database) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,job_id,status FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row and row[0] != digest_of(identity):
                raise EvalContractError("evaluation request identity changed")
            if row:
                job_id = row[1]
            else:
                job_id = "eval_" + uuid.uuid4().hex
                db.execute("INSERT INTO requests VALUES (?,?,?,?)", (request_id, digest_of(identity), job_id, "prepared"))
        run_dir = self.home.run_dir(job_id)
        on_event({"event": "eval.child.attached", "run_id": job_id, "request_id": request_id,
                  "checkpoint": checkpoint, "role": role})
        if (run_dir / "result_manifest.json").exists():
            return self.result(job_id, evaluator, checkpoint, len(seeds)*len(recipe.scenarios))
        if row and row[2] == "running":
            raise EvalContractError("child evaluation interrupted; reconcile existing job before resuming")
        source = run_dir / "checkpoint_policy"
        source.mkdir(parents=True, exist_ok=True)
        write_json(source / "policy.json", {"schema_version": "eval.tinker-sampler.v1", "base_model": model_id,
                                             "checkpoint_id": checkpoint["checkpoint_id"],
                                             "sampler_reference": checkpoint["provider_reference"]})
        candidate = stage_candidate_set(self.home,
            [CandidateSource(checkpoint["checkpoint_id"], source, "policy.json", "tinker-sampler.v1")])
        candidate_path = self.home.candidates_dir / candidate.id / "candidate_set.json"
        write_json(candidate_path, candidate.to_json())
        manifest = WorkerManifest(job_id, recipe.id, self.home.root, candidate_path, None,
            correlation={**identity, "request_id": request_id}, plan_override={"seeds": seeds})
        write_json(run_dir / "worker_manifest.json", {"schema_version": "eval.worker-manifest.v1",
            "run_id": job_id, "recipe_id": recipe.id, "home": str(self.home.root),
            "candidate_set_path": str(candidate_path), "correlation": manifest.correlation,
            "plan_override": manifest.plan_override})
        class Stream:
            def write(self, line):
                on_event(json.loads(line))
            def flush(self):
                pass
        with CheckpointGateway(provider, checkpoint, run_id=job_id, renderer_profile=renderer_profile,
                bind_host="0.0.0.0" if self.executor is None else "127.0.0.1",
                advertised_host="host.docker.internal" if self.executor is None else "127.0.0.1",
                ttl_seconds=recipe.limits.timeout_seconds) as gateway:
            runner = CheckpointRunner(manifest, gateway=gateway, model_id=model_id,
                                      should_stop=should_stop, executor=self.executor, stream=Stream(),
                                      prices=provider.budget.prices)
            with sqlite3.connect(self.database) as db:
                changed = db.execute("UPDATE requests SET status='running' WHERE request_id=? AND status='prepared'", (request_id,)).rowcount
                if changed != 1:
                    raise EvalContractError("child evaluation already has an owner")
            finished = threading.Event()
            def monitor_parent():
                while not finished.wait(0.1):
                    if should_stop():
                        runner.cancel._event.set()
                        return
            monitor = threading.Thread(target=monitor_parent, daemon=True)
            monitor.start()
            try:
                code = runner.execute()
            finally:
                finished.set()
                monitor.join()
            if gateway.gateway.provider_failures:
                # Optional scoring failures cannot waive uncertain work or budget admission.
                raise gateway.gateway.provider_failures[0]
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE requests SET status=? WHERE request_id=?", ("completed" if code == 0 else "failed", request_id))
        if code:
            raise EvalContractError(f"checkpoint eval job {job_id} failed; retained evidence at {run_dir}")
        return self.result(job_id, evaluator, checkpoint, len(seeds)*len(recipe.scenarios))

    def result(self, job_id, evaluator, checkpoint, expected):
        manifest = json.loads((self.home.run_dir(job_id) / "result_manifest.json").read_text())
        if manifest["correlation"]["checkpoint"] != checkpoint:
            raise EvalContractError("child checkpoint provenance mismatch")
        rows = [json.loads(Path(row["evidence"]).read_text()) for row in manifest["trials"]]
        rewards = [row["metrics"].get(evaluator["metric_ref"]) for row in rows]
        valid = len(rows) == expected and all(row["status"] == "evaluated" and
                all(row["gates"].values()) and not row["missing_gates"] and not row["missing_artifacts"]
                for row in rows) and all(isinstance(value, (int,float)) and math.isfinite(value) for value in rewards)
        from .checkpoint_trace import materialize_checkpoint_trace
        retained = list(manifest["artifacts"])
        if valid:
            for trial, evidence, reward in zip(manifest["trials"], rows, rewards):
                output = Path(trial["evidence"]).parent / "output"
                retained.append(materialize_checkpoint_trace(output, job_id=job_id,
                    trial_id=trial["trial_id"], checkpoint=checkpoint, evaluator=evaluator, reward=reward))
        return {"eval_job_id": job_id, "checkpoint_id": checkpoint["checkpoint_id"],
                "actual_sampler_reference": checkpoint["provider_reference"], "expected": expected,
                "completed": len(rows), "valid": valid, "status": "completed" if valid else "partial",
                "metric_ref": evaluator["metric_ref"], "reward_version": evaluator["reward_version"],
                "units": evaluator["units"], "value": sum(rewards)/expected if valid else None,
                "rollouts": manifest["trials"], "evidence_refs": retained}
