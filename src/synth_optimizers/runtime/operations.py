"""Durable provider boundaries. Unknown outcomes are never automatically replayed."""

from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
import json

from .jobs import digest_payload
from ..providers import protocols


class UncertainOperation(protocols.ProviderError):
    def __init__(self, message):
        super().__init__("operation_uncertain", message)


def encode(value):
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, list):
        return tuple(decode(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"type", "fields"}:
            allowed = {
                "ProviderCheckpoint",
                "TrainingStepResult",
                "SampleResult",
                "ForwardResult",
                "ProviderUsage",
                "ProviderSession",
            }
            if value["type"] not in allowed:
                raise ValueError("unsupported durable provider result")
            return getattr(protocols, value["type"])(
                **{key: decode(item) for key, item in value["fields"].items()}
            )
        return {key: decode(item) for key, item in value.items()}
    return value


class DurableProvider:
    """Journal updates, saves and samples, retaining confirmed results across workers."""

    def __init__(self, provider, store, job_id, owner):
        from copy import copy

        self.provider = copy(provider)
        # A transport retry after an uncertain write can double-charge or double-update.
        if hasattr(self.provider, "max_attempts"):
            self.provider.max_attempts = 1
        self.store, self.job_id, self.owner = store, job_id, owner
        with store._lock:
            store._db.execute("""CREATE TABLE IF NOT EXISTS training_operations (
                job_id TEXT NOT NULL, request_id TEXT NOT NULL, kind TEXT NOT NULL,
                input_digest TEXT NOT NULL, result_json TEXT,
                PRIMARY KEY(job_id, request_id))""")
            store._db.commit()
        from .training_budget import TrainingBudget
        config = json.loads(store.require(job_id).config_json)
        self.budget = TrainingBudget(store, job_id, config["budget"]) if "budget" in config else None
        self.restored_step = 0
        self._recovery_checked = False

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def _call(self, kind, request_id, identity, call, *, priced_request=None):
        digest = digest_payload(encode(identity))
        cached = None
        with self.store.owned(self.owner), self.store._write(self.job_id):
            row = self.store._db.execute(
                "SELECT * FROM training_operations WHERE job_id=? AND request_id=?",
                (self.job_id, request_id),
            ).fetchone()
            if row:
                if row["kind"] != kind or row["input_digest"] != digest:
                    raise UncertainOperation("provider operation identity changed")
                if row["result_json"] is None:
                    raise UncertainOperation(f"reconciliation required for {kind} {request_id}")
                cached = decode(json.loads(row["result_json"]))
            else:
                self.store._db.execute(
                    "INSERT INTO training_operations VALUES (?, ?, ?, ?, NULL)",
                    (self.job_id, request_id, kind, digest),
                )
        if cached is not None:
            if self.budget is not None:
                self.budget.settle(request_id, cached)
            return cached
        if self.budget is not None:
            try:
                self.budget.reserve(request_id, kind, priced_request)
            except Exception:
                # No provider dispatch occurred. A prior reservation is retained for reconciliation.
                if self.budget.ledger.operation(request_id) is None:
                    with self.store.owned(self.owner), self.store._write(self.job_id):
                        self.store._db.execute("DELETE FROM training_operations WHERE job_id=? AND request_id=?",
                                               (self.job_id, request_id))
                raise
        # Intent commits before dispatch. A crash/exception leaves the outcome unresolved.
        try:
            result = call()
        except Exception as exc:
            raise UncertainOperation(f"reconciliation required for {kind} {request_id}") from exc
        with self.store.owned(self.owner), self.store._write(self.job_id):
            self.store._db.execute(
                "UPDATE training_operations SET result_json=? WHERE job_id=? AND request_id=?",
                (json.dumps(encode(result)), self.job_id, request_id),
            )
            if hasattr(result, "usage"):
                job = self.store.require(self.job_id)
                usage = result.usage
                receipt = {field.name: getattr(usage, field.name) for field in fields(usage)}
                receipt.update(
                    {
                        "schema_version": "training.usage_receipt.v1",
                        "request_id": request_id,
                        "provider": job.provider,
                        "algorithm_id": job.algorithm_id,
                        "implementation_version": job.implementation_version,
                    }
                )
                self.store._db.execute(
                    "INSERT OR REPLACE INTO training_receipts VALUES (?, ?, ?)",
                    (self.job_id, request_id, json.dumps(receipt, sort_keys=True)),
                )
                self.store._insert_event(self.job_id, "training.receipt", receipt, "running")
            self.store._insert_event(
                self.job_id,
                "training.operation.confirmed",
                {"request_id": request_id, "operation": kind},
                "running",
            )
        if self.budget is not None:
            self.budget.settle(request_id, result)
            self.store.append_event_once(self.job_id, "training.budget", self.budget.ledger.snapshot(), phase="running")
        return result

    def recovery_checkpoint(self):
        with self.store._lock:
            rows = self.store._db.execute(
                "SELECT kind, result_json FROM training_operations WHERE job_id=?", (self.job_id,)
            ).fetchall()
        if not rows and not self._recovery_checked:
            with self.store._lock:
                legacy = self.store._db.execute(
                    "SELECT 1 FROM training_events WHERE job_id=? AND kind IN "
                    "('sft.training.started', 'cispo.training.started', 'sft.step.metrics', 'cispo.update.completed') LIMIT 1",
                    (self.job_id,),
                ).fetchone()
            if legacy:
                raise UncertainOperation(
                    "legacy run lacks a durable provider journal; exact resume refused"
                )
        if any(row["result_json"] is None for row in rows):
            raise UncertainOperation(
                "provider operation outcome requires reconciliation before resume"
            )
        results = [(row["kind"], decode(json.loads(row["result_json"]))) for row in rows]
        checkpoints = [
            result for kind, result in results if kind == "save" and result.kind == "training"
        ]
        checkpoint = max(checkpoints, key=lambda item: item.step, default=None)
        if checkpoint is None and any(kind == "session" for kind, _ in results):
            raise UncertainOperation("created session has no durable optimizer state; reconciliation required")
        saved_step = checkpoint.step if checkpoint else 0
        if any(kind == "train" and result.step > saved_step for kind, result in results):
            raise UncertainOperation(
                "confirmed update has no saved optimizer state; exact resume unavailable"
            )
        self._recovery_checked = True
        return checkpoint

    def create_session(self, model_id, *, rank, seed, request_id):
        checkpoint = self.recovery_checkpoint()
        if checkpoint:
            return self._restore(checkpoint, request_id)
        session = self._call("session", request_id, {"model_id": model_id, "rank": rank, "seed": seed},
                             lambda: self.provider.create_session(model_id, rank=rank, seed=seed,
                                                                  request_id=request_id))
        self.save_checkpoint(session, step=0, kind="training", request_id=f"{request_id}-initial-state")
        return session

    def _restore(self, checkpoint, request_id):
        checkpoint = replace(
            checkpoint, model_id=checkpoint.model_id or self.store.require(self.job_id).model_id
        )
        self.restored_step = checkpoint.step
        restore_id = f"{request_id}-{self.owner}"
        return self._call("restore", restore_id, checkpoint,
                          lambda: self.provider.restore_session(checkpoint, request_id=restore_id))

    def restore_session(self, checkpoint, *, request_id):
        return self._restore(self.recovery_checkpoint() or checkpoint, request_id)

    def train_step(self, session, request):
        return self._call(
            "train", request.request_id, request, lambda: self.provider.train_step(session, request),
            priced_request=request
        )

    def save_checkpoint(self, session, *, step, kind, request_id):
        return self._call(
            "save",
            request_id,
            {"step": step, "kind": kind},
            lambda: self.provider.save_checkpoint(
                session, step=step, kind=kind, request_id=request_id
            ),
        )

    def sample_checkpoint(self, checkpoint, request):
        return self._call(
            "sample_checkpoint",
            request.request_id,
            {"checkpoint": checkpoint, "request": request},
            lambda: self.provider.sample_checkpoint(checkpoint, request), priced_request=request,
        )

    def sample(self, session, request):
        return self._call(
            "sample", request.request_id, request, lambda: self.provider.sample(session, request), priced_request=request
        )

    def forward(self, session, request):
        return self._call(
            "forward", request.request_id, request, lambda: self.provider.forward(session, request), priced_request=request
        )
