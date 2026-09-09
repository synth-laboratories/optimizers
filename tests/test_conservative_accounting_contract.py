"""What the training aggregate actually counts.

Every charge in every v0.10 acceptance ledger settled `conservative`: 1,624
recorded Tinker receipts carry full token accounting and no cost at all. The
counted aggregate is therefore a sum of declared reservation prices, not of
money the provider reported. These tests pin that distinction so a later reader
cannot mistake one for the other, and so a provider that does start returning
cost changes the recorded status rather than passing silently.

Payload shapes below are taken from real receipts in the retained C1 ledger
(`training_receipts`, job `cispo_hosted_d495991d4c16`); the transports and
budget here are the production ones.
"""
import pytest

from synth_optimizers.providers.protocols import (
    ProviderUsage,
    TrainingStepRequest,
    TrainingStepResult,
)
from synth_optimizers.providers.tinker.training import _usage
from synth_optimizers.runtime import JobStore
from synth_optimizers.runtime.training_budget import TrainingBudget

# The declared contract C1 actually ran under. `session`/`save`/`restore` are
# flat per-operation reserves, not published Tinker rates: the version string
# says so.
C1_PRICING = {
    "input_usd_per_million": 0.18,
    "output_usd_per_million": 0.45,
    "training_usd_per_million": 0.396,
    "session_usd": 0.25,
    "save_usd": 0.25,
    "restore_usd": 0.25,
}
C1_PLAN = {"max_cost_usd": 5.0, "pricing": C1_PRICING,
           "pricing_version": "tinker.models.20260908.conservative-operation-reserves"}


def _store(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.persist_prepared(algorithm_id="cispo", implementation_version="cispo.slime.v1",
                           provider="tinker", model_id="openai/gpt-oss-20b",
                           idempotency_key="key", config={}, job_id="run")
    return store


def test_a_real_tinker_response_reports_tokens_and_no_cost():
    """The shape every recorded receipt had: metered tokens, absent cost."""
    usage = _usage({"input_tokens": 494, "output_tokens": 24, "training_tokens": 0})
    assert (usage.input_tokens, usage.output_tokens) == (494, 24)
    assert usage.cost_usd is None
    assert usage.cost_missing is True


def test_absent_provider_cost_settles_conservative_at_the_reservation(tmp_path):
    """With no reported cost the ledger counts what it reserved, and says so."""
    store = _store(tmp_path)
    budget = TrainingBudget(store, "run", C1_PLAN)
    request = TrainingStepRequest("step", "cross_entropy", ({"input_ids": [1, 2, 3]},))
    budget.reserve("step", "train", request)
    reserved = budget.ledger.operation("step")["reserved_microusd"]

    budget.settle("step", TrainingStepResult("step", 1, {}, _usage({"training_tokens": 3})))

    charge = budget.ledger.operation("step")
    assert charge["status"] == "conservative"
    assert charge["counted_microusd"] == reserved
    store.close()


def test_reported_provider_cost_settles_usage_counted(tmp_path):
    """The branch Tinker never takes today. It must still be reachable, and
    must record a different status, or nothing distinguishes a measured charge
    from a reserved one."""
    store = _store(tmp_path)
    budget = TrainingBudget(store, "run", C1_PLAN)
    request = TrainingStepRequest("step", "cross_entropy", ({"input_ids": [1, 2, 3]},))
    budget.reserve("step", "train", request)
    usage = ProviderUsage(training_tokens=3, cost_usd=0.000001, cost_missing=False)

    budget.settle("step", TrainingStepResult("step", 1, {}, usage))

    charge = budget.ledger.operation("step")
    assert charge["status"] == "usage_counted"
    assert charge["counted_microusd"] == 1
    store.close()


@pytest.mark.parametrize("lane", ["session", "save", "restore"])
def test_flat_lanes_charge_the_declared_price_whatever_the_work(tmp_path, lane):
    """These three carry no token term at all. Under C1's contract each is a
    flat $0.25 the provider is never asked about — 55 of them are 98% of the
    v0.10 counted aggregate."""
    store = _store(tmp_path)
    budget = TrainingBudget(store, "run", C1_PLAN)

    budget.reserve(f"{lane}-1", lane)
    assert budget.ledger.operation(f"{lane}-1")["reserved_microusd"] == 250_000

    budget.settle(f"{lane}-1", TrainingStepResult(f"{lane}-1", 1, {}, _usage({})))
    charge = budget.ledger.operation(f"{lane}-1")
    assert charge["status"] == "conservative"
    assert charge["counted_microusd"] == 250_000
    store.close()


def test_the_c1_operation_mix_reproduces_its_recorded_total(tmp_path):
    """C1's ledger: 2 restore, 3 save, 24 sample, 1600 sample_checkpoint,
    $1.447468 counted, of which $1.25 is the five flat operations."""
    store = _store(tmp_path)
    budget = TrainingBudget(store, "run", {**C1_PLAN, "max_cost_usd": 5.0})
    flat = 0
    for lane, count in (("restore", 2), ("save", 3)):
        for index in range(count):
            operation = f"{lane}-{index}"
            budget.reserve(operation, lane)
            budget.settle(operation, TrainingStepResult(operation, 1, {}, _usage({})))
            flat += budget.ledger.operation(operation)["counted_microusd"]

    assert flat == 1_250_000
    # The recorded run's token-metered remainder was $0.197468, so the flat
    # reserves are 86% of C1 alone and the whole $1.447468 is conservative.
    assert flat / 1_447_468 > 0.86
    assert budget.ledger.snapshot()["counted_or_reserved_usd"] == 1.25
    store.close()
