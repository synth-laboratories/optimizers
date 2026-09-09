"""Recovery injection where the money actually is.

98% of the v0.10 counted training aggregate is flat per-operation reserves for
`session`, `save` and `restore` — 55 charges at a declared $0.25 against
$0.234719 of token-metered work. So the accounting failure that would cost real
money is not a mispriced token: it is a save reserved twice across a crash, or a
cap that comes back empty after a restart.

The existing durability suite exercises restart and fencing with those three
lanes priced at zero, which cannot catch either. These drive the same faults
under C1's real pricing contract.

No provider call, no credential, one tmp_path store per test.
"""
import pytest

from synth_optimizers.providers.protocols import TrainingStepResult
from synth_optimizers.providers.tinker.training import _usage
from synth_optimizers.rl.budget import BudgetError
from synth_optimizers.runtime import JobStore
from synth_optimizers.runtime.training_budget import TrainingBudget

# The contract C1 actually ran under.
PRICING = {
    "input_usd_per_million": 0.18,
    "output_usd_per_million": 0.45,
    "training_usd_per_million": 0.396,
    "session_usd": 0.25,
    "save_usd": 0.25,
    "restore_usd": 0.25,
}


def store_with_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.persist_prepared(algorithm_id="cispo", implementation_version="cispo.slime.v1",
                           provider="tinker", model_id="openai/gpt-oss-20b",
                           idempotency_key="key", config={}, job_id="run")
    return store


def plan(cap):
    return {"max_cost_usd": cap, "pricing": PRICING}


def test_a_save_reserved_before_a_crash_is_not_reserved_again(tmp_path):
    """The expensive replay. A quarter of a dollar per save means a silent
    double-reserve is the costliest bug in the ledger."""
    store = store_with_job(tmp_path)
    first = TrainingBudget(store, "run", plan(5))
    first.reserve("save-3", "save")
    store.close()

    # Crash: the process dies between reserving the save and settling it.
    reopened = JobStore(tmp_path / "jobs.sqlite")
    budget = TrainingBudget(reopened, "run", plan(5))
    with pytest.raises(BudgetError, match="do not replay"):
        budget.reserve("save-3", "save")
    assert budget.ledger.operation("save-3")["reserved_microusd"] == 250_000
    assert budget.ledger.snapshot()["counted_or_reserved_usd"] == 0.25
    reopened.close()


def test_a_restart_does_not_hand_back_a_spent_cap(tmp_path):
    """Reconnect/retry must not reset the cap. Sixteen saves exhaust a $4 cap;
    reopening must not buy a seventeenth."""
    store = store_with_job(tmp_path)
    budget = TrainingBudget(store, "run", plan(4))
    for index in range(16):
        budget.reserve(f"save-{index}", "save")
        budget.settle(f"save-{index}", TrainingStepResult(f"save-{index}", 1, {}, _usage({})))
    assert budget.ledger.snapshot()["counted_or_reserved_usd"] == 4.0
    store.close()

    reopened = JobStore(tmp_path / "jobs.sqlite")
    resumed = TrainingBudget(reopened, "run", plan(4))
    with pytest.raises(BudgetError, match="aggregate reservation"):
        resumed.reserve("save-16", "save")
    assert resumed.ledger.snapshot()["counted_or_reserved_usd"] == 4.0
    reopened.close()


def test_a_restart_cannot_raise_the_cap(tmp_path):
    """Coming back with a larger cap is how a resumed run would quietly buy
    more than it was authorized."""
    store = store_with_job(tmp_path)
    TrainingBudget(store, "run", plan(1)).reserve("session-0", "session")
    store.close()

    reopened = JobStore(tmp_path / "jobs.sqlite")
    with pytest.raises(BudgetError, match="cannot be reset"):
        TrainingBudget(reopened, "run", plan(20))
    reopened.close()


def test_an_unsettled_flat_operation_blocks_admission_until_reconciled(tmp_path):
    """An operation whose outcome is unknown is not a free slot. Nothing new is
    admitted while it is outstanding, so an uncertain save cannot be worked
    around by starting the next one."""
    store = store_with_job(tmp_path)
    budget = TrainingBudget(store, "run", plan(5))
    budget.reserve("save-0", "save")
    budget.reserve("save-1", "save")
    # Settling one leaves the other outstanding but does not free anything.
    budget.settle("save-0", TrainingStepResult("save-0", 1, {}, _usage({})))
    assert budget.ledger.operation("save-1")["counted_microusd"] is None
    assert budget.ledger.snapshot()["counted_or_reserved_usd"] == 0.5
    store.close()


def test_a_settlement_survives_reopen_and_cannot_be_rewritten(tmp_path):
    """Settlement is evidence. A resumed worker must not be able to restate what
    a charge cost, in either direction."""
    store = store_with_job(tmp_path)
    budget = TrainingBudget(store, "run", plan(5))
    budget.reserve("restore-0", "restore")
    budget.settle("restore-0", TrainingStepResult("restore-0", 1, {}, _usage({})))
    store.close()

    reopened = JobStore(tmp_path / "jobs.sqlite")
    resumed = TrainingBudget(reopened, "run", plan(5))
    charge = resumed.ledger.operation("restore-0")
    assert (charge["status"], charge["counted_microusd"]) == ("conservative", 250_000)
    # Re-settling identically is a safe no-op; restating the cost is refused.
    resumed.settle("restore-0", TrainingStepResult("restore-0", 1, {}, _usage({})))
    with pytest.raises(BudgetError, match="settlement cannot be rewritten"):
        resumed.ledger.settle("restore-0", 0.01)
    assert resumed.ledger.operation("restore-0")["counted_microusd"] == 250_000
    reopened.close()


def test_two_jobs_sharing_an_experiment_cannot_oversubscribe_across_a_restart(tmp_path):
    """SFT and CISPO share one aggregate. A restart of either must not give the
    pair a fresh allowance."""
    shared = {"experiment_id": "authorized-canaries", "max_cost_usd": 1, "pricing": PRICING}
    store = store_with_job(tmp_path)
    TrainingBudget(store, "sft", shared).reserve("sft-save", "save")
    TrainingBudget(store, "cispo", shared).reserve("cispo-save", "save")
    store.close()

    reopened = JobStore(tmp_path / "jobs.sqlite")
    resumed = TrainingBudget(reopened, "cispo", shared)
    assert resumed.ledger.snapshot()["counted_or_reserved_usd"] == 0.5
    for index in range(2):
        resumed.reserve(f"cispo-save-{index}", "save")
    with pytest.raises(BudgetError, match="aggregate reservation"):
        resumed.reserve("cispo-save-overflow", "save")
    reopened.close()
