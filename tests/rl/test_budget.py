from concurrent.futures import ThreadPoolExecutor
import sqlite3
from types import SimpleNamespace

import pytest

from synth_optimizers.rl.budget import BudgetError, BudgetedProvider, ExperimentBudget
from synth_optimizers.rl.config import ConfigError, loads


def test_budget_survives_restart_and_refuses_cap_reset(tmp_path):
    path = tmp_path/'budget.db'
    b = ExperimentBudget(path, 'experiment', 1)
    b.reserve('a', 'sample', .6)
    b = ExperimentBudget(path, 'experiment', 1)
    assert b.snapshot()['unsettled_operations'] == 1
    with pytest.raises(BudgetError, match='cap cannot be reset'):
        ExperimentBudget(path, 'experiment', 2)
    with pytest.raises(BudgetError):
        b.reserve('b', 'sample', .5)
    b.settle('a', .2)
    b.settle('a', .2)
    b.reserve('b', 'sample', .5)
    assert b.snapshot()['counted_or_reserved_usd'] == .7


def test_explicit_cap_extension_preserves_charges_and_records_authority(tmp_path):
    b = ExperimentBudget(tmp_path/'budget.db', 'experiment', 1)
    b.reserve('a', 'sample', .6)
    with pytest.raises(ValueError):
        b.extend_cap(expected_cap_usd=1,new_cap_usd=2,authorization='')
    b.extend_cap(expected_cap_usd=1,new_cap_usd=2,authorization='user approval test')
    assert b.snapshot()['counted_or_reserved_usd']==.6
    assert b.snapshot()['cap_usd']==2
    assert b.events()[-1]['event_type']=='budget.cap_extended'
    with pytest.raises(BudgetError):
        b.extend_cap(expected_cap_usd=1,new_cap_usd=3,authorization='stale approval')
    assert ExperimentBudget(tmp_path/'budget.db','experiment',2).snapshot()['unsettled_operations']==1
    with pytest.raises(BudgetError):
        b.reserve('a','sample',.1)


def test_concurrent_budget_cannot_oversubscribe(tmp_path):
    b = ExperimentBudget(tmp_path/'budget.db', 'experiment', 1)
    def claim(i):
        try:
            b.reserve(str(i), 'sample', .3)
            return True
        except BudgetError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(16))) == 3
    assert b.snapshot()['counted_or_reserved_usd'] == .9


def test_budget_admission_uses_indexed_overrun_lookup(tmp_path):
    path = tmp_path/'budget.db'
    ExperimentBudget(path, 'experiment', 1)
    with sqlite3.connect(path) as db:
        plan = db.execute("""EXPLAIN QUERY PLAN SELECT 1 FROM budget_events exceeded
            WHERE exceeded.experiment=?
              AND json_extract(exceeded.payload,'$.reservation_exceeded')=1
              AND NOT EXISTS (SELECT 1 FROM budget_events reconciled
                WHERE reconciled.experiment=exceeded.experiment
                  AND reconciled.kind='budget.pricing_reconciled'
                  AND json_extract(reconciled.payload,'$.operation_id')=
                      json_extract(exceeded.payload,'$.operation_id')) LIMIT 1""",
            ('experiment',)).fetchall()
    detail = ' '.join(row[3] for row in plan)
    assert 'budget_unreconciled_lookup' in detail
    assert 'budget_reconciliation_lookup' in detail
    assert 'SCAN' not in detail


def test_overrun_liability_is_saved_and_admission_blocked(tmp_path):
    b = ExperimentBudget(tmp_path/'budget.db', 'experiment', 10)
    b.reserve('a', 'sample', .1)
    with pytest.raises(BudgetError):
        b.settle('a', 2)
    assert b.snapshot()['counted_or_reserved_usd'] == 2
    with pytest.raises(BudgetError, match='prior call'):
        b.reserve('b', 'sample', .1)
    b.reconcile_pricing_overrun('a', evidence='observed provider usage')
    b.reserve('b', 'sample', .1)
    assert any(event['event_type'] == 'budget.pricing_reconciled' for event in b.events())


def test_unknown_cost_conservatively_settles_and_replay_refused(tmp_path):
    b = ExperimentBudget(tmp_path/'budget.db', 'experiment', 1)
    b.reserve('a', 'sample', .3)
    b.settle('a')
    assert b.snapshot()['counted_or_reserved_usd'] == .3
    with pytest.raises(BudgetError, match='reconcile'):
        b.reserve('a', 'sample', .3)


def test_provider_error_keeps_reservation_and_prevents_repeat(tmp_path):
    calls = []
    def sample(identity, request):
        calls.append(request.request_id)
        raise TimeoutError('uncertain provider result')
    b = ExperimentBudget(tmp_path/'budget.db', 'experiment', 1)
    provider = BudgetedProvider(SimpleNamespace(sample=sample), b,
                                input_rate=1, output_rate=1, training_rate=1)
    request = SimpleNamespace(request_id='a', prompt_token_ids=(1,2), max_tokens=10)
    with pytest.raises(TimeoutError):
        provider.sample(None, request)
    with pytest.raises(BudgetError):
        provider.sample(None, request)
    assert calls == ['a']
    assert b.snapshot()['counted_or_reserved_usd'] == .000012


@pytest.mark.parametrize('value', ['nan', 'inf', '-1'])
def test_invalid_amounts_refused(tmp_path, value):
    with pytest.raises(ValueError):
        ExperimentBudget(tmp_path/'budget.db', 'experiment', value)


def test_config_budget_is_explicit_and_redacted(tmp_path):
    config = loads(f'''schema_version = "cispo.container.v1"
[container]
url = "http://localhost:9000"
[model]
provider = "tinker"
id = "openai/gpt-oss-20b"
family = "gpt_oss"
[plan]
preset = "cispo"
[reward]
optimized_channel = "score"
[budget]
experiment_id = "exp-a"
ledger = "{tmp_path}/budget.db"
cap_usd = 10
input_usd_per_million = 1
output_usd_per_million = 2
training_usd_per_million = 3
''')
    assert config.budget.experiment_id == 'exp-a'
    assert config.redacted_payload()['budget']['cap_usd'] == 10
    with pytest.raises(ConfigError):
        loads('schema_version="cispo.container.v1"\n[budget]\ncap_usd=nan')
