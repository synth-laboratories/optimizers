from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from synth_optimizers.rl.experiment import CoordinationError, ExperimentSpec, ExperimentStore


def spec(tmp_path):
    return ExperimentSpec.model_validate({
        'experiment_id': 'exp-a', 'updates': 3, 'segment_updates': 2,
        'evaluation_url': 'http://localhost:9998',
        'validation_updates': [1,3], 'judge_protocol_digest': 'sha256:'+'1'*64,
        'run': {'schema_version': 'cispo.container.v1', 'container': {'url': 'http://localhost:9999'},
                'model': {'provider': 'tinker', 'id': 'openai/gpt-oss-20b', 'family': 'gpt_oss'},
                'plan': {'preset': 'cispo'}, 'reward': {'optimized_channel': 'score'},
                'budget': {'experiment_id': 'exp-a', 'ledger': str(tmp_path/'budget.db'), 'cap_usd': 10,
                           'input_usd_per_million': 1, 'output_usd_per_million': 1, 'training_usd_per_million': 1}},
        **{split: [{'task_id': split, 'seed': 1, 'content_digest': 'sha256:'+'2'*64}]
           for split in ('train','validation','final')}})


def test_frozen_design_and_split_disjointness(tmp_path):
    design = spec(tmp_path)
    assert all(p.get('updates', 1) <= 2 for p in design.phases())
    payload = design.model_dump()
    payload['final'] = payload['train']
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_completion_restart_and_events(tmp_path):
    path = tmp_path/'experiment.db'
    store = ExperimentStore(path)
    design = spec(tmp_path)
    store.submit(design)
    store.submit(design)
    for _ in design.phases():
        claim = store.claim('exp-a')
        assert claim
        store.complete('exp-a', claim, {'verified': True})
        store = ExperimentStore(path)
    assert store.snapshot('exp-a')['state'] == 'completed'
    assert store.claim('exp-a') is None
    first = store.events('exp-a', limit=2)
    assert first == store.events('exp-a', limit=2)
    assert first['has_more']
    assert store.events('exp-a', after_sequence=first['next_sequence'])['events'][0]['sequence'] == 3


def test_expired_claim_never_automatically_replays(tmp_path):
    now = [100.0]
    store = ExperimentStore(tmp_path/'experiment.db', clock=lambda: now[0])
    store.submit(spec(tmp_path))
    old = store.claim('exp-a', lease_seconds=2)
    now[0] += 3
    assert store.claim('exp-a') is None
    assert store.snapshot('exp-a')['state'] == 'blocked'
    with pytest.raises(CoordinationError):
        store.complete('exp-a', old, {})
    with pytest.raises(CoordinationError):
        store.control('exp-a', 'resume')


def test_only_one_worker_claims_phase(tmp_path):
    store = ExperimentStore(tmp_path/'experiment.db')
    store.submit(spec(tmp_path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: store.claim('exp-a'), range(8)))
    assert sum(c is not None for c in claims) == 1


def test_pause_does_not_admit_next_phase(tmp_path):
    store = ExperimentStore(tmp_path/'experiment.db')
    store.submit(spec(tmp_path))
    claim = store.claim('exp-a')
    store.control('exp-a', 'pause')
    store.complete('exp-a', claim, {})
    assert store.claim('exp-a') is None
    store.control('exp-a', 'resume')
    assert store.claim('exp-a')['position'] == 1


def test_changed_spec_rejected_without_new_events(tmp_path):
    store = ExperimentStore(tmp_path/'experiment.db')
    design = spec(tmp_path)
    store.submit(design)
    before = store.events('exp-a')
    with pytest.raises(CoordinationError):
        store.submit(design.model_copy(update={'updates': 4}))
    assert store.events('exp-a') == before


def test_recovery_cannot_resurrect_stopped_experiment(tmp_path):
    store = ExperimentStore(tmp_path/'experiment.db')
    store.submit(spec(tmp_path))
    claim = store.claim('exp-a')
    store.block('exp-a', claim, 'operation_uncertain')
    store.control('exp-a', 'stop')
    with pytest.raises(CoordinationError):
        store.reconcile_completed('exp-a', 0, {}, evidence_digest='sha256:'+'a'*64)
    assert store.snapshot('exp-a')['state'] == 'stopped'
