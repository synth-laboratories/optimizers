import pytest

from synth_optimizers.rl.evidence import EvidenceStore


def test_immutable_evidence_preserves_text_and_survives_restart(tmp_path):
    path = tmp_path/'evidence.db'
    store = EvidenceStore(path)
    trace = {'answer': 'Hello, World!\n["move_left", "do"]', 'sealed': True}
    digest = store.record('rollout', trace, {'score': .5})
    assert store.record('rollout', trace, {'score': .5}) == digest
    assert EvidenceStore(path).get('rollout')['trace'] == trace
    with pytest.raises(ValueError, match='changed'):
        store.record('rollout', trace, {'score': .6})


def test_credentials_are_refused_not_silently_written(tmp_path):
    store = EvidenceStore(tmp_path/'evidence.db')
    with pytest.raises(ValueError, match='credential'):
        store.record('rollout', {'headers': {'Authorization': 'secret'}}, {})
    with pytest.raises(KeyError):
        store.get('rollout')
