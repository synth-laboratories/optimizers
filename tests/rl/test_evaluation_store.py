from dataclasses import dataclass

import pytest

from synth_optimizers.rl.evaluation_store import EvaluationStore


@dataclass
class Request:
    evaluation_id: str = 'eval-a'


def test_observation_survives_restart_and_refuses_fresh_label(tmp_path):
    path = tmp_path/'eval.db'
    store = EvaluationStore(path)
    store.begin(Request())
    store = EvaluationStore(path)
    assert store.snapshot('eval-a')['observed']
    with pytest.raises(ValueError, match='already observed'):
        store.begin(Request())
    row = {'arm': 'baseline', 'sample_index': 0, 'reward': 0.5, 'text': 'Use JSON: {"Action": "LEFT"}'}
    store.record('eval-a', row)
    store.record('eval-a', row)
    with pytest.raises(ValueError, match='immutable'):
        store.record('eval-a', {**row, 'reward': 1})
    assert store.snapshot('eval-a')['attempts'] == [row]
