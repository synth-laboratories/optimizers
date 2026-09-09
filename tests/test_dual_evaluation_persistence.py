from pathlib import Path
import sys
from types import SimpleNamespace
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/e2e'))
from evaluate_dual_benchmark import PersistedEvaluation, PairedEvaluation  # noqa: E402


def test_observed_row_and_evidence_persist_before_full_panel_completes(tmp_path, monkeypatch):
    row = SimpleNamespace(arm='baseline', sample_index=3, rollout_id='rollout-test',
                          to_payload=lambda: {'reward': .5, 'seed': 12})
    monkeypatch.setattr(PairedEvaluation, '_row', lambda *args, **kwargs: row)
    evaluator = PersistedEvaluation.__new__(PersistedEvaluation)
    evaluator._output = tmp_path
    evaluator._session = SimpleNamespace(reward_payload=lambda _: {'measure': .5},
                                         trace=lambda _: {'sealed': True})
    assert evaluator._row() is row
    assert json.loads((tmp_path/'attempts/baseline_3.json').read_text())['reward'] == .5
    assert json.loads((tmp_path/'rewards/rollout-test.json').read_text())['measure'] == .5
    assert json.loads((tmp_path/'traces/rollout-test.json').read_text())['sealed']
