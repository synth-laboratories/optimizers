from __future__ import annotations

import importlib
import json
import threading
import time
from pathlib import Path


def test_parallel_validation_uses_distinct_ports_and_fixed_selection(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'docs/e2e'))
    driver = importlib.import_module('run_banking77_fast50')
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    (tmp_path / 'training_resume25').mkdir()
    (tmp_path / 'training_resume25/manifest.json').write_text(json.dumps({'stop_reason': 'target_train_updates_reached'}))
    (tmp_path / 'artifact_digests.json').write_text('{}')
    monkeypatch.setattr(driver, 'checkpoints', lambda: [
        {'policy_revision_id': f'pg-0@{i}', 'checkpoint_id': f'checkpoint-{i}', 'artifacts': {}}
        for i in range(25, 75)
    ])
    lock = threading.Lock()
    active = set()
    peak = 0
    calls = []

    def evaluate(name, selected, baseline, panel, port):
        nonlocal peak
        with lock:
            assert port not in active
            active.add(port)
            peak = max(peak, len(active))
            calls.append((name, selected, baseline, panel))
        time.sleep(0.03)
        with lock:
            active.remove(port)
        revision = int(selected.split('-')[-1])
        return {'trained_checkpoint_id': selected, 'trained_mean': 0.9 if revision in [44, 64] else 0.8}

    monkeypatch.setattr(driver, 'evaluate', evaluate)
    driver.heldout()
    assert peak == 4
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert selection['revision'] == 44
    finals = [c for c in calls if c[3] == 'final']
    assert len(finals) == 2
    assert all(c[1] == 'checkpoint-44' for c in finals)
    assert {c[2] for c in finals} == {driver.BASELINE, driver.PARENT}
    assert not active
