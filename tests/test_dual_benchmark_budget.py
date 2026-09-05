from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/e2e'))
import dual_benchmark_budget as budget  # noqa: E402


def test_concurrent_reservations_cannot_cross_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'ROOT', tmp_path)
    def claim(_):
        try:
            return budget.reserve('test', 10)
        except RuntimeError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = [k for k in pool.map(claim, range(8)) if k]
    assert len(keys) == 4
    with budget.connect() as db:
        assert db.execute('SELECT SUM(reserved) FROM charges').fetchone()[0] == 40
    budget.settle(keys[0], 1, {'test':True})
    assert budget.reserve('test', 10)


def test_missing_usage_keeps_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'ROOT', tmp_path)
    key = budget.reserve('unknown outcome', 48)
    with pytest.raises(RuntimeError):
        budget.reserve('next', .01)
    with pytest.raises(RuntimeError):
        budget.settle(key, 49, {})


def test_low_disk_refuses_before_reserving_paid_work(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'ROOT', tmp_path)
    monkeypatch.setattr(budget.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1024**3))
    with pytest.raises(RuntimeError, match='less than 2 GiB'):
        budget.reserve('must not execute', .01)
    assert not (tmp_path / 'budget.sqlite3').exists()
