"""Adverse local custody boundaries; canonical API proof lives in tunnel E2E."""
import os
from pathlib import Path
from uuid import uuid4

import pytest

from synth_optimizers.tunnel_custody import CallerLeaseCustody
from synth_optimizers.tunnel_custody_journal import FileLeaseCustodyJournal, TunnelCustodyInDoubt
from synth_optimizers.tunnels import TunnelError


def test_receipt_cas_fences_two_recovered_writers_and_refuses_special_files(tmp_path):
    identity = uuid4()
    first = FileLeaseCustodyJournal(tmp_path, identity)
    initial = first.save({'phase': 'granted'})
    second = FileLeaseCustodyJournal(tmp_path, identity)
    assert second.read() == initial
    first.save({'phase': 'offered'})
    with pytest.raises(TunnelError, match='revision conflict'):
        second.save({'phase': 'closed'})
    identity = uuid4()
    unsafe = FileLeaseCustodyJournal(tmp_path, identity)
    os.mkfifo(unsafe.path, 0o600)
    with pytest.raises(TunnelError, match='not private'):
        unsafe.read()  # Must refuse without blocking on a FIFO writer.
    unsafe.path.unlink()
    first.path.chmod(0o644)
    with pytest.raises(TunnelError, match='not private'):
        first.read()


def test_submit_reply_loss_is_retained_and_never_replayed_after_process_recovery(tmp_path):
    class Backend:
        calls = 0
        def _json_request(self, method, path, payload=None):
            assert method == 'POST'
            self.calls += 1
            raise TimeoutError('committed reply lost')
    backend = Backend()
    journal = FileLeaseCustodyJournal(tmp_path, uuid4())
    receipt = journal.save({'connector': str(uuid4()), 'phase': 'offered',
                            'offer_request': {'job_id': 'stable_job'}})
    custody = CallerLeaseCustody(backend, journal, receipt)
    payload = {'run_id': 'stable_job', 'algorithm': 'gepa', 'config_json': {}}
    with pytest.raises(TunnelCustodyInDoubt, match='submission'):
        custody.submit_once(payload)
    other = FileLeaseCustodyJournal(tmp_path, journal.lease)
    restored = CallerLeaseCustody(backend, other, other.read())
    with pytest.raises(TunnelCustodyInDoubt):
        restored.submit_once(payload)
    with pytest.raises(TunnelError, match='different input'):
        restored.submit_once({**payload, 'algorithm': 'gelo'})
    assert backend.calls == 1
    assert 'config_json' not in restored.receipt['submission']


def test_local_commit_uncertainty_is_not_reported_as_a_success(tmp_path, monkeypatch):
    journal = FileLeaseCustodyJournal(tmp_path, uuid4())
    original = os.fsync
    def fail_directory(descriptor):
        import stat
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError('fixture directory sync failure')
        original(descriptor)
    monkeypatch.setattr(os, 'fsync', fail_directory)
    with pytest.raises(TunnelCustodyInDoubt, match='local_commit'):
        journal.save({'phase': 'grant_in_doubt'})
    assert Path(journal.path).is_file()
    assert not list(tmp_path.glob('*.tmp'))
