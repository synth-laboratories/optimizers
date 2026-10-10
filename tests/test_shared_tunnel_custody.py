"""Adverse local custody boundaries; canonical API proof lives in tunnel E2E."""
import os
from pathlib import Path
from uuid import UUID, uuid4
from datetime import UTC, datetime, timedelta

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


@pytest.mark.parametrize('uncertain_close', [False, True])
def test_external_revocation_settles_close_without_effect_replay(tmp_path, uncertain_close):
    connector, lease, owner = map(str, (uuid4(), uuid4(), uuid4()))
    class Backend:
        calls = []
        def _json_request(self, method, path, payload=None):
            self.calls.append((method, path))
            assert method == 'GET' and path.endswith('/leases/' + lease)
            return {'connector': connector, 'lease': lease, 'revision': 2,
                    'owner_binding': owner, 'policy': 'revoked'}
    backend = Backend()
    journal = FileLeaseCustodyJournal(tmp_path, journal_lease := UUID(lease))
    receipt = {'connector': connector, 'phase': 'granted', 'grant_request': {'owner_binding': owner}}
    if uncertain_close:
        receipt.update(phase='close_in_doubt', close_request={'command_id': str(uuid4())})
    custody = CallerLeaseCustody(backend, journal, journal.save(receipt))
    custody.close()
    restored = FileLeaseCustodyJournal(tmp_path, journal_lease)
    CallerLeaseCustody(backend, restored, restored.read()).close()
    assert len(backend.calls) == 1
    assert restored.read()['close_observation']['disposition'] == 'canonical_revoked'
    assert restored.read()['phase'] == 'closed'


def test_expired_owned_grant_can_close_but_cannot_regain_authority(tmp_path):
    connector, lease, owner = map(str, (uuid4(), uuid4(), uuid4()))
    command = None
    class Backend:
        calls = []
        def _json_request(self, method, path, payload=None):
            nonlocal command
            self.calls.append((method, path))
            if method == 'POST':
                command = payload['command_id']
                return {}
            if '/commands/' in path:
                return {'connector': connector, 'command': command}
            if path.endswith('/leases/' + lease):
                return {'connector': connector, 'lease': lease, 'revision': 1,
                        'owner_binding': owner, 'policy': 'granted',
                        'expires_at': (datetime.now(UTC)-timedelta(seconds=1)).isoformat()}
            return {'revision': 8}
    backend = Backend()
    journal = FileLeaseCustodyJournal(tmp_path, UUID(lease))
    custody = CallerLeaseCustody(backend, journal, journal.save({
        'connector': connector, 'phase': 'granted', 'grant_request': {'owner_binding': owner}}))
    with pytest.raises(TunnelError, match='expired or revoked'):
        custody.settle_grant()
    custody.close()
    assert [method for method, _ in backend.calls].count('POST') == 1
    assert journal.read()['phase'] == 'closed'


def test_revoked_other_owner_cannot_settle_caller_close(tmp_path):
    connector, lease, owner = map(str, (uuid4(), uuid4(), uuid4()))
    class Backend:
        def _json_request(self, method, path, payload=None):
            assert method == 'GET'
            return {'connector': connector, 'lease': lease, 'revision': 2,
                    'owner_binding': str(uuid4()), 'policy': 'revoked'}
    journal = FileLeaseCustodyJournal(tmp_path, UUID(lease))
    custody = CallerLeaseCustody(Backend(), journal, journal.save({
        'connector': connector, 'phase': 'granted', 'grant_request': {'owner_binding': owner}}))
    with pytest.raises(TunnelError, match='custody is stale'):
        custody.close()
    assert journal.read()['phase'] == 'granted'
