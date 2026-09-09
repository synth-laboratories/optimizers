from types import SimpleNamespace
import json

import pytest

from synth_optimizers.rl.budget import ExperimentBudget
from synth_optimizers.rl.daytona_substrate import DaytonaSubstrate, DaytonaWorkspace


def test_owned_sandbox_reservation_and_idempotent_cleanup(tmp_path):
    pytest.importorskip('daytona')
    calls = []
    budget = ExperimentBudget(tmp_path/'budget.db', 'test', 1)
    sandbox = SimpleNamespace(id='sandbox', labels={'experiment':'test'})
    def create(params, **kwargs):
        assert budget.snapshot()['unsettled_operations'] == 1
        assert params.network_block_all is True
        assert params.ttl_minutes == 20
        assert params.env_vars is None
        return sandbox
    client = SimpleNamespace(create=create, delete=lambda obj, **kwargs: calls.append(obj.id))
    substrate = DaytonaSubstrate(client,budget,tmp_path/'receipts',workers=1,verifier_workers=1)
    found = substrate._create('snapshot', 'rollout', 'agent')
    substrate._delete(found)
    substrate._delete(found)
    substrate.close()
    assert calls == ['sandbox']
    assert budget.snapshot()['unsettled_operations'] == 0


def test_ambiguous_create_is_reconciled_without_replay(tmp_path):
    pytest.importorskip('daytona')
    calls = []
    budget = ExperimentBudget(tmp_path/'budget.db', 'test', 1)
    sandbox = SimpleNamespace(id='sandbox', labels={'experiment':'test'})
    def create(*args, **kwargs):
        raise TimeoutError('lost response')
    client = SimpleNamespace(create=create, get=lambda name, **kwargs:sandbox,
                             delete=lambda obj, **kwargs:calls.append(obj.id))
    substrate = DaytonaSubstrate(client,budget,tmp_path/'receipts',workers=1,verifier_workers=1)
    with pytest.raises(TimeoutError):
        substrate._create('snapshot','rollout','agent')
    assert calls == ['sandbox']
    assert budget.snapshot()['unsettled_operations'] == 1
    with pytest.raises(Exception, match='reconcile'):
        substrate._create('snapshot','rollout','agent')
    substrate.close()


def test_sealed_workspace_rejects_further_commands():
    workspace = DaytonaWorkspace(None,SimpleNamespace(id='id'),SimpleNamespace(workspace='/app'),'rollout')
    workspace.manifest = {'content_digest':'sha256:frozen'}
    with pytest.raises(RuntimeError,match='sealed'):
        workspace.run('echo mutation',timeout_seconds=1)


def test_workspace_enforces_controller_command_bound():
    pytest.importorskip('harbor_tblite')
    calls=[]
    def execute(command,**kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(exit_code=124,result='timed out')
    owner=SimpleNamespace(max_command_seconds=60)
    sandbox=SimpleNamespace(id='sandbox',process=SimpleNamespace(exec=execute))
    workspace=DaytonaWorkspace(owner,sandbox,SimpleNamespace(workspace='/app'),'rollout')
    outcome=workspace.run('sleep 300',timeout_seconds=300)
    assert outcome.exit_code==124
    assert 'timeout --signal=TERM --kill-after=5s 60s' in calls[0][0]
    assert calls[0][1]['timeout']==70


def test_background_process_does_not_hold_capture_pipe():
    import os
    import shlex
    import signal
    import subprocess
    import sys
    from synth_optimizers.rl.daytona_substrate import _background_safe_capture
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']); print(p.pid); print('parent finished')"
    result=subprocess.run(shlex.split(_background_safe_capture(shlex.join([sys.executable,'-c',code]))),
        capture_output=True,text=True,timeout=3)
    pid=int(result.stdout.splitlines()[0])
    try:
        assert result.returncode==0
        assert 'parent finished' in result.stdout
        os.kill(pid,0)
    finally:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass


def test_background_safe_capture_preserves_exit_and_stderr():
    import shlex
    import subprocess
    import sys
    from synth_optimizers.rl.daytona_substrate import _background_safe_capture
    command=shlex.join([sys.executable,'-c',"import sys;print('error',file=sys.stderr);sys.exit(17)"])
    result=subprocess.run(shlex.split(_background_safe_capture(command)),capture_output=True,text=True,timeout=3)
    assert result.returncode==17
    assert result.stdout=='error\n'


def test_pipe_capture_reproduces_background_descriptor_hang():
    import os
    import signal
    import subprocess
    import sys
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']); print(p.pid,flush=True)"
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        subprocess.run([sys.executable,'-c',code],capture_output=True,timeout=1)
    pid=int(failure.value.stdout.splitlines()[0])
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass


def test_delete_lost_response_requires_independent_absence(tmp_path):
    daytona = pytest.importorskip('daytona')
    budget = ExperimentBudget(tmp_path/'budget.db', 'test', 1)
    sandbox = SimpleNamespace(id='sandbox',labels={'experiment':'test'})
    def delete(*args, **kwargs):
        raise TimeoutError('delete response lost')
    def get(*args, **kwargs):
        raise daytona.DaytonaNotFoundError('not found')
    client = SimpleNamespace(create=lambda *a, **k:sandbox, delete=delete, get=get)
    substrate = DaytonaSubstrate(client,budget,tmp_path/'receipts',workers=1,verifier_workers=1)
    substrate._delete(substrate._create('snapshot','rollout','agent'))
    assert budget.snapshot()['unsettled_operations']==0
    substrate.close()


@pytest.mark.parametrize('delete_fails',[False,True])
@pytest.mark.parametrize('score,tampered,expected',[(0,False,0),(1,False,1),(1,True,0)])
def test_direct_grader_uses_score_not_exit_and_protects_inputs(tmp_path,monkeypatch,score,tampered,expected,delete_fails):
    pytest.importorskip('harbor_tblite')
    budget=ExperimentBudget(tmp_path/'budget.db','test',1)
    runtime=DaytonaSubstrate(None,budget,tmp_path/'receipts',workers=1,verifier_workers=1,
        binary_grader_tasks=['task'],immutable_inputs={'task':('data',)})
    manifest={'content_digest':'sha256:example','baseline':{'data/input':'a'},
              'final':{'data/input':'b' if tampered else 'a'}}
    sandbox=SimpleNamespace(id='sandbox',fs=SimpleNamespace(upload_file=lambda *a:None,
        download_file=lambda *a:json.dumps({'reward':score}).encode()),
        process=SimpleNamespace(exec=lambda *a,**k:SimpleNamespace(exit_code=0,result='')))
    monkeypatch.setattr(runtime,'_create',lambda *a:sandbox)
    def delete(*args):
        if delete_fails:
            raise TimeoutError('provider DELETE response failed')
    monkeypatch.setattr(runtime,'_delete',delete)
    monkeypatch.setattr(runtime,'_exec',lambda *a:json.dumps(manifest))
    workspace=SimpleNamespace(content_digest=lambda:'sha256:example',release=lambda:None,archive=b'zip')
    trial=SimpleNamespace(task_id='task',verifier_image='image',workspace='/app',verifier_timeout_seconds=10)
    assert runtime._verify(trial,workspace,'rollout').reward==expected
    assert (runtime.root/'cleanup-pending-sandbox.json').exists()==delete_fails
    runtime.close()
