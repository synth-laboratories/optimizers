"""Owned, budgeted Daytona agent/verifier siblings for the TBLite contract.

Credentials remain in the controller. Agents run as a non-root UID with no
network. Sealing terminates that UID before hashing; verifiers receive only a
validated workspace delta in a separate pristine snapshot.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import threading
import time
import zipfile

from .daytona_workspace_control import validate
from .screening import _write_json

CONTROL = '/root/rl-workspace-control.py'
AGENT_UID = 10001


def _background_safe_capture(command):
    """Keep background descendants from holding the provider's stdout pipe open.

    The command still runs once under its existing timeout and UID boundary.
    A regular anonymous file captures output; only its bounded completion-time
    prefix is forwarded. Background services can survive until workspace seal.
    """
    body = ('import os, subprocess, sys, tempfile\n'
        'with tempfile.TemporaryFile() as output:\n'
        '    result = subprocess.run('+repr(command)+', shell=True, stdout=output, stderr=subprocess.STDOUT)\n'
        '    limit = 8 * 1024 * 1024\n'
        '    size = os.fstat(output.fileno()).st_size\n'
        '    sys.stdout.buffer.write(os.pread(output.fileno(), min(size, limit), 0))\n'
        '    if size > limit: sys.stdout.buffer.write(b"\\n[command output truncated at 8 MiB]\\n")\n'
        'sys.exit(result.returncode)\n')
    return 'python3 -I -c '+shlex.quote(body)


class DaytonaSubstrate:
    kind = 'daytona'

    def __init__(self, client, budget, root, *, workers=96, verifier_workers=16,
                 ttl_minutes=20, cpu=1, memory=2, disk=3, binary_grader_tasks=(),
                 immutable_inputs=None, max_command_seconds=None, full_credit_grader_tasks=(),
                 create_timeout_seconds=90):
        if not 1 <= workers <= 128 or not 1 <= verifier_workers <= workers:
            raise ValueError('invalid bounded Daytona concurrency')
        self.client, self.budget, self.root = client, budget, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_minutes = ttl_minutes
        self.hourly = cpu*.0504 + memory*.0162 + disk*.000108
        self._slots = threading.BoundedSemaphore(workers)
        self._pool = ThreadPoolExecutor(max_workers=verifier_workers, thread_name_prefix='daytona-verifier')
        self._jobs, self._owned, self._deleting = {}, {}, set()
        self._lock = threading.RLock()
        self._closed = False
        self.binary_grader_tasks = frozenset(binary_grader_tasks)
        self.full_credit_grader_tasks = frozenset(full_credit_grader_tasks)
        if not self.full_credit_grader_tasks <= self.binary_grader_tasks:
            raise ValueError('full-credit grading requires a reviewed direct grader')
        self.immutable_inputs = dict(immutable_inputs or {})
        if max_command_seconds is not None and not 1 <= max_command_seconds <= 600:
            raise ValueError('invalid command time bound')
        self.max_command_seconds = max_command_seconds
        if not 30 <= create_timeout_seconds <= 600:
            raise ValueError('invalid sandbox creation timeout')
        self.create_timeout_seconds = create_timeout_seconds

    def ready(self):
        return (not self._closed, 'daytona_owned_snapshot_substrate')

    def _create(self, image, rollout_id, lane):
        from daytona import CreateSandboxFromSnapshotParams
        key = hashlib.sha256(f'{rollout_id}:{lane}'.encode()).hexdigest()[:28]
        name = 'rl-'+key
        operation = 'daytona:'+name
        self._slots.acquire()
        try:
            if self._closed:
                raise RuntimeError('Daytona substrate closed')
            # Include provisioning and deletion headroom. Ambiguous outcomes
            # retain the full ceiling; never replay the same operation ID.
            self.budget.reserve(operation, 'daytona_'+lane,
                                self.hourly*(self.ttl_minutes*60+180)/3600)
            started = time.monotonic()
            receipt = {'name': name, 'operation_id': operation, 'lane': lane,
                       'snapshot': image, 'rollout_id': rollout_id, 'status': 'creating',
                       'created_at': time.time(), 'ttl_minutes': self.ttl_minutes}
            _write_json(self.root/(name+'.json'), receipt)
            sandbox = self.client.create(CreateSandboxFromSnapshotParams(
                name=name, snapshot=image,
                labels={'experiment': self.budget.experiment_id, 'rl_operation': key},
                ephemeral=True, auto_stop_interval=0, auto_delete_interval=0,
                ttl_minutes=self.ttl_minutes, network_block_all=True), timeout=self.create_timeout_seconds)
            receipt.update(sandbox_id=sandbox.id, status='running')
            with self._lock:
                self._owned[sandbox.id] = (sandbox, operation, started, receipt)
            _write_json(self.root/(name+'.json'), receipt)
            return sandbox
        except BaseException:
            # A timed-out create may have succeeded. Reconcile only this
            # pre-reserved exact name; never create a replacement blindly.
            if 'receipt' in locals() and 'sandbox' not in locals():
                try:
                    found = self.client.get(name, request_timeout=15)
                    if found.labels.get('experiment') != self.budget.experiment_id:
                        raise RuntimeError('ambiguous sandbox ownership')
                    self.client.delete(found, timeout=60, wait=True)
                    receipt.update(sandbox_id=found.id, status='deleted_after_ambiguous_create')
                except Exception:
                    receipt['status'] = 'reconciliation_required'
                _write_json(self.root/(name+'.json'), receipt)
            self._slots.release()
            raise

    def _delete(self, sandbox):
        with self._lock:
            owned = self._owned.get(sandbox.id)
            if owned is None or sandbox.id in self._deleting:
                return
            self._deleting.add(sandbox.id)
            _, operation, started, receipt = owned
        try:
            try:
                self.client.delete(sandbox, timeout=60, wait=True)
            except Exception:
                # TTL or a successful DELETE with a lost response may already
                # have removed it. Only an independent exact-ID 404 confirms
                # that; authentication and connection failures still fail shut.
                from daytona import DaytonaNotFoundError
                try:
                    self.client.get(sandbox.id, request_timeout=15)
                except DaytonaNotFoundError:
                    pass
                else:
                    raise
            duration = time.monotonic()-started
            receipt.update(status='deleted', deleted_at=time.time(), duration_seconds=duration)
            _write_json(self.root/(receipt['name']+'.json'), receipt)
            # SDK timing is a conservative local meter, not a provider invoice.
            self.budget.settle(operation, self.hourly*(duration+10)/3600, duration_seconds=duration)
            with self._lock:
                del self._owned[sandbox.id]
            self._slots.release()
        finally:
            with self._lock:
                self._deleting.discard(sandbox.id)

    @staticmethod
    def _exec(sandbox, command, timeout=60):
        outcome = sandbox.process.exec(command, timeout=timeout)
        if outcome.exit_code != 0:
            raise RuntimeError(f'Daytona trusted operation failed ({outcome.exit_code}): {outcome.result[:1500]}')
        return outcome.result

    def extract(self, *, trial, rollout_id):
        sandbox = self._create(trial.agent_image, rollout_id, 'agent')
        try:
            self._exec(sandbox, f'python3 -I {CONTROL} baseline {shlex.quote(trial.workspace)} /root/baseline.json')
            self._exec(sandbox, f'chown -R {AGENT_UID}:{AGENT_UID} {shlex.quote(trial.workspace)}')
            return DaytonaWorkspace(self, sandbox, trial, rollout_id)
        except BaseException:
            self._delete(sandbox)
            raise

    def submit_verifier(self, *, trial, workspace, rollout_id):
        with self._lock:
            if rollout_id not in self._jobs:
                if self._closed:
                    raise RuntimeError('Daytona substrate closed')
                # Runtime admission bounds the number of outstanding attempts.
                self._jobs[rollout_id] = self._pool.submit(self._verify, trial, workspace, rollout_id)
        return rollout_id

    def _verify(self, trial, workspace, rollout_id):
        from harbor_tblite.cispo import VerifierOutcome
        workspace.content_digest()
        # Reclaim the agent before allocating its verifier: a full admission
        # wave must not deadlock waiting for twice the sandbox quota.
        workspace.release()
        sandbox = self._create(trial.verifier_image, rollout_id, 'verifier')
        started = time.time()
        grade_persisted = False
        try:
            sandbox.fs.upload_file(workspace.archive, '/root/workspace.zip')
            restored = json.loads(self._exec(sandbox,
                f'python3 -I {CONTROL} restore {shlex.quote(trial.workspace)} /root/workspace.zip'))
            if restored['content_digest'] != workspace.content_digest():
                raise RuntimeError('workspace handoff mismatch')
            # These output-file tasks have a frozen, build-installed pytest
            # verifier. Its result comes from exit status, not an agent-written
            # reward file. Disable workspace conftest/plugin injection.
            direct = trial.task_id in self.binary_grader_tasks
            command = ('/opt/rl-verifier/bin/python -I /root/rl-binary-grader.py' if direct else
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/rl-verifier/bin/python -I -m pytest '
                '--confcutdir=/tests -p no:cacheprovider /tests/test_outputs.py -rA')
            if trial.task_id in self.full_credit_grader_tasks:
                command += ' --full-credit'
            result = sandbox.process.exec(command,
                cwd=trial.workspace, timeout=int(trial.verifier_timeout_seconds))
            if result.exit_code not in ((0,) if direct else (0, 1)):
                raise RuntimeError(f'verifier infrastructure failure: exit {result.exit_code}: {result.result[:1500]}')
            details = {}
            if direct:
                from .daytona_binary_grader import validate_score
                details = json.loads(sandbox.fs.download_file('/root/grade-result.json'))
                reward = validate_score(details['reward'])
            else:
                reward = float(result.exit_code == 0)
            protected = self.immutable_inputs.get(trial.task_id, ())
            tampered = [name for name in set(restored['baseline']) | set(restored['final'])
                        if any(name == p or name.startswith(p+'/') for p in protected)
                        and restored['baseline'].get(name) != restored['final'].get(name)]
            if tampered:
                reward = 0.0
            evidence = {'sandbox_id': sandbox.id, 'snapshot': trial.verifier_image,
                        'workspace_content_digest': restored['content_digest'],
                        'exit_code': result.exit_code, 'reward': reward, 'grader_details': details,
                        'modified_protected_inputs': tampered,
                        'started_at': started, 'finished_at': time.time(), 'output': result.result}
            _write_json(self.root/('grade-'+hashlib.sha256(rollout_id.encode()).hexdigest()+'.json'), evidence)
            grade_persisted = True
            return VerifierOutcome(handle=rollout_id, container=sandbox.id,
                image=trial.verifier_image, exit_code=result.exit_code, reward=reward,
                result=evidence, isolation_mechanism='daytona_separate_clean_snapshot',
                workspace_content_digest=restored['content_digest'])
        finally:
            try:
                self._delete(sandbox)
            except Exception as error:
                if not grade_persisted:
                    raise
                # Preserve an already durable grade across transient DELETE
                # failures. Ownership/reservation remains live; close() retries
                # and still fails if reconciliation cannot finish.
                _write_json(self.root/('cleanup-pending-'+sandbox.id+'.json'), {
                    'sandbox_id': sandbox.id, 'rollout_id': rollout_id,
                    'grade_persisted': True, 'error_type': type(error).__name__,
                    'recorded_at': time.time(), 'status': 'retry_at_close',
                })

    def poll_verifier(self, handle):
        future = self._jobs[handle]
        return future.result() if future.done() else None

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        errors = []
        for sandbox, *_ in list(self._owned.values()):
            try:
                self._delete(sandbox)
            except Exception as error:
                errors.append(type(error).__name__)
        if errors:
            raise RuntimeError('owned Daytona cleanup requires reconciliation: '+','.join(errors))


class DaytonaWorkspace:
    patch_media_type = 'application/vnd.synth.workspace-delta+json'

    def __init__(self, substrate, sandbox, trial, rollout_id):
        self.owner, self.sandbox, self.trial = substrate, sandbox, trial
        self.workspace_id = 'daytona:'+sandbox.id+':'+trial.workspace
        self.rollout_id, self.manifest, self.archive = rollout_id, None, None
        self._released = False
        self._lock = threading.RLock()

    def run(self, command, *, timeout_seconds):
        with self._lock:
            if self.manifest is not None or self._released:
                raise RuntimeError('workspace is sealed or released')
            from harbor_tblite.cispo import CommandOutcome
            if self.owner.max_command_seconds is not None:
                timeout_seconds = min(timeout_seconds,self.owner.max_command_seconds)
            started = time.monotonic()
            inner = 'cd '+shlex.quote(self.trial.workspace)+' && '+command
            bounded = ('timeout --signal=TERM --kill-after=5s '+str(max(1, int(timeout_seconds)))+
                's runuser -u rl-agent -- bash -lc '+shlex.quote(inner))
            outcome = self.sandbox.process.exec(
                _background_safe_capture(bounded),
                timeout=int(timeout_seconds)+10)
            return CommandOutcome(outcome.exit_code, outcome.result, '', time.monotonic()-started)

    def content_digest(self):
        with self._lock:
            if self.manifest is None:
                if self._released:
                    raise RuntimeError('unsealed workspace was released')
                self.owner._exec(self.sandbox, f'pkill -KILL -u {AGENT_UID} || test $? = 1')
                self.owner._exec(self.sandbox, f'! pgrep -u {AGENT_UID}')
                self.owner._exec(self.sandbox,
                    f'python3 -I {CONTROL} seal {shlex.quote(self.trial.workspace)} /root/baseline.json /root/workspace.zip')
                self.archive = self.sandbox.fs.download_file('/root/workspace.zip')
                with zipfile.ZipFile(io.BytesIO(self.archive)) as bundle:
                    self.manifest = validate(bundle)
                artifact = self.owner.root/('workspace-'+hashlib.sha256(self.rollout_id.encode()).hexdigest()+'.zip')
                with artifact.open('xb') as stream:
                    stream.write(self.archive)
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                self.workspace_id = str(artifact)
                _write_json(self.owner.root/('workspace-'+hashlib.sha256(self.rollout_id.encode()).hexdigest()+'.json'), self.manifest)
            return self.manifest['content_digest']

    def patch(self):
        self.content_digest()
        return json.dumps(self.manifest, sort_keys=True)

    def residual_processes(self):
        self.content_digest()
        return ()

    def release(self):
        with self._lock:
            if not self._released:
                self.owner._delete(self.sandbox)
                self._released = True
