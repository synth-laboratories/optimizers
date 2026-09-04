"""Run the frozen fast50 training and validation/final evaluation phases."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import time
import urllib.request
from contextlib import contextmanager

from freeze_banking77_panel import _atomic_json
from prepare_banking77_fast50 import CONFIGS, PARENT, PREVIOUS, REPO, ROOT, config_text, curriculum
from synth_optimizers.rl.config import load

CATALOG = ROOT.parent / 'b77_variance8_gate_15/checkpoints.sqlite3'
BASELINE = 'ckpt_b229ee0836324a7d96b0d4b0'
PORT = 8254


def environment():
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO / 'docs/e2e'), SYNTH_TINKER_ENV_FILE='/Users/joshuapurtell/GitHub/frontend/.env.local', SYNTH_E2E_ARTIFACT_DIGESTS=str(ROOT / 'artifact_digests.json'))
    return env


@contextmanager
def server(name, temperature):
    env = environment()
    env.update(SYNTH_BANKING77_SOURCE='hf', SYNTH_BANKING77_DECLARED_ROWS_PER_SPLIT='10003', SYNTH_CISPO_RENDERER_CANARY_DIGEST='43e18d1c29ee9cc6a849f8fc77c9efee', SYNTH_BANKING77_TEMPERATURE=str(temperature), SYNTH_BANKING77_HANDSHAKE_TTL_SECONDS='14400')
    with (ROOT / f'{name}.server.log').open('w') as log:
        proc = subprocess.Popen(['uv', 'run', '--with', 'pytest', '--with', 'uvicorn', 'python', 'docs/e2e/serve_banking77.py', str(PORT)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f'{name}: server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/cispo/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise RuntimeError('server did not become ready')
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)


def command(name, args):
    _atomic_json(ROOT / 'status.json', {'phase': name, 'state': 'running', 'started_unix': time.time()})
    with (ROOT / f'{name}.log').open('w') as log:
        subprocess.run(['uv', 'run', 'synth-optimizers', 'rl', *args], cwd=REPO, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)


def checkpoints():
    with sqlite3.connect(f'file:{CATALOG}?mode=ro', uri=True) as db:
        return [json.loads(row[0]) for row in db.execute('SELECT payload FROM checkpoints WHERE run_id IN (?, ?) ORDER BY seq', ('b77_fast50_19', 'b77_fast50_19_resume25'))]


def train():
    while not all((ROOT / f'screen_{i}/manifest.json').exists() for i in range(4)):
        time.sleep(5)
    curriculum()
    path = CONFIGS / 'run_b77_fast50_19.toml'
    path.write_text(path.read_text().replace('127.0.0.1:8250', f'127.0.0.1:{PORT}'))
    if checkpoints():
        raise RuntimeError('Training has catalogued checkpoints already; inspect before retrying.')
    with server('training', 1):
        command('training', ['run', '--config', str(path), '--plane', 'paid_plane:paid', '--receipts', str(ROOT / 'training'), '--max-ticks', '200000', '--json'])
    rows = checkpoints()
    revisions = {int(r['policy_revision_id'].split('@')[-1]) for r in rows}
    assert set(range(25, 75)).issubset(revisions) and max(revisions) == 74
    manifest = json.loads((ROOT / 'training/manifest.json').read_text())
    assert manifest['stop_reason'] == 'target_train_updates_reached'
    digests = json.loads((ROOT / 'artifact_digests.json').read_text())
    for row in rows:
        for artifact in row['artifacts'].values():
            digests[artifact['ref']] = artifact['digest']
    _atomic_json(ROOT / 'artifact_digests.json', digests)
    _atomic_json(ROOT / 'status.json', {'phase': 'training', 'state': 'completed', 'additional_updates': 50, 'final_revision': 74})


def evaluate(name, selected, baseline, panel_name):
    panel_path = ROOT / f'{panel_name}_panel.json'
    panel = json.loads(panel_path.read_text())
    ids = json.loads((ROOT / 'curriculum.json').read_text())['selected_train_ids']
    config_path = CONFIGS / f'{name}.toml'
    config_path.write_text(config_text(name, ids, [r['task_id'] for r in panel['rows']], PORT))
    config = load(config_path)
    pin = json.loads((PREVIOUS / 'confirmatory_5x/evaluation_pin.json').read_text())
    pin.update(run_id=name, algorithm_plan_hash=config.expanded_plan().plan_hash)
    directory = ROOT / name
    _atomic_json(directory / 'pin.json', pin)
    args = ['evaluate', '--config', str(config_path), '--plane', 'paid_plane:paid', '--catalog', str(CATALOG), '--selector', selected, '--baseline', baseline, '--evaluation-id', name, '--roster', 'instance-0=pg-0:policy-0', '--split', 'heldout', '--scope-parameter-group', 'pg-0', '--scope-policy-type', 'policy-0', '--metric', 'mean_reward', '--artifact-digests', str(ROOT / 'artifact_digests.json'), '--pin', str(directory / 'pin.json'), '--receipts-dir', str(directory), '--concurrency', '8']
    for row in panel['rows']:
        args.extend(['--seed', f'{row["task_id"]}={row["seed"]}'])
    with server(name, 0):
        command(name, args)
    receipt = directory / f'{name}.evaluation.json'
    validation_args = ['uv', 'run', 'python', 'docs/e2e/validate_banking77_eval.py', '--panel', str(panel_path), '--receipt', str(receipt), '--baseline', baseline, '--trained', selected, '--train-config', str(CONFIGS / 'run_b77_fast50_19.toml'), '--bootstrap-seed', '20260904', '--output', str(directory / 'result.json')]
    if panel_name == 'final':
        validation_args.extend(['--prior-panel', str(ROOT / 'validation_panel.json')])
    subprocess.run(validation_args, cwd=REPO, env=environment(), check=True)
    return json.loads((directory / 'result.json').read_text())


def resume25():
    """Explicit recovery of the observed one-update dispatch failure."""
    rows = checkpoints()
    parent = next(r for r in rows if r['checkpoint_id'] == 'ckpt_0278ebdd569252e2f583b9a0')
    assert max(int(r['policy_revision_id'].split('@')[-1]) for r in rows) == 25
    assert not any(r['run_id'] == 'b77_fast50_19_resume25' for r in rows)
    digests = json.loads((ROOT / 'artifact_digests.json').read_text())
    for row in rows:
        for artifact in row['artifacts'].values():
            digests[artifact['ref']] = artifact['digest']
    _atomic_json(ROOT / 'artifact_digests.json', digests)
    original = load(CONFIGS / 'run_b77_fast50_19.toml')
    ids = list(original.taskset.train_ids)
    # The failed run admitted ten groups. Continue the frozen task order.
    ids = ids[10:] + ids[:10]
    text = config_text('b77_fast50_19_resume25', ids, list(original.taskset.evaluation_ids), PORT, train=True)
    text = text.replace(PARENT, parent['checkpoint_id']).replace('target_train_updates = 50', 'target_train_updates = 49').replace('maximum_sampled_groups = 800', 'maximum_sampled_groups = 790').replace('max_open_groups = 4', 'max_open_groups = 1')
    path = CONFIGS / 'run_b77_fast50_19_resume25.toml'
    path.write_text(text)
    load(path).expanded_plan()
    _atomic_json(ROOT / 'recovery.json', {'parent_checkpoint': parent['checkpoint_id'], 'completed_updates': 1, 'remaining_updates': 49, 'already_admitted_groups': 10, 'remaining_group_cap': 790, 'reason': 'fix admitted-revision binding; prevent stale prefetch'})
    with server('training_resume25', 1):
        command('training_resume25', ['run', '--config', str(path), '--plane', 'paid_plane:paid', '--receipts', str(ROOT / 'training_resume25'), '--max-ticks', '200000', '--json'])
    rows = checkpoints()
    revisions = {int(r['policy_revision_id'].split('@')[-1]) for r in rows}
    assert set(range(25, 75)).issubset(revisions) and max(revisions) == 74
    assert json.loads((ROOT / 'training_resume25/manifest.json').read_text())['stop_reason'] == 'target_train_updates_reached'
    for row in rows:
        for artifact in row['artifacts'].values():
            digests[artifact['ref']] = artifact['digest']
    _atomic_json(ROOT / 'artifact_digests.json', digests)
    _atomic_json(ROOT / 'status.json', {'phase': 'training', 'state': 'completed', 'additional_updates': 50, 'final_revision': 74})


def heldout():
    by_revision = {int(r['policy_revision_id'].split('@')[-1]): r for r in checkpoints()}
    validation = []
    for revision in [34, 44, 54, 64, 74]:
        result = evaluate(f'b77_fast50_19_val_{revision}', by_revision[revision]['checkpoint_id'], PARENT, 'validation')
        validation.append({'revision': revision, **result})
        _atomic_json(ROOT / 'validation_results.json', {'results': validation})
    best = max(validation, key=lambda r: (r['trained_mean'], -r['revision']))
    _atomic_json(ROOT / 'selection.json', best)
    selected = best['trained_checkpoint_id']
    original = evaluate('b77_fast50_19_final_original', selected, BASELINE, 'final')
    incremental = evaluate('b77_fast50_19_final_incremental', selected, PARENT, 'final')
    _atomic_json(ROOT / 'final_results.json', {'selected_revision': best['revision'], 'original_baseline': original, 'update24_baseline': incremental})
    _atomic_json(ROOT / 'status.json', {'phase': 'complete', 'state': 'completed', 'selected_revision': best['revision']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'heldout', 'all', 'resume25'])
    args = parser.parse_args()
    try:
        if args.phase in {'train', 'all'}:
            train()
        if args.phase == 'resume25':
            resume25()
        if args.phase in {'heldout', 'all', 'resume25'}:
            heldout()
    except BaseException as error:
        _atomic_json(ROOT / 'status.json', {'state': 'failed', 'error': str(error), 'phase': args.phase})
        raise
