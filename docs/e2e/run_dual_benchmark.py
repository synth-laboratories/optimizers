"""Bounded real training and untouched paired evaluations for either benchmark."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from dual_benchmark_budget import ROOT
from prepare_dual_benchmark import REPO, config
from screen_banking77 import _write_json
from synth_optimizers.rl.config import load


def environment(name):
    return {**os.environ,'PYTHONPATH':str(REPO/'docs/e2e'),
            'SYNTH_E2E_PARAMETER_GROUP':'pg-answer' if name == 'healthbench' else 'pg-0'}


@contextmanager
def server(name, phase, port, temperature):
    directory = ROOT/name
    with (directory/f'{phase}.server.log').open('w') as log:
        proc = subprocess.Popen(['uv','run','--with','uvicorn','python','docs/e2e/serve_dual_benchmark.py',name,'--port',str(port),'--temperature',str(temperature)],cwd=REPO,env=environment(name),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f'{phase} server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/cispo/health',timeout=1) as reply:
                        assert json.load(reply)['status']=='ok'
                    break
                except OSError:
                    time.sleep(.5)
            else:
                raise RuntimeError('server readiness timeout')
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    proc.wait(timeout=30)


def command(name, phase, arguments):
    directory = ROOT/name
    _write_json(directory/'status.json',{'phase':phase,'state':'running','started':time.time()})
    with (directory/f'{phase}.log').open('w') as log:
        subprocess.run(['uv','run',*arguments],cwd=REPO,env=environment(name),stdout=log,stderr=subprocess.STDOUT,check=True)


def checkpoints(name):
    with sqlite3.connect(f'file:{ROOT/name/"checkpoints.sqlite3"}?mode=ro',uri=True) as db:
        return [json.loads(row[0]) for row in db.execute('SELECT payload FROM checkpoints ORDER BY seq')]


def train(name):
    directory = ROOT/name
    panel = json.loads((ROOT/'panels.json').read_text())[name]
    screen_phases = ('screen_full',) if (directory/'screen_full/manifest.json').exists() else ('pilot','screen_remaining')
    manifests = [json.loads((directory/phase/'manifest.json').read_text()) for phase in screen_phases]
    assert sum(m['attempt_count'] for m in manifests)==256
    selected = {task for m in manifests for task in m['selected_train_ids']}
    tasks = [r for r in panel['train'] if r['task_id'] in selected]
    assert tasks, 'no training reward variation; cannot train honestly'
    _write_json(directory/'curriculum.json',{'rows':tasks,'admission':'nonzero reward range across exactly eight completions'})
    if any(r['train_call_ids'] for r in checkpoints(name)):
        raise RuntimeError('training checkpoints exist; explicit recovery required, not blind restart')
    parent = None
    baseline = None
    # Respect the workspace's 15-update segment ceiling. Resume exact training
    # state, never sampler weights, between segments. Each update packs 3 groups.
    for target, updates in ((10,10),(25,15),(40,15),(50,10)):
        phase = f'train_{target}'
        path = config(name,phase,tasks,panel['validation'],resume=parent,updates=updates)
        load(path)
        with server(name,phase,8260 if name=='healthbench' else 8261,1):
            command(name,phase,['synth-optimizers','rl','run','--config',str(path),'--plane','dual_benchmark_budget:paid','--receipts',str(directory/phase),'--max-ticks','200000','--json'])
        assert json.loads((directory/phase/'manifest.json').read_text())['stop_reason']=='target_train_updates_reached'
        rows = checkpoints(name)
        trained = [r for r in rows if r['train_call_ids'] and r['run_id']==load(path).run_id]
        assert len(trained)==updates
        final = max(trained,key=lambda r:int(r['policy_revision_id'].split('@')[-1]))
        assert int(final['policy_revision_id'].split('@')[-1])==target
        parent = final['checkpoint_id']
        if baseline is None:
            baseline = next(r['checkpoint_id'] for r in rows if r['run_id']==load(path).run_id and not r['train_call_ids'])
            _write_json(directory/'training_baseline.json',{'checkpoint_id':baseline})
        _write_json(directory/'training_progress.json',{'completed_updates':target,'checkpoint':parent})
    return baseline


def evaluate(name, phase, panel_name, selected, baseline, port, concurrency):
    panel = json.loads((ROOT/'panels.json').read_text())[name]
    tasks = json.loads((ROOT/name/'curriculum.json').read_text())['rows']
    config(name,phase,tasks,panel[panel_name],updates=1,port=port)
    with server(name,phase,port,0):
        command(name,phase,['python','docs/e2e/evaluate_dual_benchmark.py',name,phase,'--panel',panel_name,'--baseline',baseline,'--selected',selected,'--concurrency',str(concurrency)])
    return json.loads((ROOT/name/phase/'result.json').read_text())


def heldout(name):
    directory = ROOT/name
    rows = checkpoints(name)
    by_revision = {int(r['policy_revision_id'].split('@')[-1]):r for r in rows if r['train_call_ids']}
    assert set(range(1,51)).issubset(by_revision)
    baseline = json.loads((directory/'training_baseline.json').read_text())['checkpoint_id']
    port = 8270 if name=='craftax' else 8280
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(revision,pool.submit(evaluate,name,f'validation_{revision}','validation',by_revision[revision]['checkpoint_id'],baseline,port+i,8)) for i,revision in enumerate((10,25,50))]
        results = [{'revision':revision,**future.result()} for revision,future in futures]
    _write_json(directory/'validation_results.json',results)
    best = max(results,key=lambda r:(r['trained_mean'],-r['revision']))
    _write_json(directory/'selection.json',best)
    result = evaluate(name,'final','final',best['trained_checkpoint'],baseline,port,24)
    _write_json(directory/'final_results.json',{'selected_revision':best['revision'],**result})
    _write_json(directory/'status.json',{'state':'completed','phase':'final','additional_updates':50})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('benchmark',choices=['healthbench','craftax'])
    parser.add_argument('phase',choices=['train','heldout','all'])
    args=parser.parse_args()
    try:
        if args.phase in ('train','all'):
            train(args.benchmark)
        if args.phase in ('heldout','all'):
            heldout(args.benchmark)
    except BaseException as error:
        _write_json(ROOT/args.benchmark/'status.json',{'state':'failed','phase':args.phase,'error':str(error)})
        raise
