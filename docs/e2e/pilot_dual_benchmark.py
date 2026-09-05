"""Create a fresh base-model checkpoint and run a real, parallel 8x pilot."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3

from dual_benchmark_budget import ROOT, paid
from screen_banking77 import _write_json, run_screen
from synth_optimizers.rl.config import load


def main(name, phase='pilot', concurrency=12):
    directory = ROOT / name
    if (directory / phase / 'manifest.json').exists():
        raise RuntimeError('pilot already complete; refusing duplicate')
    os.environ['SYNTH_E2E_PARAMETER_GROUP'] = 'pg-answer' if name == 'healthbench' else 'pg-0'
    config = load(directory / f'{phase}.toml')
    plane = paid(config=config)
    try:
        if (directory / 'baseline.json').exists():
            selector = json.loads((directory / 'baseline.json').read_text())['checkpoint_id']
        else:
            revision = plane.binder.baseline(run_id=config.run_id, parameter_group_id=os.environ['SYNTH_E2E_PARAMETER_GROUP'])
            selector = revision.checkpoint_id
        with sqlite3.connect(directory / 'checkpoints.sqlite3') as db:
            rows = [json.loads(row[0]) for row in db.execute('SELECT payload FROM checkpoints')]
        row = next(row for row in rows if row['checkpoint_id'] == selector)
        _write_json(directory / 'baseline.json', row)
        _write_json(directory / 'artifact_digests.json', {a['ref']:a['digest'] for r in rows for a in r['artifacts'].values()})
        result = run_screen(config, plane, selector=selector, output=directory/phase,samples=8,concurrency=concurrency,poll_limit=3600,selection_mode='reward_variance')
        print(json.dumps(result,indent=2))
    finally:
        plane.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', choices=['healthbench','craftax'])
    parser.add_argument('--phase', default='pilot')
    parser.add_argument('--concurrency', type=int, default=12)
    args = parser.parse_args()
    main(args.benchmark, args.phase, args.concurrency)
