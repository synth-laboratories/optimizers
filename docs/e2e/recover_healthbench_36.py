"""Explicit recovery after OpenRouter credit exhaustion; never restart screening."""
import json
import os

import httpx

from dual_benchmark_budget import ROOT, LEDGER_ROOT, load_credentials
from prepare_dual_benchmark import config
from run_dual_benchmark import checkpoints, command, heldout, server
from screen_banking77 import _write_json
from synth_optimizers.rl.catalog import CheckpointCatalog
from synth_optimizers.rl.config import load


def main():
    assert ROOT != LEDGER_ROOT, 'set DUAL_BENCHMARK_ROOT to the clean artifact root'
    assert os.environ.get('DUAL_PERSIST_EVIDENCE') == '1'
    directory = ROOT / 'healthbench'
    parent = 'ckpt_3b2b2e1626446de48f68587b'
    assert checkpoints('healthbench')[-1]['checkpoint_id'] == parent
    catalog = CheckpointCatalog(directory / 'checkpoints.sqlite3')
    try:
        assert catalog.publication_status(parent) == 'published'
    finally:
        catalog.close()
    load_credentials('OPENROUTER_API_KEY')
    response = httpx.get('https://openrouter.ai/api/v1/credits',
                        headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']}, timeout=30)
    response.raise_for_status()
    credits = response.json()['data']
    assert credits['total_credits'] > credits['total_usage'], 'OpenRouter account remains exhausted'
    tasks = json.loads((directory / 'curriculum.json').read_text())['rows']
    panel = json.loads((ROOT / 'panels.json').read_text())['healthbench']
    for target, updates in ((40, 4), (50, 10)):
        phase = f'recovery_train_{target}'
        assert not (directory / phase).exists(), 'explicit recovery required; preserve partial evidence'
        path = config('healthbench', phase, tasks, panel['validation'], resume=parent, updates=updates)
        with server('healthbench', phase, 8260, 1):
            command('healthbench', phase, ['synth-optimizers', 'rl', 'run', '--config', str(path),
                    '--plane', 'dual_benchmark_budget:paid', '--receipts', str(directory / phase),
                    '--max-ticks', '200000', '--json'])
        assert json.loads((directory / phase / 'manifest.json').read_text())['stop_reason'] == 'target_train_updates_reached'
        trained = [r for r in checkpoints('healthbench') if r['run_id'] == load(path).run_id and r['train_call_ids']]
        assert len(trained) == updates
        final = max(trained, key=lambda r: int(r['policy_revision_id'].split('@')[-1]))
        assert int(final['policy_revision_id'].split('@')[-1]) == target
        parent = final['checkpoint_id']
        _write_json(directory / 'training_progress.json', {'completed_updates': target, 'checkpoint': parent})
    heldout('healthbench')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        _write_json(ROOT / 'healthbench/status.json', {'state': 'failed', 'error': str(error)})
        raise
