"""Explicit disk-full recovery; preserve the interrupted segment's evidence."""
import json
import shutil

from dual_benchmark_budget import ROOT
from prepare_dual_benchmark import config
from run_dual_benchmark import checkpoints, command, heldout, server
from screen_banking77 import _write_json
from synth_optimizers.rl.catalog import CheckpointCatalog
from synth_optimizers.rl.config import load


def main():
    directory = ROOT / 'craftax'
    assert shutil.disk_usage(directory).free > 10 * 1024**3
    rows = checkpoints('craftax')
    parent = 'ckpt_965fb2c7899ae2148734b8a9'
    assert rows[-1]['checkpoint_id'] == parent
    catalog = CheckpointCatalog(directory / 'checkpoints.sqlite3')
    try:
        assert catalog.publication_status(parent) == 'published'
    finally:
        catalog.close()
    tasks = json.loads((directory / 'curriculum.json').read_text())['rows']
    panel = json.loads((ROOT / 'panels.json').read_text())['craftax']
    for target, updates in ((25, 3), (40, 15), (50, 10)):
        phase = f'recovery_train_{target}'
        assert not (directory / phase).exists(), 'do not overwrite recovery evidence'
        assert shutil.disk_usage(directory).free > 10 * 1024**3
        path = config('craftax', phase, tasks, panel['validation'], resume=parent, updates=updates)
        with server('craftax', phase, 8261, 1):
            command('craftax', phase, ['synth-optimizers', 'rl', 'run', '--config', str(path), '--plane', 'dual_benchmark_budget:paid', '--receipts', str(directory / phase), '--max-ticks', '200000', '--json'])
        assert json.loads((directory / phase / 'manifest.json').read_text())['stop_reason'] == 'target_train_updates_reached'
        trained = [r for r in checkpoints('craftax') if r['run_id'] == load(path).run_id and r['train_call_ids']]
        assert len(trained) == updates
        final = max(trained, key=lambda r: int(r['policy_revision_id'].split('@')[-1]))
        assert int(final['policy_revision_id'].split('@')[-1]) == target
        parent = final['checkpoint_id']
        _write_json(directory / 'training_progress.json', {'completed_updates': target, 'checkpoint': parent})
    heldout('craftax')


if __name__ == '__main__':
    main()
