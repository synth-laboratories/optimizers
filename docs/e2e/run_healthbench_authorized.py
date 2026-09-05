"""Continue the successful pilot under the explicitly approved $100 total cap."""
import json
import shutil

from dual_benchmark_budget import ROOT
from run_dual_benchmark import command, heldout, server, train
from screen_banking77 import _write_json


def main():
    directory = ROOT / 'healthbench'
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3
    assert (directory / 'pilot/manifest.json').exists()
    assert not (directory / 'screen_remaining').exists(), 'preserve prior screening evidence'
    baseline = json.loads((directory / 'baseline.json').read_text())['checkpoint_id']
    with server('healthbench', 'screen_remaining', 8260, 1):
        command('healthbench', 'screen_remaining', [
            'python', 'docs/e2e/screen_banking77.py',
            '--config', str(directory / 'screen_remaining.toml'),
            '--selector', baseline, '--output', str(directory / 'screen_remaining'),
            '--plane', 'dual_benchmark_budget:paid', '--samples', '8',
            '--concurrency', '24', '--selection-mode', 'reward_variance', '--poll-limit', '3600',
        ])
    train('healthbench')
    heldout('healthbench')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        _write_json(ROOT / 'healthbench/status.json', {'state': 'failed', 'error': str(error)})
        raise
