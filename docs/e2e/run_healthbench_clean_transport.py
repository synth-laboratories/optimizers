"""Fresh HealthBench training; corrected text, unchanged research grader."""
from dual_benchmark_budget import ROOT, LEDGER_ROOT
from run_dual_benchmark import command, heldout, server, train
from screen_banking77 import _write_json


def main():
    assert ROOT != LEDGER_ROOT, 'clean run must not overwrite old artifacts'
    assert not (ROOT/'healthbench/screen_full').exists(), 'explicit recovery required'
    with server('healthbench', 'screen_full', 8260, 1):
        command('healthbench', 'screen_full', ['python', 'docs/e2e/pilot_dual_benchmark.py',
                'healthbench', '--phase', 'screen_full', '--concurrency', '24'])
    train('healthbench')
    heldout('healthbench')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        _write_json(ROOT/'healthbench/status.json', {'state': 'failed', 'error': str(error)})
        raise
