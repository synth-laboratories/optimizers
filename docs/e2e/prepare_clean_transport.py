"""Freeze clean-transport artifacts while retaining the original spend ledger."""
import json
import shutil
import sqlite3

from dual_benchmark_budget import ROOT, LEDGER_ROOT
from prepare_dual_benchmark import config
from screen_banking77 import _write_json


def main():
    assert ROOT != LEDGER_ROOT
    assert not ROOT.exists(), 'do not overwrite clean experiment'
    ROOT.mkdir(parents=True)
    panel = json.loads((LEDGER_ROOT/'panels.json').read_text())
    panel['craftax']['final'] = [{'task_id': f'craftax/heldout/{seed}', 'seed': seed} for seed in range(99001,99065)]
    panel['clean_transport'] = {
        'reason': 'Generic sampler must preserve prose and JSON; old label normalization is removed.',
        'craftax_fixed_checkpoint': 'ckpt_4bfc308872dc44019328ac96',
        'craftax_final_seeds': [99001,99064],
        'healthbench_training': 'fresh base model and fresh screening; old 17-update run is diagnostic only',
        'ledger_root': str(LEDGER_ROOT),
    }
    _write_json(ROOT/'panels.json', panel)
    shutil.copy2(LEDGER_ROOT/'healthbench_dataset.jsonl',ROOT/'healthbench_dataset.jsonl')
    (ROOT/'craftax').mkdir()
    for name in ('artifact_digests.json','curriculum.json','training_baseline.json'):
        shutil.copy2(LEDGER_ROOT/'craftax'/name,ROOT/'craftax'/name)
    with sqlite3.connect(f'file:{LEDGER_ROOT/"craftax/checkpoints.sqlite3"}?mode=ro',uri=True) as source:
        with sqlite3.connect(ROOT/'craftax/checkpoints.sqlite3') as target:
            source.backup(target)
    config('healthbench','screen_full',panel['healthbench']['train'],[],updates=1)
    shutil.copy2(LEDGER_ROOT/'healthbench/judge_protocol.json',ROOT/'healthbench/judge_protocol.json')


if __name__ == '__main__':
    main()
