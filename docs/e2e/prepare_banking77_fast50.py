"""Freeze the fast 50-update experiment, then assemble its screened curriculum."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from freeze_banking77_panel import _atomic_json, audit_files, exclusions_from_files, freeze_panel, source_rows
from synth_optimizers.rl.config import load

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_fast50_19')
PREVIOUS = ROOT.parent / 'b77_real_uplift_18'
PARENT = 'ckpt_d02739fcc0546c017c3cfb94'
CONFIGS = REPO / 'docs/e2e/configs'


def ordered(rows, seed):
    return sorted(rows, key=lambda r: hashlib.sha256(f'{seed}\0{r["task_id"]}'.encode()).hexdigest())


def config_text(run_id, ids, evaluation_ids, port, *, train=False):
    text = (CONFIGS / 'run_b77_curriculum_stage2_paid_18.toml').read_text()
    text = re.sub(r'run_id = "[^"]+"', f'run_id = "{run_id}"', text, count=1)
    text = text.replace('http://127.0.0.1:8241', f'http://127.0.0.1:{port}')
    for key, values in [('train_ids', ids), ('evaluation_ids', evaluation_ids)]:
        text = re.sub(rf'{key} = \[.*?\]', key + ' = ' + json.dumps(values), text, flags=re.S)
    text = re.sub(r'^#.*\n', '', text, flags=re.M)
    text = re.sub(r'resume_from_checkpoint = .*\n', f'resume_from_checkpoint = "{PARENT}"\n' if train else '', text)
    for old, new in [('group_size = 16', 'group_size = 8'), ('groups_per_step = 1', 'groups_per_step = 4'), ('target_train_updates = 8', 'target_train_updates = 50'), ('steps_per_round = 8', 'steps_per_round = 50'), ('maximum_sampled_groups = 40', 'maximum_sampled_groups = 800'), ('train_ready_capacity = 1', 'train_ready_capacity = 4'), ('max_open_groups = 1', 'max_open_groups = 4')]:
        text = text.replace(old, new)
    text = text.replace('train_ready_capacity = 4', 'train_ready_capacity = 1')
    text = re.sub(r'directory = .*', f'directory = "{ROOT / run_id / "runs"}"', text)
    return text


def prepare():
    if (ROOT / 'experiment.json').exists():
        raise SystemExit('Experiment already frozen; refusing to overwrite it.')
    rows, source = source_rows(Path('/tmp/banking77-cache/banking77-heldout.csv'))
    excluded, inventory = exclusions_from_files(audit_files([REPO / 'docs', PREVIOUS, ROOT.parent / 'b77_variance8_gate_15']))
    validation = freeze_panel(rows, excluded=excluded, panel_seed='fast50-19-validation', source=source, exclusion_inventory=inventory, examples_per_intent=2)
    final = freeze_panel(rows, excluded=excluded | {r['task_id'] for r in validation['rows']}, panel_seed='fast50-19-final', source=source, exclusion_inventory=inventory, examples_per_intent=10)
    _atomic_json(ROOT / 'validation_panel.json', validation)
    _atomic_json(ROOT / 'final_panel.json', final)
    train_file = Path('/tmp/banking77-cache/banking77-train.csv')
    labels = defaultdict(list)
    for i, r in enumerate(csv.DictReader(train_file.read_text().splitlines())):
        labels[r['category']].append({'task_id': f'banking77/train/{i}', 'seed': i, 'label': r['category']})
    assert len(labels) == 77 and all(len(v) >= 20 for v in labels.values())
    selected = {label: ordered(group, 'fast50-19-candidates')[:20] for label, group in labels.items()}
    candidates = [selected[label][i] for i in range(20) for label in sorted(labels)]
    _atomic_json(ROOT / 'candidates.json', {'rows': candidates, 'source_sha256': hashlib.sha256(train_file.read_bytes()).hexdigest()})
    digests = json.loads((PREVIOUS / 'confirmatory_5x/artifact_digests.json').read_text())
    digests['tinker://b577d3c6-caad-5f81-b05e-d26de32c9f43:train:0/weights/optimizers-training_state-save-ca5b3e3089c456204b25946319ab29cf'] = 'sha256:efdf3c49b10fa464952191393ad0a8b3653dc26ddd5abe3fbf1ea4dcc5aa68f3'
    _atomic_json(ROOT / 'artifact_digests.json', digests)
    eval_ids = [r['task_id'] for r in validation['rows']]
    for shard in range(4):
        name = f'b77_fast50_19_screen_{shard}'
        path = CONFIGS / f'{name}.toml'
        path.write_text(config_text(name, [r['task_id'] for r in candidates[shard::4]], eval_ids, 8250 + shard))
        load(path).expanded_plan()
    _atomic_json(ROOT / 'experiment.json', {
        'parent_checkpoint': PARENT, 'additional_effective_updates': 50,
        'group_size': 8, 'groups_per_update': 4, 'maximum_sampled_groups': 800,
        'candidate_count': 1540, 'samples_per_candidate': 8,
        'admission': '1 <= successes <= 7; interleave eligible tasks by intent',
        'screen_shards': 4, 'concurrency_per_shard': 8,
        'validation_updates': [34, 44, 54, 64, 74],
        'selection': 'highest validation accuracy; ties choose earliest update',
        'final_comparisons': ['original baseline', 'update 24'],
        'primary_final_comparison': 'update 24; original baseline is secondary context',
        'validation_panel_digest': validation['panel_digest'], 'final_panel_digest': final['panel_digest'],
        'aggregate_cost_cap_usd': 15,
    })
    print('Frozen 154 validation, 770 final, and 1540 training candidates across four shards.')


def curriculum():
    candidates = json.loads((ROOT / 'candidates.json').read_text())['rows']
    labels = {r['task_id']: r['label'] for r in candidates}
    admitted = defaultdict(list)
    seen = set()
    for shard in range(4):
        directory = ROOT / f'screen_{shard}'
        manifest = json.loads((directory / 'manifest.json').read_text())
        attempts = json.loads((directory / 'attempts.json').read_text())
        assert manifest['checkpoint_id'] == PARENT
        expected = {r['task_id'] for r in candidates[shard::4]}
        assert {r['task_id'] for r in attempts} == expected
        for tid in expected:
            values = [r for r in attempts if r['task_id'] == tid]
            assert len(values) == 8 and {r['sample_index'] for r in values} == set(range(8))
            assert all(r['checkpoint_id'] == PARENT and r['terminal_status'] in {'completed', 'scored'} for r in values)
            assert all(r['reward'] in (0, 1) for r in values)
            if 1 <= sum(r['reward'] for r in values) <= 7:
                admitted[labels[tid]].append({'task_id': tid})
        assert not seen & expected
        seen |= expected
    groups = {label: ordered(group, 'fast50-19-curriculum') for label, group in admitted.items()}
    ids = [groups[label][i]['task_id'] for i in range(max(map(len, groups.values()))) for label in sorted(groups) if i < len(groups[label])]
    assert ids
    validation = json.loads((ROOT / 'validation_panel.json').read_text())
    path = CONFIGS / 'run_b77_fast50_19.toml'
    path.write_text(config_text('b77_fast50_19', ids, [r['task_id'] for r in validation['rows']], 8250, train=True))
    config = load(path)
    _atomic_json(ROOT / 'curriculum.json', {'selected_train_ids': ids, 'selected_count': len(ids), 'intent_counts': {label: len(v) for label, v in groups.items()}, 'plan_hash': config.expanded_plan().plan_hash})
    print(f'Admitted {len(ids)} tasks across {len(groups)} intents; 50-update configuration validated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'curriculum'])
    args = parser.parse_args()
    prepare() if args.phase == 'prepare' else curriculum()
