"""Freeze train/validation/final identities before either benchmark is sampled."""
from __future__ import annotations

import hashlib
import json
import urllib.request

from dual_benchmark_budget import ROOT
from screen_banking77 import _write_json

REPO = ROOT.parents[1]


def config(benchmark, phase, train_rows, evaluation_rows, *, resume=None, updates=10, port=None):
    hb = benchmark == 'healthbench'
    run = f'dual_{benchmark}_{phase}_20260904'
    text = f'''schema_version = "cispo.container.v1"
run_id = "{run}"
[container]
url = "http://127.0.0.1:{port or (8260 if hb else 8261)}"
[taskset]
train_split = "{'eval' if hb else 'train'}"
evaluation_split = "{'eval' if hb else 'heldout'}"
train_ids = {json.dumps([r['task_id'] for r in train_rows])}
evaluation_ids = {json.dumps([r['task_id'] for r in evaluation_rows])}
[model]
provider = "tinker"
id = "openai/gpt-oss-20b"
family = "gpt_oss"
policy_kind = "{'healthbench_chat' if hb else 'craftax_react'}"
'''
    if resume:
        text += f'resume_from_checkpoint = "{resume}"\n'
    text += f'''[plan]
preset = "cispo"
group_size = 4
groups_per_step = 3
target_train_updates = {updates}
maximum_sampled_groups = 240
[pipeline]
max_execution_slots = 12
rollout_queue_capacity = 24
score_queue_capacity = 24
scored_result_queue_capacity = 24
train_ready_capacity = 1
maximum_policy_lag = 0
max_open_groups = 3
bounded_on_policy_batch = true
expected_horizon_seconds = {600 if hb else 1440}.0
stale_disposition = "discard"
[topology]
expected_topology_id = "{'healthbench.answer.solo.v1' if hb else 'craftax.react.solo.v1'}"
trainable_teams = ["team-0"]
partial_roster = "refuse"
[opponents]
match_set_revision = "match-set-0001"
[reward]
optimized_channel = "score"
[evaluation]
paired = false
[lifecycle]
resume_requires_rehandshake = true
[offline]
mode = "off"
[artifacts]
catalog = "{ROOT / benchmark / 'checkpoints.sqlite3'}"
directory = "{ROOT / benchmark / 'runs'}"
'''
    path = ROOT / benchmark / f'{phase}.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def prepare():
    if (ROOT / 'panels.json').exists():
        raise RuntimeError('frozen experiment already exists; refusing overwrite')
    ROOT.mkdir(parents=True, exist_ok=True)
    url = 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl'
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    (ROOT / 'healthbench_dataset.jsonl').write_bytes(source)
    corpus = [json.loads(line) for line in source.splitlines() if line]
    assert len(corpus) == 5000
    hb = [{'task_id':r['prompt_id'], 'seed':i} for i,r in enumerate(corpus)]
    hb.sort(key=lambda r:hashlib.sha256(('dual-hb-20260904:'+r['task_id']).encode()).hexdigest())
    def crx(split, start, count):
        return [{'task_id':f'craftax/{split}/{seed}', 'seed':seed} for seed in range(start,start+count)]
    panels = {
        'healthbench': {'train':hb[:32], 'validation':hb[32:64], 'final':hb[64:192], 'source_sha256':hashlib.sha256(source).hexdigest()},
        'craftax': {'train':crx('train',96001,32), 'validation':crx('heldout',97001,16), 'final':crx('heldout',98001,64)},
        'design': {'additional_updates':50, 'group_size':4, 'groups_per_update':3, 'screen_samples':8,
                   'screen_rule':'nonzero within-task reward range; select only using training rows',
                   'validation_checkpoints':[10,25,50], 'selection':'highest validation mean; earliest checkpoint on ties',
                   'final_metric':'paired mean reward difference; bootstrap tasks, not rubric items',
                   'aggregate_max_usd':49, 'expected_usd':[20,40], 'craftax_env_steps':64, 'craftax_policy_calls':8,
                   'healthbench_judge':'gpt-4.1-2025-04-14 via OpenRouter, fixed for all phases',
                   'caveats':['custom research partitions, not official leaderboard scores','Craftax is the local GameBench Rust implementation','HealthBench uplift is not evidence of clinical readiness']}}
    _write_json(ROOT / 'panels.json', panels)
    for name in ('healthbench','craftax'):
        panel = panels[name]
        config(name,'pilot',panel['train'][:4],[],updates=1)
        config(name,'screen_remaining',panel['train'][4:],[])
    print(json.dumps({'root':str(ROOT),'healthbench_rubrics_mean':sum(len(corpus[r['seed']]['rubrics']) for r in hb[:192])/192,'healthbench_source_sha256':panels['healthbench']['source_sha256']}))


if __name__ == '__main__':
    prepare()
