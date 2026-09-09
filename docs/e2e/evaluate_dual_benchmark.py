"""Paired frozen-panel scoring; independent tasks are the uncertainty unit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from dual_benchmark_budget import ROOT, paid
from screen_banking77 import _write_json
from synth_optimizers.rl.config import load
from synth_optimizers.rl.evaluation import EvaluationRequest, HeldOutSeed, PairedEvaluation, PinTemplate, RosterSlot
from synth_optimizers.rl.resolver import EvaluationResolver, MappingArtifactProbe, ResolutionScope
from synth_optimizers.rl.catalog import CheckpointCatalog


class PersistedEvaluation(PairedEvaluation):
    """Persist each observed outcome before another task can fail."""

    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self._output = output

    def _row(self, *args, **kwargs):
        row = super()._row(*args, **kwargs)
        _write_json(self._output/'attempts'/f'{row.arm}_{row.sample_index}.json', row.to_payload())
        _write_json(self._output/'rewards'/f'{row.rollout_id}.json', self._session.reward_payload(row.rollout_id))
        _write_json(self._output/'traces'/f'{row.rollout_id}.json', self._session.trace(row.rollout_id))
        return row


def main(args):
    name = args.benchmark
    directory = ROOT / name
    group = 'pg-answer' if name == 'healthbench' else 'pg-0'
    instance = 'answer-policy' if name == 'healthbench' else 'instance-0'
    os.environ['SYNTH_E2E_PARAMETER_GROUP'] = group
    config = load(directory / f'{args.phase}.toml')
    output = directory / args.phase
    if list(output.glob('*.evaluation.json')) or (output/'evaluation_started.json').exists() or list((output/'attempts').glob('*.json')):
        raise RuntimeError('evaluation evidence already exists; refusing to overwrite an observed panel')
    rows = json.loads((ROOT/'panels.json').read_text())[name][args.panel]
    plane = paid(config=config)
    catalog = CheckpointCatalog(directory/'checkpoints.sqlite3')
    try:
        resolver = EvaluationResolver(catalog,probe=MappingArtifactProbe(json.loads((directory/'artifact_digests.json').read_text())))
        capability = plane.session.capability
        tasks = plane.session.tasks(split=config.taskset.evaluation_split,task_ids=tuple(r['task_id'] for r in rows))
        assert len(tasks) == len(rows)
        request = EvaluationRequest(evaluation_id=config.run_id,baseline_selector=args.baseline,trained_selector=args.selected,
            seeds=tuple(HeldOutSeed(task_id=r['task_id'],seed=r['seed']) for r in rows),roster=(RosterSlot(agent_instance_id=instance,parameter_group_id=group),),
            pin=PinTemplate(run_id=config.run_id,algorithm_plan_hash=config.expanded_plan().plan_hash,wire_api=config.model.wire_api,
                sampling_transport=config.model.sampling_transport,policy_kind=config.model.policy_kind,model_family=config.model.family,
                container_image_digest=capability.container_image_digest,container_contract_hash=plane.session.startup.contract.contract_hash,
                task_family=tasks[0].task_family,topology_id=capability.topology.topology_id),
            split=config.taskset.evaluation_split,scope=ResolutionScope(parameter_group_id=group),poll_limit=3600,concurrency=args.concurrency)
        _write_json(output/'evaluation_started.json', {'evaluation_id':config.run_id, 'baseline':args.baseline, 'selected':args.selected, 'panel':args.panel})
        receipt = PersistedEvaluation(resolver,output=output,session=plane.session,gateway=plane.gateway,binder=plane.binder).run(request)
        receipt_path = receipt.write(output)
        payload = receipt.to_payload()
        details = {}
        for arm in ('baseline','trained'):
            details[arm] = []
            for attempt in payload['arms'][arm]['attempts']:
                rollout = attempt['rollout_id']
                details[arm].append(json.loads((output/'rewards'/f'{rollout}.json').read_text()))
        _write_json(output/'reward_details.json',details)
        paired = payload['paired_summary']
        differences = np.asarray([r['delta'] for r in paired['rows']],dtype=float)
        rng = np.random.default_rng(20260904)
        bootstrap = np.mean(rng.choice(differences,size=(20000,len(differences))),axis=1)
        result = {k:v for k,v in paired.items() if k != 'rows'}
        result.update(benchmark=name, panel=args.panel, baseline_checkpoint=args.baseline, trained_checkpoint=args.selected,
            paired_bootstrap_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),bootstrap_seed=20260904,
            bootstrap_replicates=20000,receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            duration_seconds=payload['duration_seconds'],attempts=payload['attempt_count'],usage=payload['usage_totals'])
        _write_json(output/'result.json',result)
        print(json.dumps(result,indent=2))
    finally:
        plane.close()
        catalog.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark',choices=['healthbench','craftax'])
    parser.add_argument('phase')
    parser.add_argument('--panel',choices=['validation','final'],required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--selected',required=True)
    parser.add_argument('--concurrency',type=int,default=24)
    main(parser.parse_args())
