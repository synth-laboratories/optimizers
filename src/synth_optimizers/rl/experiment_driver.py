"""Supported container-first screen/train/evaluate phase driver.

Containers are independently managed declared endpoints. This module imports no
benchmark checkout, credentials file, recovery script, or private database table.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
import time
import urllib.request

from ..contracts.rl_records import SamplingProfile
from .config import from_mapping
from .evaluation import EvaluationRequest, HeldOutSeed, PairedEvaluation, PinTemplate, RosterSlot
from .evaluation_store import EvaluationStore
from .executor import ExecutionPlan, execute
from .plane import build_plane, ProviderArtifactProbe
from .resolver import EvaluationResolver, ResolutionScope
from .screening import run_screen


class ContainerExperimentDriver:
    def __init__(self, spec, *, plane_factory=build_plane):
        self.spec = type(spec).model_validate(spec.model_dump())
        self.plane_factory = plane_factory
        self.root = Path(from_mapping(self.spec.run).artifacts.directory) / self.spec.experiment_id
        self.admission_check = None

    def set_admission_check(self, check):
        self.admission_check = check

    def configuration(self, phase, prior):
        payload = deepcopy(self.spec.run)
        payload['run_id'] = f'{self.spec.experiment_id}.{phase["id"]}'
        train_ids = [t.task_id for t in self.spec.train]
        if phase['kind'] != 'screen':
            train_ids = prior['screen']['selected_train_ids']
        panel = self.spec.final if phase['kind'] == 'final' else self.spec.validation
        payload.setdefault('taskset', {}).update(train_ids=train_ids, evaluation_ids=[t.task_id for t in panel])
        payload.setdefault('artifacts', {})['directory'] = str(self.root / phase['id'] / 'artifacts')
        payload.setdefault('plan', {}).update(target_train_updates=phase.get('updates', 1))
        payload.setdefault('model', {}).pop('resume_from_checkpoint', None)
        if phase['kind'] == 'train':
            previous = [v for k, v in prior.items() if k.startswith('train_')]
            if previous:
                payload['model']['resume_from_checkpoint'] = max(previous, key=lambda v: v['target_update'])['checkpoint_id']
            else:
                payload['model']['resume_from_checkpoint'] = prior['screen']['baseline_checkpoint_id']
        if phase['kind'] in {'validation', 'final'}:
            payload['container']['url'] = self.spec.evaluation_url
        return from_mapping(payload)

    @staticmethod
    def verify_panel(session, split, expected):
        tasks = session.tasks(split=split, task_ids=tuple(t.task_id for t in expected))
        actual = {t.task_id: t for t in tasks}
        if len(actual) != len(expected):
            raise ValueError('container task membership differs from frozen panel')
        for task in expected:
            observed = actual[task.task_id]
            if observed.content_digest != task.content_digest or observed.seed != task.seed:
                raise ValueError('container task content/seed differs from frozen panel')
        return tasks

    def perform(self, phase, snapshot):
        from .screening import _write_json
        started = time.monotonic()
        result = self._perform(phase, snapshot)
        result['phase_seconds'] = time.monotonic() - started
        if 'admitted_examples' in result:
            result['admitted_examples_per_second'] = result['admitted_examples'] / max(result['phase_seconds'], 1e-9)
        _write_json(self.root / phase['id'] / 'phase-result.json', {
            'spec_digest': hashlib.sha256(self.spec.model_dump_json().encode()).hexdigest(),
            'phase': phase, 'result': result})
        return result

    def recovered_result(self, phase):
        body = (self.root / phase['id'] / 'phase-result.json').read_bytes()
        record = json.loads(body)
        if record['phase'] != phase or record['spec_digest'] != hashlib.sha256(self.spec.model_dump_json().encode()).hexdigest():
            raise ValueError('phase evidence does not match the frozen experiment')
        return record['result'], 'sha256:' + hashlib.sha256(body).hexdigest()

    def _perform(self, phase, snapshot):
        prior = {p['phase']['id']: p['result'] for p in snapshot['phases'] if p['state'] == 'completed'}
        if phase['kind'] == 'select':
            candidates = [v for k, v in prior.items() if k.startswith('validation_')]
            selected = max(candidates, key=lambda v: (v['trained_mean'], -v['target_update']))
            from .catalog import CheckpointCatalog
            with CheckpointCatalog(from_mapping(self.spec.run).artifacts.catalog) as catalog:
                catalog.put_alias('selected:' + self.spec.experiment_id, 'checkpoint', selected['checkpoint_id'])
            return {'checkpoint_id': selected['checkpoint_id'], 'target_update': selected['target_update'],
                    'rule': 'highest_validation_mean_then_earliest_update',
                    'evaluation_id': selected['evaluation_id'], 'panel': 'validation'}
        config = self.configuration(phase, prior)
        output = self.root / phase['id']
        evaluation = phase['kind'] in {'validation', 'final'}
        if self.spec.benchmark != 'container':
            with urllib.request.urlopen(config.container.url.rstrip('/') + '/rl/experiment', timeout=10) as response:
                manifest = json.load(response)
            expected_digest = hashlib.sha256(self.spec.model_dump_json().encode()).hexdigest()
            if (manifest.get('spec_digest') != expected_digest or manifest.get('temperature') != (0 if evaluation else 1)
                    or manifest.get('max_tokens') != self.spec.max_tokens or manifest.get('normalization') != 'none'
                    or (self.spec.benchmark == 'healthbench' and manifest.get('budgeted_grading') is not True)):
                raise ValueError('benchmark runtime does not attest the frozen sampling/budget protocol')
        with self.plane_factory(config, admission_check=self.admission_check, sampling=SamplingProfile(
                temperature=0.0 if evaluation else 1.0, max_tokens=self.spec.max_tokens)) as plane:
            instances = plane.session.capability.topology.trainable_instances
            if len(instances) != 1:
                raise ValueError('experiment driver currently requires one trainable instance')
            instance = instances[0]
            group = plane.session.capability.topology.parameter_groups[instance.policy_type_id]
            if phase['kind'] == 'screen':
                self.verify_panel(plane.session, config.taskset.train_split, self.spec.train)
                baseline = plane.binder.baseline(run_id=config.run_id, parameter_group_id=group, save_training_state=True)
                manifest = run_screen(config, plane, selector=baseline.checkpoint_id,
                    output=output / 'screening', samples=self.spec.screening.samples,
                    concurrency=self.spec.screening.concurrency,
                    selection_mode='binary' if self.spec.screening.rule == 'mixed_binary_success' else 'reward_variance')
                if not manifest['selected_train_ids']:
                    raise ValueError('screening found no trainable reward variation')
                return {**manifest, 'baseline_checkpoint_id': baseline.checkpoint_id}
            if phase['kind'] == 'train':
                expected = [t for t in self.spec.train if t.task_id in config.taskset.train_ids]
                self.verify_panel(plane.session, config.taskset.train_split, expected)
                report = execute(config, plane.session, plane.gateway, plane.binder, clock=plane.clock,
                                 plan=ExecutionPlan(receipts=output / 'receipts', max_ticks=200000,
                                                    poll_interval_seconds=0.05))
                if report.stop_reason != 'target_train_updates_reached' or len(report.updates) != phase['updates']:
                    raise ValueError('training did not reach its declared update boundary')
                final = report.final_revisions[group]
                if final.revision != phase['target_update'] or not final.training_state_reference:
                    raise ValueError('training boundary lacks exact resumable state')
                return {'checkpoint_id': final.checkpoint_id, 'target_update': final.revision,
                        'admitted_examples': sum(o.examples for u in report.updates for o in u.outcomes.values()),
                        'training_tokens': sum(o.tokens for u in report.updates for o in u.outcomes.values()),
                        'sampled_groups': report.sampled_groups, 'trained_groups': len(report.trained_groups),
                        'skipped_groups': len(report.skipped_groups), 'stale_groups': len(report.stale_groups),
                        'receipt_directory': str(report.receipt_directory)}
            panel = self.spec.final if phase['kind'] == 'final' else self.spec.validation
            tasks = self.verify_panel(plane.session, config.taskset.evaluation_split, panel)
            selected = prior['select'] if phase['kind'] == 'final' else prior[f'train_{phase["target_update"]}']
            capability = plane.session.capability
            request = EvaluationRequest(evaluation_id=config.run_id,
                baseline_selector=prior['screen']['baseline_checkpoint_id'], trained_selector=selected['checkpoint_id'],
                seeds=tuple(HeldOutSeed(task_id=t.task_id, seed=t.seed) for t in panel),
                roster=(RosterSlot(agent_instance_id=instance.agent_instance_id, parameter_group_id=group),),
                pin=PinTemplate(run_id=config.run_id, algorithm_plan_hash=config.expanded_plan().plan_hash,
                    wire_api=config.model.wire_api, sampling_transport=config.model.sampling_transport,
                    policy_kind=config.model.policy_kind, model_family=config.model.family,
                    container_image_digest=capability.container_image_digest,
                    container_contract_hash=plane.session.startup.contract.contract_hash,
                    task_family=tasks[0].task_family, topology_id=capability.topology.topology_id),
                split=config.taskset.evaluation_split, scope=ResolutionScope(parameter_group_id=group),
                poll_limit=3600, concurrency=self.spec.evaluation_concurrency)
            receipt = PairedEvaluation(EvaluationResolver(plane.catalog, probe=ProviderArtifactProbe(plane.provider)),
                session=plane.session, gateway=plane.gateway, binder=plane.binder,
                attempt_sink=EvaluationStore(self.root / 'evaluations.sqlite3')).run(request)
            receipt.write(output)
            summary = receipt.to_payload()['paired_summary']
            differences = [row['delta'] for row in summary['rows']]
            rng = random.Random(20260905)
            samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(10000))
            panel_digest = 'sha256:' + hashlib.sha256(json.dumps([t.model_dump() for t in panel], sort_keys=True).encode()).hexdigest()
            return {**summary, 'checkpoint_id': selected['checkpoint_id'],
                    'paired_bootstrap_95_interval': [samples[249], samples[9749]],
                    'bootstrap_seed': 20260905, 'bootstrap_replicates': 10000, 'panel_digest': panel_digest,
                    'target_update': selected['target_update'], 'evaluation_id': config.run_id,
                    'panel': phase['kind'], 'judge_protocol_digest': self.spec.judge_protocol_digest}
