"""Experiment operations for the existing CISPO HTTP façade and CLI.

One experiment ID owns its worker and projected event history. Source outboxes
remain authoritative; projection cursors commit alongside deduplication records.
"""
from datetime import datetime, timezone
from pathlib import Path
import threading

from .budget import ExperimentBudget
from .catalog import CheckpointCatalog
from .config import from_mapping
from .experiment import ExperimentSpec, ExperimentStore, CoordinationError
from .experiment_driver import ContainerExperimentDriver
from .experiment_runner import run_experiment
from .read_api import checkpoint_details


class ExperimentService:
    def __init__(self, path, *, driver_factory=ContainerExperimentDriver):
        self.store = ExperimentStore(path)
        self.driver_factory = driver_factory
        self._lock = threading.Lock()
        self._workers = {}

    def contains(self, experiment_id):
        try:
            self.store.specification(experiment_id)
            return True
        except CoordinationError:
            return False

    def submit(self, payload, *, start=False):
        spec = ExperimentSpec.model_validate(payload)
        self.store.submit(spec)
        if start:
            self.start(spec.experiment_id)
        return self.get(spec.experiment_id)

    def start(self, experiment_id):
        spec = self.store.specification(experiment_id)
        with self._lock:
            worker = self._workers.get(experiment_id)
            if worker and worker.is_alive():
                return
            def work():
                try:
                    run_experiment(self.store, experiment_id, self.driver_factory(spec))
                except Exception:
                    # The runner persisted a typed blocked state. Never put a
                    # raw provider exception (potential credentials) in events.
                    pass
            worker = threading.Thread(target=work, name='rl-'+experiment_id, daemon=True)
            self._workers[experiment_id] = worker
            worker.start()

    def control(self, experiment_id, action):
        if action == 'start':
            self.start(experiment_id)
        elif action == 'recover':
            self.store.expire_claim(experiment_id)
            snapshot = self.store.snapshot(experiment_id)
            phase = next((p for p in snapshot['phases'] if p['state'] == 'uncertain'), None)
            if phase is None:
                raise CoordinationError('no uncertain phase to reconcile')
            result, digest = self.driver_factory(self.store.specification(experiment_id)).recovered_result(phase['phase'])
            self.store.reconcile_completed(experiment_id, phase['position'], result, evidence_digest=digest)
        else:
            self.store.control(experiment_id, action)
            if action == 'resume':
                self.start(experiment_id)
        return self.get(experiment_id)

    def get(self, experiment_id):
        state = self.store.snapshot(experiment_id)
        config = from_mapping(self.store.specification(experiment_id).run)
        budget = config.budget
        draining = any(p['state'] == 'running' for p in state['phases'])
        status = {'stopped': 'stopping', 'paused': 'pausing'}.get(state['state'], state['state']) if draining else state['state']
        return {**state, 'run_id': experiment_id, 'algorithm': 'cispo',
                'status': status, 'control_boundary': 'phase_drain',
                'budget': ExperimentBudget(budget.ledger, experiment_id, budget.cap_usd).snapshot()}

    def sync_sources(self, experiment_id):
        caught_up = True
        spec = self.store.specification(experiment_id)
        config = from_mapping(spec.run)
        if Path(config.artifacts.catalog).exists():
            with CheckpointCatalog(config.artifacts.catalog) as catalog:
                for phase in spec.phases():
                    run = f'{experiment_id}.{phase["id"]}'
                    source = catalog.event_head(run)['log_id']
                    cursor = self.store.source_cursor(experiment_id, source)
                    page = catalog.event_page(run, after_sequence=cursor)
                    caught_up = caught_up and not page['has_more']
                    events = [{'event_id': e['event_id'], 'event_type': e['event_type'],
                               'sequence': e['sequence_number'], 'timestamp': e['timestamp'],
                               'payload': {**e['fields'], 'segment_run_id': run}} for e in page['events']]
                    for event in events:
                        checkpoint_id = event['payload'].get('checkpoint_id')
                        if checkpoint_id and catalog.has_checkpoint(checkpoint_id):
                            event['payload']['checkpoint_snapshot'] = checkpoint_details(catalog, checkpoint_id)
                    self.store.import_page(experiment_id, source, events, page['next_sequence'])
        policy = config.budget
        from .store import JournalStore
        for phase in spec.phases():
            journal_path = Path(config.artifacts.directory) / experiment_id / phase['id'] / 'receipts' / 'queue_journal.sqlite3'
            if journal_path.exists():
                with JournalStore(journal_path) as journal:
                    run = f'{experiment_id}.{phase["id"]}'
                    source = journal.event_page(run, limit=1)['log_id']
                    cursor = self.store.source_cursor(experiment_id, source)
                    page = journal.event_page(run, cursor)
                    caught_up = caught_up and not page['has_more']
                    self.store.import_page(experiment_id, source, page['events'], page['next_sequence'])
        budget = ExperimentBudget(policy.ledger, experiment_id, policy.cap_usd)
        source = 'experiment_budget'
        cursor = self.store.source_cursor(experiment_id, source)
        events = budget.events(cursor)
        caught_up = caught_up and len(events) < 500
        self.store.import_page(experiment_id, source, events, events[-1]['sequence'] if events else cursor)
        from .evaluation_store import EvaluationStore
        evaluation_path = Path(config.artifacts.directory) / experiment_id / 'evaluations.sqlite3'
        if evaluation_path.exists():
            source = 'experiment_evaluations'
            cursor = self.store.source_cursor(experiment_id, source)
            events = EvaluationStore(evaluation_path).events(cursor)
            caught_up = caught_up and len(events) < 500
            self.store.import_page(experiment_id, source, events, events[-1]['sequence'] if events else cursor)
        return caught_up

    def events(self, experiment_id, after_sequence=0, limit=500):
        caught_up = self.sync_sources(experiment_id)
        page = self.store.events(experiment_id, after_sequence, limit)
        snapshot = self.store.snapshot(experiment_id)
        for event in page['events']:
            event.update(schema_version='training.event.v1', run_id=experiment_id,
                job_id=experiment_id, phase=event['kind'].split('.')[0],
                producer={'service': 'synth-optimizers', 'version': 'rl.experiment.v1', 'commit': 'local'},
                optimizer_run_id=experiment_id, algorithm_id='cispo', attempt_id='experiment-v1',
                sequence_number=event['sequence'], event_type=event['kind'],
                occurred_at=datetime.fromtimestamp(event['timestamp'], timezone.utc).isoformat())
        return {**page, 'schema_version': 'optimizer_event_page.v1', 'run_id': experiment_id,
                'log_id': 'experiment.v1:'+self.store.events(experiment_id, 0, 1)['events'][0]['event_id'],
                'after_sequence': after_sequence,
                'has_more': page['has_more'] or not caught_up,
                'terminal': caught_up and snapshot['state'] in {'completed', 'stopped'}
                    and not any(p['state'] == 'running' for p in snapshot['phases'])}

    def checkpoints(self, experiment_id):
        spec = self.store.specification(experiment_id)
        config = from_mapping(spec.run)
        if not Path(config.artifacts.catalog).exists():
            return {'checkpoints': []}
        with CheckpointCatalog(config.artifacts.catalog) as catalog:
            rows = [checkpoint_details(catalog, view.checkpoint_id)
                    for phase in spec.phases()
                    for view in catalog.list_checkpoints(run_id=f'{experiment_id}.{phase["id"]}')]
        return {'schema_version': 'rl_experiment_checkpoints.v1', 'run_id': experiment_id, 'checkpoints': rows}

    def evaluations(self, experiment_id):
        from .evaluation_store import EvaluationStore
        spec = self.store.specification(experiment_id)
        config = from_mapping(spec.run)
        path = Path(config.artifacts.directory) / experiment_id / 'evaluations.sqlite3'
        panels = []
        if path.exists():
            store = EvaluationStore(path)
            for phase in spec.phases():
                if phase['kind'] in {'validation', 'final'}:
                    panels.append(store.snapshot(f'{experiment_id}.{phase["id"]}'))
        return {'schema_version': 'rl_experiment_evaluations.v1', 'run_id': experiment_id, 'panels': panels}

    def verify_checkpoint(self, experiment_id, checkpoint_id):
        from .plane import build_provider
        from .read_api import verify_checkpoint
        spec = self.store.specification(experiment_id)
        config = from_mapping(spec.run)
        with CheckpointCatalog(config.artifacts.catalog) as catalog:
            view = catalog.describe_checkpoint(checkpoint_id)
            if view.record.run_id not in {f'{experiment_id}.{phase["id"]}' for phase in spec.phases()}:
                raise CoordinationError('checkpoint does not belong to this experiment')
            return verify_checkpoint(catalog, checkpoint_id, build_provider(config))
