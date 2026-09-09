"""Frozen experiment specification and durable phase coordination.

The coordinator owns no training math. Drivers use existing RL ports. An expired
claim is uncertain, not permission to replay provider work. Every control and
phase transition commits with its event in the same transaction.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import from_mapping


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class PanelTask(FrozenModel):
    task_id: str = Field(min_length=1)
    seed: int
    content_digest: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')


class Screening(FrozenModel):
    samples: int = Field(default=8, ge=2)
    concurrency: int = Field(default=24, ge=8, le=128)
    rule: str = 'nonzero_reward_range'

    @model_validator(mode='after')
    def rule_supported(self):
        if self.rule not in {'mixed_binary_success', 'nonzero_reward_range'}:
            raise ValueError('unknown screening rule')
        return self


class ExperimentSpec(FrozenModel):
    schema_version: str = 'rl.experiment.v1'
    experiment_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
    run: dict
    train: tuple[PanelTask, ...]
    validation: tuple[PanelTask, ...]
    final: tuple[PanelTask, ...]
    screening: Screening = Field(default_factory=Screening)
    updates: int = Field(default=50, ge=1)
    segment_updates: int = Field(default=15, ge=1, le=15)
    validation_updates: tuple[int, ...] = (10, 25, 50)
    judge_protocol_digest: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    evaluation_url: str = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1)
    evaluation_concurrency: int = Field(default=24, ge=1, le=128)
    benchmark: str = 'container'
    renderer_profile: dict = Field(default_factory=dict)
    judge_protocol: dict = Field(default_factory=dict)
    craftax_env_steps: int = Field(default=200, ge=1, le=100000)
    craftax_policy_calls: int = Field(default=8, ge=1, le=1000)
    judge_input_usd_per_million: float = Field(default=0, ge=0, allow_inf_nan=False)
    judge_output_usd_per_million: float = Field(default=0, ge=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_design(self):
        if self.schema_version != 'rl.experiment.v1':
            raise ValueError('unsupported experiment schema')
        if self.benchmark not in {'container', 'healthbench', 'craftax'}:
            raise ValueError('unsupported benchmark')
        if self.benchmark == 'healthbench' and (not self.judge_input_usd_per_million or not self.judge_output_usd_per_million):
            raise ValueError('HealthBench requires explicit nonzero grader pricing')
        config = from_mapping(self.run)
        from .evidence import EvidenceStore
        from urllib.parse import urlsplit
        EvidenceStore._refuse_secrets(self.run)
        EvidenceStore._refuse_secrets(self.judge_protocol)
        if self.benchmark != 'container':
            digest = 'sha256:' + hashlib.sha256(json.dumps(self.judge_protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if not self.judge_protocol or digest != self.judge_protocol_digest:
                raise ValueError('benchmark judge protocol must match its frozen digest')
            if config.pipeline.max_execution_slots < 8 or config.plan.groups_per_step != 3:
                raise ValueError('benchmark recipes require at least eight rollout slots and three groups per step')
        for endpoint in (config.container.url, self.evaluation_url):
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError('container endpoints must be credential-free HTTP origins')
        if config.budget is None or config.budget.experiment_id != self.experiment_id:
            raise ValueError('experiment requires its own configured aggregate budget')
        if not all((self.train, self.validation, self.final)):
            raise ValueError('all logical panels must be nonempty')
        ids = [task.task_id for panel in (self.train, self.validation, self.final) for task in panel]
        if len(ids) != len(set(ids)):
            raise ValueError('task identities must be unique and split-disjoint')
        if not self.validation_updates or len(set(self.validation_updates)) != len(self.validation_updates):
            raise ValueError('validation checkpoints must be nonempty and distinct')
        if any(update < 1 or update > self.updates for update in self.validation_updates):
            raise ValueError('validation checkpoint outside training schedule')
        # Only references to environment secrets are allowed in the durable spec.
        if config.container.headers:
            raise ValueError('experiment uses auth_bearer_env, not persisted header values')
        return self

    def phases(self) -> list[dict]:
        phases = [{'id': 'screen', 'kind': 'screen'}]
        previous = 0
        boundaries = sorted(set(range(self.segment_updates, self.updates, self.segment_updates)) |
                            set(self.validation_updates) | {self.updates})
        for target in boundaries:
            phases.append({'id': f'train_{target}', 'kind': 'train', 'target_update': target,
                           'updates': target-previous})
            previous = target
        phases.extend({'id': f'validation_{u}', 'kind': 'validation', 'target_update': u}
                      for u in sorted(self.validation_updates))
        phases.extend([{'id': 'select', 'kind': 'select'}, {'id': 'final', 'kind': 'final'}])
        return phases


class CoordinationError(RuntimeError):
    pass


class ExperimentStore:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path, self.clock = str(path), clock
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS experiments(
                    id TEXT PRIMARY KEY, digest TEXT NOT NULL, spec TEXT NOT NULL,
                    state TEXT NOT NULL, blocked_reason TEXT);
                CREATE TABLE IF NOT EXISTS phases(
                    experiment TEXT NOT NULL, position INTEGER NOT NULL, phase TEXT NOT NULL,
                    state TEXT NOT NULL, owner TEXT, lease_until REAL, result TEXT,
                    PRIMARY KEY(experiment,position));
                CREATE TABLE IF NOT EXISTS experiment_events(
                    experiment TEXT NOT NULL, sequence INTEGER NOT NULL, event_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL, timestamp REAL NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(experiment,sequence));
                CREATE TABLE IF NOT EXISTS imported_events(
                    experiment TEXT NOT NULL, source_id TEXT NOT NULL,
                    PRIMARY KEY(experiment,source_id));
                CREATE TABLE IF NOT EXISTS source_cursors(
                    experiment TEXT NOT NULL, source TEXT NOT NULL, cursor INTEGER NOT NULL,
                    PRIMARY KEY(experiment,source));
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def event(self, db, experiment, kind, payload):
        seq = db.execute('SELECT COALESCE(MAX(sequence),0)+1 FROM experiment_events WHERE experiment=?',
                         (experiment,)).fetchone()[0]
        db.execute('INSERT INTO experiment_events VALUES (?,?,?,?,?,?)',
                   (experiment, seq, 'evt_'+uuid.uuid4().hex, kind, self.clock(), json.dumps(payload)))

    def submit(self, spec: ExperimentSpec):
        spec = ExperimentSpec.model_validate(spec.model_dump())
        body = spec.model_dump_json()
        digest = hashlib.sha256(body.encode()).hexdigest()
        with self.db() as db:
            previous = db.execute('SELECT digest FROM experiments WHERE id=?', (spec.experiment_id,)).fetchone()
            if previous:
                if previous[0] != digest:
                    raise CoordinationError('experiment identity already binds a different frozen specification')
                return
            db.execute('INSERT INTO experiments VALUES (?,?,?,\'ready\',NULL)', (spec.experiment_id, digest, body))
            db.executemany('INSERT INTO phases VALUES (?,?,?,\'pending\',NULL,NULL,NULL)',
                           [(spec.experiment_id, i, json.dumps(phase)) for i, phase in enumerate(spec.phases())])
            self.event(db, spec.experiment_id, 'experiment.prepared', {'spec_digest': digest})

    def snapshot(self, experiment):
        with self.db() as db:
            row = db.execute('SELECT * FROM experiments WHERE id=?', (experiment,)).fetchone()
            if row is None:
                raise CoordinationError('unknown experiment')
            phases = [dict(r) for r in db.execute('SELECT * FROM phases WHERE experiment=? ORDER BY position', (experiment,))]
            for phase in phases:
                phase['phase'] = json.loads(phase['phase'])
                phase['result'] = json.loads(phase['result']) if phase['result'] else None
            return {'schema_version': 'rl_experiment_state.v1', 'experiment_id': experiment,
                    'state': row['state'], 'blocked_reason': row['blocked_reason'], 'phases': phases}

    def specification(self, experiment):
        with self.db() as db:
            row = db.execute('SELECT spec FROM experiments WHERE id=?', (experiment,)).fetchone()
            if row is None:
                raise CoordinationError('unknown experiment')
            return ExperimentSpec.model_validate_json(row[0])

    def import_event(self, experiment, source_id, kind, payload):
        """At-least-once source outboxes, deduplicated with the projected event."""
        with self.db() as db:
            inserted = db.execute('INSERT OR IGNORE INTO imported_events VALUES (?,?)', (experiment, source_id))
            if inserted.rowcount:
                self.event(db, experiment, kind, {**payload, 'source_event_id': source_id})

    def assert_owned(self, experiment, claim):
        with self.db() as db:
            self._owned(db, experiment, claim)

    def source_cursor(self, experiment, source):
        with self.db() as db:
            row = db.execute('SELECT cursor FROM source_cursors WHERE experiment=? AND source=?', (experiment,source)).fetchone()
            return row[0] if row else 0

    def import_page(self, experiment, source, events, cursor):
        with self.db() as db:
            for event in events:
                source_id = source + ':' + event['event_id']
                inserted = db.execute('INSERT OR IGNORE INTO imported_events VALUES (?,?)', (experiment, source_id))
                if inserted.rowcount:
                    self.event(db, experiment, event['event_type'], {
                        **event['payload'], 'source_event_id': source_id,
                        'source_sequence': event['sequence'], 'source_timestamp': event['timestamp']})
            db.execute('INSERT INTO source_cursors VALUES (?,?,?) ON CONFLICT(experiment,source) DO UPDATE SET cursor=MAX(cursor,excluded.cursor)',
                       (experiment,source,cursor))

    def claim(self, experiment, *, lease_seconds=60):
        if lease_seconds <= 0:
            raise ValueError('lease_seconds must be positive')
        with self.db() as db:
            run = db.execute('SELECT state FROM experiments WHERE id=?', (experiment,)).fetchone()
            if run is None:
                raise CoordinationError('unknown experiment')
            if run[0] not in {'ready', 'running'}:
                return None
            row = db.execute("SELECT * FROM phases WHERE experiment=? AND state!='completed' ORDER BY position LIMIT 1",
                             (experiment,)).fetchone()
            if row is None:
                db.execute("UPDATE experiments SET state='completed' WHERE id=?", (experiment,))
                self.event(db, experiment, 'experiment.completed', {})
                return None
            if row['state'] == 'running':
                if row['lease_until'] < self.clock():
                    db.execute("UPDATE phases SET state='uncertain' WHERE experiment=? AND position=?", (experiment,row['position']))
                    db.execute("UPDATE experiments SET state='blocked',blocked_reason='expired_phase_needs_reconciliation' WHERE id=?", (experiment,))
                    self.event(db, experiment, 'phase.uncertain', {'position': row['position']})
                    self.event(db, experiment, 'experiment.blocked', {'reason': 'operation_uncertain', 'position': row['position']})
                return None
            if row['state'] != 'pending':
                return None
            owner = uuid.uuid4().hex
            db.execute("UPDATE phases SET state='running',owner=?,lease_until=? WHERE experiment=? AND position=?",
                       (owner, self.clock()+lease_seconds, experiment, row['position']))
            db.execute("UPDATE experiments SET state='running' WHERE id=?", (experiment,))
            self.event(db, experiment, 'phase.started', {'position': row['position'], 'phase': json.loads(row['phase'])})
            return {'position': row['position'], 'owner': owner, 'phase': json.loads(row['phase'])}

    def heartbeat(self, experiment, claim, *, lease_seconds=60):
        with self.db() as db:
            self._owned(db, experiment, claim)
            db.execute('UPDATE phases SET lease_until=? WHERE experiment=? AND position=?',
                       (self.clock()+lease_seconds, experiment, claim['position']))

    def expire_claim(self, experiment):
        """Reconcile liveness only; never admit a pending phase as a side effect."""
        with self.db() as db:
            state = db.execute('SELECT state FROM experiments WHERE id=?', (experiment,)).fetchone()
            if state is None or state[0] in {'completed', 'stopped'}:
                raise CoordinationError('unknown or terminal experiment')
            row = db.execute("SELECT position FROM phases WHERE experiment=? AND state='running' AND lease_until<?",
                             (experiment, self.clock())).fetchone()
            if row:
                db.execute("UPDATE phases SET state='uncertain' WHERE experiment=? AND position=?", (experiment,row[0]))
                db.execute("UPDATE experiments SET state='blocked',blocked_reason='operation_uncertain' WHERE id=?", (experiment,))
                self.event(db, experiment, 'experiment.blocked', {'reason': 'operation_uncertain', 'position': row[0]})

    def _owned(self, db, experiment, claim):
        row = db.execute('SELECT * FROM phases WHERE experiment=? AND position=?', (experiment,claim['position'])).fetchone()
        if row is None or row['state'] != 'running' or row['owner'] != claim['owner'] or row['lease_until'] < self.clock():
            raise CoordinationError('phase claim lost; provider outcome requires reconciliation')

    def complete(self, experiment, claim, result):
        body = json.dumps(result, allow_nan=False)
        with self.db() as db:
            self._owned(db, experiment, claim)
            db.execute("UPDATE phases SET state='completed',result=? WHERE experiment=? AND position=?",
                       (body, experiment, claim['position']))
            self.event(db, experiment, 'phase.completed', {'position': claim['position'], 'result': result})
            pending = db.execute("SELECT 1 FROM phases WHERE experiment=? AND state!='completed' LIMIT 1", (experiment,)).fetchone()
            state = db.execute('SELECT state FROM experiments WHERE id=?', (experiment,)).fetchone()[0]
            if pending is None and state not in {'paused', 'stopped'}:
                db.execute("UPDATE experiments SET state='completed' WHERE id=?", (experiment,))
                self.event(db, experiment, 'experiment.completed', {})

    def block(self, experiment, claim, reason):
        from .experiment_runner import FAILURE_CODES
        if reason not in FAILURE_CODES:
            raise ValueError('unknown blocked reason')
        with self.db() as db:
            self._owned(db, experiment, claim)
            db.execute("UPDATE phases SET state='uncertain' WHERE experiment=? AND position=?",
                       (experiment,claim['position']))
            db.execute("UPDATE experiments SET state='blocked',blocked_reason=? WHERE id=?", (reason,experiment))
            self.event(db, experiment, 'experiment.blocked', {'reason': reason, 'position': claim['position']})

    def reconcile_completed(self, experiment, position, result, *, evidence_digest):
        """Commit a driver-verified durable result; never re-dispatch uncertain work."""
        import re
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', evidence_digest):
            raise ValueError('reconciliation needs a durable evidence digest')
        with self.db() as db:
            state = db.execute('SELECT state FROM experiments WHERE id=?', (experiment,)).fetchone()
            if state is None or state[0] in {'completed', 'stopped'}:
                raise CoordinationError('terminal experiments cannot be resurrected by recovery')
            row = db.execute('SELECT state FROM phases WHERE experiment=? AND position=?', (experiment,position)).fetchone()
            if row is None or row[0] != 'uncertain':
                raise CoordinationError('only uncertain phases can be reconciled')
            db.execute("UPDATE phases SET state='completed',result=? WHERE experiment=? AND position=?",
                       (json.dumps(result, allow_nan=False),experiment,position))
            db.execute("UPDATE experiments SET state='ready',blocked_reason=NULL WHERE id=?", (experiment,))
            self.event(db, experiment, 'phase.reconciled', {'position': position, 'evidence_digest': evidence_digest, 'result': result})

    def control(self, experiment, action):
        if action not in {'pause', 'resume', 'stop'}:
            raise ValueError('unsupported experiment control')
        with self.db() as db:
            row = db.execute('SELECT state FROM experiments WHERE id=?', (experiment,)).fetchone()
            if row is None or row[0] in {'completed', 'stopped'}:
                raise CoordinationError('unknown or terminal experiment')
            if action == 'resume' and row[0] != 'paused':
                raise CoordinationError('only paused experiments resume without reconciliation')
            if action == 'pause' and row[0] not in {'ready', 'running', 'paused'}:
                raise CoordinationError('blocked experiments require reconciliation, not pause/resume')
            state = {'pause': 'paused', 'resume': 'ready', 'stop': 'stopped'}[action]
            db.execute('UPDATE experiments SET state=? WHERE id=?', (state,experiment))
            self.event(db, experiment, f'experiment.{state}', {})

    def events(self, experiment, after_sequence=0, limit=500):
        if type(after_sequence) is not int or after_sequence < 0 or type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError('invalid event cursor/limit')
        with self.db() as db:
            rows = db.execute('SELECT * FROM experiment_events WHERE experiment=? AND sequence>? ORDER BY sequence LIMIT ?',
                              (experiment,after_sequence,limit+1)).fetchall()
            events = [{**dict(row), 'payload': json.loads(row['payload'])} for row in rows[:limit]]
            return {'schema_version': 'rl_experiment_event_page.v1', 'events': events,
                    'next_sequence': events[-1]['sequence'] if events else after_sequence,
                    'has_more': len(rows)>limit}
