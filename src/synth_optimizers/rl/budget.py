"""Restart-safe experiment reservations. No credentials or provider payloads stored."""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import sqlite3
import time
import uuid

from ..providers.protocols import ProviderError


class BudgetError(ProviderError):
    def __init__(self, code: str, message: str):
        super().__init__(code, message, retryable=False)


def micros(value) -> int:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError('cost must be finite and nonnegative')
    return int((amount * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


class ExperimentBudget:
    """An explicitly authorized cap per experiment; operations cannot replay.

    Each call opens its own connection, supporting threaded graders and separate
    workers. Unsettled requests keep their full ceiling through process crashes.
    """
    def __init__(self, path: str | Path, experiment_id: str, cap_usd):
        if not experiment_id.strip():
            raise ValueError('experiment_id is required')
        self.path, self.experiment_id = str(path), experiment_id
        cap = micros(cap_usd)
        if cap == 0:
            raise ValueError('budget cap must be positive')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS budgets(id TEXT PRIMARY KEY, cap INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS charges(
                    experiment TEXT NOT NULL, operation TEXT NOT NULL, lane TEXT NOT NULL,
                    reserved INTEGER NOT NULL, counted INTEGER, status TEXT NOT NULL,
                    PRIMARY KEY(experiment, operation));
                CREATE TABLE IF NOT EXISTS budget_events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, experiment TEXT NOT NULL,
                    event_id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
                    timestamp REAL NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS budget_unreconciled_lookup
                    ON budget_events(experiment, json_extract(payload,'$.reservation_exceeded'));
                CREATE INDEX IF NOT EXISTS budget_reconciliation_lookup
                    ON budget_events(experiment, kind, json_extract(payload,'$.operation_id'));
            ''')
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO budgets VALUES (?,?)', (experiment_id, cap))
            if db.execute('SELECT cap FROM budgets WHERE id=?', (experiment_id,)).fetchone()[0] != cap:
                raise BudgetError('budget_cap_mismatch', 'existing experiment cap cannot be reset')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('PRAGMA synchronous=FULL')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def events(self, after_sequence=0, limit=500):
        if type(after_sequence) is not int or after_sequence < 0 or type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError('invalid event cursor/limit')
        with self._db() as db:
            rows = db.execute('SELECT seq,event_id,kind,timestamp,payload FROM budget_events WHERE experiment=? AND seq>? ORDER BY seq LIMIT ?',
                              (self.experiment_id, after_sequence, limit)).fetchall()
        return [{'sequence': r[0], 'event_id': r[1], 'event_type': r[2],
                 'timestamp': r[3], 'payload': json.loads(r[4])} for r in rows]

    def extend_cap(self, *, expected_cap_usd, new_cap_usd, authorization):
        """Explicit audited increase; construction never silently changes a cap.

        Caller must possess user authorization. A compare-and-swap prevents
        stale approvals overwriting another extension; charges remain intact.
        """
        old, new = micros(expected_cap_usd), micros(new_cap_usd)
        if new <= old or not isinstance(authorization, str) or not authorization.strip():
            raise ValueError('cap increase requires an explicit authorization reference')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT cap FROM budgets WHERE id=?', (self.experiment_id,)).fetchone()[0]
            if current != old:
                raise BudgetError('budget_cap_mismatch', 'cap changed since authorization was prepared')
            db.execute('UPDATE budgets SET cap=? WHERE id=?', (new, self.experiment_id))
            self._event(db, 'budget.cap_extended', {'previous_cap_usd': old/1e6,
                'new_cap_usd': new/1e6, 'authorization': authorization})

    def _event(self, db, kind, payload):
        used = db.execute('SELECT COALESCE(SUM(COALESCE(counted,reserved)),0) FROM charges WHERE experiment=?',
                          (self.experiment_id,)).fetchone()[0]
        cap = db.execute('SELECT cap FROM budgets WHERE id=?', (self.experiment_id,)).fetchone()[0]
        payload = {**payload, 'counted_or_reserved_usd': used / 1e6, 'cap_usd': cap / 1e6,
                   'invoice_reconciled': False}
        db.execute('INSERT INTO budget_events(experiment,event_id,kind,timestamp,payload) VALUES (?,?,?,?,?)',
                   (self.experiment_id, 'evt_' + uuid.uuid4().hex, kind, time.time(), json.dumps(payload)))

    def reserve(self, operation: str, lane: str, upper_usd) -> None:
        if not operation or not lane:
            raise ValueError('operation and lane are required')
        upper = micros(upper_usd)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("""SELECT 1 FROM budget_events exceeded
                          WHERE exceeded.experiment=?
                            AND json_extract(exceeded.payload,'$.reservation_exceeded')=1
                            AND NOT EXISTS (
                              SELECT 1 FROM budget_events reconciled
                              WHERE reconciled.experiment=exceeded.experiment
                                AND reconciled.kind='budget.pricing_reconciled'
                                AND json_extract(reconciled.payload,'$.operation_id')=
                                    json_extract(exceeded.payload,'$.operation_id'))
                          LIMIT 1""", (self.experiment_id,)).fetchone():
                raise BudgetError('pricing_reconciliation_required', 'a prior call exceeded its reservation')
            if db.execute('SELECT 1 FROM charges WHERE experiment=? AND operation=?',
                          (self.experiment_id, operation)).fetchone():
                raise BudgetError('operation_already_admitted', 'reconcile existing operation; do not replay')
            used = db.execute('SELECT COALESCE(SUM(COALESCE(counted,reserved)),0) FROM charges WHERE experiment=?',
                              (self.experiment_id,)).fetchone()[0]
            cap = db.execute('SELECT cap FROM budgets WHERE id=?', (self.experiment_id,)).fetchone()[0]
            if used + upper > cap:
                raise BudgetError('experiment_budget_exhausted', 'aggregate reservation exceeds experiment cap')
            db.execute('INSERT INTO charges VALUES (?,?,?,?,NULL,?)',
                       (self.experiment_id, operation, lane, upper, 'reserved'))
            self._event(db, 'budget.reserved', {'operation_id': operation, 'lane': lane, 'microusd': upper})

    def settle(self, operation: str, counted_usd=None, *, duration_seconds=None) -> None:
        if duration_seconds is not None:
            import math
            if not math.isfinite(duration_seconds) or duration_seconds < 0:
                raise ValueError('operation duration must be finite and nonnegative')
        overrun = False
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT reserved,counted,status,lane FROM charges WHERE experiment=? AND operation=?',
                             (self.experiment_id, operation)).fetchone()
            if row is None:
                raise BudgetError('unknown_operation', 'no reservation exists')
            counted = row[0] if counted_usd is None else micros(counted_usd)
            status = 'conservative' if counted_usd is None else 'usage_counted'
            if row[1] is not None:
                if row[1] == counted and row[2] == status:
                    return
                raise BudgetError('settlement_conflict', 'settlement cannot be rewritten')
            overrun = counted > row[0]
            db.execute('UPDATE charges SET counted=?,status=? WHERE experiment=? AND operation=?',
                       (counted, status, self.experiment_id, operation))
            self._event(db, 'budget.settled', {'operation_id': operation, 'microusd': counted,
                                            'accounting': status, 'reservation_exceeded': overrun,
                                            'lane': row[3], 'duration_seconds': duration_seconds})
        if overrun:
            # Persist the observed liability even when the pricing contract failed.
            raise BudgetError('reservation_exceeded', 'provider cost exceeded reserved ceiling')

    def reconcile_pricing_overrun(self, operation: str, *, evidence: str) -> None:
        """Acknowledge a durably counted overrun without rewriting its charge."""
        if not operation or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('pricing reconciliation requires operation and evidence')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(
                'SELECT reserved,counted,status FROM charges WHERE experiment=? AND operation=?',
                (self.experiment_id, operation),
            ).fetchone()
            if row is None or row[1] is None or row[1] <= row[0]:
                raise BudgetError('no_pricing_overrun', 'operation has no counted reservation overrun')
            if db.execute("""SELECT 1 FROM budget_events WHERE experiment=?
                              AND kind='budget.pricing_reconciled'
                              AND json_extract(payload,'$.operation_id')=?""",
                          (self.experiment_id, operation)).fetchone():
                return
            self._event(db, 'budget.pricing_reconciled', {
                'operation_id': operation, 'reserved_usd': row[0] / 1e6,
                'counted_usd': row[1] / 1e6, 'evidence': evidence,
            })

    def operation(self, operation: str):
        """Read a reservation without admitting or settling any work."""
        with self._db() as db:
            row = db.execute('SELECT reserved,counted,status,lane FROM charges WHERE experiment=? AND operation=?',
                             (self.experiment_id, operation)).fetchone()
        return None if row is None else dict(reserved_microusd=row[0], counted_microusd=row[1],
                                             status=row[2], lane=row[3])

    def snapshot(self) -> dict:
        with self._db() as db:
            db.execute('BEGIN')
            cap = db.execute('SELECT cap FROM budgets WHERE id=?', (self.experiment_id,)).fetchone()[0]
            rows = db.execute('SELECT reserved,counted,status FROM charges WHERE experiment=?',
                              (self.experiment_id,)).fetchall()
            used = sum(r if c is None else c for r, c, _ in rows)
            return {'schema_version': 'rl_budget.v1', 'experiment_id': self.experiment_id,
                    'cap_usd': cap / 1e6, 'counted_or_reserved_usd': used / 1e6,
                    'remaining_usd': max(0, cap-used) / 1e6,
                    'unsettled_operations': sum(c is None for _, c, _ in rows),
                    'provider_reported_usd': None, 'invoice_reconciled': False}


class BudgetedProvider:
    """Explicit provider decorator; pricing is supplied, never silently guessed.

    Prices are USD per million tokens. Session creation/checkpoint storage pricing
    is outside this token adapter and must be reserved separately if nonzero.
    """
    def __init__(self, provider, budget, *, input_rate, output_rate, training_rate):
        self.provider, self.budget = provider, budget
        if getattr(provider, 'max_attempts', 1) != 1:
            raise BudgetError('unsafe_provider_retry_policy',
                              'budgeted provider must disable internal retries before wrapping')
        self.input_rate, self.output_rate, self.training_rate = map(
            Decimal, map(str, (input_rate, output_rate, training_rate)))
        for rate in (self.input_rate, self.output_rate, self.training_rate):
            if not rate.is_finite() or rate < 0:
                raise ValueError('token prices must be finite and nonnegative')

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def _call(self, method, identity, request):
        sampling = method in ('sample', 'sample_checkpoint')
        if sampling:
            upper = (len(request.prompt_token_ids)*self.input_rate + request.max_tokens*self.output_rate)/1_000_000
        elif method == 'forward':
            upper = sum(map(len, request.token_ids))*self.training_rate/1_000_000
        else:
            upper = sum(len(r.get('token_ids') or r.get('input_ids') or ()) +
                        len(r.get('prompt_token_ids') or ()) for r in request.data)*self.training_rate/1_000_000
        operation = f'{method}:{request.request_id}'
        self.budget.reserve(operation, method, upper)
        # No wrapper retries. Any exception retains the reservation.
        started = time.monotonic()
        result = getattr(self.provider, method)(identity, request)
        counted = ((len(request.prompt_token_ids)*self.input_rate + len(result.token_ids)*self.output_rate)/1_000_000
                   if sampling else upper)
        self.budget.settle(operation, counted, duration_seconds=time.monotonic()-started)
        return result

    def sample(self, identity, request):
        return self._call('sample', identity, request)

    def sample_checkpoint(self, identity, request):
        return self._call('sample_checkpoint', identity, request)

    def train_step(self, identity, request):
        return self._call('train_step', identity, request)

    def forward(self, identity, request):
        return self._call('forward', identity, request)
