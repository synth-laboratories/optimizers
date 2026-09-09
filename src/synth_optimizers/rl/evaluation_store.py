"""Write-ahead observation markers and immutable paired-evaluation outcomes.

A started panel cannot be silently re-run, including when no outcome survived.
This store deliberately does not infer that absence of evidence means no spend.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .evidence import EvidenceStore


class EvaluationStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS panels(id TEXT PRIMARY KEY, protocol TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts(
                    panel TEXT NOT NULL, arm TEXT NOT NULL, sample INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(panel,arm,sample));
                CREATE TABLE IF NOT EXISTS evaluation_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
                    timestamp REAL NOT NULL, event_type TEXT NOT NULL, payload TEXT NOT NULL);
            ''')

    @staticmethod
    def _event(db, kind, payload):
        db.execute('INSERT INTO evaluation_events(event_id,timestamp,event_type,payload) VALUES (?,?,?,?)',
                   (str(uuid.uuid4()), time.time(), kind, json.dumps(payload, sort_keys=True)))

    def begin(self, request):
        protocol = asdict(request)
        EvidenceStore._refuse_secrets(protocol)
        with sqlite3.connect(self.path) as db:
            db.execute('PRAGMA synchronous=FULL')
            try:
                db.execute('INSERT INTO panels VALUES (?,?)',
                           (request.evaluation_id, json.dumps(protocol, sort_keys=True)))
                self._event(db, 'evaluation.observed', {'evaluation_id': request.evaluation_id})
            except sqlite3.IntegrityError as error:
                raise ValueError('panel already observed; explicit reconciliation required') from error

    def record(self, evaluation_id, row):
        EvidenceStore._refuse_secrets(row)
        payload = json.dumps(row, sort_keys=True, allow_nan=False)
        with sqlite3.connect(self.path) as db:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM panels WHERE id=?', (evaluation_id,)).fetchone():
                raise ValueError('evaluation must be marked observed before recording outcomes')
            key = (evaluation_id, row['arm'], row['sample_index'])
            existing = db.execute('SELECT payload FROM attempts WHERE panel=? AND arm=? AND sample=?', key).fetchone()
            if existing and existing[0] != payload:
                raise ValueError('evaluation outcome is immutable')
            db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?,?)', (*key, payload))
            if not existing:
                self._event(db, 'evaluation.attempt_completed', {'evaluation_id': evaluation_id,
                    'arm': row['arm'], 'sample_index': row['sample_index'],
                    'evidence_reference': {'store': self.path, 'panel': evaluation_id,
                                           'arm': row['arm'], 'sample': row['sample_index']}})

    def events(self, cursor=0, limit=500):
        with sqlite3.connect(self.path) as db:
            rows = db.execute('SELECT sequence,event_id,timestamp,event_type,payload FROM evaluation_events '
                              'WHERE sequence>? ORDER BY sequence LIMIT ?', (cursor, limit)).fetchall()
        return [dict(sequence=r[0], event_id=r[1], timestamp=r[2], event_type=r[3], payload=json.loads(r[4])) for r in rows]

    def snapshot(self, evaluation_id):
        with sqlite3.connect(self.path) as db:
            panel = db.execute('SELECT protocol FROM panels WHERE id=?', (evaluation_id,)).fetchone()
            rows = db.execute('SELECT payload FROM attempts WHERE panel=? ORDER BY arm,sample', (evaluation_id,)).fetchall()
        return {'evaluation_id': evaluation_id, 'observed': panel is not None,
                'protocol': json.loads(panel[0]) if panel else None,
                'attempts': [json.loads(row[0]) for row in rows]}
