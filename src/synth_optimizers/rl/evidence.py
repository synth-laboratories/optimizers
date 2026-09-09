"""Explicit immutable evidence sink, persisted before training admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS rollout_evidence(
                rollout_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL, payload TEXT NOT NULL)''')

    def record(self, rollout_id: str, trace: dict, reward: dict) -> str:
        payload = {'schema_version': 'rl_rollout_evidence.v1', 'rollout_id': rollout_id,
                   'trace': trace, 'reward': reward}
        self._refuse_secrets(payload)
        body = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = 'sha256:' + hashlib.sha256(body.encode()).hexdigest()
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT content_sha256 FROM rollout_evidence WHERE rollout_id=?',
                                  (rollout_id,)).fetchone()
            if existing and existing[0] != digest:
                raise ValueError('sealed rollout evidence changed; refusing overwrite')
            db.execute('INSERT OR IGNORE INTO rollout_evidence VALUES (?,?,?)', (rollout_id, digest, body))
        return digest

    def get(self, rollout_id: str) -> dict:
        with sqlite3.connect(self.path) as db:
            row = db.execute('SELECT payload FROM rollout_evidence WHERE rollout_id=?', (rollout_id,)).fetchone()
        if row is None:
            raise KeyError(rollout_id)
        return json.loads(row[0])

    @classmethod
    def _refuse_secrets(cls, value):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {'authorization', 'api_key', 'credential', 'access_token', 'refresh_token'}:
                    raise ValueError('credential-bearing evidence cannot be persisted')
                cls._refuse_secrets(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                cls._refuse_secrets(child)
