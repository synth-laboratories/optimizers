"""Transactional checkpoint event outbox, installed on the authoritative catalog.

SQLite triggers commit facts with their source writes, including writes nested in
a publisher transaction. Consumers page by per-run sequence and deduplicate by
event_id. Historical catalogs start streaming at migration; no invented backfill.
"""
from __future__ import annotations

import json
import sqlite3


def install(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        INSERT OR IGNORE INTO catalog_meta (key, value)
        VALUES ('event_log_id', lower(hex(randomblob(16))));
        CREATE TABLE IF NOT EXISTS checkpoint_event_outbox (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            sequence_number INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            fields TEXT NOT NULL,
            UNIQUE(run_id, sequence_number)
        );
        CREATE TABLE IF NOT EXISTS checkpoint_artifact_observations (
            observation_id TEXT PRIMARY KEY, checkpoint_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoint_alias_history (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, alias TEXT NOT NULL,
            previous_kind TEXT, previous_id TEXT, target_kind TEXT NOT NULL,
            target_id TEXT NOT NULL, recorded_at TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS checkpoint_alias_insert_audit
        AFTER INSERT ON aliases BEGIN
            INSERT INTO checkpoint_alias_history
            (alias,previous_kind,previous_id,target_kind,target_id,recorded_at)
            VALUES (NEW.alias,NULL,NULL,NEW.target_kind,NEW.target_id,NEW.updated_at);
        END;
        CREATE TRIGGER IF NOT EXISTS checkpoint_alias_update_audit
        AFTER UPDATE ON aliases BEGIN
            INSERT INTO checkpoint_alias_history
            (alias,previous_kind,previous_id,target_kind,target_id,recorded_at)
            VALUES (NEW.alias,OLD.target_kind,OLD.target_id,NEW.target_kind,NEW.target_id,NEW.updated_at);
        END;
        CREATE TRIGGER IF NOT EXISTS checkpoint_event_no_update
        BEFORE UPDATE ON checkpoint_event_outbox BEGIN
            SELECT RAISE(ABORT, 'checkpoint events are append-only');
        END;
        CREATE TRIGGER IF NOT EXISTS checkpoint_event_no_delete
        BEFORE DELETE ON checkpoint_event_outbox BEGIN
            SELECT RAISE(ABORT, 'checkpoint events are append-only');
        END;
    """)
    sources = (
        ("checkpoints", "NEW.run_id", "checkpoint.registered", "NEW.created_at",
         "json_object('checkpoint_id', NEW.checkpoint_id, 'update_id', NEW.update_id, "
         "'policy_revision_id', NEW.policy_revision_id, 'parent_checkpoint_id', NEW.parent_checkpoint_id)"),
        ("publication_events", "(SELECT run_id FROM checkpoints WHERE checkpoint_id=NEW.checkpoint_id)",
         "checkpoint.publication_changed", "NEW.recorded_at",
         "json_object('checkpoint_id', NEW.checkpoint_id, 'publication_status', NEW.status)"),
        ("save_attempts", "NEW.run_id", "checkpoint.save_recorded", "NEW.recorded_at",
         "json_object('checkpoint_id', NEW.checkpoint_id, 'update_id', NEW.update_id, "
         "'parameter_group_id', NEW.parameter_group_id, 'outcome', NEW.outcome)"),
        ("checkpoint_artifact_observations", "(SELECT run_id FROM checkpoints WHERE checkpoint_id=NEW.checkpoint_id)",
         "checkpoint.availability_checked", "NEW.recorded_at",
         "json_object('checkpoint_id', NEW.checkpoint_id, 'observation_id', NEW.observation_id, 'health', json(NEW.payload))"),
    )
    for table, run, kind, timestamp, fields in sources:
        # All fragments are implementation constants, never user-supplied SQL.
        conn.executescript(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_checkpoint_event_v1
            AFTER INSERT ON {table} BEGIN
                INSERT INTO checkpoint_event_outbox
                (event_id, run_id, sequence_number, event_type, timestamp, fields)
                VALUES ('evt_' || lower(hex(randomblob(16))), {run},
                    (SELECT COALESCE(MAX(sequence_number), 0) + 1
                     FROM checkpoint_event_outbox WHERE run_id={run}),
                    '{kind}', {timestamp}, {fields});
            END;
        """)
    for table in ('checkpoint_artifact_observations', 'checkpoint_alias_history'):
        for operation in ('UPDATE', 'DELETE'):
            conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                         f"BEFORE {operation} ON {table} BEGIN "
                         "SELECT RAISE(ABORT, 'checkpoint audit records are append-only'); END")
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS checkpoint_evaluation_event_v1
        AFTER INSERT ON evaluation_bindings BEGIN
            INSERT INTO checkpoint_event_outbox
            (event_id,run_id,sequence_number,event_type,timestamp,fields)
            SELECT 'evt_' || lower(hex(randomblob(16))),c.run_id,
                (SELECT COALESCE(MAX(sequence_number),0) FROM checkpoint_event_outbox WHERE run_id=c.run_id)
                    + ROW_NUMBER() OVER (PARTITION BY c.run_id ORDER BY c.checkpoint_id),
                'checkpoint.evaluation_bound',NEW.recorded_at,
                json_object('checkpoint_id',c.checkpoint_id,'evaluation_id',NEW.evaluation_id)
            FROM checkpoints c WHERE c.checkpoint_id IN
                (SELECT value FROM json_each(NEW.payload,'$.resolved_checkpoint_ids'));
        END;
        CREATE TRIGGER IF NOT EXISTS checkpoint_alias_event_v1
        AFTER INSERT ON checkpoint_alias_history
        WHEN NEW.target_kind='checkpoint' BEGIN
            INSERT INTO checkpoint_event_outbox
            (event_id,run_id,sequence_number,event_type,timestamp,fields)
            SELECT 'evt_' || lower(hex(randomblob(16))), c.run_id,
                (SELECT COALESCE(MAX(sequence_number),0)+1 FROM checkpoint_event_outbox WHERE run_id=c.run_id),
                'checkpoint.alias_changed',NEW.recorded_at,
                json_object('checkpoint_id',NEW.target_id,'alias',NEW.alias,
                    'previous_target_id',NEW.previous_id,'previous_target_kind',NEW.previous_kind)
            FROM checkpoints c WHERE c.checkpoint_id=NEW.target_id;
            INSERT INTO checkpoint_event_outbox
            (event_id,run_id,sequence_number,event_type,timestamp,fields)
            SELECT 'evt_' || lower(hex(randomblob(16))), c.run_id,
                (SELECT COALESCE(MAX(sequence_number),0)+1 FROM checkpoint_event_outbox WHERE run_id=c.run_id),
                'checkpoint.alias_removed',NEW.recorded_at,
                json_object('checkpoint_id',NEW.previous_id,'alias',NEW.alias,'new_target_id',NEW.target_id)
            FROM checkpoints c WHERE NEW.previous_kind='checkpoint'
                AND c.checkpoint_id=NEW.previous_id AND NEW.previous_id!=NEW.target_id;
        END;
    """)


def page(conn: sqlite3.Connection, run_id: str, *, after_sequence: int = 0,
         limit: int = 500) -> dict:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError('run_id must be nonempty')
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError('after_sequence must be a nonnegative integer')
    if type(limit) is not int or not 1 <= limit <= 2000:
        raise ValueError('limit must be an integer between 1 and 2000')
    rows = conn.execute(
        'SELECT * FROM checkpoint_event_outbox WHERE run_id=? AND sequence_number>? '
        'ORDER BY sequence_number LIMIT ?', (run_id, after_sequence, limit + 1),
    ).fetchall()
    events = [{
        'schema_version': 'rl_checkpoint_event.v1',
        'event_id': row['event_id'], 'run_id': row['run_id'],
        'sequence_number': row['sequence_number'], 'event_type': row['event_type'],
        'timestamp': row['timestamp'], 'fields': json.loads(row['fields']),
    } for row in rows[:limit]]
    log_id = conn.execute("SELECT value FROM catalog_meta WHERE key='event_log_id'").fetchone()[0]
    return {
        'schema_version': 'optimizer_event_page.v1', 'run_id': run_id,
        'log_id': f'checkpoint_event.v1:{log_id}:{run_id}', 'after_sequence': after_sequence,
        'next_sequence': events[-1]['sequence_number'] if events else after_sequence,
        'has_more': len(rows) > limit, 'terminal': None, 'events': events,
    }
