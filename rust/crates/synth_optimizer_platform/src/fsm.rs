use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{OptimizerError, Result};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionRow {
    pub seq: i64,
    pub ts_unix_ms: i64,
    pub entity_type: String,
    pub entity_id: String,
    pub from_state: Option<String>,
    pub to_state: String,
    pub trigger: String,
    pub generation: Option<i64>,
    pub parent_id: Option<String>,
    pub metadata: Value,
}

#[derive(Clone, Debug)]
pub struct TransitionInput<'a> {
    pub ts_unix_ms: Option<i64>,
    pub entity_type: &'a str,
    pub entity_id: &'a str,
    pub from_state: Option<&'a str>,
    pub to_state: &'a str,
    pub trigger: &'a str,
    pub generation: Option<i64>,
    pub parent_id: Option<&'a str>,
    pub metadata: Value,
}

pub trait StateMachineEntity {
    type State: Copy + Eq + Debug;
    type Trigger: Copy + Debug;

    const ENTITY_TYPE: &'static str;

    fn initial_state() -> Self::State;
    fn state_name(state: Self::State) -> &'static str;
    fn state_from_name(name: &str) -> Option<Self::State>;
    fn trigger_name(trigger: Self::Trigger) -> &'static str;
    fn transition_allowed(from: Self::State, to: Self::State, trigger: Self::Trigger) -> bool;
}

#[derive(Clone)]
pub struct TransitionLog {
    path: PathBuf,
    sink: TransitionSink,
}

impl TransitionLog {
    pub fn open(run_dir: impl AsRef<Path>) -> Result<Self> {
        let path = run_dir.as_ref().join("transitions.sqlite");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS transitions (
                seq INTEGER PRIMARY KEY,
                ts_unix_ms INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                trigger TEXT NOT NULL,
                generation INTEGER,
                parent_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS ix_transitions_entity
                ON transitions(entity_type, entity_id, seq);
            CREATE INDEX IF NOT EXISTS ix_transitions_time
                ON transitions(entity_type, ts_unix_ms);
            CREATE INDEX IF NOT EXISTS ix_transitions_state
                ON transitions(entity_type, to_state, seq);
            "#,
        )?;
        Ok(Self {
            path,
            sink: TransitionSink {
                conn: Arc::new(Mutex::new(conn)),
            },
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sink(&self) -> TransitionSink {
        self.sink.clone()
    }

    pub fn read_run_dir(run_dir: impl AsRef<Path>) -> Result<Vec<TransitionRow>> {
        Self::read_path(run_dir.as_ref().join("transitions.sqlite"))
    }

    pub fn read_path(path: impl AsRef<Path>) -> Result<Vec<TransitionRow>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut stmt = conn.prepare(
            r#"
            SELECT seq, ts_unix_ms, entity_type, entity_id, from_state, to_state,
                   trigger, generation, parent_id, metadata
            FROM transitions
            ORDER BY seq
            "#,
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<i64>>(7)?,
                row.get::<_, Option<String>>(8)?,
                row.get::<_, String>(9)?,
            ))
        })?;
        let mut transitions = Vec::new();
        for row in rows {
            let (
                seq,
                ts_unix_ms,
                entity_type,
                entity_id,
                from_state,
                to_state,
                trigger,
                generation,
                parent_id,
                metadata_json,
            ) = row?;
            transitions.push(TransitionRow {
                seq,
                ts_unix_ms,
                entity_type,
                entity_id,
                from_state,
                to_state,
                trigger,
                generation,
                parent_id,
                metadata: serde_json::from_str(&metadata_json)?,
            });
        }
        Ok(transitions)
    }
}

#[derive(Clone)]
pub struct TransitionSink {
    conn: Arc<Mutex<Connection>>,
}

impl TransitionSink {
    pub fn record(&self, input: TransitionInput<'_>) -> Result<TransitionRow> {
        let metadata_json = serde_json::to_string(&input.metadata)?;
        let ts_unix_ms = input.ts_unix_ms.unwrap_or_else(current_unix_ms);
        let conn = self
            .conn
            .lock()
            .map_err(|_| OptimizerError::Invariant("transition log mutex poisoned".to_string()))?;
        conn.execute(
            r#"
            INSERT INTO transitions (
                ts_unix_ms, entity_type, entity_id, from_state, to_state,
                trigger, generation, parent_id, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![
                ts_unix_ms,
                input.entity_type,
                input.entity_id,
                input.from_state,
                input.to_state,
                input.trigger,
                input.generation,
                input.parent_id,
                metadata_json,
            ],
        )?;
        let seq = conn.last_insert_rowid();
        Ok(TransitionRow {
            seq,
            ts_unix_ms,
            entity_type: input.entity_type.to_string(),
            entity_id: input.entity_id.to_string(),
            from_state: input.from_state.map(str::to_string),
            to_state: input.to_state.to_string(),
            trigger: input.trigger.to_string(),
            generation: input.generation,
            parent_id: input.parent_id.map(str::to_string),
            metadata: input.metadata,
        })
    }

    pub fn latest_state(&self, entity_type: &str, entity_id: &str) -> Result<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| OptimizerError::Invariant("transition log mutex poisoned".to_string()))?;
        conn.query_row(
            r#"
            SELECT to_state
            FROM transitions
            WHERE entity_type = ?1 AND entity_id = ?2
            ORDER BY seq DESC
            LIMIT 1
            "#,
            params![entity_type, entity_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(OptimizerError::from)
    }

    pub fn transition_entity<E: StateMachineEntity>(
        &self,
        entity_id: &str,
        to: E::State,
        trigger: E::Trigger,
        generation: Option<i64>,
        parent_id: Option<&str>,
        metadata: Value,
    ) -> Result<TransitionRow> {
        self.transition_entity_at::<E>(
            None, entity_id, to, trigger, generation, parent_id, metadata,
        )
    }

    pub fn transition_entity_at<E: StateMachineEntity>(
        &self,
        ts_unix_ms: Option<i64>,
        entity_id: &str,
        to: E::State,
        trigger: E::Trigger,
        generation: Option<i64>,
        parent_id: Option<&str>,
        metadata: Value,
    ) -> Result<TransitionRow> {
        let latest = self.latest_state(E::ENTITY_TYPE, entity_id)?;
        let from = match latest.as_deref() {
            Some(name) => Some(E::state_from_name(name).ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "transition log contains unknown {} state {name:?} for entity {entity_id}",
                    E::ENTITY_TYPE
                ))
            })?),
            None => None,
        };
        if let Some(from) = from {
            if !E::transition_allowed(from, to, trigger) {
                return Err(OptimizerError::StateTransition {
                    from: E::state_name(from).to_string(),
                    to: E::state_name(to).to_string(),
                    trigger: E::trigger_name(trigger).to_string(),
                });
            }
        } else if to != E::initial_state() {
            return Err(OptimizerError::StateTransition {
                from: "<missing>".to_string(),
                to: E::state_name(to).to_string(),
                trigger: E::trigger_name(trigger).to_string(),
            });
        }
        self.record(TransitionInput {
            ts_unix_ms,
            entity_type: E::ENTITY_TYPE,
            entity_id,
            from_state: from.map(E::state_name),
            to_state: E::state_name(to),
            trigger: E::trigger_name(trigger),
            generation,
            parent_id,
            metadata,
        })
    }
}

#[derive(Clone)]
pub struct EntityMachine<E: StateMachineEntity> {
    sink: TransitionSink,
    entity_id: String,
    state: E::State,
    generation: Option<i64>,
    parent_id: Option<String>,
    _entity: PhantomData<E>,
}

impl<E: StateMachineEntity> EntityMachine<E> {
    pub fn create(
        sink: TransitionSink,
        entity_id: impl Into<String>,
        trigger: E::Trigger,
        generation: Option<i64>,
        parent_id: Option<String>,
        metadata: Value,
    ) -> Result<Self> {
        let entity_id = entity_id.into();
        let state = E::initial_state();
        sink.transition_entity::<E>(
            &entity_id,
            state,
            trigger,
            generation,
            parent_id.as_deref(),
            metadata,
        )?;
        Ok(Self {
            sink,
            entity_id,
            state,
            generation,
            parent_id,
            _entity: PhantomData,
        })
    }

    pub fn transition(
        &mut self,
        to: E::State,
        trigger: E::Trigger,
        metadata: Value,
    ) -> Result<TransitionRow> {
        let row = self.sink.transition_entity::<E>(
            &self.entity_id,
            to,
            trigger,
            self.generation,
            self.parent_id.as_deref(),
            metadata,
        )?;
        self.state = to;
        Ok(row)
    }

    pub fn state(&self) -> E::State {
        self.state
    }
}

fn current_unix_ms() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = duration.as_millis();
    i64::try_from(millis).unwrap_or(i64::MAX)
}
