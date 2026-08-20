use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use synth_optimizer_platform::{
    CheckpointInput, CheckpointRecord, OptimizationRunStartedInput, OptimizerError, Result,
    WorkspaceStore,
};

use crate::planner::{GepaCursor, GEPA_CURSOR_CHECKPOINT_KIND};

pub const GEPA_CURSOR_FIXTURE_SCHEMA: &str = "gepa_cursor_fixture.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaCursorFixture {
    pub schema: String,
    pub fixture_id: String,
    pub source_run_id: String,
    pub source_checkpoint_id: String,
    pub generation: Option<u64>,
    pub snapshot_sha256: String,
    pub checkpoint: CheckpointRecord,
}

pub fn export_cursor_fixture(record: &CheckpointRecord) -> Result<GepaCursorFixture> {
    if record.is_storage_compacted() {
        return Err(OptimizerError::Config(
            "cannot export a compacted checkpoint as a fixture".to_string(),
        ));
    }
    let snapshot_sha256 = sha256_hex(&serde_json::to_vec(&record.snapshot)?);
    let fixture_id = format!("gepa_fixture_{}", &snapshot_sha256[..16]);
    let source_run_id = record
        .snapshot
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok(GepaCursorFixture {
        schema: GEPA_CURSOR_FIXTURE_SCHEMA.to_string(),
        fixture_id,
        source_run_id,
        source_checkpoint_id: record.checkpoint_id.clone(),
        generation: record.generation,
        snapshot_sha256,
        checkpoint: record.clone(),
    })
}

pub fn fork_cursor_checkpoint(
    record: &CheckpointRecord,
    new_run_id: &str,
) -> Result<CheckpointRecord> {
    if record.is_storage_compacted() {
        return Err(OptimizerError::Config(
            "cannot fork a compacted checkpoint".to_string(),
        ));
    }
    let mut cursor: GepaCursor = serde_json::from_value(record.snapshot.clone())?;
    let parent_run_id = cursor.run_id.clone();
    cursor.run_id = new_run_id.to_string();
    cursor.pending_job_id = None;
    cursor.pending_effect_id = None;
    cursor.pending_reservation_ids.clear();
    cursor.checkpoint_sequence = 1;
    cursor.metadata = merge_metadata(
        cursor.metadata,
        json!({
            "retain": true,
            "fork": {
                "parent_run_id": parent_run_id,
                "parent_checkpoint_id": record.checkpoint_id,
                "parent_sequence": record.sequence_number,
            }
        }),
    );
    crate::episode::reset_episode_origin(&mut cursor);
    let snapshot = serde_json::to_value(&cursor)?;
    let mut metadata = record.metadata.clone();
    metadata.insert("retain".to_string(), json!(true));
    metadata.insert(
        "fork".to_string(),
        json!({
            "parent_run_id": parent_run_id,
            "parent_checkpoint_id": record.checkpoint_id,
            "parent_sequence": record.sequence_number,
        }),
    );
    Ok(CheckpointRecord::from_input(CheckpointInput {
        sequence_number: 1,
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status: "forked",
        run_state: cursor.phase.as_str(),
        reason: Some("forked from retained fixture"),
        generation: record.generation,
        candidate_id: record.best_candidate_id.as_deref(),
        evaluation_stage: Some(cursor.phase.as_str()),
        best_candidate_id: record.best_candidate_id.as_deref(),
        candidate_count: record.candidate_count,
        frontier_count: record.frontier_count,
        rollout_count: record.rollout_count,
        cost_usd: record.cost_usd,
        usage: record.usage.clone(),
        snapshot,
        metadata,
    }))
}

pub fn load_fixture_file(path: impl AsRef<std::path::Path>) -> Result<GepaCursorFixture> {
    let raw = std::fs::read_to_string(path.as_ref())
        .map_err(|source| OptimizerError::io(path.as_ref(), source))?;
    let value: Value = serde_json::from_str(&raw)?;
    if value.get("checkpoint").is_some() {
        return serde_json::from_value(value).map_err(OptimizerError::from);
    }
    let Some(cursor_value) = value.get("cursor") else {
        return Err(OptimizerError::Config(
            "fixture JSON must contain checkpoint or cursor".to_string(),
        ));
    };
    let cursor: GepaCursor = serde_json::from_value(cursor_value.clone())?;
    let candidate_count = cursor.candidates.as_array().map(Vec::len).unwrap_or(0) as u64;
    cursor_fixture_from_cursor(&cursor, candidate_count, candidate_count)
}

pub fn import_cursor_fixture(
    store: &mut WorkspaceStore,
    new_run_id: &str,
    fixture: &GepaCursorFixture,
) -> Result<CheckpointRecord> {
    if fixture.schema != GEPA_CURSOR_FIXTURE_SCHEMA {
        return Err(OptimizerError::Config(format!(
            "unsupported fixture schema {}",
            fixture.schema
        )));
    }
    ensure_run_row(store, new_run_id)?;
    let forked = fork_cursor_checkpoint(&fixture.checkpoint, new_run_id)?;
    store.record_checkpoint(new_run_id, &forked)?;
    if let Ok(cursor) = serde_json::from_value::<GepaCursor>(forked.snapshot.clone()) {
        if let Some(candidates) = cursor.candidates.as_array() {
            store.persist_candidate_registry(new_run_id, candidates)?;
        }
    }
    Ok(forked)
}

pub fn pin_run_checkpoint(
    store: &mut WorkspaceStore,
    run_id: &str,
    checkpoint_id: &str,
) -> Result<CheckpointRecord> {
    store.pin_checkpoint(run_id, checkpoint_id)
}

fn ensure_run_row(store: &WorkspaceStore, run_id: &str) -> Result<()> {
    let output = std::path::Path::new("/tmp/gepa-fixture-import");
    store.record_optimization_run_started(OptimizationRunStartedInput {
        run_id,
        state: "created",
        config: &json!({"run": {"run_id": run_id}}),
        cache_mode: "readwrite",
        cache_namespace: run_id,
        output_dir: output,
        run_dir: output,
        manifest_path: &output.join("manifest.json"),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn merge_metadata(base: Value, update: Value) -> Value {
    let mut merged = base.as_object().cloned().unwrap_or_default();
    if let Some(update) = update.as_object() {
        for (key, value) in update {
            merged.insert(key.clone(), value.clone());
        }
    }
    Value::Object(merged)
}

pub fn cursor_fixture_from_cursor(
    cursor: &GepaCursor,
    candidate_count: u64,
    frontier_count: u64,
) -> Result<GepaCursorFixture> {
    let mut metadata = Map::new();
    metadata.insert("retain".to_string(), json!(true));
    let record = CheckpointRecord::from_input(CheckpointInput {
        sequence_number: cursor.checkpoint_sequence.max(1),
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status: "fixture",
        run_state: cursor.phase.as_str(),
        reason: Some("synthetic fixture"),
        generation: Some(cursor.generation as u64),
        candidate_id: cursor.best_candidate_id.as_deref(),
        evaluation_stage: Some(cursor.phase.as_str()),
        best_candidate_id: cursor.best_candidate_id.as_deref(),
        candidate_count,
        frontier_count,
        rollout_count: cursor.rollout_count as u64,
        cost_usd: cursor.cost_usd,
        usage: cursor.usage.clone(),
        snapshot: serde_json::to_value(cursor)?,
        metadata,
    });
    export_cursor_fixture(&record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::{GepaCursor, GepaCursorPhase};
    use serde_json::json;

    fn candidate(id: &str, parent: Option<&str>, reward: f64, n_scores: usize) -> Value {
        let scores: Vec<Value> = (0..n_scores)
            .map(|i| {
                json!({
                    "example_id": format!("train:{i}"),
                    "task_id": format!("train:{i}"),
                    "reward": reward,
                })
            })
            .collect();
        json!({
            "candidate_id": id,
            "payload": {"stage2_system": format!("prompt for {id}")},
            "lever_bundle": {
                "schema_version": "lever_bundle.v1",
                "bundle_id": id,
                "parent_ids": parent.into_iter().collect::<Vec<_>>(),
                "values": {"stage2_system": format!("prompt for {id}")},
                "mutated_lever_ids": ["stage2_system"],
                "metadata": {}
            },
            "parent_id": parent,
            "source": "reflector:frontier_variation",
            "status": "accepted",
            "minibatch_reward": reward,
            "train_reward": reward,
            "heldout_reward": reward,
            "minibatch_scores": scores,
            "train_scores": [],
            "sensor_frames": [],
            "acceptance_score": null,
            "acceptance_metadata": {}
        })
    }

    fn cursor_with_candidates(
        run_id: &str,
        generation: usize,
        candidates: Vec<Value>,
    ) -> GepaCursor {
        let mut cursor = GepaCursor::new(run_id);
        cursor.phase = GepaCursorPhase::GenerationStart;
        cursor.generation = generation;
        cursor.best_candidate_id = candidates
            .first()
            .and_then(|row| row.get("candidate_id"))
            .and_then(Value::as_str)
            .map(str::to_string);
        cursor.candidates = Value::Array(candidates);
        cursor.train_rows = json!([
            {"task_id": "train:0", "split": "train", "seed": 0},
            {"task_id": "train:1", "split": "train", "seed": 1},
            {"task_id": "train:2", "split": "train", "seed": 2}
        ]);
        cursor.minibatch_rows = cursor.train_rows.clone();
        cursor.reflection_rows = cursor.train_rows.clone();
        cursor.heldout_rows = json!([{"task_id": "test:0", "split": "test", "seed": 0}]);
        cursor.checkpoint_sequence = 4;
        cursor
    }

    fn open_store(label: &str) -> (std::path::PathBuf, WorkspaceStore) {
        let dir = std::env::temp_dir().join(format!(
            "gepa_fixture_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let store = WorkspaceStore::open(dir.join("workspace.sqlite")).expect("open workspace");
        (dir, store)
    }

    #[test]
    fn compacting_skips_retained_generation_start() {
        let (_dir, mut store) = open_store("retain");
        ensure_run_row(&store, "run-a").unwrap();
        let cursor = cursor_with_candidates("run-a", 1, vec![candidate("seed", None, 0.4, 3)]);
        let fixture = cursor_fixture_from_cursor(&cursor, 1, 1).unwrap();
        store
            .record_checkpoint("run-a", &fixture.checkpoint)
            .unwrap();
        let tick = CheckpointRecord::from_input(CheckpointInput {
            sequence_number: 5,
            checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
            status: "running",
            run_state: "proposer_waiting",
            reason: Some("tick"),
            generation: Some(1),
            candidate_id: Some("seed"),
            evaluation_stage: Some("proposer_waiting"),
            best_candidate_id: Some("seed"),
            candidate_count: 1,
            frontier_count: 1,
            rollout_count: 3,
            cost_usd: 0.0,
            usage: json!({}),
            snapshot: json!({"run_id": "run-a", "phase": "proposer_waiting"}),
            metadata: Map::new(),
        });
        store
            .record_checkpoint_compacting_previous("run-a", &tick)
            .unwrap();
        let pinned = store
            .checkpoint_by_id("run-a", &fixture.checkpoint.checkpoint_id)
            .unwrap()
            .unwrap();
        assert!(pinned.is_retained());
        assert!(!pinned.is_storage_compacted());
        assert_eq!(
            pinned
                .snapshot
                .get("candidates")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn fork_twice_is_byte_identical_on_archive_frontier_and_rows() {
        let (_dir, mut store) = open_store("fork");
        let cursor = cursor_with_candidates(
            "parent",
            2,
            vec![
                candidate("seed", None, 0.50, 3),
                candidate("c1", Some("seed"), 0.62, 3),
                candidate("c2", Some("seed"), 0.58, 3),
            ],
        );
        let fixture = cursor_fixture_from_cursor(&cursor, 3, 2).unwrap();
        let a = import_cursor_fixture(&mut store, "arm-low", &fixture).unwrap();
        let b = import_cursor_fixture(&mut store, "arm-medium", &fixture).unwrap();
        let cursor_a: GepaCursor = serde_json::from_value(a.snapshot.clone()).unwrap();
        let cursor_b: GepaCursor = serde_json::from_value(b.snapshot.clone()).unwrap();
        assert_eq!(cursor_a.candidates, cursor_b.candidates);
        assert_eq!(cursor_a.train_rows, cursor_b.train_rows);
        assert_eq!(cursor_a.minibatch_rows, cursor_b.minibatch_rows);
        assert_eq!(cursor_a.heldout_rows, cursor_b.heldout_rows);
        assert_eq!(cursor_a.generation, 2);
        assert_eq!(cursor_a.run_id, "arm-low");
        assert_eq!(cursor_b.run_id, "arm-medium");
        assert!(a.is_retained());
        assert!(b.is_retained());
        let episode_a = crate::episode::read_episode(&cursor_a).expect("fork stamps episode");
        let episode_b = crate::episode::read_episode(&cursor_b).expect("fork stamps episode");
        assert_eq!(episode_a.proposer_rounds, 0);
        assert_eq!(episode_b.proposer_rounds, 0);
        assert_eq!(episode_a.origin.generation, 2);
        assert_eq!(episode_a.origin.rollout_count, cursor_a.rollout_count);
        assert_eq!(episode_a.origin.cost_usd, cursor_a.cost_usd);
    }

    #[test]
    fn compacted_checkpoint_cannot_be_exported() {
        let mut record = CheckpointRecord::from_input(CheckpointInput {
            sequence_number: 1,
            checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
            status: "running",
            run_state: "generation_start",
            reason: None,
            generation: Some(0),
            candidate_id: None,
            evaluation_stage: None,
            best_candidate_id: None,
            candidate_count: 0,
            frontier_count: 0,
            rollout_count: 0,
            cost_usd: 0.0,
            usage: json!({}),
            snapshot: json!({"run_id": "x"}),
            metadata: Map::new(),
        });
        record = record.storage_compacted_summary(12);
        assert!(export_cursor_fixture(&record).is_err());
    }

    #[test]
    fn load_fixture_file_accepts_cursor_json() {
        let dir = std::env::temp_dir().join(format!(
            "gepa_fixture_file_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cursor.json");
        let cursor = cursor_with_candidates("src", 1, vec![candidate("seed", None, 0.5, 2)]);
        std::fs::write(
            &path,
            serde_json::to_vec(&json!({
                "schema": GEPA_CURSOR_FIXTURE_SCHEMA,
                "cursor": cursor,
            }))
            .unwrap(),
        )
        .unwrap();
        let fixture = load_fixture_file(&path).unwrap();
        assert_eq!(fixture.generation, Some(1));
        assert_eq!(fixture.checkpoint.run_state, "generation_start");
    }
}
