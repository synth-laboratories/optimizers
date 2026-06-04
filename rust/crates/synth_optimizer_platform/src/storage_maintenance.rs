use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{directory_size_bytes, CheckpointRecord, OptimizerError, Result, WorkspaceStore};

const COMPACTION_MANIFEST_NAME: &str = "compaction_manifest.json";
const GEPA_CURSOR_CHECKPOINT_KIND: &str = "gepa_cursor";

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageMaintenanceProfile {
    Debug,
    Compact,
    Minimal,
}

impl StorageMaintenanceProfile {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "debug" => Ok(Self::Debug),
            "compact" => Ok(Self::Compact),
            "minimal" => Ok(Self::Minimal),
            other => Err(OptimizerError::Config(format!(
                "unknown storage maintenance profile {other:?}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Compact => "compact",
            Self::Minimal => "minimal",
        }
    }

    fn compact_checkpoints(self) -> bool {
        matches!(self, Self::Compact | Self::Minimal)
    }

    fn remove_rollout_traces(self) -> bool {
        matches!(self, Self::Minimal)
    }

    fn remove_request_cache(self) -> bool {
        matches!(self, Self::Minimal)
    }
}

#[derive(Clone, Debug)]
pub struct RunStorageMaintenanceInput {
    pub run_dir: PathBuf,
    pub run_id: Option<String>,
    pub profile: StorageMaintenanceProfile,
    pub dry_run: bool,
}

pub fn compact_run_storage(input: RunStorageMaintenanceInput) -> Result<Value> {
    let run_dir = input.run_dir;
    let before_bytes = directory_size_bytes(&run_dir)?;
    let mut removed_paths = Vec::new();
    let mut estimated_reclaim_bytes = 0u64;

    let mut candidates = generated_runtime_dirs(&run_dir)?;
    if input.profile.remove_rollout_traces() {
        candidates.push(run_dir.join("rollout_traces"));
    }
    if input.profile.remove_request_cache() {
        candidates.push(run_dir.join("request_cache.sqlite"));
    }
    candidates.sort();
    candidates.dedup();

    for path in candidates {
        if !path.exists() {
            continue;
        }
        let bytes = path_size_bytes(&path)?;
        estimated_reclaim_bytes = estimated_reclaim_bytes.saturating_add(bytes);
        removed_paths.push(json!({
            "path": path.display().to_string(),
            "bytes": bytes,
        }));
        if !input.dry_run {
            remove_path(&path)?;
        }
    }

    let checkpoint_report = if input.profile.compact_checkpoints() {
        compact_workspace_checkpoints(&run_dir, input.run_id.as_deref(), input.dry_run)?
    } else {
        json!({
            "enabled": false,
            "dry_run": input.dry_run,
        })
    };

    let after_bytes = if input.dry_run {
        before_bytes
    } else {
        directory_size_bytes(&run_dir)?
    };
    let report = json!({
        "schema": "synth.optimizer.storage_compaction.v1",
        "run_dir": run_dir.display().to_string(),
        "run_id": input.run_id,
        "profile": input.profile.as_str(),
        "dry_run": input.dry_run,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "estimated_reclaim_bytes": estimated_reclaim_bytes,
        "actual_reclaim_bytes": before_bytes.saturating_sub(after_bytes),
        "removed_paths": removed_paths,
        "checkpoint_compaction": checkpoint_report,
    });
    if !input.dry_run {
        let manifest_path = run_dir.join(COMPACTION_MANIFEST_NAME);
        let text = serde_json::to_string_pretty(&report)?;
        fs::write(&manifest_path, format!("{text}\n"))
            .map_err(|source| OptimizerError::io(&manifest_path, source))?;
    }
    Ok(report)
}

pub fn delete_run_storage(run_dir: impl AsRef<Path>, dry_run: bool) -> Result<Value> {
    let run_dir = run_dir.as_ref();
    let bytes = directory_size_bytes(run_dir)?;
    if !dry_run && run_dir.exists() {
        fs::remove_dir_all(run_dir).map_err(|source| OptimizerError::io(run_dir, source))?;
    }
    Ok(json!({
        "schema": "synth.optimizer.storage_delete.v1",
        "run_dir": run_dir.display().to_string(),
        "dry_run": dry_run,
        "bytes": bytes,
        "deleted": !dry_run,
    }))
}

fn compact_workspace_checkpoints(
    run_dir: &Path,
    run_id: Option<&str>,
    dry_run: bool,
) -> Result<Value> {
    let db_path = run_dir.join("workspace.sqlite");
    if !db_path.exists() {
        return Ok(json!({
            "enabled": true,
            "dry_run": dry_run,
            "workspace_db_path": db_path.display().to_string(),
            "checkpoint_kind": GEPA_CURSOR_CHECKPOINT_KIND,
            "rewritten": 0,
            "kept_full": 0,
            "original_snapshot_bytes": 0,
            "compacted_snapshot_bytes": 0,
            "missing_workspace": true,
        }));
    }
    let run_id = run_id
        .map(str::to_string)
        .or_else(|| {
            run_dir
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            OptimizerError::Config(format!("cannot infer run_id from run dir {run_dir:?}"))
        })?;
    let mut store = WorkspaceStore::open_existing(&db_path)?;
    let checkpoints = store.checkpoint_history(&run_id, Some(GEPA_CURSOR_CHECKPOINT_KIND))?;
    let latest_checkpoint_id = checkpoints
        .last()
        .map(|record| record.checkpoint_id.clone());
    let mut rewritten = 0u64;
    let mut kept_full = 0u64;
    let mut original_snapshot_bytes = 0u64;
    let mut compacted_snapshot_bytes = 0u64;

    for record in checkpoints {
        if Some(record.checkpoint_id.as_str()) == latest_checkpoint_id.as_deref() {
            kept_full = kept_full.saturating_add(1);
            continue;
        }
        if checkpoint_is_compacted(&record) {
            continue;
        }
        let original_bytes = serde_json::to_vec(&record.snapshot)?.len() as u64;
        let compacted = compact_checkpoint_record(record, original_bytes);
        let compacted_bytes = serde_json::to_vec(&compacted.snapshot)?.len() as u64;
        original_snapshot_bytes = original_snapshot_bytes.saturating_add(original_bytes);
        compacted_snapshot_bytes = compacted_snapshot_bytes.saturating_add(compacted_bytes);
        rewritten = rewritten.saturating_add(1);
        if !dry_run {
            store.record_checkpoint(&run_id, &compacted)?;
        }
    }
    if !dry_run && rewritten > 0 {
        store.vacuum()?;
    }
    Ok(json!({
        "enabled": true,
        "dry_run": dry_run,
        "workspace_db_path": db_path.display().to_string(),
        "checkpoint_kind": GEPA_CURSOR_CHECKPOINT_KIND,
        "rewritten": rewritten,
        "kept_full": kept_full,
        "original_snapshot_bytes": original_snapshot_bytes,
        "compacted_snapshot_bytes": compacted_snapshot_bytes,
        "estimated_reclaim_bytes": original_snapshot_bytes.saturating_sub(compacted_snapshot_bytes),
    }))
}

fn compact_checkpoint_record(
    mut record: CheckpointRecord,
    original_snapshot_bytes: u64,
) -> CheckpointRecord {
    let original_checkpoint_id = record.checkpoint_id.clone();
    let original_snapshot_kind = record
        .snapshot
        .get("schema")
        .or_else(|| record.snapshot.get("schema_version"))
        .and_then(Value::as_str)
        .map(str::to_string);
    record.snapshot = json!({
        "schema": "synth.optimizer.checkpoint_summary.v1",
        "compacted": true,
        "original_checkpoint_id": original_checkpoint_id,
        "original_snapshot_kind": original_snapshot_kind,
        "original_snapshot_bytes": original_snapshot_bytes,
        "summary": record.summary(),
    });
    record
        .metadata
        .insert("storage_compacted".to_string(), json!(true));
    record.metadata.insert(
        "storage_compaction_schema".to_string(),
        json!("synth.optimizer.checkpoint_summary.v1"),
    );
    record
}

fn checkpoint_is_compacted(record: &CheckpointRecord) -> bool {
    record
        .snapshot
        .get("compacted")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || record
            .metadata
            .get("storage_compacted")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

fn generated_runtime_dirs(run_dir: &Path) -> Result<Vec<PathBuf>> {
    let proposer_workspaces = run_dir.join("proposer_workspaces");
    if !proposer_workspaces.exists() {
        return Ok(Vec::new());
    }
    let mut matches = Vec::new();
    let mut stack = vec![proposer_workspaces.clone()];
    while let Some(current) = stack.pop() {
        let read_dir = match fs::read_dir(&current) {
            Ok(read_dir) => read_dir,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => return Err(OptimizerError::io(&current, source)),
        };
        for entry in read_dir {
            let entry = entry.map_err(|source| OptimizerError::io(&current, source))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|source| OptimizerError::io(&path, source))?;
            if !file_type.is_dir() {
                continue;
            }
            let name = path.file_name().and_then(|name| name.to_str());
            if matches!(name, Some(".codex_home" | ".codex_api_key_home")) {
                matches.push(path);
            } else {
                stack.push(path);
            }
        }
    }
    Ok(matches)
}

fn path_size_bytes(path: &Path) -> Result<u64> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(source) => return Err(OptimizerError::io(path, source)),
    };
    if metadata.file_type().is_dir() {
        directory_size_bytes(path)
    } else if metadata.file_type().is_file() {
        Ok(metadata.len())
    } else {
        Ok(0)
    }
}

fn remove_path(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(OptimizerError::io(path, source)),
    };
    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path).map_err(|source| OptimizerError::io(path, source))
    } else {
        fs::remove_file(path).map_err(|source| OptimizerError::io(path, source))
    }
}
