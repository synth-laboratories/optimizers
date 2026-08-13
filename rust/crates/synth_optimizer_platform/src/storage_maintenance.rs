use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{directory_size_bytes, CheckpointRecord, OptimizerError, Result, WorkspaceStore};

const COMPACTION_MANIFEST_NAME: &str = "compaction_manifest.json";
const GEPA_CURSOR_CHECKPOINT_KIND: &str = "gepa_cursor";
const STORAGE_REPORT_NAME: &str = "storage_report.json";
const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_RUN_WARN_BYTES: u64 = 5 * GIB;
const DEFAULT_ROOT_WARN_BYTES: u64 = 20 * GIB;
const DEFAULT_STALE_PARTIAL_WARN_BYTES: u64 = 2 * GIB;
const DEFAULT_PARTIAL_STALE_AFTER_SECONDS: i64 = 2 * 60 * 60;

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

#[derive(Clone, Debug)]
pub struct RunStorageInspectionInput {
    pub run_dir: PathBuf,
    pub run_id: Option<String>,
    pub terminal: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct WorkspaceStorageHealthInput {
    pub roots: Vec<PathBuf>,
    pub thresholds: StorageHealthThresholds,
    pub now_unix_seconds: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct StorageHealthThresholds {
    pub run_warn_bytes: u64,
    pub root_warn_bytes: u64,
    pub stale_partial_warn_bytes: u64,
    pub partial_stale_after_seconds: i64,
}

impl Default for StorageHealthThresholds {
    fn default() -> Self {
        Self {
            run_warn_bytes: DEFAULT_RUN_WARN_BYTES,
            root_warn_bytes: DEFAULT_ROOT_WARN_BYTES,
            stale_partial_warn_bytes: DEFAULT_STALE_PARTIAL_WARN_BYTES,
            partial_stale_after_seconds: DEFAULT_PARTIAL_STALE_AFTER_SECONDS,
        }
    }
}

pub fn inspect_run_storage(input: RunStorageInspectionInput) -> Result<Value> {
    let run_dir = input.run_dir;
    let exists = run_dir.is_dir();
    let bytes = if exists {
        directory_size_bytes(&run_dir)?
    } else {
        0
    };
    let terminal = infer_terminal_status(&run_dir, input.run_id.as_deref(), input.terminal)?;
    let generated_runtime = generated_runtime_report(&run_dir)?;
    let request_cache_bytes = path_size_bytes(&run_dir.join("request_cache.sqlite"))?;
    let rollout_trace_bytes = path_size_bytes(&run_dir.join("rollout_traces"))?;
    let checkpoint_compaction = checkpoint_inspection(&run_dir, input.run_id.as_deref());
    let checkpoint_estimate = checkpoint_compaction
        .get("estimated_reclaim_bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reclaimable_bytes = generated_runtime
        .get("bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .saturating_add(request_cache_bytes)
        .saturating_add(rollout_trace_bytes)
        .saturating_add(checkpoint_estimate);
    let recommendation = storage_recommendation(
        terminal.terminal,
        reclaimable_bytes,
        request_cache_bytes.saturating_add(rollout_trace_bytes),
    );
    Ok(json!({
        "schema": "synth.optimizer.run_storage_report.v1",
        "run_dir": run_dir.display().to_string(),
        "run_id": input.run_id,
        "exists": exists,
        "terminal": terminal.terminal,
        "terminal_status": terminal.status,
        "terminal_source": terminal.source,
        "bytes": bytes,
        "artifact_summary": known_artifact_report(&run_dir)?,
        "top_files": top_file_report(&run_dir, 12)?,
        "sqlite": sqlite_report(&run_dir)?,
        "generated_runtime": generated_runtime,
        "request_cache_bytes": request_cache_bytes,
        "rollout_trace_bytes": rollout_trace_bytes,
        "checkpoint_compaction": checkpoint_compaction,
        "reclaimable_bytes": reclaimable_bytes,
        "compaction_manifest": compaction_manifest_report(&run_dir)?,
        "recommendation": recommendation,
        "safe_actions": if terminal.terminal {
            json!(["inspect", "compact", "delete"])
        } else {
            json!(["inspect"])
        },
    }))
}

pub fn inspect_run_storage_summary(input: RunStorageInspectionInput) -> Result<Value> {
    let run_dir = input.run_dir;
    let exists = run_dir.is_dir();
    let bytes = if exists {
        directory_size_bytes(&run_dir)?
    } else {
        0
    };
    let terminal = infer_terminal_status(&run_dir, input.run_id.as_deref(), input.terminal)?;
    let generated_runtime = generated_runtime_report(&run_dir)?;
    let request_cache_bytes = path_size_bytes(&run_dir.join("request_cache.sqlite"))?;
    let rollout_trace_bytes = path_size_bytes(&run_dir.join("rollout_traces"))?;
    let reclaimable_bytes = generated_runtime
        .get("bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .saturating_add(request_cache_bytes)
        .saturating_add(rollout_trace_bytes);
    let recommendation = storage_recommendation(
        terminal.terminal,
        reclaimable_bytes,
        request_cache_bytes.saturating_add(rollout_trace_bytes),
    );
    Ok(json!({
        "schema": "synth.optimizer.run_storage_summary.v1",
        "run_dir": run_dir.display().to_string(),
        "run_id": input.run_id,
        "exists": exists,
        "terminal": terminal.terminal,
        "terminal_status": terminal.status,
        "terminal_source": terminal.source,
        "bytes": bytes,
        "artifact_summary": known_artifact_report(&run_dir)?,
        "generated_runtime": generated_runtime,
        "request_cache_bytes": request_cache_bytes,
        "rollout_trace_bytes": rollout_trace_bytes,
        "reclaimable_bytes": reclaimable_bytes,
        "compaction_manifest": compaction_manifest_report(&run_dir)?,
        "storage_report_path": run_dir.join(STORAGE_REPORT_NAME).display().to_string(),
        "recommendation": recommendation,
        "safe_actions": if terminal.terminal {
            json!(["inspect", "compact", "delete"])
        } else {
            json!(["inspect"])
        },
    }))
}

pub fn write_run_storage_report(input: RunStorageInspectionInput) -> Result<Value> {
    let run_dir = input.run_dir.clone();
    let mut report = inspect_run_storage_summary(input)?;
    if let Some(object) = report.as_object_mut() {
        object.insert(
            "recorded_at_unix_seconds".to_string(),
            json!(now_unix_seconds()),
        );
    }
    let path = run_dir.join(STORAGE_REPORT_NAME);
    let text = serde_json::to_string_pretty(&report)?;
    fs::write(&path, format!("{text}\n")).map_err(|source| OptimizerError::io(&path, source))?;
    Ok(report)
}

pub fn inspect_workspace_storage_health(input: WorkspaceStorageHealthInput) -> Result<Value> {
    let thresholds = input.thresholds;
    let now = input.now_unix_seconds.unwrap_or_else(now_unix_seconds);
    let mut root_reports = Vec::new();
    let mut alerts = Vec::new();
    let mut total_bytes = 0u64;
    let mut total_reclaimable_bytes = 0u64;
    let mut total_partial_bytes = 0u64;
    let mut stale_partial_bytes = 0u64;
    let mut run_count = 0u64;
    let mut terminal_run_count = 0u64;
    let mut partial_count = 0u64;

    for root in input.roots {
        let root_report = inspect_storage_root(&root, &thresholds, now)?;
        total_bytes = total_bytes.saturating_add(
            root_report
                .get("bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        total_reclaimable_bytes = total_reclaimable_bytes.saturating_add(
            root_report
                .get("reclaimable_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        total_partial_bytes = total_partial_bytes.saturating_add(
            root_report
                .get("partial_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        stale_partial_bytes = stale_partial_bytes.saturating_add(
            root_report
                .get("stale_partial_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        run_count = run_count.saturating_add(
            root_report
                .get("run_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        terminal_run_count = terminal_run_count.saturating_add(
            root_report
                .get("terminal_run_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        partial_count = partial_count.saturating_add(
            root_report
                .get("partial_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        if let Some(items) = root_report.get("alerts").and_then(Value::as_array) {
            alerts.extend(items.iter().cloned());
        }
        root_reports.push(root_report);
    }

    Ok(json!({
        "schema": "synth.optimizer.storage_health.v1",
        "generated_at_unix_seconds": now,
        "thresholds": thresholds_json(&thresholds),
        "summary": {
            "root_count": root_reports.len(),
            "run_count": run_count,
            "terminal_run_count": terminal_run_count,
            "partial_count": partial_count,
            "bytes": total_bytes,
            "reclaimable_bytes": total_reclaimable_bytes,
            "partial_bytes": total_partial_bytes,
            "stale_partial_bytes": stale_partial_bytes,
            "alert_count": alerts.len(),
        },
        "alerts": alerts,
        "roots": root_reports,
    }))
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

fn inspect_storage_root(
    root: &Path,
    thresholds: &StorageHealthThresholds,
    now: i64,
) -> Result<Value> {
    let exists = root.is_dir();
    let mut runs = Vec::new();
    let mut partials = Vec::new();
    let mut alerts = Vec::new();
    let mut bytes = 0u64;
    let mut reclaimable_bytes = 0u64;
    let mut terminal_run_count = 0u64;
    let mut partial_bytes = 0u64;
    let mut stale_partial_bytes = 0u64;

    if exists {
        let read_dir = fs::read_dir(root).map_err(|source| OptimizerError::io(root, source))?;
        for entry in read_dir {
            let entry = entry.map_err(|source| OptimizerError::io(root, source))?;
            let path = entry.path();
            let metadata =
                fs::symlink_metadata(&path).map_err(|source| OptimizerError::io(&path, source))?;
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_string();
            if name.contains(".partial_") {
                let partial = partial_artifact_report(&path, &metadata, now, thresholds)?;
                let item_bytes = partial.get("bytes").and_then(Value::as_u64).unwrap_or(0);
                bytes = bytes.saturating_add(item_bytes);
                partial_bytes = partial_bytes.saturating_add(item_bytes);
                if partial
                    .get("stale")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    stale_partial_bytes = stale_partial_bytes.saturating_add(item_bytes);
                }
                partials.push(partial);
                continue;
            }
            if !metadata.file_type().is_dir() {
                continue;
            }
            let run_id = name;
            let summary = inspect_run_storage_summary(RunStorageInspectionInput {
                run_dir: path.clone(),
                run_id: Some(run_id.clone()),
                terminal: None,
            })?;
            let run_bytes = summary.get("bytes").and_then(Value::as_u64).unwrap_or(0);
            let run_reclaimable = summary
                .get("reclaimable_bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            bytes = bytes.saturating_add(run_bytes);
            reclaimable_bytes = reclaimable_bytes.saturating_add(run_reclaimable);
            if summary
                .get("terminal")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                terminal_run_count = terminal_run_count.saturating_add(1);
            }
            if run_bytes >= thresholds.run_warn_bytes {
                alerts.push(json!({
                    "kind": "run_storage_high",
                    "severity": "warning",
                    "run_id": run_id,
                    "path": path.display().to_string(),
                    "bytes": run_bytes,
                    "threshold_bytes": thresholds.run_warn_bytes,
                    "message": "GEPA run storage exceeds configured per-run threshold",
                }));
            }
            runs.push(run_health_projection(&summary));
        }
    }

    if bytes >= thresholds.root_warn_bytes {
        alerts.push(json!({
            "kind": "root_storage_high",
            "severity": "warning",
            "root": root.display().to_string(),
            "bytes": bytes,
            "threshold_bytes": thresholds.root_warn_bytes,
            "message": "GEPA run root storage exceeds configured threshold",
        }));
    }
    if stale_partial_bytes >= thresholds.stale_partial_warn_bytes {
        alerts.push(json!({
            "kind": "stale_partial_storage_high",
            "severity": "warning",
            "root": root.display().to_string(),
            "bytes": stale_partial_bytes,
            "threshold_bytes": thresholds.stale_partial_warn_bytes,
            "partial_stale_after_seconds": thresholds.partial_stale_after_seconds,
            "message": "stale GEPA partial artifacts exceed configured threshold",
        }));
    }

    runs.sort_by(|left, right| {
        right
            .get("bytes")
            .and_then(Value::as_u64)
            .cmp(&left.get("bytes").and_then(Value::as_u64))
    });
    partials.sort_by(|left, right| {
        right
            .get("bytes")
            .and_then(Value::as_u64)
            .cmp(&left.get("bytes").and_then(Value::as_u64))
    });

    Ok(json!({
        "root": root.display().to_string(),
        "exists": exists,
        "bytes": bytes,
        "run_count": runs.len(),
        "terminal_run_count": terminal_run_count,
        "partial_count": partials.len(),
        "partial_bytes": partial_bytes,
        "stale_partial_bytes": stale_partial_bytes,
        "reclaimable_bytes": reclaimable_bytes,
        "alerts": alerts,
        "runs": runs,
        "partials": partials,
    }))
}

fn partial_artifact_report(
    path: &Path,
    metadata: &fs::Metadata,
    now: i64,
    thresholds: &StorageHealthThresholds,
) -> Result<Value> {
    let modified_at_unix_seconds = metadata
        .modified()
        .ok()
        .and_then(system_time_unix_seconds)
        .unwrap_or(0);
    let age_seconds = if modified_at_unix_seconds > 0 {
        now.saturating_sub(modified_at_unix_seconds)
    } else {
        0
    };
    let bytes = path_size_bytes(path)?;
    Ok(json!({
        "path": path.display().to_string(),
        "owner_hint": partial_owner_hint(path),
        "kind": if metadata.file_type().is_dir() { "directory" } else { "file" },
        "bytes": bytes,
        "modified_at_unix_seconds": modified_at_unix_seconds,
        "age_seconds": age_seconds,
        "stale": age_seconds >= thresholds.partial_stale_after_seconds,
    }))
}

fn partial_owner_hint(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.split(".partial_")
        .next()
        .map(|value| value.trim_end_matches("_cache.sqlite").to_string())
}

fn run_health_projection(summary: &Value) -> Value {
    json!({
        "run_id": summary.get("run_id").cloned().unwrap_or(Value::Null),
        "run_dir": summary.get("run_dir").cloned().unwrap_or(Value::Null),
        "bytes": summary.get("bytes").cloned().unwrap_or(json!(0)),
        "reclaimable_bytes": summary.get("reclaimable_bytes").cloned().unwrap_or(json!(0)),
        "terminal": summary.get("terminal").cloned().unwrap_or(json!(false)),
        "terminal_status": summary.get("terminal_status").cloned().unwrap_or(Value::Null),
        "storage_report_path": summary.get("storage_report_path").cloned().unwrap_or(Value::Null),
        "recommendation": summary.get("recommendation").cloned().unwrap_or(Value::Null),
    })
}

fn thresholds_json(thresholds: &StorageHealthThresholds) -> Value {
    json!({
        "run_warn_bytes": thresholds.run_warn_bytes,
        "root_warn_bytes": thresholds.root_warn_bytes,
        "stale_partial_warn_bytes": thresholds.stale_partial_warn_bytes,
        "partial_stale_after_seconds": thresholds.partial_stale_after_seconds,
    })
}

fn now_unix_seconds() -> i64 {
    system_time_unix_seconds(SystemTime::now()).unwrap_or(0)
}

fn system_time_unix_seconds(value: SystemTime) -> Option<i64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

struct TerminalStatus {
    terminal: bool,
    status: String,
    source: String,
}

fn infer_terminal_status(
    run_dir: &Path,
    run_id: Option<&str>,
    override_terminal: Option<bool>,
) -> Result<TerminalStatus> {
    if let Some(terminal) = override_terminal {
        return Ok(TerminalStatus {
            terminal,
            status: if terminal { "terminal" } else { "non_terminal" }.to_string(),
            source: "caller".to_string(),
        });
    }
    let manifest_path = run_dir.join("result_manifest.json");
    if manifest_path.is_file() {
        let text = fs::read_to_string(&manifest_path)
            .map_err(|source| OptimizerError::io(&manifest_path, source))?;
        let data: Value = serde_json::from_str(&text)?;
        let status = data
            .get("status")
            .or_else(|| data.get("final_status"))
            .or_else(|| data.get("run_status"))
            .and_then(Value::as_str)
            .unwrap_or("terminal")
            .to_string();
        return Ok(TerminalStatus {
            terminal: true,
            status,
            source: "result_manifest".to_string(),
        });
    }
    if let Some(status) = registry_status(run_dir, run_id)? {
        let terminal = matches!(
            status.as_str(),
            "finished" | "succeeded" | "failed" | "cancelled" | "completed"
        );
        return Ok(TerminalStatus {
            terminal,
            status,
            source: "run_registry".to_string(),
        });
    }
    Ok(TerminalStatus {
        terminal: false,
        status: "unknown".to_string(),
        source: "none".to_string(),
    })
}

fn registry_status(run_dir: &Path, run_id: Option<&str>) -> Result<Option<String>> {
    let Some(run_id) = run_id.or_else(|| run_dir.file_name().and_then(|name| name.to_str())) else {
        return Ok(None);
    };
    let registry_path = run_dir
        .parent()
        .map(|parent| parent.join("run_registry.jsonl"));
    let Some(registry_path) = registry_path.filter(|path| path.is_file()) else {
        return Ok(None);
    };
    let text = fs::read_to_string(&registry_path)
        .map_err(|source| OptimizerError::io(&registry_path, source))?;
    let mut latest: Option<String> = None;
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if value.get("run_id").and_then(Value::as_str) != Some(run_id) {
            continue;
        }
        if let Some(status) = value.get("status").and_then(Value::as_str) {
            latest = Some(status.to_string());
        }
    }
    Ok(latest)
}

fn known_artifact_report(run_dir: &Path) -> Result<Vec<Value>> {
    let artifacts = [
        ("result_manifest", "result_manifest.json"),
        ("candidate_registry", "candidate_registry.json"),
        ("best_candidate", "best_candidate.json"),
        ("score_chart", "score_chart.json"),
        ("events", "events.jsonl"),
        ("optimizer_events", "events.optimizer.jsonl"),
        ("normalized_events", "events.normalized.jsonl"),
        ("workspace", "workspace.sqlite"),
        ("transitions", "transitions.sqlite"),
        ("storage_report", STORAGE_REPORT_NAME),
        ("request_cache", "request_cache.sqlite"),
        ("rollout_traces", "rollout_traces"),
        ("proposer_workspaces", "proposer_workspaces"),
    ];
    let mut report = Vec::new();
    for (name, relative) in artifacts {
        let path = run_dir.join(relative);
        if !path.exists() {
            continue;
        }
        report.push(json!({
            "name": name,
            "path": path.display().to_string(),
            "bytes": path_size_bytes(&path)?,
            "kind": if path.is_dir() { "directory" } else { "file" },
        }));
    }
    report.sort_by(|left, right| {
        right
            .get("bytes")
            .and_then(Value::as_u64)
            .cmp(&left.get("bytes").and_then(Value::as_u64))
    });
    Ok(report)
}

fn top_file_report(run_dir: &Path, limit: usize) -> Result<Vec<Value>> {
    if !run_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    collect_file_sizes(run_dir, &mut files)?;
    files.sort_by(|left, right| right.1.cmp(&left.1));
    Ok(files
        .into_iter()
        .take(limit)
        .map(|(path, bytes)| {
            json!({
                "path": path.display().to_string(),
                "relative_path": path
                    .strip_prefix(run_dir)
                    .unwrap_or(path.as_path())
                    .display()
                    .to_string(),
                "bytes": bytes,
            })
        })
        .collect())
}

fn collect_file_sizes(current: &Path, files: &mut Vec<(PathBuf, u64)>) -> Result<()> {
    let read_dir = match fs::read_dir(current) {
        Ok(read_dir) => read_dir,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(OptimizerError::io(current, source)),
    };
    for entry in read_dir {
        let entry = entry.map_err(|source| OptimizerError::io(current, source))?;
        let path = entry.path();
        let metadata =
            fs::symlink_metadata(&path).map_err(|source| OptimizerError::io(&path, source))?;
        if metadata.file_type().is_file() {
            files.push((path, metadata.len()));
        } else if metadata.file_type().is_dir() {
            collect_file_sizes(&path, files)?;
        }
    }
    Ok(())
}

fn sqlite_report(run_dir: &Path) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for name in [
        "workspace.sqlite",
        "request_cache.sqlite",
        "transitions.sqlite",
    ] {
        let path = run_dir.join(name);
        if !path.is_file() {
            continue;
        }
        reports.push(sqlite_file_report(&path)?);
    }
    Ok(reports)
}

fn sqlite_file_report(path: &Path) -> Result<Value> {
    let bytes = path_size_bytes(path)?;
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let connection = match Connection::open_with_flags(path, flags) {
        Ok(connection) => connection,
        Err(error) => {
            return Ok(json!({
                "path": path.display().to_string(),
                "bytes": bytes,
                "dbstat_available": false,
                "error": error.to_string(),
            }));
        }
    };
    let mut statement = match connection.prepare(
        "SELECT name, SUM(pgsize) AS bytes \
         FROM dbstat \
         GROUP BY name \
         ORDER BY bytes DESC \
         LIMIT 20",
    ) {
        Ok(statement) => statement,
        Err(error) => {
            return Ok(json!({
                "path": path.display().to_string(),
                "bytes": bytes,
                "dbstat_available": false,
                "error": error.to_string(),
            }));
        }
    };
    let rows = statement.query_map([], |row| {
        let bytes = row.get::<_, i64>(1)?.max(0) as u64;
        Ok(json!({
            "name": row.get::<_, String>(0)?,
            "bytes": bytes,
        }))
    })?;
    let mut tables = Vec::new();
    for row in rows {
        tables.push(row?);
    }
    Ok(json!({
        "path": path.display().to_string(),
        "bytes": bytes,
        "dbstat_available": true,
        "objects": tables,
    }))
}

fn generated_runtime_report(run_dir: &Path) -> Result<Value> {
    let mut paths = Vec::new();
    let mut bytes = 0u64;
    for path in generated_runtime_dirs(run_dir)? {
        let path_bytes = path_size_bytes(&path)?;
        bytes = bytes.saturating_add(path_bytes);
        paths.push(json!({
            "path": path.display().to_string(),
            "bytes": path_bytes,
        }));
    }
    Ok(json!({
        "bytes": bytes,
        "paths": paths,
    }))
}

fn checkpoint_inspection(run_dir: &Path, run_id: Option<&str>) -> Value {
    match compact_workspace_checkpoints(run_dir, run_id, true) {
        Ok(report) => report,
        Err(error) => json!({
            "enabled": true,
            "dry_run": true,
            "error": error.to_string(),
        }),
    }
}

fn compaction_manifest_report(run_dir: &Path) -> Result<Value> {
    let path = run_dir.join(COMPACTION_MANIFEST_NAME);
    if !path.is_file() {
        return Ok(json!({
            "exists": false,
            "path": path.display().to_string(),
        }));
    }
    let text = fs::read_to_string(&path).map_err(|source| OptimizerError::io(&path, source))?;
    match serde_json::from_str::<Value>(&text) {
        Ok(value) => Ok(json!({
            "exists": true,
            "path": path.display().to_string(),
            "bytes": path_size_bytes(&path)?,
            "profile": value.get("profile"),
            "applied_at": value.get("applied_at"),
            "actual_reclaim_bytes": value.get("actual_reclaim_bytes"),
        })),
        Err(error) => Ok(json!({
            "exists": true,
            "path": path.display().to_string(),
            "bytes": path_size_bytes(&path)?,
            "error": error.to_string(),
        })),
    }
}

fn storage_recommendation(
    terminal: bool,
    reclaimable_bytes: u64,
    destructive_profile_bytes: u64,
) -> Value {
    if !terminal {
        return json!({
            "action": "none",
            "profile": null,
            "reason": "run is not terminal; cleanup actions are disabled",
        });
    }
    if reclaimable_bytes == 0 {
        return json!({
            "action": "none",
            "profile": null,
            "reason": "no known reclaimable GEPA artifacts found",
        });
    }
    let profile = if destructive_profile_bytes > 0 {
        "minimal"
    } else {
        "compact"
    };
    json!({
        "action": "compact",
        "profile": profile,
        "reason": "terminal run has reclaimable local artifacts",
        "estimated_reclaim_bytes": reclaimable_bytes,
    })
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
