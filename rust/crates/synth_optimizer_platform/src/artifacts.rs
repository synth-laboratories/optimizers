use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::artifact_store::{LocalDevStore, RunArtifactStore};
use crate::error::{OptimizerError, Result};

#[derive(Clone, Debug)]
pub struct ArtifactPaths {
    pub run_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub event_feed_path: PathBuf,
    pub normalized_event_feed_path: PathBuf,
    pub optimizer_event_feed_path: PathBuf,
    pub cache_profile_path: PathBuf,
    pub best_candidate_path: PathBuf,
    pub candidate_registry_path: PathBuf,
    pub frontier_path: PathBuf,
    pub score_chart_path: PathBuf,
    pub storage_report_path: PathBuf,
    pub run_registry_path: PathBuf,
    pub workspace_db_path: PathBuf,
    artifact_store: Arc<dyn RunArtifactStore>,
}

impl ArtifactPaths {
    pub fn new(output_dir: impl AsRef<Path>, run_id: &str) -> Self {
        let output_dir = output_dir.as_ref();
        let run_dir = output_dir.join(run_id);
        Self::with_artifact_store(output_dir, run_id, Arc::new(LocalDevStore::new(run_dir)))
    }

    pub fn with_artifact_store(
        output_dir: impl AsRef<Path>,
        run_id: &str,
        artifact_store: Arc<dyn RunArtifactStore>,
    ) -> Self {
        let output_dir = output_dir.as_ref();
        let run_dir = output_dir.join(run_id);
        Self {
            manifest_path: run_dir.join("result_manifest.json"),
            event_feed_path: run_dir.join("events.jsonl"),
            normalized_event_feed_path: run_dir.join("events.normalized.jsonl"),
            optimizer_event_feed_path: run_dir.join("events.optimizer.jsonl"),
            cache_profile_path: run_dir.join("cache_profile.json"),
            best_candidate_path: run_dir.join("best_candidate.json"),
            candidate_registry_path: run_dir.join("candidate_registry.json"),
            frontier_path: run_dir.join("frontier.json"),
            score_chart_path: run_dir.join("score_chart.svg"),
            storage_report_path: run_dir.join("storage_report.json"),
            run_registry_path: output_dir.join("run_registry.jsonl"),
            workspace_db_path: run_dir.join("workspace.sqlite"),
            artifact_store,
            run_dir,
        }
    }

    pub fn create(&self) -> Result<()> {
        fs::create_dir_all(&self.run_dir)
            .map_err(|source| OptimizerError::io(&self.run_dir, source))
    }

    pub fn artifact_store(&self) -> &dyn RunArtifactStore {
        self.artifact_store.as_ref()
    }

    pub fn write_json(&self, path: &Path, value: &Value) -> Result<()> {
        if let Some(path_key) = self.path_key_for_path(path) {
            self.artifact_store.write_json(&path_key, value)?;
            return Ok(());
        }
        let text = serde_json::to_string_pretty(value)?;
        fs::write(path, format!("{text}\n")).map_err(|source| OptimizerError::io(path, source))
    }

    pub fn write_text(&self, path: &Path, value: &str) -> Result<()> {
        if let Some(path_key) = self.path_key_for_path(path) {
            self.artifact_store
                .write_bytes(&path_key, value.as_bytes())?;
            return Ok(());
        }
        fs::write(path, value).map_err(|source| OptimizerError::io(path, source))
    }

    pub fn artifact_ref(&self, path: &Path, kind: &str, retention: &str) -> Result<ArtifactRef> {
        artifact_ref(path, kind, retention)
    }

    fn path_key_for_path(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.run_dir).ok()?;
        let key = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if key.is_empty() {
            None
        } else {
            Some(key)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub path: String,
    pub kind: String,
    pub sha256: String,
    pub bytes: u64,
    pub retention: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GepaCandidateIdentity {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub split: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GepaDeploymentCandidate {
    pub id: String,
    pub rule: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GepaHeldoutMeasurement {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GepaReconciliationStatus {
    pub status: String,
    pub authorities: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GepaRunResult {
    pub best_candidate: Value,
    pub manifest_path: String,
    pub event_feed_path: String,
    pub normalized_event_feed_path: String,
    pub cache_profile_path: String,
    pub candidate_registry_path: String,
    pub frontier_path: String,
    pub score_chart_path: String,
    pub storage_report_path: String,
    pub run_registry_path: String,
    pub workspace_db_path: String,
    pub artifact_refs: Vec<ArtifactRef>,
    /// Complete settled/provider-priced cost, or null when any chargeable
    /// boundary omitted cost authority.
    pub cost_usd: Option<f64>,
    pub usage: Value,
    pub state_history: Value,
    #[serde(default)]
    pub optimization_selected_candidate: Option<GepaCandidateIdentity>,
    #[serde(default)]
    pub heldout_measurements_by_candidate: Vec<GepaHeldoutMeasurement>,
    #[serde(default)]
    pub heldout_best_candidate: Option<GepaCandidateIdentity>,
    #[serde(default)]
    pub deployment_candidate: Option<GepaDeploymentCandidate>,
    #[serde(default)]
    pub improvement_verdict: Option<String>,
    #[serde(default)]
    pub rollout_count_authority: Value,
    #[serde(default)]
    pub cost_authority: Value,
    #[serde(default)]
    pub reconciliation_status: Option<GepaReconciliationStatus>,
    #[serde(default)]
    pub resolved_pipeline: Value,
}

pub fn artifact_ref(path: &Path, kind: &str, retention: &str) -> Result<ArtifactRef> {
    let mut file = fs::File::open(path).map_err(|source| OptimizerError::io(path, source))?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 8192];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| OptimizerError::io(path, source))?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        digest.update(&buffer[..count]);
    }
    Ok(ArtifactRef {
        path: path.display().to_string(),
        kind: kind.to_string(),
        sha256: format!("{:x}", digest.finalize()),
        bytes,
        retention: retention.to_string(),
    })
}
