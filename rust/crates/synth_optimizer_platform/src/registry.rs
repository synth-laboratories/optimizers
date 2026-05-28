use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

use crate::artifacts::ArtifactPaths;
use crate::cache::CacheMode;
use crate::config::SynthOptimizerConfig;
use crate::error::{OptimizerError, Result};

#[derive(Clone, Debug)]
pub struct RunRegistry {
    path: PathBuf,
}

impl RunRegistry {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn append(&self, entry: &RunRegistryEntry) -> Result<()> {
        if self.contains_terminal_entry(entry)? {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|source| OptimizerError::io(&self.path, source))?;
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{line}").map_err(|source| OptimizerError::io(&self.path, source))
    }

    fn contains_terminal_entry(&self, entry: &RunRegistryEntry) -> Result<bool> {
        if !matches!(
            entry.status.as_str(),
            "started" | "finished" | "failed" | "cancelled"
        ) {
            return Ok(false);
        }
        if !self.path.exists() {
            return Ok(false);
        }
        let file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|source| OptimizerError::io(&self.path, source))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|source| OptimizerError::io(&self.path, source))?;
            if line.trim().is_empty() {
                continue;
            }
            let existing: RunRegistryEntry = serde_json::from_str(&line)?;
            if existing.run_id == entry.run_id && existing.status == entry.status {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRegistryEntry {
    pub ts: String,
    pub run_id: String,
    pub status: String,
    pub run_dir: String,
    pub manifest_path: String,
    pub event_feed_path: String,
    pub normalized_event_feed_path: String,
    pub cache_profile_path: String,
    pub best_candidate_path: String,
    pub candidate_registry_path: String,
    pub frontier_path: String,
    pub cache_mode: String,
    pub cache_namespace: String,
    pub proposer_backend: String,
    pub best_candidate_id: Option<String>,
    pub cost_usd: Option<f64>,
    pub usage: Option<Value>,
}

impl RunRegistryEntry {
    pub fn started(
        paths: &ArtifactPaths,
        config: &SynthOptimizerConfig,
        cache_mode: CacheMode,
        cache_namespace: &str,
    ) -> Self {
        Self::new(paths, config, cache_mode, cache_namespace, "started", None)
    }

    pub fn finished(
        paths: &ArtifactPaths,
        config: &SynthOptimizerConfig,
        cache_mode: CacheMode,
        cache_namespace: &str,
        best_candidate_id: String,
        cost_usd: f64,
        usage: Value,
    ) -> Self {
        let mut entry = Self::new(paths, config, cache_mode, cache_namespace, "finished", None);
        entry.best_candidate_id = Some(best_candidate_id);
        entry.cost_usd = Some(cost_usd);
        entry.usage = Some(usage);
        entry
    }

    pub fn failed(
        paths: &ArtifactPaths,
        config: &SynthOptimizerConfig,
        cache_mode: CacheMode,
        cache_namespace: &str,
        cost_usd: f64,
        usage: Value,
    ) -> Self {
        let mut entry = Self::new(paths, config, cache_mode, cache_namespace, "failed", None);
        entry.cost_usd = Some(cost_usd);
        entry.usage = Some(usage);
        entry
    }

    pub fn cancelled(
        paths: &ArtifactPaths,
        config: &SynthOptimizerConfig,
        cache_mode: CacheMode,
        cache_namespace: &str,
        cost_usd: f64,
        usage: Value,
    ) -> Self {
        let mut entry = Self::new(
            paths,
            config,
            cache_mode,
            cache_namespace,
            "cancelled",
            None,
        );
        entry.cost_usd = Some(cost_usd);
        entry.usage = Some(usage);
        entry
    }

    fn new(
        paths: &ArtifactPaths,
        config: &SynthOptimizerConfig,
        cache_mode: CacheMode,
        cache_namespace: &str,
        status: &str,
        usage: Option<Value>,
    ) -> Self {
        Self {
            ts: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
            run_id: config.run.run_id.clone(),
            status: status.to_string(),
            run_dir: paths.run_dir.display().to_string(),
            manifest_path: paths.manifest_path.display().to_string(),
            event_feed_path: paths.event_feed_path.display().to_string(),
            normalized_event_feed_path: paths.normalized_event_feed_path.display().to_string(),
            cache_profile_path: paths.cache_profile_path.display().to_string(),
            best_candidate_path: paths.best_candidate_path.display().to_string(),
            candidate_registry_path: paths.candidate_registry_path.display().to_string(),
            frontier_path: paths.frontier_path.display().to_string(),
            cache_mode: cache_mode_name(cache_mode).to_string(),
            cache_namespace: cache_namespace.to_string(),
            proposer_backend: config.proposer.backend.clone(),
            best_candidate_id: None,
            cost_usd: None,
            usage,
        }
    }
}

fn cache_mode_name(mode: CacheMode) -> &'static str {
    match mode {
        CacheMode::Off => "off",
        CacheMode::Readwrite => "readwrite",
        CacheMode::Readonly => "readonly",
    }
}
