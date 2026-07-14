use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use synth_optimizer_platform::{
    CacheConfig, ContainerConfig, OptimizerError, PolicyConfig, ProposerConfig, Result,
};
use uuid::Uuid;

use crate::candidate::MapoCandidate;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoConfig {
    #[serde(default)]
    pub run: MapoRunConfig,
    #[serde(default)]
    pub container: ContainerConfig,
    #[serde(default)]
    pub taskset: MapoTasksetConfig,
    #[serde(default)]
    pub policy: PolicyConfig,
    #[serde(default)]
    pub proposer: ProposerConfig,
    #[serde(default)]
    pub mapo: MapoAlgorithmConfig,
    #[serde(default)]
    pub evidence: MapoEvidenceConfig,
    #[serde(default)]
    pub seed_candidate: MapoCandidate,
    #[serde(default)]
    pub cache: CacheConfig,
}

impl Default for MapoConfig {
    fn default() -> Self {
        Self {
            run: MapoRunConfig::default(),
            container: ContainerConfig::default(),
            taskset: MapoTasksetConfig::default(),
            policy: PolicyConfig::default(),
            proposer: ProposerConfig::default(),
            mapo: MapoAlgorithmConfig::default(),
            evidence: MapoEvidenceConfig::default(),
            seed_candidate: MapoCandidate::seed("mapo_seed"),
            cache: CacheConfig::default(),
        }
    }
}

impl MapoConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut config = Self::from_file_unvalidated(path)?;
        config.resolve_runtime_targets()?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut config = Self::from_toml_file_unvalidated(path)?;
        config.resolve_runtime_targets()?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_file_unvalidated(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if extension == "json" {
            let text =
                fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
            let mut config: Self = serde_json::from_str(&text)?;
            config.resolve_relative_paths(path.parent().unwrap_or_else(|| Path::new(".")));
            Ok(config)
        } else {
            Self::from_toml_file_unvalidated(path)
        }
    }

    pub fn from_toml_file_unvalidated(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
        let mut config: Self = toml::from_str(&text)?;
        config.resolve_relative_paths(path.parent().unwrap_or_else(|| Path::new(".")));
        Ok(config)
    }

    pub fn resolve_runtime_targets(&mut self) -> Result<()> {
        self.container.resolve_pool_target()
    }

    pub fn validate(&self) -> Result<()> {
        if self.run.run_id.trim().is_empty() {
            return Err(OptimizerError::Config("run.run_id is required".to_string()));
        }
        if self
            .container
            .url
            .as_deref()
            .unwrap_or_default()
            .trim()
            .is_empty()
        {
            return Err(OptimizerError::Config(
                "container.url or container.pool.pool_id is required".to_string(),
            ));
        }
        if self.taskset.train_seeds.is_empty() {
            return Err(OptimizerError::Config(
                "taskset.train_seeds must contain at least one seed".to_string(),
            ));
        }
        if self.taskset.heldout_seeds.is_empty() {
            return Err(OptimizerError::Config(
                "taskset.heldout_seeds must contain at least one seed".to_string(),
            ));
        }
        for (left_name, left, right_name, right) in [
            (
                "train_seeds",
                &self.taskset.train_seeds,
                "selection_seeds",
                &self.taskset.selection_seeds,
            ),
            (
                "train_seeds",
                &self.taskset.train_seeds,
                "heldout_seeds",
                &self.taskset.heldout_seeds,
            ),
            (
                "selection_seeds",
                &self.taskset.selection_seeds,
                "heldout_seeds",
                &self.taskset.heldout_seeds,
            ),
        ] {
            if let Some(seed) = left.iter().find(|seed| right.contains(seed)) {
                return Err(OptimizerError::Config(format!(
                    "taskset.{left_name} must not overlap {right_name}; seed {seed} appears in both splits"
                )));
            }
        }
        let has_selection = !self.taskset.selection_seeds.is_empty()
            || !self.taskset.selection_task_instance_ids.is_empty();
        if self.mapo.max_generations == 0 {
            return Err(OptimizerError::Config(
                "mapo.max_generations must be positive".to_string(),
            ));
        }
        if self.mapo.proposals_per_generation == 0 {
            return Err(OptimizerError::Config(
                "mapo.proposals_per_generation must be positive".to_string(),
            ));
        }
        if self.mapo.rollouts_per_candidate == 0 {
            return Err(OptimizerError::Config(
                "mapo.rollouts_per_candidate must be positive".to_string(),
            ));
        }
        if has_selection && self.mapo.selection_rollouts_per_candidate == 0 {
            return Err(OptimizerError::Config(
                "mapo.selection_rollouts_per_candidate must be positive when selection is set"
                    .to_string(),
            ));
        }
        if has_selection && self.mapo.selection_top_k == 0 {
            return Err(OptimizerError::Config(
                "mapo.selection_top_k must be positive when selection is set".to_string(),
            ));
        }
        if self.mapo.branch_selection_enabled {
            if !has_selection {
                return Err(OptimizerError::Config(
                    "mapo.branch_selection_enabled requires selection seeds or task instance ids"
                        .to_string(),
                ));
            }
            if self.mapo.branch_discovery_steps == 0 {
                return Err(OptimizerError::Config(
                    "mapo.branch_discovery_steps must be positive when branch selection is enabled"
                        .to_string(),
                ));
            }
            if self.mapo.branch_rollout_steps == 0 {
                return Err(OptimizerError::Config(
                    "mapo.branch_rollout_steps must be positive when branch selection is enabled"
                        .to_string(),
                ));
            }
            if self.mapo.branch_checkpoints_per_rollout == 0 {
                return Err(OptimizerError::Config(
                    "mapo.branch_checkpoints_per_rollout must be positive when branch selection is enabled"
                        .to_string(),
                ));
            }
            if !matches!(
                self.mapo.branch_checkpoint_strategy.as_str(),
                "reward" | "early" | "late"
            ) {
                return Err(OptimizerError::Config(
                    "mapo.branch_checkpoint_strategy must be reward, early, or late".to_string(),
                ));
            }
        }
        if self.mapo.max_steps == 0 {
            return Err(OptimizerError::Config(
                "mapo.max_steps must be positive".to_string(),
            ));
        }
        if let Some(request_timeout_seconds) = self.mapo.request_timeout_seconds {
            if !request_timeout_seconds.is_finite() || request_timeout_seconds <= 0.0 {
                return Err(OptimizerError::Config(
                    "mapo.request_timeout_seconds must be a finite positive number".to_string(),
                ));
            }
        }
        if !self.mapo.container_connect_timeout_seconds.is_finite()
            || self.mapo.container_connect_timeout_seconds <= 0.0
        {
            return Err(OptimizerError::Config(
                "mapo.container_connect_timeout_seconds must be a finite positive number"
                    .to_string(),
            ));
        }
        if !matches!(
            self.mapo.proposer_mode.as_str(),
            "deterministic_grid" | "external"
        ) {
            return Err(OptimizerError::Config(
                "mapo.proposer_mode must be deterministic_grid or external".to_string(),
            ));
        }
        if !matches!(
            self.evidence.claim_label.as_str(),
            "paper_reported"
                | "reproduced"
                | "directionally_reproduced"
                | "synth_measured"
                | "exploratory"
        ) {
            return Err(OptimizerError::Config(
                "evidence.claim_label is not a debrief.v1 claim label".to_string(),
            ));
        }
        Ok(())
    }

    pub fn run_dir(&self) -> PathBuf {
        self.run.output_dir.join(&self.run.run_id)
    }

    pub fn resolved_config_value(&self) -> Result<Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn resolve_relative_paths(&mut self, base_dir: &Path) {
        if self.run.output_dir.is_relative() {
            self.run.output_dir = base_dir.join(&self.run.output_dir);
        }
        if let Some(cwd) = &self.container.cwd {
            if cwd.is_relative() {
                self.container.cwd = Some(base_dir.join(cwd));
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoEvidenceConfig {
    #[serde(default)]
    pub benchmark_id: String,
    #[serde(default)]
    pub paper_reference: String,
    #[serde(default)]
    pub environment_digest: String,
    #[serde(default)]
    pub model_snapshot: String,
    #[serde(default)]
    pub team_topology: String,
    #[serde(default)]
    pub agent_roles: BTreeMap<String, String>,
    #[serde(default)]
    pub prompt_protocol_digest: String,
    #[serde(default)]
    pub search_budget: Value,
    #[serde(default)]
    pub primary_metric: String,
    #[serde(default)]
    pub noninferiority_margin: Value,
    #[serde(default)]
    pub scorer_version: String,
    #[serde(default)]
    pub token_cost: Value,
    #[serde(default)]
    pub latency: Value,
    #[serde(default)]
    pub known_differences: Vec<String>,
    #[serde(default = "default_claim_label")]
    pub claim_label: String,
}

impl Default for MapoEvidenceConfig {
    fn default() -> Self {
        Self {
            benchmark_id: String::new(),
            paper_reference: String::new(),
            environment_digest: String::new(),
            model_snapshot: String::new(),
            team_topology: String::new(),
            agent_roles: BTreeMap::new(),
            prompt_protocol_digest: String::new(),
            search_budget: Value::Null,
            primary_metric: String::new(),
            noninferiority_margin: Value::Null,
            scorer_version: String::new(),
            token_cost: Value::Null,
            latency: Value::Null,
            known_differences: Vec::new(),
            claim_label: default_claim_label(),
        }
    }
}

fn default_claim_label() -> String {
    "exploratory".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoRunConfig {
    #[serde(default = "default_run_id")]
    pub run_id: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default)]
    pub seed: u64,
}

impl Default for MapoRunConfig {
    fn default() -> Self {
        Self {
            run_id: default_run_id(),
            output_dir: default_output_dir(),
            seed: 0,
        }
    }
}

fn default_run_id() -> String {
    format!("mapo_{}", Uuid::new_v4().simple())
}

fn default_output_dir() -> PathBuf {
    PathBuf::from(".out/mapo_runs")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoTasksetConfig {
    #[serde(default = "default_train_seeds")]
    pub train_seeds: Vec<i64>,
    #[serde(default = "default_selection_seeds")]
    pub selection_seeds: Vec<i64>,
    #[serde(default)]
    pub selection_task_instance_ids: Vec<String>,
    #[serde(default = "default_heldout_seeds")]
    pub heldout_seeds: Vec<i64>,
    #[serde(default)]
    pub task_instance_template: Option<String>,
    #[serde(default)]
    pub task_instance_id: Option<String>,
    #[serde(default)]
    pub env_config: Map<String, Value>,
    #[serde(default)]
    pub context: BTreeMap<String, Value>,
}

impl Default for MapoTasksetConfig {
    fn default() -> Self {
        Self {
            train_seeds: default_train_seeds(),
            selection_seeds: default_selection_seeds(),
            selection_task_instance_ids: Vec::new(),
            heldout_seeds: default_heldout_seeds(),
            task_instance_template: None,
            task_instance_id: None,
            env_config: Map::new(),
            context: BTreeMap::new(),
        }
    }
}

fn default_train_seeds() -> Vec<i64> {
    vec![11]
}

fn default_selection_seeds() -> Vec<i64> {
    Vec::new()
}

fn default_heldout_seeds() -> Vec<i64> {
    vec![12, 13, 14]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapoAlgorithmConfig {
    #[serde(default = "default_proposer_mode")]
    pub proposer_mode: String,
    #[serde(default = "default_max_generations")]
    pub max_generations: usize,
    #[serde(default = "default_proposals_per_generation")]
    pub proposals_per_generation: usize,
    #[serde(default = "default_rollouts_per_candidate")]
    pub rollouts_per_candidate: usize,
    #[serde(default = "default_selection_rollouts_per_candidate")]
    pub selection_rollouts_per_candidate: usize,
    #[serde(default = "default_selection_top_k")]
    pub selection_top_k: usize,
    #[serde(default)]
    pub selection_min_messages_delivered: u64,
    #[serde(default)]
    pub branch_selection_enabled: bool,
    #[serde(default = "default_branch_discovery_steps")]
    pub branch_discovery_steps: usize,
    #[serde(default = "default_branch_rollout_steps")]
    pub branch_rollout_steps: usize,
    #[serde(default = "default_branch_checkpoints_per_rollout")]
    pub branch_checkpoints_per_rollout: usize,
    #[serde(default = "default_branch_checkpoint_min_step")]
    pub branch_checkpoint_min_step: usize,
    #[serde(default = "default_branch_checkpoint_strategy")]
    pub branch_checkpoint_strategy: String,
    #[serde(default = "default_heldout_rollouts_per_arm")]
    pub heldout_rollouts_per_arm: usize,
    #[serde(default = "default_heldout_min_paired_episodes_per_arm")]
    pub heldout_min_paired_episodes_per_arm: usize,
    #[serde(default = "default_heldout_min_success_delta_pp")]
    pub heldout_min_success_delta_pp: f64,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: Option<f64>,
    #[serde(default = "default_container_connect_timeout_seconds")]
    pub container_connect_timeout_seconds: f64,
}

impl Default for MapoAlgorithmConfig {
    fn default() -> Self {
        Self {
            proposer_mode: default_proposer_mode(),
            max_generations: default_max_generations(),
            proposals_per_generation: default_proposals_per_generation(),
            rollouts_per_candidate: default_rollouts_per_candidate(),
            selection_rollouts_per_candidate: default_selection_rollouts_per_candidate(),
            selection_top_k: default_selection_top_k(),
            selection_min_messages_delivered: 0,
            branch_selection_enabled: false,
            branch_discovery_steps: default_branch_discovery_steps(),
            branch_rollout_steps: default_branch_rollout_steps(),
            branch_checkpoints_per_rollout: default_branch_checkpoints_per_rollout(),
            branch_checkpoint_min_step: default_branch_checkpoint_min_step(),
            branch_checkpoint_strategy: default_branch_checkpoint_strategy(),
            heldout_rollouts_per_arm: default_heldout_rollouts_per_arm(),
            heldout_min_paired_episodes_per_arm: default_heldout_min_paired_episodes_per_arm(),
            heldout_min_success_delta_pp: default_heldout_min_success_delta_pp(),
            max_steps: default_max_steps(),
            request_timeout_seconds: default_request_timeout_seconds(),
            container_connect_timeout_seconds: default_container_connect_timeout_seconds(),
        }
    }
}

fn default_proposer_mode() -> String {
    "deterministic_grid".to_string()
}

fn default_max_generations() -> usize {
    4
}

fn default_proposals_per_generation() -> usize {
    4
}

fn default_rollouts_per_candidate() -> usize {
    6
}

fn default_selection_rollouts_per_candidate() -> usize {
    1
}

fn default_selection_top_k() -> usize {
    3
}

fn default_branch_discovery_steps() -> usize {
    24
}

fn default_branch_rollout_steps() -> usize {
    24
}

fn default_branch_checkpoints_per_rollout() -> usize {
    2
}

fn default_branch_checkpoint_min_step() -> usize {
    4
}

fn default_branch_checkpoint_strategy() -> String {
    "reward".to_string()
}

fn default_heldout_rollouts_per_arm() -> usize {
    20
}

fn default_heldout_min_paired_episodes_per_arm() -> usize {
    20
}

fn default_heldout_min_success_delta_pp() -> f64 {
    10.0
}

fn default_max_steps() -> usize {
    160
}

fn default_request_timeout_seconds() -> Option<f64> {
    Some(600.0)
}

fn default_container_connect_timeout_seconds() -> f64 {
    30.0
}

#[derive(Clone, Debug, Default)]
pub struct MapoExecutionOptions {
    pub resume: bool,
    pub dry_run: bool,
}
