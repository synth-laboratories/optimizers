use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::configured_limits::validate_gepa_limit_config;
use crate::disk_budget::DiskBudgetConfig;
use crate::error::{OptimizerError, Result};

fn default_output_dir() -> PathBuf {
    PathBuf::from("runs")
}

fn default_startup_timeout_seconds() -> u64 {
    30
}

fn default_train_split() -> String {
    "train".to_string()
}

fn default_heldout_split() -> String {
    "test".to_string()
}

fn default_policy_provider() -> String {
    "openai".to_string()
}

fn default_policy_model() -> String {
    "gpt-4.1-nano".to_string()
}

fn default_proposer_backend() -> String {
    "codex_app_server".to_string()
}

fn default_execution_mode() -> String {
    "local_process".to_string()
}

fn default_proposer_auth_mode() -> String {
    "auto".to_string()
}

fn default_timeout_seconds() -> u64 {
    300
}

fn default_max_generations() -> usize {
    2
}

fn default_proposals_per_generation() -> usize {
    2
}

fn default_minibatch_size() -> usize {
    8
}

fn default_max_total_rollouts() -> usize {
    256
}

fn default_rollout_submission_mode() -> String {
    "sync".to_string()
}

fn default_rollout_poll_interval_ms() -> u64 {
    250
}

fn default_rollout_async_timeout_seconds() -> u64 {
    600
}

fn default_frontier_type() -> String {
    "per_example".to_string()
}

fn default_candidate_selector_name() -> String {
    "pareto_weighted".to_string()
}

fn default_candidate_selector_config() -> GepaCandidateSelectorConfig {
    GepaCandidateSelectorConfig::default()
}

fn default_batch_sampler_name() -> String {
    "seeded_shuffle".to_string()
}

fn default_batch_sampler_config() -> GepaBatchSamplerConfig {
    GepaBatchSamplerConfig::default()
}

fn default_seed_pools_config() -> GepaSeedPoolsConfig {
    GepaSeedPoolsConfig::default()
}

fn default_acceptance_criterion() -> String {
    "primary_improvement".to_string()
}

fn default_objective_acceptance_config() -> GepaObjectiveAcceptanceConfig {
    GepaObjectiveAcceptanceConfig::default()
}

fn default_gepa_pipeline_config() -> GepaPipelineConfig {
    GepaPipelineConfig::default()
}

fn default_pipeline_max_in_flight_candidates() -> usize {
    8
}

fn default_pipeline_proposal_workers() -> usize {
    1
}

fn default_pipeline_rollout_workers() -> usize {
    8
}

fn default_pipeline_evaluate_workers() -> usize {
    1
}

fn default_gepa_adaptive_rollout_concurrency_enabled() -> bool {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_ENABLED
}

fn default_gepa_adaptive_rollout_concurrency_initial() -> usize {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_INITIAL
}

fn default_gepa_adaptive_rollout_concurrency_min() -> usize {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_MIN
}

fn default_gepa_adaptive_rollout_concurrency_max() -> usize {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_MAX
}

fn default_gepa_adaptive_rollout_concurrency_increase_step() -> usize {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_INCREASE_STEP
}

fn default_gepa_adaptive_rollout_concurrency_decrease_step() -> usize {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_DECREASE_STEP
}

fn default_gepa_adaptive_rollout_concurrency_increase_after_successes() -> usize {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_INCREASE_AFTER_SUCCESSES
}

fn default_gepa_adaptive_rollout_concurrency_overload_status_codes() -> Vec<u16> {
    crate::limits::GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_OVERLOAD_STATUS_CODES.to_vec()
}

fn default_cache_mode() -> CacheConfigMode {
    CacheConfigMode::Readwrite
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthOptimizerConfig {
    #[serde(default)]
    pub run: RunConfig,
    #[serde(default)]
    pub container: ContainerConfig,
    #[serde(default)]
    pub dataset: DatasetConfig,
    #[serde(default)]
    pub candidate: CandidateConfig,
    #[serde(default)]
    pub seed_candidate: BTreeMap<String, String>,
    #[serde(default)]
    pub policy: PolicyConfig,
    #[serde(default)]
    pub proposer: ProposerConfig,
    #[serde(default)]
    pub gepa: GepaConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub disk_budget: DiskBudgetConfig,
}

impl SynthOptimizerConfig {
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
        let mut config: Self = toml::from_str(&text)?;
        config.apply_env_overrides()?;
        config.resolve_relative_paths(path.parent().unwrap_or_else(|| Path::new(".")));
        config.validate()?;
        Ok(config)
    }

    fn apply_env_overrides(&mut self) -> Result<()> {
        if let Some(run_id) =
            read_env_override(&["SYNTH_OPTIMIZERS_RUN_ID", "GEPA_PLATFORM_RUN_ID"])
        {
            self.run.run_id = run_id;
        }
        if let Some(output_dir) =
            read_env_override(&["SYNTH_OPTIMIZERS_OUTPUT_DIR", "GEPA_PLATFORM_OUTPUT_DIR"])
        {
            self.run.output_dir = PathBuf::from(output_dir);
        }
        if let Some(cache_namespace) = read_env_override(&[
            "SYNTH_OPTIMIZERS_CACHE_NAMESPACE",
            "GEPA_PLATFORM_CACHE_NAMESPACE",
        ]) {
            self.cache.namespace = Some(cache_namespace);
        }
        if let Some(cache_path) =
            read_env_override(&["SYNTH_OPTIMIZERS_CACHE_PATH", "GEPA_PLATFORM_CACHE_PATH"])
        {
            self.cache.path = Some(PathBuf::from(cache_path));
        }
        if let Some(cache_mode) =
            read_env_override(&["SYNTH_OPTIMIZERS_CACHE_MODE", "GEPA_PLATFORM_CACHE_MODE"])
        {
            self.cache.mode = parse_cache_mode_override(&cache_mode)?;
        }
        if let Some(proposer_backend) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_BACKEND",
            "GEPA_PLATFORM_PROPOSER_BACKEND",
        ]) {
            self.proposer.backend = proposer_backend;
        }
        if let Some(rollout_submission_mode) =
            read_env_override(&["SYNTH_OPTIMIZERS_ROLLOUT_SUBMISSION_MODE"])
        {
            self.gepa.rollout_submission_mode = rollout_submission_mode.trim().to_ascii_lowercase();
        }
        if let Some(poll_interval_ms) =
            read_env_override(&["SYNTH_OPTIMIZERS_ROLLOUT_POLL_INTERVAL_MS"])
        {
            self.gepa.rollout_poll_interval_ms = parse_u64_override(
                "SYNTH_OPTIMIZERS_ROLLOUT_POLL_INTERVAL_MS",
                &poll_interval_ms,
            )?;
        }
        if let Some(timeout_seconds) =
            read_env_override(&["SYNTH_OPTIMIZERS_ROLLOUT_ASYNC_TIMEOUT_SECONDS"])
        {
            self.gepa.rollout_async_timeout_seconds = parse_u64_override(
                "SYNTH_OPTIMIZERS_ROLLOUT_ASYNC_TIMEOUT_SECONDS",
                &timeout_seconds,
            )?;
        }
        if let Some(pipeline_mode) = read_env_override(&["SYNTH_OPTIMIZERS_GEPA_PIPELINE_MODE"]) {
            self.gepa.pipeline.mode = parse_gepa_pipeline_mode_override(&pipeline_mode)?;
        }
        if let Some(staleness_policy) =
            read_env_override(&["SYNTH_OPTIMIZERS_GEPA_STALENESS_POLICY"])
        {
            self.gepa.pipeline.staleness_policy =
                parse_gepa_staleness_policy_override(&staleness_policy)?;
        }
        if let Some(max_in_flight) =
            read_env_override(&["SYNTH_OPTIMIZERS_GEPA_MAX_IN_FLIGHT_CANDIDATES"])
        {
            self.gepa.pipeline.max_in_flight_candidates = parse_usize_override(
                "SYNTH_OPTIMIZERS_GEPA_MAX_IN_FLIGHT_CANDIDATES",
                &max_in_flight,
            )?;
        }
        if let Some(propose_workers) = read_env_override(&["SYNTH_OPTIMIZERS_GEPA_WORKERS_PROPOSE"])
        {
            self.gepa.pipeline.workers.propose =
                parse_usize_override("SYNTH_OPTIMIZERS_GEPA_WORKERS_PROPOSE", &propose_workers)?;
        }
        if let Some(rollout_workers) = read_env_override(&["SYNTH_OPTIMIZERS_GEPA_WORKERS_ROLLOUT"])
        {
            self.gepa.pipeline.workers.rollout =
                parse_usize_override("SYNTH_OPTIMIZERS_GEPA_WORKERS_ROLLOUT", &rollout_workers)?;
        }
        if let Some(evaluate_workers) =
            read_env_override(&["SYNTH_OPTIMIZERS_GEPA_WORKERS_EVALUATE"])
        {
            self.gepa.pipeline.workers.evaluate =
                parse_usize_override("SYNTH_OPTIMIZERS_GEPA_WORKERS_EVALUATE", &evaluate_workers)?;
        }
        if let Some(rollout_chunk_size) =
            read_env_override(&["SYNTH_OPTIMIZERS_GEPA_ROLLOUT_CHUNK_SIZE"])
        {
            self.gepa.rollout_chunk_size = Some(parse_usize_override(
                "SYNTH_OPTIMIZERS_GEPA_ROLLOUT_CHUNK_SIZE",
                &rollout_chunk_size,
            )?);
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_DISK_BUDGET_ENABLED"]) {
            self.disk_budget.enabled =
                parse_bool_override("SYNTH_OPTIMIZERS_DISK_BUDGET_ENABLED", &raw)?;
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_DISK_BUDGET_SOFT_LIMIT_GB"]) {
            self.disk_budget.soft_limit_gb =
                parse_f64_override("SYNTH_OPTIMIZERS_DISK_BUDGET_SOFT_LIMIT_GB", &raw)?;
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_DISK_BUDGET_HARD_LIMIT_GB"]) {
            self.disk_budget.hard_limit_gb =
                parse_f64_override("SYNTH_OPTIMIZERS_DISK_BUDGET_HARD_LIMIT_GB", &raw)?;
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_DISK_BUDGET_PATH"]) {
            self.disk_budget.path = Some(PathBuf::from(raw));
        }
        Ok(())
    }

    fn resolve_relative_paths(&mut self, base_dir: &Path) {
        self.run.output_dir = absolutize(base_dir, &self.run.output_dir);
        if let Some(cwd) = &self.container.cwd {
            self.container.cwd = Some(absolutize(base_dir, cwd));
        }
        if let Some(path) = &self.cache.path {
            self.cache.path = Some(absolutize(base_dir, path));
        }
        resolve_command_path_args(base_dir, &mut self.proposer.command);
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
                "container.url is required".to_string(),
            ));
        }
        if self.dataset.train_seeds.is_empty() {
            return Err(OptimizerError::Config(
                "dataset.train_seeds must contain at least one seed".to_string(),
            ));
        }
        if self.dataset.heldout_seeds.is_empty() {
            return Err(OptimizerError::Config(
                "dataset.heldout_seeds must contain at least one seed".to_string(),
            ));
        }
        if self.candidate.target_modules.is_empty() {
            return Err(OptimizerError::Config(
                "candidate.target_modules must contain at least one module id".to_string(),
            ));
        }
        if self.gepa.minibatch_size == 0 {
            return Err(OptimizerError::Config(
                "gepa.minibatch_size must be positive".to_string(),
            ));
        }
        if self.gepa.max_total_rollouts == 0 {
            return Err(OptimizerError::Config(
                "gepa.max_total_rollouts must be positive".to_string(),
            ));
        }
        if self.gepa.max_train_rollouts == Some(0) {
            return Err(OptimizerError::Config(
                "gepa.max_train_rollouts must be positive when set".to_string(),
            ));
        }
        if self.gepa.max_heldout_rollouts == Some(0) {
            return Err(OptimizerError::Config(
                "gepa.max_heldout_rollouts must be positive when set".to_string(),
            ));
        }
        let rollout_submission_mode = self
            .gepa
            .rollout_submission_mode
            .trim()
            .to_ascii_lowercase();
        if !matches!(rollout_submission_mode.as_str(), "sync" | "async") {
            return Err(OptimizerError::Config(format!(
                "gepa.rollout_submission_mode must be sync or async, got {:?}",
                self.gepa.rollout_submission_mode
            )));
        }
        if self.gepa.rollout_poll_interval_ms == 0 {
            return Err(OptimizerError::Config(
                "gepa.rollout_poll_interval_ms must be positive".to_string(),
            ));
        }
        if self.gepa.rollout_async_timeout_seconds == 0 {
            return Err(OptimizerError::Config(
                "gepa.rollout_async_timeout_seconds must be positive".to_string(),
            ));
        }
        let frontier_type = self
            .gepa
            .frontier_type
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_");
        if !matches!(
            frontier_type.as_str(),
            "per_example" | "per_objective" | "per_example_objective"
        ) {
            return Err(OptimizerError::Config(format!(
                "gepa.frontier_type must be per_example, per_objective, or per_example_objective, got {:?}",
                self.gepa.frontier_type
            )));
        }
        if self
            .gepa
            .selection_objective
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(OptimizerError::Config(
                "gepa.selection_objective must be non-empty when set".to_string(),
            ));
        }
        for objective in &self.gepa.objective_keys {
            if objective.trim().is_empty() {
                return Err(OptimizerError::Config(
                    "gepa.objective_keys entries must be non-empty".to_string(),
                ));
            }
        }
        for (objective, direction) in &self.gepa.objective_directions {
            if objective.trim().is_empty() {
                return Err(OptimizerError::Config(
                    "gepa.objective_directions keys must be non-empty".to_string(),
                ));
            }
            validate_gepa_objective_direction("gepa.objective_directions", direction)?;
        }
        if !self.gepa.minibatch_accept_margin.is_finite() || self.gepa.minibatch_accept_margin < 0.0
        {
            return Err(OptimizerError::Config(
                "gepa.minibatch_accept_margin must be finite and non-negative".to_string(),
            ));
        }
        validate_gepa_acceptance_criterion(&self.gepa.acceptance_criterion)?;
        validate_gepa_objective_acceptance_config(&self.gepa.objective_acceptance)?;
        validate_gepa_candidate_selector_config(&self.gepa.candidate_selector)?;
        validate_gepa_batch_sampler_config(&self.gepa.batch_sampler)?;
        validate_gepa_pipeline_config(&self.gepa.pipeline)?;
        if !self.gepa.max_cost_usd.is_finite() || self.gepa.max_cost_usd < 0.0 {
            return Err(OptimizerError::Config(
                "gepa.max_cost_usd must be finite and non-negative".to_string(),
            ));
        }
        validate_positive_option("gepa.max_time_seconds", self.gepa.max_time_seconds)?;
        validate_positive_option("gepa.max_prompt_tokens", self.gepa.max_prompt_tokens)?;
        validate_positive_option(
            "gepa.max_completion_tokens",
            self.gepa.max_completion_tokens,
        )?;
        validate_positive_option("gepa.max_total_tokens", self.gepa.max_total_tokens)?;
        validate_positive_f64_option(
            "gepa.proposer_estimated_cost_usd",
            self.gepa.proposer_estimated_cost_usd,
        )?;
        validate_positive_f64_option(
            "gepa.rollout_estimated_cost_usd",
            self.gepa.rollout_estimated_cost_usd,
        )?;
        validate_positive_option(
            "gepa.proposer_estimated_prompt_tokens",
            self.gepa.proposer_estimated_prompt_tokens,
        )?;
        validate_positive_option(
            "gepa.proposer_estimated_completion_tokens",
            self.gepa.proposer_estimated_completion_tokens,
        )?;
        validate_positive_option(
            "gepa.proposer_estimated_total_tokens",
            self.gepa.proposer_estimated_total_tokens,
        )?;
        validate_positive_option(
            "gepa.rollout_estimated_prompt_tokens",
            self.gepa.rollout_estimated_prompt_tokens,
        )?;
        validate_positive_option(
            "gepa.rollout_estimated_completion_tokens",
            self.gepa.rollout_estimated_completion_tokens,
        )?;
        validate_positive_option(
            "gepa.rollout_estimated_total_tokens",
            self.gepa.rollout_estimated_total_tokens,
        )?;
        validate_positive_option(
            "gepa.rollout_estimated_wall_seconds",
            self.gepa.rollout_estimated_wall_seconds,
        )?;
        if self.gepa.rollout_chunk_size == Some(0) {
            return Err(OptimizerError::Config(
                "gepa.rollout_chunk_size must be positive".to_string(),
            ));
        }
        validate_gepa_limit_config(&self.gepa)?;
        self.disk_budget.validate()?;
        let backend = self.proposer.backend.trim();
        match backend {
            "codex_app_server" => {}
            "local_process_json" => {
                return Err(OptimizerError::Config(
                    "unsupported proposer.backend \"local_process_json\"; GEPA proposer work must use codex_app_server workspace-backed proposing".to_string(),
                ));
            }
            _ => {
                return Err(OptimizerError::Config(format!(
                    "unsupported proposer.backend {backend:?}; expected codex_app_server"
                )));
            }
        }
        if self.proposer.execution_mode.trim() != "local_process" {
            return Err(OptimizerError::Config(format!(
                "unsupported proposer.execution_mode {:?}; expected local_process",
                self.proposer.execution_mode
            )));
        }
        match self.proposer.auth_mode.trim() {
            "auto" | "host" | "api_key" => {}
            mode => {
                return Err(OptimizerError::Config(format!(
                    "unsupported proposer.auth_mode {mode:?}; expected auto, host, or api_key"
                )));
            }
        }
        if self.proposer.auth_mode.trim() == "api_key" && self.proposer.copy_host_auth {
            return Err(OptimizerError::Config(
                "proposer.auth_mode = \"api_key\" cannot be combined with proposer.copy_host_auth = true".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    #[serde(default = "default_run_id")]
    pub run_id: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default)]
    pub seed: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            run_id: default_run_id(),
            output_dir: default_output_dir(),
            seed: 0,
        }
    }
}

fn default_run_id() -> String {
    format!("gepa_{}", uuid::Uuid::new_v4().simple())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerConfig {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_startup_timeout_seconds")]
    pub startup_timeout_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    #[serde(default = "default_train_split")]
    pub train_split: String,
    #[serde(default = "default_heldout_split")]
    pub heldout_split: String,
    #[serde(default)]
    pub train_seeds: Vec<i64>,
    #[serde(default)]
    pub heldout_seeds: Vec<i64>,
    #[serde(default)]
    pub filters: Map<String, Value>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            train_split: default_train_split(),
            heldout_split: default_heldout_split(),
            train_seeds: Vec::new(),
            heldout_seeds: Vec::new(),
            filters: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateConfig {
    #[serde(default)]
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub candidate_id_prefix: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    #[serde(default = "default_policy_provider")]
    pub provider: String,
    #[serde(default = "default_policy_model")]
    pub model: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub extra: Map<String, Value>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            provider: default_policy_provider(),
            model: default_policy_model(),
            base_url: None,
            api_key_env: None,
            extra: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposerConfig {
    #[serde(default = "default_proposer_backend")]
    pub backend: String,
    #[serde(default = "default_execution_mode")]
    pub execution_mode: String,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub sandbox_mode: Option<String>,
    #[serde(default)]
    pub approval_policy: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default = "default_proposer_auth_mode")]
    pub auth_mode: String,
    #[serde(default)]
    pub copy_host_auth: bool,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default)]
    pub model: Option<String>,
}

impl Default for ProposerConfig {
    fn default() -> Self {
        Self {
            backend: default_proposer_backend(),
            execution_mode: default_execution_mode(),
            command: Vec::new(),
            sandbox_mode: None,
            approval_policy: None,
            reasoning_effort: None,
            auth_mode: default_proposer_auth_mode(),
            copy_host_auth: false,
            api_key_env: None,
            timeout_seconds: default_timeout_seconds(),
            model: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaCandidateSelectorConfig {
    #[serde(default = "default_candidate_selector_name")]
    pub name: String,
    #[serde(default)]
    pub epsilon: Option<f64>,
    #[serde(default)]
    pub k: Option<usize>,
}

impl Default for GepaCandidateSelectorConfig {
    fn default() -> Self {
        Self {
            name: default_candidate_selector_name(),
            epsilon: None,
            k: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaBatchSamplerConfig {
    #[serde(default = "default_batch_sampler_name")]
    pub name: String,
    #[serde(default)]
    pub epoch_width: Option<usize>,
    #[serde(default)]
    pub field: Option<String>,
}

impl Default for GepaBatchSamplerConfig {
    fn default() -> Self {
        Self {
            name: default_batch_sampler_name(),
            epoch_width: None,
            field: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaSeedPoolsConfig {
    #[serde(default)]
    pub pareto_eval: Vec<i64>,
    #[serde(default)]
    pub minibatch: Vec<i64>,
    #[serde(default)]
    pub reflection: Vec<i64>,
    #[serde(default)]
    pub validation: Vec<i64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaObjectiveAcceptanceConfig {
    #[serde(default)]
    pub min_objective_delta: Option<f64>,
    #[serde(default)]
    pub objective_regression_tolerance: Option<f64>,
    #[serde(default)]
    pub protected_objectives: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaConfig {
    #[serde(default = "default_max_generations")]
    pub max_generations: usize,
    #[serde(default = "default_proposals_per_generation")]
    pub proposals_per_generation: usize,
    #[serde(default = "default_minibatch_size")]
    pub minibatch_size: usize,
    #[serde(default)]
    pub minibatch_accept_margin: f64,
    #[serde(default = "default_max_total_rollouts")]
    pub max_total_rollouts: usize,
    #[serde(default)]
    pub max_train_rollouts: Option<usize>,
    #[serde(default)]
    pub max_heldout_rollouts: Option<usize>,
    #[serde(default = "default_rollout_submission_mode")]
    pub rollout_submission_mode: String,
    #[serde(default = "default_rollout_poll_interval_ms")]
    pub rollout_poll_interval_ms: u64,
    #[serde(default = "default_rollout_async_timeout_seconds")]
    pub rollout_async_timeout_seconds: u64,
    #[serde(default)]
    pub rollout_chunk_size: Option<usize>,
    #[serde(default = "default_frontier_type")]
    pub frontier_type: String,
    #[serde(default)]
    pub selection_objective: Option<String>,
    #[serde(default)]
    pub objective_keys: Vec<String>,
    #[serde(default)]
    pub objective_directions: BTreeMap<String, String>,
    #[serde(default = "default_acceptance_criterion")]
    pub acceptance_criterion: String,
    #[serde(default = "default_objective_acceptance_config")]
    pub objective_acceptance: GepaObjectiveAcceptanceConfig,
    #[serde(default = "default_candidate_selector_config")]
    pub candidate_selector: GepaCandidateSelectorConfig,
    #[serde(default = "default_batch_sampler_config")]
    pub batch_sampler: GepaBatchSamplerConfig,
    #[serde(default = "default_seed_pools_config")]
    pub seed_pools: GepaSeedPoolsConfig,
    #[serde(default = "default_gepa_pipeline_config")]
    pub pipeline: GepaPipelineConfig,
    #[serde(default)]
    pub max_cost_usd: f64,
    #[serde(default)]
    pub max_time_seconds: Option<u64>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    #[serde(default)]
    pub proposer_estimated_cost_usd: Option<f64>,
    #[serde(default)]
    pub proposer_estimated_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub proposer_estimated_completion_tokens: Option<u64>,
    #[serde(default)]
    pub proposer_estimated_total_tokens: Option<u64>,
    #[serde(default)]
    pub rollout_estimated_cost_usd: Option<f64>,
    #[serde(default)]
    pub rollout_estimated_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub rollout_estimated_completion_tokens: Option<u64>,
    #[serde(default)]
    pub rollout_estimated_total_tokens: Option<u64>,
    #[serde(default)]
    pub rollout_estimated_wall_seconds: Option<u64>,
}

impl Default for GepaConfig {
    fn default() -> Self {
        Self {
            max_generations: default_max_generations(),
            proposals_per_generation: default_proposals_per_generation(),
            minibatch_size: default_minibatch_size(),
            minibatch_accept_margin: 0.0,
            max_total_rollouts: default_max_total_rollouts(),
            max_train_rollouts: None,
            max_heldout_rollouts: None,
            rollout_submission_mode: default_rollout_submission_mode(),
            rollout_poll_interval_ms: default_rollout_poll_interval_ms(),
            rollout_async_timeout_seconds: default_rollout_async_timeout_seconds(),
            rollout_chunk_size: None,
            frontier_type: default_frontier_type(),
            selection_objective: None,
            objective_keys: Vec::new(),
            objective_directions: BTreeMap::new(),
            acceptance_criterion: default_acceptance_criterion(),
            objective_acceptance: default_objective_acceptance_config(),
            candidate_selector: default_candidate_selector_config(),
            batch_sampler: default_batch_sampler_config(),
            seed_pools: default_seed_pools_config(),
            pipeline: default_gepa_pipeline_config(),
            max_cost_usd: 0.0,
            max_time_seconds: None,
            max_prompt_tokens: None,
            max_completion_tokens: None,
            max_total_tokens: None,
            proposer_estimated_cost_usd: None,
            proposer_estimated_prompt_tokens: None,
            proposer_estimated_completion_tokens: None,
            proposer_estimated_total_tokens: None,
            rollout_estimated_cost_usd: None,
            rollout_estimated_prompt_tokens: None,
            rollout_estimated_completion_tokens: None,
            rollout_estimated_total_tokens: None,
            rollout_estimated_wall_seconds: None,
        }
    }
}

impl GepaConfig {
    pub fn split_rollout_budgets_enabled(&self) -> bool {
        self.max_train_rollouts.is_some() || self.max_heldout_rollouts.is_some()
    }

    pub fn train_rollout_limit(&self) -> usize {
        self.max_train_rollouts.unwrap_or(self.max_total_rollouts)
    }

    pub fn heldout_rollout_limit(&self) -> usize {
        self.max_heldout_rollouts.unwrap_or(self.max_total_rollouts)
    }

    pub fn effective_max_total_rollouts(&self) -> usize {
        if self.split_rollout_budgets_enabled() {
            self.train_rollout_limit()
                .saturating_add(self.heldout_rollout_limit())
        } else {
            self.max_total_rollouts
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GepaPipelineMode {
    #[default]
    SyncSerial,
    AsyncPipelined,
}

impl GepaPipelineMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SyncSerial => "sync_serial",
            Self::AsyncPipelined => "async_pipelined",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GepaStalenessPolicy {
    #[default]
    Full,
    Guarded,
    Reflective,
}

impl GepaStalenessPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Guarded => "guarded",
            Self::Reflective => "reflective",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaPipelineConfig {
    #[serde(default)]
    pub mode: GepaPipelineMode,
    #[serde(default)]
    pub staleness_policy: GepaStalenessPolicy,
    #[serde(default = "default_pipeline_max_in_flight_candidates")]
    pub max_in_flight_candidates: usize,
    #[serde(default)]
    pub workers: GepaPipelineWorkers,
    #[serde(default)]
    pub adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyConfig,
}

impl Default for GepaPipelineConfig {
    fn default() -> Self {
        Self {
            mode: GepaPipelineMode::SyncSerial,
            staleness_policy: GepaStalenessPolicy::Full,
            max_in_flight_candidates: default_pipeline_max_in_flight_candidates(),
            workers: GepaPipelineWorkers::default(),
            adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaAdaptiveRolloutConcurrencyConfig {
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_enabled")]
    pub enabled: bool,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_initial")]
    pub initial: usize,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_min")]
    pub min: usize,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_max")]
    pub max: usize,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_increase_step")]
    pub increase_step: usize,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_decrease_step")]
    pub decrease_step: usize,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_increase_after_successes")]
    pub increase_after_successes: usize,
    #[serde(default = "default_gepa_adaptive_rollout_concurrency_overload_status_codes")]
    pub overload_status_codes: Vec<u16>,
}

impl Default for GepaAdaptiveRolloutConcurrencyConfig {
    fn default() -> Self {
        Self {
            enabled: default_gepa_adaptive_rollout_concurrency_enabled(),
            initial: default_gepa_adaptive_rollout_concurrency_initial(),
            min: default_gepa_adaptive_rollout_concurrency_min(),
            max: default_gepa_adaptive_rollout_concurrency_max(),
            increase_step: default_gepa_adaptive_rollout_concurrency_increase_step(),
            decrease_step: default_gepa_adaptive_rollout_concurrency_decrease_step(),
            increase_after_successes:
                default_gepa_adaptive_rollout_concurrency_increase_after_successes(),
            overload_status_codes: default_gepa_adaptive_rollout_concurrency_overload_status_codes(
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaPipelineWorkers {
    #[serde(default = "default_pipeline_proposal_workers")]
    pub propose: usize,
    #[serde(default = "default_pipeline_rollout_workers")]
    pub rollout: usize,
    #[serde(default = "default_pipeline_evaluate_workers")]
    pub evaluate: usize,
}

impl Default for GepaPipelineWorkers {
    fn default() -> Self {
        Self {
            propose: default_pipeline_proposal_workers(),
            rollout: default_pipeline_rollout_workers(),
            evaluate: default_pipeline_evaluate_workers(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    #[serde(default = "default_cache_mode")]
    pub mode: CacheConfigMode,
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub namespace: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            mode: default_cache_mode(),
            path: None,
            namespace: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheConfigMode {
    Off,
    Readwrite,
    Readonly,
}

fn absolutize(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let base_dir = if base_dir.is_absolute() {
            base_dir.to_path_buf()
        } else {
            env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(base_dir)
        };
        base_dir.join(path)
    }
}

fn resolve_command_path_args(base_dir: &Path, command: &mut [String]) {
    for arg in command.iter_mut() {
        let path = Path::new(arg);
        if path.is_absolute() || !looks_like_relative_path(path) {
            continue;
        }
        if let Some(resolved) = find_existing_relative_path(base_dir, path) {
            *arg = resolved.display().to_string();
        }
    }
}

fn looks_like_relative_path(path: &Path) -> bool {
    path.components().count() > 1
}

fn find_existing_relative_path(base_dir: &Path, path: &Path) -> Option<PathBuf> {
    let absolute_base = if base_dir.is_absolute() {
        base_dir.to_path_buf()
    } else {
        env::current_dir().ok()?.join(base_dir)
    };
    for ancestor in absolute_base.ancestors() {
        let candidate = ancestor.join(path);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn validate_positive_option(name: &str, value: Option<u64>) -> Result<()> {
    if value == Some(0) {
        return Err(OptimizerError::Config(format!("{name} must be positive")));
    }
    Ok(())
}

fn validate_positive_f64_option(name: &str, value: Option<f64>) -> Result<()> {
    if value.is_some_and(|item| !item.is_finite() || item <= 0.0) {
        return Err(OptimizerError::Config(format!("{name} must be positive")));
    }
    Ok(())
}

fn validate_gepa_candidate_selector_config(config: &GepaCandidateSelectorConfig) -> Result<()> {
    let strategy = config.name.trim().to_ascii_lowercase().replace('-', "_");
    if !matches!(
        strategy.as_str(),
        "pareto_weighted"
            | "pareto"
            | "uniform_pareto"
            | "random"
            | "current_best"
            | "top_k_pareto"
            | "epsilon_greedy"
    ) {
        return Err(OptimizerError::Config(format!(
            "gepa.candidate_selector.name must be pareto_weighted, pareto, uniform_pareto, random, current_best, top_k_pareto, or epsilon_greedy; got {:?}",
            config.name
        )));
    }
    if config
        .epsilon
        .is_some_and(|epsilon| !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon))
    {
        return Err(OptimizerError::Config(
            "gepa.candidate_selector.epsilon must be finite and between 0.0 and 1.0".to_string(),
        ));
    }
    if config.k == Some(0) {
        return Err(OptimizerError::Config(
            "gepa.candidate_selector.k must be positive when set".to_string(),
        ));
    }
    Ok(())
}

fn validate_gepa_batch_sampler_config(config: &GepaBatchSamplerConfig) -> Result<()> {
    let strategy = config.name.trim().to_ascii_lowercase().replace('-', "_");
    if !matches!(
        strategy.as_str(),
        "seeded_shuffle"
            | "epoch_shuffled"
            | "ordered_epoch"
            | "sequential_epoch"
            | "stratified"
            | "stratified_by_field"
    ) {
        return Err(OptimizerError::Config(format!(
            "gepa.batch_sampler.name must be seeded_shuffle, epoch_shuffled, ordered_epoch, or stratified; got {:?}",
            config.name
        )));
    }
    if config.epoch_width == Some(0) {
        return Err(OptimizerError::Config(
            "gepa.batch_sampler.epoch_width must be positive when set".to_string(),
        ));
    }
    if config
        .field
        .as_deref()
        .is_some_and(|field| field.trim().is_empty())
    {
        return Err(OptimizerError::Config(
            "gepa.batch_sampler.field must be non-empty when set".to_string(),
        ));
    }
    Ok(())
}

fn validate_gepa_acceptance_criterion(criterion: &str) -> Result<()> {
    let criterion = criterion.trim().to_ascii_lowercase().replace('-', "_");
    if matches!(
        criterion.as_str(),
        "primary_improvement"
            | "improvement_or_equal"
            | "primary_or_objective"
            | "any_objective_improved"
            | "protected_objective_guard"
    ) {
        Ok(())
    } else {
        Err(OptimizerError::Config(format!(
            "gepa.acceptance_criterion must be primary_improvement, improvement_or_equal, primary_or_objective, any_objective_improved, or protected_objective_guard; got {:?}",
            criterion
        )))
    }
}

fn validate_gepa_objective_acceptance_config(config: &GepaObjectiveAcceptanceConfig) -> Result<()> {
    if config
        .min_objective_delta
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(OptimizerError::Config(
            "gepa.objective_acceptance.min_objective_delta must be finite and non-negative"
                .to_string(),
        ));
    }
    if config
        .objective_regression_tolerance
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(OptimizerError::Config(
            "gepa.objective_acceptance.objective_regression_tolerance must be finite and non-negative"
                .to_string(),
        ));
    }
    if config
        .protected_objectives
        .iter()
        .any(|objective| objective.trim().is_empty())
    {
        return Err(OptimizerError::Config(
            "gepa.objective_acceptance.protected_objectives entries must be non-empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_gepa_objective_direction(name: &str, direction: &str) -> Result<()> {
    match direction.trim().to_ascii_lowercase().as_str() {
        "max" | "maximize" | "higher" | "higher_is_better" | "up" | "min" | "minimize"
        | "lower" | "lower_is_better" | "down" => Ok(()),
        _ => Err(OptimizerError::Config(format!(
            "{name} values must be maximize/higher_is_better or minimize/lower_is_better; got {direction:?}"
        ))),
    }
}

fn read_env_override(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn parse_cache_mode_override(raw_mode: &str) -> Result<CacheConfigMode> {
    match raw_mode.trim().to_ascii_lowercase().as_str() {
        "off" => Ok(CacheConfigMode::Off),
        "readwrite" => Ok(CacheConfigMode::Readwrite),
        "readonly" => Ok(CacheConfigMode::Readonly),
        _ => Err(OptimizerError::Config(format!(
            "unknown cache mode override: {raw_mode}"
        ))),
    }
}

fn parse_u64_override(name: &str, raw_value: &str) -> Result<u64> {
    raw_value.trim().parse::<u64>().map_err(|source| {
        OptimizerError::Config(format!("invalid {name} override {raw_value:?}: {source}"))
    })
}

fn parse_usize_override(name: &str, raw_value: &str) -> Result<usize> {
    raw_value.trim().parse::<usize>().map_err(|source| {
        OptimizerError::Config(format!("invalid {name} override {raw_value:?}: {source}"))
    })
}

fn parse_f64_override(name: &str, raw_value: &str) -> Result<f64> {
    raw_value.trim().parse::<f64>().map_err(|source| {
        OptimizerError::Config(format!("invalid {name} override {raw_value:?}: {source}"))
    })
}

fn parse_bool_override(name: &str, raw_value: &str) -> Result<bool> {
    match raw_value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "y" => Ok(true),
        "0" | "false" | "no" | "off" | "n" | "" => Ok(false),
        other => Err(OptimizerError::Config(format!(
            "invalid {name} override {other:?}: expected one of 0/1/true/false/yes/no/on/off"
        ))),
    }
}

fn parse_gepa_pipeline_mode_override(raw_mode: &str) -> Result<GepaPipelineMode> {
    match raw_mode.trim().to_ascii_lowercase().as_str() {
        "sync_serial" | "sync" | "serial" => Ok(GepaPipelineMode::SyncSerial),
        "async_pipelined" | "async" | "pipelined" => Ok(GepaPipelineMode::AsyncPipelined),
        _ => Err(OptimizerError::Config(format!(
            "unknown GEPA pipeline mode override: {raw_mode}"
        ))),
    }
}

fn parse_gepa_staleness_policy_override(raw_policy: &str) -> Result<GepaStalenessPolicy> {
    match raw_policy.trim().to_ascii_lowercase().as_str() {
        "full" | "full_async" => Ok(GepaStalenessPolicy::Full),
        "guarded" => Ok(GepaStalenessPolicy::Guarded),
        "reflective" => Ok(GepaStalenessPolicy::Reflective),
        _ => Err(OptimizerError::Config(format!(
            "unknown GEPA staleness policy override: {raw_policy}"
        ))),
    }
}

fn validate_gepa_pipeline_config(config: &GepaPipelineConfig) -> Result<()> {
    if matches!(config.mode, GepaPipelineMode::AsyncPipelined)
        && !matches!(config.staleness_policy, GepaStalenessPolicy::Full)
    {
        return Err(OptimizerError::Config(format!(
            "gepa.pipeline.staleness_policy = {:?} is reserved for a later async-pipelined phase; use full",
            config.staleness_policy
        )));
    }
    if config.max_in_flight_candidates == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.max_in_flight_candidates must be positive".to_string(),
        ));
    }
    if config.workers.propose == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.workers.propose must be positive".to_string(),
        ));
    }
    if config.workers.rollout == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.workers.rollout must be positive".to_string(),
        ));
    }
    if config.workers.evaluate == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.workers.evaluate must be positive".to_string(),
        ));
    }
    let adaptive = &config.adaptive_rollout_concurrency;
    if adaptive.min == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_rollout_concurrency.min must be positive".to_string(),
        ));
    }
    if adaptive.max < adaptive.min {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_rollout_concurrency.max must be >= min".to_string(),
        ));
    }
    if adaptive.initial < adaptive.min || adaptive.initial > adaptive.max {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_rollout_concurrency.initial must be between min and max"
                .to_string(),
        ));
    }
    if adaptive.increase_step == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_rollout_concurrency.increase_step must be positive".to_string(),
        ));
    }
    if adaptive.decrease_step == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_rollout_concurrency.decrease_step must be positive".to_string(),
        ));
    }
    if adaptive.increase_after_successes == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_rollout_concurrency.increase_after_successes must be positive"
                .to_string(),
        ));
    }
    Ok(())
}
