use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agent_runtime::{validate_execution_mode_compat, ExecutionSubstrate};
use crate::configured_limits::validate_gepa_limit_config;
use crate::disk_budget::DiskBudgetConfig;
use crate::error::{OptimizerError, Result};

fn default_output_dir() -> PathBuf {
    PathBuf::from("runs")
}

fn default_startup_timeout_seconds() -> u64 {
    30
}

fn default_container_pool_api_key_env() -> String {
    "SYNTH_API_KEY".to_string()
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

fn default_policy_enabled() -> bool {
    true
}

fn default_policy_model() -> String {
    "gpt-4.1-nano".to_string()
}

fn default_policy_api_family() -> String {
    "chat_completions".to_string()
}

fn default_policy_disable_reasoning() -> String {
    "auto".to_string()
}

fn default_policy_tool_call_style() -> String {
    "none".to_string()
}

fn default_policy_proxy_mode() -> String {
    "proxy_only".to_string()
}

fn default_policy_credential_mode() -> String {
    "byok".to_string()
}

fn default_proposer_backend() -> String {
    "codex_app_server".to_string()
}

fn default_execution_mode() -> String {
    "local_process".to_string()
}

fn default_runtime_substrate() -> ExecutionSubstrate {
    ExecutionSubstrate::Local
}

fn default_docker_workspace_mount_path() -> String {
    crate::agent_runtime::limits::DOCKER_WORKSPACE_MOUNT_PATH.to_string()
}

fn default_docker_network() -> String {
    crate::agent_runtime::limits::DOCKER_NETWORK.to_string()
}

fn default_daytona_api_key_env() -> String {
    "DAYTONA_API_KEY".to_string()
}

fn default_daytona_api_url() -> String {
    "https://app.daytona.io/api".to_string()
}

fn default_daytona_language() -> String {
    "python".to_string()
}

fn default_daytona_sandbox_name_prefix() -> String {
    "synth-optimizer-proposer".to_string()
}

fn default_daytona_remote_workspace_dir() -> String {
    crate::agent_runtime::limits::DOCKER_WORKSPACE_MOUNT_PATH.to_string()
}

fn default_daytona_auto_stop_interval_minutes() -> u64 {
    30
}

fn default_daytona_startup_timeout_seconds() -> u64 {
    120
}

fn default_daytona_poll_interval_ms() -> u64 {
    250
}

fn default_daytona_public() -> bool {
    true
}

fn default_proposer_auth_mode() -> String {
    "auto".to_string()
}

fn default_timeout_seconds() -> u64 {
    300
}

fn default_message_stall_timeout_seconds() -> u64 {
    120
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
    "async".to_string()
}

fn default_rollout_poll_interval_ms() -> u64 {
    250
}

fn default_rollout_async_timeout_seconds() -> u64 {
    600
}

fn default_rollout_failure_rate_tolerance() -> f64 {
    0.25
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

fn default_task_pools_config() -> GepaTaskPoolsConfig {
    GepaTaskPoolsConfig::default()
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

fn default_pipeline_staleness_delta_max() -> u64 {
    2
}

fn default_gepa_speculative_completion_enabled() -> bool {
    false
}

fn default_gepa_speculative_completion_alpha() -> f64 {
    0.25
}

fn default_gepa_adaptive_stage_workers_enabled() -> bool {
    false
}

fn default_gepa_adaptive_stage_workers_min() -> usize {
    1
}

fn default_gepa_adaptive_stage_workers_max() -> usize {
    128
}

fn default_gepa_adaptive_stage_workers_backlog_threshold() -> usize {
    2
}

fn default_gepa_adaptive_stage_workers_stale_gap_threshold() -> u64 {
    2
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthOptimizerConfig {
    #[serde(default)]
    pub run: RunConfig,
    #[serde(default)]
    pub container: ContainerConfig,
    #[serde(default)]
    pub taskset: TasksetConfig,
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
    /// Optional jesterky trace-annotate workflow before each GEPA proposer turn.
    /// Default disabled (Arm A). When enabled, export rollouts → annotate →
    /// materialize `state/jesterky_*` into the proposer workspace.
    #[serde(default)]
    pub jesterky_workflow: JesterkyWorkflowConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub disk_budget: DiskBudgetConfig,
}

/// Per-run toggle for jesterky trace workflows inside GEPA.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct JesterkyWorkflowConfig {
    pub enabled: bool,
    /// Path to a jesterky workflow spec (absolute or relative).
    pub spec: String,
    /// Binary name or absolute path. Falls back to `STACK_JESTERKY_COMMAND` then `jesterky`.
    pub command: String,
    /// jesterky `--actor` (fake|codex).
    pub actor: String,
    /// Optional annotate actor model (`--model`).
    pub model: Option<String>,
    pub concurrency: usize,
    pub timeout_seconds: u64,
    /// When true, annotate/export failures fail the proposer turn.
    pub fail_closed: bool,
    /// When true, annotate every exported rollout instead of the default cap of 6.
    #[serde(default)]
    pub bulk: bool,
}

impl Default for JesterkyWorkflowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            spec: default_jesterky_workflow_spec(),
            command: default_jesterky_workflow_command(),
            actor: default_jesterky_workflow_actor(),
            model: None,
            concurrency: default_jesterky_workflow_concurrency(),
            timeout_seconds: default_jesterky_workflow_timeout_seconds(),
            fail_closed: true,
            bulk: false,
        }
    }
}

fn default_jesterky_workflow_spec() -> String {
    "examples/gepa_trace_annotate.json".to_string()
}

fn default_jesterky_workflow_command() -> String {
    "jesterky".to_string()
}

fn default_jesterky_workflow_actor() -> String {
    "codex".to_string()
}

fn default_jesterky_workflow_concurrency() -> usize {
    4
}

fn default_jesterky_workflow_timeout_seconds() -> u64 {
    600
}

impl SynthOptimizerConfig {
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
        let mut config: Self = toml::from_str(&text)?;
        config.apply_env_overrides()?;
        config.resolve_relative_paths(path.parent().unwrap_or_else(|| Path::new(".")));
        config.resolve_runtime_targets()?;
        config.validate()?;
        Ok(config)
    }

    fn apply_env_overrides(&mut self) -> Result<()> {
        if let Some(run_id) =
            read_env_override(&["SYNTH_OPTIMIZERS_RUN_ID", "GEPA_PLATFORM_RUN_ID"])
        {
            self.run.run_id = run_id;
        }
        if let Some(fixture_path) = read_env_override(&["SYNTH_OPTIMIZERS_FIXTURE_PATH"]) {
            let trimmed = fixture_path.trim();
            self.run.fixture_path = if trimmed.is_empty() {
                None
            } else {
                Some(PathBuf::from(trimmed))
            };
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
        if let Some(execution_mode) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_EXECUTION_MODE",
            "GEPA_PLATFORM_PROPOSER_EXECUTION_MODE",
        ]) {
            self.proposer.execution_mode = execution_mode.trim().to_ascii_lowercase();
        }
        if let Some(model) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_MODEL",
            "GEPA_PLATFORM_PROPOSER_MODEL",
        ]) {
            self.proposer.model = Some(model.trim().to_string());
        }
        if let Some(reasoning_effort) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT",
            "GEPA_PLATFORM_PROPOSER_REASONING_EFFORT",
        ]) {
            self.proposer.reasoning_effort = Some(normalize_enum_value(&reasoning_effort));
        }
        if let Some(service_tier) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_SERVICE_TIER",
            "GEPA_PLATFORM_PROPOSER_SERVICE_TIER",
        ]) {
            self.proposer.service_tier = normalize_proposer_service_tier(&service_tier);
        }
        if let Some(auth_mode) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_AUTH_MODE",
            "GEPA_PLATFORM_PROPOSER_AUTH_MODE",
        ]) {
            self.proposer.auth_mode = proposer_auth_mode_normalized(&auth_mode);
            if proposer_uses_chatgpt_auth(&self.proposer.auth_mode) {
                self.proposer.api_key_env = None;
            }
        }
        if let Some(codex_home) = read_env_override(&[
            "SYNTH_OPTIMIZERS_PROPOSER_CODEX_HOME",
            "GEPA_PLATFORM_PROPOSER_CODEX_HOME",
        ]) {
            self.proposer.codex_home = Some(PathBuf::from(codex_home));
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
        if let Some(raw) =
            read_env_override(&["SYNTH_OPTIMIZERS_GEPA_ROLLOUT_FAILURE_RATE_TOLERANCE"])
        {
            self.gepa.rollout_failure_rate_tolerance =
                parse_f64_override("SYNTH_OPTIMIZERS_GEPA_ROLLOUT_FAILURE_RATE_TOLERANCE", &raw)?;
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
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_EPISODE_PROPOSER_ROUNDS"]) {
            self.gepa.episode.proposer_rounds = Some(parse_usize_override(
                "SYNTH_OPTIMIZERS_EPISODE_PROPOSER_ROUNDS",
                &raw,
            )?);
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_EPISODE_MAX_ROLLOUTS"]) {
            self.gepa.episode.max_rollouts = Some(parse_usize_override(
                "SYNTH_OPTIMIZERS_EPISODE_MAX_ROLLOUTS",
                &raw,
            )?);
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_EPISODE_MAX_WALL_SECONDS"]) {
            self.gepa.episode.max_wall_seconds = Some(parse_u64_override(
                "SYNTH_OPTIMIZERS_EPISODE_MAX_WALL_SECONDS",
                &raw,
            )?);
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_EPISODE_MAX_SPEND_USD"]) {
            self.gepa.episode.max_spend_usd = Some(parse_f64_override(
                "SYNTH_OPTIMIZERS_EPISODE_MAX_SPEND_USD",
                &raw,
            )?);
        }
        if let Some(raw) = read_env_override(&["SYNTH_OPTIMIZERS_EPISODE_SKIP_HELDOUT"]) {
            self.gepa.episode.skip_heldout =
                parse_bool_override("SYNTH_OPTIMIZERS_EPISODE_SKIP_HELDOUT", &raw)?;
        }
        Ok(())
    }

    fn resolve_relative_paths(&mut self, base_dir: &Path) {
        self.run.output_dir = absolutize(base_dir, &self.run.output_dir);
        if let Some(cwd) = &self.container.cwd {
            self.container.cwd = Some(absolutize(base_dir, cwd));
        }
        if let Some(path) = &self.proposer.prompt.best_practices_path {
            self.proposer.prompt.best_practices_path = Some(absolutize(base_dir, path));
        }
        if let Some(path) = &self.cache.path {
            self.cache.path = Some(absolutize(base_dir, path));
        }
        resolve_command_path_args(base_dir, &mut self.proposer.command);
    }

    pub fn resolve_runtime_targets(&mut self) -> Result<()> {
        self.container.resolve_pool_target()
    }

    pub fn validate(&self) -> Result<()> {
        if self.run.run_id.trim().is_empty() {
            return Err(OptimizerError::Config("run.run_id is required".to_string()));
        }
        let has_container_url = !self
            .container
            .url
            .as_deref()
            .unwrap_or_default()
            .trim()
            .is_empty();
        let has_pool_target = self
            .container
            .pool
            .as_ref()
            .is_some_and(|pool| !pool.pool_id.trim().is_empty());
        if !has_container_url && !has_pool_target {
            return Err(OptimizerError::Config(
                "container.url or container.pool.pool_id is required".to_string(),
            ));
        }
        if self.taskset.train_ids.is_empty() {
            return Err(OptimizerError::Config(
                "taskset.train_ids must contain at least one task id".to_string(),
            ));
        }
        if self.taskset.heldout_ids.is_empty() {
            return Err(OptimizerError::Config(
                "taskset.heldout_ids must contain at least one task id".to_string(),
            ));
        }
        for module_id in &self.candidate.target_modules {
            if module_id.trim().is_empty() {
                return Err(OptimizerError::Config(
                    "candidate.target_modules entries must be non-empty".to_string(),
                ));
            }
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
        if !(0.0..=1.0).contains(&self.gepa.rollout_failure_rate_tolerance) {
            return Err(OptimizerError::Config(
                "gepa.rollout_failure_rate_tolerance must be between 0.0 and 1.0".to_string(),
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
        validate_policy_config(&self.policy)?;
        if self.policy.enabled {
            if self.candidate.target_modules.is_empty() {
                return Err(OptimizerError::Config(
                    "GEPA policy runs require candidate.target_modules so prompt delivery assertions can be bound to candidate fields".to_string(),
                ));
            }
            if normalize_enum_value(&self.policy.proxy_mode) == "allow_direct" {
                return Err(OptimizerError::Config(
                    "policy.proxy_mode = \"allow_direct\" is forbidden for GEPA policy runs; use proxy_only or assert_proxy so candidate prompts are applied through the inference proxy".to_string(),
                ));
            }
        }
        validate_gepa_objective_acceptance_config(&self.gepa.objective_acceptance)?;
        validate_gepa_candidate_selector_config(&self.gepa.candidate_selector)?;
        validate_gepa_batch_sampler_config(&self.gepa.batch_sampler)?;
        validate_gepa_task_pools_config(&self.gepa.task_pools)?;
        validate_task_pools_against_taskset(
            &self.gepa.task_pools,
            &self.taskset.train_ids,
            &self.taskset.heldout_ids,
        )?;
        validate_gepa_pipeline_config(&self.gepa.pipeline)?;
        validate_gepa_episode_config(&self.gepa.episode)?;
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
        validate_jesterky_workflow_config(&self.jesterky_workflow)?;
        validate_gepa_operator_config(&self.gepa.operator)?;
        self.disk_budget.validate()?;
        let backend = self.proposer.backend.trim();
        match backend {
            "codex_app_server" | "deepseek_chat" | "chat_completions" => {}
            "local_process_json" => {
                return Err(OptimizerError::Config(
                    "unsupported proposer.backend \"local_process_json\"; GEPA proposer work must use codex_app_server workspace-backed proposing".to_string(),
                ));
            }
            _ => {
                return Err(OptimizerError::Config(format!(
                    "unsupported proposer.backend {backend:?}; expected codex_app_server, chat_completions, or deepseek_chat"
                )));
            }
        }
        validate_execution_mode_compat(&self.proposer.execution_mode)?;
        validate_proposer_runtime_substrate_config(&self.proposer)?;
        if matches!(backend, "deepseek_chat" | "chat_completions") {
            validate_chat_completions_proposer_config(&self.proposer)?;
        }
        validate_openrouter_proposer_config(&self.proposer)?;
        let proposer_api_family = normalize_enum_value(&self.proposer.api_family);
        if !matches!(
            proposer_api_family.as_str(),
            "chat_completions" | "responses"
        ) {
            return Err(OptimizerError::Config(format!(
                "proposer.api_family must be chat_completions or responses; got {:?}",
                self.proposer.api_family
            )));
        }
        validate_proposer_reasoning_effort(&self.proposer)?;
        validate_proposer_service_tier(&self.proposer)?;
        validate_proposer_auth_config(&self.proposer)?;
        validate_proposer_prompt_config(&self.proposer.prompt)?;
        validate_mcp_agent_config("proposer.mcp", &self.proposer.mcp)?;
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
    #[serde(default)]
    pub fixture_path: Option<PathBuf>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            run_id: default_run_id(),
            output_dir: default_output_dir(),
            seed: 0,
            fixture_path: None,
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
    pub pool: Option<ContainerPoolTargetConfig>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub auth_bearer_env: Option<String>,
    #[serde(default)]
    pub auth_refresh: Option<ContainerAuthRefreshConfig>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_startup_timeout_seconds")]
    pub startup_timeout_seconds: u64,
}

impl ContainerConfig {
    pub fn resolve_pool_target(&mut self) -> Result<()> {
        let Some(pool) = &self.pool else {
            return Ok(());
        };
        let pool_id = pool.pool_id.trim();
        if pool_id.is_empty() {
            return Err(OptimizerError::Config(
                "container.pool.pool_id is required when container.pool is set".to_string(),
            ));
        }
        reject_path_segment("container.pool.pool_id", pool_id)?;
        let task_id = pool
            .task_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some(task_id) = task_id {
            reject_path_segment("container.pool.task_id", task_id)?;
        }

        let base_url_source = pool
            .backend_base_url
            .clone()
            .or_else(resolve_backend_base_url_from_env)
            .unwrap_or_else(|| "https://api.usesynth.ai".to_string());
        let base_url = normalize_backend_base_url(&base_url_source);
        self.url = Some(match task_id {
            Some(task_id) => {
                format!("{base_url}/v1/pools/{pool_id}/tasks/{task_id}/container")
            }
            None => format!("{base_url}/v1/pools/{pool_id}/container"),
        });
        if self
            .auth_bearer_env
            .as_deref()
            .unwrap_or_default()
            .trim()
            .is_empty()
            && !headers_contain_authorization(&self.headers)
        {
            self.auth_bearer_env = Some(pool.api_key_env.clone());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerAuthRefreshConfig {
    pub provider: String,
    pub lease_id: String,
    #[serde(default)]
    pub refresh_interval_seconds: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerPoolTargetConfig {
    pub pool_id: String,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub backend_base_url: Option<String>,
    #[serde(default = "default_container_pool_api_key_env")]
    pub api_key_env: String,
}

impl Default for ContainerPoolTargetConfig {
    fn default() -> Self {
        Self {
            pool_id: String::new(),
            task_id: None,
            backend_base_url: None,
            api_key_env: default_container_pool_api_key_env(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TasksetConfig {
    #[serde(default = "default_train_split")]
    pub train_split: String,
    #[serde(default = "default_heldout_split")]
    pub heldout_split: String,
    #[serde(default)]
    pub train_ids: Vec<String>,
    #[serde(default)]
    pub heldout_ids: Vec<String>,
    #[serde(default)]
    pub filters: Map<String, Value>,
}

impl Default for TasksetConfig {
    fn default() -> Self {
        Self {
            train_split: default_train_split(),
            heldout_split: default_heldout_split(),
            train_ids: Vec::new(),
            heldout_ids: Vec::new(),
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
    #[serde(default = "default_policy_enabled")]
    pub enabled: bool,
    #[serde(default = "default_policy_provider")]
    pub provider: String,
    #[serde(default = "default_policy_model")]
    pub model: String,
    #[serde(default)]
    pub policy_type: Option<String>,
    #[serde(default = "default_policy_api_family")]
    pub api_family: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub inference_url: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default = "default_policy_disable_reasoning")]
    pub disable_reasoning: String,
    #[serde(default = "default_policy_tool_call_style")]
    pub tool_call_style: String,
    #[serde(default = "default_policy_proxy_mode")]
    pub proxy_mode: String,
    #[serde(default = "default_policy_credential_mode")]
    pub credential_mode: String,
    #[serde(default, skip_serializing)]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub config: Map<String, Value>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            enabled: default_policy_enabled(),
            provider: default_policy_provider(),
            model: default_policy_model(),
            policy_type: None,
            api_family: default_policy_api_family(),
            base_url: None,
            inference_url: None,
            max_tokens: None,
            disable_reasoning: default_policy_disable_reasoning(),
            tool_call_style: default_policy_tool_call_style(),
            proxy_mode: default_policy_proxy_mode(),
            credential_mode: default_policy_credential_mode(),
            api_key_env: None,
            config: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposerConfig {
    #[serde(default = "default_proposer_backend")]
    pub backend: String,
    #[serde(default = "default_runtime_substrate")]
    pub runtime_substrate: ExecutionSubstrate,
    #[serde(default = "default_execution_mode")]
    pub execution_mode: String,
    #[serde(default = "default_policy_provider")]
    pub provider: String,
    #[serde(default = "default_policy_api_family")]
    pub api_family: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub sandbox_mode: Option<String>,
    #[serde(default)]
    pub approval_policy: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default = "default_proposer_auth_mode")]
    pub auth_mode: String,
    #[serde(default)]
    pub copy_host_auth: bool,
    #[serde(default)]
    pub codex_home: Option<PathBuf>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    /// Max gap between JSON-RPC messages before a turn is flagged as stalled.
    /// Independent of [`Self::timeout_seconds`], the overall turn budget.
    #[serde(default = "default_message_stall_timeout_seconds")]
    pub message_stall_timeout_seconds: u64,
    #[serde(default)]
    pub model: Option<String>,
    /// Opt out of the curated OpenRouter model allowlist so any OpenRouter
    /// slug can be used as a proposer. Cost is taken from OpenRouter's reported
    /// usage instead of a verified static price.
    #[serde(default)]
    pub allow_unverified_model: bool,
    /// Codex `model_context_window`. Codex has no per-call output cap, so when it
    /// does not recognise a model slug it falls back to a conservative window and
    /// can compact or truncate a large proposer turn mid-flight. Set this for any
    /// model Codex does not ship metadata for (e.g. OpenRouter slugs).
    #[serde(default)]
    pub model_context_window: Option<u64>,
    /// Codex `model_auto_compact_token_limit`. Raise alongside the context window
    /// so a long reflection turn is not auto-compacted before it completes.
    #[serde(default)]
    pub model_auto_compact_token_limit: Option<u64>,
    #[serde(default)]
    pub prompt: ProposerPromptConfig,
    #[serde(default)]
    pub docker: Option<ProposerDockerConfig>,
    #[serde(default)]
    pub daytona: Option<ProposerDaytonaConfig>,
    /// Optional MCP server the proposer may call. Off by default.
    #[serde(default)]
    pub mcp: McpAgentConfig,
    /// Follow-up turns after a `validate_manifest_contract` failure. 0 = current
    /// fail-closed behaviour.
    #[serde(default)]
    pub schema_repair_rounds: u32,
}

impl Default for ProposerConfig {
    fn default() -> Self {
        Self {
            backend: default_proposer_backend(),
            runtime_substrate: default_runtime_substrate(),
            execution_mode: default_execution_mode(),
            provider: default_policy_provider(),
            api_family: default_policy_api_family(),
            base_url: None,
            command: Vec::new(),
            sandbox_mode: None,
            approval_policy: None,
            reasoning_effort: None,
            service_tier: None,
            auth_mode: default_proposer_auth_mode(),
            copy_host_auth: false,
            codex_home: None,
            api_key_env: None,
            timeout_seconds: default_timeout_seconds(),
            message_stall_timeout_seconds: default_message_stall_timeout_seconds(),
            model: None,
            allow_unverified_model: false,
            model_context_window: None,
            model_auto_compact_token_limit: None,
            prompt: ProposerPromptConfig::default(),
            docker: None,
            daytona: None,
            mcp: McpAgentConfig::default(),
            schema_repair_rounds: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposerDockerConfig {
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default = "default_docker_workspace_mount_path")]
    pub workspace_mount_path: String,
    #[serde(default = "default_docker_network")]
    pub network: String,
    #[serde(default)]
    pub extra_env: BTreeMap<String, String>,
}

impl Default for ProposerDockerConfig {
    fn default() -> Self {
        Self {
            image: None,
            workspace_mount_path: default_docker_workspace_mount_path(),
            network: default_docker_network(),
            extra_env: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposerDaytonaConfig {
    #[serde(default = "default_daytona_api_url")]
    pub api_url: String,
    #[serde(default = "default_daytona_api_key_env")]
    pub api_key_env: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub snapshot: Option<String>,
    #[serde(default)]
    pub dockerfile_content: Option<String>,
    #[serde(default = "default_daytona_language")]
    pub language: String,
    #[serde(default = "default_daytona_sandbox_name_prefix")]
    pub sandbox_name_prefix: String,
    #[serde(default = "default_daytona_remote_workspace_dir")]
    pub remote_workspace_dir: String,
    #[serde(default = "default_daytona_auto_stop_interval_minutes")]
    pub auto_stop_interval_minutes: u64,
    #[serde(default = "default_daytona_startup_timeout_seconds")]
    pub startup_timeout_seconds: u64,
    #[serde(default = "default_daytona_poll_interval_ms")]
    pub poll_interval_ms: u64,
    #[serde(default = "default_daytona_public")]
    pub public: bool,
    #[serde(default)]
    pub keep_sandbox: bool,
    #[serde(default)]
    pub sync_workspace_back: bool,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub extra_env: BTreeMap<String, String>,
}

impl Default for ProposerDaytonaConfig {
    fn default() -> Self {
        Self {
            api_url: default_daytona_api_url(),
            api_key_env: default_daytona_api_key_env(),
            target: None,
            image: None,
            snapshot: None,
            dockerfile_content: None,
            language: default_daytona_language(),
            sandbox_name_prefix: default_daytona_sandbox_name_prefix(),
            remote_workspace_dir: default_daytona_remote_workspace_dir(),
            auto_stop_interval_minutes: default_daytona_auto_stop_interval_minutes(),
            startup_timeout_seconds: default_daytona_startup_timeout_seconds(),
            poll_interval_ms: default_daytona_poll_interval_ms(),
            public: default_daytona_public(),
            keep_sandbox: false,
            sync_workspace_back: false,
            env: BTreeMap::new(),
            extra_env: BTreeMap::new(),
        }
    }
}

/// Proposer models allowed when using ChatGPT subscription auth (`auth_mode = chatgpt`).
/// Proposer models a ChatGPT-subscription Codex app-server can launch.
///
/// This is not "every model that exists" — it is the set the Codex harness can
/// actually serve under subscription auth. Adding an id that ChatGPT auth cannot
/// serve trades a clean config error for a runtime failure mid-run, which is the
/// worse of the two.
///
/// The gpt-5.6 Codex family is luna / sol / terra, corroborated by the backend
/// model catalog (`packages/smr/config/supported_models_catalog.py`, all three
/// `harnesses = ["codex"]` on `openai_chatgpt_pool`) and by the desktop app's
/// model capabilities. Each supports reasoning efforts low/medium/high/xhigh and
/// defaults to medium.
pub const CHATGPT_PROPOSER_MODELS: &[&str] = &[
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
];

pub fn proposer_auth_mode_normalized(auth_mode: &str) -> String {
    let normalized = normalize_enum_value(auth_mode);
    if normalized == "host" {
        "chatgpt".to_string()
    } else {
        normalized
    }
}

pub fn proposer_uses_chatgpt_auth(auth_mode: &str) -> bool {
    proposer_auth_mode_normalized(auth_mode) == "chatgpt"
}

/// Resolved proposer credential path for Codex app-server launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProposerAuthLaunchMode {
    ApiKey,
    Chatgpt,
}

pub fn resolve_proposer_auth_launch_mode(
    proposer: &ProposerConfig,
    api_key_present: bool,
) -> Result<ProposerAuthLaunchMode> {
    let auth_mode = proposer_auth_mode_normalized(&proposer.auth_mode);
    match auth_mode.as_str() {
        "api_key" => Ok(ProposerAuthLaunchMode::ApiKey),
        "chatgpt" => Ok(ProposerAuthLaunchMode::Chatgpt),
        "auto" => {
            if proposer_model_requires_chatgpt_auth(proposer) {
                if proposer.codex_home.is_some() {
                    return Ok(ProposerAuthLaunchMode::Chatgpt);
                }
                return Err(OptimizerError::Config(
                    "proposer.auth_mode = \"auto\" did not resolve: ChatGPT-subscription \
                     proposer models require proposer.codex_home for ChatGPT subscription auth"
                        .to_string(),
                ));
            }
            if api_key_present {
                Ok(ProposerAuthLaunchMode::ApiKey)
            } else if proposer.codex_home.is_some() {
                Ok(ProposerAuthLaunchMode::Chatgpt)
            } else {
                Err(OptimizerError::Config(
                    "proposer.auth_mode = \"auto\" did not resolve: export an API key \
                     (proposer.api_key_env, default OPENAI_API_KEY) or set proposer.codex_home \
                     for ChatGPT subscription auth"
                        .to_string(),
                ))
            }
        }
        mode => Err(OptimizerError::Config(format!(
            "unsupported proposer.auth_mode {mode:?}; expected auto, api_key, or chatgpt \
             (legacy host maps to chatgpt)"
        ))),
    }
}

fn proposer_model_requires_chatgpt_auth(proposer: &ProposerConfig) -> bool {
    proposer
        .model
        .as_deref()
        .map(normalize_chatgpt_proposer_model_id)
        .is_some_and(|model| CHATGPT_PROPOSER_MODELS.contains(&model.as_str()))
}

pub fn validate_chatgpt_proposer_model(model: &str) -> Result<()> {
    let normalized = normalize_chatgpt_proposer_model_id(model);
    if CHATGPT_PROPOSER_MODELS.contains(&normalized.as_str()) {
        return Ok(());
    }
    Err(OptimizerError::Config(format!(
        "proposer.model {model:?} is not allowed for proposer.auth_mode = \"chatgpt\"; \
         allowed models: {}",
        CHATGPT_PROPOSER_MODELS.join(", ")
    )))
}

pub fn validate_chatgpt_proposer_config(proposer: &ProposerConfig) -> Result<()> {
    let codex_home = proposer.codex_home.as_ref().ok_or_else(|| {
        OptimizerError::Config(
            "proposer.auth_mode = \"chatgpt\" requires proposer.codex_home pointing at the \
             ChatGPT-authenticated Codex directory (for example ~/.codex after `codex auth login`)"
                .to_string(),
        )
    })?;
    if codex_home.as_os_str().is_empty() {
        return Err(OptimizerError::Config(
            "proposer.codex_home must be non-empty when proposer.auth_mode = \"chatgpt\""
                .to_string(),
        ));
    }
    let source = fs::canonicalize(codex_home).map_err(|source| {
        OptimizerError::Config(format!(
            "proposer.codex_home {codex_home:?} must exist when proposer.auth_mode = \"chatgpt\": \
             {source}"
        ))
    })?;
    if !source.is_dir() {
        return Err(OptimizerError::Config(format!(
            "proposer.codex_home {source:?} must be a directory when proposer.auth_mode = \"chatgpt\""
        )));
    }
    let auth_json = source.join("auth.json");
    if !auth_json.is_file() {
        return Err(OptimizerError::Config(format!(
            "proposer.codex_home {source:?} is missing auth.json; run `codex auth login` or fix \
             proposer.codex_home"
        )));
    }
    let model = proposer
        .model
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Config(
                "proposer.auth_mode = \"chatgpt\" requires proposer.model to be set".to_string(),
            )
        })?;
    validate_chatgpt_proposer_model(model)
}

pub fn resolve_chatgpt_codex_home_source(proposer: &ProposerConfig) -> Result<PathBuf> {
    validate_chatgpt_proposer_config(proposer)?;
    let Some(codex_home) = proposer.codex_home.as_ref() else {
        return Err(OptimizerError::Config(
            "proposer.auth_mode = \"chatgpt\" requires proposer.codex_home".to_string(),
        ));
    };
    fs::canonicalize(codex_home).map_err(|source| OptimizerError::io(codex_home, source))
}

fn validate_proposer_auth_config(proposer: &ProposerConfig) -> Result<()> {
    let auth_mode = proposer_auth_mode_normalized(&proposer.auth_mode);
    match auth_mode.as_str() {
        "auto" | "api_key" | "chatgpt" => {}
        mode => {
            return Err(OptimizerError::Config(format!(
                "unsupported proposer.auth_mode {mode:?}; expected auto, api_key, or chatgpt \
                 (legacy host maps to chatgpt)"
            )));
        }
    }
    if auth_mode == "api_key" && proposer.copy_host_auth {
        return Err(OptimizerError::Config(
            "proposer.auth_mode = \"api_key\" cannot be combined with proposer.copy_host_auth = true".to_string(),
        ));
    }
    if proposer.copy_host_auth && auth_mode != "api_key" && proposer.codex_home.is_none() {
        return Err(OptimizerError::Config(
            "proposer.copy_host_auth requires proposer.codex_home".to_string(),
        ));
    }
    if auth_mode == "chatgpt" {
        if proposer.api_key_env.is_some() {
            return Err(OptimizerError::Config(
                "proposer.auth_mode = \"chatgpt\" cannot be combined with proposer.api_key_env"
                    .to_string(),
            ));
        }
        validate_chatgpt_proposer_config(proposer)?;
    }
    Ok(())
}

fn validate_proposer_reasoning_effort(proposer: &ProposerConfig) -> Result<()> {
    let Some(reasoning_effort) = proposer.reasoning_effort.as_deref() else {
        return Ok(());
    };
    let normalized = normalize_enum_value(reasoning_effort);
    if matches!(normalized.as_str(), "none" | "low" | "medium" | "high") {
        return Ok(());
    }
    Err(OptimizerError::Config(format!(
        "proposer.reasoning_effort must be none, low, medium, or high; got {reasoning_effort:?}. \
         Use proposer.service_tier = \"fast\" for Codex Fast mode."
    )))
}

fn validate_proposer_service_tier(proposer: &ProposerConfig) -> Result<()> {
    let Some(service_tier) = proposer.service_tier.as_deref() else {
        return Ok(());
    };
    let normalized = normalize_enum_value(service_tier);
    if normalized != "fast" {
        return Err(OptimizerError::Config(format!(
            "unsupported proposer.service_tier {service_tier:?}; expected fast or omit the field \
             for the default Codex tier"
        )));
    }
    if proposer.backend.trim() != "codex_app_server" {
        return Err(OptimizerError::Config(
            "proposer.service_tier = \"fast\" requires proposer.backend = \"codex_app_server\""
                .to_string(),
        ));
    }
    if !proposer_uses_chatgpt_auth(&proposer.auth_mode) {
        return Err(OptimizerError::Config(
            "proposer.service_tier = \"fast\" requires proposer.auth_mode = \"chatgpt\"; \
             API-key Codex runs use standard API pricing, not Codex Fast mode"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_proposer_runtime_substrate_config(proposer: &ProposerConfig) -> Result<()> {
    match proposer.runtime_substrate {
        ExecutionSubstrate::Local => Ok(()),
        ExecutionSubstrate::Docker => validate_proposer_docker_config(proposer),
        ExecutionSubstrate::Daytona => validate_proposer_daytona_config(proposer),
    }
}

fn validate_chat_completions_proposer_config(proposer: &ProposerConfig) -> Result<()> {
    let provider = proposer.provider.trim().to_ascii_lowercase();
    if !matches!(provider.as_str(), "deepseek" | "nvidia" | "openai") {
        return Err(OptimizerError::Config(format!(
            "chat-completions proposer backend requires proposer.provider = \"deepseek\", \"nvidia\", or \"openai\"; got {:?}",
            proposer.provider
        )));
    }
    if proposer_auth_mode_normalized(&proposer.auth_mode) != "api_key" {
        return Err(OptimizerError::Config(
            "chat-completions proposer backend requires proposer.auth_mode = \"api_key\""
                .to_string(),
        ));
    }
    if !matches!(proposer.runtime_substrate, ExecutionSubstrate::Local) {
        return Err(OptimizerError::Config(
            "chat-completions proposer backend requires proposer.runtime_substrate = \"local\""
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_openrouter_proposer_config(proposer: &ProposerConfig) -> Result<()> {
    if !proposer.provider.eq_ignore_ascii_case("openrouter") {
        return Ok(());
    }
    let model = proposer
        .model
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Config(
                "proposer.provider = \"openrouter\" requires proposer.model".to_string(),
            )
        })?;
    // Curated allowlist of first-class OpenRouter proposers (verified, with a
    // known static price). Any other slug requires the explicit
    // proposer.allow_unverified_model opt-in, after which OpenRouter validates
    // the slug and cost flows through from its reported usage.
    const VERIFIED_OPENROUTER_MODELS: [&str; 1] = ["x-ai/grok-4.3"];
    let normalized_model = model.trim().to_ascii_lowercase();
    if !proposer.allow_unverified_model
        && !VERIFIED_OPENROUTER_MODELS.contains(&normalized_model.as_str())
    {
        return Err(OptimizerError::Config(format!(
            "OpenRouter proposer.model {model:?} is not in the verified allowlist ({}); \
             set proposer.allow_unverified_model = true to use any OpenRouter model",
            VERIFIED_OPENROUTER_MODELS.join(", ")
        )));
    }
    if proposer.backend != "codex_app_server" {
        return Err(OptimizerError::Config(
            "OpenRouter proposer requires proposer.backend = \"codex_app_server\"".to_string(),
        ));
    }
    let auth_mode = proposer_auth_mode_normalized(&proposer.auth_mode);
    if !matches!(auth_mode.as_str(), "api_key" | "auto") {
        return Err(OptimizerError::Config(
            "OpenRouter proposer requires proposer.auth_mode = \"api_key\" or \"auto\"".to_string(),
        ));
    }
    let api_key_env = proposer
        .api_key_env
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Config(
                "OpenRouter proposer requires proposer.api_key_env, usually OPENROUTER_API_KEY"
                    .to_string(),
            )
        })?;
    if api_key_env == "OPENAI_API_KEY" {
        return Err(OptimizerError::Config(
            "OpenRouter proposer must not use OPENAI_API_KEY; set \
             proposer.api_key_env = \"OPENROUTER_API_KEY\""
                .to_string(),
        ));
    }
    let api_family = normalize_enum_value(&proposer.api_family);
    if !matches!(api_family.as_str(), "chat_completions" | "responses") {
        return Err(OptimizerError::Config(format!(
            "OpenRouter proposer supports chat_completions or responses; got {:?}",
            proposer.api_family
        )));
    }
    if let Some(reasoning_effort) = proposer.reasoning_effort.as_deref() {
        let normalized_effort = normalize_enum_value(reasoning_effort);
        if !matches!(
            normalized_effort.as_str(),
            "none" | "low" | "medium" | "high"
        ) {
            return Err(OptimizerError::Config(format!(
                "OpenRouter proposer.reasoning_effort must be none, low, medium, \
                 or high; got {reasoning_effort:?}"
            )));
        }
    }
    Ok(())
}

fn validate_proposer_docker_config(proposer: &ProposerConfig) -> Result<()> {
    let docker = proposer.docker.as_ref().ok_or_else(|| {
        OptimizerError::Config(
            "proposer.runtime_substrate = \"docker\" requires [proposer.docker]".to_string(),
        )
    })?;
    let image = docker.image.as_deref().unwrap_or_default().trim();
    if image.is_empty() {
        return Err(OptimizerError::Config(
            "proposer.runtime_substrate = \"docker\" requires [proposer.docker].image; pin a tag or digest, do not rely on latest".to_string(),
        ));
    }
    if image.ends_with(":latest") || image == "latest" {
        return Err(OptimizerError::Config(
            "proposer.docker.image must be pinned to a non-latest tag or digest".to_string(),
        ));
    }
    if !docker.workspace_mount_path.starts_with('/') {
        return Err(OptimizerError::Config(format!(
            "proposer.docker.workspace_mount_path must be an absolute container path; got {:?}",
            docker.workspace_mount_path
        )));
    }
    if !matches!(docker.network.as_str(), "bridge" | "host" | "none") {
        return Err(OptimizerError::Config(format!(
            "proposer.docker.network must be bridge, host, or none; got {:?}",
            docker.network
        )));
    }
    if proposer_uses_chatgpt_auth(&proposer.auth_mode) {
        return Err(OptimizerError::Config(
            "proposer.runtime_substrate = \"docker\" does not support auth_mode = \"chatgpt\" in v1; use runtime_substrate = \"local\" or api_key proposer auth"
                .to_string(),
        ));
    }
    for (container_key, host_key) in &docker.extra_env {
        if container_key.trim().is_empty() || host_key.trim().is_empty() {
            return Err(OptimizerError::Config(
                "proposer.docker.extra_env entries must map non-empty container env names to non-empty host env names".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_proposer_daytona_config(proposer: &ProposerConfig) -> Result<()> {
    let daytona = proposer.daytona.as_ref().ok_or_else(|| {
        OptimizerError::Config(
            "proposer.runtime_substrate = \"daytona\" requires [proposer.daytona]".to_string(),
        )
    })?;
    if daytona.api_url.trim().is_empty() {
        return Err(OptimizerError::Config(
            "proposer.daytona.api_url must be non-empty".to_string(),
        ));
    }
    if daytona.api_key_env.trim().is_empty() {
        return Err(OptimizerError::Config(
            "proposer.daytona.api_key_env must name the environment variable holding the Daytona API key".to_string(),
        ));
    }
    let dockerfile_content = daytona
        .dockerfile_content
        .as_deref()
        .unwrap_or_default()
        .trim();
    let image = daytona.image.as_deref().unwrap_or_default().trim();
    let snapshot = daytona.snapshot.as_deref().unwrap_or_default().trim();
    if dockerfile_content.is_empty() && image.is_empty() && snapshot.is_empty() {
        return Err(OptimizerError::Config(
            "proposer.runtime_substrate = \"daytona\" requires [proposer.daytona].dockerfile_content, [proposer.daytona].image, or [proposer.daytona].snapshot".to_string(),
        ));
    }
    if !image.is_empty() && (image.ends_with(":latest") || image == "latest") {
        return Err(OptimizerError::Config(
            "proposer.daytona.image must be pinned to a non-latest tag or digest".to_string(),
        ));
    }
    if daytona.language.trim().is_empty() {
        return Err(OptimizerError::Config(
            "proposer.daytona.language must be non-empty".to_string(),
        ));
    }
    if daytona.sandbox_name_prefix.trim().is_empty() {
        return Err(OptimizerError::Config(
            "proposer.daytona.sandbox_name_prefix must be non-empty".to_string(),
        ));
    }
    if !daytona.remote_workspace_dir.starts_with('/') {
        return Err(OptimizerError::Config(format!(
            "proposer.daytona.remote_workspace_dir must be an absolute path; got {:?}",
            daytona.remote_workspace_dir
        )));
    }
    if daytona.startup_timeout_seconds == 0 {
        return Err(OptimizerError::Config(
            "proposer.daytona.startup_timeout_seconds must be greater than zero".to_string(),
        ));
    }
    if daytona.poll_interval_ms == 0 {
        return Err(OptimizerError::Config(
            "proposer.daytona.poll_interval_ms must be greater than zero".to_string(),
        ));
    }
    if proposer.execution_mode.trim() != "local_process"
        && proposer.execution_mode.trim() != "stdio"
    {
        return Err(OptimizerError::Config(
            "proposer.runtime_substrate = \"daytona\" requires proposer.execution_mode = \"local_process\" or \"stdio\"; websocket mode is local-process only".to_string(),
        ));
    }
    if proposer_uses_chatgpt_auth(&proposer.auth_mode) {
        return Err(OptimizerError::Config(
            "proposer.runtime_substrate = \"daytona\" does not support auth_mode = \"chatgpt\" in v1; use api_key proposer auth".to_string(),
        ));
    }
    for (key, value) in &daytona.env {
        if key.trim().is_empty() || value.trim().is_empty() {
            return Err(OptimizerError::Config(
                "proposer.daytona.env entries must have non-empty keys and values".to_string(),
            ));
        }
    }
    for (container_key, host_key) in &daytona.extra_env {
        if container_key.trim().is_empty() || host_key.trim().is_empty() {
            return Err(OptimizerError::Config(
                "proposer.daytona.extra_env entries must map non-empty sandbox env names to non-empty host env names".to_string(),
            ));
        }
    }
    Ok(())
}

fn normalize_chatgpt_proposer_model_id(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposerPromptConfig {
    #[serde(default)]
    pub best_practices: Option<String>,
    #[serde(default)]
    pub best_practices_path: Option<PathBuf>,
    /// Extra prompt-opt style guides concatenated after best_practices.
    #[serde(default)]
    pub style_guides: Vec<PathBuf>,
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
pub struct GepaTaskPoolsConfig {
    #[serde(default)]
    pub pareto: Vec<String>,
    #[serde(default)]
    pub minibatch: Vec<String>,
    #[serde(default)]
    pub reflection: Vec<String>,
    #[serde(default)]
    pub heldout: Vec<String>,
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
    #[serde(default = "default_rollout_failure_rate_tolerance")]
    pub rollout_failure_rate_tolerance: f64,
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
    #[serde(default = "default_task_pools_config")]
    pub task_pools: GepaTaskPoolsConfig,
    #[serde(default = "default_gepa_pipeline_config")]
    pub pipeline: GepaPipelineConfig,
    #[serde(default)]
    pub max_cost_usd: f64,
    #[serde(default)]
    pub max_time_seconds: Option<u64>,
    #[serde(default)]
    pub no_improvement_generations: Option<usize>,
    #[serde(default)]
    pub no_improvement_metric: Option<String>,
    #[serde(default)]
    pub score_threshold_value: Option<f64>,
    #[serde(default)]
    pub score_threshold_metric: Option<String>,
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
    /// Delta-from-restart episode horizon. When a limit here is set, it
    /// replaces the matching absolute GEPA budget for search-loop stop.
    #[serde(default)]
    pub episode: GepaEpisodeConfig,
    /// Optional operator surfaces (manderqueue, scratchpad, hypotheses, MCP).
    /// All default off so existing runs are unchanged.
    #[serde(default)]
    pub operator: GepaOperatorConfig,
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
            rollout_failure_rate_tolerance: default_rollout_failure_rate_tolerance(),
            frontier_type: default_frontier_type(),
            selection_objective: None,
            objective_keys: Vec::new(),
            objective_directions: BTreeMap::new(),
            acceptance_criterion: default_acceptance_criterion(),
            objective_acceptance: default_objective_acceptance_config(),
            candidate_selector: default_candidate_selector_config(),
            batch_sampler: default_batch_sampler_config(),
            task_pools: default_task_pools_config(),
            pipeline: default_gepa_pipeline_config(),
            max_cost_usd: 0.0,
            max_time_seconds: None,
            no_improvement_generations: None,
            no_improvement_metric: None,
            score_threshold_value: None,
            score_threshold_metric: None,
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
            episode: GepaEpisodeConfig::default(),
            operator: GepaOperatorConfig::default(),
        }
    }
}

/// Search-loop stop measured from restart (fresh run or fixture fork).
/// First matching limit wins. Unset fields fall through to the absolute
/// `[gepa]` budgets (`max_generations`, `max_train_rollouts`, `max_cost_usd`).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GepaEpisodeConfig {
    /// Completed proposer rounds since restart.
    #[serde(default)]
    pub proposer_rounds: Option<usize>,
    /// Inner train rollouts since restart.
    #[serde(default)]
    pub max_rollouts: Option<usize>,
    /// Wall seconds since restart.
    #[serde(default)]
    pub max_wall_seconds: Option<u64>,
    /// Spend USD since restart.
    #[serde(default)]
    pub max_spend_usd: Option<f64>,
    /// Complete on train evidence and do not enter heldout.
    #[serde(default)]
    pub skip_heldout: bool,
}

impl GepaEpisodeConfig {
    pub fn has_delta_limits(&self) -> bool {
        self.proposer_rounds.is_some()
            || self.max_rollouts.is_some()
            || self.max_wall_seconds.is_some()
            || self.max_spend_usd.is_some()
    }
}

/// Optional GEPA operator surfaces. Default-off; enabling a nested block is the
/// opt-in. HTTP pause/resume/fork already exist; these knobs expose them to the
/// proposer workspace and turn on MQ / scratchpad / hypotheses / MCP when wanted.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct GepaOperatorConfig {
    pub manderqueue: ManderqueueConfig,
    pub scratchpad: ScratchpadConfig,
    pub hypotheses: HypothesesConfig,
    pub control: ControlSurfaceConfig,
    pub levers: LeverSurfaceConfig,
    pub reward: RewardSurfaceConfig,
    pub mcp_agent: McpAgentConfig,
}

impl GepaOperatorConfig {
    pub fn any_enabled(&self) -> bool {
        self.manderqueue.enabled
            || self.scratchpad.enabled
            || self.hypotheses.enabled
            || self.mcp_agent.enabled
    }
}

/// Proposer ↔ operator comms via the Manderqueue HTTP API (`mq-sdk` contract).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ManderqueueConfig {
    pub enabled: bool,
    pub base_url: Option<String>,
    #[serde(default = "default_manderqueue_token_env")]
    pub token_env: String,
    pub thread_id: Option<String>,
    pub org_id: Option<String>,
    pub scope_kind: Option<String>,
    pub scope_id: Option<String>,
    #[serde(default = "default_manderqueue_poll_seconds")]
    pub poll_seconds: u64,
    pub fail_closed: bool,
}

impl Default for ManderqueueConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: None,
            token_env: default_manderqueue_token_env(),
            thread_id: None,
            org_id: None,
            scope_kind: None,
            scope_id: None,
            poll_seconds: default_manderqueue_poll_seconds(),
            fail_closed: false,
        }
    }
}

fn default_manderqueue_token_env() -> String {
    "MANDERQUEUE_TOKEN".to_string()
}

fn default_manderqueue_poll_seconds() -> u64 {
    5
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ScratchpadConfig {
    pub enabled: bool,
    #[serde(default = "default_scratchpad_path")]
    pub path: String,
    pub shared: bool,
}

impl Default for ScratchpadConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: default_scratchpad_path(),
            shared: true,
        }
    }
}

fn default_scratchpad_path() -> String {
    "state/scratchpad.md".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct HypothesesConfig {
    pub enabled: bool,
    #[serde(default = "default_hypotheses_max_open")]
    pub max_open: usize,
}

impl Default for HypothesesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_open: default_hypotheses_max_open(),
        }
    }
}

fn default_hypotheses_max_open() -> usize {
    8
}

/// Pause / resume / fork are always on the service. This records whether the
/// proposer workspace should treat them as part of the episode contract.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ControlSurfaceConfig {
    pub pause: bool,
    pub restart: bool,
    pub branch: bool,
}

impl Default for ControlSurfaceConfig {
    fn default() -> Self {
        Self {
            pause: true,
            restart: true,
            branch: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct LeverSurfaceConfig {
    pub prompt: bool,
    pub code: bool,
    pub harness: bool,
}

impl Default for LeverSurfaceConfig {
    fn default() -> Self {
        Self {
            prompt: true,
            code: true,
            harness: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct RewardSurfaceConfig {
    /// How a missing optional term is treated (`zero` or `fail`).
    #[serde(default = "default_reward_missing")]
    pub missing: String,
    pub confidence: bool,
    pub time: bool,
    pub cost: bool,
    pub milestones: bool,
    pub rubrics: bool,
    #[serde(default = "default_exploration_reduce")]
    pub exploration_reduce: String,
}

impl Default for RewardSurfaceConfig {
    fn default() -> Self {
        Self {
            missing: default_reward_missing(),
            confidence: false,
            time: false,
            cost: false,
            milestones: false,
            rubrics: false,
            exploration_reduce: default_exploration_reduce(),
        }
    }
}

fn default_reward_missing() -> String {
    "zero".to_string()
}

fn default_exploration_reduce() -> String {
    "mean".to_string()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct McpAgentConfig {
    pub enabled: bool,
    /// MCP server command, e.g. `npx -y @modelcontextprotocol/server-github`.
    pub command: Option<String>,
    /// Named MCP server the proposer may call. Empty unless `enabled`.
    pub server: Option<String>,
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
    /// Async pipeline that overlaps proposer reflection with policy rollouts to
    /// realize the paper's wall-clock speedup over `SyncSerial`.
    ///
    /// HISTORY (Banking77 matrix `20260602052705`, measured 2026-06-02):
    /// FlashEvolve was ~31% *slower* than `SyncSerial` (0.687x, 741s vs 509s)
    /// at identical heldout quality (0.750), with proposer/policy overlap of
    /// ~0.33s. The 2026-06-02 note attributed this to admission order
    /// (`candidate_full_train` draining before the next proposer is admitted).
    /// That diagnosis was incomplete: the driver executed every leased lane job
    /// **inline on the tick**, so no two lanes could occupy wall clock at once
    /// no matter what order they were admitted in. Lane leases modelled
    /// concurrency that the executor never provided.
    ///
    /// FlashEvolve now defaults to `pipeline.background_execution = true`,
    /// which dispatches leased jobs to worker threads (each with its own
    /// workspace and request-cache handle on the shared sqlite files) and
    /// leaves the tick loop free to admit the next proposer. Admission order
    /// was fixed alongside it: with background execution on, a pending job no
    /// longer preempts proposer admission.
    ///
    /// `overlap_seconds` on `cursor.pipeline_state` reports measured
    /// propose/rollout wall-clock overlap; it is the metric that decides
    /// whether this mode is actually earning its name on a given workload.
    ///
    /// `combee` is an alias for this mode.
    #[serde(alias = "combee", alias = "flash", alias = "flashevolve")]
    FlashEvolve,
}

impl GepaPipelineMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SyncSerial => "sync_serial",
            Self::AsyncPipelined => "async_pipelined",
            Self::FlashEvolve => "flash_evolve",
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
    #[serde(default = "default_pipeline_staleness_delta_max")]
    pub delta_max: u64,
    #[serde(default)]
    pub speculative_completion: GepaSpeculativeCompletionConfig,
    #[serde(default)]
    pub adaptive_stage_workers: GepaAdaptiveStageWorkersConfig,
    #[serde(default)]
    pub adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyConfig,
    /// Run leased lane jobs on background worker threads instead of inline on
    /// the driver tick.
    ///
    /// Without this the "async" pipeline modes are cooperative only: the tick
    /// loop executes one leased job to completion before it can plan or execute
    /// the next, so a propose lane and a rollout lane hold leases at the same
    /// time but never occupy wall clock at the same time. That is what made
    /// FlashEvolve a wall-clock regression on the 2026-06-02 Banking77 matrix.
    ///
    /// Defaults to on for `flash_evolve` and off for every other mode, so
    /// `async_pipelined` keeps its measured 2026-06-02 behaviour and stays a
    /// clean control arm.
    #[serde(default)]
    pub background_execution: Option<bool>,
    /// Worker threads backing `background_execution`. Defaults to
    /// `workers.propose + workers.rollout + workers.evaluate` so every lane can
    /// be resident at once.
    #[serde(default)]
    pub background_workers: Option<usize>,
}

impl Default for GepaPipelineConfig {
    fn default() -> Self {
        Self {
            mode: GepaPipelineMode::SyncSerial,
            staleness_policy: GepaStalenessPolicy::Full,
            max_in_flight_candidates: default_pipeline_max_in_flight_candidates(),
            workers: GepaPipelineWorkers::default(),
            delta_max: default_pipeline_staleness_delta_max(),
            speculative_completion: GepaSpeculativeCompletionConfig::default(),
            adaptive_stage_workers: GepaAdaptiveStageWorkersConfig::default(),
            adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyConfig::default(),
            background_execution: None,
            background_workers: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaSpeculativeCompletionConfig {
    #[serde(default = "default_gepa_speculative_completion_enabled")]
    pub enabled: bool,
    #[serde(default = "default_gepa_speculative_completion_alpha")]
    pub alpha: f64,
}

impl Default for GepaSpeculativeCompletionConfig {
    fn default() -> Self {
        Self {
            enabled: default_gepa_speculative_completion_enabled(),
            alpha: default_gepa_speculative_completion_alpha(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GepaAdaptiveStageWorkersConfig {
    #[serde(default = "default_gepa_adaptive_stage_workers_enabled")]
    pub enabled: bool,
    #[serde(default = "default_gepa_adaptive_stage_workers_min")]
    pub min: usize,
    #[serde(default = "default_gepa_adaptive_stage_workers_max")]
    pub max: usize,
    #[serde(default = "default_gepa_adaptive_stage_workers_backlog_threshold")]
    pub backlog_threshold: usize,
    #[serde(default = "default_gepa_adaptive_stage_workers_stale_gap_threshold")]
    pub stale_gap_threshold: u64,
}

impl Default for GepaAdaptiveStageWorkersConfig {
    fn default() -> Self {
        Self {
            enabled: default_gepa_adaptive_stage_workers_enabled(),
            min: default_gepa_adaptive_stage_workers_min(),
            max: default_gepa_adaptive_stage_workers_max(),
            backlog_threshold: default_gepa_adaptive_stage_workers_backlog_threshold(),
            stale_gap_threshold: default_gepa_adaptive_stage_workers_stale_gap_threshold(),
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

fn headers_contain_authorization(headers: &BTreeMap<String, String>) -> bool {
    headers
        .keys()
        .any(|name| name.trim().eq_ignore_ascii_case("authorization"))
}

fn reject_path_segment(field: &str, value: &str) -> Result<()> {
    if value.contains('/') || value.contains('?') || value.contains('#') {
        return Err(OptimizerError::Config(format!(
            "{field} must be a single URL path segment"
        )));
    }
    Ok(())
}

fn resolve_backend_base_url_from_env() -> Option<String> {
    for name in [
        "SYNTH_BACKEND_URL_OVERRIDE",
        "SYNTH_BACKEND_URL",
        "SYNTH_API_URL",
        "DEV_SYNTH_BACKEND_URL",
        "DEV_BACKEND_URL",
        "PROD_SYNTH_BACKEND_URL",
        "PROD_BACKEND_URL",
        "BACKEND_URL",
    ] {
        if let Ok(value) = env::var(name) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn normalize_backend_base_url(raw: &str) -> String {
    let mut base = raw.trim().trim_end_matches('/').to_string();
    for suffix in ["/v1", "/api"] {
        if base.ends_with(suffix) {
            base.truncate(base.len() - suffix.len());
            break;
        }
    }
    base
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

fn validate_jesterky_workflow_config(config: &JesterkyWorkflowConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    if config.spec.trim().is_empty() {
        return Err(OptimizerError::Config(
            "jesterky_workflow.spec must be non-empty when enabled".to_string(),
        ));
    }
    if config.command.trim().is_empty() {
        return Err(OptimizerError::Config(
            "jesterky_workflow.command must be non-empty when enabled".to_string(),
        ));
    }
    let actor = config.actor.trim().to_ascii_lowercase();
    if !matches!(actor.as_str(), "fake" | "codex") {
        return Err(OptimizerError::Config(format!(
            "jesterky_workflow.actor must be fake or codex when enabled, got {:?}",
            config.actor
        )));
    }
    if config.concurrency == 0 {
        return Err(OptimizerError::Config(
            "jesterky_workflow.concurrency must be > 0 when enabled".to_string(),
        ));
    }
    if config.timeout_seconds == 0 {
        return Err(OptimizerError::Config(
            "jesterky_workflow.timeout_seconds must be > 0 when enabled".to_string(),
        ));
    }
    Ok(())
}

fn validate_policy_config(config: &PolicyConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    if config.provider.trim().is_empty() {
        return Err(OptimizerError::Config(
            "policy.provider must be non-empty".to_string(),
        ));
    }
    if config.model.trim().is_empty() {
        return Err(OptimizerError::Config(
            "policy.model must be non-empty".to_string(),
        ));
    }
    if let Some(policy_type) = config.policy_type.as_deref() {
        let normalized = normalize_enum_value(policy_type);
        if !matches!(normalized.as_str(), "dag" | "react" | "codex") {
            return Err(OptimizerError::Config(format!(
                "policy.policy_type must be dag, react, or codex; got {policy_type:?}"
            )));
        }
    }
    validate_positive_option("policy.max_tokens", config.max_tokens)?;
    let api_family = normalize_enum_value(&config.api_family);
    if !matches!(api_family.as_str(), "chat_completions" | "responses") {
        return Err(OptimizerError::Config(format!(
            "policy.api_family must be chat_completions or responses; got {:?}",
            config.api_family
        )));
    }
    let disable_reasoning = normalize_enum_value(&config.disable_reasoning);
    if !matches!(
        disable_reasoning.as_str(),
        "auto" | "on" | "off" | "true" | "false" | "1" | "0"
    ) {
        return Err(OptimizerError::Config(format!(
            "policy.disable_reasoning must be auto, on, or off; got {:?}",
            config.disable_reasoning
        )));
    }
    let tool_call_style = normalize_enum_value(&config.tool_call_style);
    if !matches!(
        tool_call_style.as_str(),
        "openai_chat" | "openai_responses" | "codex_session_native" | "none"
    ) {
        return Err(OptimizerError::Config(format!(
            "policy.tool_call_style must be openai_chat, openai_responses, codex_session_native, or none; got {:?}",
            config.tool_call_style
        )));
    }
    let proxy_mode = normalize_enum_value(&config.proxy_mode);
    if !matches!(proxy_mode.as_str(), "proxy_only" | "assert_proxy") {
        return Err(OptimizerError::Config(format!(
            "policy.proxy_mode must be proxy_only or assert_proxy; got {:?}",
            config.proxy_mode
        )));
    }
    let credential_mode = normalize_enum_value(&config.credential_mode);
    if !matches!(credential_mode.as_str(), "byok" | "proxy") {
        return Err(OptimizerError::Config(format!(
            "policy.credential_mode must be byok or proxy; got {:?}",
            config.credential_mode
        )));
    }
    if credential_mode == "proxy"
        && config
            .inference_url
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
    {
        return Err(OptimizerError::Config(
            "policy.inference_url must be set when policy.credential_mode is proxy".to_string(),
        ));
    }
    for key in config.config.keys() {
        let normalized = normalize_enum_value(key);
        if normalized.contains("api_key")
            || normalized.contains("authorization")
            || normalized == "token"
            || normalized.ends_with("_token")
        {
            return Err(OptimizerError::Config(format!(
                "policy.config must not contain credential-shaped key {key:?}"
            )));
        }
    }
    Ok(())
}

fn normalize_enum_value(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn normalize_proposer_service_tier(value: &str) -> Option<String> {
    match normalize_enum_value(value).as_str() {
        "" | "default" | "normal" | "standard" => None,
        "fast" => Some("fast".to_string()),
        _ => Some(value.trim().to_string()),
    }
}

fn validate_proposer_prompt_config(config: &ProposerPromptConfig) -> Result<()> {
    if config.best_practices.is_some() && config.best_practices_path.is_some() {
        return Err(OptimizerError::Config(
            "proposer.prompt must set at most one of best_practices or best_practices_path"
                .to_string(),
        ));
    }
    if config
        .best_practices
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(OptimizerError::Config(
            "proposer.prompt.best_practices must be non-empty when set".to_string(),
        ));
    }
    if config
        .best_practices_path
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(OptimizerError::Config(
            "proposer.prompt.best_practices_path must be non-empty when set".to_string(),
        ));
    }
    for path in &config.style_guides {
        if path.as_os_str().is_empty() {
            return Err(OptimizerError::Config(
                "proposer.prompt.style_guides entries must be non-empty".to_string(),
            ));
        }
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

fn validate_gepa_task_pools_config(config: &GepaTaskPoolsConfig) -> Result<()> {
    for (name, values) in [
        ("gepa.task_pools.pareto", &config.pareto),
        ("gepa.task_pools.minibatch", &config.minibatch),
        ("gepa.task_pools.reflection", &config.reflection),
        ("gepa.task_pools.heldout", &config.heldout),
    ] {
        if values.is_empty() {
            return Err(OptimizerError::Config(format!(
                "{name} must contain at least one task id"
            )));
        }
        if values.iter().any(|value| value.trim().is_empty()) {
            return Err(OptimizerError::Config(format!(
                "{name} entries must be non-empty"
            )));
        }
    }

    let minibatch = config.minibatch.iter().cloned().collect::<BTreeSet<_>>();
    let reflection = config.reflection.iter().cloned().collect::<BTreeSet<_>>();
    let pareto = config.pareto.iter().cloned().collect::<BTreeSet<_>>();
    let heldout = config.heldout.iter().cloned().collect::<BTreeSet<_>>();

    let minibatch_not_reflected = minibatch
        .difference(&reflection)
        .cloned()
        .collect::<Vec<_>>();
    if !minibatch_not_reflected.is_empty() {
        return Err(OptimizerError::Config(format!(
            "gepa.task_pools.minibatch must be a subset of gepa.task_pools.reflection; missing from reflection: {:?}",
            minibatch_not_reflected
        )));
    }

    let heldout_overlaps = heldout
        .intersection(&pareto)
        .chain(heldout.intersection(&minibatch))
        .chain(heldout.intersection(&reflection))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !heldout_overlaps.is_empty() {
        return Err(OptimizerError::Config(format!(
            "gepa.task_pools.heldout must be disjoint from pareto, minibatch, and reflection pools; overlaps: {:?}",
            heldout_overlaps
        )));
    }
    Ok(())
}

/// Pools are split-local: search pools (pareto/minibatch/reflection) draw from
/// `taskset.train_ids`, the heldout pool from `taskset.heldout_ids`. A pool id
/// outside its split would silently mis-select (or fail) at rollout time.
fn validate_task_pools_against_taskset(
    config: &GepaTaskPoolsConfig,
    train_ids: &[String],
    heldout_ids: &[String],
) -> Result<()> {
    let train = train_ids.iter().cloned().collect::<BTreeSet<_>>();
    let unknown_search = config
        .pareto
        .iter()
        .chain(&config.minibatch)
        .chain(&config.reflection)
        .filter(|id| !train.contains(*id))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !unknown_search.is_empty() {
        return Err(OptimizerError::Config(format!(
            "gepa.task_pools pareto/minibatch/reflection ids must come from taskset.train_ids; unknown: {:?}",
            unknown_search.into_iter().collect::<Vec<_>>()
        )));
    }
    let held = heldout_ids.iter().cloned().collect::<BTreeSet<_>>();
    let unknown_heldout = config
        .heldout
        .iter()
        .filter(|id| !held.contains(*id))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !unknown_heldout.is_empty() {
        return Err(OptimizerError::Config(format!(
            "gepa.task_pools.heldout ids must come from taskset.heldout_ids; unknown: {:?}",
            unknown_heldout.into_iter().collect::<Vec<_>>()
        )));
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
        "flash_evolve" | "flashevolve" | "flash" | "combee" => Ok(GepaPipelineMode::FlashEvolve),
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

fn validate_gepa_episode_config(config: &GepaEpisodeConfig) -> Result<()> {
    if config.proposer_rounds == Some(0) {
        return Err(OptimizerError::Config(
            "gepa.episode.proposer_rounds must be positive".to_string(),
        ));
    }
    if config.max_rollouts == Some(0) {
        return Err(OptimizerError::Config(
            "gepa.episode.max_rollouts must be positive".to_string(),
        ));
    }
    validate_positive_option("gepa.episode.max_wall_seconds", config.max_wall_seconds)?;
    validate_positive_f64_option("gepa.episode.max_spend_usd", config.max_spend_usd)?;
    Ok(())
}

fn validate_gepa_operator_config(config: &GepaOperatorConfig) -> Result<()> {
    if config.manderqueue.enabled {
        let url = config
            .manderqueue
            .base_url
            .as_deref()
            .unwrap_or("")
            .trim();
        if url.is_empty() && config.manderqueue.fail_closed {
            return Err(OptimizerError::Config(
                "gepa.operator.manderqueue.base_url is required when enabled and fail_closed"
                    .to_string(),
            ));
        }
        if config.manderqueue.poll_seconds == 0 {
            return Err(OptimizerError::Config(
                "gepa.operator.manderqueue.poll_seconds must be positive when enabled".to_string(),
            ));
        }
    }
    if config.scratchpad.enabled && config.scratchpad.path.trim().is_empty() {
        return Err(OptimizerError::Config(
            "gepa.operator.scratchpad.path must be non-empty when enabled".to_string(),
        ));
    }
    if config.hypotheses.enabled && config.hypotheses.max_open == 0 {
        return Err(OptimizerError::Config(
            "gepa.operator.hypotheses.max_open must be positive when enabled".to_string(),
        ));
    }
    let missing = config.reward.missing.trim().to_ascii_lowercase();
    if missing != "zero" && missing != "fail" {
        return Err(OptimizerError::Config(
            "gepa.operator.reward.missing must be zero or fail".to_string(),
        ));
    }
    let reduce = config
        .reward
        .exploration_reduce
        .trim()
        .to_ascii_lowercase();
    if reduce != "mean" && reduce != "sum" {
        return Err(OptimizerError::Config(
            "gepa.operator.reward.exploration_reduce must be mean or sum".to_string(),
        ));
    }
    if config.mcp_agent.enabled {
        validate_mcp_agent_config("gepa.operator.mcp_agent", &config.mcp_agent)?;
    }
    Ok(())
}

fn validate_mcp_agent_config(name: &str, config: &McpAgentConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    let command = config.command.as_deref().unwrap_or("").trim();
    let server = config.server.as_deref().unwrap_or("").trim();
    if command.is_empty() && server.is_empty() {
        return Err(OptimizerError::Config(format!(
            "{name}.command or server is required when enabled"
        )));
    }
    Ok(())
}

fn validate_gepa_pipeline_config(config: &GepaPipelineConfig) -> Result<()> {
    match (config.mode, config.staleness_policy) {
        (GepaPipelineMode::SyncSerial, GepaStalenessPolicy::Full) => {}
        (GepaPipelineMode::SyncSerial, policy) => {
            return Err(OptimizerError::Config(format!(
                "gepa.pipeline.staleness_policy = {policy:?} is incompatible with sync_serial; use full"
            )));
        }
        (GepaPipelineMode::AsyncPipelined, GepaStalenessPolicy::Full) => {}
        (GepaPipelineMode::AsyncPipelined, policy) => {
            return Err(OptimizerError::Config(format!(
                "gepa.pipeline.staleness_policy = {policy:?} is reserved for flash_evolve; use full with async_pipelined"
            )));
        }
        (
            GepaPipelineMode::FlashEvolve,
            GepaStalenessPolicy::Full
            | GepaStalenessPolicy::Guarded
            | GepaStalenessPolicy::Reflective,
        ) => {}
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
    if config.speculative_completion.enabled {
        if !matches!(config.mode, GepaPipelineMode::FlashEvolve) {
            return Err(OptimizerError::Config(
                "gepa.pipeline.speculative_completion requires mode = \"flash_evolve\"".to_string(),
            ));
        }
        if !config.speculative_completion.alpha.is_finite()
            || config.speculative_completion.alpha <= 0.0
            || config.speculative_completion.alpha > 1.0
        {
            return Err(OptimizerError::Config(
                "gepa.pipeline.speculative_completion.alpha must be in (0, 1]".to_string(),
            ));
        }
    }
    if config.adaptive_stage_workers.min == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_stage_workers.min must be positive".to_string(),
        ));
    }
    if config.adaptive_stage_workers.max < config.adaptive_stage_workers.min {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_stage_workers.max must be >= min".to_string(),
        ));
    }
    if config.adaptive_stage_workers.backlog_threshold == 0 {
        return Err(OptimizerError::Config(
            "gepa.pipeline.adaptive_stage_workers.backlog_threshold must be positive".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_evolve_accepts_reflective_speculative_and_adaptive_stage_workers() {
        let mut config = GepaPipelineConfig::default();
        config.mode = GepaPipelineMode::FlashEvolve;
        config.staleness_policy = GepaStalenessPolicy::Reflective;
        config.speculative_completion.enabled = true;
        config.speculative_completion.alpha = 0.25;
        config.adaptive_stage_workers.enabled = true;

        validate_gepa_pipeline_config(&config).expect("flash_evolve full feature config");
    }

    #[test]
    fn chatgpt_proposer_allowlist_covers_the_gpt_5_6_codex_family() {
        for model in ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"] {
            validate_chatgpt_proposer_model(model)
                .unwrap_or_else(|error| panic!("{model} should be allowed: {error}"));
            // Case and surrounding whitespace are normalized before the lookup.
            validate_chatgpt_proposer_model(&format!("  {}  ", model.to_ascii_uppercase()))
                .unwrap_or_else(|error| panic!("{model} should normalize: {error}"));
        }
    }

    #[test]
    fn chatgpt_proposer_allowlist_still_rejects_unknown_models() {
        // The allowlist is the set Codex can serve under subscription auth, not
        // a naming pattern. A plausible-looking sibling must still fail closed.
        let error = validate_chatgpt_proposer_model("gpt-5.6-nonesuch")
            .expect_err("unknown gpt-5.6 variants must not be admitted");
        assert!(error.to_string().contains("is not allowed"));
        assert!(error.to_string().contains("gpt-5.6-luna"));
    }

    #[test]
    fn gepa_episode_toml_parses_delta_from_restart_limits() {
        let gepa: GepaConfig = toml::from_str(
            r#"
max_generations = 2

[episode]
proposer_rounds = 3
max_rollouts = 200
max_wall_seconds = 600
max_spend_usd = 0.5
skip_heldout = true
"#,
        )
        .expect("parse [gepa.episode]");
        assert_eq!(gepa.episode.proposer_rounds, Some(3));
        assert_eq!(gepa.episode.max_rollouts, Some(200));
        assert_eq!(gepa.episode.max_wall_seconds, Some(600));
        assert_eq!(gepa.episode.max_spend_usd, Some(0.5));
        assert!(gepa.episode.skip_heldout);
        validate_gepa_episode_config(&gepa.episode).expect("episode config");
    }

    #[test]
    fn pipeline_toml_accepts_background_execution_keys() {
        // `GepaPipelineConfig` is `deny_unknown_fields`, so a cookbook TOML
        // carrying these keys would hard-fail if they were not declared.
        let config: GepaPipelineConfig = toml::from_str(
            r#"
mode = "flash_evolve"
staleness_policy = "full"
max_in_flight_candidates = 8
background_execution = true
background_workers = 12
"#,
        )
        .expect("background execution keys should deserialize");
        assert_eq!(config.mode, GepaPipelineMode::FlashEvolve);
        assert_eq!(config.background_execution, Some(true));
        assert_eq!(config.background_workers, Some(12));
        validate_gepa_pipeline_config(&config).expect("config should validate");
    }

    #[test]
    fn pipeline_toml_omitting_background_keys_leaves_the_engine_default() {
        let config: GepaPipelineConfig =
            toml::from_str("mode = \"flash_evolve\"\n").expect("minimal pipeline config");
        assert_eq!(config.background_execution, None);
        assert_eq!(config.background_workers, None);
    }

    #[test]
    fn speculative_completion_requires_flash_evolve_mode() {
        let mut config = GepaPipelineConfig::default();
        config.mode = GepaPipelineMode::AsyncPipelined;
        config.speculative_completion.enabled = true;

        let error = validate_gepa_pipeline_config(&config)
            .expect_err("speculative completion should reject async_pipelined");

        assert!(error
            .to_string()
            .contains("speculative_completion requires mode"));
    }

    #[test]
    fn sync_serial_rejects_staleness_policy() {
        let mut config = GepaPipelineConfig::default();
        config.mode = GepaPipelineMode::SyncSerial;
        config.staleness_policy = GepaStalenessPolicy::Guarded;

        let error = validate_gepa_pipeline_config(&config)
            .expect_err("sync_serial should reject guarded staleness");

        assert!(error.to_string().contains("incompatible with sync_serial"));
    }

    #[test]
    fn adaptive_stage_worker_bounds_are_validated() {
        let mut config = GepaPipelineConfig::default();
        config.adaptive_stage_workers.min = 4;
        config.adaptive_stage_workers.max = 2;

        let error =
            validate_gepa_pipeline_config(&config).expect_err("max below min should be invalid");

        assert!(error
            .to_string()
            .contains("adaptive_stage_workers.max must be >= min"));
    }

    #[test]
    fn pipeline_toml_accepts_combee_as_flash_evolve_alias() {
        let config: GepaPipelineConfig =
            toml::from_str("mode = \"combee\"\n").expect("combee alias");
        assert_eq!(config.mode, GepaPipelineMode::FlashEvolve);
    }

    #[test]
    fn operator_toml_defaults_off_and_parses_opt_in_blocks() {
        let gepa: GepaConfig = toml::from_str("").expect("empty gepa");
        assert!(!gepa.operator.manderqueue.enabled);
        assert!(!gepa.operator.scratchpad.enabled);
        assert!(!gepa.operator.hypotheses.enabled);
        assert!(!gepa.operator.mcp_agent.enabled);
        assert_eq!(gepa.operator.reward.exploration_reduce, "mean");
        validate_gepa_operator_config(&gepa.operator).expect("defaults");

        let gepa: GepaConfig = toml::from_str(
            r#"
[operator.scratchpad]
enabled = true
shared = true

[operator.hypotheses]
enabled = true
max_open = 4

[operator.manderqueue]
enabled = true
base_url = "http://127.0.0.1:7400"
fail_closed = false
"#,
        )
        .expect("parse operator");
        assert!(gepa.operator.scratchpad.enabled);
        assert!(gepa.operator.hypotheses.enabled);
        assert_eq!(gepa.operator.hypotheses.max_open, 4);
        assert_eq!(
            gepa.operator.manderqueue.base_url.as_deref(),
            Some("http://127.0.0.1:7400")
        );
        validate_gepa_operator_config(&gepa.operator).expect("opt-in operator");
    }

    #[test]
    fn manderqueue_fail_closed_requires_base_url() {
        let gepa: GepaConfig = toml::from_str(
            r#"
[operator.manderqueue]
enabled = true
fail_closed = true
"#,
        )
        .expect("parse");
        let error = validate_gepa_operator_config(&gepa.operator)
            .expect_err("fail_closed manderqueue needs a URL");
        assert!(error.to_string().contains("base_url"));
    }
}
