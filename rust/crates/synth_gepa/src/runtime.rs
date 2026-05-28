use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    BudgetReservationRecord, ContainerClient, FailurePayload, GepaPipelineMode, OptimizerError,
    OptimizerJob, OptimizerJobKind, OptimizerJobStatus, PromptProgram, RequestCache, Result,
    RolloutResponse, RuntimeEffectInput, RuntimeEffectRecord, SynthOptimizerConfig, WorkspaceStore,
};

use crate::{
    cached_profiled_call_with_access, record_runtime_effect_completed, run_proposer,
    CandidateRecord, ProposedCandidate, RuntimeEffectCompletionInput, UsageTotals,
    GEPA_ALGORITHM_ID,
};

pub const GEPA_RUNTIME_JOB_SCHEMA_VERSION: &str = "gepa_runtime_job.v1";
const DEFAULT_RUNTIME_WORKER_ID: &str = "gepa_inline_executor";
const DEFAULT_RUNTIME_LEASE_SECONDS: u64 = 3600;
const DEFAULT_ROLLOUT_CONCURRENCY: usize = 128;
const DEFAULT_ROLLOUT_HTTP_RETRIES: usize = 2;
const DEFAULT_ROLLOUT_RETRY_BACKOFF_MS: u64 = 200;

#[derive(Clone, Debug)]
pub struct RuntimeEffectExecutorConfig {
    pub worker_id: String,
    pub lease_seconds: u64,
}

impl RuntimeEffectExecutorConfig {
    pub fn inline_default() -> Self {
        Self {
            worker_id: DEFAULT_RUNTIME_WORKER_ID.to_string(),
            lease_seconds: DEFAULT_RUNTIME_LEASE_SECONDS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueuedRuntimeEffect {
    pub effect: RuntimeEffectRecord,
    pub reservation: BudgetReservationRecord,
    pub job: OptimizerJob,
    pub dispatch: RuntimeEffectDispatchPayload,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeEffectDispatchPayload {
    pub schema_version: String,
    #[serde(flatten)]
    pub dispatch: RuntimeEffectDispatchKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "dispatch_kind", rename_all = "snake_case")]
pub enum RuntimeEffectDispatchKind {
    Proposer {
        cache_namespace: String,
        cache_profile: String,
        cache_metadata: Map<String, Value>,
        request: Value,
        generation: usize,
        parent_candidate_id: String,
        proposer_workspace_dir: String,
    },
    Rollout {
        cache_namespace: String,
        cache_profile: String,
        cache_metadata: Map<String, Value>,
        request: Value,
        candidate_id: String,
        stage: String,
        example_id: String,
        task_id: String,
    },
    RolloutBatch {
        cache_namespace: String,
        cache_profile: String,
        rollouts: Vec<RuntimeRolloutDispatchItem>,
    },
}

#[derive(Clone, Debug)]
pub struct RuntimeRolloutDispatchInput {
    pub cache_namespace: String,
    pub cache_profile: String,
    pub cache_metadata: Map<String, Value>,
    pub request: Value,
    pub candidate_id: String,
    pub stage: String,
    pub example_id: String,
    pub task_id: String,
}

#[derive(Clone, Debug)]
pub struct RuntimeRolloutBatchDispatchInput {
    pub cache_namespace: String,
    pub cache_profile: String,
    pub rollouts: Vec<RuntimeRolloutDispatchItem>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeRolloutDispatchItem {
    pub cache_metadata: Map<String, Value>,
    pub request: Value,
    pub candidate_id: String,
    pub stage: String,
    pub example_id: String,
    pub task_id: String,
}

impl RuntimeEffectDispatchPayload {
    pub fn proposer(
        cache_namespace: String,
        cache_profile: &str,
        cache_metadata: Map<String, Value>,
        request: Value,
        generation: usize,
        parent_candidate_id: String,
        proposer_workspace_dir: String,
    ) -> Self {
        Self {
            schema_version: GEPA_RUNTIME_JOB_SCHEMA_VERSION.to_string(),
            dispatch: RuntimeEffectDispatchKind::Proposer {
                cache_namespace,
                cache_profile: cache_profile.to_string(),
                cache_metadata,
                request,
                generation,
                parent_candidate_id,
                proposer_workspace_dir,
            },
        }
    }

    pub fn rollout(input: RuntimeRolloutDispatchInput) -> Self {
        Self {
            schema_version: GEPA_RUNTIME_JOB_SCHEMA_VERSION.to_string(),
            dispatch: RuntimeEffectDispatchKind::Rollout {
                cache_namespace: input.cache_namespace,
                cache_profile: input.cache_profile,
                cache_metadata: input.cache_metadata,
                request: input.request,
                candidate_id: input.candidate_id,
                stage: input.stage,
                example_id: input.example_id,
                task_id: input.task_id,
            },
        }
    }

    pub fn rollout_batch(input: RuntimeRolloutBatchDispatchInput) -> Self {
        Self {
            schema_version: GEPA_RUNTIME_JOB_SCHEMA_VERSION.to_string(),
            dispatch: RuntimeEffectDispatchKind::RolloutBatch {
                cache_namespace: input.cache_namespace,
                cache_profile: input.cache_profile,
                rollouts: input.rollouts,
            },
        }
    }

    pub fn from_job(job: &OptimizerJob) -> Result<Self> {
        let payload: RuntimeEffectDispatchPayload =
            serde_json::from_value(Value::Object(job.payload.clone())).map_err(|source| {
                OptimizerError::Invariant(format!(
                    "optimizer job {} has invalid GEPA runtime dispatch payload: {source}",
                    job.job_id
                ))
            })?;
        payload.validate_for_job(&job.job_id)?;
        Ok(payload)
    }

    fn validate_for_job(&self, job_id: &str) -> Result<()> {
        if self.schema_version != GEPA_RUNTIME_JOB_SCHEMA_VERSION {
            return Err(OptimizerError::Invariant(format!(
                "optimizer job {job_id} has unsupported GEPA runtime job schema_version {}",
                self.schema_version
            )));
        }
        match &self.dispatch {
            RuntimeEffectDispatchKind::Proposer {
                cache_namespace,
                cache_profile,
                proposer_workspace_dir,
                ..
            } => {
                require_non_empty(job_id, "cache_namespace", cache_namespace)?;
                require_non_empty(job_id, "cache_profile", cache_profile)?;
                require_non_empty(job_id, "proposer_workspace_dir", proposer_workspace_dir)?;
            }
            RuntimeEffectDispatchKind::Rollout {
                cache_namespace,
                cache_profile,
                candidate_id,
                stage,
                example_id,
                task_id,
                ..
            } => {
                require_non_empty(job_id, "cache_namespace", cache_namespace)?;
                require_non_empty(job_id, "cache_profile", cache_profile)?;
                require_non_empty(job_id, "candidate_id", candidate_id)?;
                require_non_empty(job_id, "stage", stage)?;
                require_non_empty(job_id, "example_id", example_id)?;
                require_non_empty(job_id, "task_id", task_id)?;
            }
            RuntimeEffectDispatchKind::RolloutBatch {
                cache_namespace,
                cache_profile,
                rollouts,
            } => {
                require_non_empty(job_id, "cache_namespace", cache_namespace)?;
                require_non_empty(job_id, "cache_profile", cache_profile)?;
                if rollouts.is_empty() {
                    return Err(OptimizerError::Invariant(format!(
                        "optimizer job {job_id} has empty rollout batch"
                    )));
                }
                for (idx, rollout) in rollouts.iter().enumerate() {
                    let prefix = format!("rollouts[{idx}]");
                    require_non_empty(
                        job_id,
                        &format!("{prefix}.candidate_id"),
                        &rollout.candidate_id,
                    )?;
                    require_non_empty(job_id, &format!("{prefix}.stage"), &rollout.stage)?;
                    require_non_empty(
                        job_id,
                        &format!("{prefix}.example_id"),
                        &rollout.example_id,
                    )?;
                    require_non_empty(job_id, &format!("{prefix}.task_id"), &rollout.task_id)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum RuntimeEffectOutcome {
    Proposer(Box<RuntimeProposerOutcome>),
    Rollout(Box<RuntimeRolloutOutcome>),
    RolloutBatch(Vec<RuntimeRolloutOutcome>),
}

#[derive(Clone, Debug)]
pub struct RuntimeProposerOutcome {
    pub response: Value,
    pub proposals: Vec<ProposedCandidate>,
    pub usage: UsageTotals,
    pub cost_usd: f64,
    pub backend: String,
    pub workspace: Option<String>,
    pub evidence_warnings: Vec<String>,
    pub cache_key: String,
    pub cache_hit: bool,
}

#[derive(Clone, Debug)]
pub struct RuntimeRolloutOutcome {
    pub candidate_id: String,
    pub response: Value,
    pub typed_response: RolloutResponse,
    pub reward: f64,
    pub usage: UsageTotals,
    pub cost_usd: f64,
    pub cache_key: String,
    pub cache_hit: bool,
    pub stage: String,
    pub example_id: String,
    pub dispatch_wall_seconds: Option<f64>,
    pub dispatch_chunk_index: Option<usize>,
    pub dispatch_chunk_size: Option<usize>,
}

pub struct GepaRuntimeExecutor<'a> {
    workspace: &'a WorkspaceStore,
    cache: &'a mut RequestCache,
    config: &'a SynthOptimizerConfig,
    client: &'a ContainerClient,
    executor_config: RuntimeEffectExecutorConfig,
}

pub fn execute_one_pending_optimizer_job_from_run_workspace(
    workspace: &WorkspaceStore,
    cache: &mut RequestCache,
    config: &SynthOptimizerConfig,
    client: &ContainerClient,
    run_id: &str,
    job_id: &str,
    executor_config: RuntimeEffectExecutorConfig,
) -> Result<RuntimeEffectOutcome> {
    let mut executor = GepaRuntimeExecutor::new(workspace, cache, config, client, executor_config);
    executor.execute_one_pending_optimizer_job(run_id, job_id)
}

impl<'a> GepaRuntimeExecutor<'a> {
    pub fn new(
        workspace: &'a WorkspaceStore,
        cache: &'a mut RequestCache,
        config: &'a SynthOptimizerConfig,
        client: &'a ContainerClient,
        executor_config: RuntimeEffectExecutorConfig,
    ) -> Self {
        Self {
            workspace,
            cache,
            config,
            client,
            executor_config,
        }
    }

    pub fn execute_queued_runtime_effect(
        &mut self,
        queued: &QueuedRuntimeEffect,
    ) -> Result<RuntimeEffectOutcome> {
        queued.dispatch.validate_for_job(&queued.job.job_id)?;
        self.execute_one_pending_optimizer_job(&queued.effect.run_id, &queued.job.job_id)
    }

    pub fn execute_one_pending_optimizer_job(
        &mut self,
        run_id: &str,
        job_id: &str,
    ) -> Result<RuntimeEffectOutcome> {
        let existing_job = self.workspace.optimizer_job(run_id, job_id)?;
        let runtime_effect_id = required_job_string(&existing_job, "runtime_effect_id")?;
        let budget_reservation_id = required_job_string(&existing_job, "budget_reservation_id")?;
        let planned = self.workspace.runtime_effect(run_id, &runtime_effect_id)?;
        let reservation = self
            .workspace
            .budget_reservation(run_id, &budget_reservation_id)?;
        let lease_id = runtime_lease_id(&self.executor_config.worker_id, job_id);
        let claimed = self
            .workspace
            .claim_optimizer_job(
                run_id,
                job_id,
                &lease_id,
                Some(&self.executor_config.worker_id),
                self.executor_config.lease_seconds,
            )?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "GEPA runtime executor could not claim pending optimizer job run_id={run_id} job_id={job_id}"
                ))
            })?;
        let running_job = self
            .workspace
            .mark_optimizer_job_running(
                run_id,
                job_id,
                &lease_id,
                self.executor_config.lease_seconds,
            )?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "GEPA runtime executor lost optimizer job lease before running run_id={run_id} job_id={job_id} lease_id={lease_id}"
                ))
            })?;
        let running_effect =
            record_runtime_effect_running(self.workspace, &planned, &reservation, job_id)?;
        let dispatch = match RuntimeEffectDispatchPayload::from_job(&running_job) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                return crate::fail_runtime_effect_and_return(
                    self.workspace,
                    &running_effect,
                    &reservation,
                    error,
                    "dispatch_payload_decode",
                );
            }
        };
        let dispatch_started = Instant::now();
        let outcome = match self.execute_dispatch(dispatch) {
            Ok(outcome) => outcome,
            Err(error) => {
                if let Some(retry) =
                    self.schedule_rollout_retry_if_allowed(&running_job, &lease_id, &error)?
                {
                    return Err(retry);
                }
                return crate::fail_runtime_effect_and_return(
                    self.workspace,
                    &running_effect,
                    &reservation,
                    error,
                    "runtime_dispatch_execute",
                );
            }
        };
        let wall_seconds = dispatch_started.elapsed().as_secs_f64();
        let (usage, cost_usd, rollout_count, mut metadata) = terminal_metadata(&outcome);
        metadata.insert("wall_seconds".to_string(), json!(wall_seconds));
        if let Some(estimated_serial_wall_seconds) = metadata
            .get("estimated_serial_wall_seconds")
            .and_then(Value::as_f64)
            .filter(|_| wall_seconds > 0.0)
        {
            metadata.insert(
                "estimated_effective_concurrency".to_string(),
                json!(estimated_serial_wall_seconds / wall_seconds),
            );
        }
        if rollout_count > 0 {
            metadata.insert(
                "avg_wall_seconds_per_rollout".to_string(),
                json!(wall_seconds / rollout_count as f64),
            );
            metadata.insert(
                "rollout_concurrency".to_string(),
                json!(rollout_concurrency(self.config).max(1)),
            );
            metadata.insert(
                "rollout_submission_mode".to_string(),
                json!(self.config.gepa.rollout_submission_mode),
            );
        }
        record_runtime_effect_completed(
            self.workspace,
            RuntimeEffectCompletionInput {
                planned: &running_effect,
                reservation: &reservation,
                status: "completed",
                cost_usd,
                usage: &usage,
                rollout_count,
                failure: None,
                metadata,
            },
        )?;
        ensure_job_lease(self.workspace, run_id, &claimed.job_id, &lease_id)?;
        Ok(outcome)
    }

    fn schedule_rollout_retry_if_allowed(
        &self,
        job: &OptimizerJob,
        lease_id: &str,
        error: &OptimizerError,
    ) -> Result<Option<OptimizerError>> {
        if !matches!(
            self.config.gepa.pipeline.mode,
            GepaPipelineMode::AsyncPipelined
        ) || !matches!(job.kind, OptimizerJobKind::Rollout)
            || !is_retryable_rollout_runtime_error(error)
            || job.attempt >= job.retry_policy.max_attempts
        {
            return Ok(None);
        }
        let failure = FailurePayload::from_optimizer_error(error);
        let backoff_seconds = job
            .retry_policy
            .backoff_seconds
            .saturating_mul(1_u64 << job.attempt.saturating_sub(1).min(8));
        let Some(updated) = self.workspace.schedule_optimizer_job_retry(
            &job.run_id,
            &job.job_id,
            lease_id,
            backoff_seconds.max(1),
            &failure,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(OptimizerError::Failed(format!(
            "retryable rollout runtime failure scheduled for retry job_id={} attempt={}/{} next_retry_at={}",
            updated.job_id,
            updated.attempt.saturating_add(1),
            updated.retry_policy.max_attempts,
            updated.next_retry_at.unwrap_or_else(|| "now".to_string())
        ))))
    }

    fn execute_dispatch(
        &mut self,
        dispatch: RuntimeEffectDispatchPayload,
    ) -> Result<RuntimeEffectOutcome> {
        match dispatch.dispatch {
            RuntimeEffectDispatchKind::Proposer {
                cache_namespace,
                cache_profile,
                cache_metadata,
                request,
                generation,
                proposer_workspace_dir,
                ..
            } => self.execute_proposer_dispatch(
                cache_namespace,
                cache_profile,
                cache_metadata,
                request,
                generation,
                proposer_workspace_dir,
            ),
            RuntimeEffectDispatchKind::Rollout {
                cache_namespace,
                cache_profile,
                cache_metadata,
                request,
                candidate_id,
                stage,
                example_id,
                ..
            } => self.execute_rollout_dispatch(
                cache_namespace,
                cache_profile,
                cache_metadata,
                request,
                candidate_id,
                stage,
                example_id,
            ),
            RuntimeEffectDispatchKind::RolloutBatch {
                cache_namespace,
                cache_profile,
                rollouts,
            } => self.execute_rollout_batch_dispatch(cache_namespace, cache_profile, rollouts),
        }
    }

    fn execute_proposer_dispatch(
        &mut self,
        cache_namespace: String,
        cache_profile: String,
        cache_metadata: Map<String, Value>,
        request: Value,
        generation: usize,
        proposer_workspace_dir: String,
    ) -> Result<RuntimeEffectOutcome> {
        let workspace_dir = PathBuf::from(&proposer_workspace_dir);
        let call = cached_profiled_call_with_access(
            self.cache,
            &cache_namespace,
            &request,
            &cache_profile,
            cache_metadata,
            || {
                let program: PromptProgram = required_request_value(&request, "program")?;
                let parent: CandidateRecord = required_request_value(&request, "parent")?;
                let candidates: Vec<CandidateRecord> =
                    required_request_value(&request, "candidates")?;
                let seed_pool_rows = request
                    .get("seed_pool_rows")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                run_proposer(
                    self.config,
                    &program,
                    &parent,
                    &candidates,
                    generation,
                    seed_pool_rows,
                    workspace_dir.clone(),
                )
            },
        )?;
        let mut response = call.value;
        if let Some(map) = response.as_object_mut() {
            map.insert(
                "workspace".to_string(),
                Value::String(workspace_dir.display().to_string()),
            );
        }
        let proposals = proposed_candidates(&response);
        let mut usage = UsageTotals {
            proposer_calls: 1,
            ..Default::default()
        };
        if let Some(response_usage) = response.get("usage") {
            usage.add_usage_payload(response_usage);
        }
        let cost_usd = response
            .get("usage")
            .and_then(|usage| usage.get("cost_usd"))
            .or_else(|| response.get("cost_usd"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let backend = response
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or(self.config.proposer.backend.as_str())
            .to_string();
        let workspace = response
            .get("workspace")
            .and_then(Value::as_str)
            .map(str::to_string);
        let evidence_warnings = response
            .get("evidence_warnings")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        Ok(RuntimeEffectOutcome::Proposer(Box::new(
            RuntimeProposerOutcome {
                response,
                proposals,
                usage,
                cost_usd,
                backend,
                workspace,
                evidence_warnings,
                cache_key: call.cache_key,
                cache_hit: call.cache_hit,
            },
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_rollout_dispatch(
        &mut self,
        cache_namespace: String,
        cache_profile: String,
        cache_metadata: Map<String, Value>,
        request: Value,
        candidate_id: String,
        stage: String,
        example_id: String,
    ) -> Result<RuntimeEffectOutcome> {
        let cache_request = rollout_cache_request(&request);
        let dispatch_config = RolloutDispatchConfig::from_config(self.config);
        let dispatch_started = Instant::now();
        let call = cached_profiled_call_with_access(
            self.cache,
            &cache_namespace,
            &cache_request,
            &cache_profile,
            cache_metadata,
            || dispatch_rollout_with_retries(self.client, &request, &dispatch_config),
        )?;
        let dispatch_wall_seconds = dispatch_started.elapsed().as_secs_f64();
        let mut outcome = rollout_outcome_from_value(
            candidate_id,
            call.value,
            call.cache_key,
            call.cache_hit,
            stage,
            example_id,
        )?;
        if !outcome.cache_hit {
            outcome.dispatch_wall_seconds = Some(dispatch_wall_seconds);
        }
        Ok(RuntimeEffectOutcome::Rollout(Box::new(outcome)))
    }

    fn execute_rollout_batch_dispatch(
        &mut self,
        cache_namespace: String,
        cache_profile: String,
        rollouts: Vec<RuntimeRolloutDispatchItem>,
    ) -> Result<RuntimeEffectOutcome> {
        let mut outcomes: Vec<Option<RuntimeRolloutOutcome>> = vec![None; rollouts.len()];
        let mut misses = Vec::new();
        for (index, rollout) in rollouts.into_iter().enumerate() {
            let cache_request = rollout_cache_request(&rollout.request);
            if let Some(entry) =
                self.cache
                    .find_equivalent(&cache_namespace, &cache_request, &cache_profile)?
            {
                outcomes[index] = Some(rollout_outcome_from_value(
                    rollout.candidate_id,
                    entry.response,
                    entry.cache_key,
                    true,
                    rollout.stage,
                    rollout.example_id,
                )?);
                continue;
            }
            let cache_key = RequestCache::cache_key_with_profile(
                &cache_namespace,
                &cache_request,
                &cache_profile,
            );
            misses.push(PreparedRolloutMiss {
                index,
                rollout,
                cache_request,
                cache_key,
            });
        }

        let concurrency = rollout_concurrency(self.config).max(1);
        let dispatch_config = RolloutDispatchConfig::from_config(self.config);
        for (chunk_index, chunk) in misses.chunks(concurrency).enumerate() {
            let mut handles = Vec::with_capacity(chunk.len());
            for miss in chunk.iter().cloned() {
                let client = self.client.clone();
                let dispatch_config = dispatch_config.clone();
                handles.push(thread::spawn(move || {
                    let started = Instant::now();
                    let response = dispatch_rollout_with_retries(
                        &client,
                        &miss.rollout.request,
                        &dispatch_config,
                    )?;
                    Ok::<_, OptimizerError>((miss, response, started.elapsed().as_secs_f64()))
                }));
            }
            let chunk_size = chunk.len();
            for handle in handles {
                let (miss, value, dispatch_wall_seconds) = handle.join().map_err(|_| {
                    OptimizerError::Invariant("rollout worker thread panicked".to_string())
                })??;
                self.cache.put_with_metadata(
                    &cache_namespace,
                    &miss.cache_key,
                    &miss.cache_request,
                    &value,
                    &cache_profile,
                    miss.rollout.cache_metadata.clone(),
                )?;
                let mut outcome = rollout_outcome_from_value(
                    miss.rollout.candidate_id,
                    value,
                    miss.cache_key,
                    false,
                    miss.rollout.stage,
                    miss.rollout.example_id,
                )?;
                outcome.dispatch_wall_seconds = Some(dispatch_wall_seconds);
                outcome.dispatch_chunk_index = Some(chunk_index);
                outcome.dispatch_chunk_size = Some(chunk_size);
                outcomes[miss.index] = Some(outcome);
            }
        }

        let outcomes = outcomes
            .into_iter()
            .enumerate()
            .map(|(index, outcome)| {
                outcome.ok_or_else(|| {
                    OptimizerError::Invariant(format!(
                        "rollout batch finished without outcome for index {index}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RuntimeEffectOutcome::RolloutBatch(outcomes))
    }
}

#[derive(Clone, Debug)]
struct PreparedRolloutMiss {
    index: usize,
    rollout: RuntimeRolloutDispatchItem,
    cache_request: Value,
    cache_key: String,
}

#[derive(Clone, Debug)]
struct RolloutDispatchConfig {
    submission_mode: String,
    poll_interval: Duration,
    async_timeout: Duration,
    http_retries: usize,
    retry_backoff: Duration,
}

impl RolloutDispatchConfig {
    fn from_config(config: &SynthOptimizerConfig) -> Self {
        Self {
            submission_mode: config
                .gepa
                .rollout_submission_mode
                .trim()
                .to_ascii_lowercase(),
            poll_interval: Duration::from_millis(config.gepa.rollout_poll_interval_ms.max(1)),
            async_timeout: Duration::from_secs(config.gepa.rollout_async_timeout_seconds.max(1)),
            http_retries: env_usize("SYNTH_OPTIMIZERS_GEPA_ROLLOUT_HTTP_RETRIES")
                .unwrap_or(DEFAULT_ROLLOUT_HTTP_RETRIES)
                .min(10),
            retry_backoff: Duration::from_millis(
                env_u64("SYNTH_OPTIMIZERS_GEPA_ROLLOUT_RETRY_BACKOFF_MS")
                    .unwrap_or(DEFAULT_ROLLOUT_RETRY_BACKOFF_MS),
            ),
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn env_u64(name: &str) -> Option<u64> {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn rollout_concurrency(config: &SynthOptimizerConfig) -> usize {
    if matches!(config.gepa.pipeline.mode, GepaPipelineMode::AsyncPipelined) {
        return config.gepa.pipeline.workers.rollout.max(1);
    }
    env::var("GEPA_ROLLOUT_CONCURRENCY")
        .or_else(|_| env::var("SYNTH_OPTIMIZERS_MAX_CONCURRENT_ROLLOUTS"))
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_ROLLOUT_CONCURRENCY)
}

fn rollout_cache_request(request: &Value) -> Value {
    let mut cache_request = request.clone();
    if let Some(map) = cache_request.as_object_mut() {
        map.remove("submission_mode");
    }
    cache_request
}

fn dispatch_rollout(
    client: &ContainerClient,
    request: &Value,
    config: &RolloutDispatchConfig,
) -> Result<Value> {
    match config.submission_mode.as_str() {
        "sync" => {
            let response = client.rollout_typed(request)?;
            Ok(serde_json::to_value(response)?)
        }
        "async" => dispatch_async_rollout(client, request, config),
        other => Err(OptimizerError::Config(format!(
            "unsupported GEPA rollout submission mode {other:?}"
        ))),
    }
}

fn dispatch_rollout_with_retries(
    client: &ContainerClient,
    request: &Value,
    config: &RolloutDispatchConfig,
) -> Result<Value> {
    let max_attempts = config.http_retries.saturating_add(1);
    let mut attempt = 0usize;
    loop {
        match dispatch_rollout(client, request, config) {
            Ok(value) => return Ok(value),
            Err(error)
                if is_retryable_rollout_dispatch_error(&error) && attempt + 1 < max_attempts =>
            {
                attempt += 1;
                thread::sleep(config.retry_backoff.saturating_mul(attempt as u32));
            }
            Err(error) => return Err(error),
        }
    }
}

fn is_retryable_rollout_dispatch_error(error: &OptimizerError) -> bool {
    matches!(error, OptimizerError::Http(_))
}

fn is_retryable_rollout_runtime_error(error: &OptimizerError) -> bool {
    matches!(
        error,
        OptimizerError::Http(_) | OptimizerError::Container(_) | OptimizerError::Failed(_)
    )
}

fn dispatch_async_rollout(
    client: &ContainerClient,
    request: &Value,
    config: &RolloutDispatchConfig,
) -> Result<Value> {
    let mut async_request = request.clone();
    let Some(map) = async_request.as_object_mut() else {
        return Err(OptimizerError::Invariant(
            "GEPA rollout request must be a JSON object".to_string(),
        ));
    };
    map.insert(
        "submission_mode".to_string(),
        Value::String("async".to_string()),
    );
    let initial = client.rollout(&async_request)?;
    if is_terminal_rollout_success(&initial) {
        let response = RolloutResponse::from_value(initial.clone())?;
        response.validate_for_gepa()?;
        return Ok(initial);
    }
    if is_terminal_rollout_failure(&initial) {
        return Err(OptimizerError::Container(format!(
            "async rollout submission finished with status {:?}: {}",
            rollout_status(&initial),
            rollout_status_detail(&initial)
        )));
    }
    ensure_active_rollout_status(&initial, "async rollout submission")?;
    let rollout_id = rollout_id_from_payload(&initial)?;
    let deadline = Instant::now() + config.async_timeout;
    loop {
        let state = client.rollout_state(&rollout_id)?;
        if is_terminal_rollout_success(&state) {
            let record = client.rollout_record(&rollout_id)?;
            let response = RolloutResponse::from_value(record.clone())?;
            response.validate_for_gepa()?;
            return Ok(record);
        }
        if is_terminal_rollout_failure(&state) {
            return Err(OptimizerError::Container(format!(
                "async rollout {rollout_id} finished with status {:?}: {}",
                rollout_status(&state),
                rollout_status_detail(&state)
            )));
        }
        ensure_active_rollout_status(&state, &format!("async rollout {rollout_id}"))?;
        let now = Instant::now();
        if now >= deadline {
            let terminate_result = match client.rollout_terminate(&rollout_id, "gepa_async_timeout")
            {
                Ok(_) => "terminate requested".to_string(),
                Err(error) => format!("terminate failed: {error}"),
            };
            return Err(OptimizerError::Container(format!(
                "async rollout {rollout_id} timed out after {} seconds; {terminate_result}",
                config.async_timeout.as_secs()
            )));
        }
        thread::sleep(
            config
                .poll_interval
                .min(deadline.saturating_duration_since(now)),
        );
    }
}

fn rollout_id_from_payload(value: &Value) -> Result<String> {
    value
        .get("rollout_id")
        .or_else(|| value.get("trace_correlation_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|rollout_id| !rollout_id.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            OptimizerError::Container("async /rollout response must include rollout_id".to_string())
        })
}

fn rollout_status(value: &Value) -> String {
    value
        .get("status")
        .or_else(|| value.get("state"))
        .or_else(|| value.get("phase"))
        .or_else(|| value.get("success_status"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

fn rollout_status_detail(value: &Value) -> String {
    value
        .get("status_detail")
        .or_else(|| value.get("detail"))
        .or_else(|| value.get("error"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

fn is_terminal_rollout_success(value: &Value) -> bool {
    matches!(
        rollout_status(value).as_str(),
        "completed" | "success" | "succeeded" | "ok" | "done"
    )
}

fn is_terminal_rollout_failure(value: &Value) -> bool {
    matches!(
        rollout_status(value).as_str(),
        "failed"
            | "error"
            | "cancelled"
            | "canceled"
            | "terminated"
            | "expired"
            | "timeout"
            | "timed_out"
    )
}

fn ensure_active_rollout_status(value: &Value, label: &str) -> Result<()> {
    if matches!(
        rollout_status(value).as_str(),
        "queued" | "pending" | "running" | "in_progress" | "starting" | "submitted" | "paused"
    ) {
        return Ok(());
    }
    Err(OptimizerError::Container(format!(
        "{label} returned unsupported non-terminal status {:?}: {}",
        rollout_status(value),
        rollout_status_detail(value)
    )))
}

fn rollout_outcome_from_value(
    candidate_id: String,
    value: Value,
    cache_key: String,
    cache_hit: bool,
    stage: String,
    example_id: String,
) -> Result<RuntimeRolloutOutcome> {
    let typed_response = RolloutResponse::from_value(value.clone())?;
    typed_response.validate_for_gepa()?;
    let reward = typed_response.outcome_reward()?;
    let mut usage = UsageTotals::default();
    let mut cost_usd = 0.0;
    if let Some(response_usage) = value.get("usage") {
        usage.add_usage_payload(response_usage);
        cost_usd = response_usage
            .get("cost_usd")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
    }
    usage.rollout_calls = 1;
    Ok(RuntimeRolloutOutcome {
        candidate_id,
        response: value,
        typed_response,
        reward,
        usage,
        cost_usd,
        cache_key,
        cache_hit,
        stage,
        example_id,
        dispatch_wall_seconds: None,
        dispatch_chunk_index: None,
        dispatch_chunk_size: None,
    })
}

fn record_runtime_effect_running(
    workspace: &WorkspaceStore,
    planned: &RuntimeEffectRecord,
    reservation: &BudgetReservationRecord,
    job_id: &str,
) -> Result<RuntimeEffectRecord> {
    let mut metadata = planned.metadata.clone();
    metadata.insert("runtime_executor".to_string(), json!("gepa"));
    metadata.insert("algorithm_id".to_string(), json!(GEPA_ALGORITHM_ID));
    let running = RuntimeEffectRecord::from_input(RuntimeEffectInput {
        run_id: &planned.run_id,
        effect_kind: &planned.effect_kind,
        lane: &planned.lane,
        status: "running",
        subject_type: &planned.subject_type,
        subject_id: &planned.subject_id,
        idempotency_key: &planned.idempotency_key,
        cache_key: planned.cache_key.clone(),
        job_id: Some(job_id.to_string()),
        budget_reservation_id: Some(reservation.budget_reservation_id.clone()),
        attempt: planned.attempt,
        failure_class: None,
        payload: planned.payload.clone(),
        metadata,
    });
    workspace.record_runtime_effect(&running)?;
    Ok(running)
}

fn terminal_metadata(
    outcome: &RuntimeEffectOutcome,
) -> (UsageTotals, f64, u64, Map<String, Value>) {
    match outcome {
        RuntimeEffectOutcome::Proposer(outcome) => {
            let mut metadata = Map::new();
            metadata.insert("proposal_count".to_string(), json!(outcome.proposals.len()));
            metadata.insert("backend".to_string(), json!(&outcome.backend));
            metadata.insert("cache_hit".to_string(), json!(outcome.cache_hit));
            metadata.insert("cache_key".to_string(), json!(&outcome.cache_key));
            (outcome.usage.clone(), outcome.cost_usd, 0, metadata)
        }
        RuntimeEffectOutcome::Rollout(outcome) => {
            let mut metadata = Map::new();
            metadata.insert("cache_hit".to_string(), json!(outcome.cache_hit));
            metadata.insert("cache_key".to_string(), json!(&outcome.cache_key));
            metadata.insert("reward".to_string(), json!(outcome.reward));
            metadata.insert("stage".to_string(), json!(&outcome.stage));
            metadata.insert("example_id".to_string(), json!(&outcome.example_id));
            if let Some(dispatch_wall_seconds) = outcome.dispatch_wall_seconds {
                metadata.insert(
                    "uncached_dispatch_wall_seconds".to_string(),
                    json!(dispatch_wall_seconds),
                );
                metadata.insert(
                    "estimated_serial_wall_seconds".to_string(),
                    json!(dispatch_wall_seconds),
                );
            }
            (outcome.usage.clone(), outcome.cost_usd, 1, metadata)
        }
        RuntimeEffectOutcome::RolloutBatch(outcomes) => {
            let mut usage = UsageTotals::default();
            let mut cost_usd = 0.0;
            let mut cache_hits = 0usize;
            let mut stages = BTreeMap::<String, usize>::new();
            let mut dispatch_latencies = Vec::new();
            for outcome in outcomes {
                usage.merge(&outcome.usage);
                cost_usd += outcome.cost_usd;
                if outcome.cache_hit {
                    cache_hits += 1;
                } else if let Some(dispatch_wall_seconds) = outcome.dispatch_wall_seconds {
                    dispatch_latencies.push(dispatch_wall_seconds);
                }
                *stages.entry(outcome.stage.clone()).or_insert(0) += 1;
            }
            dispatch_latencies.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut metadata = Map::new();
            metadata.insert("rollout_count".to_string(), json!(outcomes.len()));
            metadata.insert("cache_hits".to_string(), json!(cache_hits));
            metadata.insert(
                "cache_misses".to_string(),
                json!(outcomes.len().saturating_sub(cache_hits)),
            );
            metadata.insert("stages".to_string(), json!(stages));
            if !dispatch_latencies.is_empty() {
                metadata.insert(
                    "uncached_latency_p50_seconds".to_string(),
                    json!(percentile_sorted(&dispatch_latencies, 0.50)),
                );
                metadata.insert(
                    "uncached_latency_p95_seconds".to_string(),
                    json!(percentile_sorted(&dispatch_latencies, 0.95)),
                );
                metadata.insert(
                    "uncached_latency_max_seconds".to_string(),
                    json!(dispatch_latencies.last().copied().unwrap_or(0.0)),
                );
                metadata.insert(
                    "estimated_serial_wall_seconds".to_string(),
                    json!(dispatch_latencies.iter().sum::<f64>()),
                );
            }
            (usage, cost_usd, outcomes.len() as u64, metadata)
        }
    }
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index =
        ((values.len().saturating_sub(1)) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[index.min(values.len().saturating_sub(1))]
}

fn proposed_candidates(response: &Value) -> Vec<ProposedCandidate> {
    let proposals = response
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for item in proposals {
        let default_evidence = response
            .get("manifest")
            .and_then(|manifest| manifest.get("evidence"))
            .cloned()
            .unwrap_or(Value::Null);
        let proposal_type = item
            .get("proposal_type")
            .and_then(Value::as_str)
            .unwrap_or("frontier_variation")
            .to_string();
        let parent_candidate_ids = item
            .get("parent_candidate_ids")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let rationale = item
            .get("rationale")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let evidence = item.get("evidence").cloned().unwrap_or(default_evidence);
        let candidate = item
            .get("candidate")
            .or_else(|| item.get("proposed_payload"))
            .cloned()
            .unwrap_or_else(|| item.clone());
        let lever_bundle = item
            .get("lever_bundle")
            .or_else(|| candidate.get("lever_bundle"))
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .unwrap_or(None);
        let Some(map) = candidate.as_object() else {
            continue;
        };
        let mut payload = BTreeMap::new();
        for (key, value) in map {
            if let Some(text) = value.as_str() {
                payload.insert(key.clone(), text.to_string());
            }
        }
        if !payload.is_empty() {
            let mut metadata = Map::new();
            if let Some(object) = item.as_object() {
                for (key, value) in object {
                    if !matches!(
                        key.as_str(),
                        "candidate"
                            | "proposed_payload"
                            | "proposal_type"
                            | "parent_candidate_ids"
                            | "rationale"
                            | "lever_bundle"
                            | "evidence"
                    ) {
                        metadata.insert(key.clone(), value.clone());
                    }
                }
            }
            out.push(ProposedCandidate {
                payload,
                lever_bundle,
                proposal_type,
                parent_candidate_ids,
                rationale,
                evidence,
                metadata,
                extra: Map::new(),
            });
        }
    }
    out
}

fn required_request_value<T: serde::de::DeserializeOwned>(
    request: &Value,
    field: &str,
) -> Result<T> {
    let value = request.get(field).cloned().ok_or_else(|| {
        OptimizerError::Invariant(format!("GEPA runtime proposer request missing {field}"))
    })?;
    serde_json::from_value(value).map_err(|source| {
        OptimizerError::Invariant(format!(
            "GEPA runtime proposer request field {field} has invalid payload: {source}"
        ))
    })
}

fn required_job_string(job: &OptimizerJob, field: &str) -> Result<String> {
    job.payload
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "optimizer job {} missing required GEPA runtime payload field {field}",
                job.job_id
            ))
        })
}

fn require_non_empty(job_id: &str, field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "optimizer job {job_id} has empty GEPA runtime payload field {field}"
        )));
    }
    Ok(())
}

fn ensure_job_lease(
    workspace: &WorkspaceStore,
    run_id: &str,
    job_id: &str,
    lease_id: &str,
) -> Result<()> {
    let job = workspace.optimizer_job(run_id, job_id)?;
    if job.lease_id.as_deref() != Some(lease_id) || job.status != OptimizerJobStatus::Completed {
        return Err(OptimizerError::Invariant(format!(
            "GEPA runtime executor lost optimizer job lease before terminal state run_id={run_id} job_id={job_id} lease_id={lease_id}"
        )));
    }
    Ok(())
}

fn runtime_lease_id(worker_id: &str, job_id: &str) -> String {
    format!("lease_{worker_id}_{job_id}_{}", now_millis())
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}
