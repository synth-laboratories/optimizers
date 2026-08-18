use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    fold_reported_cost, BudgetReservationRecord, ContainerClient, FailurePayload, OptimizerError,
    OptimizerJob, OptimizerJobKind, OptimizerJobStatus, PromptProgram, RequestCache, Result,
    RolloutResponse, RuntimeEffectInput, RuntimeEffectRecord, SynthOptimizerConfig, WorkspaceStore,
};

use crate::{
    cached_profiled_call_with_access, record_runtime_effect_completed, run_proposer,
    usage_completion_tokens, usage_prompt_tokens, CandidateRecord, ProposedCandidate,
    RuntimeEffectCompletionInput, UsageTotals, GEPA_ALGORITHM_ID,
};

pub const GEPA_RUNTIME_JOB_SCHEMA_VERSION: &str = "gepa_runtime_job.v1";
const DEFAULT_RUNTIME_WORKER_ID: &str = "gepa_inline_executor";
const DEFAULT_RUNTIME_LEASE_SECONDS: u64 = 3600;
const DEFAULT_ROLLOUT_HTTP_RETRIES: usize = 2;
const DEFAULT_ROLLOUT_RETRY_BACKOFF_MS: u64 = 200;
const DEEPSEEK_INPUT_USD_PER_MILLION: f64 = 0.27;
const DEEPSEEK_OUTPUT_USD_PER_MILLION: f64 = 1.10;

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
    RolloutBatch(RuntimeRolloutBatchOutcome),
}

#[derive(Clone, Debug)]
pub struct RuntimeRolloutBatchOutcome {
    pub outcomes: Vec<RuntimeRolloutOutcome>,
    pub failures: Vec<RuntimeRolloutFailure>,
}

impl std::ops::Deref for RuntimeRolloutBatchOutcome {
    type Target = Vec<RuntimeRolloutOutcome>;

    fn deref(&self) -> &Self::Target {
        &self.outcomes
    }
}

impl<'a> IntoIterator for &'a RuntimeRolloutBatchOutcome {
    type Item = &'a RuntimeRolloutOutcome;
    type IntoIter = std::slice::Iter<'a, RuntimeRolloutOutcome>;

    fn into_iter(self) -> Self::IntoIter {
        self.outcomes.iter()
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeRolloutFailure {
    pub candidate_id: String,
    pub stage: String,
    pub example_id: String,
    pub failure: FailurePayload,
}

#[derive(Clone, Debug)]
pub enum RuntimeRolloutProgress {
    Started {
        active_workers: usize,
        semaphore_size: usize,
        queued: usize,
    },
    Completed {
        outcome: RuntimeRolloutOutcome,
        active_workers: usize,
        semaphore_size: usize,
        queued: usize,
    },
    Failed {
        failure: RuntimeRolloutFailure,
        active_workers: usize,
        semaphore_size: usize,
        queued: usize,
    },
}

#[derive(Clone, Debug)]
pub struct RuntimeProposerOutcome {
    pub response: Value,
    pub proposals: Vec<ProposedCandidate>,
    pub usage: UsageTotals,
    pub cost_usd: f64,
    /// Provider-reported or explicitly priced cost. `None` means unknown;
    /// `cost_usd` remains the internal budget accumulator only.
    pub reported_cost_usd: Option<f64>,
    pub backend: String,
    pub runtime_substrate: String,
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
    pub reported_cost_usd: Option<f64>,
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
    progress_observer: Option<&'a mut dyn FnMut(&RuntimeRolloutProgress) -> Result<()>>,
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

#[allow(clippy::too_many_arguments)]
pub fn execute_one_pending_optimizer_job_with_progress(
    workspace: &WorkspaceStore,
    cache: &mut RequestCache,
    config: &SynthOptimizerConfig,
    client: &ContainerClient,
    run_id: &str,
    job_id: &str,
    executor_config: RuntimeEffectExecutorConfig,
    progress_observer: &mut dyn FnMut(&RuntimeRolloutProgress) -> Result<()>,
) -> Result<RuntimeEffectOutcome> {
    let mut executor = GepaRuntimeExecutor {
        workspace,
        cache,
        config,
        client,
        executor_config,
        progress_observer: Some(progress_observer),
    };
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
            progress_observer: None,
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
                    self.schedule_runtime_retry_if_allowed(&running_job, &lease_id, &error)?
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
        let (usage, cost_usd, reported_cost_usd, rollout_count, mut metadata) =
            terminal_metadata(&outcome);
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
        persist_runtime_outcome_before_completion(self.workspace, run_id, job_id, &outcome)?;
        record_runtime_effect_completed(
            self.workspace,
            RuntimeEffectCompletionInput {
                planned: &running_effect,
                reservation: &reservation,
                status: "completed",
                cost_usd,
                reported_cost_usd,
                usage: &usage,
                rollout_count,
                failure: None,
                metadata,
            },
        )?;
        ensure_job_lease(self.workspace, run_id, &claimed.job_id, &lease_id)?;
        Ok(outcome)
    }

    fn schedule_runtime_retry_if_allowed(
        &self,
        job: &OptimizerJob,
        lease_id: &str,
        error: &OptimizerError,
    ) -> Result<Option<OptimizerError>> {
        if !is_retryable_runtime_job_error(job, error)
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
            "retryable runtime failure scheduled for retry job_id={} kind={} attempt={}/{} next_retry_at={}",
            updated.job_id,
            updated.kind.as_str(),
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
                let task_pool_rows = request
                    .get("task_pool_rows")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                run_proposer(
                    self.config,
                    &program,
                    &parent,
                    &candidates,
                    generation,
                    task_pool_rows,
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
        let proposals = proposed_candidates(&response)?;
        let mut usage = UsageTotals {
            proposer_calls: 1,
            ..Default::default()
        };
        if let Some(response_usage) = response.get("usage") {
            usage.add_usage_payload(response_usage);
        }
        let reported_cost_usd = response
            .get("usage")
            .and_then(|usage| usage.get("cost_usd"))
            .or_else(|| response.get("cost_usd"))
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .or_else(|| proposer_static_cost_usd(self.config, &usage, &response));
        let cost_usd = reported_cost_usd.unwrap_or(0.0);
        let backend = response
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or(self.config.proposer.backend.as_str())
            .to_string();
        let runtime_substrate = response
            .get("runtime_substrate")
            .and_then(Value::as_str)
            .unwrap_or(self.config.proposer.runtime_substrate.as_str())
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
                reported_cost_usd,
                backend,
                runtime_substrate,
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
            &self.config.policy.provider,
            &self.config.policy.model,
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
        let mut failures = Vec::new();
        let mut misses = Vec::new();
        for (index, rollout) in rollouts.into_iter().enumerate() {
            let cache_request = rollout_cache_request(&rollout.request);
            if let Some(entry) =
                self.cache
                    .find_equivalent(&cache_namespace, &cache_request, &cache_profile)?
            {
                match rollout_outcome_from_value(
                    rollout.candidate_id.clone(),
                    entry.response,
                    entry.cache_key,
                    true,
                    rollout.stage.clone(),
                    rollout.example_id.clone(),
                    &self.config.policy.provider,
                    &self.config.policy.model,
                ) {
                    Ok(outcome) => outcomes[index] = Some(outcome),
                    Err(error) => failures.push(RuntimeRolloutFailure {
                        candidate_id: rollout.candidate_id,
                        stage: rollout.stage,
                        example_id: rollout.example_id,
                        failure: FailurePayload::from_optimizer_error(&error),
                    }),
                }
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
        let miss_count = misses.len();
        let queue = Arc::new(Mutex::new(VecDeque::from(misses)));
        let (sender, receiver) = mpsc::channel();
        let worker_count = concurrency.min(miss_count);
        if let Some(observer) = self.progress_observer.as_deref_mut() {
            observer(&RuntimeRolloutProgress::Started {
                active_workers: worker_count,
                semaphore_size: worker_count,
                queued: miss_count.saturating_sub(worker_count),
            })?;
        }
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let queue = Arc::clone(&queue);
            let sender = sender.clone();
            let client = self.client.clone();
            let dispatch_config = dispatch_config.clone();
            handles.push(thread::spawn(move || loop {
                let miss = {
                    let mut queue = queue.lock().expect("rollout queue mutex poisoned");
                    queue.pop_front()
                };
                let Some(miss) = miss else {
                    break;
                };
                let started = Instant::now();
                let result = match dispatch_rollout_with_retries(
                    &client,
                    &miss.rollout.request,
                    &dispatch_config,
                ) {
                    Ok(response) => Ok((miss, response, started.elapsed().as_secs_f64())),
                    Err(error) => Err((miss, error)),
                };
                if sender.send(result).is_err() {
                    break;
                }
            }));
        }
        drop(sender);
        for completion_index in 0..miss_count {
            let result = receiver.recv().map_err(|_| {
                OptimizerError::Invariant("rollout worker pool stopped early".to_string())
            })?;
            let (miss, value, dispatch_wall_seconds) = match result {
                Ok(value) => value,
                Err((miss, error)) => {
                    let failure = RuntimeRolloutFailure {
                        candidate_id: miss.rollout.candidate_id,
                        stage: miss.rollout.stage,
                        example_id: miss.rollout.example_id,
                        failure: FailurePayload::from_optimizer_error(&error),
                    };
                    let remaining = miss_count.saturating_sub(completion_index + 1);
                    let active_workers = worker_count.min(remaining);
                    let queued = remaining.saturating_sub(active_workers);
                    if let Some(observer) = self.progress_observer.as_deref_mut() {
                        observer(&RuntimeRolloutProgress::Failed {
                            failure: failure.clone(),
                            active_workers,
                            semaphore_size: worker_count,
                            queued,
                        })?;
                    }
                    failures.push(failure);
                    continue;
                }
            };
            let mut outcome = match rollout_outcome_from_value(
                miss.rollout.candidate_id.clone(),
                value.clone(),
                miss.cache_key.clone(),
                false,
                miss.rollout.stage.clone(),
                miss.rollout.example_id.clone(),
                &self.config.policy.provider,
                &self.config.policy.model,
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    let failure = RuntimeRolloutFailure {
                        candidate_id: miss.rollout.candidate_id,
                        stage: miss.rollout.stage,
                        example_id: miss.rollout.example_id,
                        failure: FailurePayload::from_optimizer_error(&error),
                    };
                    let remaining = miss_count.saturating_sub(completion_index + 1);
                    let active_workers = worker_count.min(remaining);
                    let queued = remaining.saturating_sub(active_workers);
                    if let Some(observer) = self.progress_observer.as_deref_mut() {
                        observer(&RuntimeRolloutProgress::Failed {
                            failure: failure.clone(),
                            active_workers,
                            semaphore_size: worker_count,
                            queued,
                        })?;
                    }
                    failures.push(failure);
                    continue;
                }
            };
            self.cache.put_with_metadata(
                &cache_namespace,
                &miss.cache_key,
                &miss.cache_request,
                &value,
                &cache_profile,
                miss.rollout.cache_metadata,
            )?;
            outcome.dispatch_wall_seconds = Some(dispatch_wall_seconds);
            outcome.dispatch_chunk_index = Some(completion_index);
            outcome.dispatch_chunk_size = Some(worker_count);
            let remaining = miss_count.saturating_sub(completion_index + 1);
            let active_workers = worker_count.min(remaining);
            let queued = remaining.saturating_sub(active_workers);
            if let Some(observer) = self.progress_observer.as_deref_mut() {
                observer(&RuntimeRolloutProgress::Completed {
                    outcome: outcome.clone(),
                    active_workers,
                    semaphore_size: worker_count,
                    queued,
                })?;
            }
            outcomes[miss.index] = Some(outcome);
        }
        for handle in handles {
            handle.join().map_err(|_| {
                OptimizerError::Invariant("rollout worker thread panicked".to_string())
            })?;
        }

        let outcomes = outcomes.into_iter().flatten().collect::<Vec<_>>();
        Ok(RuntimeEffectOutcome::RolloutBatch(
            RuntimeRolloutBatchOutcome { outcomes, failures },
        ))
    }
}

fn proposer_static_cost_usd(
    config: &SynthOptimizerConfig,
    usage: &UsageTotals,
    response: &Value,
) -> Option<f64> {
    let provider = response
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or(config.proposer.provider.as_str())
        .trim()
        .to_ascii_lowercase();
    let model = response
        .get("model")
        .and_then(Value::as_str)
        .or(config.proposer.model.as_deref())
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if provider == "deepseek" || model.contains("deepseek") {
        if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
            return None;
        }
        return Some(
            usage.prompt_tokens as f64 * DEEPSEEK_INPUT_USD_PER_MILLION / 1_000_000.0
                + usage.completion_tokens as f64 * DEEPSEEK_OUTPUT_USD_PER_MILLION / 1_000_000.0,
        );
    }
    None
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
    config.gepa.pipeline.workers.rollout.max(1)
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
        OptimizerError::Http(_)
            | OptimizerError::Container(_)
            | OptimizerError::ContainerHttpStatus {
                status_code: 408 | 409 | 425 | 429 | 500..=599,
                ..
            }
            | OptimizerError::Failed(_)
            | OptimizerError::Json(_)
    )
}

fn is_retryable_runtime_job_error(job: &OptimizerJob, error: &OptimizerError) -> bool {
    match job.kind {
        OptimizerJobKind::Rollout => is_retryable_rollout_runtime_error(error),
        OptimizerJobKind::Proposer => matches!(error, OptimizerError::Proposer(_)),
        _ => false,
    }
}

#[cfg(test)]
mod runtime_job_retry_tests {
    use super::*;

    #[test]
    fn proposer_errors_are_retryable_runtime_job_errors() {
        let job = OptimizerJob::new("proposer-job", "run", OptimizerJobKind::Proposer);
        assert!(is_retryable_runtime_job_error(
            &job,
            &OptimizerError::Proposer("app-server stalled".to_string())
        ));
        assert!(!is_retryable_runtime_job_error(
            &job,
            &OptimizerError::Config("invalid proposer config".to_string())
        ));
    }
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
    provider: &str,
    model: &str,
) -> Result<RuntimeRolloutOutcome> {
    let mut value = value;
    normalize_rollout_cost(provider, model, &mut value);
    let typed_response = RolloutResponse::from_value(value.clone())?;
    typed_response.validate_for_gepa()?;
    let reward = typed_response.outcome_reward()?;
    let mut usage = UsageTotals::default();
    let mut reported_cost_usd = None;
    if let Some(response_usage) = value.get("usage") {
        usage.add_usage_payload(response_usage);
        reported_cost_usd = response_usage
            .get("cost_usd")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0);
    }
    let cost_usd = reported_cost_usd.unwrap_or(0.0);
    usage.rollout_calls = 1;
    Ok(RuntimeRolloutOutcome {
        candidate_id,
        response: value,
        typed_response,
        reward,
        usage,
        cost_usd,
        reported_cost_usd,
        cache_key,
        cache_hit,
        stage,
        example_id,
        dispatch_wall_seconds: None,
        dispatch_chunk_index: None,
        dispatch_chunk_size: None,
    })
}

/// Fill a missing provider cost only when the policy model has a pinned public
/// token price. Container adapters commonly return token counts but omit USD;
/// treating that omission as a real zero silently disables the hard cost gate.
/// Unknown models deliberately remain unpriced rather than inheriting a made-up
/// default rate.
fn normalize_rollout_cost(provider: &str, model: &str, response: &mut Value) {
    let Some(usage) = response.get_mut("usage").and_then(Value::as_object_mut) else {
        return;
    };
    if usage
        .get("cost_usd")
        .and_then(Value::as_f64)
        .is_some_and(|cost| cost.is_finite() && cost > 0.0)
    {
        return;
    }
    let Some((input_per_million, output_per_million, source)) =
        pinned_rollout_price(provider, model)
    else {
        return;
    };
    let prompt_tokens = usage_prompt_tokens(&Value::Object(usage.clone()));
    let completion_tokens = usage_completion_tokens(&Value::Object(usage.clone()));
    if prompt_tokens == 0 && completion_tokens == 0 {
        return;
    }
    let cost_usd = prompt_tokens as f64 * input_per_million / 1_000_000.0
        + completion_tokens as f64 * output_per_million / 1_000_000.0;
    usage.insert("cost_usd".to_string(), json!(cost_usd));
    usage.insert("cost_source".to_string(), json!(source));
    usage.insert(
        "cost_pricing".to_string(),
        json!({
            "input_usd_per_million": input_per_million,
            "output_usd_per_million": output_per_million,
        }),
    );
}

fn pinned_rollout_price(provider: &str, model: &str) -> Option<(f64, f64, &'static str)> {
    if !provider.trim().eq_ignore_ascii_case("openai") {
        return None;
    }
    match model.trim().to_ascii_lowercase().as_str() {
        "gpt-4.1-nano" => Some((0.10, 0.40, "openai_gpt_4_1_nano_static_price")),
        "gpt-4o-mini" => Some((0.15, 0.60, "openai_gpt_4o_mini_static_price")),
        _ => None,
    }
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
) -> (UsageTotals, f64, Option<f64>, u64, Map<String, Value>) {
    match outcome {
        RuntimeEffectOutcome::Proposer(outcome) => {
            let mut metadata = Map::new();
            metadata.insert("proposal_count".to_string(), json!(outcome.proposals.len()));
            metadata.insert("backend".to_string(), json!(&outcome.backend));
            metadata.insert("cache_hit".to_string(), json!(outcome.cache_hit));
            metadata.insert("cache_key".to_string(), json!(&outcome.cache_key));
            (
                outcome.usage.clone(),
                outcome.cost_usd,
                outcome.reported_cost_usd,
                0,
                metadata,
            )
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
            (
                outcome.usage.clone(),
                outcome.cost_usd,
                outcome.reported_cost_usd,
                1,
                metadata,
            )
        }
        RuntimeEffectOutcome::RolloutBatch(outcomes) => {
            let mut usage = UsageTotals::default();
            let mut cost_usd = 0.0;
            let mut cost_receipts = Vec::with_capacity(outcomes.len() + outcomes.failures.len());
            cost_receipts.extend(std::iter::repeat_n(None, outcomes.failures.len()));
            let mut cache_hits = 0usize;
            let mut stages = BTreeMap::<String, usize>::new();
            let mut dispatch_latencies = Vec::new();
            for outcome in outcomes {
                usage.merge(&outcome.usage);
                cost_usd += outcome.cost_usd;
                cost_receipts.push(outcome.reported_cost_usd);
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
            metadata.insert(
                "failed_rollout_count".to_string(),
                json!(outcomes.failures.len()),
            );
            metadata.insert(
                "failed_example_ids".to_string(),
                json!(outcomes
                    .failures
                    .iter()
                    .map(|failure| failure.example_id.as_str())
                    .collect::<Vec<_>>()),
            );
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
            (
                usage,
                cost_usd,
                fold_reported_cost(cost_receipts),
                outcomes.len() as u64,
                metadata,
            )
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

fn proposed_candidates(response: &Value) -> Result<Vec<ProposedCandidate>> {
    let proposals = response
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for (proposal_index, item) in proposals.into_iter().enumerate() {
        let default_evidence = response
            .get("manifest")
            .and_then(|manifest| manifest.get("evidence"))
            .cloned()
            .unwrap_or(Value::Null);
        let mut proposal = serde_json::from_value::<ProposedCandidate>(item.clone()).map_err(
            |source| {
                OptimizerError::Proposer(format!(
                    "runtime proposer proposal index={proposal_index} is not a valid proposal object: {source}"
                ))
            },
        )?;
        if proposal.evidence.is_null() {
            proposal.evidence = default_evidence;
        }
        if proposal.payload_map().is_empty() {
            return Err(OptimizerError::Proposer(format!(
                "runtime proposer proposal index={proposal_index} returned no mutable payload; shape={}",
                proposal.payload_shape_summary()
            )));
        }
        out.push(proposal);
    }
    Ok(out)
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

fn persist_runtime_outcome_before_completion(
    workspace: &WorkspaceStore,
    run_id: &str,
    job_id: &str,
    outcome: &RuntimeEffectOutcome,
) -> Result<()> {
    let stored = crate::stored_runtime_outcome(outcome)?;
    let mut job = workspace.optimizer_job(run_id, job_id)?;
    job.payload
        .insert("runtime_outcome".to_string(), serde_json::to_value(stored)?);
    workspace.record_optimizer_job(&job)
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

#[cfg(test)]
mod partial_batch_tests {
    use super::*;

    #[test]
    fn a_failed_child_stays_null_and_does_not_discard_successful_siblings() {
        let successful = RuntimeRolloutOutcome {
            candidate_id: "candidate".to_string(),
            response: json!({"reward": 0.75}),
            typed_response: RolloutResponse::from_value(json!({"reward": 0.75})).unwrap(),
            reward: 0.75,
            usage: UsageTotals::default(),
            cost_usd: 0.01,
            reported_cost_usd: Some(0.01),
            cache_key: "cache-success".to_string(),
            cache_hit: false,
            stage: "seed_full_train".to_string(),
            example_id: "train:0".to_string(),
            dispatch_wall_seconds: Some(1.0),
            dispatch_chunk_index: Some(0),
            dispatch_chunk_size: Some(2),
        };
        let failed = RuntimeRolloutFailure {
            candidate_id: "candidate".to_string(),
            stage: "seed_full_train".to_string(),
            example_id: "train:1".to_string(),
            failure: FailurePayload::from_optimizer_error(&OptimizerError::Container(
                "provider_connect_error".to_string(),
            )),
        };
        let outcome = RuntimeEffectOutcome::RolloutBatch(RuntimeRolloutBatchOutcome {
            outcomes: vec![successful],
            failures: vec![failed],
        });

        let (_, _, reported_cost, rollout_count, metadata) = terminal_metadata(&outcome);
        assert_eq!(rollout_count, 1);
        assert_eq!(reported_cost, None, "partial spend must remain unknown");
        assert_eq!(metadata.get("rollout_count"), Some(&json!(1)));
        assert_eq!(metadata.get("failed_rollout_count"), Some(&json!(1)));
        assert_eq!(
            metadata.get("failed_example_ids"),
            Some(&json!(["train:1"]))
        );
    }

    #[test]
    fn serial_pipeline_uses_its_configured_rollout_worker_bound() {
        let mut config = SynthOptimizerConfig::default();
        config.gepa.pipeline.workers.rollout = 10;
        assert_eq!(rollout_concurrency(&config), 10);
        config.gepa.pipeline.workers.rollout = 0;
        assert_eq!(rollout_concurrency(&config), 1);
    }
}

#[cfg(test)]
mod cost_tests {
    use super::*;

    #[test]
    fn prices_known_openai_rollout_when_container_omits_cost() {
        let mut response = json!({
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000
            }
        });
        normalize_rollout_cost("openai", "gpt-4.1-nano", &mut response);
        assert_eq!(response.pointer("/usage/cost_usd"), Some(&json!(0.5)));
        assert_eq!(
            response.pointer("/usage/cost_source"),
            Some(&json!("openai_gpt_4_1_nano_static_price"))
        );
    }

    #[test]
    fn preserves_positive_provider_reported_cost() {
        let mut response = json!({
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
                "cost_usd": 0.73
            }
        });
        normalize_rollout_cost("openai", "gpt-4.1-nano", &mut response);
        assert_eq!(response.pointer("/usage/cost_usd"), Some(&json!(0.73)));
        assert!(response.pointer("/usage/cost_source").is_none());
    }

    #[test]
    fn leaves_unknown_model_cost_unknown() {
        let mut response = json!({
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}
        });
        normalize_rollout_cost("openai", "future-model", &mut response);
        assert!(response.pointer("/usage/cost_usd").is_none());
    }
}
