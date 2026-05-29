use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use synth_optimizer_platform::limits::{
    BudgetCommitInput, BudgetCommitRecord, BudgetLimitBreach, BudgetReleaseInput,
    BudgetReleaseRecord, BudgetReservationInput, BudgetReservationRecord, RunLimitPolicy,
    RuntimeEffectAdmissionInput, RuntimeEffectAdmissionRecord, RuntimeEffectBudgetEstimate,
};
use synth_optimizer_platform::{
    dataset_row_identity, normalize_event_feed, ArtifactPaths, ArtifactRef, CacheMode,
    CacheProfileRecord, CandidateOverlay, CheckpointInput, CheckpointRecord,
    ConfiguredGepaRunLimits, ContainerClient, ContainerContractSnapshotInput,
    ContainerContractSnapshotRecord, DatasetResponse, DatasetRowsRequest, DatasetRowsResponse,
    DatasetSnapshotInput, DatasetSnapshotRecord, DiskBudget, EvaluationCacheRecord,
    EvaluationCacheRecordInput, EventStreamRecord, EventWriter, EvidenceFrame, FailurePayload,
    GepaBatchSamplerConfig, GepaCandidateSelectorConfig, GepaObjectiveAcceptanceConfig,
    GepaPipelineMode, GepaRunResult, LeverBundle, LeverKind, LeverManifest,
    ManagedContainerProcess, MaterializationRecord, MaterializationRecordInput, ObjectiveScore,
    ObjectiveSetRecord, ObjectiveSpec, OptimizerError, OptimizerJob, OptimizerJobKind,
    OptimizerJobStatus, OptimizerRunState, OptimizerStateMachine, OptimizerTransition,
    OptimizerTransitionTrigger, ParetoComparisonRecord, PlanLinkInput, PlanLinkRecord,
    PromptCandidatePayload, PromptProgram, PromptProgramSnapshotInput, PromptProgramSnapshotRecord,
    RequestCache, ResolvedRunConfigInput, ResolvedRunConfigRecord, Result, RetryPolicy,
    RolloutMaterializationIdentity, RunRegistry, RunRegistryEntry, RuntimeEffectInput,
    RuntimeEffectRecord, ScoreVectorRecord, SensorFrame, SensorScoreRecords, StopperStateInput,
    StopperStateRecord, SynthOptimizerConfig, UsageLedgerInput, UsageLedgerRecord, WorkspaceStore,
};

mod codex_app_server;
pub mod pipeline;
pub mod planner;
pub mod runtime;
pub mod service;

use pipeline::{GepaAsyncPipelinedPlan, GepaPipelineRuntimePlan};
use planner::{
    GepaAdaptiveRolloutConcurrencyAdjustment, GepaAsyncCandidatePartial, GepaAsyncLaneLease,
    GepaAsyncLaneWorkItem, GepaCursor, GepaCursorPhase, GepaRolloutChunkPartial,
    GEPA_CURSOR_CHECKPOINT_KIND,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub candidate_id: String,
    pub payload: BTreeMap<String, String>,
    pub lever_bundle: LeverBundle,
    pub parent_id: Option<String>,
    pub source: String,
    pub status: String,
    pub minibatch_reward: Option<f64>,
    pub train_reward: Option<f64>,
    pub heldout_reward: Option<f64>,
    pub minibatch_scores: Vec<RolloutScore>,
    pub train_scores: Vec<RolloutScore>,
    #[serde(default)]
    pub sensor_frames: Vec<SensorFrame>,
    #[serde(default)]
    pub acceptance_score: Value,
    #[serde(default)]
    pub acceptance_metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub average_reward: f64,
    pub rollout_count: usize,
    pub usage: UsageTotals,
    pub cost_usd: f64,
    pub scores: Vec<RolloutScore>,
    pub sensor_frames: Vec<SensorFrame>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposerOutcome {
    pub proposals: Vec<ProposedCandidate>,
    pub usage: UsageTotals,
    pub cost_usd: f64,
    pub backend: String,
    pub workspace: Option<String>,
    #[serde(default)]
    pub evidence_warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProposedCandidate {
    #[serde(default)]
    pub payload: BTreeMap<String, String>,
    #[serde(default)]
    pub lever_bundle: Option<LeverBundle>,
    #[serde(default)]
    pub proposal_type: String,
    #[serde(default)]
    pub parent_candidate_ids: Vec<String>,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub evidence: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

impl ProposedCandidate {
    fn payload_map(&self) -> BTreeMap<String, String> {
        if !self.payload.is_empty() {
            return self.payload.clone();
        }
        if let Some(bundle) = &self.lever_bundle {
            let payload = bundle.to_prompt_payload();
            if !payload.is_empty() {
                return payload;
            }
        }
        self.extra
            .iter()
            .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
            .collect()
    }

    fn proposal_type_or_default(&self) -> String {
        let proposal_type = self.proposal_type.trim();
        if proposal_type.is_empty() {
            "frontier_variation".to_string()
        } else {
            proposal_type.to_string()
        }
    }

    fn metadata_value(&self) -> Value {
        let mut metadata = self.metadata.clone();
        metadata.insert(
            "proposal_type".to_string(),
            json!(self.proposal_type_or_default()),
        );
        metadata.insert(
            "parent_candidate_ids".to_string(),
            json!(self.parent_candidate_ids),
        );
        if let Some(bundle) = &self.lever_bundle {
            metadata.insert("lever_bundle".to_string(), json!(bundle));
        }
        metadata.insert("rationale".to_string(), json!(self.rationale));
        if !self.evidence.is_null() {
            metadata.insert("evidence".to_string(), self.evidence.clone());
        }
        if !self.extra.is_empty() {
            metadata.insert("raw_extra".to_string(), Value::Object(self.extra.clone()));
        }
        Value::Object(metadata)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutScore {
    pub example_id: String,
    pub seed: i64,
    pub reward: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptanceDecision {
    pub candidate_id: String,
    pub parent_id: String,
    pub accepted_minibatch: bool,
    pub accepted_full_train: bool,
    pub reason: String,
    pub candidate_minibatch_reward: f64,
    pub parent_minibatch_reward: f64,
    pub candidate_train_reward: Option<f64>,
    pub best_train_reward: f64,
    pub comparison_result: String,
    #[serde(default)]
    pub score: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrontierMember {
    pub candidate_id: String,
    pub parent_id: Option<String>,
    pub source: String,
    pub train_reward: f64,
    pub heldout_reward: Option<f64>,
}

#[derive(Clone, Debug)]
struct ParentSelectionDecision {
    candidate_index: usize,
    metadata: Value,
}

struct ProposerCall<'a> {
    client: &'a ContainerClient,
    workspace: &'a WorkspaceStore,
    cache: &'a mut RequestCache,
    cache_namespace: &'a str,
    config: &'a SynthOptimizerConfig,
    program: &'a PromptProgram,
    parent: &'a CandidateRecord,
    candidates: &'a [CandidateRecord],
    generation: usize,
    seed_pool_rows: Value,
    paths: &'a ArtifactPaths,
}

struct EvaluationCall<'a> {
    client: &'a ContainerClient,
    workspace: &'a WorkspaceStore,
    paths: &'a ArtifactPaths,
    cache: &'a mut RequestCache,
    cache_namespace: &'a str,
    config: &'a SynthOptimizerConfig,
    program: &'a PromptProgram,
    task_id: &'a str,
    objective_set: &'a ObjectiveSetRecord,
    candidate: &'a CandidateRecord,
    rows: &'a [Value],
    stage: &'a str,
    cancellation: Option<&'a GepaCancellationSource>,
}

struct CachedCallOutcome {
    value: Value,
    cache_key: String,
    cache_hit: bool,
}

struct ScoreVectorPreferenceInput<'a> {
    objective_set: &'a ObjectiveSetRecord,
    split: &'a str,
    evaluation_stage: &'a str,
    challenger: &'a ScoreVectorRecord,
    incumbent: &'a ScoreVectorRecord,
    accept_equal: bool,
    acceptance_criterion: Option<&'a str>,
    objective_acceptance: Option<&'a GepaObjectiveAcceptanceConfig>,
    margin: f64,
}

struct ScoreVectorPreference {
    preferred: bool,
    result: String,
    reason: String,
    score: Value,
    metadata: Map<String, Value>,
}

struct CandidateScoreVectorInput<'a> {
    objective_set: &'a ObjectiveSetRecord,
    candidate: &'a CandidateRecord,
    rows: &'a [Value],
    split: &'a str,
    source_stages: &'a [&'a str],
    evaluation_stage: &'a str,
}

struct HeldoutSelectionInput<'a> {
    candidates: &'a [CandidateRecord],
    evaluated_indices: &'a [usize],
    objective_set: &'a ObjectiveSetRecord,
    heldout_split: &'a str,
    heldout_rows: &'a [Value],
    train_split: &'a str,
    train_rows: &'a [Value],
    incumbent_idx: Option<usize>,
}

const ROLLOUT_CACHE_PROFILE: &str = "rollout_request";
const PROPOSER_CACHE_PROFILE: &str = "gepa_proposer";
const GEPA_ALGORITHM_ID: &str = "synth_gepa.v1";

struct StopperSnapshot<'a> {
    status: &'a str,
    reason: Option<&'a str>,
    generation: Option<usize>,
    candidate_id: Option<&'a str>,
    evaluation_stage: Option<&'a str>,
    rollout_count: usize,
    cost_usd: f64,
    metadata: Map<String, Value>,
}

struct CheckpointSnapshot<'a> {
    checkpoint_kind: &'a str,
    status: &'a str,
    reason: Option<&'a str>,
    generation: Option<usize>,
    candidate_id: Option<&'a str>,
    evaluation_stage: Option<&'a str>,
    best_candidate_id: Option<&'a str>,
    candidate_count: usize,
    frontier_count: usize,
    rollout_count: usize,
    cost_usd: f64,
    usage: Value,
    snapshot: Value,
    metadata: Map<String, Value>,
}

struct CheckpointSnapshotState<'a> {
    config: &'a SynthOptimizerConfig,
    candidates: &'a [CandidateRecord],
    frontier: Vec<FrontierMember>,
    best_idx: Option<usize>,
    state_machine: &'a OptimizerStateMachine,
    rollout_count: usize,
    total_usage: &'a UsageTotals,
    total_cost: f64,
}

struct GepaRunContext {
    paths: ArtifactPaths,
    workspace: WorkspaceStore,
    registry: RunRegistry,
    events: EventWriter,
    state_machine: OptimizerStateMachine,
    cache: RequestCache,
    config: SynthOptimizerConfig,
    cache_mode: CacheMode,
    cache_namespace: String,
    container_process: Option<ManagedContainerProcess>,
    client: Option<ContainerClient>,
    program: Option<PromptProgram>,
    objective_set: Option<ObjectiveSetRecord>,
    train_rows: Vec<Value>,
    minibatch_rows: Vec<Value>,
    reflection_rows: Vec<Value>,
    heldout_rows: Vec<Value>,
    rollout_task_id: Option<String>,
}

struct GepaContainerInputs {
    _container_process: Option<ManagedContainerProcess>,
    client: ContainerClient,
    program: PromptProgram,
    objective_set: ObjectiveSetRecord,
    train_rows: Vec<Value>,
    minibatch_rows: Vec<Value>,
    reflection_rows: Vec<Value>,
    heldout_rows: Vec<Value>,
    rollout_task_id: String,
}

struct GepaCursorState<'a> {
    phase: GepaCursorPhase,
    generation: usize,
    proposal_index: usize,
    pending_job_id: Option<String>,
    pending_effect_id: Option<String>,
    pending_reservation_ids: Vec<String>,
    active_evaluation: Option<Value>,
    candidates: &'a [CandidateRecord],
    best_idx: Option<usize>,
    train_rows: &'a [Value],
    minibatch_rows: &'a [Value],
    reflection_rows: &'a [Value],
    heldout_rows: &'a [Value],
    program: &'a PromptProgram,
    objective_set: &'a ObjectiveSetRecord,
    rollout_task_id: &'a str,
    total_usage: &'a UsageTotals,
    total_cost: f64,
    rollout_count: usize,
    stopper_sequence: u64,
    state_machine: &'a OptimizerStateMachine,
    terminal_summary: Option<Value>,
    error_summary: Option<Value>,
    metadata: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GepaAdvanceMode {
    RunLoop,
    ServiceTick,
}

const ASYNC_PIPELINE_NOOP_SLEEP: Duration = Duration::from_millis(250);

#[derive(Clone, Debug)]
pub struct GepaAdvanceOutcome {
    pub action: planner::GepaTickAction,
    pub terminal: bool,
    pub result: Option<GepaRunResult>,
    pub message: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct GepaActiveEvaluation {
    #[serde(default)]
    stage: String,
    #[serde(default)]
    candidate_id: Option<String>,
    #[serde(default)]
    candidate_index: Option<usize>,
    #[serde(default)]
    generation: usize,
    #[serde(default)]
    proposal_index: usize,
    #[serde(default)]
    row_ids: Vec<String>,
    #[serde(default)]
    next_row_index: usize,
    #[serde(default)]
    planned_job_id: Option<String>,
    #[serde(default)]
    effect_id: Option<String>,
    #[serde(default)]
    reservation_id: Option<String>,
    #[serde(default)]
    heldout_candidate_index: Option<usize>,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    scores: Vec<RolloutScore>,
    #[serde(default)]
    sensor_frames: Vec<SensorFrame>,
    #[serde(default)]
    reward_sum: f64,
    #[serde(default)]
    usage: UsageTotals,
    #[serde(default)]
    cost_usd: f64,
    #[serde(default)]
    rollout_count: usize,
    #[serde(default)]
    parent_minibatch_reward: Option<f64>,
    #[serde(default)]
    decision: Option<AcceptanceDecision>,
    #[serde(default)]
    candidate_evaluations: Vec<GepaActiveCandidateEvaluation>,
}

impl GepaActiveEvaluation {
    fn is_rollout_stage(&self) -> bool {
        matches!(
            self.stage.as_str(),
            "seed_full_train"
                | "parent_minibatch_reference"
                | "candidate_minibatch"
                | "candidate_full_train"
                | "heldout"
        )
    }

    fn average_reward(&self) -> f64 {
        if self.rollout_count == 0 {
            0.0
        } else {
            self.reward_sum / self.rollout_count as f64
        }
    }

    fn is_group(&self) -> bool {
        !self.candidate_evaluations.is_empty()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct GepaActiveCandidateEvaluation {
    #[serde(default)]
    candidate_id: String,
    #[serde(default)]
    candidate_index: usize,
    #[serde(default)]
    generation: usize,
    #[serde(default)]
    proposal_index: usize,
    #[serde(default)]
    row_ids: Vec<String>,
    #[serde(default)]
    next_row_index: usize,
    #[serde(default)]
    heldout_candidate_index: Option<usize>,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    scores: Vec<RolloutScore>,
    #[serde(default)]
    sensor_frames: Vec<SensorFrame>,
    #[serde(default)]
    reward_sum: f64,
    #[serde(default)]
    usage: UsageTotals,
    #[serde(default)]
    cost_usd: f64,
    #[serde(default)]
    rollout_count: usize,
    #[serde(default)]
    parent_minibatch_reward: Option<f64>,
    #[serde(default)]
    decision: Option<AcceptanceDecision>,
}

impl GepaActiveCandidateEvaluation {
    fn average_reward(&self) -> f64 {
        if self.rollout_count == 0 {
            0.0
        } else {
            self.reward_sum / self.rollout_count as f64
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StoredRuntimeOutcome {
    Proposer {
        #[serde(default)]
        response: Value,
        proposals: Vec<ProposedCandidate>,
        usage: UsageTotals,
        cost_usd: f64,
        backend: String,
        workspace: Option<String>,
    },
    Rollout {
        response: Value,
        reward: f64,
        usage: UsageTotals,
        cost_usd: f64,
        cache_key: String,
        cache_hit: bool,
        stage: String,
        example_id: String,
        #[serde(default)]
        dispatch_wall_seconds: Option<f64>,
        #[serde(default)]
        dispatch_chunk_index: Option<usize>,
        #[serde(default)]
        dispatch_chunk_size: Option<usize>,
    },
    RolloutBatch {
        outcomes: Vec<StoredRolloutOutcome>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredRolloutOutcome {
    candidate_id: String,
    response: Value,
    reward: f64,
    usage: UsageTotals,
    cost_usd: f64,
    cache_key: String,
    cache_hit: bool,
    stage: String,
    example_id: String,
    #[serde(default)]
    dispatch_wall_seconds: Option<f64>,
    #[serde(default)]
    dispatch_chunk_index: Option<usize>,
    #[serde(default)]
    dispatch_chunk_size: Option<usize>,
}

#[derive(Clone, Debug)]
struct GepaRunState {
    cursor: GepaCursor,
    candidates: Vec<CandidateRecord>,
    best_idx: Option<usize>,
    proposal_queue: Vec<ProposedCandidate>,
    active_evaluation: Option<GepaActiveEvaluation>,
    heldout_candidate_index: usize,
    total_usage: UsageTotals,
    total_cost: f64,
    rollout_count: usize,
    usage_ledger: Vec<UsageLedgerRecord>,
    stopper_states: Vec<StopperStateRecord>,
    stopper_sequence: u64,
    checkpoint_sequence: u64,
}

struct GepaStepResources {
    client: ContainerClient,
    program: PromptProgram,
    objective_set: ObjectiveSetRecord,
    train_rows: Vec<Value>,
    minibatch_rows: Vec<Value>,
    reflection_rows: Vec<Value>,
    heldout_rows: Vec<Value>,
    rollout_task_id: String,
}

#[derive(Clone, Debug, Default)]
pub struct GepaExecutionOptions {
    pub cancellation: Option<GepaCancellationSource>,
}

#[derive(Clone, Debug)]
pub struct GepaCancellationSource {
    pub service_db_path: PathBuf,
    pub request_id: String,
    pub lease_id: Option<String>,
    pub lease_seconds: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageTotals {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub rollout_calls: u64,
    pub proposer_calls: u64,
}

impl UsageTotals {
    fn add_usage_payload(&mut self, usage: &Value) {
        self.prompt_tokens += usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.completion_tokens += usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.total_tokens += usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }

    fn merge(&mut self, other: &UsageTotals) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        self.rollout_calls += other.rollout_calls;
        self.proposer_calls += other.proposer_calls;
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RuntimeUsageSummary {
    policy: RuntimeUsageBucket,
    proposer: RuntimeUsageBucket,
    candidates: BTreeMap<String, RuntimeUsageBucket>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RuntimeUsageBucket {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    calls: u64,
    jobs: u64,
    cost_usd: f64,
    wall_seconds: f64,
}

impl RuntimeUsageBucket {
    fn add_usage_totals(&mut self, usage: &UsageTotals) {
        self.prompt_tokens += usage.prompt_tokens;
        self.completion_tokens += usage.completion_tokens;
        self.total_tokens += usage.total_tokens;
    }

    fn add_record(&mut self, fields: &Value) {
        if self.model.is_none() {
            self.model = fields
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        let usage = fields.get("usage").unwrap_or(&Value::Null);
        self.prompt_tokens += usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.completion_tokens += usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.total_tokens += usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .or_else(|| fields.get("total_tokens").and_then(Value::as_u64))
            .unwrap_or(0);
        self.cost_usd += fields
            .get("cost_usd")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        self.wall_seconds += fields
            .get("wall_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        self.jobs = self.jobs.saturating_add(1);
    }

    fn add_rollout_record(&mut self, fields: &Value) {
        self.add_record(fields);
        self.calls = self.calls.saturating_add(
            fields
                .get("rollout_count")
                .and_then(Value::as_u64)
                .unwrap_or(1),
        );
    }

    fn add_proposer_record(&mut self, fields: &Value) {
        self.add_record(fields);
        self.calls = self.calls.saturating_add(1);
    }

    fn merge_candidate_record(&mut self, fields: &Value) {
        if self.model.is_none() {
            self.model = fields
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        self.prompt_tokens += field_u64(fields, "prompt_tokens");
        self.completion_tokens += field_u64(fields, "completion_tokens");
        self.total_tokens += field_u64(fields, "total_tokens");
        self.calls = self.calls.saturating_add(field_u64(fields, "calls"));
        self.jobs = self.jobs.saturating_add(field_u64(fields, "jobs"));
        self.cost_usd += fields
            .get("cost_usd")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        self.wall_seconds += fields
            .get("wall_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
    }
}

fn runtime_usage_summary_from_events(records: &[EventStreamRecord]) -> RuntimeUsageSummary {
    let mut summary = RuntimeUsageSummary::default();
    for record in records {
        if record.event_type != "runtime.job.completed" {
            continue;
        }
        match record
            .fields
            .get("runtime_kind")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "proposer" => summary.proposer.add_proposer_record(&record.fields),
            "rollout" | "rollout_batch" => {
                summary.policy.add_rollout_record(&record.fields);
                if let Some(candidate_usage) = record
                    .fields
                    .get("candidate_usage")
                    .and_then(Value::as_object)
                {
                    for (candidate_id, fields) in candidate_usage {
                        summary
                            .candidates
                            .entry(candidate_id.clone())
                            .or_default()
                            .merge_candidate_record(fields);
                    }
                }
            }
            _ => {}
        }
    }
    summary
}

fn field_u64(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

pub fn execute_gepa_from_toml(path: impl AsRef<Path>) -> Result<GepaRunResult> {
    let config = SynthOptimizerConfig::from_toml_file(path)?;
    execute_gepa(config)
}

pub fn execute_gepa(config: SynthOptimizerConfig) -> Result<GepaRunResult> {
    execute_gepa_with_options(config, GepaExecutionOptions::default())
}

pub fn execute_gepa_from_toml_with_options(
    path: impl AsRef<Path>,
    options: GepaExecutionOptions,
) -> Result<GepaRunResult> {
    let config = SynthOptimizerConfig::from_toml_file(path)?;
    execute_gepa_with_options(config, options)
}

fn open_gepa_run_context(
    config: SynthOptimizerConfig,
    options: &GepaExecutionOptions,
) -> Result<GepaRunContext> {
    // Soft-limit gate: refuse the new run if `output_dir` already holds
    // more bytes than the configured soft floor. Fires before any
    // workspace, paths, or registry write so the launcher exits cleanly
    // with `synth_optimizer_disk_budget_exceeded` instead of partial-
    // initializing an unusable run.
    let disk_budget = DiskBudget::new(config.disk_budget.clone(), &config.run.output_dir)?;
    disk_budget.require_below_soft()?;
    let paths = ArtifactPaths::new(&config.run.output_dir, &config.run.run_id);
    paths.create()?;
    let cache_path = config
        .cache
        .path
        .clone()
        .unwrap_or_else(|| paths.run_dir.join("request_cache.sqlite"));
    let cache_mode = CacheMode::from(config.cache.mode);
    let cache_namespace = config
        .cache
        .namespace
        .clone()
        .unwrap_or_else(|| format!("gepa:{}", config.run.run_id));
    let workspace = WorkspaceStore::open(&paths.workspace_db_path)?;
    workspace.record_run_started(&paths, &config, cache_mode, &cache_namespace)?;
    record_initial_platform_snapshots(&workspace, &config, cache_mode, &cache_namespace, &paths)?;
    let is_resumed_run = workspace
        .latest_checkpoint(&config.run.run_id, GEPA_CURSOR_CHECKPOINT_KIND)?
        .is_some();
    let registry = RunRegistry::new(&paths.run_registry_path);
    registry.append(&RunRegistryEntry::started(
        &paths,
        &config,
        cache_mode,
        &cache_namespace,
    ))?;
    // Hard-limit gate: every emit checks the budget and refuses the
    // write when usage is at or above the hard floor, so the jsonl
    // never partial-appends under ENOSPC pressure. `with_disk_budget`
    // is a no-op when the budget itself is disabled, so we always
    // attach.
    let mut events = if options.cancellation.is_some() || is_resumed_run {
        EventWriter::append(&paths.event_feed_path)?
    } else {
        EventWriter::new(&paths.event_feed_path)?
    }
    .with_disk_budget(disk_budget.clone());
    let mut state_machine = OptimizerStateMachine::new(config.run.run_id.clone());
    transition_run(
        &workspace,
        &mut events,
        &mut state_machine,
        OptimizerRunState::Initializing,
        OptimizerTransitionTrigger::RunStarted,
        "GEPA run initializing",
        json!({"run_id": config.run.run_id}),
    )?;
    events.emit(
        "gepa.run.started",
        "GEPA run started",
        json!({
            "run_id": config.run.run_id,
            "container_url": config.container.url,
            "run_registry_path": paths.run_registry_path,
            "state": state_machine.state().as_str(),
        }),
    )?;
    let cache = RequestCache::open(cache_path, cache_mode)?;
    Ok(GepaRunContext {
        paths,
        workspace,
        registry,
        events,
        state_machine,
        cache,
        config,
        cache_mode,
        cache_namespace,
        container_process: None,
        client: None,
        program: None,
        objective_set: None,
        train_rows: Vec::new(),
        minibatch_rows: Vec::new(),
        reflection_rows: Vec::new(),
        heldout_rows: Vec::new(),
        rollout_task_id: None,
    })
}

fn ensure_container_inputs(context: &mut GepaRunContext) -> Result<GepaContainerInputs> {
    let container_process = ManagedContainerProcess::maybe_start(&context.config.container)?;
    let container_url = context
        .config
        .container
        .url
        .clone()
        .ok_or_else(|| OptimizerError::Config("container.url is required".to_string()))?;
    let client = ContainerClient::new(container_url.clone())?;
    let metadata = client.verify_gepa_contract()?;
    let gepa_contract = metadata.gepa_contract()?;
    context.workspace.record_container_contract_snapshot(
        &ContainerContractSnapshotRecord::from_input(ContainerContractSnapshotInput {
            run_id: &context.config.run.run_id,
            container_url: &container_url,
            contract_kind: "gepa",
            contract_version: &gepa_contract.version,
            validation_status: "valid",
            metadata_response: &serde_json::to_value(&metadata)?,
            health_response: None,
            metadata: Map::new(),
        }),
    )?;
    context.events.emit(
        "container.contract.verified",
        "Container advertised GEPA contract",
        serde_json::to_value(&metadata)?,
    )?;

    let program_value = cached_call(
        &mut context.cache,
        &format!("{}:container.program", context.cache_namespace),
        &json!({"url": container_url, "route": "/program"}),
        || {
            let program = client.program_typed()?;
            Ok(serde_json::to_value(program)?)
        },
    )?;
    let mut program = PromptProgram::from_value(program_value)?;
    match cached_call(
        &mut context.cache,
        &format!("{}:container.task_info", context.cache_namespace),
        &json!({"url": container_url, "route": "/task_info"}),
        || client.task_info(),
    ) {
        Ok(task_info) => {
            program
                .metadata
                .insert("task_info".to_string(), task_info.clone());
            context.events.emit(
                "container.task_info.loaded",
                "Container task info loaded",
                task_info,
            )?;
        }
        Err(error) => {
            context.events.emit(
                "container.task_info.missing",
                "Container task info route was unavailable; proposer will infer task from program and rollouts",
                json!({"error": error.to_string()}),
            )?;
        }
    }
    program.validate_for_gepa(
        &context.config.candidate.target_modules,
        &context.config.seed_candidate,
    )?;
    let lever_manifest = LeverManifest::from_prompt_program(&program);
    let rollout_task_id = rollout_task_id(&program);
    context
        .workspace
        .record_prompt_program_snapshot(&PromptProgramSnapshotRecord::from_input(
            PromptProgramSnapshotInput {
                run_id: &context.config.run.run_id,
                program_id: &program.program_id,
                target_modules: &context.config.candidate.target_modules,
                mutable_field_ids: program.mutable_field_ids(),
                validation_status: "valid",
                program: &serde_json::to_value(&program)?,
                metadata: Map::new(),
            },
        ))?;
    context.events.emit(
        "container.program.loaded",
        "Prompt program loaded",
        json!({
            "program_id": program.program_id,
            "mutable_fields": program.mutable_field_ids(),
            "lever_manifest": lever_manifest,
        }),
    )?;

    let dataset_value = cached_call(
        &mut context.cache,
        &format!("{}:container.dataset", context.cache_namespace),
        &json!({"url": container_url, "route": "/dataset"}),
        || {
            let response = client.dataset_typed()?;
            Ok(serde_json::to_value(response)?)
        },
    )?;
    let dataset_response: DatasetResponse = serde_json::from_value(dataset_value.clone())?;
    let dataset_id = dataset_response
        .dataset_id
        .clone()
        .unwrap_or_else(|| "container_dataset".to_string());
    let seed_pool_seeds = effective_gepa_seed_pool_seeds(&context.config);
    let pareto_eval_seeds = seed_pool_seeds
        .get("pareto_eval")
        .cloned()
        .unwrap_or_else(|| context.config.dataset.train_seeds.clone());
    let minibatch_seeds = seed_pool_seeds
        .get("minibatch")
        .cloned()
        .unwrap_or_else(|| pareto_eval_seeds.clone());
    let reflection_seeds = seed_pool_seeds
        .get("reflection")
        .cloned()
        .unwrap_or_else(|| pareto_eval_seeds.clone());
    let validation_seeds = seed_pool_seeds
        .get("validation")
        .cloned()
        .unwrap_or_else(|| context.config.dataset.heldout_seeds.clone());
    let train_response = load_rows(
        &client,
        &mut context.cache,
        &context.cache_namespace,
        &context.config.dataset.train_split,
        &pareto_eval_seeds,
        Value::Object(context.config.dataset.filters.clone()),
    )?;
    let heldout_response = load_rows(
        &client,
        &mut context.cache,
        &context.cache_namespace,
        &context.config.dataset.heldout_split,
        &validation_seeds,
        Value::Object(context.config.dataset.filters.clone()),
    )?;
    let train_rows = train_response.rows.clone();
    let heldout_rows = heldout_response.rows.clone();
    let minibatch_rows = if minibatch_seeds == pareto_eval_seeds {
        train_rows.clone()
    } else {
        load_rows(
            &client,
            &mut context.cache,
            &context.cache_namespace,
            &context.config.dataset.train_split,
            &minibatch_seeds,
            Value::Object(context.config.dataset.filters.clone()),
        )?
        .rows
    };
    let reflection_rows = if reflection_seeds == pareto_eval_seeds {
        train_rows.clone()
    } else if reflection_seeds == minibatch_seeds {
        minibatch_rows.clone()
    } else {
        load_rows(
            &client,
            &mut context.cache,
            &context.cache_namespace,
            &context.config.dataset.train_split,
            &reflection_seeds,
            Value::Object(context.config.dataset.filters.clone()),
        )?
        .rows
    };
    record_dataset_snapshot(
        &context.workspace,
        DatasetSnapshotCall {
            run_id: &context.config.run.run_id,
            dataset_id: &dataset_id,
            split: &context.config.dataset.train_split,
            seeds: &pareto_eval_seeds,
            filters: &Value::Object(context.config.dataset.filters.clone()),
            response: &train_response,
            dataset_metadata: &dataset_value,
        },
    )?;
    record_dataset_snapshot(
        &context.workspace,
        DatasetSnapshotCall {
            run_id: &context.config.run.run_id,
            dataset_id: &dataset_id,
            split: &context.config.dataset.heldout_split,
            seeds: &validation_seeds,
            filters: &Value::Object(context.config.dataset.filters.clone()),
            response: &heldout_response,
            dataset_metadata: &dataset_value,
        },
    )?;
    context.events.emit(
        "dataset.rows.loaded",
        "Dataset rows loaded",
        json!({
            "train_rows": train_rows.len(),
            "minibatch_rows": minibatch_rows.len(),
            "reflection_rows": reflection_rows.len(),
            "heldout_rows": heldout_rows.len(),
            "seed_pools": {
                "pareto_eval": pareto_eval_seeds,
                "minibatch": minibatch_seeds,
                "reflection": reflection_seeds,
                "validation": validation_seeds,
            },
        }),
    )?;
    let objective_set =
        declared_objective_set(&context.config, &program, &train_rows, &heldout_rows);
    context
        .workspace
        .record_objective_set(&context.config.run.run_id, &objective_set)?;
    context.events.emit(
        "objective_set.declared",
        "Objective set declared",
        json!({
            "objective_set_id": objective_set.objective_set_id.clone(),
            "objective_set_hash": objective_set.objective_set_hash.clone(),
            "selection_objective": objective_set.selection_objective.clone(),
            "frontier_type": objective_set.frontier_type.clone(),
            "objectives": objective_set.objectives.clone(),
        }),
    )?;
    if matches!(
        context.state_machine.state(),
        OptimizerRunState::Initializing | OptimizerRunState::Restoring
    ) {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Ready,
            OptimizerTransitionTrigger::ContainerReady,
            "Container, program, and dataset ready",
            json!({
                "train_rows": train_rows.len(),
                "minibatch_rows": minibatch_rows.len(),
                "reflection_rows": reflection_rows.len(),
                "heldout_rows": heldout_rows.len(),
            }),
        )?;
    }
    Ok(GepaContainerInputs {
        _container_process: container_process,
        client,
        program,
        objective_set,
        train_rows,
        minibatch_rows,
        reflection_rows,
        heldout_rows,
        rollout_task_id,
    })
}

fn initialize_or_restore_cursor(workspace: &WorkspaceStore, run_id: &str) -> Result<GepaCursor> {
    let Some(checkpoint) = workspace.latest_checkpoint(run_id, GEPA_CURSOR_CHECKPOINT_KIND)? else {
        return Ok(GepaCursor::new(run_id.to_string()));
    };
    let mut cursor: GepaCursor = serde_json::from_value(checkpoint.snapshot)?;
    if cursor.run_id.is_empty() {
        cursor.run_id = run_id.to_string();
    }
    Ok(cursor)
}

fn restore_gepa_run_state(context: &mut GepaRunContext) -> Result<GepaRunState> {
    let cursor = initialize_or_restore_cursor(&context.workspace, &context.config.run.run_id)?;
    restore_state_machine_from_cursor(context, &cursor)?;
    let candidates: Vec<CandidateRecord> = cursor
        .candidates
        .as_array()
        .filter(|rows| !rows.is_empty())
        .map(|_| serde_json::from_value(cursor.candidates.clone()))
        .transpose()?
        .unwrap_or_default();
    let best_idx = cursor.best_candidate_id.as_ref().and_then(|candidate_id| {
        candidates
            .iter()
            .position(|candidate| &candidate.candidate_id == candidate_id)
    });
    let proposal_queue = cursor
        .proposal_queue
        .as_array()
        .filter(|rows| !rows.is_empty())
        .map(|_| serde_json::from_value(cursor.proposal_queue.clone()))
        .transpose()?
        .unwrap_or_default();
    let active_evaluation = cursor
        .active_evaluation
        .clone()
        .filter(|value| !value.is_null())
        .map(serde_json::from_value)
        .transpose()?;
    let total_usage = if cursor.usage.is_null() {
        UsageTotals::default()
    } else {
        serde_json::from_value(cursor.usage.clone())?
    };
    let mut usage_ledger: Vec<UsageLedgerRecord> = cursor
        .usage_ledger
        .as_array()
        .filter(|rows| !rows.is_empty())
        .map(|_| serde_json::from_value(cursor.usage_ledger.clone()))
        .transpose()?
        .unwrap_or_default();
    if usage_ledger.is_empty() {
        usage_ledger.extend(
            candidates
                .iter()
                .flat_map(|candidate| candidate.sensor_frames.iter())
                .map(UsageLedgerRecord::from_sensor_frame),
        );
    }
    let stopper_states = cursor
        .stopper_states
        .as_array()
        .filter(|rows| !rows.is_empty())
        .map(|_| serde_json::from_value(cursor.stopper_states.clone()))
        .transpose()?
        .unwrap_or_default();
    Ok(GepaRunState {
        checkpoint_sequence: cursor.checkpoint_sequence,
        stopper_sequence: cursor.stopper_sequence,
        heldout_candidate_index: cursor.heldout_candidate_index,
        total_cost: cursor.cost_usd,
        rollout_count: cursor.rollout_count,
        cursor,
        candidates,
        best_idx,
        proposal_queue,
        active_evaluation,
        total_usage,
        usage_ledger,
        stopper_states,
    })
}

fn restore_state_machine_from_cursor(
    context: &mut GepaRunContext,
    cursor: &GepaCursor,
) -> Result<()> {
    if cursor.state_history.is_null() {
        return Ok(());
    }
    let history: Vec<OptimizerTransition> = serde_json::from_value(cursor.state_history.clone())?;
    if history.is_empty() {
        return Ok(());
    }
    let state = history
        .last()
        .map(|transition| transition.to)
        .unwrap_or(OptimizerRunState::Created);
    context.state_machine.history = history;
    context.state_machine.state = state;
    Ok(())
}

fn ensure_step_resources(
    context: &mut GepaRunContext,
    state: &GepaRunState,
) -> Result<GepaStepResources> {
    if context.client.is_none() {
        let inputs = ensure_container_inputs(context)?;
        context.container_process = inputs._container_process;
        context.client = Some(inputs.client);
        context.program = Some(inputs.program);
        context.objective_set = Some(inputs.objective_set);
        context.train_rows = inputs.train_rows;
        context.minibatch_rows = inputs.minibatch_rows;
        context.reflection_rows = inputs.reflection_rows;
        context.heldout_rows = inputs.heldout_rows;
        context.rollout_task_id = Some(inputs.rollout_task_id);
    }
    if !state.cursor.program.is_null() {
        context.program = Some(serde_json::from_value(state.cursor.program.clone())?);
    }
    if !state.cursor.objective_set.is_null() {
        context.objective_set = Some(serde_json::from_value(state.cursor.objective_set.clone())?);
    }
    if state
        .cursor
        .train_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        context.train_rows = serde_json::from_value(state.cursor.train_rows.clone())?;
    }
    if state
        .cursor
        .minibatch_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        context.minibatch_rows = serde_json::from_value(state.cursor.minibatch_rows.clone())?;
    }
    if state
        .cursor
        .reflection_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        context.reflection_rows = serde_json::from_value(state.cursor.reflection_rows.clone())?;
    }
    if state
        .cursor
        .heldout_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        context.heldout_rows = serde_json::from_value(state.cursor.heldout_rows.clone())?;
    }
    if let Some(task_id) = state.cursor.rollout_task_id.clone() {
        context.rollout_task_id = Some(task_id);
    }
    Ok(GepaStepResources {
        client: context.client.clone().ok_or_else(|| {
            OptimizerError::Invariant("GEPA context missing container client".to_string())
        })?,
        program: context.program.clone().ok_or_else(|| {
            OptimizerError::Invariant("GEPA context missing prompt program".to_string())
        })?,
        objective_set: context.objective_set.clone().ok_or_else(|| {
            OptimizerError::Invariant("GEPA context missing objective set".to_string())
        })?,
        train_rows: context.train_rows.clone(),
        minibatch_rows: if context.minibatch_rows.is_empty() {
            context.train_rows.clone()
        } else {
            context.minibatch_rows.clone()
        },
        reflection_rows: if context.reflection_rows.is_empty() {
            context.train_rows.clone()
        } else {
            context.reflection_rows.clone()
        },
        heldout_rows: context.heldout_rows.clone(),
        rollout_task_id: context.rollout_task_id.clone().ok_or_else(|| {
            OptimizerError::Invariant("GEPA context missing rollout task id".to_string())
        })?,
    })
}

fn persist_gepa_run_state(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    phase: GepaCursorPhase,
    status: &str,
    reason: &str,
    metadata: Map<String, Value>,
) -> Result<()> {
    state.checkpoint_sequence += 1;
    state.cursor.schema_version = planner::GEPA_CURSOR_SCHEMA_VERSION.to_string();
    state.cursor.run_id = context.config.run.run_id.clone();
    state.cursor.phase = phase;
    state.cursor.proposal_queue = serde_json::to_value(&state.proposal_queue)?;
    state.cursor.heldout_candidate_index = state.heldout_candidate_index;
    state.cursor.active_evaluation = state
        .active_evaluation
        .as_ref()
        .map(serde_json::to_value)
        .transpose()?;
    state.cursor.candidates = serde_json::to_value(&state.candidates)?;
    state.cursor.best_candidate_id = state
        .best_idx
        .and_then(|idx| state.candidates.get(idx))
        .map(|candidate| candidate.candidate_id.clone());
    state.cursor.rollout_task_id = Some(resources.rollout_task_id.clone());
    state.cursor.rollout_count = state.rollout_count;
    state.cursor.cost_usd = state.total_cost;
    state.cursor.usage = serde_json::to_value(&state.total_usage)?;
    state.cursor.usage_ledger = serde_json::to_value(&state.usage_ledger)?;
    state.cursor.stopper_states = serde_json::to_value(&state.stopper_states)?;
    state.cursor.stopper_sequence = state.stopper_sequence;
    state.cursor.checkpoint_sequence = state.checkpoint_sequence;
    state.cursor.train_rows = serde_json::to_value(&resources.train_rows)?;
    state.cursor.minibatch_rows = serde_json::to_value(&resources.minibatch_rows)?;
    state.cursor.reflection_rows = serde_json::to_value(&resources.reflection_rows)?;
    state.cursor.heldout_rows = serde_json::to_value(&resources.heldout_rows)?;
    state.cursor.program = serde_json::to_value(&resources.program)?;
    state.cursor.objective_set = serde_json::to_value(&resources.objective_set)?;
    state.cursor.state_history = serde_json::to_value(&context.state_machine.history)?;
    let metadata = metadata_with_pipeline_state(context, state, metadata)?;
    state.cursor.metadata = Value::Object(metadata.clone());
    let cursor_value = serde_json::to_value(&state.cursor)?;
    let checkpoint = CheckpointRecord::from_input(CheckpointInput {
        sequence_number: state.checkpoint_sequence,
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status,
        run_state: state.cursor.phase.as_str(),
        reason: Some(reason),
        generation: Some(state.cursor.generation as u64),
        candidate_id: state.cursor.best_candidate_id.as_deref(),
        evaluation_stage: Some(state.cursor.phase.as_str()),
        best_candidate_id: state.cursor.best_candidate_id.as_deref(),
        candidate_count: state.candidates.len() as u64,
        frontier_count: frontier_members(&state.candidates).len() as u64,
        rollout_count: state.rollout_count as u64,
        cost_usd: state.total_cost,
        usage: state.cursor.usage.clone(),
        snapshot: cursor_value,
        metadata,
    });
    context
        .workspace
        .record_checkpoint(&context.config.run.run_id, &checkpoint)
}

fn metadata_with_pipeline_state(
    context: &GepaRunContext,
    state: &mut GepaRunState,
    mut metadata: Map<String, Value>,
) -> Result<Map<String, Value>> {
    if let GepaPipelineRuntimePlan::AsyncPipelined(plan) =
        GepaPipelineRuntimePlan::from_config(&context.config)?
    {
        refresh_async_pipeline_cursor_state(context, state, &plan);
        metadata.insert("pipeline".to_string(), plan_metadata(&plan));
        metadata.insert(
            "pipeline_state".to_string(),
            serde_json::to_value(&state.cursor.pipeline_state)?,
        );
    }
    Ok(metadata)
}

pub(crate) fn advance_gepa_config_once(
    config: SynthOptimizerConfig,
    options: GepaExecutionOptions,
    mode: GepaAdvanceMode,
) -> Result<GepaAdvanceOutcome> {
    let mut context = open_gepa_run_context(config, &options)?;
    let mut state = restore_gepa_run_state(&mut context)?;
    advance_gepa_once(&mut context, &mut state, mode, &options)
}

fn advance_gepa_once(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    mode: GepaAdvanceMode,
    options: &GepaExecutionOptions,
) -> Result<GepaAdvanceOutcome> {
    match GepaPipelineRuntimePlan::from_config(&context.config)? {
        GepaPipelineRuntimePlan::SyncSerial(_) => {
            advance_gepa_sync_serial_once(context, state, mode, options)
        }
        GepaPipelineRuntimePlan::AsyncPipelined(plan) => {
            advance_gepa_async_pipelined_once(context, state, mode, options, &plan)
        }
    }
}

fn refresh_terminal_run_projection(
    context: &mut GepaRunContext,
    state: &GepaRunState,
) -> Result<()> {
    let usage_value = serde_json::to_value(&state.total_usage)?;
    match state.cursor.phase {
        GepaCursorPhase::Completed => {
            if let Some(best_candidate_id) = state
                .best_idx
                .and_then(|idx| state.candidates.get(idx))
                .map(|candidate| candidate.candidate_id.as_str())
                .or(state.cursor.best_candidate_id.as_deref())
            {
                context.workspace.record_run_finished(
                    &context.config.run.run_id,
                    best_candidate_id,
                    state.total_cost,
                    &usage_value,
                )?;
            }
        }
        GepaCursorPhase::Failed => {
            context.workspace.record_run_failed(
                &context.config.run.run_id,
                state.cursor.best_candidate_id.as_deref(),
                state.total_cost,
                &usage_value,
            )?;
        }
        GepaCursorPhase::Cancelled => {
            context.workspace.record_run_cancelled_result(
                &context.config.run.run_id,
                state.cursor.best_candidate_id.as_deref(),
                state.total_cost,
                &usage_value,
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn advance_gepa_sync_serial_once(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    mode: GepaAdvanceMode,
    options: &GepaExecutionOptions,
) -> Result<GepaAdvanceOutcome> {
    if matches!(state.cursor.phase, GepaCursorPhase::Completed) {
        refresh_terminal_run_projection(context, state)?;
        let result = state
            .cursor
            .terminal_summary
            .clone()
            .map(serde_json::from_value)
            .transpose()?;
        return Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::TerminalizeRun {
                run_id: context.config.run.run_id.clone(),
                status: "completed".to_string(),
            },
            terminal: true,
            result,
            message: "GEPA run already completed".to_string(),
        });
    }
    if state.cursor.phase.is_terminal() {
        refresh_terminal_run_projection(context, state)?;
        return Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::TerminalizeRun {
                run_id: context.config.run.run_id.clone(),
                status: state.cursor.phase.as_str().to_string(),
            },
            terminal: true,
            result: None,
            message: format!("GEPA run already {}", state.cursor.phase.as_str()),
        });
    }
    if let Err(error) = check_cancelled(options.cancellation.as_ref()) {
        return terminalize_aborted_gepa_run(context, state, error, "GEPA run cancelled");
    }
    if matches!(state.cursor.phase, GepaCursorPhase::Paused) {
        return Ok(paused_gepa_outcome(context));
    }
    let resources = ensure_step_resources(context, state)?;
    if let Some(job_id) = state.cursor.pending_job_id.clone() {
        return advance_pending_runtime_job(context, state, &resources, mode, &job_id, None);
    }
    match state.cursor.phase {
        GepaCursorPhase::Initializing => advance_initializing(context, state, &resources),
        GepaCursorPhase::SeedFullTrain => {
            advance_rollout_stage(context, state, &resources, "seed_full_train")
        }
        GepaCursorPhase::GenerationStart => advance_generation_start(context, state, &resources),
        GepaCursorPhase::ProposerWaiting => advance_proposer_waiting(context, state, &resources),
        GepaCursorPhase::CandidateMinibatch => {
            advance_rollout_stage(context, state, &resources, "candidate_minibatch")
        }
        GepaCursorPhase::CandidateFullTrain => {
            advance_rollout_stage(context, state, &resources, "candidate_full_train")
        }
        GepaCursorPhase::Heldout => advance_heldout(context, state, &resources),
        GepaCursorPhase::Finalizing => finalize_completed_gepa_run(context, state, &resources),
        GepaCursorPhase::Paused => Ok(paused_gepa_outcome(context)),
        GepaCursorPhase::Completed | GepaCursorPhase::Failed | GepaCursorPhase::Cancelled => {
            unreachable!("terminal cursor phases are handled before phase dispatch")
        }
    }
}

fn advance_gepa_async_pipelined_once(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    mode: GepaAdvanceMode,
    options: &GepaExecutionOptions,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<GepaAdvanceOutcome> {
    refresh_async_pipeline_cursor_state(context, state, plan);
    if matches!(state.cursor.phase, GepaCursorPhase::Completed) {
        refresh_terminal_run_projection(context, state)?;
        let result = state
            .cursor
            .terminal_summary
            .clone()
            .map(serde_json::from_value)
            .transpose()?;
        return Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::TerminalizeRun {
                run_id: context.config.run.run_id.clone(),
                status: "completed".to_string(),
            },
            terminal: true,
            result,
            message: "async-pipelined: GEPA run already completed".to_string(),
        });
    }
    if state.cursor.phase.is_terminal() {
        refresh_terminal_run_projection(context, state)?;
        return Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::TerminalizeRun {
                run_id: context.config.run.run_id.clone(),
                status: state.cursor.phase.as_str().to_string(),
            },
            terminal: true,
            result: None,
            message: format!(
                "async-pipelined: GEPA run already {}",
                state.cursor.phase.as_str()
            ),
        });
    }
    if let Err(error) = check_cancelled(options.cancellation.as_ref()) {
        state.cursor.pipeline_state.propose_queue.clear();
        state.cursor.pipeline_state.rollout_queue.clear();
        state.cursor.pipeline_state.evaluate_queue.clear();
        state.cursor.pipeline_state.lane_leases.clear();
        state.cursor.pipeline_state.candidate_partials.clear();
        return terminalize_aborted_gepa_run(context, state, error, "GEPA run cancelled");
    }
    if matches!(state.cursor.phase, GepaCursorPhase::Paused) {
        return Ok(paused_gepa_outcome(context));
    }

    if let Some(outcome) = terminalize_restored_unclaimable_async_job(context, state, mode)? {
        return Ok(outcome);
    }

    let resources = ensure_step_resources(context, state)?;
    ensure_adaptive_rollout_concurrency_state(state, plan);

    // Old Phase-1 async cursors used the serial pending-job slot. Finish that
    // in-place before switching the cursor over to lane leases.
    if let Some(job_id) = state.cursor.pending_job_id.clone() {
        let mut outcome =
            advance_pending_runtime_job(context, state, &resources, mode, &job_id, Some(plan))?;
        outcome.message = format!("async-pipelined legacy lane: {}", outcome.message);
        return Ok(outcome);
    }

    if matches!(state.cursor.phase, GepaCursorPhase::Initializing) {
        let mut outcome = advance_gepa_sync_serial_once(context, state, mode, options)?;
        outcome.message = format!("async-pipelined seed: {}", outcome.message);
        return Ok(outcome);
    }

    if matches!(state.cursor.phase, GepaCursorPhase::SeedFullTrain)
        && state.active_evaluation.is_none()
        && !state
            .cursor
            .pipeline_state
            .candidate_partials
            .contains_key("async:seed_full_train:generation_000")
    {
        let outcome = plan_async_seed_full_train(context, state, &resources, plan)?;
        return Ok(outcome);
    }

    if let Some(outcome) = consume_async_lane_work(context, state, &resources, plan)? {
        return Ok(outcome);
    }
    if let Some(mut outcome) =
        schedule_async_lane_transition(context, state, &resources, mode, plan)?
    {
        let mut planned_count = 1usize;
        while matches!(
            outcome.action,
            planner::GepaTickAction::PlanRuntimeJob { .. }
        ) {
            let before_rollout_leases = async_lane_lease_count(state, "rollout");
            if before_rollout_leases >= adaptive_rollout_worker_limit(state, plan) {
                break;
            }
            let Some(next_outcome) = schedule_async_rollout_job(context, state, &resources, plan)?
            else {
                break;
            };
            if !matches!(
                next_outcome.action,
                planner::GepaTickAction::PlanRuntimeJob { .. }
            ) {
                break;
            }
            planned_count += 1;
            outcome = next_outcome;
        }
        if planned_count > 1 {
            outcome.message = format!(
                "async-pipelined: planned {planned_count} rollout jobs to fill lane capacity"
            );
        }
        return Ok(outcome);
    }
    if async_pipeline_has_no_lane_work(state)
        && matches!(
            state.cursor.phase,
            GepaCursorPhase::Heldout | GepaCursorPhase::Finalizing
        )
    {
        let mut outcome = advance_gepa_sync_serial_once(context, state, mode, options)?;
        outcome.message = format!("async-pipelined terminal: {}", outcome.message);
        return Ok(outcome);
    }
    if async_pipeline_idle(state) && async_pipeline_stopper_satisfied(context, state) {
        let mut outcome = advance_gepa_sync_serial_once(context, state, mode, options)?;
        outcome.message = format!("async-pipelined terminal: {}", outcome.message);
        return Ok(outcome);
    }

    refresh_async_pipeline_cursor_state(context, state, plan);
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::Noop,
        terminal: false,
        result: None,
        message: "async-pipelined: waiting for lane capacity or completions".to_string(),
    })
}

fn paused_gepa_outcome(context: &GepaRunContext) -> GepaAdvanceOutcome {
    GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "paused".to_string(),
        },
        terminal: false,
        result: None,
        message: "GEPA run is paused".to_string(),
    }
}

fn terminalize_restored_unclaimable_async_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    mode: GepaAdvanceMode,
) -> Result<Option<GepaAdvanceOutcome>> {
    if !matches!(mode, GepaAdvanceMode::RunLoop) {
        return Ok(None);
    }
    let mut leases = state
        .cursor
        .pipeline_state
        .lane_leases
        .values()
        .cloned()
        .collect::<Vec<_>>();
    leases.sort_by(|left, right| left.lease_id.cmp(&right.lease_id));
    for lease in leases {
        let Some(job_id) = lease.job_id.as_deref() else {
            continue;
        };
        let job = context
            .workspace
            .optimizer_job(&context.config.run.run_id, job_id)?;
        if !matches!(
            job.status,
            OptimizerJobStatus::Leased
                | OptimizerJobStatus::Running
                | OptimizerJobStatus::Annotating
                | OptimizerJobStatus::Verifying
        ) {
            continue;
        }
        state.cursor.pipeline_state.lane_leases.clear();
        state.cursor.pipeline_state.propose_queue.clear();
        state.cursor.pipeline_state.rollout_queue.clear();
        state.cursor.pipeline_state.evaluate_queue.clear();
        state.cursor.pipeline_state.candidate_partials.clear();
        let error = OptimizerError::Invariant(format!(
            "restored async lane job {} is already {}; direct GEPA run-loop cannot complete a runtime job left running by a previous process (worker={:?}, lease_expires_at={:?})",
            job.job_id,
            job.status.as_str(),
            job.worker_id,
            job.lease_expires_at,
        ));
        return terminalize_aborted_gepa_run(
            context,
            state,
            error,
            "GEPA async runtime job was left running by a previous process",
        )
        .map(Some);
    }
    Ok(None)
}

fn plan_metadata(plan: &GepaAsyncPipelinedPlan) -> Value {
    GepaPipelineRuntimePlan::AsyncPipelined(plan.clone()).metadata()
}

fn refresh_async_pipeline_cursor_state(
    context: &GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinedPlan,
) {
    let pool_version = state
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.train_reward.is_some()
                && matches!(
                    candidate.status.as_str(),
                    "full_train_evaluated" | "accepted"
                )
        })
        .count() as u64;
    let in_flight_candidate_count = async_pipeline_in_flight_candidate_count(state);
    state.cursor.pipeline_state.pool_version = pool_version;
    state.cursor.pipeline_state.in_flight_candidate_count = in_flight_candidate_count;
    state.cursor.pipeline_state.pending_job_ids = state
        .cursor
        .pipeline_state
        .lane_leases
        .values()
        .filter_map(|lease| lease.job_id.clone())
        .collect();
    state.cursor.pipeline_state.pending_effect_ids = state
        .cursor
        .pipeline_state
        .lane_leases
        .values()
        .filter_map(|lease| lease.effect_id.clone())
        .collect();
    state.cursor.pipeline_state.terminal_readiness = json!({
        "phase": state.cursor.phase.as_str(),
        "phase_terminal": state.cursor.phase.is_terminal(),
        "stopper_satisfied": async_pipeline_stopper_satisfied(context, state),
        "rollout_chunks_folded": async_rollout_chunks_folded(state),
        "retry_scheduled": async_pipeline_retry_scheduled(state),
        "pending_jobs_empty": state.cursor.pipeline_state.pending_job_ids.is_empty(),
        "pending_effects_empty": state.cursor.pipeline_state.pending_effect_ids.is_empty(),
        "leases_empty": state.cursor.pipeline_state.lane_leases.is_empty(),
        "propose_queue_empty": state.cursor.pipeline_state.propose_queue.is_empty(),
        "rollout_queue_empty": state.cursor.pipeline_state.rollout_queue.is_empty(),
        "evaluate_queue_empty": state.cursor.pipeline_state.evaluate_queue.is_empty(),
        "proposal_queue_empty": state.proposal_queue.is_empty(),
        "active_evaluation_empty": state.active_evaluation.is_none(),
        "max_in_flight_candidates": plan.max_in_flight_candidates,
        "adaptive_rollout_concurrency": adaptive_rollout_snapshot(state, plan),
    });
}

fn async_pipeline_in_flight_candidate_count(state: &GepaRunState) -> usize {
    let mut candidate_ids = BTreeSet::new();
    for partial in state.cursor.pipeline_state.candidate_partials.values() {
        candidate_ids.extend(partial.candidate_ids.iter().cloned());
    }
    for item in state
        .cursor
        .pipeline_state
        .rollout_queue
        .iter()
        .chain(state.cursor.pipeline_state.evaluate_queue.iter())
    {
        candidate_ids.extend(item.candidate_ids.iter().cloned());
    }
    for lease in state.cursor.pipeline_state.lane_leases.values() {
        if let Some(ids) = lease
            .metadata
            .get("candidate_ids")
            .and_then(Value::as_array)
        {
            candidate_ids.extend(ids.iter().filter_map(Value::as_str).map(str::to_string));
        }
    }
    candidate_ids.len()
}

fn async_stage_work_pending(state: &GepaRunState, stages: &[&str]) -> bool {
    state
        .cursor
        .pipeline_state
        .candidate_partials
        .values()
        .any(|partial| stages.contains(&partial.stage.as_str()))
        || state
            .cursor
            .pipeline_state
            .rollout_queue
            .iter()
            .chain(state.cursor.pipeline_state.evaluate_queue.iter())
            .any(|item| stages.contains(&item.stage.as_str()))
        || state
            .cursor
            .pipeline_state
            .lane_leases
            .values()
            .any(|lease| stages.contains(&lease.stage.as_str()))
}

fn consume_async_lane_work(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    let mut lease_keys = state
        .cursor
        .pipeline_state
        .lane_leases
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    lease_keys.sort();
    for lease_key in lease_keys {
        let Some(lease) = state
            .cursor
            .pipeline_state
            .lane_leases
            .get(&lease_key)
            .cloned()
        else {
            continue;
        };
        let Some(job_id) = lease.job_id.clone() else {
            continue;
        };
        let job = context
            .workspace
            .optimizer_job(&context.config.run.run_id, &job_id)?;
        match job.status {
            OptimizerJobStatus::Completed => {
                restore_async_partial_as_active(state, lease.partial_id.as_deref())?;
                if lease.lane == "propose" {
                    if let Some(parent_id) = lease
                        .metadata
                        .get("parent_candidate_id")
                        .and_then(Value::as_str)
                    {
                        state.cursor.pipeline_state.parent_candidate_id =
                            Some(parent_id.to_string());
                    }
                    if let Some(active) = state.active_evaluation.as_ref() {
                        state.cursor.generation = active.generation;
                    }
                }
                state.cursor.pending_job_id = Some(job_id.clone());
                state.cursor.pending_effect_id = lease.effect_id.clone();
                state.cursor.pending_reservation_ids = lease.reservation_ids.clone();
                let mut outcome = consume_completed_runtime_job(context, state, resources, job)?;
                state.cursor.pipeline_state.lane_leases.remove(&lease_key);
                state.cursor.pipeline_state.propose_queue.retain(|item| {
                    item.job_id.as_deref() != Some(job_id.as_str())
                        && item.partial_id.as_deref() != lease.partial_id.as_deref()
                });
                state.cursor.pending_job_id = None;
                state.cursor.pending_effect_id = None;
                state.cursor.pending_reservation_ids.clear();

                if lease.lane == "rollout" {
                    if let Some(active) = state.active_evaluation.clone() {
                        let partial_id = lease
                            .partial_id
                            .clone()
                            .unwrap_or_else(|| async_partial_id(&active.stage, active.generation));
                        let chunk_id = lease
                            .metadata
                            .get("chunk_id")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        let chunk_rows = lease
                            .metadata
                            .get("chunk_rows")
                            .and_then(Value::as_u64)
                            .unwrap_or(0);
                        if let Some(chunk_id) =
                            lease.metadata.get("chunk_id").and_then(Value::as_str)
                        {
                            if let Some(partial) = state
                                .cursor
                                .pipeline_state
                                .candidate_partials
                                .get_mut(&partial_id)
                            {
                                if let Some(chunk) = partial.rollout_chunks.get_mut(chunk_id) {
                                    chunk.status = "folded".to_string();
                                    chunk.folded = true;
                                }
                            }
                        }
                        let wall_seconds = lease
                            .effect_id
                            .as_ref()
                            .and_then(|effect_id| {
                                context
                                    .workspace
                                    .runtime_effect(&context.config.run.run_id, effect_id)
                                    .ok()
                            })
                            .and_then(|effect| {
                                effect.metadata.get("wall_seconds").and_then(Value::as_f64)
                            })
                            .unwrap_or(0.0);
                        context.events.emit(
                            "rollout.chunk.finished",
                            "Rollout chunk finished",
                            json!({
                                "chunk_id": chunk_id,
                                "job_id": job_id,
                                "stage": active.stage,
                                "rows": chunk_rows,
                                "completed_rows": active_rollout_completed_rows(&active),
                                "total_rows": active_rollout_total_rows(&active),
                                "active_rollout_workers": async_lane_lease_count(state, "rollout").saturating_sub(1),
                                "wall_seconds": wall_seconds,
                                "rows_per_second": if wall_seconds > 0.0 { chunk_rows as f64 / wall_seconds } else { 0.0 },
                            }),
                        )?;
                        if active_rollout_evaluation_complete(&active) {
                            upsert_async_partial_from_active(
                                state,
                                &partial_id,
                                "evaluate",
                                lease.parent_pool_version,
                            )?;
                            let item = async_work_item_from_active(
                                &active,
                                "evaluate",
                                lease.parent_pool_version,
                                Some(partial_id),
                            )?;
                            state.cursor.pipeline_state.evaluate_queue.push(item);
                        } else {
                            upsert_async_partial_from_active(
                                state,
                                &partial_id,
                                "rollout",
                                lease.parent_pool_version,
                            )?;
                            let item = async_work_item_from_active(
                                &active,
                                "rollout",
                                lease.parent_pool_version,
                                Some(partial_id),
                            )?;
                            state.cursor.pipeline_state.rollout_queue.push(item);
                        }
                    }
                    state.active_evaluation = None;
                } else if let Some(partial_id) = lease.partial_id.as_ref() {
                    state
                        .cursor
                        .pipeline_state
                        .candidate_partials
                        .remove(partial_id);
                    state.active_evaluation = None;
                }

                refresh_async_pipeline_cursor_state(context, state, plan);
                persist_gepa_run_state(
                    context,
                    state,
                    resources,
                    state.cursor.phase.clone(),
                    "completed",
                    "consumed async lane runtime outcome",
                    Map::new(),
                )?;
                outcome.message = format!("async-pipelined {}: {}", lease.lane, outcome.message);
                return Ok(Some(outcome));
            }
            OptimizerJobStatus::Failed
            | OptimizerJobStatus::Cancelled
            | OptimizerJobStatus::Expired => {
                state.cursor.pipeline_state.lane_leases.clear();
                state.cursor.pipeline_state.propose_queue.clear();
                state.cursor.pipeline_state.rollout_queue.clear();
                state.cursor.pipeline_state.evaluate_queue.clear();
                state.cursor.pipeline_state.candidate_partials.clear();
                return consume_failed_runtime_job(context, state, resources, job).map(Some);
            }
            _ => {}
        }
    }
    if let Some(item) = state.cursor.pipeline_state.evaluate_queue.first().cloned() {
        state.cursor.pipeline_state.evaluate_queue.remove(0);
        restore_async_partial_as_active(state, item.partial_id.as_deref())?;
        let mut outcome = finalize_active_rollout_evaluation(context, state, resources)?;
        if let Some(partial_id) = item.partial_id.as_ref() {
            state
                .cursor
                .pipeline_state
                .candidate_partials
                .remove(partial_id);
        }
        queue_async_active_rollout_continuation(state, item.parent_pool_version)?;
        refresh_async_pipeline_cursor_state(context, state, plan);
        persist_gepa_run_state(
            context,
            state,
            resources,
            state.cursor.phase.clone(),
            "completed",
            "folded async evaluate work",
            Map::new(),
        )?;
        outcome.message = format!("async-pipelined evaluate: {}", outcome.message);
        return Ok(Some(outcome));
    }
    Ok(None)
}

fn schedule_async_lane_transition(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    mode: GepaAdvanceMode,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if let Some(outcome) = execute_async_leased_runtime_job(context, state, resources, mode, plan)?
    {
        return Ok(Some(outcome));
    }
    if let Some(outcome) = schedule_async_rollout_job(context, state, resources, plan)? {
        return Ok(Some(outcome));
    }
    if async_stage_work_pending(state, &["seed_full_train"]) {
        return Ok(None);
    }
    if let Some(outcome) = schedule_async_candidate_minibatches(context, state, resources, plan)? {
        return Ok(Some(outcome));
    }
    if let Some(outcome) = schedule_async_proposer_job(context, state, resources, plan)? {
        return Ok(Some(outcome));
    }
    Ok(None)
}

fn plan_async_seed_full_train(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<GepaAdvanceOutcome> {
    if state
        .candidates
        .first()
        .and_then(|candidate| candidate.train_reward)
        .is_some()
    {
        state.best_idx = Some(0);
        let mut outcome =
            move_to_generation_start(context, state, resources, "seed already evaluated")?;
        outcome.message = format!("async-pipelined seed: {}", outcome.message);
        return Ok(outcome);
    }
    let capacity =
        remaining_train_rollout_capacity(&context.workspace, &context.config, state.rollout_count)?;
    if capacity < resources.train_rows.len() {
        return Err(rollout_budget_exceeded_error(
            &context.config.run.run_id,
            rollout_budget_limit_name(&context.config),
            resources.train_rows.len(),
            capacity,
        ));
    }
    if let Some(breach) = next_rollout_budget_breach(&context.workspace, &context.config)? {
        return Err(budget_exceeded_error(&context.config.run.run_id, &breach));
    }
    transition_to_rollout_running(
        context,
        "Seed candidate rollouts started",
        json!({
            "candidate_id": state.candidates[0].candidate_id,
            "stage": "seed_full_train",
            "row_count": resources.train_rows.len(),
            "rollout_count": resources.train_rows.len(),
        }),
    )?;
    state.active_evaluation = Some(new_rollout_evaluation(
        "seed_full_train",
        0,
        &resources.train_rows,
        state.cursor.generation,
        state.cursor.proposal_index,
        None,
    )?);
    let partial_id = async_partial_id("seed_full_train", state.cursor.generation);
    upsert_async_partial_from_active(state, &partial_id, "rollout", 0)?;
    let item = async_work_item_from_active(
        &state.active_evaluation.clone().ok_or_else(|| {
            OptimizerError::Invariant("seed active evaluation disappeared".to_string())
        })?,
        "rollout",
        0,
        Some(partial_id),
    )?;
    state.cursor.pipeline_state.rollout_queue.push(item);
    state.active_evaluation = None;
    refresh_async_pipeline_cursor_state(context, state, plan);
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::SeedFullTrain,
        "planned",
        "queued async seed full-train rollout work",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "seed_full_train".to_string(),
        },
        terminal: false,
        result: None,
        message: "async-pipelined seed: queued seed full-train rollout work".to_string(),
    })
}

fn execute_async_leased_runtime_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    mode: GepaAdvanceMode,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    let mut leases = state
        .cursor
        .pipeline_state
        .lane_leases
        .iter()
        .map(|(key, lease)| (key.clone(), lease.clone()))
        .collect::<Vec<_>>();
    leases.sort_by(|left, right| left.0.cmp(&right.0));
    for (lease_key, lease) in leases {
        let Some(job_id) = lease.job_id.clone() else {
            continue;
        };
        let job = context
            .workspace
            .optimizer_job(&context.config.run.run_id, &job_id)?;
        if !matches!(
            job.status,
            OptimizerJobStatus::Pending | OptimizerJobStatus::RetryScheduled
        ) {
            continue;
        }
        if matches!(job.status, OptimizerJobStatus::RetryScheduled)
            && !context
                .workspace
                .optimizer_job_claimable(&context.config.run.run_id, &job_id)?
        {
            continue;
        }
        restore_async_partial_as_active(state, lease.partial_id.as_deref())?;
        state.cursor.pending_job_id = Some(job_id.clone());
        state.cursor.pending_effect_id = lease.effect_id.clone();
        state.cursor.pending_reservation_ids = lease.reservation_ids.clone();
        let mut outcome =
            advance_pending_runtime_job(context, state, resources, mode, &job_id, Some(plan))?;
        if let Some(active) = state.active_evaluation.as_ref() {
            let partial_id = lease
                .partial_id
                .clone()
                .unwrap_or_else(|| async_partial_id(&active.stage, active.generation));
            upsert_async_partial_from_active(
                state,
                &partial_id,
                &lease.lane,
                lease.parent_pool_version,
            )?;
        }
        state.cursor.pending_job_id = None;
        state.cursor.pending_effect_id = None;
        state.cursor.pending_reservation_ids.clear();
        state.active_evaluation = None;
        if let Some(updated) = state.cursor.pipeline_state.lane_leases.get_mut(&lease_key) {
            updated.status = context
                .workspace
                .optimizer_job(&context.config.run.run_id, &job_id)?
                .status
                .as_str()
                .to_string();
        }
        refresh_async_pipeline_cursor_state(context, state, plan);
        persist_gepa_run_state(
            context,
            state,
            resources,
            state.cursor.phase.clone(),
            "running",
            "executed async lane runtime job",
            Map::new(),
        )?;
        outcome.message = format!("async-pipelined {}: {}", lease.lane, outcome.message);
        return Ok(Some(outcome));
    }
    Ok(None)
}

fn schedule_async_rollout_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if async_lane_lease_count(state, "rollout") >= adaptive_rollout_worker_limit(state, plan) {
        return Ok(None);
    }
    let item = loop {
        let Some(item) = state.cursor.pipeline_state.rollout_queue.first().cloned() else {
            return Ok(None);
        };
        state.cursor.pipeline_state.rollout_queue.remove(0);
        restore_async_partial_as_active(state, item.partial_id.as_deref())?;
        if state.active_evaluation.is_some() {
            break item;
        }
        if let Some(partial_id) = item.partial_id.as_ref() {
            state
                .cursor
                .pipeline_state
                .candidate_partials
                .remove(partial_id);
        }
        context.events.emit(
            "rollout.stale_skipped",
            "Stale rollout work skipped",
            json!({
                "item_id": item.item_id,
                "partial_id": item.partial_id,
                "stage": item.stage,
                "generation": item.generation,
            }),
        )?;
    };
    let active = state.active_evaluation.as_ref().ok_or_else(|| {
        OptimizerError::Invariant("async rollout work item has no active partial".to_string())
    })?;
    state.cursor.phase = phase_for_rollout_stage(&active.stage)?;
    let mut outcome = plan_next_rollout_batch(context, state, resources)?;
    let Some(job_id) = state.cursor.pending_job_id.clone() else {
        queue_async_active_rollout_continuation(state, item.parent_pool_version)?;
        refresh_async_pipeline_cursor_state(context, state, plan);
        persist_gepa_run_state(
            context,
            state,
            resources,
            state.cursor.phase.clone(),
            "completed",
            "folded async rollout work without new runtime job",
            Map::new(),
        )?;
        outcome.message = format!("async-pipelined rollout: {}", outcome.message);
        return Ok(Some(outcome));
    };
    let active = state.active_evaluation.clone().ok_or_else(|| {
        OptimizerError::Invariant("async rollout planning lost active partial".to_string())
    })?;
    let partial_id = item
        .partial_id
        .clone()
        .unwrap_or_else(|| async_partial_id(&active.stage, active.generation));
    upsert_async_partial_from_active(state, &partial_id, "rollout", item.parent_pool_version)?;
    let chunk_rows = rows_for_active_rollout_chunk(&context.config, resources, &active)?;
    let pending_effect_id = state.cursor.pending_effect_id.clone();
    let pending_reservation_ids = state.cursor.pending_reservation_ids.clone();
    let chunk_id = record_async_rollout_chunk(
        state,
        &partial_id,
        &active,
        &chunk_rows,
        &job_id,
        pending_effect_id,
        pending_reservation_ids,
    )?;
    let lease = GepaAsyncLaneLease {
        lease_id: async_lease_id("rollout", &job_id),
        lane: "rollout".to_string(),
        stage: active.stage.clone(),
        generation: active.generation,
        parent_pool_version: item.parent_pool_version,
        partial_id: Some(partial_id),
        job_id: Some(job_id.clone()),
        effect_id: state.cursor.pending_effect_id.clone(),
        reservation_ids: state.cursor.pending_reservation_ids.clone(),
        status: "pending".to_string(),
        metadata: json!({
            "candidate_ids": candidate_ids_for_active(&active),
            "chunk_id": chunk_id.clone(),
            "chunk_rows": chunk_rows.len(),
            "proposal_index": active.proposal_index,
        }),
    };
    state
        .cursor
        .pipeline_state
        .lane_leases
        .insert(job_id.clone(), lease);
    state.cursor.pending_job_id = None;
    state.cursor.pending_effect_id = None;
    state.cursor.pending_reservation_ids.clear();
    state.active_evaluation = None;
    context.events.emit(
        "rollout.chunk.started",
        "Rollout chunk started",
        json!({
            "chunk_id": chunk_id,
            "job_id": job_id,
            "stage": active.stage,
            "rows": chunk_rows.len(),
            "completed_rows": active_rollout_completed_rows(&active),
            "total_rows": active_rollout_total_rows(&active),
            "active_rollout_workers": async_lane_lease_count(state, "rollout"),
            "configured_rollout_workers": adaptive_rollout_worker_limit(state, plan),
            "static_rollout_workers": plan.rollout_workers,
            "adaptive_rollout_concurrency": adaptive_rollout_snapshot(state, plan),
        }),
    )?;
    refresh_async_pipeline_cursor_state(context, state, plan);
    persist_gepa_run_state(
        context,
        state,
        resources,
        state.cursor.phase.clone(),
        "planned",
        "leased async rollout job",
        Map::new(),
    )?;
    outcome.message = format!("async-pipelined rollout: {}", outcome.message);
    Ok(Some(outcome))
}

fn schedule_async_candidate_minibatches(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if state.proposal_queue.is_empty()
        || state.cursor.pipeline_state.in_flight_candidate_count >= plan.max_in_flight_candidates
    {
        return Ok(None);
    }
    if async_stage_work_pending(
        state,
        &["parent_minibatch_reference", "candidate_minibatch"],
    ) {
        return Ok(None);
    }
    state.cursor.phase = GepaCursorPhase::ProposerWaiting;
    let before_generation = state.cursor.generation;
    let mut outcome = advance_proposer_waiting(context, state, resources)?;
    if let Some(active) = state.active_evaluation.clone() {
        let partial_id = async_partial_id(&active.stage, active.generation);
        upsert_async_partial_from_active(
            state,
            &partial_id,
            "rollout",
            state.cursor.pipeline_state.pool_version,
        )?;
        let item = async_work_item_from_active(
            &active,
            "rollout",
            state.cursor.pipeline_state.pool_version,
            Some(partial_id),
        )?;
        state.cursor.pipeline_state.rollout_queue.push(item);
        state.active_evaluation = None;
        if state.cursor.proposal_index >= state.proposal_queue.len() {
            state.proposal_queue.clear();
            state.cursor.proposal_index = 0;
            state.cursor.generation = before_generation.saturating_add(1);
            state.cursor.pipeline_state.parent_candidate_id = None;
        }
        refresh_async_pipeline_cursor_state(context, state, plan);
        persist_gepa_run_state(
            context,
            state,
            resources,
            state.cursor.phase.clone(),
            "planned",
            "queued async candidate minibatch work",
            Map::new(),
        )?;
    }
    outcome.message = format!("async-pipelined candidate queue: {}", outcome.message);
    Ok(Some(outcome))
}

fn schedule_async_proposer_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinedPlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if async_lane_lease_count(state, "propose") >= plan.propose_workers
        || !state.cursor.pipeline_state.propose_queue.is_empty()
        || !state.proposal_queue.is_empty()
        || state.cursor.generation >= context.config.gepa.max_generations
        || state.cursor.pipeline_state.in_flight_candidate_count >= plan.max_in_flight_candidates
        || train_rollout_budget_reached(&context.config, state.rollout_count)
        || cost_budget_reached(&context.config, state.total_cost)
    {
        return Ok(None);
    }
    if let Some(train_best_idx) = select_best_train_candidate(
        &state.candidates,
        &resources.objective_set,
        &context.config.dataset.train_split,
        &resources.train_rows,
    )? {
        state.best_idx = Some(train_best_idx);
    }
    let parent_selection = select_proposer_parent_candidate(
        &state.candidates,
        &resources.train_rows,
        &resources.objective_set,
        &context.config.gepa.candidate_selector,
        state.cursor.generation,
        &context.config.run.run_id,
        state.best_idx,
    )?;
    let parent_idx = parent_selection.candidate_index;
    let parent_id = state
        .candidates
        .get(parent_idx)
        .map(|candidate| candidate.candidate_id.clone())
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "parent index {parent_idx} is outside candidate registry"
            ))
        })?;
    let proposer_started_details = proposer_started_details(
        &context.config,
        &state.candidates,
        state.cursor.generation,
        &parent_id,
        parent_selection.metadata.clone(),
        &context.paths.run_dir,
    );
    if context.state_machine.state() == OptimizerRunState::Ready {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Proposing,
            OptimizerTransitionTrigger::ProposerStarted,
            "Async proposer started",
            proposer_started_details.clone(),
        )?;
    } else {
        context.events.emit(
            "proposer.started",
            "Async proposer started",
            proposer_started_details,
        )?;
    }
    let queued = plan_proposer_runtime_job(context, resources, parent_idx, state)?;
    state.cursor.pipeline_state.parent_pool_version =
        Some(state.cursor.pipeline_state.pool_version);
    state.cursor.pipeline_state.parent_candidate_id = Some(parent_id.clone());
    let active = GepaActiveEvaluation {
        stage: "proposer".to_string(),
        candidate_id: Some(parent_id.clone()),
        candidate_index: Some(parent_idx),
        generation: state.cursor.generation,
        proposal_index: 0,
        row_ids: Vec::new(),
        next_row_index: 0,
        planned_job_id: Some(queued.job.job_id.clone()),
        effect_id: Some(queued.effect.runtime_effect_id.clone()),
        reservation_id: Some(queued.reservation.budget_reservation_id.clone()),
        heldout_candidate_index: None,
        parent_id: None,
        scores: Vec::new(),
        sensor_frames: Vec::new(),
        reward_sum: 0.0,
        usage: UsageTotals::default(),
        cost_usd: 0.0,
        rollout_count: 0,
        parent_minibatch_reward: None,
        decision: None,
        candidate_evaluations: Vec::new(),
    };
    let partial_id = async_partial_id("proposer", active.generation);
    state.active_evaluation = Some(active.clone());
    upsert_async_partial_from_active(
        state,
        &partial_id,
        "propose",
        state.cursor.pipeline_state.pool_version,
    )?;
    let lease = GepaAsyncLaneLease {
        lease_id: async_lease_id("propose", &queued.job.job_id),
        lane: "propose".to_string(),
        stage: "proposer".to_string(),
        generation: active.generation,
        parent_pool_version: state.cursor.pipeline_state.pool_version,
        partial_id: Some(partial_id.clone()),
        job_id: Some(queued.job.job_id.clone()),
        effect_id: Some(queued.effect.runtime_effect_id.clone()),
        reservation_ids: vec![queued.reservation.budget_reservation_id.clone()],
        status: "pending".to_string(),
        metadata: json!({
            "parent_candidate_id": parent_id,
            "parent_selection": parent_selection.metadata.clone(),
        }),
    };
    state
        .cursor
        .pipeline_state
        .lane_leases
        .insert(queued.job.job_id.clone(), lease);
    state
        .cursor
        .pipeline_state
        .propose_queue
        .push(GepaAsyncLaneWorkItem {
            item_id: partial_id,
            lane: "propose".to_string(),
            stage: "proposer".to_string(),
            generation: active.generation,
            proposal_index: 0,
            parent_candidate_id: active.candidate_id.clone(),
            parent_pool_version: state.cursor.pipeline_state.pool_version,
            current_pool_version: Some(state.cursor.pipeline_state.pool_version),
            stale_gap: Some(0),
            candidate_ids: Vec::new(),
            partial_id: active
                .candidate_id
                .as_ref()
                .map(|_| async_partial_id("proposer", active.generation)),
            job_id: Some(queued.job.job_id.clone()),
            effect_id: Some(queued.effect.runtime_effect_id.clone()),
            reservation_ids: vec![queued.reservation.budget_reservation_id.clone()],
            status: "leased".to_string(),
            metadata: json!({"parent_candidate_id": active.candidate_id}),
        });
    state.active_evaluation = None;
    refresh_async_pipeline_cursor_state(context, state, plan);
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::ProposerWaiting,
        "planned",
        "leased async proposer job",
        Map::new(),
    )?;
    Ok(Some(GepaAdvanceOutcome {
        action: planner::GepaTickAction::PlanRuntimeJob {
            run_id: context.config.run.run_id.clone(),
            job_id: queued.job.job_id,
        },
        terminal: false,
        result: None,
        message: "async-pipelined propose: planned proposer job".to_string(),
    }))
}

fn restore_async_partial_as_active(
    state: &mut GepaRunState,
    partial_id: Option<&str>,
) -> Result<()> {
    let Some(partial_id) = partial_id else {
        return Ok(());
    };
    let Some(partial) = state
        .cursor
        .pipeline_state
        .candidate_partials
        .get(partial_id)
    else {
        return Ok(());
    };
    state.active_evaluation = partial
        .active_evaluation
        .clone()
        .map(serde_json::from_value)
        .transpose()?;
    Ok(())
}

fn upsert_async_partial_from_active(
    state: &mut GepaRunState,
    partial_id: &str,
    lane: &str,
    parent_pool_version: u64,
) -> Result<()> {
    let active = state.active_evaluation.as_ref().ok_or_else(|| {
        OptimizerError::Invariant(
            "cannot persist async partial without active evaluation".to_string(),
        )
    })?;
    let rollout_chunks = state
        .cursor
        .pipeline_state
        .candidate_partials
        .get(partial_id)
        .map(|partial| partial.rollout_chunks.clone())
        .unwrap_or_default();
    state.cursor.pipeline_state.candidate_partials.insert(
        partial_id.to_string(),
        GepaAsyncCandidatePartial {
            partial_id: partial_id.to_string(),
            lane: lane.to_string(),
            stage: active.stage.clone(),
            generation: active.generation,
            parent_pool_version,
            parent_candidate_id: active.candidate_id.clone().or(active.parent_id.clone()),
            candidate_ids: candidate_ids_for_active(active),
            active_evaluation: Some(serde_json::to_value(active)?),
            proposal_queue: Value::Null,
            rollout_chunks,
            metadata: json!({
                "proposal_index": active.proposal_index,
                "is_group": active.is_group(),
            }),
        },
    );
    Ok(())
}

fn async_work_item_from_active(
    active: &GepaActiveEvaluation,
    lane: &str,
    parent_pool_version: u64,
    partial_id: Option<String>,
) -> Result<GepaAsyncLaneWorkItem> {
    let current_pool_version = Some(parent_pool_version);
    Ok(GepaAsyncLaneWorkItem {
        item_id: partial_id
            .clone()
            .unwrap_or_else(|| async_partial_id(&active.stage, active.generation)),
        lane: lane.to_string(),
        stage: active.stage.clone(),
        generation: active.generation,
        proposal_index: active.proposal_index,
        parent_candidate_id: active.parent_id.clone().or(active.candidate_id.clone()),
        parent_pool_version,
        current_pool_version,
        stale_gap: Some(0),
        candidate_ids: candidate_ids_for_active(active),
        partial_id,
        job_id: active.planned_job_id.clone(),
        effect_id: active.effect_id.clone(),
        reservation_ids: active.reservation_id.iter().cloned().collect(),
        status: "queued".to_string(),
        metadata: json!({
            "next_row_index": active.next_row_index,
            "row_count": active.row_ids.len(),
            "candidate_count": active.candidate_evaluations.len(),
        }),
    })
}

fn queue_async_active_rollout_continuation(
    state: &mut GepaRunState,
    parent_pool_version: u64,
) -> Result<bool> {
    let Some(active) = state
        .active_evaluation
        .clone()
        .filter(GepaActiveEvaluation::is_rollout_stage)
        .filter(|active| !active_rollout_evaluation_complete(active))
    else {
        state.active_evaluation = None;
        return Ok(false);
    };
    let partial_id = async_partial_id(&active.stage, active.generation);
    upsert_async_partial_from_active(state, &partial_id, "rollout", parent_pool_version)?;
    let rollout_item =
        async_work_item_from_active(&active, "rollout", parent_pool_version, Some(partial_id))?;
    state.cursor.pipeline_state.rollout_queue.push(rollout_item);
    state.active_evaluation = None;
    Ok(true)
}

fn candidate_ids_for_active(active: &GepaActiveEvaluation) -> Vec<String> {
    if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .map(|candidate| candidate.candidate_id.clone())
            .collect()
    } else {
        active.candidate_id.iter().cloned().collect()
    }
}

fn phase_for_rollout_stage(stage: &str) -> Result<GepaCursorPhase> {
    match stage {
        "seed_full_train" => Ok(GepaCursorPhase::SeedFullTrain),
        "parent_minibatch_reference" => Ok(GepaCursorPhase::CandidateMinibatch),
        "candidate_minibatch" => Ok(GepaCursorPhase::CandidateMinibatch),
        "candidate_full_train" => Ok(GepaCursorPhase::CandidateFullTrain),
        "heldout" => Ok(GepaCursorPhase::Heldout),
        _ => Err(OptimizerError::Invariant(format!(
            "async rollout stage {stage} is not supported"
        ))),
    }
}

fn async_lane_lease_count(state: &GepaRunState, lane: &str) -> usize {
    state
        .cursor
        .pipeline_state
        .lane_leases
        .values()
        .filter(|lease| lease.lane == lane)
        .count()
}

fn ensure_adaptive_rollout_concurrency_state(
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinedPlan,
) {
    let adaptive = &plan.adaptive_rollout_concurrency;
    let fallback = plan.rollout_workers.max(1);
    if !adaptive.enabled {
        state
            .cursor
            .pipeline_state
            .adaptive_rollout_concurrency
            .current_limit = fallback;
        return;
    }
    let adaptive_state = &mut state.cursor.pipeline_state.adaptive_rollout_concurrency;
    if !adaptive_state.initialized {
        adaptive_state.initialized = true;
        adaptive_state.current_limit = adaptive.initial.clamp(adaptive.min, adaptive.max);
        adaptive_state.successes_since_adjustment = 0;
    }
}

fn adaptive_rollout_worker_limit(state: &GepaRunState, plan: &GepaAsyncPipelinedPlan) -> usize {
    if !plan.adaptive_rollout_concurrency.enabled {
        return plan.rollout_workers.max(1);
    }
    state
        .cursor
        .pipeline_state
        .adaptive_rollout_concurrency
        .current_limit
        .clamp(
            plan.adaptive_rollout_concurrency.min,
            plan.adaptive_rollout_concurrency.max,
        )
        .max(1)
}

fn adaptive_rollout_snapshot(state: &GepaRunState, plan: &GepaAsyncPipelinedPlan) -> Value {
    let adaptive = &plan.adaptive_rollout_concurrency;
    let adaptive_state = &state.cursor.pipeline_state.adaptive_rollout_concurrency;
    json!({
        "enabled": adaptive.enabled,
        "current_limit": adaptive_rollout_worker_limit(state, plan),
        "initial": adaptive.initial,
        "min": adaptive.min,
        "max": adaptive.max,
        "increase_step": adaptive.increase_step,
        "decrease_step": adaptive.decrease_step,
        "increase_after_successes": adaptive.increase_after_successes,
        "successes_since_adjustment": adaptive_state.successes_since_adjustment,
        "completed_rollouts": adaptive_state.completed_rollouts,
        "overload_count": adaptive_state.overload_count,
        "last_adjustment": adaptive_state.last_adjustment,
    })
}

fn runtime_rollout_success_count(outcome: &runtime::RuntimeEffectOutcome) -> usize {
    match outcome {
        runtime::RuntimeEffectOutcome::Rollout(_) => 1,
        runtime::RuntimeEffectOutcome::RolloutBatch(outcomes) => outcomes.len(),
        runtime::RuntimeEffectOutcome::Proposer(_) => 0,
    }
}

fn record_adaptive_rollout_success(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinedPlan,
    rollout_count: usize,
) -> Result<()> {
    if rollout_count == 0 || !plan.adaptive_rollout_concurrency.enabled {
        return Ok(());
    }
    ensure_adaptive_rollout_concurrency_state(state, plan);
    let adaptive = &plan.adaptive_rollout_concurrency;
    let adaptive_state = &mut state.cursor.pipeline_state.adaptive_rollout_concurrency;
    adaptive_state.completed_rollouts = adaptive_state
        .completed_rollouts
        .saturating_add(rollout_count);
    adaptive_state.successes_since_adjustment = adaptive_state
        .successes_since_adjustment
        .saturating_add(rollout_count);
    if adaptive_state.successes_since_adjustment < adaptive.increase_after_successes {
        return Ok(());
    }
    let old_limit = adaptive_state.current_limit;
    let new_limit = old_limit
        .saturating_add(adaptive.increase_step)
        .clamp(adaptive.min, adaptive.max);
    adaptive_state.successes_since_adjustment = 0;
    if new_limit == old_limit {
        return Ok(());
    }
    adaptive_state.current_limit = new_limit;
    let adjustment = GepaAdaptiveRolloutConcurrencyAdjustment {
        direction: "up".to_string(),
        old_limit,
        new_limit,
        reason: format!(
            "healthy_after_{}_rollout_successes",
            adaptive.increase_after_successes
        ),
        completed_rollouts: adaptive_state.completed_rollouts,
    };
    adaptive_state.last_adjustment = Some(adjustment.clone());
    adaptive_state.adjustments.push(adjustment.clone());
    if adaptive_state.adjustments.len() > 32 {
        adaptive_state.adjustments.remove(0);
    }
    context.events.emit(
        "rollout.concurrency.adjusted",
        "Adaptive rollout concurrency adjusted",
        json!({
            "direction": adjustment.direction,
            "old_limit": adjustment.old_limit,
            "new_limit": adjustment.new_limit,
            "reason": adjustment.reason,
            "completed_rollouts": adjustment.completed_rollouts,
        }),
    )?;
    Ok(())
}

fn adaptive_error_is_overload(error_text: &str, plan: &GepaAsyncPipelinedPlan) -> bool {
    let lowered = error_text.to_ascii_lowercase();
    plan.adaptive_rollout_concurrency
        .overload_status_codes
        .iter()
        .any(|status| lowered.contains(&status.to_string()))
        || [
            "rate limit",
            "rate_limit",
            "overload",
            "overloaded",
            "too many requests",
            "temporarily unavailable",
            "gateway timeout",
            "timed out",
            "timeout",
        ]
        .iter()
        .any(|needle| lowered.contains(needle))
}

fn record_adaptive_rollout_overload(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinedPlan,
    error_text: &str,
) -> Result<()> {
    if !plan.adaptive_rollout_concurrency.enabled || !adaptive_error_is_overload(error_text, plan) {
        return Ok(());
    }
    ensure_adaptive_rollout_concurrency_state(state, plan);
    let adaptive = &plan.adaptive_rollout_concurrency;
    let adaptive_state = &mut state.cursor.pipeline_state.adaptive_rollout_concurrency;
    adaptive_state.overload_count = adaptive_state.overload_count.saturating_add(1);
    adaptive_state.successes_since_adjustment = 0;
    let old_limit = adaptive_state.current_limit;
    let new_limit = old_limit
        .saturating_sub(adaptive.decrease_step)
        .clamp(adaptive.min, adaptive.max);
    if new_limit == old_limit {
        return Ok(());
    }
    adaptive_state.current_limit = new_limit;
    let adjustment = GepaAdaptiveRolloutConcurrencyAdjustment {
        direction: "down".to_string(),
        old_limit,
        new_limit,
        reason: "provider_overload_or_timeout".to_string(),
        completed_rollouts: adaptive_state.completed_rollouts,
    };
    adaptive_state.last_adjustment = Some(adjustment.clone());
    adaptive_state.adjustments.push(adjustment.clone());
    if adaptive_state.adjustments.len() > 32 {
        adaptive_state.adjustments.remove(0);
    }
    context.events.emit(
        "rollout.concurrency.adjusted",
        "Adaptive rollout concurrency adjusted",
        json!({
            "direction": adjustment.direction,
            "old_limit": adjustment.old_limit,
            "new_limit": adjustment.new_limit,
            "reason": adjustment.reason,
            "completed_rollouts": adjustment.completed_rollouts,
            "error": error_text.chars().take(240).collect::<String>(),
        }),
    )?;
    Ok(())
}

fn async_pipeline_idle(state: &GepaRunState) -> bool {
    async_pipeline_has_no_lane_work(state)
        && state.proposal_queue.is_empty()
        && state.active_evaluation.is_none()
}

fn async_pipeline_has_no_lane_work(state: &GepaRunState) -> bool {
    state.cursor.pipeline_state.lane_leases.is_empty()
        && state.cursor.pipeline_state.propose_queue.is_empty()
        && state.cursor.pipeline_state.rollout_queue.is_empty()
        && state.cursor.pipeline_state.evaluate_queue.is_empty()
}

fn async_rollout_chunks_folded(state: &GepaRunState) -> bool {
    state
        .cursor
        .pipeline_state
        .candidate_partials
        .values()
        .flat_map(|partial| partial.rollout_chunks.values())
        .all(|chunk| chunk.folded || chunk.status == "folded")
}

fn async_pipeline_retry_scheduled(state: &GepaRunState) -> bool {
    state
        .cursor
        .pipeline_state
        .lane_leases
        .values()
        .any(|lease| lease.status == OptimizerJobStatus::RetryScheduled.as_str())
}

fn async_pipeline_stopper_satisfied(context: &GepaRunContext, state: &GepaRunState) -> bool {
    state.cursor.generation >= context.config.gepa.max_generations
        || train_rollout_budget_reached(&context.config, state.rollout_count)
        || cost_budget_reached(&context.config, state.total_cost)
        || score_threshold_reached(&context.config, state)
        || no_improvement_reached(&context.config, state)
}

fn service_stop_condition_reached(config: &SynthOptimizerConfig, state: &GepaRunState) -> bool {
    score_threshold_reached(config, state) || no_improvement_reached(config, state)
}

fn score_threshold_reached(config: &SynthOptimizerConfig, state: &GepaRunState) -> bool {
    let Some(threshold) = config.gepa.score_threshold_value else {
        return false;
    };
    let metric = config
        .gepa
        .score_threshold_metric
        .as_deref()
        .unwrap_or("heldout_score");
    state
        .candidates
        .iter()
        .filter_map(|candidate| candidate_metric_score(candidate, metric))
        .any(|score| score >= threshold)
}

fn no_improvement_reached(config: &SynthOptimizerConfig, state: &GepaRunState) -> bool {
    let Some(window) = config.gepa.no_improvement_generations else {
        return false;
    };
    if window == 0 || state.cursor.generation <= window {
        return false;
    }
    let metric = config
        .gepa
        .no_improvement_metric
        .as_deref()
        .unwrap_or("heldout_score");
    let mut best_by_generation: BTreeMap<usize, f64> = BTreeMap::new();
    for candidate in &state.candidates {
        let Some(generation) = candidate_generation(candidate) else {
            continue;
        };
        let Some(score) = candidate_metric_score(candidate, metric) else {
            continue;
        };
        best_by_generation
            .entry(generation)
            .and_modify(|best| *best = best.max(score))
            .or_insert(score);
    }
    if best_by_generation.len() <= window {
        return false;
    }
    let latest_generation = best_by_generation.keys().copied().max().unwrap_or(0);
    if latest_generation < window {
        return false;
    }
    let prior_best = best_by_generation
        .iter()
        .filter(|(generation, _)| **generation < latest_generation.saturating_sub(window - 1))
        .map(|(_, score)| *score)
        .reduce(f64::max);
    let Some(prior_best) = prior_best else {
        return false;
    };
    let recent_best = best_by_generation
        .iter()
        .filter(|(generation, _)| **generation >= latest_generation.saturating_sub(window - 1))
        .map(|(_, score)| *score)
        .reduce(f64::max)
        .unwrap_or(f64::NEG_INFINITY);
    recent_best <= prior_best
}

fn candidate_metric_score(candidate: &CandidateRecord, metric: &str) -> Option<f64> {
    match metric {
        "train_score" | "train_reward" => candidate.train_reward.or(candidate.minibatch_reward),
        "heldout_score" | "heldout_reward" => candidate.heldout_reward.or(candidate.train_reward),
        _ => None,
    }
}

fn candidate_generation(candidate: &CandidateRecord) -> Option<usize> {
    if candidate.source == "seed" || candidate.parent_id.is_none() {
        return Some(0);
    }
    candidate
        .acceptance_metadata
        .get("generation")
        .and_then(Value::as_u64)
        .map(|generation| generation as usize)
}

fn async_partial_id(stage: &str, generation: usize) -> String {
    format!("async:{stage}:generation_{generation:03}")
}

fn async_rollout_chunk_size(config: &SynthOptimizerConfig) -> usize {
    config
        .gepa
        .rollout_chunk_size
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            if config.gepa.pipeline.adaptive_rollout_concurrency.enabled {
                config
                    .gepa
                    .pipeline
                    .adaptive_rollout_concurrency
                    .initial
                    .clamp(1, 128)
            } else {
                config.gepa.pipeline.workers.rollout.clamp(1, 128)
            }
        })
        .max(1)
}

fn rollout_rows_for_chunk<'a>(config: &SynthOptimizerConfig, rows: &'a [Value]) -> &'a [Value] {
    if matches!(config.gepa.pipeline.mode, GepaPipelineMode::AsyncPipelined) {
        let chunk_size = async_rollout_chunk_size(config).min(rows.len()).max(1);
        &rows[..chunk_size]
    } else {
        rows
    }
}

fn take_count_for_group_rollout_chunk(
    config: &SynthOptimizerConfig,
    remaining_len: usize,
    remaining_budget: &mut Option<usize>,
) -> usize {
    if !matches!(config.gepa.pipeline.mode, GepaPipelineMode::AsyncPipelined) {
        return remaining_len;
    }
    let Some(budget) = remaining_budget.as_mut() else {
        return remaining_len;
    };
    let count = remaining_len.min(*budget);
    *budget = budget.saturating_sub(count);
    count
}

fn rows_for_active_rollout_chunk(
    config: &SynthOptimizerConfig,
    resources: &GepaStepResources,
    active: &GepaActiveEvaluation,
) -> Result<Vec<Value>> {
    if active.is_group() {
        let mut rows = Vec::new();
        let mut remaining_budget = Some(async_rollout_chunk_size(config));
        for candidate_eval in &active.candidate_evaluations {
            let stage_rows = rows_for_rollout_stage(
                config,
                resources,
                &active.stage,
                candidate_eval.generation,
                candidate_eval.proposal_index,
            );
            let remaining = stage_rows
                .get(candidate_eval.next_row_index..)
                .ok_or_else(|| {
                    OptimizerError::Invariant(format!(
                        "active evaluation stage {} row index {} is outside {} rows",
                        active.stage,
                        candidate_eval.next_row_index,
                        stage_rows.len()
                    ))
                })?;
            let take_count =
                take_count_for_group_rollout_chunk(config, remaining.len(), &mut remaining_budget);
            if take_count == 0 {
                break;
            }
            rows.extend(remaining[..take_count].iter().cloned());
        }
        return Ok(rows);
    }
    let rows = rows_for_rollout_stage(
        config,
        resources,
        &active.stage,
        active.generation,
        active.proposal_index,
    );
    let remaining = rows.get(active.next_row_index..).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "active evaluation stage {} row index {} is outside {} rows",
            active.stage,
            active.next_row_index,
            rows.len()
        ))
    })?;
    Ok(rollout_rows_for_chunk(config, remaining).to_vec())
}

fn row_ids_for_chunk(rows: &[Value]) -> Result<Vec<String>> {
    rows.iter().map(row_example_id).collect()
}

fn async_rollout_chunk_id(active: &GepaActiveEvaluation, rows: &[Value]) -> Result<String> {
    let first_row_index = if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .map(|candidate| candidate.next_row_index)
            .min()
            .unwrap_or(active.next_row_index)
    } else {
        active.next_row_index
    };
    let row_ids = row_ids_for_chunk(rows)?;
    let candidate_ids = candidate_ids_for_active(active);
    Ok(stable_gepa_id(
        "gepa_chunk",
        &[
            &active.stage,
            &active.generation.to_string(),
            &active.proposal_index.to_string(),
            &first_row_index.to_string(),
            &candidate_ids.join(","),
            &row_ids.join(","),
        ],
    ))
}

fn stable_gepa_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}

fn record_async_rollout_chunk(
    state: &mut GepaRunState,
    partial_id: &str,
    active: &GepaActiveEvaluation,
    rows: &[Value],
    job_id: &str,
    effect_id: Option<String>,
    reservation_ids: Vec<String>,
) -> Result<String> {
    let chunk_id = async_rollout_chunk_id(active, rows)?;
    let first_row_index = if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .map(|candidate| candidate.next_row_index)
            .min()
            .unwrap_or(active.next_row_index)
    } else {
        active.next_row_index
    };
    let partial = state
        .cursor
        .pipeline_state
        .candidate_partials
        .get_mut(partial_id)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "async rollout partial {partial_id} missing while recording chunk"
            ))
        })?;
    partial.rollout_chunks.insert(
        chunk_id.clone(),
        GepaRolloutChunkPartial {
            chunk_id: chunk_id.clone(),
            stage: active.stage.clone(),
            generation: active.generation,
            candidate_ids: candidate_ids_for_active(active),
            row_ids: row_ids_for_chunk(rows)?,
            first_row_index,
            row_count: rows.len(),
            job_id: Some(job_id.to_string()),
            effect_id,
            reservation_ids,
            status: "leased".to_string(),
            folded: false,
            attempt: 0,
            metadata: json!({
                "chunk_size": rows.len(),
            }),
        },
    );
    Ok(chunk_id)
}

fn async_lease_id(lane: &str, job_id: &str) -> String {
    format!("async:{lane}:{job_id}")
}

fn advance_pending_runtime_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    _mode: GepaAdvanceMode,
    job_id: &str,
    async_plan: Option<&GepaAsyncPipelinedPlan>,
) -> Result<GepaAdvanceOutcome> {
    let job = context
        .workspace
        .optimizer_job(&context.config.run.run_id, job_id)?;
    match job.status {
        OptimizerJobStatus::Pending | OptimizerJobStatus::RetryScheduled => {
            let runtime_started = Instant::now();
            let outcome = match runtime::execute_one_pending_optimizer_job_from_run_workspace(
                &context.workspace,
                &mut context.cache,
                &context.config,
                &resources.client,
                &context.config.run.run_id,
                job_id,
                runtime::RuntimeEffectExecutorConfig::inline_default(),
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    if let Some(plan) = async_plan {
                        if job.payload.get("lane").and_then(Value::as_str) == Some("rollout") {
                            record_adaptive_rollout_overload(
                                context,
                                state,
                                plan,
                                &error.to_string(),
                            )?;
                        }
                    }
                    if let Ok(updated_job) = context
                        .workspace
                        .optimizer_job(&context.config.run.run_id, job_id)
                    {
                        if matches!(updated_job.status, OptimizerJobStatus::RetryScheduled) {
                            persist_gepa_run_state(
                                context,
                                state,
                                resources,
                                state.cursor.phase.clone(),
                                "retry_scheduled",
                                "scheduled GEPA runtime job retry",
                                Map::new(),
                            )?;
                            return Ok(GepaAdvanceOutcome {
                                action: planner::GepaTickAction::Noop,
                                terminal: false,
                                result: None,
                                message: format!(
                                    "scheduled GEPA runtime job retry: {}",
                                    updated_job.job_id
                                ),
                            });
                        }
                        if updated_job.status.is_terminal() {
                            return consume_failed_runtime_job(
                                context,
                                state,
                                resources,
                                updated_job,
                            );
                        }
                    }
                    return terminalize_aborted_gepa_run(
                        context,
                        state,
                        error,
                        "GEPA runtime job failed",
                    );
                }
            };
            let wall_seconds = runtime_started.elapsed().as_secs_f64();
            emit_runtime_job_completed_event(context, state, job_id, &job, &outcome, wall_seconds)?;
            if let Some(plan) = async_plan {
                let rollout_count = runtime_rollout_success_count(&outcome);
                record_adaptive_rollout_success(context, state, plan, rollout_count)?;
            }
            let stored = stored_runtime_outcome(&outcome)?;
            let mut updated_job = context
                .workspace
                .optimizer_job(&context.config.run.run_id, job_id)?;
            updated_job
                .payload
                .insert("runtime_outcome".to_string(), serde_json::to_value(stored)?);
            context.workspace.record_optimizer_job(&updated_job)?;
            if let Some(active) = state.active_evaluation.as_mut() {
                active.planned_job_id = Some(job_id.to_string());
            }
            persist_gepa_run_state(
                context,
                state,
                resources,
                state.cursor.phase.clone(),
                "running",
                "executed GEPA runtime job",
                Map::new(),
            )?;
            Ok(GepaAdvanceOutcome {
                action: planner::GepaTickAction::ExecuteRuntimeJob {
                    run_id: context.config.run.run_id.clone(),
                    job_id: job_id.to_string(),
                },
                terminal: false,
                result: None,
                message: "executed GEPA runtime job".to_string(),
            })
        }
        OptimizerJobStatus::Completed => {
            consume_completed_runtime_job(context, state, resources, job)
        }
        OptimizerJobStatus::Failed
        | OptimizerJobStatus::Cancelled
        | OptimizerJobStatus::Expired => consume_failed_runtime_job(context, state, resources, job),
        _ => Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::Noop,
            terminal: false,
            result: None,
            message: format!(
                "runtime job {} is already {}",
                job.job_id,
                job.status.as_str()
            ),
        }),
    }
}

fn stored_runtime_outcome(outcome: &runtime::RuntimeEffectOutcome) -> Result<StoredRuntimeOutcome> {
    Ok(match outcome {
        runtime::RuntimeEffectOutcome::Proposer(outcome) => StoredRuntimeOutcome::Proposer {
            response: outcome.response.clone(),
            proposals: outcome.proposals.clone(),
            usage: outcome.usage.clone(),
            cost_usd: outcome.cost_usd,
            backend: outcome.backend.clone(),
            workspace: outcome.workspace.clone(),
        },
        runtime::RuntimeEffectOutcome::Rollout(outcome) => StoredRuntimeOutcome::Rollout {
            response: outcome.response.clone(),
            reward: outcome.reward,
            usage: outcome.usage.clone(),
            cost_usd: outcome.cost_usd,
            cache_key: outcome.cache_key.clone(),
            cache_hit: outcome.cache_hit,
            stage: outcome.stage.clone(),
            example_id: outcome.example_id.clone(),
            dispatch_wall_seconds: outcome.dispatch_wall_seconds,
            dispatch_chunk_index: outcome.dispatch_chunk_index,
            dispatch_chunk_size: outcome.dispatch_chunk_size,
        },
        runtime::RuntimeEffectOutcome::RolloutBatch(outcomes) => {
            StoredRuntimeOutcome::RolloutBatch {
                outcomes: outcomes
                    .iter()
                    .map(|outcome| StoredRolloutOutcome {
                        candidate_id: outcome.candidate_id.clone(),
                        response: outcome.response.clone(),
                        reward: outcome.reward,
                        usage: outcome.usage.clone(),
                        cost_usd: outcome.cost_usd,
                        cache_key: outcome.cache_key.clone(),
                        cache_hit: outcome.cache_hit,
                        stage: outcome.stage.clone(),
                        example_id: outcome.example_id.clone(),
                        dispatch_wall_seconds: outcome.dispatch_wall_seconds,
                        dispatch_chunk_index: outcome.dispatch_chunk_index,
                        dispatch_chunk_size: outcome.dispatch_chunk_size,
                    })
                    .collect(),
            }
        }
    })
}

fn emit_runtime_job_completed_event(
    context: &mut GepaRunContext,
    state: &GepaRunState,
    job_id: &str,
    job: &OptimizerJob,
    outcome: &runtime::RuntimeEffectOutcome,
    wall_seconds: f64,
) -> Result<()> {
    let mut fields = Map::new();
    fields.insert("job_id".to_string(), json!(job_id));
    if let Some(runtime_effect_id) = job.payload.get("runtime_effect_id").and_then(Value::as_str) {
        fields.insert("runtime_effect_id".to_string(), json!(runtime_effect_id));
    }
    if let Some(effect_kind) = job.payload.get("effect_kind").and_then(Value::as_str) {
        fields.insert("effect_kind".to_string(), json!(effect_kind));
    }
    if let Some(lane) = job.payload.get("lane").and_then(Value::as_str) {
        fields.insert("lane".to_string(), json!(lane));
    }
    let adaptive = &context.config.gepa.pipeline.adaptive_rollout_concurrency;
    let adaptive_state = &state.cursor.pipeline_state.adaptive_rollout_concurrency;
    let configured_rollout_workers = if adaptive.enabled {
        adaptive_state
            .current_limit
            .clamp(adaptive.min, adaptive.max)
            .max(1)
    } else {
        context.config.gepa.pipeline.workers.rollout.max(1)
    };
    fields.insert(
        "configured_rollout_workers".to_string(),
        json!(configured_rollout_workers),
    );
    fields.insert(
        "static_rollout_workers".to_string(),
        json!(context.config.gepa.pipeline.workers.rollout),
    );
    fields.insert(
        "adaptive_rollout_concurrency".to_string(),
        json!({
            "enabled": adaptive.enabled,
            "current_limit": configured_rollout_workers,
            "initial": adaptive.initial,
            "min": adaptive.min,
            "max": adaptive.max,
            "increase_step": adaptive.increase_step,
            "decrease_step": adaptive.decrease_step,
            "increase_after_successes": adaptive.increase_after_successes,
            "successes_since_adjustment": adaptive_state.successes_since_adjustment,
            "completed_rollouts": adaptive_state.completed_rollouts,
            "overload_count": adaptive_state.overload_count,
            "last_adjustment": adaptive_state.last_adjustment,
        }),
    );
    fields.insert(
        "rollout_submission_mode".to_string(),
        json!(context.config.gepa.rollout_submission_mode),
    );
    if let Some(active) = state.active_evaluation.as_ref() {
        fields.insert("generation".to_string(), json!(active.generation));
        fields.insert("active_stage".to_string(), json!(active.stage));
        fields.insert("proposal_index".to_string(), json!(active.proposal_index));
    }
    fields.insert("wall_seconds".to_string(), json!(wall_seconds));

    match outcome {
        runtime::RuntimeEffectOutcome::Proposer(outcome) => {
            fields.insert("runtime_kind".to_string(), json!("proposer"));
            fields.insert("model".to_string(), json!(context.config.proposer.model));
            fields.insert("proposal_count".to_string(), json!(outcome.proposals.len()));
            fields.insert("backend".to_string(), json!(&outcome.backend));
            fields.insert("cache_hit".to_string(), json!(outcome.cache_hit));
            fields.insert("cost_usd".to_string(), json!(outcome.cost_usd));
            fields.insert("usage".to_string(), serde_json::to_value(&outcome.usage)?);
            fields.insert(
                "total_tokens".to_string(),
                json!(outcome.usage.total_tokens),
            );
        }
        runtime::RuntimeEffectOutcome::Rollout(outcome) => {
            let mut candidate_usage = BTreeMap::<String, RuntimeUsageBucket>::new();
            let mut bucket = RuntimeUsageBucket {
                model: Some(context.config.policy.model.clone()),
                ..Default::default()
            };
            bucket.add_usage_totals(&outcome.usage);
            bucket.calls = 1;
            bucket.jobs = 1;
            bucket.cost_usd = outcome.cost_usd;
            bucket.wall_seconds = outcome.dispatch_wall_seconds.unwrap_or(wall_seconds);
            candidate_usage.insert(outcome.candidate_id.clone(), bucket);
            fields.insert("runtime_kind".to_string(), json!("rollout"));
            fields.insert("model".to_string(), json!(&context.config.policy.model));
            fields.insert("stage".to_string(), json!(&outcome.stage));
            fields.insert("candidate_id".to_string(), json!(&outcome.candidate_id));
            fields.insert("example_id".to_string(), json!(&outcome.example_id));
            fields.insert("rollout_count".to_string(), json!(1));
            fields.insert(
                "cache_hits".to_string(),
                json!(usize::from(outcome.cache_hit)),
            );
            fields.insert(
                "cache_misses".to_string(),
                json!(usize::from(!outcome.cache_hit)),
            );
            fields.insert(
                "avg_wall_seconds_per_rollout".to_string(),
                json!(wall_seconds),
            );
            fields.insert("cost_usd".to_string(), json!(outcome.cost_usd));
            fields.insert("usage".to_string(), serde_json::to_value(&outcome.usage)?);
            fields.insert(
                "total_tokens".to_string(),
                json!(outcome.usage.total_tokens),
            );
            fields.insert(
                "candidate_usage".to_string(),
                serde_json::to_value(candidate_usage)?,
            );
            if let Some(dispatch_wall_seconds) = outcome.dispatch_wall_seconds {
                fields.insert(
                    "uncached_dispatch_wall_seconds".to_string(),
                    json!(dispatch_wall_seconds),
                );
                fields.insert(
                    "uncached_latency_max_seconds".to_string(),
                    json!(dispatch_wall_seconds),
                );
                fields.insert("estimated_effective_concurrency".to_string(), json!(1.0));
            }
        }
        runtime::RuntimeEffectOutcome::RolloutBatch(outcomes) => {
            let mut usage = UsageTotals::default();
            let mut cost_usd = 0.0;
            let mut cache_hits = 0usize;
            let mut candidate_ids = BTreeSet::new();
            let mut candidate_usage = BTreeMap::<String, RuntimeUsageBucket>::new();
            let mut stages = BTreeMap::<String, usize>::new();
            let mut dispatch_latencies = Vec::new();
            let mut max_chunk_index = None::<usize>;
            let mut max_chunk_size = 0usize;
            for outcome in outcomes {
                usage.merge(&outcome.usage);
                cost_usd += outcome.cost_usd;
                if outcome.cache_hit {
                    cache_hits += 1;
                } else if let Some(dispatch_wall_seconds) = outcome.dispatch_wall_seconds {
                    dispatch_latencies.push(dispatch_wall_seconds);
                }
                candidate_ids.insert(outcome.candidate_id.clone());
                let bucket = candidate_usage
                    .entry(outcome.candidate_id.clone())
                    .or_default();
                if bucket.model.is_none() {
                    bucket.model = Some(context.config.policy.model.clone());
                }
                bucket.add_usage_totals(&outcome.usage);
                bucket.calls = bucket.calls.saturating_add(1);
                bucket.cost_usd += outcome.cost_usd;
                bucket.wall_seconds += outcome.dispatch_wall_seconds.unwrap_or(0.0);
                *stages.entry(outcome.stage.clone()).or_insert(0) += 1;
                if let Some(chunk_index) = outcome.dispatch_chunk_index {
                    max_chunk_index =
                        Some(max_chunk_index.map_or(chunk_index, |max| max.max(chunk_index)));
                }
                if let Some(chunk_size) = outcome.dispatch_chunk_size {
                    max_chunk_size = max_chunk_size.max(chunk_size);
                }
            }
            for bucket in candidate_usage.values_mut() {
                bucket.jobs = bucket.jobs.saturating_add(1);
            }
            let rollout_count = outcomes.len();
            dispatch_latencies.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            fields.insert("runtime_kind".to_string(), json!("rollout_batch"));
            fields.insert("model".to_string(), json!(&context.config.policy.model));
            fields.insert("rollout_count".to_string(), json!(rollout_count));
            fields.insert("candidate_count".to_string(), json!(candidate_ids.len()));
            fields.insert("candidate_ids".to_string(), json!(candidate_ids));
            if stages.len() == 1 {
                if let Some(stage) = stages.keys().next() {
                    fields.insert("stage".to_string(), json!(stage));
                }
            }
            fields.insert("stages".to_string(), json!(stages));
            fields.insert("cache_hits".to_string(), json!(cache_hits));
            fields.insert(
                "cache_misses".to_string(),
                json!(rollout_count.saturating_sub(cache_hits)),
            );
            if rollout_count > 0 {
                fields.insert(
                    "avg_wall_seconds_per_rollout".to_string(),
                    json!(wall_seconds / rollout_count as f64),
                );
            }
            fields.insert("cost_usd".to_string(), json!(cost_usd));
            fields.insert("usage".to_string(), serde_json::to_value(&usage)?);
            fields.insert("total_tokens".to_string(), json!(usage.total_tokens));
            fields.insert(
                "candidate_usage".to_string(),
                serde_json::to_value(candidate_usage)?,
            );
            if !dispatch_latencies.is_empty() {
                let estimated_serial_wall_seconds = dispatch_latencies.iter().sum::<f64>();
                let effective_concurrency = if wall_seconds > 0.0 {
                    estimated_serial_wall_seconds / wall_seconds
                } else {
                    0.0
                };
                fields.insert(
                    "uncached_latency_p50_seconds".to_string(),
                    json!(percentile_sorted(&dispatch_latencies, 0.50)),
                );
                fields.insert(
                    "uncached_latency_p95_seconds".to_string(),
                    json!(percentile_sorted(&dispatch_latencies, 0.95)),
                );
                fields.insert(
                    "uncached_latency_max_seconds".to_string(),
                    json!(dispatch_latencies.last().copied().unwrap_or(0.0)),
                );
                fields.insert(
                    "estimated_serial_wall_seconds".to_string(),
                    json!(estimated_serial_wall_seconds),
                );
                fields.insert(
                    "estimated_effective_concurrency".to_string(),
                    json!(effective_concurrency),
                );
                fields.insert("max_dispatch_chunk_size".to_string(), json!(max_chunk_size));
                if let Some(max_chunk_index) = max_chunk_index {
                    fields.insert(
                        "dispatch_chunk_count".to_string(),
                        json!(max_chunk_index.saturating_add(1)),
                    );
                }
            }
        }
    }

    let warning_fields = runtime_throughput_warning_fields(&fields);
    context.events.emit(
        "runtime.job.completed",
        "Runtime job completed",
        Value::Object(fields),
    )?;
    if let Some(warning_fields) = warning_fields {
        context.events.emit(
            "runtime.throughput.warning",
            "Runtime throughput lower than expected",
            Value::Object(warning_fields),
        )?;
    }
    Ok(())
}

fn runtime_throughput_warning_fields(fields: &Map<String, Value>) -> Option<Map<String, Value>> {
    let runtime_kind = fields.get("runtime_kind").and_then(Value::as_str)?;
    if !matches!(runtime_kind, "rollout" | "rollout_batch") {
        return None;
    }
    let cache_misses = fields
        .get("cache_misses")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let wall_seconds = fields
        .get("wall_seconds")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let workers = fields
        .get("configured_rollout_workers")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1);
    if cache_misses < workers || wall_seconds <= 10.0 {
        return None;
    }
    let observed_per_second = cache_misses as f64 / wall_seconds;
    let expected_min_per_second = workers as f64 * 0.05;
    let effective_concurrency = fields
        .get("estimated_effective_concurrency")
        .and_then(Value::as_f64);
    if observed_per_second >= expected_min_per_second {
        return None;
    }
    let mut warning = Map::new();
    for key in [
        "runtime_kind",
        "stage",
        "rollout_count",
        "cache_hits",
        "cache_misses",
        "wall_seconds",
        "configured_rollout_workers",
        "rollout_submission_mode",
        "job_id",
        "generation",
        "uncached_latency_p50_seconds",
        "uncached_latency_p95_seconds",
        "uncached_latency_max_seconds",
        "estimated_serial_wall_seconds",
        "estimated_effective_concurrency",
        "dispatch_chunk_count",
        "max_dispatch_chunk_size",
    ] {
        if let Some(value) = fields.get(key) {
            warning.insert(key.to_string(), value.clone());
        }
    }
    warning.insert(
        "observed_uncached_rollouts_per_second".to_string(),
        json!(observed_per_second),
    );
    warning.insert(
        "expected_min_uncached_rollouts_per_second".to_string(),
        json!(expected_min_per_second),
    );
    warning.insert(
        "diagnostic".to_string(),
        json!(rollout_throughput_diagnostic(
            effective_concurrency,
            workers as f64
        )),
    );
    Some(warning)
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index =
        ((values.len().saturating_sub(1)) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[index.min(values.len().saturating_sub(1))]
}

fn rollout_throughput_diagnostic(effective_concurrency: Option<f64>, workers: f64) -> String {
    let Some(effective_concurrency) = effective_concurrency else {
        return "rollout throughput is low for the configured worker count; check container semaphore, provider throttling, or synchronous container bottlenecks".to_string();
    };
    if effective_concurrency < workers * 0.25 {
        format!(
            "rollout throughput is low and estimated effective concurrency is {:.1} vs configured workers {:.0}; check container semaphore, provider throttling, synchronous container bottlenecks, or rollout chunking",
            effective_concurrency, workers
        )
    } else {
        format!(
            "rollout throughput is low despite estimated effective concurrency {:.1} vs configured workers {:.0}; likely high per-rollout provider latency or provider throttling",
            effective_concurrency, workers
        )
    }
}

fn runtime_outcome_from_job(job: &OptimizerJob) -> Result<StoredRuntimeOutcome> {
    let value = job.payload.get("runtime_outcome").cloned().ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "completed GEPA runtime job {} has no runtime_outcome payload",
            job.job_id
        ))
    })?;
    serde_json::from_value(value).map_err(OptimizerError::from)
}

fn advance_initializing(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    if state.candidates.is_empty() {
        let seed_payload = seed_candidate_payload(&context.config, &resources.program)?;
        let seed_id = candidate_id(&seed_payload);
        let seed_bundle = LeverBundle::from_prompt_payload(seed_id.clone(), None, &seed_payload);
        state.candidates.push(CandidateRecord {
            candidate_id: seed_id.clone(),
            payload: seed_payload,
            lever_bundle: seed_bundle,
            parent_id: None,
            source: "seed".to_string(),
            status: "registered".to_string(),
            minibatch_reward: None,
            train_reward: None,
            heldout_reward: None,
            minibatch_scores: Vec::new(),
            train_scores: Vec::new(),
            sensor_frames: Vec::new(),
            acceptance_score: Value::Null,
            acceptance_metadata: Map::new(),
        });
        context.events.emit(
            "candidate.registered",
            "Seed candidate registered",
            json!({"candidate_id": state.candidates[0].candidate_id, "source": "seed"}),
        )?;
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            &state.candidates[0],
        )?;
        let mut metadata = Map::new();
        metadata.insert("stage".to_string(), Value::String("run_start".to_string()));
        metadata.insert(
            "max_generations".to_string(),
            json!(context.config.gepa.max_generations),
        );
        metadata.insert(
            "proposals_per_generation".to_string(),
            json!(context.config.gepa.proposals_per_generation),
        );
        push_stopper_snapshot(
            &mut state.stopper_states,
            &mut state.stopper_sequence,
            &context.config,
            StopperSnapshot {
                status: "within_budget",
                reason: Some("run initialized within budget"),
                generation: None,
                candidate_id: None,
                evaluation_stage: Some("run_start"),
                rollout_count: state.rollout_count,
                cost_usd: state.total_cost,
                metadata,
            },
        );
    }
    state.best_idx = None;
    state.cursor.generation = 0;
    state.cursor.proposal_index = 0;
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &context.config,
        candidates: &state.candidates,
        frontier: Vec::new(),
        best_idx: None,
        state_machine: &context.state_machine,
        rollout_count: state.rollout_count,
        total_usage: &state.total_usage,
        total_cost: state.total_cost,
    });
    let mut checkpoint_metadata = Map::new();
    checkpoint_metadata.insert(
        "stage".to_string(),
        Value::String("seed_registered".to_string()),
    );
    record_checkpoint_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        &mut state.checkpoint_sequence,
        &context.state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "candidate_registry",
            status: "completed",
            reason: Some("seed candidate registered"),
            generation: None,
            candidate_id: Some(&state.candidates[0].candidate_id),
            evaluation_stage: Some("seed_registered"),
            best_candidate_id: None,
            candidate_count: state.candidates.len(),
            frontier_count: 0,
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            usage: serde_json::to_value(&state.total_usage)?,
            snapshot,
            metadata: checkpoint_metadata,
        },
    )?;
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::SeedFullTrain,
        "completed",
        "seed candidate registered",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::SetupRun {
            run_id: context.config.run.run_id.clone(),
        },
        terminal: false,
        result: None,
        message: "seed candidate registered".to_string(),
    })
}

fn advance_rollout_stage(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    expected_stage: &str,
) -> Result<GepaAdvanceOutcome> {
    if state
        .active_evaluation
        .as_ref()
        .is_some_and(|active| active.stage == expected_stage)
    {
        if state
            .active_evaluation
            .as_ref()
            .is_some_and(active_rollout_evaluation_complete)
        {
            return finalize_active_rollout_evaluation(context, state, resources);
        }
        return plan_next_rollout_batch(context, state, resources);
    }
    match expected_stage {
        "seed_full_train" => {
            if state
                .candidates
                .first()
                .and_then(|candidate| candidate.train_reward)
                .is_some()
            {
                state.best_idx = Some(0);
                return move_to_generation_start(
                    context,
                    state,
                    resources,
                    "seed already evaluated",
                );
            }
            let capacity = remaining_train_rollout_capacity(
                &context.workspace,
                &context.config,
                state.rollout_count,
            )?;
            if capacity < resources.train_rows.len() {
                return Err(rollout_budget_exceeded_error(
                    &context.config.run.run_id,
                    rollout_budget_limit_name(&context.config),
                    resources.train_rows.len(),
                    capacity,
                ));
            }
            if let Some(breach) = next_rollout_budget_breach(&context.workspace, &context.config)? {
                return Err(budget_exceeded_error(&context.config.run.run_id, &breach));
            }
            transition_to_rollout_running(
                context,
                "Seed candidate rollouts started",
                json!({
                    "candidate_id": state.candidates[0].candidate_id,
                    "stage": "seed_full_train",
                    "row_count": resources.train_rows.len(),
                    "rollout_count": resources.train_rows.len(),
                }),
            )?;
            state.active_evaluation = Some(new_rollout_evaluation(
                "seed_full_train",
                0,
                &resources.train_rows,
                state.cursor.generation,
                state.cursor.proposal_index,
                None,
            )?);
            persist_gepa_run_state(
                context,
                state,
                resources,
                GepaCursorPhase::SeedFullTrain,
                "planned",
                "seed full-train evaluation started",
                Map::new(),
            )?;
            plan_next_rollout_batch(context, state, resources)
        }
        "candidate_minibatch" | "candidate_full_train" => {
            if state.active_evaluation.is_none() {
                return Err(OptimizerError::Invariant(format!(
                    "phase {expected_stage} has no active candidate evaluation"
                )));
            }
            plan_next_rollout_batch(context, state, resources)
        }
        _ => Err(OptimizerError::Invariant(format!(
            "unsupported rollout stage {expected_stage}"
        ))),
    }
}

fn active_rollout_evaluation_complete(active: &GepaActiveEvaluation) -> bool {
    if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .all(|candidate| candidate.next_row_index >= candidate.row_ids.len())
    } else {
        active.next_row_index >= active.row_ids.len()
    }
}

fn active_rollout_completed_rows(active: &GepaActiveEvaluation) -> usize {
    if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .map(|candidate| candidate.next_row_index.min(candidate.row_ids.len()))
            .sum()
    } else {
        active.next_row_index.min(active.row_ids.len())
    }
}

fn active_rollout_total_rows(active: &GepaActiveEvaluation) -> usize {
    if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .map(|candidate| candidate.row_ids.len())
            .sum()
    } else {
        active.row_ids.len()
    }
}

fn new_rollout_evaluation(
    stage: &str,
    candidate_index: usize,
    rows: &[Value],
    generation: usize,
    proposal_index: usize,
    heldout_candidate_index: Option<usize>,
) -> Result<GepaActiveEvaluation> {
    let row_ids = rows
        .iter()
        .map(row_example_id)
        .collect::<Result<Vec<String>>>()?;
    Ok(GepaActiveEvaluation {
        stage: stage.to_string(),
        candidate_id: None,
        candidate_index: Some(candidate_index),
        generation,
        proposal_index,
        row_ids,
        next_row_index: 0,
        planned_job_id: None,
        effect_id: None,
        reservation_id: None,
        heldout_candidate_index,
        parent_id: None,
        scores: Vec::new(),
        sensor_frames: Vec::new(),
        reward_sum: 0.0,
        usage: UsageTotals::default(),
        cost_usd: 0.0,
        rollout_count: 0,
        parent_minibatch_reward: None,
        decision: None,
        candidate_evaluations: Vec::new(),
    })
}

fn new_active_candidate_evaluation(
    candidate_id: String,
    candidate_index: usize,
    _stage: &str,
    rows: &[Value],
    generation: usize,
    proposal_index: usize,
    heldout_candidate_index: Option<usize>,
) -> Result<GepaActiveCandidateEvaluation> {
    let row_ids = rows
        .iter()
        .map(row_example_id)
        .collect::<Result<Vec<String>>>()?;
    Ok(GepaActiveCandidateEvaluation {
        candidate_id,
        candidate_index,
        generation,
        proposal_index,
        row_ids,
        next_row_index: 0,
        heldout_candidate_index,
        parent_id: None,
        scores: Vec::new(),
        sensor_frames: Vec::new(),
        reward_sum: 0.0,
        usage: UsageTotals::default(),
        cost_usd: 0.0,
        rollout_count: 0,
        parent_minibatch_reward: None,
        decision: None,
    })
}

fn new_rollout_group_evaluation(
    stage: &str,
    candidate_evaluations: Vec<GepaActiveCandidateEvaluation>,
    generation: usize,
) -> GepaActiveEvaluation {
    let row_ids = candidate_evaluations
        .iter()
        .flat_map(|candidate| {
            candidate
                .row_ids
                .iter()
                .map(|row_id| format!("{}:{row_id}", candidate.candidate_id))
        })
        .collect();
    GepaActiveEvaluation {
        stage: stage.to_string(),
        candidate_id: None,
        candidate_index: None,
        generation,
        proposal_index: 0,
        row_ids,
        next_row_index: 0,
        planned_job_id: None,
        effect_id: None,
        reservation_id: None,
        heldout_candidate_index: None,
        parent_id: None,
        scores: Vec::new(),
        sensor_frames: Vec::new(),
        reward_sum: 0.0,
        usage: UsageTotals::default(),
        cost_usd: 0.0,
        rollout_count: 0,
        parent_minibatch_reward: None,
        decision: None,
        candidate_evaluations,
    }
}

fn transition_to_rollout_running(
    context: &mut GepaRunContext,
    message: &str,
    details: Value,
) -> Result<()> {
    if matches!(
        context.state_machine.state(),
        OptimizerRunState::Ready | OptimizerRunState::Evaluating
    ) {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::RolloutsQueued,
            message,
            details.clone(),
        )?;
    }
    if context.state_machine.state() == OptimizerRunState::RolloutQueueing {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::RolloutRunning,
            OptimizerTransitionTrigger::RolloutsStarted,
            message,
            details,
        )?;
    }
    Ok(())
}

fn plan_next_rollout_batch(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    let active = state.active_evaluation.as_ref().ok_or_else(|| {
        OptimizerError::Invariant("cannot plan rollout batch without active evaluation".to_string())
    })?;
    if !active.is_rollout_stage() {
        return Err(OptimizerError::Invariant(format!(
            "active evaluation stage {} is not a rollout stage",
            active.stage
        )));
    }
    if active.is_group() {
        return plan_next_rollout_group_batch(context, state, resources);
    }
    let candidate_index = active.candidate_index.ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "active evaluation stage {} has no candidate_index",
            active.stage
        ))
    })?;
    let candidate = state.candidates.get(candidate_index).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "active evaluation candidate index {candidate_index} is outside candidate registry"
        ))
    })?;
    let rows = rows_for_rollout_stage(
        &context.config,
        resources,
        &active.stage,
        active.generation,
        active.proposal_index,
    );
    if active.next_row_index == 0 {
        transition_to_rollout_running(
            context,
            match active.stage.as_str() {
                "seed_full_train" => "Seed candidate rollouts started",
                "parent_minibatch_reference" => "Parent minibatch reference rollouts started",
                "candidate_minibatch" => "Candidate minibatch rollouts started",
                "candidate_full_train" => "Candidate full-train rollouts started",
                "heldout" => "Heldout rollouts started",
                _ => "Rollouts started",
            },
            json!({
                "candidate_id": candidate.candidate_id,
                "generation": active.generation,
                "stage": active.stage,
                "row_count": rows.len(),
            }),
        )?;
    }
    let remaining_rows = rows.get(active.next_row_index..).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "active evaluation stage {} row index {} is outside {} rows",
            active.stage,
            active.next_row_index,
            rows.len()
        ))
    })?;
    if remaining_rows.is_empty() {
        return finalize_active_rollout_evaluation(context, state, resources);
    }
    let chunk_rows = rollout_rows_for_chunk(&context.config, remaining_rows);
    let chunk_id = async_rollout_chunk_id(active, chunk_rows)?;
    let chunk_row_count = chunk_rows.len();
    let queued =
        plan_rollout_runtime_batch_job(context, resources, candidate, chunk_rows, &active.stage)?;
    let active = state.active_evaluation.as_mut().ok_or_else(|| {
        OptimizerError::Invariant(
            "active evaluation disappeared while planning rollout".to_string(),
        )
    })?;
    active.candidate_id = Some(candidate.candidate_id.clone());
    active.planned_job_id = Some(queued.job.job_id.clone());
    active.effect_id = Some(queued.effect.runtime_effect_id.clone());
    active.reservation_id = Some(queued.reservation.budget_reservation_id.clone());
    state.cursor.pending_job_id = Some(queued.job.job_id.clone());
    state.cursor.pending_effect_id = Some(queued.effect.runtime_effect_id.clone());
    state.cursor.pending_reservation_ids = vec![queued.reservation.budget_reservation_id.clone()];
    persist_gepa_run_state(
        context,
        state,
        resources,
        state.cursor.phase.clone(),
        "planned",
        "planned rollout batch job",
        {
            let mut metadata = Map::new();
            metadata.insert("rollout_chunk_id".to_string(), json!(chunk_id));
            metadata.insert("rollout_chunk_rows".to_string(), json!(chunk_row_count));
            metadata
        },
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::PlanRuntimeJob {
            run_id: context.config.run.run_id.clone(),
            job_id: queued.job.job_id,
        },
        terminal: false,
        result: None,
        message: "planned rollout batch job".to_string(),
    })
}

fn plan_next_rollout_group_batch(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    let active = state.active_evaluation.as_ref().ok_or_else(|| {
        OptimizerError::Invariant("cannot plan rollout group without active evaluation".to_string())
    })?;
    let mut groups = Vec::new();
    let mut candidate_ids = Vec::new();
    let mut remaining_budget = Some(async_rollout_chunk_size(&context.config));
    for candidate_eval in &active.candidate_evaluations {
        let candidate = state
            .candidates
            .get(candidate_eval.candidate_index)
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "active evaluation candidate index {} is outside candidate registry",
                    candidate_eval.candidate_index
                ))
            })?;
        let rows = rows_for_rollout_stage(
            &context.config,
            resources,
            &active.stage,
            candidate_eval.generation,
            candidate_eval.proposal_index,
        );
        let remaining_rows = rows
            .get(candidate_eval.next_row_index..)
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "active evaluation stage {} row index {} is outside {} rows",
                    active.stage,
                    candidate_eval.next_row_index,
                    rows.len()
                ))
            })?
            .to_vec();
        if remaining_rows.is_empty() {
            continue;
        }
        let take_count = take_count_for_group_rollout_chunk(
            &context.config,
            remaining_rows.len(),
            &mut remaining_budget,
        );
        if take_count == 0 {
            break;
        }
        let chunk_rows = remaining_rows[..take_count].to_vec();
        candidate_ids.push(candidate.candidate_id.clone());
        groups.push(RolloutBatchCandidate {
            candidate: candidate.clone(),
            rows: chunk_rows,
            stage: active.stage.clone(),
        });
    }
    if groups.is_empty() {
        return finalize_active_rollout_evaluation(context, state, resources);
    }
    if active
        .candidate_evaluations
        .iter()
        .all(|candidate| candidate.next_row_index == 0)
    {
        transition_to_rollout_running(
            context,
            match active.stage.as_str() {
                "parent_minibatch_reference" => "Parent minibatch reference rollouts started",
                "candidate_minibatch" => "Candidate minibatch rollouts started",
                "candidate_full_train" => "Candidate full-train rollouts started",
                "heldout" => "Heldout rollouts started",
                _ => "Rollouts started",
            },
            json!({
                "generation": active.generation,
                "stage": active.stage,
                "candidate_count": active.candidate_evaluations.len(),
                "rollout_count": groups.iter().map(|group| group.rows.len()).sum::<usize>(),
            }),
        )?;
    }
    let queued = plan_rollout_runtime_batch_job_for_candidates(context, resources, &groups)?;
    let active = state.active_evaluation.as_mut().ok_or_else(|| {
        OptimizerError::Invariant(
            "active evaluation disappeared while planning rollout group".to_string(),
        )
    })?;
    active.candidate_id = None;
    active.planned_job_id = Some(queued.job.job_id.clone());
    active.effect_id = Some(queued.effect.runtime_effect_id.clone());
    active.reservation_id = Some(queued.reservation.budget_reservation_id.clone());
    state.cursor.pending_job_id = Some(queued.job.job_id.clone());
    state.cursor.pending_effect_id = Some(queued.effect.runtime_effect_id.clone());
    state.cursor.pending_reservation_ids = vec![queued.reservation.budget_reservation_id.clone()];
    let mut metadata = Map::new();
    metadata.insert("stage".to_string(), json!(active.stage.clone()));
    metadata.insert("candidate_ids".to_string(), json!(candidate_ids));
    metadata.insert(
        "rollout_chunk_rows".to_string(),
        json!(groups.iter().map(|group| group.rows.len()).sum::<usize>()),
    );
    metadata.insert(
        "rollout_chunk_size".to_string(),
        json!(async_rollout_chunk_size(&context.config)),
    );
    persist_gepa_run_state(
        context,
        state,
        resources,
        state.cursor.phase.clone(),
        "planned",
        "planned rollout group batch job",
        metadata,
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::PlanRuntimeJob {
            run_id: context.config.run.run_id.clone(),
            job_id: queued.job.job_id,
        },
        terminal: false,
        result: None,
        message: "planned rollout group batch job".to_string(),
    })
}

fn rows_for_rollout_stage(
    config: &SynthOptimizerConfig,
    resources: &GepaStepResources,
    stage: &str,
    generation: usize,
    proposal_index: usize,
) -> Vec<Value> {
    match stage {
        "parent_minibatch_reference" | "candidate_minibatch" => minibatch_rows(
            &resources.minibatch_rows,
            &config.gepa.batch_sampler,
            config.gepa.minibatch_size,
            generation,
            proposal_index,
            config.gepa.proposals_per_generation,
        ),
        "seed_full_train" | "candidate_full_train" => resources.train_rows.clone(),
        "heldout" => resources.heldout_rows.clone(),
        _ => Vec::new(),
    }
}

struct RolloutBatchCandidate {
    candidate: CandidateRecord,
    rows: Vec<Value>,
    stage: String,
}

fn plan_rollout_runtime_batch_job(
    context: &GepaRunContext,
    resources: &GepaStepResources,
    candidate: &CandidateRecord,
    rows: &[Value],
    stage: &str,
) -> Result<runtime::QueuedRuntimeEffect> {
    plan_rollout_runtime_batch_job_for_candidates(
        context,
        resources,
        &[RolloutBatchCandidate {
            candidate: candidate.clone(),
            rows: rows.to_vec(),
            stage: stage.to_string(),
        }],
    )
}

fn plan_rollout_runtime_batch_job_for_candidates(
    context: &GepaRunContext,
    resources: &GepaStepResources,
    candidate_groups: &[RolloutBatchCandidate],
) -> Result<runtime::QueuedRuntimeEffect> {
    let rollout_namespace = format!("{}:container.rollout", context.cache_namespace);
    let rollout_count = candidate_groups
        .iter()
        .map(|group| group.rows.len())
        .sum::<usize>();
    let mut dispatch_items = Vec::with_capacity(rollout_count);
    let mut example_refs = Vec::with_capacity(rollout_count);
    let mut candidate_ids = Vec::new();
    let mut stages = BTreeSet::new();
    let mut batch_requests = Vec::with_capacity(rollout_count);
    for group in candidate_groups {
        if !candidate_ids.contains(&group.candidate.candidate_id) {
            candidate_ids.push(group.candidate.candidate_id.clone());
        }
        stages.insert(group.stage.clone());
    }
    let max_rows = candidate_groups
        .iter()
        .map(|group| group.rows.len())
        .max()
        .unwrap_or(0);
    for row_index in 0..max_rows {
        for group in candidate_groups {
            let Some(row) = group.rows.get(row_index) else {
                continue;
            };
            let seed = row.get("seed").and_then(Value::as_i64).unwrap_or(0);
            let overlay = CandidateOverlay {
                candidate: PromptCandidatePayload::from_map(group.candidate.payload.clone()),
                metadata: Map::new(),
            };
            let request = json!({
                "submission_mode": rollout_submission_mode_for_request(&context.config),
                "task_id": resources.rollout_task_id,
                "candidate": overlay.candidate.to_value(),
                "candidate_overlay": overlay,
                "policy": context.config.policy,
                "dataset_row": row,
                "metadata": {
                    "candidate_id": group.candidate.candidate_id,
                    "seed": seed,
                },
            });
            let mut cache_metadata = Map::new();
            cache_metadata.insert(
                "candidate_id".to_string(),
                json!(group.candidate.candidate_id),
            );
            cache_metadata.insert("evaluation_stage".to_string(), json!(group.stage));
            let example_id = row_example_id(row)?;
            cache_metadata.insert("example_id".to_string(), json!(example_id.clone()));
            cache_metadata.insert("task_id".to_string(), json!(resources.rollout_task_id));
            batch_requests.push(request.clone());
            example_refs.push(json!({
                "candidate_id": group.candidate.candidate_id,
                "example_id": example_id,
            }));
            dispatch_items.push(runtime::RuntimeRolloutDispatchItem {
                cache_metadata,
                request,
                candidate_id: group.candidate.candidate_id.clone(),
                stage: group.stage.clone(),
                example_id,
                task_id: resources.rollout_task_id.clone(),
            });
        }
    }
    let planned_effect_key = RequestCache::cache_key_with_profile(
        &rollout_namespace,
        &json!({
            "stages": stages.clone(),
            "candidate_ids": candidate_ids.clone(),
            "rollout_batch": batch_requests,
        }),
        ROLLOUT_CACHE_PROFILE,
    );
    let mut effect_metadata = Map::new();
    effect_metadata.insert("algorithm_id".to_string(), json!(GEPA_ALGORITHM_ID));
    effect_metadata.insert("candidate_ids".to_string(), json!(candidate_ids.clone()));
    effect_metadata.insert("evaluation_stages".to_string(), json!(stages.clone()));
    effect_metadata.insert("rollout_count".to_string(), json!(rollout_count));
    effect_metadata.insert("task_id".to_string(), json!(resources.rollout_task_id));
    let dispatch_payload = runtime::RuntimeEffectDispatchPayload::rollout_batch(
        runtime::RuntimeRolloutBatchDispatchInput {
            cache_namespace: rollout_namespace.clone(),
            cache_profile: ROLLOUT_CACHE_PROFILE.to_string(),
            rollouts: dispatch_items,
        },
    );
    record_runtime_effect_planned(
        &context.workspace,
        RuntimeEffectPlanInput {
            run_id: &context.config.run.run_id,
            effect_kind: "container_rollout",
            lane: "rollout",
            subject_type: "candidate_examples",
            subject_id: &format!("{}:{}rollouts", candidate_ids.join(","), rollout_count),
            idempotency_key: &planned_effect_key,
            job_kind: OptimizerJobKind::Rollout,
            candidate_id: candidate_ids.first().map(String::as_str),
            cache_key: Some(planned_effect_key.clone()),
            budget_estimate: rollout_budget_estimate_for_count(&context.config, rollout_count),
            payload: json!({
                "candidate_ids": candidate_ids.clone(),
                "example_refs": example_refs,
                "rollout_count": rollout_count,
                "stages": stages,
                "task_id": resources.rollout_task_id,
            }),
            dispatch_payload,
            metadata: effect_metadata,
        },
    )
}

fn rollout_budget_estimate_for_count(
    config: &SynthOptimizerConfig,
    rollout_count: usize,
) -> RuntimeEffectBudgetEstimate {
    let estimate = ConfiguredGepaRunLimits::from_config(config).rollout_budget_estimate();
    let count = rollout_count as u64;
    RuntimeEffectBudgetEstimate {
        max_cost_usd: estimate
            .max_cost_usd
            .map(|value| value * rollout_count as f64),
        max_prompt_tokens: scale_u64_budget(estimate.max_prompt_tokens, count),
        max_completion_tokens: scale_u64_budget(estimate.max_completion_tokens, count),
        max_total_tokens: scale_u64_budget(estimate.max_total_tokens, count),
        max_rollouts: scale_u64_budget(estimate.max_rollouts, count),
        max_wall_seconds: scale_u64_budget(estimate.max_wall_seconds, count),
    }
}

fn scale_u64_budget(value: Option<u64>, count: u64) -> Option<u64> {
    value.map(|item| item.saturating_mul(count))
}

fn rollout_submission_mode_for_request(config: &SynthOptimizerConfig) -> String {
    let mode = config
        .gepa
        .rollout_submission_mode
        .trim()
        .to_ascii_lowercase();
    if mode.is_empty() {
        "sync".to_string()
    } else {
        mode
    }
}

fn consume_completed_runtime_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    job: OptimizerJob,
) -> Result<GepaAdvanceOutcome> {
    let outcome = runtime_outcome_from_job(&job)?;
    match outcome {
        StoredRuntimeOutcome::Proposer {
            response,
            proposals,
            usage,
            cost_usd,
            backend,
            workspace,
        } => {
            consume_proposer_outcome(
                context,
                state,
                resources,
                &job.job_id,
                response,
                proposals,
                usage,
                cost_usd,
                backend,
                workspace,
            )?;
        }
        StoredRuntimeOutcome::Rollout {
            response,
            reward,
            usage,
            cost_usd,
            cache_key,
            cache_hit,
            stage,
            example_id,
            ..
        } => {
            consume_rollout_outcome(
                context, state, resources, None, response, reward, usage, cost_usd, cache_key,
                cache_hit, stage, example_id,
            )?;
        }
        StoredRuntimeOutcome::RolloutBatch { outcomes } => {
            for outcome in outcomes {
                consume_rollout_outcome(
                    context,
                    state,
                    resources,
                    Some(outcome.candidate_id),
                    outcome.response,
                    outcome.reward,
                    outcome.usage,
                    outcome.cost_usd,
                    outcome.cache_key,
                    outcome.cache_hit,
                    outcome.stage,
                    outcome.example_id,
                )?;
            }
        }
    }
    let consumed_phase = state.cursor.phase.clone();
    state.cursor.pending_job_id = None;
    state.cursor.pending_effect_id = None;
    state.cursor.pending_reservation_ids.clear();
    if let Some(active) = state.active_evaluation.as_mut() {
        active.planned_job_id = None;
        active.effect_id = None;
        active.reservation_id = None;
    }
    persist_gepa_run_state(
        context,
        state,
        resources,
        consumed_phase,
        "completed",
        "consumed GEPA runtime outcome",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::ConsumeRuntimeOutcome {
            run_id: context.config.run.run_id.clone(),
            job_id: job.job_id,
        },
        terminal: false,
        result: None,
        message: "consumed GEPA runtime outcome".to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
fn consume_proposer_outcome(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    _resources: &GepaStepResources,
    job_id: &str,
    response: Value,
    proposals: Vec<ProposedCandidate>,
    usage: UsageTotals,
    cost_usd: f64,
    backend: String,
    workspace: Option<String>,
) -> Result<()> {
    let active = state.active_evaluation.take().ok_or_else(|| {
        OptimizerError::Invariant("proposer outcome has no active evaluation".to_string())
    })?;
    let parent_idx = active
        .candidate_index
        .or_else(|| {
            active.candidate_id.as_ref().and_then(|candidate_id| {
                state
                    .candidates
                    .iter()
                    .position(|candidate| &candidate.candidate_id == candidate_id)
            })
        })
        .ok_or_else(|| {
            OptimizerError::Invariant(
                "proposer outcome has no selected parent candidate".to_string(),
            )
        })?;
    let parent = state.candidates.get(parent_idx).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "selected parent index {parent_idx} is outside candidate registry"
        ))
    })?;
    let outcome = ProposerOutcome {
        proposals: proposals.clone(),
        usage: usage.clone(),
        cost_usd,
        backend: backend.clone(),
        workspace: workspace.clone(),
        evidence_warnings: response
            .get("evidence_warnings")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect(),
    };
    state.total_usage.merge(&usage);
    state.total_cost += cost_usd;
    state.usage_ledger.push(proposer_usage_record(
        &context.config,
        parent,
        active.generation,
        &outcome,
    )?);
    record_proposer_workspace_evidence(
        context,
        job_id,
        &parent.candidate_id,
        active.generation,
        &proposals,
        &response,
        workspace.as_deref(),
    )?;
    let mut metadata = Map::new();
    metadata.insert("stage".to_string(), Value::String("proposer".to_string()));
    metadata.insert("generation".to_string(), json!(active.generation));
    metadata.insert("proposal_count".to_string(), json!(proposals.len()));
    metadata.insert("backend".to_string(), Value::String(backend.clone()));
    metadata.insert(
        "warning_count".to_string(),
        json!(outcome.evidence_warnings.len()),
    );
    push_stopper_snapshot(
        &mut state.stopper_states,
        &mut state.stopper_sequence,
        &context.config,
        StopperSnapshot {
            status: budget_status(&context.config, state.rollout_count, state.total_cost),
            reason: Some("proposer completed"),
            generation: Some(active.generation),
            candidate_id: Some(&parent.candidate_id),
            evaluation_stage: Some("proposer"),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            metadata,
        },
    );
    context.events.emit(
        "proposer.completed",
        "Proposer returned candidates",
        json!({
            "generation": active.generation,
            "proposal_count": proposals.len(),
            "backend": backend,
            "workspace": workspace,
            "warning_count": outcome.evidence_warnings.len(),
            "warnings": outcome.evidence_warnings,
        }),
    )?;
    state.proposal_queue = proposals;
    state.cursor.proposal_index = 0;
    state.cursor.pipeline_state.parent_candidate_id = Some(parent.candidate_id.clone());
    state.cursor.phase = GepaCursorPhase::ProposerWaiting;
    if context.state_machine.state() == OptimizerRunState::Proposing {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::ProposerFinished,
            "Proposer returned candidates; rollout queue ready",
            json!({"generation": active.generation, "proposal_count": state.proposal_queue.len()}),
        )?;
    }
    Ok(())
}

fn record_proposer_workspace_evidence(
    context: &mut GepaRunContext,
    job_id: &str,
    parent_candidate_id: &str,
    generation: usize,
    proposals: &[ProposedCandidate],
    response: &Value,
    workspace: Option<&str>,
) -> Result<()> {
    let artifact_refs = proposer_workspace_artifact_refs(context, workspace)?;
    if !artifact_refs.is_empty() {
        context
            .workspace
            .record_artifact_refs(&context.config.run.run_id, &artifact_refs)?;
    }

    let proposed_candidate_ids = proposals
        .iter()
        .map(|proposal| candidate_id(&proposal.payload_map()))
        .collect::<Vec<_>>();
    let manifest = response.get("manifest").cloned().unwrap_or(Value::Null);
    let manifest_evidence = manifest.get("evidence").cloned().unwrap_or(Value::Null);
    let artifact_values = artifact_refs
        .iter()
        .map(|artifact| {
            json!({
                "path": artifact.path,
                "kind": artifact.kind,
                "sha256": artifact.sha256,
                "bytes": artifact.bytes,
                "retention": artifact.retention,
            })
        })
        .collect::<Vec<_>>();
    let frame_id = format!("evidence:gepa_proposer_output:{job_id}");
    let mut metadata = Map::new();
    metadata.insert("generation".to_string(), json!(generation));
    metadata.insert(
        "parent_candidate_id".to_string(),
        json!(parent_candidate_id),
    );
    metadata.insert("proposal_count".to_string(), json!(proposals.len()));
    metadata.insert("artifact_count".to_string(), json!(artifact_refs.len()));
    if let Some(workspace) = workspace {
        metadata.insert("workspace".to_string(), json!(workspace));
    }
    let frame = EvidenceFrame {
        schema_version: "evidence_frame.v1".to_string(),
        evidence_frame_id: frame_id.clone(),
        subject_type: "proposer_job".to_string(),
        subject_id: job_id.to_string(),
        candidate_id: Some(parent_candidate_id.to_string()),
        sensor_frame_id: None,
        kind: "gepa_proposer_output".to_string(),
        source: "codex_app_server".to_string(),
        summary: format!(
            "GEPA proposer job {job_id} returned {} proposals for generation {generation}",
            proposals.len()
        ),
        score: None,
        severity: "info".to_string(),
        evidence: json!({
            "schema_version": "gepa_proposer_output_evidence.v1",
            "generation": generation,
            "parent_candidate_id": parent_candidate_id,
            "proposal_count": proposals.len(),
            "proposed_candidate_ids": proposed_candidate_ids,
            "workspace": workspace,
            "manifest": manifest,
            "manifest_evidence": manifest_evidence,
            "artifact_refs": artifact_values,
        }),
        metadata,
    };

    let mut plan_links = Vec::new();
    for artifact in &artifact_refs {
        plan_links.push(proposer_artifact_plan_link(
            job_id,
            artifact,
            generation,
            "subagent_artifact",
        ));
    }
    plan_links.push(PlanLinkRecord::from_input(PlanLinkInput {
        source_type: "evidence_frame",
        source_id: &frame.evidence_frame_id,
        target_type: "proposer_job",
        target_id: job_id,
        relation: "evidence_source",
        status: "active",
        confidence: 1.0,
        metadata: json_map(vec![
            ("evidence_kind", json!("gepa_proposer_output")),
            ("generation", json!(generation)),
        ]),
    }));
    context.workspace.record_evidence_frames_and_plan_links(
        &context.config.run.run_id,
        &[frame],
        &plan_links,
    )?;
    Ok(())
}

fn proposer_workspace_artifact_refs(
    context: &GepaRunContext,
    workspace: Option<&str>,
) -> Result<Vec<ArtifactRef>> {
    let Some(workspace) = workspace else {
        return Ok(Vec::new());
    };
    let workspace = PathBuf::from(workspace);
    let specs = [
        ("proposal/manifest.json", "proposal_manifest"),
        (
            "state/workspace_pack_manifest.json",
            "workspace_pack_manifest",
        ),
        (".agent_artifacts/opencode_session.json", "agent_session"),
        (".agent_artifacts/opencode_messages.json", "agent_messages"),
        (".agent_artifacts/opencode_response.json", "agent_response"),
        (".agent_artifacts/opencode_sse_events.jsonl", "agent_events"),
    ];
    let mut refs = Vec::new();
    for (relative, kind) in specs {
        let path = workspace.join(relative);
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if metadata.is_file() {
            refs.push(context.paths.artifact_ref(&path, kind, "run_evidence")?);
        }
    }
    Ok(refs)
}

fn proposer_artifact_plan_link(
    job_id: &str,
    artifact: &ArtifactRef,
    generation: usize,
    relation: &str,
) -> PlanLinkRecord {
    PlanLinkRecord::from_input(PlanLinkInput {
        source_type: "proposer_job",
        source_id: job_id,
        target_type: "artifact",
        target_id: &artifact.path,
        relation,
        status: "active",
        confidence: 1.0,
        metadata: json_map(vec![
            ("artifact_kind", json!(artifact.kind)),
            ("artifact_sha256", json!(artifact.sha256)),
            ("generation", json!(generation)),
        ]),
    })
}

fn json_map(items: Vec<(&str, Value)>) -> Map<String, Value> {
    items
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn consume_rollout_outcome(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    candidate_id: Option<String>,
    response: Value,
    reward: f64,
    usage: UsageTotals,
    cost_usd: f64,
    cache_key: String,
    cache_hit: bool,
    stage: String,
    example_id: String,
) -> Result<()> {
    if state
        .active_evaluation
        .as_ref()
        .is_some_and(GepaActiveEvaluation::is_group)
    {
        return consume_group_rollout_outcome(
            context,
            state,
            resources,
            candidate_id,
            response,
            reward,
            usage,
            cost_usd,
            cache_key,
            cache_hit,
            stage,
            example_id,
        );
    }
    let active = state.active_evaluation.as_mut().ok_or_else(|| {
        OptimizerError::Invariant("rollout outcome has no active evaluation".to_string())
    })?;
    if active.stage != stage {
        return Err(OptimizerError::Invariant(format!(
            "rollout outcome stage {stage} does not match active stage {}",
            active.stage
        )));
    }
    let candidate_index = active.candidate_index.ok_or_else(|| {
        OptimizerError::Invariant(
            "rollout outcome active evaluation has no candidate_index".to_string(),
        )
    })?;
    let candidate = state.candidates.get(candidate_index).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "rollout outcome candidate index {candidate_index} is outside candidate registry"
        ))
    })?;
    let rows = rows_for_rollout_stage(
        &context.config,
        resources,
        &stage,
        active.generation,
        active.proposal_index,
    );
    let row = rows.get(active.next_row_index).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "rollout outcome row index {} is outside {} rows",
            active.next_row_index,
            rows.len()
        ))
    })?;
    let row_example = row_example_id(row)?;
    if row_example != example_id {
        return Err(OptimizerError::Invariant(format!(
            "rollout outcome example_id {example_id} does not match active row {row_example}"
        )));
    }
    let typed_response = synth_optimizer_platform::RolloutResponse::from_value(response.clone())?;
    typed_response.validate_for_gepa()?;
    let mut sensor_frame =
        SensorFrame::from_rollout_response(&candidate.candidate_id, row, &stage, &response)?;
    align_sensor_frame_objectives(&mut sensor_frame, &resources.objective_set, reward);
    attach_rollout_trace_artifact(
        &context.paths,
        &context.config.run.run_id,
        &mut sensor_frame,
    )?;
    record_rollout_materialization_from_outcome(
        context,
        resources,
        candidate,
        row,
        &stage,
        &response,
        &typed_response,
        &sensor_frame,
        &cache_key,
        cache_hit,
    )?;
    active.reward_sum += reward;
    active.rollout_count += 1;
    active.usage.merge(&usage);
    active.cost_usd += cost_usd;
    let seed = row.get("seed").and_then(Value::as_i64).unwrap_or(0);
    active.scores.push(RolloutScore {
        example_id,
        seed,
        reward,
    });
    active.sensor_frames.push(sensor_frame);
    active.next_row_index += 1;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn consume_group_rollout_outcome(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    candidate_id: Option<String>,
    response: Value,
    reward: f64,
    usage: UsageTotals,
    cost_usd: f64,
    cache_key: String,
    cache_hit: bool,
    stage: String,
    example_id: String,
) -> Result<()> {
    let candidate_id = candidate_id.ok_or_else(|| {
        OptimizerError::Invariant(
            "rollout batch outcome for grouped evaluation has no candidate_id".to_string(),
        )
    })?;
    let active = state.active_evaluation.as_mut().ok_or_else(|| {
        OptimizerError::Invariant("rollout outcome has no active evaluation".to_string())
    })?;
    if active.stage != stage {
        return Err(OptimizerError::Invariant(format!(
            "rollout outcome stage {stage} does not match active stage {}",
            active.stage
        )));
    }
    let candidate_eval_index = active
        .candidate_evaluations
        .iter()
        .position(|candidate| candidate.candidate_id == candidate_id)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "rollout batch outcome candidate_id {candidate_id} is not active"
            ))
        })?;
    let candidate_eval = active
        .candidate_evaluations
        .get_mut(candidate_eval_index)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "active candidate evaluation index {candidate_eval_index} is missing"
            ))
        })?;
    let candidate = state
        .candidates
        .get(candidate_eval.candidate_index)
        .cloned()
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "rollout outcome candidate index {} is outside candidate registry",
                candidate_eval.candidate_index
            ))
        })?;
    let rows = rows_for_rollout_stage(
        &context.config,
        resources,
        &stage,
        candidate_eval.generation,
        candidate_eval.proposal_index,
    );
    let row = rows.get(candidate_eval.next_row_index).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "rollout outcome row index {} is outside {} rows",
            candidate_eval.next_row_index,
            rows.len()
        ))
    })?;
    let row_example = row_example_id(row)?;
    if row_example != example_id {
        return Err(OptimizerError::Invariant(format!(
            "rollout outcome example_id {example_id} does not match active row {row_example}"
        )));
    }
    let typed_response = synth_optimizer_platform::RolloutResponse::from_value(response.clone())?;
    typed_response.validate_for_gepa()?;
    let mut sensor_frame =
        SensorFrame::from_rollout_response(&candidate.candidate_id, row, &stage, &response)?;
    align_sensor_frame_objectives(&mut sensor_frame, &resources.objective_set, reward);
    attach_rollout_trace_artifact(
        &context.paths,
        &context.config.run.run_id,
        &mut sensor_frame,
    )?;
    record_rollout_materialization_from_outcome(
        context,
        resources,
        &candidate,
        row,
        &stage,
        &response,
        &typed_response,
        &sensor_frame,
        &cache_key,
        cache_hit,
    )?;
    candidate_eval.reward_sum += reward;
    candidate_eval.rollout_count += 1;
    candidate_eval.usage.merge(&usage);
    candidate_eval.cost_usd += cost_usd;
    let seed = row.get("seed").and_then(Value::as_i64).unwrap_or(0);
    candidate_eval.scores.push(RolloutScore {
        example_id,
        seed,
        reward,
    });
    candidate_eval.sensor_frames.push(sensor_frame);
    candidate_eval.next_row_index += 1;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_rollout_materialization_from_outcome(
    context: &GepaRunContext,
    resources: &GepaStepResources,
    candidate: &CandidateRecord,
    row: &Value,
    stage: &str,
    response: &Value,
    typed_response: &synth_optimizer_platform::RolloutResponse,
    sensor_frame: &SensorFrame,
    cache_key: &str,
    cache_hit: bool,
) -> Result<()> {
    let seed = row.get("seed").and_then(Value::as_i64).unwrap_or(0);
    let overlay = CandidateOverlay {
        candidate: PromptCandidatePayload::from_map(candidate.payload.clone()),
        metadata: Map::new(),
    };
    let request = json!({
        "submission_mode": rollout_submission_mode_for_request(&context.config),
        "task_id": resources.rollout_task_id,
        "candidate": overlay.candidate.to_value(),
        "candidate_overlay": overlay,
        "policy": context.config.policy,
        "dataset_row": row,
        "metadata": {
            "candidate_id": candidate.candidate_id,
            "seed": seed,
        },
    });
    let objective_scores = serde_json::to_value(&sensor_frame.objective_scores)?;
    let materialization =
        rollout_materialization_identity(&resources.program, candidate, &resources.objective_set);
    let candidate_payload_value = serde_json::to_value(&candidate.payload)?;
    let example_id = row_example_id(row)?;
    let mut materialization_metadata = Map::new();
    materialization_metadata.insert("cache_hit".to_string(), json!(cache_hit));
    materialization_metadata.insert(
        "rollout_status".to_string(),
        json!(sensor_frame.status.clone()),
    );
    materialization_metadata.insert(
        "rollout_id".to_string(),
        sensor_frame
            .rollout_id
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    context.workspace.record_materialization(
        &context.config.run.run_id,
        &MaterializationRecord::from_input(MaterializationRecordInput {
            candidate_id: &candidate.candidate_id,
            candidate_payload: &candidate_payload_value,
            example: row,
            request: &request,
            example_id: &example_id,
            seed,
            split: &sensor_frame.split,
            evaluation_stage: stage,
            task_id: &resources.rollout_task_id,
            materialization: materialization.clone(),
            status: "materialized",
            platform_cache_key: Some(cache_key.to_string()),
            metadata: materialization_metadata,
        }),
    )?;
    context.workspace.record_evaluation_cache(
        &context.config.run.run_id,
        &EvaluationCacheRecord::from_input(EvaluationCacheRecordInput {
            candidate_payload: &candidate_payload_value,
            example: row,
            request: &request,
            example_id: &example_id,
            materialization,
            source_rollout_id: typed_response
                .rollout_id
                .clone()
                .or_else(|| sensor_frame.rollout_id.clone()),
            reward: sensor_frame.reward,
            objective_scores,
            actionable_side_info: sensor_frame
                .actionable_side_info
                .clone()
                .unwrap_or_else(|| json!({})),
            usage: sensor_frame.usage.clone(),
            trace_ref: sensor_frame
                .trace_digest
                .as_ref()
                .map(|digest| format!("trace_sha256:{}", digest.sha256)),
            status: &sensor_frame.status,
            cache_hit,
            platform_cache_key: Some(cache_key.to_string()),
            rollout_payload: response,
            metadata: Map::new(),
        }),
    )
}

fn attach_rollout_trace_artifact(
    paths: &ArtifactPaths,
    _run_id: &str,
    sensor_frame: &mut SensorFrame,
) -> Result<()> {
    let Some(trace_payload) = sensor_frame.metadata.get("rollout_trace").cloned() else {
        return Ok(());
    };
    let trace_dir = paths.run_dir.join("rollout_traces");
    fs::create_dir_all(&trace_dir).map_err(|source| OptimizerError::io(&trace_dir, source))?;
    let trace_path = trace_dir.join(format!("{}.json", sensor_frame.sensor_frame_id));
    paths.write_json(&trace_path, &trace_payload)?;
    let artifact = paths.artifact_ref(&trace_path, "rollout_trace_payload", "debug")?;
    sensor_frame.artifact_refs.push(artifact);
    Ok(())
}

fn rollout_materialization_identity(
    program: &PromptProgram,
    candidate: &CandidateRecord,
    objective_set: &ObjectiveSetRecord,
) -> RolloutMaterializationIdentity {
    let lever_manifest = LeverManifest::from_prompt_program(program);
    let has_non_prompt_lever = lever_manifest.levers.iter().any(|lever| {
        !matches!(
            lever.kind,
            LeverKind::TextPrompt | LeverKind::SystemPrompt | LeverKind::UserPrompt
        )
    });
    if has_non_prompt_lever {
        RolloutMaterializationIdentity::lever_bundle(
            GEPA_ALGORITHM_ID,
            &program.program_id,
            &candidate.lever_bundle.schema_version,
            &objective_set.objective_set_hash,
        )
    } else {
        RolloutMaterializationIdentity::prompt_overlay(
            GEPA_ALGORITHM_ID,
            &program.program_id,
            &candidate.lever_bundle.schema_version,
            &objective_set.objective_set_hash,
        )
    }
}

fn consume_failed_runtime_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    _resources: &GepaStepResources,
    job: OptimizerJob,
) -> Result<GepaAdvanceOutcome> {
    let (phase, status, message) = match job.status {
        OptimizerJobStatus::Cancelled => (
            GepaCursorPhase::Cancelled,
            "cancelled",
            "GEPA runtime job cancelled",
        ),
        _ => (GepaCursorPhase::Failed, "failed", "GEPA runtime job failed"),
    };
    let error_summary = job
        .payload
        .get("error")
        .cloned()
        .or_else(|| {
            job.failure.as_ref().map(|failure| {
                json!({
                    "error_code": "synth_optimizer_failed",
                    "message": format!("GEPA runtime job {} {}: {}", job.job_id, job.status.as_str(), failure.message),
                    "failure": failure,
                })
            })
        })
        .unwrap_or_else(|| {
            json!({
                "error_code": "synth_optimizer_failed",
                "message": format!("GEPA runtime job {} {}", job.job_id, job.status.as_str()),
            })
        });
    state.cursor.pending_job_id = None;
    state.cursor.pending_effect_id = None;
    state.cursor.pending_reservation_ids.clear();
    terminalize_gepa_run_state(context, state, phase, status, message, error_summary)?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::TerminalizeRun {
            run_id: context.config.run.run_id.clone(),
            status: status.to_string(),
        },
        terminal: true,
        result: None,
        message: format!("{message}: {}", job.job_id),
    })
}

fn terminalize_aborted_gepa_run(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    error: OptimizerError,
    message: &str,
) -> Result<GepaAdvanceOutcome> {
    let (phase, status) = if matches!(error, OptimizerError::Cancelled { .. }) {
        (GepaCursorPhase::Cancelled, "cancelled")
    } else {
        (GepaCursorPhase::Failed, "failed")
    };
    terminalize_pending_runtime_work_for_abort(context, state, status, &error)?;
    let error_summary = json!({
        "error_code": error.error_code(),
        "message": error.to_string(),
    });
    terminalize_gepa_run_state(context, state, phase, status, message, error_summary)?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::TerminalizeRun {
            run_id: context.config.run.run_id.clone(),
            status: status.to_string(),
        },
        terminal: true,
        result: None,
        message: message.to_string(),
    })
}

fn terminalize_pending_runtime_job_for_abort(
    context: &mut GepaRunContext,
    state: &GepaRunState,
    status: &str,
    error: &OptimizerError,
) -> Result<()> {
    let Some(job_id) = state.cursor.pending_job_id.as_deref() else {
        return Ok(());
    };
    let job = context
        .workspace
        .optimizer_job(&context.config.run.run_id, job_id)?;
    if job.status.is_terminal() {
        return Ok(());
    }
    let Some(effect_id) = job.payload.get("runtime_effect_id").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(reservation_id) = job
        .payload
        .get("budget_reservation_id")
        .and_then(Value::as_str)
    else {
        return Ok(());
    };
    let effect = context
        .workspace
        .runtime_effect(&context.config.run.run_id, effect_id)?;
    let reservation = context
        .workspace
        .budget_reservation(&context.config.run.run_id, reservation_id)?;
    let failure = FailurePayload::from_optimizer_error(error);
    let mut metadata = Map::new();
    metadata.insert("abort_status".to_string(), json!(status));
    metadata.insert("error_code".to_string(), json!(error.error_code()));
    record_runtime_effect_completed(
        &context.workspace,
        RuntimeEffectCompletionInput {
            planned: &effect,
            reservation: &reservation,
            status,
            cost_usd: 0.0,
            usage: &UsageTotals::default(),
            rollout_count: 0,
            failure: Some(&failure),
            metadata,
        },
    )
}

fn terminalize_pending_runtime_work_for_abort(
    context: &mut GepaRunContext,
    state: &GepaRunState,
    status: &str,
    error: &OptimizerError,
) -> Result<()> {
    terminalize_pending_runtime_job_for_abort(context, state, status, error)?;

    let run_id = &context.config.run.run_id;
    let failure = FailurePayload::from_optimizer_error(error);
    let mut terminalized_effect_ids = BTreeSet::new();
    for effect in context.workspace.view().runtime_effect_records(run_id)? {
        if runtime_effect_status_is_terminal(&effect.status)
            || !terminalized_effect_ids.insert(effect.runtime_effect_id.clone())
        {
            continue;
        }
        let mut metadata = Map::new();
        metadata.insert("abort_status".to_string(), json!(status));
        metadata.insert("abort_scope".to_string(), json!("pending_runtime_work"));
        metadata.insert("error_code".to_string(), json!(error.error_code()));
        let Some(reservation_id) = effect.budget_reservation_id.as_deref() else {
            terminalize_runtime_effect_without_reservation(
                &context.workspace,
                &effect,
                status,
                &failure,
                metadata,
            )?;
            continue;
        };
        let reservation = context
            .workspace
            .budget_reservation(run_id, reservation_id)?;
        record_runtime_effect_completed(
            &context.workspace,
            RuntimeEffectCompletionInput {
                planned: &effect,
                reservation: &reservation,
                status,
                cost_usd: 0.0,
                usage: &UsageTotals::default(),
                rollout_count: 0,
                failure: Some(&failure),
                metadata,
            },
        )?;
    }
    Ok(())
}

fn terminalize_runtime_effect_without_reservation(
    workspace: &WorkspaceStore,
    effect: &RuntimeEffectRecord,
    status: &str,
    failure: &FailurePayload,
    mut metadata: Map<String, Value>,
) -> Result<()> {
    metadata.insert("failure".to_string(), serde_json::to_value(failure)?);
    let mut payload = effect.payload.clone();
    if let Some(object) = payload.as_object_mut() {
        object.insert("completion_status".to_string(), json!(status));
        object.insert("failure".to_string(), serde_json::to_value(failure)?);
    }
    let terminal_effect = RuntimeEffectRecord::from_input(RuntimeEffectInput {
        run_id: &effect.run_id,
        effect_kind: &effect.effect_kind,
        lane: &effect.lane,
        status,
        subject_type: &effect.subject_type,
        subject_id: &effect.subject_id,
        idempotency_key: &effect.idempotency_key,
        cache_key: effect.cache_key.clone(),
        job_id: effect.job_id.clone(),
        budget_reservation_id: None,
        attempt: effect.attempt,
        failure_class: Some(failure.failure_class().to_string()),
        payload,
        metadata,
    });
    workspace.record_runtime_effect(&terminal_effect)?;
    if let Some(job_id) = effect.job_id.as_deref() {
        record_runtime_effect_job(
            workspace,
            RuntimeEffectJobInput {
                job_id,
                run_id: &effect.run_id,
                kind: runtime_effect_job_kind(effect),
                status: optimizer_job_status_from_effect_status(status),
                candidate_id: runtime_effect_candidate_id(effect).as_deref(),
                effect,
                reservation: None,
                dispatch_payload: None,
                queue_state: status,
                failure: Some(failure),
            },
        )?;
    }
    Ok(())
}

fn runtime_effect_status_is_terminal(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "canceled" | "expired" | "rejected"
    )
}

fn terminalize_gepa_run_state(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    phase: GepaCursorPhase,
    status: &str,
    message: &str,
    error_summary: Value,
) -> Result<()> {
    let terminal_state = if matches!(phase, GepaCursorPhase::Cancelled) {
        OptimizerRunState::Cancelled
    } else {
        OptimizerRunState::Failed
    };
    let (trigger, terminal_event_type, terminal_message) =
        if matches!(terminal_state, OptimizerRunState::Cancelled) {
            (
                OptimizerTransitionTrigger::CancelRequested,
                "gepa.run.cancelled",
                "GEPA run cancelled",
            )
        } else {
            (
                OptimizerTransitionTrigger::FailureRaised,
                "gepa.run.failed",
                "GEPA run failed",
            )
        };
    let usage_value = serde_json::to_value(&state.total_usage)?;
    context
        .workspace
        .record_usage_ledger(&context.config.run.run_id, &state.usage_ledger)?;
    context
        .workspace
        .record_stopper_states(&context.config.run.run_id, &state.stopper_states)?;
    let cache_profile_record = CacheProfileRecord::from_profile(context.cache.profile()?);
    let cache_access_log = context.cache.access_log().to_vec();
    let cache_profile = serde_json::to_value(&cache_profile_record.profile)?;
    context
        .paths
        .write_json(&context.paths.cache_profile_path, &cache_profile)?;
    context.workspace.record_cache_profile(
        &context.config.run.run_id,
        &cache_profile_record,
        &cache_access_log,
    )?;
    let best_candidate_id = state
        .best_idx
        .and_then(|idx| state.candidates.get(idx))
        .map(|candidate| candidate.candidate_id.clone());
    let manifest_best_candidate_id = best_candidate_id.as_deref().unwrap_or("unavailable");
    let mut details = Map::new();
    details.insert("error".to_string(), error_summary.clone());
    if let Some(best_candidate_id) = best_candidate_id.as_ref() {
        details.insert("best_candidate_id".to_string(), json!(best_candidate_id));
    }
    if !context.state_machine.state().is_terminal() {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            terminal_state,
            trigger,
            terminal_message,
            Value::Object(details),
        )?;
    }
    let state_history = serde_json::to_value(&context.state_machine.history)?;
    let failure_manifest = json!({
        "schema_version": "gepa_failure_manifest.v1",
        "run_id": context.config.run.run_id,
        "status": terminal_state.as_str(),
        "best_candidate_id": manifest_best_candidate_id,
        "cost_usd": state.total_cost,
        "usage": usage_value,
        "failure": error_summary,
        "state_history": state_history,
        "event_feed_path": context.paths.event_feed_path.display().to_string(),
        "normalized_event_feed_path": context.paths.normalized_event_feed_path.display().to_string(),
        "cache_profile_path": context.paths.cache_profile_path.display().to_string(),
        "workspace_db_path": context.paths.workspace_db_path.display().to_string(),
    });
    context
        .paths
        .write_json(&context.paths.manifest_path, &failure_manifest)?;
    context.workspace.record_manifest(
        &context.config.run.run_id,
        &context.paths.manifest_path,
        manifest_best_candidate_id,
        state.total_cost,
        &usage_value,
        &failure_manifest,
    )?;
    context.events.emit(
        terminal_event_type,
        message,
        json!({
            "run_id": context.config.run.run_id,
            "state": context.state_machine.state().as_str(),
            "cost_usd": state.total_cost,
            "usage": usage_value,
            "failure": failure_manifest["failure"],
        }),
    )?;
    context.events.flush()?;
    context
        .workspace
        .record_event_stream(&context.config.run.run_id, context.events.records())?;
    normalize_event_feed(
        &context.paths.event_feed_path,
        &context.paths.normalized_event_feed_path,
        &context.paths.run_dir,
    )?;
    if matches!(terminal_state, OptimizerRunState::Cancelled) {
        context.workspace.record_run_cancelled_result(
            &context.config.run.run_id,
            best_candidate_id.as_deref(),
            state.total_cost,
            &usage_value,
        )?;
        context.registry.append(&RunRegistryEntry::cancelled(
            &context.paths,
            &context.config,
            context.cache_mode,
            &context.cache_namespace,
            state.total_cost,
            usage_value.clone(),
        ))?;
    } else {
        context.workspace.record_run_failed(
            &context.config.run.run_id,
            best_candidate_id.as_deref(),
            state.total_cost,
            &usage_value,
        )?;
        context.registry.append(&RunRegistryEntry::failed(
            &context.paths,
            &context.config,
            context.cache_mode,
            &context.cache_namespace,
            state.total_cost,
            usage_value.clone(),
        ))?;
    }
    state.checkpoint_sequence += 1;
    state.cursor.schema_version = planner::GEPA_CURSOR_SCHEMA_VERSION.to_string();
    state.cursor.run_id = context.config.run.run_id.clone();
    state.cursor.phase = phase;
    state.cursor.proposal_queue = serde_json::to_value(&state.proposal_queue)?;
    state.cursor.heldout_candidate_index = state.heldout_candidate_index;
    state.cursor.active_evaluation = state
        .active_evaluation
        .as_ref()
        .map(serde_json::to_value)
        .transpose()?;
    state.cursor.candidates = serde_json::to_value(&state.candidates)?;
    state.cursor.best_candidate_id = best_candidate_id;
    state.cursor.rollout_count = state.rollout_count;
    state.cursor.cost_usd = state.total_cost;
    state.cursor.usage = usage_value.clone();
    state.cursor.usage_ledger = serde_json::to_value(&state.usage_ledger)?;
    state.cursor.stopper_states = serde_json::to_value(&state.stopper_states)?;
    state.cursor.stopper_sequence = state.stopper_sequence;
    state.cursor.checkpoint_sequence = state.checkpoint_sequence;
    state.cursor.state_history = serde_json::to_value(&context.state_machine.history)?;
    state.cursor.pending_job_id = None;
    state.cursor.pending_effect_id = None;
    state.cursor.pending_reservation_ids.clear();
    state.cursor.error_summary = Some(failure_manifest.clone());
    state.cursor.metadata = json!({
        "status": status,
        "failure_manifest_path": context.paths.manifest_path,
    });
    let cursor_value = serde_json::to_value(&state.cursor)?;
    let checkpoint = CheckpointRecord::from_input(CheckpointInput {
        sequence_number: state.checkpoint_sequence,
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status,
        run_state: state.cursor.phase.as_str(),
        reason: Some(message),
        generation: Some(state.cursor.generation as u64),
        candidate_id: state.cursor.best_candidate_id.as_deref(),
        evaluation_stage: Some(state.cursor.phase.as_str()),
        best_candidate_id: state.cursor.best_candidate_id.as_deref(),
        candidate_count: state.candidates.len() as u64,
        frontier_count: frontier_members(&state.candidates).len() as u64,
        rollout_count: state.rollout_count as u64,
        cost_usd: state.total_cost,
        usage: usage_value,
        snapshot: cursor_value,
        metadata: Map::new(),
    });
    context
        .workspace
        .record_checkpoint(&context.config.run.run_id, &checkpoint)
}

fn finalize_active_rollout_evaluation(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    let active = state.active_evaluation.clone().ok_or_else(|| {
        OptimizerError::Invariant("cannot finalize without active evaluation".to_string())
    })?;
    if active.is_group() {
        return finalize_active_rollout_group(context, state, resources, active);
    }
    if context.state_machine.state() == OptimizerRunState::RolloutRunning {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Evaluating,
            OptimizerTransitionTrigger::RolloutsFinished,
            "Rollouts finished",
            json!({"stage": active.stage, "candidate_id": active.candidate_id}),
        )?;
    }
    let eval = CandidateEvaluation {
        average_reward: active.average_reward(),
        rollout_count: active.rollout_count,
        usage: active.usage.clone(),
        cost_usd: active.cost_usd,
        scores: active.scores.clone(),
        sensor_frames: active.sensor_frames.clone(),
    };
    state.total_usage.merge(&eval.usage);
    state.total_cost += eval.cost_usd;
    state.rollout_count += eval.rollout_count;
    append_rollout_usage(&mut state.usage_ledger, &eval);
    match active.stage.as_str() {
        "seed_full_train" => finalize_seed_full_train(context, state, resources, active, eval),
        "parent_minibatch_reference" => {
            finalize_parent_minibatch_reference(context, state, resources, active, eval)
        }
        "candidate_minibatch" => {
            finalize_candidate_minibatch(context, state, resources, active, eval)
        }
        "candidate_full_train" => {
            finalize_candidate_full_train(context, state, resources, active, eval)
        }
        "heldout" => finalize_heldout_candidate(context, state, resources, active, eval),
        stage => Err(OptimizerError::Invariant(format!(
            "unsupported active rollout stage {stage}"
        ))),
    }
}

fn finalize_active_rollout_group(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
) -> Result<GepaAdvanceOutcome> {
    if context.state_machine.state() == OptimizerRunState::RolloutRunning {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Evaluating,
            OptimizerTransitionTrigger::RolloutsFinished,
            "Rollout batch finished",
            json!({
                "stage": active.stage,
                "candidate_count": active.candidate_evaluations.len(),
            }),
        )?;
    }
    let evaluations = active
        .candidate_evaluations
        .iter()
        .cloned()
        .map(|candidate| {
            let eval = evaluation_from_active_candidate(&candidate);
            state.total_usage.merge(&eval.usage);
            state.total_cost += eval.cost_usd;
            state.rollout_count += eval.rollout_count;
            append_rollout_usage(&mut state.usage_ledger, &eval);
            (candidate, eval)
        })
        .collect::<Vec<_>>();
    match active.stage.as_str() {
        "candidate_minibatch" => {
            finalize_candidate_minibatch_group(context, state, resources, active, evaluations)
        }
        "candidate_full_train" => {
            finalize_candidate_full_train_group(context, state, resources, active, evaluations)
        }
        "heldout" => finalize_heldout_group(context, state, resources, active, evaluations),
        stage => Err(OptimizerError::Invariant(format!(
            "unsupported grouped rollout stage {stage}"
        ))),
    }
}

fn evaluation_from_active_candidate(
    candidate: &GepaActiveCandidateEvaluation,
) -> CandidateEvaluation {
    CandidateEvaluation {
        average_reward: candidate.average_reward(),
        rollout_count: candidate.rollout_count,
        usage: candidate.usage.clone(),
        cost_usd: candidate.cost_usd,
        scores: candidate.scores.clone(),
        sensor_frames: candidate.sensor_frames.clone(),
    }
}

fn finalize_seed_full_train(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
    eval: CandidateEvaluation,
) -> Result<GepaAdvanceOutcome> {
    let candidate_idx = active.candidate_index.unwrap_or(0);
    let candidate_id = {
        let candidate = state.candidates.get_mut(candidate_idx).ok_or_else(|| {
            OptimizerError::Invariant(format!("seed candidate index {candidate_idx} is missing"))
        })?;
        candidate.status = "full_train_evaluated".to_string();
        candidate.train_reward = Some(eval.average_reward);
        candidate.train_scores = eval.scores.clone();
        candidate.sensor_frames.extend(eval.sensor_frames.clone());
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            candidate,
        )?;
        candidate.candidate_id.clone()
    };
    state.best_idx = Some(candidate_idx);
    let mut metadata = Map::new();
    metadata.insert(
        "stage".to_string(),
        Value::String("seed_full_train".to_string()),
    );
    metadata.insert("rollout_delta".to_string(), json!(eval.rollout_count));
    metadata.insert("average_reward".to_string(), json!(eval.average_reward));
    push_stopper_snapshot(
        &mut state.stopper_states,
        &mut state.stopper_sequence,
        &context.config,
        StopperSnapshot {
            status: budget_status(&context.config, state.rollout_count, state.total_cost),
            reason: Some("seed full-train evaluation completed"),
            generation: None,
            candidate_id: Some(&candidate_id),
            evaluation_stage: Some("seed_full_train"),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            metadata,
        },
    );
    let frontier = frontier_members(&state.candidates);
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &context.config,
        candidates: &state.candidates,
        frontier: frontier.clone(),
        best_idx: state.best_idx,
        state_machine: &context.state_machine,
        rollout_count: state.rollout_count,
        total_usage: &state.total_usage,
        total_cost: state.total_cost,
    });
    let mut checkpoint_metadata = Map::new();
    checkpoint_metadata.insert(
        "stage".to_string(),
        Value::String("seed_full_train".to_string()),
    );
    record_checkpoint_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        &mut state.checkpoint_sequence,
        &context.state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "evaluation_boundary",
            status: "completed",
            reason: Some("seed full-train evaluation completed"),
            generation: None,
            candidate_id: Some(&candidate_id),
            evaluation_stage: Some("seed_full_train"),
            best_candidate_id: Some(&candidate_id),
            candidate_count: state.candidates.len(),
            frontier_count: frontier.len(),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            usage: serde_json::to_value(&state.total_usage)?,
            snapshot,
            metadata: checkpoint_metadata,
        },
    )?;
    context.events.emit(
        "candidate.evaluated",
        "Seed candidate evaluated",
        json!({"candidate_id": candidate_id, "train_reward": eval.average_reward}),
    )?;
    context.events.emit(
        "frontier.updated",
        "Frontier updated",
        frontier_snapshot_value(
            &state.candidates,
            &resources.train_rows,
            state.best_idx,
            None,
            "seed_full_train",
            Some(&candidate_id),
            Some(BTreeSet::new()),
        )?,
    )?;
    state.active_evaluation = None;
    if context.state_machine.state() == OptimizerRunState::Evaluating {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Ready,
            OptimizerTransitionTrigger::EvaluationFinished,
            "Seed candidate evaluation finished",
            json!({"candidate_id": candidate_id}),
        )?;
    }
    move_to_generation_start(
        context,
        state,
        resources,
        "seed full-train evaluation completed",
    )
}

fn move_to_generation_start(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    reason: &str,
) -> Result<GepaAdvanceOutcome> {
    state.active_evaluation = None;
    state.cursor.pending_job_id = None;
    state.cursor.pending_effect_id = None;
    state.cursor.pending_reservation_ids.clear();
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::GenerationStart,
        "completed",
        reason,
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "generation_start".to_string(),
        },
        terminal: false,
        result: None,
        message: reason.to_string(),
    })
}

fn finalize_parent_minibatch_reference(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
    eval: CandidateEvaluation,
) -> Result<GepaAdvanceOutcome> {
    let parent_idx = active.candidate_index.ok_or_else(|| {
        OptimizerError::Invariant(
            "parent minibatch reference missing parent candidate index".to_string(),
        )
    })?;
    let parent = state.candidates.get_mut(parent_idx).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "parent minibatch reference index {parent_idx} is outside candidate registry"
        ))
    })?;
    parent.sensor_frames.extend(eval.sensor_frames.clone());
    persist_candidate_snapshot(&mut context.workspace, &context.config.run.run_id, parent)?;
    context.events.emit(
        "parent_minibatch_reference.completed",
        "Parent minibatch reference completed",
        json!({
            "candidate_id": parent.candidate_id,
            "generation": active.generation,
            "proposal_index": active.proposal_index,
            "row_count": active.row_ids.len(),
            "reward": eval.average_reward,
        }),
    )?;
    move_to_proposer_waiting(
        context,
        state,
        resources,
        "parent minibatch reference evaluated",
    )
}

fn finalize_candidate_minibatch(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
    eval: CandidateEvaluation,
) -> Result<GepaAdvanceOutcome> {
    let candidate_idx = active.candidate_index.ok_or_else(|| {
        OptimizerError::Invariant("candidate minibatch missing candidate index".to_string())
    })?;
    let parent_id = state
        .candidates
        .get(candidate_idx)
        .and_then(|candidate| candidate.parent_id.clone())
        .ok_or_else(|| {
            OptimizerError::Invariant("candidate minibatch missing parent".to_string())
        })?;
    let parent_idx = state
        .candidates
        .iter()
        .position(|candidate| candidate.candidate_id == parent_id)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!("parent candidate {parent_id} is missing"))
        })?;
    let minibatch_rows = minibatch_rows(
        &resources.minibatch_rows,
        &context.config.gepa.batch_sampler,
        context.config.gepa.minibatch_size,
        active.generation,
        active.proposal_index,
        context.config.gepa.proposals_per_generation,
    );
    let parent_minibatch_reward = parent_minibatch_reward_for_rows(
        &state.candidates[parent_idx],
        &minibatch_rows,
        &context.config.dataset.train_split,
    )?
    .ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "parent candidate {} is missing minibatch reference scores for generation {}",
            parent_id, active.generation
        ))
    })?;
    {
        let candidate = state.candidates.get_mut(candidate_idx).ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "candidate minibatch index {candidate_idx} is outside candidate registry"
            ))
        })?;
        candidate.status = "minibatch_evaluated".to_string();
        candidate.minibatch_reward = Some(eval.average_reward);
        candidate.minibatch_scores = eval.scores.clone();
        candidate.sensor_frames.extend(eval.sensor_frames.clone());
    }
    let candidate_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
        objective_set: &resources.objective_set,
        candidate: &state.candidates[candidate_idx],
        rows: &minibatch_rows,
        split: &context.config.dataset.train_split,
        source_stages: &["candidate_minibatch"],
        evaluation_stage: "candidate_minibatch",
    })?;
    let parent_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
        objective_set: &resources.objective_set,
        candidate: &state.candidates[parent_idx],
        rows: &minibatch_rows,
        split: &context.config.dataset.train_split,
        source_stages: parent_minibatch_reference_source_stages(),
        evaluation_stage: "parent_minibatch_reference",
    })?;
    let minibatch_preference = compare_score_vectors(ScoreVectorPreferenceInput {
        objective_set: &resources.objective_set,
        split: &context.config.dataset.train_split,
        evaluation_stage: "candidate_minibatch",
        challenger: &candidate_minibatch_vector,
        incumbent: &parent_minibatch_vector,
        accept_equal: true,
        acceptance_criterion: Some(&context.config.gepa.acceptance_criterion),
        objective_acceptance: Some(&context.config.gepa.objective_acceptance),
        margin: context.config.gepa.minibatch_accept_margin,
    })?;
    let best_idx = state.best_idx.unwrap_or(parent_idx);
    let mut decision = AcceptanceDecision {
        candidate_id: state.candidates[candidate_idx].candidate_id.clone(),
        parent_id: parent_id.clone(),
        accepted_minibatch: minibatch_preference.preferred,
        accepted_full_train: false,
        reason: String::new(),
        candidate_minibatch_reward: eval.average_reward,
        parent_minibatch_reward,
        candidate_train_reward: None,
        best_train_reward: state.candidates[best_idx]
            .train_reward
            .unwrap_or(f64::NEG_INFINITY),
        comparison_result: minibatch_preference.result.clone(),
        score: minibatch_preference.score.clone(),
    };
    {
        let candidate = &mut state.candidates[candidate_idx];
        candidate.acceptance_score = minibatch_preference.score.clone();
        candidate.acceptance_metadata = minibatch_preference.metadata.clone();
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            candidate,
        )?;
    }
    let mut metadata = Map::new();
    metadata.insert(
        "stage".to_string(),
        Value::String("candidate_minibatch".to_string()),
    );
    metadata.insert("generation".to_string(), json!(active.generation));
    metadata.insert("rollout_delta".to_string(), json!(eval.rollout_count));
    metadata.insert("average_reward".to_string(), json!(eval.average_reward));
    push_stopper_snapshot(
        &mut state.stopper_states,
        &mut state.stopper_sequence,
        &context.config,
        StopperSnapshot {
            status: budget_status(&context.config, state.rollout_count, state.total_cost),
            reason: Some("candidate minibatch evaluation completed"),
            generation: Some(active.generation),
            candidate_id: Some(&state.candidates[candidate_idx].candidate_id),
            evaluation_stage: Some("candidate_minibatch"),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            metadata,
        },
    );
    context.events.emit(
        "candidate.minibatch_evaluated",
        "Candidate minibatch evaluated",
        json!({
            "candidate_id": state.candidates[candidate_idx].candidate_id,
            "parent_id": parent_id,
            "minibatch_reward": eval.average_reward,
            "parent_minibatch_reward": parent_minibatch_reward,
        }),
    )?;
    if !decision.accepted_minibatch {
        decision.reason = minibatch_preference.reason;
        state.candidates[candidate_idx].status = "rejected_minibatch".to_string();
        context.events.emit(
            "candidate.rejected",
            "Candidate rejected at minibatch",
            serde_json::to_value(&decision)?,
        )?;
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            &state.candidates[candidate_idx],
        )?;
        state.cursor.proposal_index += 1;
        state.active_evaluation = None;
        return move_to_proposer_waiting(
            context,
            state,
            resources,
            "candidate rejected at minibatch",
        );
    }
    let full_train_capacity =
        remaining_train_rollout_capacity(&context.workspace, &context.config, state.rollout_count)?;
    if full_train_capacity < resources.train_rows.len() {
        state.candidates[candidate_idx].status = "deferred_budget".to_string();
        decision.reason = "insufficient rollout budget for full-train evaluation".to_string();
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            &state.candidates[candidate_idx],
        )?;
        context.events.emit(
            "candidate.deferred",
            "Candidate deferred before full-train",
            serde_json::to_value(&decision)?,
        )?;
        state.cursor.proposal_index = state.proposal_queue.len();
        state.active_evaluation = None;
        return move_to_proposer_waiting(
            context,
            state,
            resources,
            "candidate deferred before full-train",
        );
    }
    if let Some(breach) = next_rollout_budget_breach(&context.workspace, &context.config)? {
        state.candidates[candidate_idx].status = "deferred_budget".to_string();
        decision.reason = "insufficient budget for full-train evaluation".to_string();
        let mut metadata = Map::new();
        metadata.insert("limit".to_string(), json!(breach.limit));
        metadata.insert("requested".to_string(), json!(breach.requested));
        metadata.insert("available".to_string(), json!(breach.available));
        state.candidates[candidate_idx].acceptance_metadata = metadata;
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            &state.candidates[candidate_idx],
        )?;
        state.cursor.proposal_index = state.proposal_queue.len();
        state.active_evaluation = None;
        return move_to_proposer_waiting(
            context,
            state,
            resources,
            "candidate deferred before full-train",
        );
    }
    let mut next = new_rollout_evaluation(
        "candidate_full_train",
        candidate_idx,
        &resources.train_rows,
        active.generation,
        active.proposal_index,
        None,
    )?;
    next.candidate_id = Some(state.candidates[candidate_idx].candidate_id.clone());
    next.parent_id = Some(parent_id);
    next.parent_minibatch_reward = Some(parent_minibatch_reward);
    next.decision = Some(decision);
    state.active_evaluation = Some(next);
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::CandidateFullTrain,
        "completed",
        "candidate minibatch accepted",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "candidate_full_train".to_string(),
        },
        terminal: false,
        result: None,
        message: "candidate minibatch accepted".to_string(),
    })
}

fn finalize_candidate_minibatch_group(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
    evaluations: Vec<(GepaActiveCandidateEvaluation, CandidateEvaluation)>,
) -> Result<GepaAdvanceOutcome> {
    let mut full_train_evaluations = Vec::new();
    let mut planned_full_train_rollouts = 0usize;
    for (candidate_active, eval) in evaluations {
        let candidate_idx = candidate_active.candidate_index;
        let parent_id = state
            .candidates
            .get(candidate_idx)
            .and_then(|candidate| candidate.parent_id.clone())
            .or(candidate_active.parent_id.clone())
            .ok_or_else(|| {
                OptimizerError::Invariant("candidate minibatch missing parent".to_string())
            })?;
        let parent_idx = state
            .candidates
            .iter()
            .position(|candidate| candidate.candidate_id == parent_id)
            .ok_or_else(|| {
                OptimizerError::Invariant(format!("parent candidate {parent_id} is missing"))
            })?;
        let minibatch_rows = minibatch_rows(
            &resources.minibatch_rows,
            &context.config.gepa.batch_sampler,
            context.config.gepa.minibatch_size,
            candidate_active.generation,
            candidate_active.proposal_index,
            context.config.gepa.proposals_per_generation,
        );
        let parent_minibatch_reward = parent_minibatch_reward_for_rows(
            &state.candidates[parent_idx],
            &minibatch_rows,
            &context.config.dataset.train_split,
        )?
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "parent candidate {} is missing minibatch reference scores for generation {}",
                parent_id, candidate_active.generation
            ))
        })?;
        {
            let candidate = state.candidates.get_mut(candidate_idx).ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "candidate minibatch index {candidate_idx} is outside candidate registry"
                ))
            })?;
            candidate.status = "minibatch_evaluated".to_string();
            candidate.minibatch_reward = Some(eval.average_reward);
            candidate.minibatch_scores = eval.scores.clone();
            candidate.sensor_frames.extend(eval.sensor_frames.clone());
        }
        let candidate_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set: &resources.objective_set,
            candidate: &state.candidates[candidate_idx],
            rows: &minibatch_rows,
            split: &context.config.dataset.train_split,
            source_stages: &["candidate_minibatch"],
            evaluation_stage: "candidate_minibatch",
        })?;
        let parent_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set: &resources.objective_set,
            candidate: &state.candidates[parent_idx],
            rows: &minibatch_rows,
            split: &context.config.dataset.train_split,
            source_stages: parent_minibatch_reference_source_stages(),
            evaluation_stage: "parent_minibatch_reference",
        })?;
        let minibatch_preference = compare_score_vectors(ScoreVectorPreferenceInput {
            objective_set: &resources.objective_set,
            split: &context.config.dataset.train_split,
            evaluation_stage: "candidate_minibatch",
            challenger: &candidate_minibatch_vector,
            incumbent: &parent_minibatch_vector,
            accept_equal: true,
            acceptance_criterion: Some(&context.config.gepa.acceptance_criterion),
            objective_acceptance: Some(&context.config.gepa.objective_acceptance),
            margin: context.config.gepa.minibatch_accept_margin,
        })?;
        let best_idx = state.best_idx.unwrap_or(parent_idx);
        let mut decision = AcceptanceDecision {
            candidate_id: state.candidates[candidate_idx].candidate_id.clone(),
            parent_id: parent_id.clone(),
            accepted_minibatch: minibatch_preference.preferred,
            accepted_full_train: false,
            reason: String::new(),
            candidate_minibatch_reward: eval.average_reward,
            parent_minibatch_reward,
            candidate_train_reward: None,
            best_train_reward: state.candidates[best_idx]
                .train_reward
                .unwrap_or(f64::NEG_INFINITY),
            comparison_result: minibatch_preference.result.clone(),
            score: minibatch_preference.score.clone(),
        };
        {
            let candidate = &mut state.candidates[candidate_idx];
            candidate.acceptance_score = minibatch_preference.score.clone();
            candidate.acceptance_metadata = minibatch_preference.metadata.clone();
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                candidate,
            )?;
        }
        let mut metadata = Map::new();
        metadata.insert(
            "stage".to_string(),
            Value::String("candidate_minibatch".to_string()),
        );
        metadata.insert("generation".to_string(), json!(candidate_active.generation));
        metadata.insert("rollout_delta".to_string(), json!(eval.rollout_count));
        metadata.insert("average_reward".to_string(), json!(eval.average_reward));
        push_stopper_snapshot(
            &mut state.stopper_states,
            &mut state.stopper_sequence,
            &context.config,
            StopperSnapshot {
                status: budget_status(&context.config, state.rollout_count, state.total_cost),
                reason: Some("candidate minibatch evaluation completed"),
                generation: Some(candidate_active.generation),
                candidate_id: Some(&state.candidates[candidate_idx].candidate_id),
                evaluation_stage: Some("candidate_minibatch"),
                rollout_count: state.rollout_count,
                cost_usd: state.total_cost,
                metadata,
            },
        );
        context.events.emit(
            "candidate.minibatch_evaluated",
            "Candidate minibatch evaluated",
            json!({
                "candidate_id": state.candidates[candidate_idx].candidate_id,
                "parent_id": parent_id,
                "minibatch_reward": eval.average_reward,
                "parent_minibatch_reward": parent_minibatch_reward,
            }),
        )?;
        if !decision.accepted_minibatch {
            decision.reason = minibatch_preference.reason;
            state.candidates[candidate_idx].status = "rejected_minibatch".to_string();
            context.events.emit(
                "candidate.rejected",
                "Candidate rejected at minibatch",
                serde_json::to_value(&decision)?,
            )?;
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                &state.candidates[candidate_idx],
            )?;
            continue;
        }
        let full_train_capacity = remaining_train_rollout_capacity(
            &context.workspace,
            &context.config,
            state.rollout_count,
        )?
        .saturating_sub(planned_full_train_rollouts);
        if full_train_capacity < resources.train_rows.len()
            || next_rollout_budget_breach(&context.workspace, &context.config)?.is_some()
        {
            state.candidates[candidate_idx].status = "deferred_budget".to_string();
            decision.reason = "insufficient budget for full-train evaluation".to_string();
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                &state.candidates[candidate_idx],
            )?;
            context.events.emit(
                "candidate.deferred",
                "Candidate deferred before full-train",
                serde_json::to_value(&decision)?,
            )?;
            continue;
        }
        let mut next = new_active_candidate_evaluation(
            state.candidates[candidate_idx].candidate_id.clone(),
            candidate_idx,
            "candidate_full_train",
            &resources.train_rows,
            candidate_active.generation,
            candidate_active.proposal_index,
            None,
        )?;
        next.parent_id = Some(parent_id);
        next.parent_minibatch_reward = Some(parent_minibatch_reward);
        next.decision = Some(decision);
        full_train_evaluations.push(next);
        planned_full_train_rollouts =
            planned_full_train_rollouts.saturating_add(resources.train_rows.len());
    }
    if full_train_evaluations.is_empty() {
        state.cursor.proposal_index = state.proposal_queue.len();
        state.active_evaluation = None;
        return move_to_proposer_waiting(
            context,
            state,
            resources,
            "candidate minibatch group finished",
        );
    }
    state.active_evaluation = Some(new_rollout_group_evaluation(
        "candidate_full_train",
        full_train_evaluations,
        active.generation,
    ));
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::CandidateFullTrain,
        "completed",
        "candidate minibatch group accepted",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "candidate_full_train".to_string(),
        },
        terminal: false,
        result: None,
        message: "candidate minibatch group accepted".to_string(),
    })
}

fn move_to_proposer_waiting(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    reason: &str,
) -> Result<GepaAdvanceOutcome> {
    if context.state_machine.state() == OptimizerRunState::Evaluating {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::EvaluationFinished,
            "Candidate evaluation finished",
            json!({"generation": state.cursor.generation}),
        )?;
    }
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::ProposerWaiting,
        "completed",
        reason,
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "proposer_waiting".to_string(),
        },
        terminal: false,
        result: None,
        message: reason.to_string(),
    })
}

fn finalize_candidate_full_train(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
    eval: CandidateEvaluation,
) -> Result<GepaAdvanceOutcome> {
    let candidate_idx = active.candidate_index.ok_or_else(|| {
        OptimizerError::Invariant("candidate full-train missing candidate index".to_string())
    })?;
    let best_idx = state.best_idx.unwrap_or(0);
    let previous_frontier_member_ids = frontier_member_ids(&frontier_members(&state.candidates));
    {
        let candidate = state.candidates.get_mut(candidate_idx).ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "candidate full-train index {candidate_idx} is outside candidate registry"
            ))
        })?;
        candidate.status = "full_train_evaluated".to_string();
        candidate.train_reward = Some(eval.average_reward);
        candidate.train_scores = eval.scores.clone();
        candidate.sensor_frames.extend(eval.sensor_frames.clone());
    }
    let candidate_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
        objective_set: &resources.objective_set,
        candidate: &state.candidates[candidate_idx],
        rows: &resources.train_rows,
        split: &context.config.dataset.train_split,
        source_stages: &["candidate_full_train"],
        evaluation_stage: "candidate_full_train",
    })?;
    let best_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
        objective_set: &resources.objective_set,
        candidate: &state.candidates[best_idx],
        rows: &resources.train_rows,
        split: &context.config.dataset.train_split,
        source_stages: &["seed_full_train", "candidate_full_train"],
        evaluation_stage: "best_full_train_reference",
    })?;
    let train_preference = compare_score_vectors(ScoreVectorPreferenceInput {
        objective_set: &resources.objective_set,
        split: &context.config.dataset.train_split,
        evaluation_stage: "candidate_full_train",
        challenger: &candidate_train_vector,
        incumbent: &best_train_vector,
        accept_equal: true,
        acceptance_criterion: Some(&context.config.gepa.acceptance_criterion),
        objective_acceptance: Some(&context.config.gepa.objective_acceptance),
        margin: 0.0,
    })?;
    let accepted = train_preference.preferred;
    let mut decision = active.decision.unwrap_or_else(|| AcceptanceDecision {
        candidate_id: state.candidates[candidate_idx].candidate_id.clone(),
        parent_id: state.candidates[candidate_idx]
            .parent_id
            .clone()
            .unwrap_or_default(),
        accepted_minibatch: true,
        accepted_full_train: false,
        reason: String::new(),
        candidate_minibatch_reward: state.candidates[candidate_idx]
            .minibatch_reward
            .unwrap_or(0.0),
        parent_minibatch_reward: active.parent_minibatch_reward.unwrap_or(0.0),
        candidate_train_reward: None,
        best_train_reward: state.candidates[best_idx]
            .train_reward
            .unwrap_or(f64::NEG_INFINITY),
        comparison_result: train_preference.result.clone(),
        score: train_preference.score.clone(),
    });
    decision.candidate_train_reward = Some(eval.average_reward);
    decision.accepted_full_train = accepted;
    decision.reason = train_preference.reason;
    decision.comparison_result = train_preference.result;
    decision.score = train_preference.score.clone();
    {
        let candidate = &mut state.candidates[candidate_idx];
        candidate.acceptance_score = train_preference.score;
        candidate.acceptance_metadata = train_preference.metadata;
        candidate.status = if accepted {
            "accepted".to_string()
        } else {
            "rejected_full_train".to_string()
        };
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            candidate,
        )?;
    }
    let mut metadata = Map::new();
    metadata.insert(
        "stage".to_string(),
        Value::String("candidate_full_train".to_string()),
    );
    metadata.insert("generation".to_string(), json!(active.generation));
    metadata.insert("rollout_delta".to_string(), json!(eval.rollout_count));
    metadata.insert("average_reward".to_string(), json!(eval.average_reward));
    push_stopper_snapshot(
        &mut state.stopper_states,
        &mut state.stopper_sequence,
        &context.config,
        StopperSnapshot {
            status: budget_status(&context.config, state.rollout_count, state.total_cost),
            reason: Some("candidate full-train evaluation completed"),
            generation: Some(active.generation),
            candidate_id: Some(&state.candidates[candidate_idx].candidate_id),
            evaluation_stage: Some("candidate_full_train"),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            metadata,
        },
    );
    context.events.emit(
        "candidate.full_train_evaluated",
        "Candidate full train evaluated",
        json!({
            "candidate_id": state.candidates[candidate_idx].candidate_id,
            "parent_id": state.candidates[candidate_idx].parent_id,
            "train_reward": eval.average_reward,
            "best_train_reward": state.candidates[best_idx].train_reward,
        }),
    )?;
    context.events.emit(
        if accepted {
            "candidate.accepted"
        } else {
            "candidate.rejected"
        },
        if accepted {
            "Candidate accepted"
        } else {
            "Candidate rejected"
        },
        serde_json::to_value(&decision)?,
    )?;
    if accepted {
        state.best_idx = Some(candidate_idx);
        context.events.emit(
            "frontier.updated",
            "Frontier updated",
            frontier_snapshot_value(
                &state.candidates,
                &resources.train_rows,
                state.best_idx,
                Some(active.generation),
                "candidate_accepted",
                Some(&state.candidates[candidate_idx].candidate_id),
                Some(previous_frontier_member_ids),
            )?,
        )?;
    }
    state.cursor.proposal_index += 1;
    state.active_evaluation = None;
    move_to_proposer_waiting(
        context,
        state,
        resources,
        "candidate full-train evaluation finished",
    )
}

fn finalize_candidate_full_train_group(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    _active: GepaActiveEvaluation,
    evaluations: Vec<(GepaActiveCandidateEvaluation, CandidateEvaluation)>,
) -> Result<GepaAdvanceOutcome> {
    for (candidate_active, eval) in evaluations {
        let candidate_idx = candidate_active.candidate_index;
        let best_idx = state.best_idx.unwrap_or(0);
        let previous_frontier_member_ids =
            frontier_member_ids(&frontier_members(&state.candidates));
        {
            let candidate = state.candidates.get_mut(candidate_idx).ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "candidate full-train index {candidate_idx} is outside candidate registry"
                ))
            })?;
            candidate.status = "full_train_evaluated".to_string();
            candidate.train_reward = Some(eval.average_reward);
            candidate.train_scores = eval.scores.clone();
            candidate.sensor_frames.extend(eval.sensor_frames.clone());
        }
        let candidate_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set: &resources.objective_set,
            candidate: &state.candidates[candidate_idx],
            rows: &resources.train_rows,
            split: &context.config.dataset.train_split,
            source_stages: &["candidate_full_train"],
            evaluation_stage: "candidate_full_train",
        })?;
        let best_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set: &resources.objective_set,
            candidate: &state.candidates[best_idx],
            rows: &resources.train_rows,
            split: &context.config.dataset.train_split,
            source_stages: &["seed_full_train", "candidate_full_train"],
            evaluation_stage: "best_full_train_reference",
        })?;
        let train_preference = compare_score_vectors(ScoreVectorPreferenceInput {
            objective_set: &resources.objective_set,
            split: &context.config.dataset.train_split,
            evaluation_stage: "candidate_full_train",
            challenger: &candidate_train_vector,
            incumbent: &best_train_vector,
            accept_equal: true,
            acceptance_criterion: Some(&context.config.gepa.acceptance_criterion),
            objective_acceptance: Some(&context.config.gepa.objective_acceptance),
            margin: 0.0,
        })?;
        let accepted = train_preference.preferred;
        let mut decision =
            candidate_active
                .decision
                .clone()
                .unwrap_or_else(|| AcceptanceDecision {
                    candidate_id: state.candidates[candidate_idx].candidate_id.clone(),
                    parent_id: state.candidates[candidate_idx]
                        .parent_id
                        .clone()
                        .unwrap_or_default(),
                    accepted_minibatch: true,
                    accepted_full_train: false,
                    reason: String::new(),
                    candidate_minibatch_reward: state.candidates[candidate_idx]
                        .minibatch_reward
                        .unwrap_or(0.0),
                    parent_minibatch_reward: candidate_active
                        .parent_minibatch_reward
                        .unwrap_or(0.0),
                    candidate_train_reward: None,
                    best_train_reward: state.candidates[best_idx]
                        .train_reward
                        .unwrap_or(f64::NEG_INFINITY),
                    comparison_result: train_preference.result.clone(),
                    score: train_preference.score.clone(),
                });
        decision.candidate_train_reward = Some(eval.average_reward);
        decision.accepted_full_train = accepted;
        decision.reason = train_preference.reason;
        decision.comparison_result = train_preference.result;
        decision.score = train_preference.score.clone();
        {
            let candidate = &mut state.candidates[candidate_idx];
            candidate.acceptance_score = train_preference.score;
            candidate.acceptance_metadata = train_preference.metadata;
            candidate.status = if accepted {
                "accepted".to_string()
            } else {
                "rejected_full_train".to_string()
            };
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                candidate,
            )?;
        }
        let mut metadata = Map::new();
        metadata.insert(
            "stage".to_string(),
            Value::String("candidate_full_train".to_string()),
        );
        metadata.insert("generation".to_string(), json!(candidate_active.generation));
        metadata.insert("rollout_delta".to_string(), json!(eval.rollout_count));
        metadata.insert("average_reward".to_string(), json!(eval.average_reward));
        push_stopper_snapshot(
            &mut state.stopper_states,
            &mut state.stopper_sequence,
            &context.config,
            StopperSnapshot {
                status: budget_status(&context.config, state.rollout_count, state.total_cost),
                reason: Some("candidate full-train evaluation completed"),
                generation: Some(candidate_active.generation),
                candidate_id: Some(&state.candidates[candidate_idx].candidate_id),
                evaluation_stage: Some("candidate_full_train"),
                rollout_count: state.rollout_count,
                cost_usd: state.total_cost,
                metadata,
            },
        );
        context.events.emit(
            "candidate.full_train_evaluated",
            "Candidate full train evaluated",
            json!({
                "candidate_id": state.candidates[candidate_idx].candidate_id,
                "parent_id": state.candidates[candidate_idx].parent_id,
                "train_reward": eval.average_reward,
                "best_train_reward": state.candidates[best_idx].train_reward,
            }),
        )?;
        context.events.emit(
            if accepted {
                "candidate.accepted"
            } else {
                "candidate.rejected"
            },
            if accepted {
                "Candidate accepted"
            } else {
                "Candidate rejected"
            },
            serde_json::to_value(&decision)?,
        )?;
        if accepted {
            state.best_idx = Some(candidate_idx);
            context.events.emit(
                "frontier.updated",
                "Frontier updated",
                frontier_snapshot_value(
                    &state.candidates,
                    &resources.train_rows,
                    state.best_idx,
                    Some(candidate_active.generation),
                    "candidate_accepted",
                    Some(&state.candidates[candidate_idx].candidate_id),
                    Some(previous_frontier_member_ids),
                )?,
            )?;
        }
    }
    state.cursor.proposal_index = state.proposal_queue.len();
    state.active_evaluation = None;
    move_to_proposer_waiting(
        context,
        state,
        resources,
        "candidate full-train group finished",
    )
}

fn advance_generation_start(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    if state.cursor.generation >= context.config.gepa.max_generations {
        return move_to_pre_heldout(context, state, resources);
    }
    if train_rollout_budget_reached(&context.config, state.rollout_count)
        || cost_budget_reached(&context.config, state.total_cost)
        || service_stop_condition_reached(&context.config, state)
    {
        return move_to_pre_heldout(context, state, resources);
    }
    if let Some(train_best_idx) = select_best_train_candidate(
        &state.candidates,
        &resources.objective_set,
        &context.config.dataset.train_split,
        &resources.train_rows,
    )? {
        state.best_idx = Some(train_best_idx);
    }
    let parent_selection = select_proposer_parent_candidate(
        &state.candidates,
        &resources.train_rows,
        &resources.objective_set,
        &context.config.gepa.candidate_selector,
        state.cursor.generation,
        &context.config.run.run_id,
        state.best_idx,
    )?;
    let parent_idx = parent_selection.candidate_index;
    let parent_id = state
        .candidates
        .get(parent_idx)
        .map(|candidate| candidate.candidate_id.clone())
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "parent index {parent_idx} is outside candidate registry"
            ))
        })?;
    if context.state_machine.state() == OptimizerRunState::Ready {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Proposing,
            OptimizerTransitionTrigger::ProposerStarted,
            "Proposer started",
            proposer_started_details(
                &context.config,
                &state.candidates,
                state.cursor.generation,
                &parent_id,
                parent_selection.metadata.clone(),
                &context.paths.run_dir,
            ),
        )?;
    }
    let queued = plan_proposer_runtime_job(context, resources, parent_idx, state)?;
    state.cursor.pipeline_state.parent_pool_version =
        Some(state.cursor.pipeline_state.pool_version);
    state.cursor.pipeline_state.parent_candidate_id = Some(parent_id.clone());
    state.active_evaluation = Some(GepaActiveEvaluation {
        stage: "proposer".to_string(),
        candidate_id: Some(parent_id),
        candidate_index: Some(parent_idx),
        generation: state.cursor.generation,
        proposal_index: 0,
        row_ids: Vec::new(),
        next_row_index: 0,
        planned_job_id: Some(queued.job.job_id.clone()),
        effect_id: Some(queued.effect.runtime_effect_id.clone()),
        reservation_id: Some(queued.reservation.budget_reservation_id.clone()),
        heldout_candidate_index: None,
        parent_id: None,
        scores: Vec::new(),
        sensor_frames: Vec::new(),
        reward_sum: 0.0,
        usage: UsageTotals::default(),
        cost_usd: 0.0,
        rollout_count: 0,
        parent_minibatch_reward: None,
        decision: None,
        candidate_evaluations: Vec::new(),
    });
    state.cursor.pending_job_id = Some(queued.job.job_id.clone());
    state.cursor.pending_effect_id = Some(queued.effect.runtime_effect_id.clone());
    state.cursor.pending_reservation_ids = vec![queued.reservation.budget_reservation_id.clone()];
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::ProposerWaiting,
        "planned",
        "planned proposer job",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::PlanRuntimeJob {
            run_id: context.config.run.run_id.clone(),
            job_id: queued.job.job_id,
        },
        terminal: false,
        result: None,
        message: "planned proposer job".to_string(),
    })
}

fn plan_proposer_runtime_job(
    context: &GepaRunContext,
    resources: &GepaStepResources,
    parent_idx: usize,
    state: &GepaRunState,
) -> Result<runtime::QueuedRuntimeEffect> {
    let configured_limits = ConfiguredGepaRunLimits::from_config(&context.config);
    let parent = state.candidates.get(parent_idx).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "parent index {parent_idx} is outside candidate registry"
        ))
    })?;
    let workspace_dir = context
        .paths
        .run_dir
        .join("proposer_workspaces")
        .join(format!("generation_{:03}", state.cursor.generation));
    let request = json!({
        "backend": context.config.proposer.backend,
        "execution_mode": context.config.proposer.execution_mode,
        "model": context.config.proposer.model,
        "generation": state.cursor.generation,
        "parent": parent,
        "candidates": state.candidates,
        "program": resources.program,
        "seed_pool_rows": seed_pool_rows_value(
            &resources.train_rows,
            &resources.minibatch_rows,
            &resources.reflection_rows,
            &resources.heldout_rows,
        ),
        "workspace_root": context.paths.run_dir,
        "run_artifact_dir": context.paths.run_dir,
        "proposal_artifact_dir": workspace_dir,
        "lever_manifest": LeverManifest::from_prompt_program(&resources.program),
        "frontier_summary": proposer_frontier_summary(
            &state.candidates,
            &resources.train_rows,
            state.best_idx,
        )?,
        "minibatch_failures": proposer_minibatch_failures(&state.candidates),
        "rollout_trace_artifact_refs": proposer_rollout_trace_artifact_refs(&state.candidates),
        "merge_evidence_artifacts": proposer_merge_evidence_artifacts(&context.paths)?,
        "target_modules": context.config.candidate.target_modules,
        "proposal_count": context.config.gepa.proposals_per_generation,
    });
    let mut cache_metadata = Map::new();
    cache_metadata.insert(
        "backend".to_string(),
        json!(&context.config.proposer.backend),
    );
    cache_metadata.insert("generation".to_string(), json!(state.cursor.generation));
    cache_metadata.insert(
        "parent_candidate_id".to_string(),
        json!(&parent.candidate_id),
    );
    cache_metadata.insert(
        "proposal_count".to_string(),
        json!(context.config.gepa.proposals_per_generation),
    );
    let proposer_namespace = format!("{}:proposer.codex", context.cache_namespace);
    let planned_cache_key =
        RequestCache::cache_key_with_profile(&proposer_namespace, &request, PROPOSER_CACHE_PROFILE);
    let mut effect_metadata = cache_metadata.clone();
    effect_metadata.insert("algorithm_id".to_string(), json!(GEPA_ALGORITHM_ID));
    let dispatch_payload = runtime::RuntimeEffectDispatchPayload::proposer(
        proposer_namespace,
        PROPOSER_CACHE_PROFILE,
        cache_metadata,
        request.clone(),
        state.cursor.generation,
        parent.candidate_id.clone(),
        workspace_dir.display().to_string(),
    );
    record_runtime_effect_planned(
        &context.workspace,
        RuntimeEffectPlanInput {
            run_id: &context.config.run.run_id,
            effect_kind: "candidate_proposal",
            lane: "proposer",
            subject_type: "generation",
            subject_id: &format!("generation_{:03}", state.cursor.generation),
            idempotency_key: &planned_cache_key,
            job_kind: OptimizerJobKind::Proposer,
            candidate_id: Some(&parent.candidate_id),
            cache_key: Some(planned_cache_key.clone()),
            budget_estimate: configured_limits.proposer_budget_estimate(),
            payload: json!({
                "generation": state.cursor.generation,
                "parent_candidate_id": parent.candidate_id,
                "backend": context.config.proposer.backend,
            }),
            dispatch_payload,
            metadata: effect_metadata,
        },
    )
}

fn advance_proposer_waiting(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    if state.cursor.proposal_index >= state.proposal_queue.len() {
        return complete_generation_boundary(context, state, resources);
    }
    if train_rollout_budget_reached(&context.config, state.rollout_count)
        || cost_budget_reached(&context.config, state.total_cost)
    {
        state.cursor.proposal_index = state.proposal_queue.len();
        return complete_generation_boundary(context, state, resources);
    }
    let parent_idx = current_proposal_parent_idx(state)?;
    let minibatch_capacity =
        remaining_train_rollout_capacity(&context.workspace, &context.config, state.rollout_count)?;
    let mut active_candidates = Vec::new();
    let mut planned_candidate_ids = BTreeSet::new();
    let mut planned_rollouts = 0usize;
    let mut proposal_index = state.cursor.proposal_index;
    let admission_limit = pipeline_candidate_admission_limit(&context.config);
    while proposal_index < state.proposal_queue.len() && active_candidates.len() < admission_limit {
        let proposal = state
            .proposal_queue
            .get(proposal_index)
            .cloned()
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "proposal index {proposal_index} is outside proposal queue"
                ))
            })?;
        let proposal_parent_idx = proposal_parent_idx(state, &proposal, parent_idx);
        let proposal_parent = state
            .candidates
            .get(proposal_parent_idx)
            .cloned()
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "proposal parent index {proposal_parent_idx} is outside candidate registry"
                ))
            })?;
        let payload = normalize_candidate_payload(
            &resources.program,
            &context.config,
            &proposal_parent.payload,
            proposal.payload_map(),
        )?;
        let candidate_id = candidate_id(&payload);
        if planned_candidate_ids.contains(&candidate_id) {
            context.events.emit(
                "candidate.duplicate_skipped",
                "Duplicate candidate skipped",
                json!({"candidate_id": candidate_id, "generation": state.cursor.generation}),
            )?;
            proposal_index += 1;
            continue;
        }
        let candidate_idx = if let Some(existing_idx) = state
            .candidates
            .iter()
            .position(|candidate| candidate.candidate_id == candidate_id)
        {
            let existing = &state.candidates[existing_idx];
            if matches!(existing.status.as_str(), "registered")
                && existing.minibatch_reward.is_none()
                && existing.train_reward.is_none()
            {
                planned_candidate_ids.insert(candidate_id.clone());
                existing_idx
            } else {
                context.events.emit(
                    "candidate.duplicate_skipped",
                    "Duplicate candidate skipped",
                    json!({"candidate_id": candidate_id, "generation": state.cursor.generation}),
                )?;
                state.best_idx.get_or_insert(existing_idx);
                proposal_index += 1;
                continue;
            }
        } else {
            planned_candidate_ids.insert(candidate_id.clone());
            let proposal_type = proposal.proposal_type_or_default();
            let proposal_parent_id = proposal_parent.candidate_id.clone();
            let mut acceptance_metadata = Map::new();
            acceptance_metadata.insert("proposal".to_string(), proposal.metadata_value());
            acceptance_metadata.insert("generation".to_string(), json!(state.cursor.generation));
            let lever_bundle = proposal.lever_bundle.clone().unwrap_or_else(|| {
                LeverBundle::from_prompt_payload(
                    candidate_id.clone(),
                    Some(proposal_parent_id.clone()),
                    &payload,
                )
            });
            let candidate = CandidateRecord {
                lever_bundle,
                candidate_id,
                payload,
                parent_id: Some(proposal_parent_id),
                source: format!("reflector:{proposal_type}"),
                status: "registered".to_string(),
                minibatch_reward: None,
                train_reward: None,
                heldout_reward: None,
                minibatch_scores: Vec::new(),
                train_scores: Vec::new(),
                sensor_frames: Vec::new(),
                acceptance_score: Value::Null,
                acceptance_metadata,
            };
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                &candidate,
            )?;
            state.candidates.push(candidate);
            state.candidates.len() - 1
        };
        let minibatch_rows = minibatch_rows(
            &resources.minibatch_rows,
            &context.config.gepa.batch_sampler,
            context.config.gepa.minibatch_size,
            state.cursor.generation,
            proposal_index,
            context.config.gepa.proposals_per_generation,
        );
        if parent_minibatch_reward_for_rows(
            &proposal_parent,
            &minibatch_rows,
            &context.config.dataset.train_split,
        )?
        .is_none()
        {
            let remaining_capacity = minibatch_capacity.saturating_sub(planned_rollouts);
            if remaining_capacity < minibatch_rows.len() {
                state.cursor.proposal_index = state.proposal_queue.len();
                return complete_generation_boundary(context, state, resources);
            }
            if let Some(_breach) = next_rollout_budget_breach(&context.workspace, &context.config)?
            {
                state.cursor.proposal_index = state.proposal_queue.len();
                return complete_generation_boundary(context, state, resources);
            }
            let mut parent_reference = new_rollout_evaluation(
                "parent_minibatch_reference",
                proposal_parent_idx,
                &minibatch_rows,
                state.cursor.generation,
                proposal_index,
                None,
            )?;
            parent_reference.candidate_id = Some(proposal_parent.candidate_id.clone());
            state.active_evaluation = Some(parent_reference);
            persist_gepa_run_state(
                context,
                state,
                resources,
                GepaCursorPhase::CandidateMinibatch,
                "planned",
                "parent minibatch reference evaluation started",
                Map::new(),
            )?;
            return Ok(GepaAdvanceOutcome {
                action: planner::GepaTickAction::CheckpointRun {
                    run_id: context.config.run.run_id.clone(),
                    phase: "parent_minibatch_reference".to_string(),
                },
                terminal: false,
                result: None,
                message: "parent minibatch reference evaluation started".to_string(),
            });
        }
        let remaining_capacity = minibatch_capacity.saturating_sub(planned_rollouts);
        if remaining_capacity < minibatch_rows.len() {
            state.candidates[candidate_idx].status = "deferred_budget".to_string();
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                &state.candidates[candidate_idx],
            )?;
            context.events.emit(
                "candidate.deferred",
                "Candidate deferred before minibatch",
                json!({
                    "candidate_id": state.candidates[candidate_idx].candidate_id,
                    "generation": state.cursor.generation,
                    "stage": "candidate_minibatch",
                    "required_rollouts": minibatch_rows.len(),
                    "available_rollouts": remaining_capacity,
                }),
            )?;
            proposal_index = state.proposal_queue.len();
            break;
        }
        if let Some(breach) = next_rollout_budget_breach(&context.workspace, &context.config)? {
            state.candidates[candidate_idx].status = "deferred_budget".to_string();
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                &state.candidates[candidate_idx],
            )?;
            context.events.emit(
                "candidate.deferred",
                "Candidate deferred before minibatch",
                json!({
                    "candidate_id": state.candidates[candidate_idx].candidate_id,
                    "generation": state.cursor.generation,
                    "stage": "candidate_minibatch",
                    "limit": breach.limit,
                    "requested": breach.requested,
                    "available": breach.available,
                }),
            )?;
            proposal_index = state.proposal_queue.len();
            break;
        }
        let mut active = new_active_candidate_evaluation(
            state.candidates[candidate_idx].candidate_id.clone(),
            candidate_idx,
            "candidate_minibatch",
            &minibatch_rows,
            state.cursor.generation,
            proposal_index,
            None,
        )?;
        active.parent_id = state.candidates[candidate_idx].parent_id.clone();
        active_candidates.push(active);
        planned_rollouts = planned_rollouts.saturating_add(minibatch_rows.len());
        proposal_index += 1;
    }
    state.cursor.proposal_index = proposal_index;
    if active_candidates.is_empty() {
        return move_to_proposer_waiting(
            context,
            state,
            resources,
            "no new candidate minibatches queued",
        );
    }
    state.active_evaluation = Some(new_rollout_group_evaluation(
        "candidate_minibatch",
        active_candidates,
        state.cursor.generation,
    ));
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::CandidateMinibatch,
        "planned",
        "candidate minibatch evaluation started",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "candidate_minibatch".to_string(),
        },
        terminal: false,
        result: None,
        message: "candidate minibatch evaluation started".to_string(),
    })
}

fn pipeline_candidate_admission_limit(config: &SynthOptimizerConfig) -> usize {
    match GepaPipelineRuntimePlan::from_config(config) {
        Ok(GepaPipelineRuntimePlan::AsyncPipelined(plan)) => plan.max_in_flight_candidates.max(1),
        _ => usize::MAX,
    }
}

fn complete_generation_boundary(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    if context.state_machine.state() != OptimizerRunState::Ready
        && context.state_machine.state() == OptimizerRunState::RolloutQueueing
    {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Ready,
            OptimizerTransitionTrigger::EvaluationFinished,
            "Generation evaluation finished",
            json!({"generation": state.cursor.generation}),
        )?;
    }
    context.events.emit(
        "frontier.snapshot",
        "Frontier generation snapshot",
        frontier_snapshot_value(
            &state.candidates,
            &resources.train_rows,
            state.best_idx,
            Some(state.cursor.generation),
            "generation_complete",
            None,
            None,
        )?,
    )?;
    let frontier = frontier_members(&state.candidates);
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &context.config,
        candidates: &state.candidates,
        frontier: frontier.clone(),
        best_idx: state.best_idx,
        state_machine: &context.state_machine,
        rollout_count: state.rollout_count,
        total_usage: &state.total_usage,
        total_cost: state.total_cost,
    });
    let mut metadata = Map::new();
    metadata.insert("generation".to_string(), json!(state.cursor.generation));
    metadata.insert(
        "stage".to_string(),
        Value::String("generation_complete".to_string()),
    );
    record_checkpoint_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        &mut state.checkpoint_sequence,
        &context.state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "generation_boundary",
            status: "completed",
            reason: Some("generation evaluation completed"),
            generation: Some(state.cursor.generation),
            candidate_id: state
                .best_idx
                .and_then(|idx| state.candidates.get(idx))
                .map(|candidate| candidate.candidate_id.as_str()),
            evaluation_stage: Some("generation_complete"),
            best_candidate_id: state
                .best_idx
                .and_then(|idx| state.candidates.get(idx))
                .map(|candidate| candidate.candidate_id.as_str()),
            candidate_count: state.candidates.len(),
            frontier_count: frontier.len(),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            usage: serde_json::to_value(&state.total_usage)?,
            snapshot,
            metadata,
        },
    )?;
    state.cursor.generation += 1;
    state.cursor.proposal_index = 0;
    state.proposal_queue.clear();
    state.active_evaluation = None;
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::GenerationStart,
        "completed",
        "generation evaluation completed",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "generation_boundary".to_string(),
        },
        terminal: false,
        result: None,
        message: "generation evaluation completed".to_string(),
    })
}

fn move_to_pre_heldout(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    let best_idx = state.best_idx.unwrap_or(0);
    let frontier = frontier_members(&state.candidates);
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &context.config,
        candidates: &state.candidates,
        frontier: frontier.clone(),
        best_idx: Some(best_idx),
        state_machine: &context.state_machine,
        rollout_count: state.rollout_count,
        total_usage: &state.total_usage,
        total_cost: state.total_cost,
    });
    let mut metadata = Map::new();
    metadata.insert(
        "stage".to_string(),
        Value::String("pre_heldout".to_string()),
    );
    metadata.insert(
        "heldout_rows".to_string(),
        json!(resources.heldout_rows.len()),
    );
    record_checkpoint_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        &mut state.checkpoint_sequence,
        &context.state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "pre_heldout",
            status: "completed",
            reason: Some("optimization loop completed before heldout"),
            generation: None,
            candidate_id: Some(&state.candidates[best_idx].candidate_id),
            evaluation_stage: Some("pre_heldout"),
            best_candidate_id: Some(&state.candidates[best_idx].candidate_id),
            candidate_count: state.candidates.len(),
            frontier_count: frontier.len(),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            usage: serde_json::to_value(&state.total_usage)?,
            snapshot,
            metadata,
        },
    )?;
    state.heldout_candidate_index = 0;
    state.active_evaluation = None;
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::Heldout,
        "completed",
        "optimization loop completed before heldout",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "pre_heldout".to_string(),
        },
        terminal: false,
        result: None,
        message: "optimization loop completed before heldout".to_string(),
    })
}

fn advance_heldout(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    if state
        .active_evaluation
        .as_ref()
        .is_some_and(|active| active.stage == "heldout")
    {
        if state
            .active_evaluation
            .as_ref()
            .is_some_and(active_rollout_evaluation_complete)
        {
            return finalize_active_rollout_evaluation(context, state, resources);
        }
        return plan_next_rollout_batch(context, state, resources);
    }
    let all_heldout_indices = heldout_candidate_indices(state);
    let mut heldout_indices = all_heldout_indices.clone();
    let evaluated_heldout_indices = all_heldout_indices
        .iter()
        .copied()
        .filter(|idx| state.candidates[*idx].heldout_reward.is_some())
        .collect::<Vec<_>>();
    if !evaluated_heldout_indices.is_empty() {
        heldout_indices = evaluated_heldout_indices;
    }
    if heldout_indices.is_empty() || resources.heldout_rows.is_empty() {
        return move_to_finalizing(context, state, resources, "heldout evaluation skipped");
    }
    if state.heldout_candidate_index == 0 {
        let available_rollouts = remaining_heldout_rollout_capacity(
            &context.workspace,
            &context.config,
            &state.candidates,
        )?;
        let budget_breach = next_rollout_budget_breach(&context.workspace, &context.config)?;
        if budget_breach.is_none() {
            heldout_indices = budgeted_heldout_candidate_indices(
                &state.candidates,
                all_heldout_indices.clone(),
                state.best_idx,
                available_rollouts,
                resources.heldout_rows.len(),
            );
        }
        let required_rollouts = heldout_indices
            .len()
            .saturating_mul(resources.heldout_rows.len());
        if heldout_indices.is_empty()
            || available_rollouts < required_rollouts
            || budget_breach.is_some()
        {
            let best_idx = state.best_idx.unwrap_or(0);
            let mut metadata = Map::new();
            metadata.insert("stage".to_string(), Value::String("heldout".to_string()));
            metadata.insert("required_rollouts".to_string(), json!(required_rollouts));
            metadata.insert("available_rollouts".to_string(), json!(available_rollouts));
            push_stopper_snapshot(
                &mut state.stopper_states,
                &mut state.stopper_sequence,
                &context.config,
                StopperSnapshot {
                    status: "heldout_skipped_limit_reached",
                    reason: Some("insufficient rollout budget for heldout evaluation"),
                    generation: None,
                    candidate_id: Some(&state.candidates[best_idx].candidate_id),
                    evaluation_stage: Some("heldout"),
                    rollout_count: state.rollout_count,
                    cost_usd: state.total_cost,
                    metadata,
                },
            );
            context.events.emit(
                "heldout.skipped",
                "Heldout skipped due to limits",
                json!({
                    "best_candidate_id": state.candidates[best_idx].candidate_id,
                    "required_rollouts": required_rollouts,
                    "available_rollouts": available_rollouts,
                }),
            )?;
            return move_to_finalizing(context, state, resources, "heldout skipped due to limits");
        }
        let total_heldout_candidates = all_heldout_indices.len();
        if heldout_indices.len() < total_heldout_candidates {
            context.events.emit(
                "heldout.partial",
                "Heldout limited to budgeted candidate subset",
                json!({
                    "candidate_count": heldout_indices.len(),
                    "total_candidate_count": total_heldout_candidates,
                    "available_rollouts": available_rollouts,
                    "required_rollouts": required_rollouts,
                }),
            )?;
        }
        transition_to_rollout_running(
            context,
            "Heldout rollouts started",
            json!({
                "stage": "heldout",
                "row_count": resources.heldout_rows.len(),
                "candidate_count": heldout_indices.len(),
                "rollout_count": required_rollouts,
            }),
        )?;
    }
    if state.heldout_candidate_index >= heldout_indices.len() {
        if let Some(best_heldout_idx) = select_best_heldout_candidate(HeldoutSelectionInput {
            candidates: &state.candidates,
            evaluated_indices: &heldout_indices,
            objective_set: &resources.objective_set,
            heldout_split: &context.config.dataset.heldout_split,
            heldout_rows: &resources.heldout_rows,
            train_split: &context.config.dataset.train_split,
            train_rows: &resources.train_rows,
            incumbent_idx: state.best_idx,
        })? {
            state.best_idx = Some(best_heldout_idx);
        }
        return move_to_finalizing(context, state, resources, "heldout evaluation completed");
    }
    let mut active_candidates = Vec::new();
    for (heldout_offset, candidate_idx) in heldout_indices
        .iter()
        .copied()
        .enumerate()
        .skip(state.heldout_candidate_index)
    {
        active_candidates.push(new_active_candidate_evaluation(
            state.candidates[candidate_idx].candidate_id.clone(),
            candidate_idx,
            "heldout",
            &resources.heldout_rows,
            state.cursor.generation,
            0,
            Some(heldout_offset),
        )?);
    }
    state.active_evaluation = Some(new_rollout_group_evaluation(
        "heldout",
        active_candidates,
        state.cursor.generation,
    ));
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::Heldout,
        "planned",
        "heldout candidate evaluation started",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "heldout".to_string(),
        },
        terminal: false,
        result: None,
        message: "heldout candidate evaluation started".to_string(),
    })
}

fn heldout_candidate_indices(state: &GepaRunState) -> Vec<usize> {
    let mut indices = state
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| candidate.train_reward.map(|_| idx))
        .collect::<Vec<_>>();
    if indices.is_empty() {
        if let Some(best_idx) = state.best_idx {
            indices.push(best_idx);
        }
    }
    indices
}

fn budgeted_heldout_candidate_indices(
    candidates: &[CandidateRecord],
    indices: Vec<usize>,
    best_idx: Option<usize>,
    available_rollouts: usize,
    heldout_row_count: usize,
) -> Vec<usize> {
    if heldout_row_count == 0 || indices.is_empty() {
        return indices;
    }
    let max_candidates = available_rollouts / heldout_row_count;
    if max_candidates >= indices.len() {
        return indices;
    }
    if max_candidates == 0 {
        return Vec::new();
    }
    let mut selected = Vec::new();
    let push_index = |idx: usize, selected: &mut Vec<usize>| {
        if indices.contains(&idx) && !selected.contains(&idx) && selected.len() < max_candidates {
            selected.push(idx);
        }
    };
    if let Some(best_idx) = best_idx {
        push_index(best_idx, &mut selected);
    }
    push_index(0, &mut selected);
    let mut by_train = indices.clone();
    by_train.sort_by(|left, right| {
        let left_reward = candidates[*left].train_reward.unwrap_or(f64::NEG_INFINITY);
        let right_reward = candidates[*right].train_reward.unwrap_or(f64::NEG_INFINITY);
        right_reward.total_cmp(&left_reward).then_with(|| {
            candidates[*left]
                .candidate_id
                .cmp(&candidates[*right].candidate_id)
        })
    });
    for idx in by_train {
        push_index(idx, &mut selected);
    }
    selected
}

fn finalize_heldout_candidate(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    active: GepaActiveEvaluation,
    eval: CandidateEvaluation,
) -> Result<GepaAdvanceOutcome> {
    let candidate_idx = active.candidate_index.ok_or_else(|| {
        OptimizerError::Invariant("heldout evaluation missing candidate index".to_string())
    })?;
    state.candidates[candidate_idx].heldout_reward = Some(eval.average_reward);
    state.candidates[candidate_idx]
        .sensor_frames
        .extend(eval.sensor_frames.clone());
    persist_candidate_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        &state.candidates[candidate_idx],
    )?;
    context.events.emit(
        "heldout.completed",
        "Heldout evaluation completed",
        json!({
            "candidate_id": state.candidates[candidate_idx].candidate_id,
            "train_reward": state.candidates[candidate_idx].train_reward,
            "heldout_reward": eval.average_reward,
        }),
    )?;
    state.heldout_candidate_index += 1;
    state.active_evaluation = None;
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::Heldout,
        "completed",
        "heldout candidate evaluation completed",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "heldout".to_string(),
        },
        terminal: false,
        result: None,
        message: "heldout candidate evaluation completed".to_string(),
    })
}

fn finalize_heldout_group(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    _active: GepaActiveEvaluation,
    evaluations: Vec<(GepaActiveCandidateEvaluation, CandidateEvaluation)>,
) -> Result<GepaAdvanceOutcome> {
    for (candidate_active, eval) in evaluations {
        let candidate_idx = candidate_active.candidate_index;
        state.candidates[candidate_idx].heldout_reward = Some(eval.average_reward);
        state.candidates[candidate_idx]
            .sensor_frames
            .extend(eval.sensor_frames.clone());
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            &state.candidates[candidate_idx],
        )?;
        context.events.emit(
            "heldout.completed",
            "Heldout evaluation completed",
            json!({
                "candidate_id": state.candidates[candidate_idx].candidate_id,
                "train_reward": state.candidates[candidate_idx].train_reward,
                "heldout_reward": eval.average_reward,
            }),
        )?;
        state.heldout_candidate_index = state.heldout_candidate_index.max(
            candidate_active
                .heldout_candidate_index
                .unwrap_or(0)
                .saturating_add(1),
        );
    }
    state.active_evaluation = None;
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::Heldout,
        "completed",
        "heldout candidate group completed",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "heldout".to_string(),
        },
        terminal: false,
        result: None,
        message: "heldout candidate group completed".to_string(),
    })
}

fn move_to_finalizing(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    reason: &str,
) -> Result<GepaAdvanceOutcome> {
    if context.state_machine.state() == OptimizerRunState::RolloutRunning {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Evaluating,
            OptimizerTransitionTrigger::RolloutsFinished,
            "Heldout rollouts finished",
            json!({"stage": "heldout"}),
        )?;
    }
    state.active_evaluation = None;
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::Finalizing,
        "completed",
        reason,
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: "finalizing".to_string(),
        },
        terminal: false,
        result: None,
        message: reason.to_string(),
    })
}

fn finalize_completed_gepa_run(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
) -> Result<GepaAdvanceOutcome> {
    let best_idx = state.best_idx.unwrap_or(0);
    let heldout_best_reward = state.candidates[best_idx]
        .heldout_reward
        .or(state.candidates[best_idx].train_reward)
        .unwrap_or(0.0);
    let heldout_skipped = !state
        .candidates
        .iter()
        .any(|candidate| candidate.heldout_reward.is_some());
    let mut stopper_metadata = Map::new();
    stopper_metadata.insert("stage".to_string(), Value::String("heldout".to_string()));
    stopper_metadata.insert("heldout_reward".to_string(), json!(heldout_best_reward));
    push_stopper_snapshot(
        &mut state.stopper_states,
        &mut state.stopper_sequence,
        &context.config,
        StopperSnapshot {
            status: if heldout_skipped {
                "completed_limit_reached"
            } else {
                "completed"
            },
            reason: Some(if heldout_skipped {
                "heldout skipped due to limits"
            } else {
                "heldout evaluation completed"
            }),
            generation: None,
            candidate_id: Some(&state.candidates[best_idx].candidate_id),
            evaluation_stage: Some("heldout"),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            metadata: stopper_metadata,
        },
    );
    let score_chart = score_chart_value(
        &state.candidates,
        0,
        best_idx,
        &context.paths.score_chart_path,
    );
    context.paths.write_text(
        &context.paths.score_chart_path,
        &render_score_chart_svg(&context.config.run.run_id, &score_chart),
    )?;
    context
        .events
        .emit("score_chart.written", "Score chart written", score_chart)?;
    let frontier = frontier_members(&state.candidates);
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &context.config,
        candidates: &state.candidates,
        frontier: frontier.clone(),
        best_idx: Some(best_idx),
        state_machine: &context.state_machine,
        rollout_count: state.rollout_count,
        total_usage: &state.total_usage,
        total_cost: state.total_cost,
    });
    let mut metadata = Map::new();
    metadata.insert("stage".to_string(), Value::String("heldout".to_string()));
    metadata.insert("heldout_reward".to_string(), json!(heldout_best_reward));
    metadata.insert("heldout_skipped".to_string(), json!(heldout_skipped));
    record_checkpoint_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        &mut state.checkpoint_sequence,
        &context.state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "terminal",
            status: "completed",
            reason: Some(if heldout_skipped {
                "heldout skipped due to limits"
            } else {
                "heldout evaluation completed"
            }),
            generation: None,
            candidate_id: Some(&state.candidates[best_idx].candidate_id),
            evaluation_stage: Some("heldout"),
            best_candidate_id: Some(&state.candidates[best_idx].candidate_id),
            candidate_count: state.candidates.len(),
            frontier_count: frontier.len(),
            rollout_count: state.rollout_count,
            cost_usd: state.total_cost,
            usage: serde_json::to_value(&state.total_usage)?,
            snapshot,
            metadata,
        },
    )?;
    if !context.state_machine.state().is_terminal() {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            OptimizerRunState::Completed,
            OptimizerTransitionTrigger::RunCompleted,
            "GEPA run completed",
            json!({
                "best_candidate_id": state.candidates[best_idx].candidate_id,
                "heldout_reward": heldout_best_reward,
                "heldout_skipped": heldout_skipped,
            }),
        )?;
    }
    let best_candidate = serde_json::to_value(&state.candidates[best_idx])?;
    let candidate_registry = serde_json::to_value(&state.candidates)?;
    let frontier_value = serde_json::to_value(frontier_members(&state.candidates))?;
    let cache_profile_record = CacheProfileRecord::from_profile(context.cache.profile()?);
    let cache_access_log = context.cache.access_log().to_vec();
    let cache_profile = serde_json::to_value(&cache_profile_record.profile)?;
    let usage_value = serde_json::to_value(&state.total_usage)?;
    let state_history = serde_json::to_value(&context.state_machine.history)?;
    let candidate_values = candidate_registry.as_array().cloned().unwrap_or_default();
    context
        .workspace
        .persist_candidate_registry(&context.config.run.run_id, &candidate_values)?;
    context
        .workspace
        .persist_state_history(&context.state_machine.history)?;
    context
        .paths
        .write_json(&context.paths.best_candidate_path, &best_candidate)?;
    context
        .paths
        .write_json(&context.paths.candidate_registry_path, &candidate_registry)?;
    context
        .paths
        .write_json(&context.paths.frontier_path, &frontier_value)?;
    context
        .paths
        .write_json(&context.paths.cache_profile_path, &cache_profile)?;
    let sensor_frame_count = state
        .candidates
        .iter()
        .map(|candidate| candidate.sensor_frames.len())
        .sum::<usize>();
    context.events.emit(
        "workspace.persisted",
        "SQLite workspace persisted",
        json!({
            "workspace_db_path": context.paths.workspace_db_path,
            "candidate_count": state.candidates.len(),
            "sensor_frame_count": sensor_frame_count,
            "state_transition_count": context.state_machine.history.len(),
        }),
    )?;
    let runtime_summary =
        serde_json::to_value(runtime_usage_summary_from_events(context.events.records()))?;
    context.events.emit(
        "gepa.run.finished",
        "GEPA run finished",
        json!({
            "best_candidate_id": state.candidates[best_idx].candidate_id,
            "cost_usd": state.total_cost,
            "heldout_reward": heldout_best_reward,
            "heldout_skipped": heldout_skipped,
            "rollout_count": state.rollout_count,
            "runtime_summary": runtime_summary,
            "usage": usage_value,
            "state": context.state_machine.state().as_str(),
        }),
    )?;
    context.events.flush()?;
    context
        .workspace
        .record_event_stream(&context.config.run.run_id, context.events.records())?;
    normalize_event_feed(
        &context.paths.event_feed_path,
        &context.paths.normalized_event_feed_path,
        &context.paths.run_dir,
    )?;
    context.registry.append(&RunRegistryEntry::finished(
        &context.paths,
        &context.config,
        context.cache_mode,
        &context.cache_namespace,
        state.candidates[best_idx].candidate_id.clone(),
        state.total_cost,
        usage_value.clone(),
    ))?;
    let artifact_refs = vec![
        context.paths.artifact_ref(
            &context.paths.best_candidate_path,
            "best_candidate",
            "release_evidence",
        )?,
        context.paths.artifact_ref(
            &context.paths.candidate_registry_path,
            "candidate_registry",
            "release_evidence",
        )?,
        context
            .paths
            .artifact_ref(&context.paths.frontier_path, "frontier", "release_evidence")?,
        context.paths.artifact_ref(
            &context.paths.score_chart_path,
            "score_chart_svg",
            "release_evidence",
        )?,
        context.paths.artifact_ref(
            &context.paths.event_feed_path,
            "events_jsonl",
            "release_evidence",
        )?,
        context.paths.artifact_ref(
            &context.paths.normalized_event_feed_path,
            "events_normalized_jsonl",
            "release_evidence",
        )?,
        context.paths.artifact_ref(
            &context.paths.cache_profile_path,
            "cache_profile",
            "release_evidence",
        )?,
        context.paths.artifact_ref(
            &context.paths.run_registry_path,
            "run_registry_jsonl",
            "release_evidence",
        )?,
    ];
    let result = GepaRunResult {
        best_candidate,
        manifest_path: context.paths.manifest_path.display().to_string(),
        event_feed_path: context.paths.event_feed_path.display().to_string(),
        normalized_event_feed_path: context
            .paths
            .normalized_event_feed_path
            .display()
            .to_string(),
        cache_profile_path: context.paths.cache_profile_path.display().to_string(),
        candidate_registry_path: context.paths.candidate_registry_path.display().to_string(),
        frontier_path: context.paths.frontier_path.display().to_string(),
        score_chart_path: context.paths.score_chart_path.display().to_string(),
        run_registry_path: context.paths.run_registry_path.display().to_string(),
        workspace_db_path: context.paths.workspace_db_path.display().to_string(),
        artifact_refs,
        cost_usd: state.total_cost,
        usage: usage_value,
        state_history,
    };
    let mut result_value = serde_json::to_value(&result)?;
    if let Some(result_object) = result_value.as_object_mut() {
        result_object.insert(
            "stopped_by".to_string(),
            stopped_by_value(&context.config, state),
        );
    }
    context
        .workspace
        .record_artifact_refs(&context.config.run.run_id, &result.artifact_refs)?;
    context.workspace.record_cache_profile(
        &context.config.run.run_id,
        &cache_profile_record,
        &cache_access_log,
    )?;
    context
        .workspace
        .record_usage_ledger(&context.config.run.run_id, &state.usage_ledger)?;
    context
        .workspace
        .record_stopper_states(&context.config.run.run_id, &state.stopper_states)?;
    context.workspace.record_manifest(
        &context.config.run.run_id,
        &context.paths.manifest_path,
        &state.candidates[best_idx].candidate_id,
        state.total_cost,
        &result.usage,
        &result_value,
    )?;
    context.workspace.record_run_finished(
        &context.config.run.run_id,
        &state.candidates[best_idx].candidate_id,
        state.total_cost,
        &result.usage,
    )?;
    context
        .paths
        .write_json(&context.paths.manifest_path, &result_value)?;
    state.cursor.terminal_summary = Some(result_value);
    persist_gepa_run_state(
        context,
        state,
        resources,
        GepaCursorPhase::Completed,
        "completed",
        "GEPA run completed",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::TerminalizeRun {
            run_id: context.config.run.run_id.clone(),
            status: "completed".to_string(),
        },
        terminal: true,
        result: Some(result),
        message: "GEPA run completed".to_string(),
    })
}

fn stopped_by_value(config: &SynthOptimizerConfig, state: &GepaRunState) -> Value {
    if score_threshold_reached(config, state) {
        return json!({
            "kind": "score_threshold",
            "metric": config.gepa.score_threshold_metric.as_deref().unwrap_or("heldout_score"),
            "value": config.gepa.score_threshold_value,
        });
    }
    if no_improvement_reached(config, state) {
        return json!({
            "kind": "no_improvement",
            "metric": config.gepa.no_improvement_metric.as_deref().unwrap_or("heldout_score"),
            "generations": config.gepa.no_improvement_generations,
        });
    }
    if cost_budget_reached(config, state.total_cost) {
        return json!({"kind": "max_cost_usd", "value": config.gepa.max_cost_usd});
    }
    if train_rollout_budget_reached(config, state.rollout_count) {
        return json!({"kind": "max_rollouts", "n": config.gepa.max_total_rollouts});
    }
    json!({"kind": "max_generations", "n": config.gepa.max_generations})
}

pub fn execute_gepa_with_options(
    config: SynthOptimizerConfig,
    options: GepaExecutionOptions,
) -> Result<GepaRunResult> {
    let mut context = open_gepa_run_context(config, &options)?;
    let mut state = restore_gepa_run_state(&mut context)?;
    loop {
        let outcome =
            advance_gepa_once(&mut context, &mut state, GepaAdvanceMode::RunLoop, &options)?;
        if outcome.terminal {
            if let Some(result) = outcome.result {
                return Ok(result);
            }
            return Err(error_from_terminal_gepa_outcome(
                &context,
                &state.cursor,
                &outcome,
            ));
        }
        if matches!(outcome.action, planner::GepaTickAction::Noop) {
            thread::sleep(ASYNC_PIPELINE_NOOP_SLEEP);
        }
    }
}

fn error_from_terminal_gepa_outcome(
    context: &GepaRunContext,
    cursor: &GepaCursor,
    outcome: &GepaAdvanceOutcome,
) -> OptimizerError {
    let run_id = &context.config.run.run_id;
    if matches!(cursor.phase, GepaCursorPhase::Cancelled) {
        return OptimizerError::Cancelled {
            request_id: run_id.to_string(),
        };
    }
    if matches!(cursor.phase, GepaCursorPhase::Failed) {
        let runtime_failure = latest_failed_runtime_effect_message(&context.workspace, run_id);
        return OptimizerError::Failed(terminal_failure_message(
            run_id,
            cursor,
            outcome,
            runtime_failure.as_deref(),
        ));
    }
    OptimizerError::Invariant(format!(
        "GEPA run {run_id} reached terminal state {} without a result",
        cursor.phase.as_str()
    ))
}

fn terminal_failure_message(
    run_id: &str,
    cursor: &GepaCursor,
    outcome: &GepaAdvanceOutcome,
    runtime_failure: Option<&str>,
) -> String {
    let summary = cursor.error_summary.as_ref();
    let failure = summary.and_then(|value| value.get("failure"));
    let failure_message = failure
        .and_then(|value| value.get("message"))
        .or_else(|| summary.and_then(|value| value.get("message")))
        .and_then(Value::as_str)
        .filter(|message| !message.trim().is_empty());
    let error_code = failure
        .and_then(|value| value.get("error_code"))
        .or_else(|| summary.and_then(|value| value.get("error_code")))
        .and_then(Value::as_str)
        .filter(|code| !code.trim().is_empty());
    let manifest_path = summary
        .and_then(|value| value.get("failure_manifest_path"))
        .or_else(|| cursor.metadata.get("failure_manifest_path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty());

    let mut message = format!("GEPA run {run_id} failed");
    if let Some(error_code) = error_code {
        let _ = write!(message, " ({error_code})");
    }
    if let Some(failure_message) = failure_message {
        let _ = write!(message, ": {failure_message}");
    } else if !outcome.message.trim().is_empty() {
        let _ = write!(message, ": {}", outcome.message);
    }
    if let Some(runtime_failure) = runtime_failure {
        if !message.contains(runtime_failure) {
            let _ = write!(message, "; runtime failure: {runtime_failure}");
        }
    }
    if let Some(manifest_path) = manifest_path {
        let _ = write!(message, " [manifest: {manifest_path}]");
    }
    message
}

fn latest_failed_runtime_effect_message(
    workspace: &WorkspaceStore,
    run_id: &str,
) -> Option<String> {
    workspace
        .view()
        .runtime_effect_records(run_id)
        .ok()?
        .into_iter()
        .filter(|effect| effect.status == "failed")
        .max_by(|left, right| left.updated_at.cmp(&right.updated_at))
        .and_then(|effect| {
            effect
                .metadata
                .get("failure")
                .and_then(|failure| failure.get("message"))
                .and_then(Value::as_str)
                .filter(|message| !message.trim().is_empty())
                .map(str::to_string)
                .or_else(|| {
                    effect
                        .metadata
                        .get("message")
                        .and_then(Value::as_str)
                        .filter(|message| !message.trim().is_empty())
                        .map(str::to_string)
                })
        })
}

#[allow(dead_code)]
fn execute_gepa_monolithic_with_options(
    config: SynthOptimizerConfig,
    options: GepaExecutionOptions,
) -> Result<GepaRunResult> {
    let mut context = open_gepa_run_context(config, &options)?;
    let restored_cursor =
        initialize_or_restore_cursor(&context.workspace, &context.config.run.run_id)?;
    if matches!(restored_cursor.phase, GepaCursorPhase::Completed) {
        if let Some(summary) = restored_cursor.terminal_summary.clone() {
            return serde_json::from_value(summary).map_err(OptimizerError::from);
        }
    }
    let container_inputs = ensure_container_inputs(&mut context)?;
    let GepaRunContext {
        paths,
        mut workspace,
        registry,
        mut events,
        mut state_machine,
        mut cache,
        config,
        cache_mode,
        cache_namespace,
        ..
    } = context;
    let GepaContainerInputs {
        _container_process,
        client,
        mut program,
        mut objective_set,
        mut train_rows,
        minibatch_rows: mut minibatch_pool_rows,
        mut reflection_rows,
        mut heldout_rows,
        mut rollout_task_id,
    } = container_inputs;
    if !restored_cursor.program.is_null() {
        program = serde_json::from_value(restored_cursor.program.clone())?;
    }
    if !restored_cursor.objective_set.is_null() {
        objective_set = serde_json::from_value(restored_cursor.objective_set.clone())?;
    }
    if let Some(restored_task_id) = restored_cursor.rollout_task_id.clone() {
        rollout_task_id = restored_task_id;
    }
    if restored_cursor
        .train_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        train_rows = serde_json::from_value(restored_cursor.train_rows.clone())?;
    }
    if restored_cursor
        .minibatch_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        minibatch_pool_rows = serde_json::from_value(restored_cursor.minibatch_rows.clone())?;
    }
    if restored_cursor
        .reflection_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        reflection_rows = serde_json::from_value(restored_cursor.reflection_rows.clone())?;
    }
    if restored_cursor
        .heldout_rows
        .as_array()
        .is_some_and(|rows| !rows.is_empty())
    {
        heldout_rows = serde_json::from_value(restored_cursor.heldout_rows.clone())?;
    }
    check_cancelled(options.cancellation.as_ref())?;

    let mut candidates: Vec<CandidateRecord> = restored_cursor
        .candidates
        .as_array()
        .filter(|rows| !rows.is_empty())
        .map(|_| serde_json::from_value(restored_cursor.candidates.clone()))
        .transpose()?
        .unwrap_or_default();
    let seed_restored = !candidates.is_empty();
    if candidates.is_empty() {
        let seed_payload = seed_candidate_payload(&config, &program)?;
        let seed_id = candidate_id(&seed_payload);
        let seed_bundle = LeverBundle::from_prompt_payload(seed_id.clone(), None, &seed_payload);
        candidates.push(CandidateRecord {
            candidate_id: seed_id.clone(),
            payload: seed_payload,
            lever_bundle: seed_bundle,
            parent_id: None,
            source: "seed".to_string(),
            status: "registered".to_string(),
            minibatch_reward: None,
            train_reward: None,
            heldout_reward: None,
            minibatch_scores: Vec::new(),
            train_scores: Vec::new(),
            sensor_frames: Vec::new(),
            acceptance_score: Value::Null,
            acceptance_metadata: Map::new(),
        });
        events.emit(
            "candidate.registered",
            "Seed candidate registered",
            json!({"candidate_id": candidates[0].candidate_id, "source": "seed"}),
        )?;
        persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidates[0])?;
    }

    let mut total_usage: UsageTotals = if restored_cursor.usage.is_null() {
        UsageTotals::default()
    } else {
        serde_json::from_value(restored_cursor.usage.clone())?
    };
    let mut total_cost = restored_cursor.cost_usd;
    let mut rollout_count = restored_cursor.rollout_count;
    let mut usage_ledger = Vec::new();
    let mut stopper_states = Vec::new();
    let mut stopper_sequence = restored_cursor.stopper_sequence;
    let mut checkpoint_sequence = restored_cursor.checkpoint_sequence;
    if !seed_restored {
        let mut metadata = Map::new();
        metadata.insert("stage".to_string(), Value::String("run_start".to_string()));
        metadata.insert(
            "max_generations".to_string(),
            json!(config.gepa.max_generations),
        );
        metadata.insert(
            "proposals_per_generation".to_string(),
            json!(config.gepa.proposals_per_generation),
        );
        push_stopper_snapshot(
            &mut stopper_states,
            &mut stopper_sequence,
            &config,
            StopperSnapshot {
                status: "within_budget",
                reason: Some("run initialized within budget"),
                generation: None,
                candidate_id: None,
                evaluation_stage: Some("run_start"),
                rollout_count,
                cost_usd: total_cost,
                metadata,
            },
        );
    }
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &config,
        candidates: &candidates,
        frontier: Vec::new(),
        best_idx: None,
        state_machine: &state_machine,
        rollout_count,
        total_usage: &total_usage,
        total_cost,
    });
    let mut metadata = Map::new();
    metadata.insert(
        "stage".to_string(),
        Value::String("seed_registered".to_string()),
    );
    record_checkpoint_snapshot(
        &mut workspace,
        &config.run.run_id,
        &mut checkpoint_sequence,
        &state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "candidate_registry",
            status: "completed",
            reason: Some("seed candidate registered"),
            generation: None,
            candidate_id: Some(&candidates[0].candidate_id),
            evaluation_stage: Some("seed_registered"),
            best_candidate_id: None,
            candidate_count: candidates.len(),
            frontier_count: 0,
            rollout_count,
            cost_usd: total_cost,
            usage: serde_json::to_value(&total_usage)?,
            snapshot,
            metadata,
        },
    )?;
    persist_gepa_cursor(
        &mut workspace,
        &config,
        &mut checkpoint_sequence,
        GepaCursorState {
            phase: GepaCursorPhase::SeedFullTrain,
            generation: 0,
            proposal_index: 0,
            pending_job_id: None,
            pending_effect_id: None,
            pending_reservation_ids: Vec::new(),
            active_evaluation: None,
            candidates: &candidates,
            best_idx: None,
            train_rows: &train_rows,
            minibatch_rows: &minibatch_pool_rows,
            reflection_rows: &reflection_rows,
            heldout_rows: &heldout_rows,
            program: &program,
            objective_set: &objective_set,
            rollout_task_id: &rollout_task_id,
            total_usage: &total_usage,
            total_cost,
            rollout_count,
            stopper_sequence,
            state_machine: &state_machine,
            terminal_summary: None,
            error_summary: None,
            metadata: Map::new(),
        },
        "completed",
        "seed candidate registered",
    )?;

    let mut best_idx = restored_cursor
        .best_candidate_id
        .as_ref()
        .and_then(|candidate_id| {
            candidates
                .iter()
                .position(|candidate| &candidate.candidate_id == candidate_id)
        })
        .unwrap_or(0);
    if candidates[0].train_reward.is_none() {
        let seed_rollout_capacity =
            remaining_train_rollout_capacity(&workspace, &config, rollout_count)?;
        if seed_rollout_capacity < train_rows.len() {
            let error = rollout_budget_exceeded_error(
                &config.run.run_id,
                rollout_budget_limit_name(&config),
                train_rows.len(),
                seed_rollout_capacity,
            );
            return fail_gepa_run_and_return(
                FailedGepaRunInput {
                    workspace: &mut workspace,
                    events: &mut events,
                    state_machine: &mut state_machine,
                    paths: &paths,
                    registry: &registry,
                    cache: &mut cache,
                    config: &config,
                    cache_mode,
                    cache_namespace: &cache_namespace,
                    best_candidate_id: Some(&candidates[0].candidate_id),
                    total_cost,
                    total_usage: &total_usage,
                    usage_ledger: &usage_ledger,
                    stopper_states: &stopper_states,
                    message: "Seed candidate cannot be fully evaluated within rollout limits",
                    details: json!({
                        "candidate_id": candidates[0].candidate_id,
                        "stage": "seed_full_train",
                        "required_rollouts": train_rows.len(),
                        "available_rollouts": seed_rollout_capacity,
                    }),
                },
                error,
            );
        }
        if let Some(breach) = next_rollout_budget_breach(&workspace, &config)? {
            let error = budget_exceeded_error(&config.run.run_id, &breach);
            return fail_gepa_run_and_return(
                FailedGepaRunInput {
                    workspace: &mut workspace,
                    events: &mut events,
                    state_machine: &mut state_machine,
                    paths: &paths,
                    registry: &registry,
                    cache: &mut cache,
                    config: &config,
                    cache_mode,
                    cache_namespace: &cache_namespace,
                    best_candidate_id: Some(&candidates[0].candidate_id),
                    total_cost,
                    total_usage: &total_usage,
                    usage_ledger: &usage_ledger,
                    stopper_states: &stopper_states,
                    message: "Seed candidate cannot reserve rollout budget",
                    details: json!({
                        "candidate_id": candidates[0].candidate_id,
                        "stage": "seed_full_train",
                        "limit": breach.limit,
                        "requested": breach.requested,
                        "available": breach.available,
                    }),
                },
                error,
            );
        }

        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::RolloutsQueued,
            "Seed candidate rollouts queued",
            json!({"candidate_id": candidates[0].candidate_id, "stage": "seed_full_train"}),
        )?;
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::RolloutRunning,
            OptimizerTransitionTrigger::RolloutsStarted,
            "Seed candidate rollouts started",
            json!({"candidate_id": candidates[0].candidate_id, "stage": "seed_full_train"}),
        )?;
        let seed_eval = match evaluate_candidate(EvaluationCall {
            client: &client,
            workspace: &workspace,
            paths: &paths,
            cache: &mut cache,
            cache_namespace: &cache_namespace,
            config: &config,
            program: &program,
            task_id: &rollout_task_id,
            objective_set: &objective_set,
            candidate: &candidates[0],
            rows: &train_rows,
            stage: "seed_full_train",
            cancellation: options.cancellation.as_ref(),
        }) {
            Ok(eval) => eval,
            Err(error) => {
                return fail_gepa_run_and_return(
                    FailedGepaRunInput {
                        workspace: &mut workspace,
                        events: &mut events,
                        state_machine: &mut state_machine,
                        paths: &paths,
                        registry: &registry,
                        cache: &mut cache,
                        config: &config,
                        cache_mode,
                        cache_namespace: &cache_namespace,
                        best_candidate_id: Some(&candidates[0].candidate_id),
                        total_cost,
                        total_usage: &total_usage,
                        usage_ledger: &usage_ledger,
                        stopper_states: &stopper_states,
                        message: "Seed candidate rollout failed",
                        details: json!({
                            "candidate_id": candidates[0].candidate_id,
                            "stage": "seed_full_train",
                        }),
                    },
                    error,
                );
            }
        };
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::Evaluating,
            OptimizerTransitionTrigger::RolloutsFinished,
            "Seed candidate rollouts finished",
            json!({"candidate_id": candidates[0].candidate_id, "stage": "seed_full_train"}),
        )?;
        candidates[0].status = "full_train_evaluated".to_string();
        candidates[0].train_reward = Some(seed_eval.average_reward);
        candidates[0].train_scores = seed_eval.scores.clone();
        candidates[0]
            .sensor_frames
            .extend(seed_eval.sensor_frames.clone());
        persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidates[0])?;
        total_usage.merge(&seed_eval.usage);
        total_cost += seed_eval.cost_usd;
        rollout_count += seed_eval.rollout_count;
        append_rollout_usage(&mut usage_ledger, &seed_eval);
        let mut metadata = Map::new();
        metadata.insert(
            "stage".to_string(),
            Value::String("seed_full_train".to_string()),
        );
        metadata.insert("rollout_delta".to_string(), json!(seed_eval.rollout_count));
        metadata.insert(
            "average_reward".to_string(),
            json!(seed_eval.average_reward),
        );
        push_stopper_snapshot(
            &mut stopper_states,
            &mut stopper_sequence,
            &config,
            StopperSnapshot {
                status: budget_status(&config, rollout_count, total_cost),
                reason: Some("seed full-train evaluation completed"),
                generation: None,
                candidate_id: Some(&candidates[0].candidate_id),
                evaluation_stage: Some("seed_full_train"),
                rollout_count,
                cost_usd: total_cost,
                metadata,
            },
        );
        best_idx = 0usize;
        let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
            config: &config,
            candidates: &candidates,
            frontier: frontier_members(&candidates),
            best_idx: Some(best_idx),
            state_machine: &state_machine,
            rollout_count,
            total_usage: &total_usage,
            total_cost,
        });
        let mut metadata = Map::new();
        metadata.insert(
            "stage".to_string(),
            Value::String("seed_full_train".to_string()),
        );
        record_checkpoint_snapshot(
            &mut workspace,
            &config.run.run_id,
            &mut checkpoint_sequence,
            &state_machine,
            CheckpointSnapshot {
                checkpoint_kind: "evaluation_boundary",
                status: "completed",
                reason: Some("seed full-train evaluation completed"),
                generation: None,
                candidate_id: Some(&candidates[0].candidate_id),
                evaluation_stage: Some("seed_full_train"),
                best_candidate_id: Some(&candidates[best_idx].candidate_id),
                candidate_count: candidates.len(),
                frontier_count: frontier_members(&candidates).len(),
                rollout_count,
                cost_usd: total_cost,
                usage: serde_json::to_value(&total_usage)?,
                snapshot,
                metadata,
            },
        )?;
        events.emit(
        "candidate.evaluated",
        "Seed candidate evaluated",
        json!({"candidate_id": candidates[0].candidate_id, "train_reward": seed_eval.average_reward}),
    )?;
        events.emit(
            "frontier.updated",
            "Frontier updated",
            frontier_snapshot_value(
                &candidates,
                &train_rows,
                Some(best_idx),
                None,
                "seed_full_train",
                Some(&candidates[0].candidate_id),
                Some(BTreeSet::new()),
            )?,
        )?;
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::Ready,
            OptimizerTransitionTrigger::EvaluationFinished,
            "Seed candidate evaluation finished",
            json!({"candidate_id": candidates[0].candidate_id}),
        )?;
    }
    persist_gepa_cursor(
        &mut workspace,
        &config,
        &mut checkpoint_sequence,
        GepaCursorState {
            phase: GepaCursorPhase::GenerationStart,
            generation: restored_cursor.generation,
            proposal_index: restored_cursor.proposal_index,
            pending_job_id: None,
            pending_effect_id: None,
            pending_reservation_ids: Vec::new(),
            active_evaluation: None,
            candidates: &candidates,
            best_idx: Some(best_idx),
            train_rows: &train_rows,
            minibatch_rows: &minibatch_pool_rows,
            reflection_rows: &reflection_rows,
            heldout_rows: &heldout_rows,
            program: &program,
            objective_set: &objective_set,
            rollout_task_id: &rollout_task_id,
            total_usage: &total_usage,
            total_cost,
            rollout_count,
            stopper_sequence,
            state_machine: &state_machine,
            terminal_summary: None,
            error_summary: None,
            metadata: Map::new(),
        },
        "completed",
        "seed full-train evaluation completed",
    )?;

    let generation_start = if matches!(
        restored_cursor.phase,
        GepaCursorPhase::GenerationStart
            | GepaCursorPhase::ProposerWaiting
            | GepaCursorPhase::CandidateMinibatch
            | GepaCursorPhase::CandidateFullTrain
            | GepaCursorPhase::Heldout
            | GepaCursorPhase::Finalizing
            | GepaCursorPhase::Completed
    ) {
        restored_cursor.generation
    } else {
        0
    };

    for generation in generation_start..config.gepa.max_generations {
        check_cancelled(options.cancellation.as_ref())?;
        if train_rollout_budget_reached(&config, rollout_count) {
            let mut metadata = Map::new();
            metadata.insert(
                "stage".to_string(),
                Value::String("generation_start".to_string()),
            );
            metadata.insert("generation".to_string(), json!(generation));
            push_stopper_snapshot(
                &mut stopper_states,
                &mut stopper_sequence,
                &config,
                StopperSnapshot {
                    status: "rollout_budget_reached",
                    reason: Some("rollout budget reached before generation"),
                    generation: Some(generation),
                    candidate_id: Some(&candidates[best_idx].candidate_id),
                    evaluation_stage: Some("generation_start"),
                    rollout_count,
                    cost_usd: total_cost,
                    metadata,
                },
            );
            events.emit(
                "gepa.stop",
                "Rollout budget reached",
                json!({"rollout_count": rollout_count}),
            )?;
            break;
        }
        if cost_budget_reached(&config, total_cost) {
            let mut metadata = Map::new();
            metadata.insert(
                "stage".to_string(),
                Value::String("generation_start".to_string()),
            );
            metadata.insert("generation".to_string(), json!(generation));
            push_stopper_snapshot(
                &mut stopper_states,
                &mut stopper_sequence,
                &config,
                StopperSnapshot {
                    status: "cost_budget_reached",
                    reason: Some("cost budget reached before generation"),
                    generation: Some(generation),
                    candidate_id: Some(&candidates[best_idx].candidate_id),
                    evaluation_stage: Some("generation_start"),
                    rollout_count,
                    cost_usd: total_cost,
                    metadata,
                },
            );
            events.emit(
                "gepa.stop",
                "Cost budget reached",
                json!({"cost_usd": total_cost, "max_cost_usd": config.gepa.max_cost_usd}),
            )?;
            break;
        }
        if let Some(train_best_idx) = select_best_train_candidate(
            &candidates,
            &objective_set,
            &config.dataset.train_split,
            &train_rows,
        )? {
            best_idx = train_best_idx;
        }
        let parent_selection = select_proposer_parent_candidate(
            &candidates,
            &train_rows,
            &objective_set,
            &config.gepa.candidate_selector,
            generation,
            &config.run.run_id,
            Some(best_idx),
        )?;
        let parent = candidates[parent_selection.candidate_index].clone();
        let proposer_rollout_row_count = candidates
            .iter()
            .map(|candidate| {
                candidate.minibatch_scores.len()
                    + candidate.train_scores.len()
                    + candidate.sensor_frames.len()
            })
            .sum::<usize>();
        let proposer_loss_count = candidates
            .iter()
            .flat_map(|candidate| candidate.sensor_frames.iter())
            .filter(|frame| frame.reward < 1.0)
            .count();
        let proposer_win_count = candidates
            .iter()
            .flat_map(|candidate| candidate.sensor_frames.iter())
            .filter(|frame| frame.reward >= 1.0)
            .count();
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::Proposing,
            OptimizerTransitionTrigger::ProposerStarted,
            "Proposer started",
            json!({
                "generation": generation,
                "backend": config.proposer.backend,
                "model": config.proposer.model,
                "proposal_count": config.gepa.proposals_per_generation,
                "parent_candidate_id": parent.candidate_id,
                "frontier_size": frontier_members(&candidates).len(),
                "candidate_count": candidates.len(),
                "rollout_row_count": proposer_rollout_row_count,
                "loss_count": proposer_loss_count,
                "win_count": proposer_win_count,
                "workspace": paths.run_dir
                    .join("proposer_workspaces")
                    .join(format!("generation_{:03}", generation))
                    .display()
                    .to_string(),
                "parent_selection": parent_selection.metadata,
            }),
        )?;
        check_cancelled(options.cancellation.as_ref())?;
        let proposer_outcome = match propose_candidates(ProposerCall {
            client: &client,
            workspace: &workspace,
            cache: &mut cache,
            cache_namespace: &cache_namespace,
            config: &config,
            program: &program,
            parent: &parent,
            candidates: &candidates,
            generation,
            seed_pool_rows: seed_pool_rows_value(
                &train_rows,
                &minibatch_pool_rows,
                &reflection_rows,
                &heldout_rows,
            ),
            paths: &paths,
        }) {
            Ok(outcome) => outcome,
            Err(error) => {
                return fail_gepa_run_and_return(
                    FailedGepaRunInput {
                        workspace: &mut workspace,
                        events: &mut events,
                        state_machine: &mut state_machine,
                        paths: &paths,
                        registry: &registry,
                        cache: &mut cache,
                        config: &config,
                        cache_mode,
                        cache_namespace: &cache_namespace,
                        best_candidate_id: Some(&parent.candidate_id),
                        total_cost,
                        total_usage: &total_usage,
                        usage_ledger: &usage_ledger,
                        stopper_states: &stopper_states,
                        message: "Candidate proposer failed",
                        details: json!({
                            "generation": generation,
                            "parent_candidate_id": parent.candidate_id,
                            "stage": "proposer",
                        }),
                    },
                    error,
                );
            }
        };
        total_usage.merge(&proposer_outcome.usage);
        total_cost += proposer_outcome.cost_usd;
        usage_ledger.push(proposer_usage_record(
            &config,
            &parent,
            generation,
            &proposer_outcome,
        )?);
        let mut metadata = Map::new();
        metadata.insert("stage".to_string(), Value::String("proposer".to_string()));
        metadata.insert("generation".to_string(), json!(generation));
        metadata.insert(
            "proposal_count".to_string(),
            json!(proposer_outcome.proposals.len()),
        );
        metadata.insert(
            "backend".to_string(),
            Value::String(proposer_outcome.backend.clone()),
        );
        metadata.insert(
            "warning_count".to_string(),
            json!(proposer_outcome.evidence_warnings.len()),
        );
        push_stopper_snapshot(
            &mut stopper_states,
            &mut stopper_sequence,
            &config,
            StopperSnapshot {
                status: budget_status(&config, rollout_count, total_cost),
                reason: Some("proposer completed"),
                generation: Some(generation),
                candidate_id: Some(&parent.candidate_id),
                evaluation_stage: Some("proposer"),
                rollout_count,
                cost_usd: total_cost,
                metadata,
            },
        );
        events.emit(
            "proposer.completed",
            "Proposer returned candidates",
            json!({
                "generation": generation,
                "proposal_count": proposer_outcome.proposals.len(),
                "backend": proposer_outcome.backend,
                "workspace": proposer_outcome.workspace,
                "warning_count": proposer_outcome.evidence_warnings.len(),
                "warnings": proposer_outcome.evidence_warnings,
            }),
        )?;
        if proposer_outcome.proposals.is_empty() {
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::Ready,
                OptimizerTransitionTrigger::ProposerFinished,
                "Proposer returned no candidates",
                json!({"generation": generation}),
            )?;
            continue;
        }
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::ProposerFinished,
            "Proposer returned candidates; rollout queue ready",
            json!({
                "generation": generation,
                "proposal_count": proposer_outcome.proposals.len(),
            }),
        )?;

        for (proposal_index, proposal) in proposer_outcome.proposals.into_iter().enumerate() {
            check_cancelled(options.cancellation.as_ref())?;
            if train_rollout_budget_reached(&config, rollout_count) {
                let mut metadata = Map::new();
                metadata.insert(
                    "stage".to_string(),
                    Value::String("candidate_loop".to_string()),
                );
                metadata.insert("generation".to_string(), json!(generation));
                push_stopper_snapshot(
                    &mut stopper_states,
                    &mut stopper_sequence,
                    &config,
                    StopperSnapshot {
                        status: "rollout_budget_reached",
                        reason: Some("rollout budget reached before candidate evaluation"),
                        generation: Some(generation),
                        candidate_id: Some(&parent.candidate_id),
                        evaluation_stage: Some("candidate_loop"),
                        rollout_count,
                        cost_usd: total_cost,
                        metadata,
                    },
                );
                break;
            }
            if cost_budget_reached(&config, total_cost) {
                let mut metadata = Map::new();
                metadata.insert(
                    "stage".to_string(),
                    Value::String("candidate_loop".to_string()),
                );
                metadata.insert("generation".to_string(), json!(generation));
                push_stopper_snapshot(
                    &mut stopper_states,
                    &mut stopper_sequence,
                    &config,
                    StopperSnapshot {
                        status: "cost_budget_reached",
                        reason: Some("cost budget reached before candidate evaluation"),
                        generation: Some(generation),
                        candidate_id: Some(&parent.candidate_id),
                        evaluation_stage: Some("candidate_loop"),
                        rollout_count,
                        cost_usd: total_cost,
                        metadata,
                    },
                );
                break;
            }
            let minibatch_rows = minibatch_rows(
                &minibatch_pool_rows,
                &config.gepa.batch_sampler,
                config.gepa.minibatch_size,
                generation,
                proposal_index,
                config.gepa.proposals_per_generation,
            );
            let proposal_parent_idx = proposal
                .parent_candidate_ids
                .iter()
                .find_map(|candidate_id| {
                    candidates
                        .iter()
                        .position(|candidate| &candidate.candidate_id == candidate_id)
                })
                .unwrap_or(parent_selection.candidate_index);
            let proposal_parent =
                candidates
                    .get(proposal_parent_idx)
                    .cloned()
                    .ok_or_else(|| {
                        OptimizerError::Invariant(format!(
                        "proposal parent index {proposal_parent_idx} is outside candidate registry"
                    ))
                    })?;
            let mut proposal_parent = proposal_parent;
            if parent_minibatch_reward_for_rows(
                &proposal_parent,
                &minibatch_rows,
                &config.dataset.train_split,
            )?
            .is_none()
            {
                let parent_reference_capacity =
                    remaining_train_rollout_capacity(&workspace, &config, rollout_count)?;
                if parent_reference_capacity < minibatch_rows.len() {
                    let mut metadata = Map::new();
                    metadata.insert(
                        "stage".to_string(),
                        Value::String("parent_minibatch_reference".to_string()),
                    );
                    metadata.insert("generation".to_string(), json!(generation));
                    metadata.insert(
                        "remaining_rollouts".to_string(),
                        json!(parent_reference_capacity),
                    );
                    metadata.insert("required_rollouts".to_string(), json!(minibatch_rows.len()));
                    push_stopper_snapshot(
                        &mut stopper_states,
                        &mut stopper_sequence,
                        &config,
                        StopperSnapshot {
                            status: "deferred_budget",
                            reason: Some(
                                "insufficient rollout budget for parent minibatch reference",
                            ),
                            generation: Some(generation),
                            candidate_id: Some(&proposal_parent.candidate_id),
                            evaluation_stage: Some("parent_minibatch_reference"),
                            rollout_count,
                            cost_usd: total_cost,
                            metadata,
                        },
                    );
                    break;
                }
                transition_run(
                    &workspace,
                    &mut events,
                    &mut state_machine,
                    OptimizerRunState::RolloutRunning,
                    OptimizerTransitionTrigger::RolloutsStarted,
                    "Parent minibatch reference rollouts started",
                    json!({
                        "candidate_id": proposal_parent.candidate_id,
                        "generation": generation,
                        "stage": "parent_minibatch_reference",
                        "row_count": minibatch_rows.len(),
                    }),
                )?;
                let parent_reference_eval = evaluate_candidate(EvaluationCall {
                    client: &client,
                    workspace: &workspace,
                    paths: &paths,
                    cache: &mut cache,
                    cache_namespace: &cache_namespace,
                    config: &config,
                    program: &program,
                    task_id: &rollout_task_id,
                    objective_set: &objective_set,
                    candidate: &proposal_parent,
                    rows: &minibatch_rows,
                    stage: "parent_minibatch_reference",
                    cancellation: options.cancellation.as_ref(),
                })?;
                transition_run(
                    &workspace,
                    &mut events,
                    &mut state_machine,
                    OptimizerRunState::Evaluating,
                    OptimizerTransitionTrigger::RolloutsFinished,
                    "Parent minibatch reference rollouts finished",
                    json!({
                        "candidate_id": proposal_parent.candidate_id,
                        "generation": generation,
                        "stage": "parent_minibatch_reference",
                    }),
                )?;
                rollout_count += parent_reference_eval.rollout_count;
                total_usage.merge(&parent_reference_eval.usage);
                total_cost += parent_reference_eval.cost_usd;
                append_rollout_usage(&mut usage_ledger, &parent_reference_eval);
                proposal_parent
                    .sensor_frames
                    .extend(parent_reference_eval.sensor_frames.clone());
                persist_candidate_snapshot(&mut workspace, &config.run.run_id, &proposal_parent)?;
                candidates[proposal_parent_idx] = proposal_parent.clone();
            }
            let parent_minibatch_reward = parent_minibatch_reward_for_rows(
                &proposal_parent,
                &minibatch_rows,
                &config.dataset.train_split,
            )?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "parent candidate {} is missing minibatch reference scores for generation {}",
                    proposal_parent.candidate_id, generation
                ))
            })?;
            let payload = normalize_candidate_payload(
                &program,
                &config,
                &proposal_parent.payload,
                proposal.payload_map(),
            )?;
            let candidate_id = candidate_id(&payload);
            if candidates
                .iter()
                .any(|candidate| candidate.candidate_id == candidate_id)
            {
                events.emit(
                    "candidate.duplicate_skipped",
                    "Duplicate candidate skipped",
                    json!({"candidate_id": candidate_id, "generation": generation}),
                )?;
                continue;
            }
            let proposal_type = proposal.proposal_type_or_default();
            let proposal_parent_id = proposal_parent.candidate_id.clone();
            let mut acceptance_metadata = Map::new();
            acceptance_metadata.insert("proposal".to_string(), proposal.metadata_value());
            acceptance_metadata.insert("generation".to_string(), json!(generation));
            let mut candidate = CandidateRecord {
                lever_bundle: LeverBundle::from_prompt_payload(
                    candidate_id.clone(),
                    Some(proposal_parent_id.clone()),
                    &payload,
                ),
                candidate_id,
                payload,
                parent_id: Some(proposal_parent_id),
                source: format!("reflector:{proposal_type}"),
                status: "registered".to_string(),
                minibatch_reward: None,
                train_reward: None,
                heldout_reward: None,
                minibatch_scores: Vec::new(),
                train_scores: Vec::new(),
                sensor_frames: Vec::new(),
                acceptance_score: Value::Null,
                acceptance_metadata,
            };
            persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
            let minibatch_rollout_capacity =
                remaining_train_rollout_capacity(&workspace, &config, rollout_count)?;
            if minibatch_rollout_capacity < minibatch_rows.len() {
                candidate.status = "deferred_budget".to_string();
                let mut metadata = Map::new();
                metadata.insert(
                    "stage".to_string(),
                    Value::String("candidate_minibatch".to_string()),
                );
                metadata.insert("generation".to_string(), json!(generation));
                metadata.insert(
                    "remaining_rollouts".to_string(),
                    json!(minibatch_rollout_capacity),
                );
                metadata.insert("required_rollouts".to_string(), json!(minibatch_rows.len()));
                push_stopper_snapshot(
                    &mut stopper_states,
                    &mut stopper_sequence,
                    &config,
                    StopperSnapshot {
                        status: "deferred_budget",
                        reason: Some("insufficient rollout budget for minibatch evaluation"),
                        generation: Some(generation),
                        candidate_id: Some(&candidate.candidate_id),
                        evaluation_stage: Some("candidate_minibatch"),
                        rollout_count,
                        cost_usd: total_cost,
                        metadata,
                    },
                );
                events.emit(
                    "candidate.deferred",
                    "Candidate deferred before minibatch",
                    json!({
                        "candidate_id": candidate.candidate_id,
                        "generation": generation,
                        "stage": "candidate_minibatch",
                        "required_rollouts": minibatch_rows.len(),
                        "available_rollouts": minibatch_rollout_capacity,
                    }),
                )?;
                persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
                candidates.push(candidate);
                break;
            }
            if let Some(breach) = next_rollout_budget_breach(&workspace, &config)? {
                candidate.status = "deferred_budget".to_string();
                let mut metadata = Map::new();
                metadata.insert(
                    "stage".to_string(),
                    Value::String("candidate_minibatch".to_string()),
                );
                metadata.insert("generation".to_string(), json!(generation));
                metadata.insert("limit".to_string(), json!(breach.limit.clone()));
                metadata.insert("requested".to_string(), json!(breach.requested.clone()));
                metadata.insert("available".to_string(), json!(breach.available.clone()));
                push_stopper_snapshot(
                    &mut stopper_states,
                    &mut stopper_sequence,
                    &config,
                    StopperSnapshot {
                        status: "deferred_budget",
                        reason: Some("insufficient budget for minibatch evaluation"),
                        generation: Some(generation),
                        candidate_id: Some(&candidate.candidate_id),
                        evaluation_stage: Some("candidate_minibatch"),
                        rollout_count,
                        cost_usd: total_cost,
                        metadata,
                    },
                );
                events.emit(
                    "candidate.deferred",
                    "Candidate deferred before minibatch",
                    json!({
                        "candidate_id": candidate.candidate_id,
                        "generation": generation,
                        "stage": "candidate_minibatch",
                        "limit": breach.limit,
                        "requested": breach.requested,
                        "available": breach.available,
                    }),
                )?;
                persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
                candidates.push(candidate);
                break;
            }
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::RolloutRunning,
                OptimizerTransitionTrigger::RolloutsStarted,
                "Candidate minibatch rollouts started",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "generation": generation,
                    "stage": "candidate_minibatch",
                    "row_count": minibatch_rows.len(),
                }),
            )?;
            let eval = match evaluate_candidate(EvaluationCall {
                client: &client,
                workspace: &workspace,
                paths: &paths,
                cache: &mut cache,
                cache_namespace: &cache_namespace,
                config: &config,
                program: &program,
                task_id: &rollout_task_id,
                objective_set: &objective_set,
                candidate: &candidate,
                rows: &minibatch_rows,
                stage: "candidate_minibatch",
                cancellation: options.cancellation.as_ref(),
            }) {
                Ok(eval) => eval,
                Err(error) => {
                    return fail_gepa_run_and_return(
                        FailedGepaRunInput {
                            workspace: &mut workspace,
                            events: &mut events,
                            state_machine: &mut state_machine,
                            paths: &paths,
                            registry: &registry,
                            cache: &mut cache,
                            config: &config,
                            cache_mode,
                            cache_namespace: &cache_namespace,
                            best_candidate_id: Some(&candidates[best_idx].candidate_id),
                            total_cost,
                            total_usage: &total_usage,
                            usage_ledger: &usage_ledger,
                            stopper_states: &stopper_states,
                            message: "Candidate minibatch rollout failed",
                            details: json!({
                                "candidate_id": candidate.candidate_id,
                                "generation": generation,
                                "stage": "candidate_minibatch",
                            }),
                        },
                        error,
                    );
                }
            };
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::Evaluating,
                OptimizerTransitionTrigger::RolloutsFinished,
                "Candidate minibatch rollouts finished",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "generation": generation,
                    "stage": "candidate_minibatch",
                }),
            )?;
            rollout_count += eval.rollout_count;
            total_usage.merge(&eval.usage);
            total_cost += eval.cost_usd;
            append_rollout_usage(&mut usage_ledger, &eval);
            let mut metadata = Map::new();
            metadata.insert(
                "stage".to_string(),
                Value::String("candidate_minibatch".to_string()),
            );
            metadata.insert("generation".to_string(), json!(generation));
            metadata.insert("rollout_delta".to_string(), json!(eval.rollout_count));
            metadata.insert("average_reward".to_string(), json!(eval.average_reward));
            push_stopper_snapshot(
                &mut stopper_states,
                &mut stopper_sequence,
                &config,
                StopperSnapshot {
                    status: budget_status(&config, rollout_count, total_cost),
                    reason: Some("candidate minibatch evaluation completed"),
                    generation: Some(generation),
                    candidate_id: Some(&candidate.candidate_id),
                    evaluation_stage: Some("candidate_minibatch"),
                    rollout_count,
                    cost_usd: total_cost,
                    metadata,
                },
            );
            candidate.status = "minibatch_evaluated".to_string();
            candidate.minibatch_reward = Some(eval.average_reward);
            candidate.minibatch_scores = eval.scores.clone();
            candidate.sensor_frames.extend(eval.sensor_frames.clone());
            let candidate_minibatch_vector =
                score_vector_for_candidate(CandidateScoreVectorInput {
                    objective_set: &objective_set,
                    candidate: &candidate,
                    rows: &minibatch_rows,
                    split: &config.dataset.train_split,
                    source_stages: &["candidate_minibatch"],
                    evaluation_stage: "candidate_minibatch",
                })?;
            let parent_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set: &objective_set,
                candidate: &proposal_parent,
                rows: &minibatch_rows,
                split: &config.dataset.train_split,
                source_stages: parent_minibatch_reference_source_stages(),
                evaluation_stage: "parent_minibatch_reference",
            })?;
            let minibatch_preference = compare_score_vectors(ScoreVectorPreferenceInput {
                objective_set: &objective_set,
                split: &config.dataset.train_split,
                evaluation_stage: "candidate_minibatch",
                challenger: &candidate_minibatch_vector,
                incumbent: &parent_minibatch_vector,
                accept_equal: true,
                acceptance_criterion: Some(&config.gepa.acceptance_criterion),
                objective_acceptance: Some(&config.gepa.objective_acceptance),
                margin: config.gepa.minibatch_accept_margin,
            })?;
            candidate.acceptance_score = minibatch_preference.score.clone();
            candidate.acceptance_metadata = minibatch_preference.metadata.clone();
            persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
            events.emit(
                "candidate.minibatch_evaluated",
                "Candidate minibatch evaluated",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "minibatch_reward": eval.average_reward,
                    "parent_minibatch_reward": parent_minibatch_reward,
                }),
            )?;
            let mut decision = AcceptanceDecision {
                candidate_id: candidate.candidate_id.clone(),
                parent_id: proposal_parent.candidate_id.clone(),
                accepted_minibatch: minibatch_preference.preferred,
                accepted_full_train: false,
                reason: String::new(),
                candidate_minibatch_reward: eval.average_reward,
                parent_minibatch_reward,
                candidate_train_reward: None,
                best_train_reward: candidates[best_idx]
                    .train_reward
                    .unwrap_or(f64::NEG_INFINITY),
                comparison_result: minibatch_preference.result.clone(),
                score: minibatch_preference.score.clone(),
            };
            if !decision.accepted_minibatch {
                candidate.status = "rejected_minibatch".to_string();
                decision.reason = minibatch_preference.reason;
                events.emit(
                    "candidate.rejected",
                    "Candidate rejected at minibatch",
                    serde_json::to_value(&decision)?,
                )?;
                persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
                candidates.push(candidate);
                transition_run(
                    &workspace,
                    &mut events,
                    &mut state_machine,
                    OptimizerRunState::RolloutQueueing,
                    OptimizerTransitionTrigger::EvaluationFinished,
                    "Candidate minibatch evaluation finished",
                    json!({"generation": generation}),
                )?;
                continue;
            }
            let full_train_rollout_budget =
                remaining_train_rollout_capacity(&workspace, &config, rollout_count)?;
            if full_train_rollout_budget < train_rows.len() {
                candidate.status = "deferred_budget".to_string();
                decision.reason =
                    "insufficient rollout budget for full-train evaluation".to_string();
                let mut metadata = Map::new();
                metadata.insert(
                    "stage".to_string(),
                    Value::String("candidate_full_train".to_string()),
                );
                metadata.insert("generation".to_string(), json!(generation));
                metadata.insert(
                    "remaining_rollouts".to_string(),
                    json!(full_train_rollout_budget),
                );
                metadata.insert("required_rollouts".to_string(), json!(train_rows.len()));
                push_stopper_snapshot(
                    &mut stopper_states,
                    &mut stopper_sequence,
                    &config,
                    StopperSnapshot {
                        status: "deferred_budget",
                        reason: Some("insufficient rollout budget for full-train evaluation"),
                        generation: Some(generation),
                        candidate_id: Some(&candidate.candidate_id),
                        evaluation_stage: Some("candidate_full_train"),
                        rollout_count,
                        cost_usd: total_cost,
                        metadata,
                    },
                );
                events.emit(
                    "candidate.deferred",
                    "Candidate deferred before full-train",
                    serde_json::to_value(&decision)?,
                )?;
                persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
                candidates.push(candidate);
                transition_run(
                    &workspace,
                    &mut events,
                    &mut state_machine,
                    OptimizerRunState::Ready,
                    OptimizerTransitionTrigger::EvaluationFinished,
                    "Candidate deferred before full-train",
                    json!({"generation": generation}),
                )?;
                break;
            }
            if let Some(breach) = next_rollout_budget_breach(&workspace, &config)? {
                candidate.status = "deferred_budget".to_string();
                decision.reason = "insufficient budget for full-train evaluation".to_string();
                let mut metadata = Map::new();
                metadata.insert(
                    "stage".to_string(),
                    Value::String("candidate_full_train".to_string()),
                );
                metadata.insert("generation".to_string(), json!(generation));
                metadata.insert("limit".to_string(), json!(breach.limit.clone()));
                metadata.insert("requested".to_string(), json!(breach.requested.clone()));
                metadata.insert("available".to_string(), json!(breach.available.clone()));
                push_stopper_snapshot(
                    &mut stopper_states,
                    &mut stopper_sequence,
                    &config,
                    StopperSnapshot {
                        status: "deferred_budget",
                        reason: Some("insufficient budget for full-train evaluation"),
                        generation: Some(generation),
                        candidate_id: Some(&candidate.candidate_id),
                        evaluation_stage: Some("candidate_full_train"),
                        rollout_count,
                        cost_usd: total_cost,
                        metadata,
                    },
                );
                events.emit(
                    "candidate.deferred",
                    "Candidate deferred before full-train",
                    serde_json::to_value(&decision)?,
                )?;
                persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
                candidates.push(candidate);
                transition_run(
                    &workspace,
                    &mut events,
                    &mut state_machine,
                    OptimizerRunState::Ready,
                    OptimizerTransitionTrigger::EvaluationFinished,
                    "Candidate deferred before full-train",
                    json!({"generation": generation}),
                )?;
                break;
            }
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::RolloutQueueing,
                OptimizerTransitionTrigger::RolloutsQueued,
                "Candidate full-train rollouts queued",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "generation": generation,
                    "stage": "candidate_full_train",
                    "row_count": train_rows.len(),
                }),
            )?;
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::RolloutRunning,
                OptimizerTransitionTrigger::RolloutsStarted,
                "Candidate full-train rollouts started",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "generation": generation,
                    "stage": "candidate_full_train",
                }),
            )?;
            let train_eval = match evaluate_candidate(EvaluationCall {
                client: &client,
                workspace: &workspace,
                paths: &paths,
                cache: &mut cache,
                cache_namespace: &cache_namespace,
                config: &config,
                program: &program,
                task_id: &rollout_task_id,
                objective_set: &objective_set,
                candidate: &candidate,
                rows: &train_rows,
                stage: "candidate_full_train",
                cancellation: options.cancellation.as_ref(),
            }) {
                Ok(eval) => eval,
                Err(error) => {
                    return fail_gepa_run_and_return(
                        FailedGepaRunInput {
                            workspace: &mut workspace,
                            events: &mut events,
                            state_machine: &mut state_machine,
                            paths: &paths,
                            registry: &registry,
                            cache: &mut cache,
                            config: &config,
                            cache_mode,
                            cache_namespace: &cache_namespace,
                            best_candidate_id: Some(&candidates[best_idx].candidate_id),
                            total_cost,
                            total_usage: &total_usage,
                            usage_ledger: &usage_ledger,
                            stopper_states: &stopper_states,
                            message: "Candidate full-train rollout failed",
                            details: json!({
                                "candidate_id": candidate.candidate_id,
                                "generation": generation,
                                "stage": "candidate_full_train",
                            }),
                        },
                        error,
                    );
                }
            };
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::Evaluating,
                OptimizerTransitionTrigger::RolloutsFinished,
                "Candidate full-train rollouts finished",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "generation": generation,
                    "stage": "candidate_full_train",
                }),
            )?;
            rollout_count += train_eval.rollout_count;
            total_usage.merge(&train_eval.usage);
            total_cost += train_eval.cost_usd;
            append_rollout_usage(&mut usage_ledger, &train_eval);
            let mut metadata = Map::new();
            metadata.insert(
                "stage".to_string(),
                Value::String("candidate_full_train".to_string()),
            );
            metadata.insert("generation".to_string(), json!(generation));
            metadata.insert("rollout_delta".to_string(), json!(train_eval.rollout_count));
            metadata.insert(
                "average_reward".to_string(),
                json!(train_eval.average_reward),
            );
            push_stopper_snapshot(
                &mut stopper_states,
                &mut stopper_sequence,
                &config,
                StopperSnapshot {
                    status: budget_status(&config, rollout_count, total_cost),
                    reason: Some("candidate full-train evaluation completed"),
                    generation: Some(generation),
                    candidate_id: Some(&candidate.candidate_id),
                    evaluation_stage: Some("candidate_full_train"),
                    rollout_count,
                    cost_usd: total_cost,
                    metadata,
                },
            );
            candidate.status = "full_train_evaluated".to_string();
            candidate.train_reward = Some(train_eval.average_reward);
            candidate.train_scores = train_eval.scores.clone();
            candidate
                .sensor_frames
                .extend(train_eval.sensor_frames.clone());
            let candidate_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set: &objective_set,
                candidate: &candidate,
                rows: &train_rows,
                split: &config.dataset.train_split,
                source_stages: &["candidate_full_train"],
                evaluation_stage: "candidate_full_train",
            })?;
            let best_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set: &objective_set,
                candidate: &candidates[best_idx],
                rows: &train_rows,
                split: &config.dataset.train_split,
                source_stages: &["seed_full_train", "candidate_full_train"],
                evaluation_stage: "best_full_train_reference",
            })?;
            let train_preference = compare_score_vectors(ScoreVectorPreferenceInput {
                objective_set: &objective_set,
                split: &config.dataset.train_split,
                evaluation_stage: "candidate_full_train",
                challenger: &candidate_train_vector,
                incumbent: &best_train_vector,
                accept_equal: true,
                acceptance_criterion: Some(&config.gepa.acceptance_criterion),
                objective_acceptance: Some(&config.gepa.objective_acceptance),
                margin: 0.0,
            })?;
            candidate.acceptance_score = train_preference.score.clone();
            candidate.acceptance_metadata = train_preference.metadata.clone();
            persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
            decision.candidate_train_reward = Some(train_eval.average_reward);
            events.emit(
                "candidate.full_train_evaluated",
                "Candidate full train evaluated",
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "train_reward": train_eval.average_reward,
                    "best_train_reward": candidates[best_idx].train_reward,
                }),
            )?;
            let accepted = train_preference.preferred;
            decision.accepted_full_train = accepted;
            decision.reason = train_preference.reason;
            decision.comparison_result = train_preference.result;
            decision.score = train_preference.score;
            candidate.status = if accepted {
                "accepted".to_string()
            } else {
                "rejected_full_train".to_string()
            };
            events.emit(
                if accepted {
                    "candidate.accepted"
                } else {
                    "candidate.rejected"
                },
                if accepted {
                    "Candidate accepted"
                } else {
                    "Candidate rejected"
                },
                serde_json::to_value(&decision)?,
            )?;
            persist_candidate_snapshot(&mut workspace, &config.run.run_id, &candidate)?;
            let previous_frontier_member_ids = frontier_member_ids(&frontier_members(&candidates));
            candidates.push(candidate);
            if accepted {
                best_idx = candidates.len() - 1;
                let changed_candidate_id = candidates[candidates.len() - 1].candidate_id.clone();
                events.emit(
                    "frontier.updated",
                    "Frontier updated",
                    frontier_snapshot_value(
                        &candidates,
                        &train_rows,
                        Some(best_idx),
                        Some(generation),
                        "candidate_accepted",
                        Some(&changed_candidate_id),
                        Some(previous_frontier_member_ids),
                    )?,
                )?;
            }
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::RolloutQueueing,
                OptimizerTransitionTrigger::EvaluationFinished,
                "Candidate full-train evaluation finished",
                json!({"generation": generation}),
            )?;
        }
        if state_machine.state() != OptimizerRunState::Ready {
            transition_run(
                &workspace,
                &mut events,
                &mut state_machine,
                OptimizerRunState::Ready,
                OptimizerTransitionTrigger::EvaluationFinished,
                "Generation evaluation finished",
                json!({"generation": generation}),
            )?;
        }
        events.emit(
            "frontier.snapshot",
            "Frontier generation snapshot",
            frontier_snapshot_value(
                &candidates,
                &train_rows,
                Some(best_idx),
                Some(generation),
                "generation_complete",
                None,
                None,
            )?,
        )?;
        let frontier = frontier_members(&candidates);
        let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
            config: &config,
            candidates: &candidates,
            frontier: frontier.clone(),
            best_idx: Some(best_idx),
            state_machine: &state_machine,
            rollout_count,
            total_usage: &total_usage,
            total_cost,
        });
        let mut metadata = Map::new();
        metadata.insert("generation".to_string(), json!(generation));
        metadata.insert(
            "stage".to_string(),
            Value::String("generation_complete".to_string()),
        );
        record_checkpoint_snapshot(
            &mut workspace,
            &config.run.run_id,
            &mut checkpoint_sequence,
            &state_machine,
            CheckpointSnapshot {
                checkpoint_kind: "generation_boundary",
                status: "completed",
                reason: Some("generation evaluation completed"),
                generation: Some(generation),
                candidate_id: Some(&candidates[best_idx].candidate_id),
                evaluation_stage: Some("generation_complete"),
                best_candidate_id: Some(&candidates[best_idx].candidate_id),
                candidate_count: candidates.len(),
                frontier_count: frontier.len(),
                rollout_count,
                cost_usd: total_cost,
                usage: serde_json::to_value(&total_usage)?,
                snapshot,
                metadata,
            },
        )?;
        persist_gepa_cursor(
            &mut workspace,
            &config,
            &mut checkpoint_sequence,
            GepaCursorState {
                phase: GepaCursorPhase::GenerationStart,
                generation: generation + 1,
                proposal_index: 0,
                pending_job_id: None,
                pending_effect_id: None,
                pending_reservation_ids: Vec::new(),
                active_evaluation: None,
                candidates: &candidates,
                best_idx: Some(best_idx),
                train_rows: &train_rows,
                minibatch_rows: &minibatch_pool_rows,
                reflection_rows: &reflection_rows,
                heldout_rows: &heldout_rows,
                program: &program,
                objective_set: &objective_set,
                rollout_task_id: &rollout_task_id,
                total_usage: &total_usage,
                total_cost,
                rollout_count,
                stopper_sequence,
                state_machine: &state_machine,
                terminal_summary: None,
                error_summary: None,
                metadata: Map::new(),
            },
            "completed",
            "generation evaluation completed",
        )?;
    }

    check_cancelled(options.cancellation.as_ref())?;
    let frontier = frontier_members(&candidates);
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &config,
        candidates: &candidates,
        frontier: frontier.clone(),
        best_idx: Some(best_idx),
        state_machine: &state_machine,
        rollout_count,
        total_usage: &total_usage,
        total_cost,
    });
    let mut metadata = Map::new();
    metadata.insert(
        "stage".to_string(),
        Value::String("pre_heldout".to_string()),
    );
    metadata.insert("heldout_rows".to_string(), json!(heldout_rows.len()));
    record_checkpoint_snapshot(
        &mut workspace,
        &config.run.run_id,
        &mut checkpoint_sequence,
        &state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "pre_heldout",
            status: "completed",
            reason: Some("optimization loop completed before heldout"),
            generation: None,
            candidate_id: Some(&candidates[best_idx].candidate_id),
            evaluation_stage: Some("pre_heldout"),
            best_candidate_id: Some(&candidates[best_idx].candidate_id),
            candidate_count: candidates.len(),
            frontier_count: frontier.len(),
            rollout_count,
            cost_usd: total_cost,
            usage: serde_json::to_value(&total_usage)?,
            snapshot,
            metadata,
        },
    )?;
    persist_gepa_cursor(
        &mut workspace,
        &config,
        &mut checkpoint_sequence,
        GepaCursorState {
            phase: GepaCursorPhase::Heldout,
            generation: config.gepa.max_generations,
            proposal_index: 0,
            pending_job_id: None,
            pending_effect_id: None,
            pending_reservation_ids: Vec::new(),
            active_evaluation: None,
            candidates: &candidates,
            best_idx: Some(best_idx),
            train_rows: &train_rows,
            minibatch_rows: &minibatch_pool_rows,
            reflection_rows: &reflection_rows,
            heldout_rows: &heldout_rows,
            program: &program,
            objective_set: &objective_set,
            rollout_task_id: &rollout_task_id,
            total_usage: &total_usage,
            total_cost,
            rollout_count,
            stopper_sequence,
            state_machine: &state_machine,
            terminal_summary: None,
            error_summary: None,
            metadata: Map::new(),
        },
        "completed",
        "optimization loop completed before heldout",
    )?;
    let mut heldout_indices = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| candidate.train_reward.map(|_| idx))
        .collect::<Vec<_>>();
    if heldout_indices.is_empty() {
        heldout_indices.push(best_idx);
    }
    let mut heldout_rollout_delta = 0usize;
    let mut heldout_cost_delta = 0.0;
    let heldout_required_rollouts = heldout_indices.len().saturating_mul(heldout_rows.len());
    let heldout_available_rollouts =
        remaining_heldout_rollout_capacity(&workspace, &config, &candidates)?;
    let heldout_budget_breach = next_rollout_budget_breach(&workspace, &config)?;
    let heldout_skipped =
        heldout_available_rollouts < heldout_required_rollouts || heldout_budget_breach.is_some();
    if heldout_skipped {
        let mut metadata = Map::new();
        metadata.insert("stage".to_string(), Value::String("heldout".to_string()));
        metadata.insert(
            "required_rollouts".to_string(),
            json!(heldout_required_rollouts),
        );
        metadata.insert(
            "available_rollouts".to_string(),
            json!(heldout_available_rollouts),
        );
        if let Some(breach) = heldout_budget_breach.as_ref() {
            metadata.insert("limit".to_string(), json!(breach.limit.clone()));
            metadata.insert("requested".to_string(), json!(breach.requested.clone()));
            metadata.insert("available".to_string(), json!(breach.available.clone()));
        }
        metadata.insert("candidate_count".to_string(), json!(heldout_indices.len()));
        push_stopper_snapshot(
            &mut stopper_states,
            &mut stopper_sequence,
            &config,
            StopperSnapshot {
                status: "heldout_skipped_limit_reached",
                reason: Some("insufficient rollout budget for heldout evaluation"),
                generation: None,
                candidate_id: Some(&candidates[best_idx].candidate_id),
                evaluation_stage: Some("heldout"),
                rollout_count,
                cost_usd: total_cost,
                metadata,
            },
        );
        events.emit(
            "heldout.skipped",
            "Heldout skipped due to limits",
            json!({
                "best_candidate_id": candidates[best_idx].candidate_id,
                "required_rollouts": heldout_required_rollouts,
                "available_rollouts": heldout_available_rollouts,
                "budget_breach": heldout_budget_breach.as_ref().map(|breach| json!({
                    "limit": breach.limit.clone(),
                    "requested": breach.requested.clone(),
                    "available": breach.available.clone(),
                })),
            }),
        )?;
    } else {
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::RolloutsQueued,
            "Heldout rollouts queued",
            json!({
                "stage": "heldout",
                "row_count": heldout_rows.len(),
                "candidate_count": heldout_indices.len(),
                "rollout_count": heldout_required_rollouts,
            }),
        )?;
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::RolloutRunning,
            OptimizerTransitionTrigger::RolloutsStarted,
            "Heldout rollouts started",
            json!({
                "stage": "heldout",
            }),
        )?;
        for candidate_idx in heldout_indices.iter().copied() {
            check_cancelled(options.cancellation.as_ref())?;
            let heldout_eval = match evaluate_candidate(EvaluationCall {
                client: &client,
                workspace: &workspace,
                paths: &paths,
                cache: &mut cache,
                cache_namespace: &cache_namespace,
                config: &config,
                program: &program,
                task_id: &rollout_task_id,
                objective_set: &objective_set,
                candidate: &candidates[candidate_idx],
                rows: &heldout_rows,
                stage: "heldout",
                cancellation: options.cancellation.as_ref(),
            }) {
                Ok(eval) => eval,
                Err(error) => {
                    return fail_gepa_run_and_return(
                        FailedGepaRunInput {
                            workspace: &mut workspace,
                            events: &mut events,
                            state_machine: &mut state_machine,
                            paths: &paths,
                            registry: &registry,
                            cache: &mut cache,
                            config: &config,
                            cache_mode,
                            cache_namespace: &cache_namespace,
                            best_candidate_id: Some(&candidates[best_idx].candidate_id),
                            total_cost,
                            total_usage: &total_usage,
                            usage_ledger: &usage_ledger,
                            stopper_states: &stopper_states,
                            message: "Heldout rollout failed",
                            details: json!({
                                "candidate_id": candidates[candidate_idx].candidate_id,
                                "stage": "heldout",
                            }),
                        },
                        error,
                    );
                }
            };
            candidates[candidate_idx].heldout_reward = Some(heldout_eval.average_reward);
            candidates[candidate_idx]
                .sensor_frames
                .extend(heldout_eval.sensor_frames.clone());
            persist_candidate_snapshot(
                &mut workspace,
                &config.run.run_id,
                &candidates[candidate_idx],
            )?;
            total_usage.merge(&heldout_eval.usage);
            total_cost += heldout_eval.cost_usd;
            rollout_count += heldout_eval.rollout_count;
            heldout_rollout_delta += heldout_eval.rollout_count;
            heldout_cost_delta += heldout_eval.cost_usd;
            append_rollout_usage(&mut usage_ledger, &heldout_eval);
            events.emit(
                "heldout.completed",
                "Heldout evaluation completed",
                json!({
                    "candidate_id": candidates[candidate_idx].candidate_id,
                    "train_reward": candidates[candidate_idx].train_reward,
                    "heldout_reward": heldout_eval.average_reward,
                }),
            )?;
        }
        best_idx = select_best_heldout_candidate(HeldoutSelectionInput {
            candidates: &candidates,
            evaluated_indices: &heldout_indices,
            objective_set: &objective_set,
            heldout_split: &config.dataset.heldout_split,
            heldout_rows: &heldout_rows,
            train_split: &config.dataset.train_split,
            train_rows: &train_rows,
            incumbent_idx: Some(best_idx),
        })?
        .unwrap_or(best_idx);
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            OptimizerRunState::Evaluating,
            OptimizerTransitionTrigger::RolloutsFinished,
            "Heldout rollouts finished",
            json!({
                "candidate_id": candidates[best_idx].candidate_id,
                "stage": "heldout",
                "candidate_count": heldout_indices.len(),
            }),
        )?;
    }
    let heldout_best_reward = candidates[best_idx]
        .heldout_reward
        .or(candidates[best_idx].train_reward)
        .unwrap_or(0.0);
    let mut metadata = Map::new();
    metadata.insert("stage".to_string(), Value::String("heldout".to_string()));
    metadata.insert("rollout_delta".to_string(), json!(heldout_rollout_delta));
    metadata.insert("candidate_count".to_string(), json!(heldout_indices.len()));
    metadata.insert("cost_delta_usd".to_string(), json!(heldout_cost_delta));
    metadata.insert("heldout_reward".to_string(), json!(heldout_best_reward));
    push_stopper_snapshot(
        &mut stopper_states,
        &mut stopper_sequence,
        &config,
        StopperSnapshot {
            status: if heldout_skipped {
                "completed_limit_reached"
            } else {
                "completed"
            },
            reason: Some(if heldout_skipped {
                "heldout skipped due to limits"
            } else {
                "heldout evaluation completed"
            }),
            generation: None,
            candidate_id: Some(&candidates[best_idx].candidate_id),
            evaluation_stage: Some("heldout"),
            rollout_count,
            cost_usd: total_cost,
            metadata,
        },
    );
    let score_chart = score_chart_value(&candidates, 0, best_idx, &paths.score_chart_path);
    paths.write_text(
        &paths.score_chart_path,
        &render_score_chart_svg(&config.run.run_id, &score_chart),
    )?;
    events.emit("score_chart.written", "Score chart written", score_chart)?;
    let frontier = frontier_members(&candidates);
    let snapshot = checkpoint_snapshot_value(CheckpointSnapshotState {
        config: &config,
        candidates: &candidates,
        frontier: frontier.clone(),
        best_idx: Some(best_idx),
        state_machine: &state_machine,
        rollout_count,
        total_usage: &total_usage,
        total_cost,
    });
    let mut metadata = Map::new();
    metadata.insert("stage".to_string(), Value::String("heldout".to_string()));
    metadata.insert("heldout_reward".to_string(), json!(heldout_best_reward));
    metadata.insert("heldout_skipped".to_string(), json!(heldout_skipped));
    record_checkpoint_snapshot(
        &mut workspace,
        &config.run.run_id,
        &mut checkpoint_sequence,
        &state_machine,
        CheckpointSnapshot {
            checkpoint_kind: "terminal",
            status: "completed",
            reason: Some(if heldout_skipped {
                "heldout skipped due to limits"
            } else {
                "heldout evaluation completed"
            }),
            generation: None,
            candidate_id: Some(&candidates[best_idx].candidate_id),
            evaluation_stage: Some("heldout"),
            best_candidate_id: Some(&candidates[best_idx].candidate_id),
            candidate_count: candidates.len(),
            frontier_count: frontier.len(),
            rollout_count,
            cost_usd: total_cost,
            usage: serde_json::to_value(&total_usage)?,
            snapshot,
            metadata,
        },
    )?;
    transition_run(
        &workspace,
        &mut events,
        &mut state_machine,
        OptimizerRunState::Completed,
        OptimizerTransitionTrigger::RunCompleted,
        "GEPA run completed",
        json!({
            "best_candidate_id": candidates[best_idx].candidate_id,
            "heldout_reward": heldout_best_reward,
            "heldout_skipped": heldout_skipped,
        }),
    )?;

    let best_candidate = serde_json::to_value(&candidates[best_idx])?;
    let candidate_registry = serde_json::to_value(&candidates)?;
    let frontier = serde_json::to_value(frontier_members(&candidates))?;
    let cache_profile_record = CacheProfileRecord::from_profile(cache.profile()?);
    let cache_access_log = cache.access_log().to_vec();
    let cache_profile = serde_json::to_value(&cache_profile_record.profile)?;
    let usage_value = serde_json::to_value(&total_usage)?;
    let state_history = serde_json::to_value(&state_machine.history)?;
    let candidate_values = candidate_registry.as_array().cloned().unwrap_or_default();
    workspace.persist_candidate_registry(&config.run.run_id, &candidate_values)?;
    workspace.persist_state_history(&state_machine.history)?;
    paths.write_json(&paths.best_candidate_path, &best_candidate)?;
    paths.write_json(&paths.candidate_registry_path, &candidate_registry)?;
    paths.write_json(&paths.frontier_path, &frontier)?;
    paths.write_json(&paths.cache_profile_path, &cache_profile)?;
    let sensor_frame_count = candidates
        .iter()
        .map(|candidate| candidate.sensor_frames.len())
        .sum::<usize>();
    events.emit(
        "workspace.persisted",
        "SQLite workspace persisted",
        json!({
            "workspace_db_path": paths.workspace_db_path,
            "candidate_count": candidates.len(),
            "sensor_frame_count": sensor_frame_count,
            "state_transition_count": state_machine.history.len(),
        }),
    )?;
    let runtime_summary =
        serde_json::to_value(runtime_usage_summary_from_events(events.records()))?;
    events.emit(
        "gepa.run.finished",
        "GEPA run finished",
        json!({
            "best_candidate_id": candidates[best_idx].candidate_id,
            "cost_usd": total_cost,
            "heldout_reward": heldout_best_reward,
            "heldout_skipped": heldout_skipped,
            "rollout_count": rollout_count,
            "runtime_summary": runtime_summary,
            "usage": usage_value,
            "state": state_machine.state().as_str(),
        }),
    )?;
    events.flush()?;
    workspace.record_event_stream(&config.run.run_id, events.records())?;
    normalize_event_feed(
        &paths.event_feed_path,
        &paths.normalized_event_feed_path,
        &paths.run_dir,
    )?;
    registry.append(&RunRegistryEntry::finished(
        &paths,
        &config,
        cache_mode,
        &cache_namespace,
        candidates[best_idx].candidate_id.clone(),
        total_cost,
        usage_value.clone(),
    ))?;
    let artifact_refs = vec![
        paths.artifact_ref(
            &paths.best_candidate_path,
            "best_candidate",
            "release_evidence",
        )?,
        paths.artifact_ref(
            &paths.candidate_registry_path,
            "candidate_registry",
            "release_evidence",
        )?,
        paths.artifact_ref(&paths.frontier_path, "frontier", "release_evidence")?,
        paths.artifact_ref(
            &paths.score_chart_path,
            "score_chart_svg",
            "release_evidence",
        )?,
        paths.artifact_ref(&paths.event_feed_path, "events_jsonl", "release_evidence")?,
        paths.artifact_ref(
            &paths.normalized_event_feed_path,
            "events_normalized_jsonl",
            "release_evidence",
        )?,
        paths.artifact_ref(
            &paths.cache_profile_path,
            "cache_profile",
            "release_evidence",
        )?,
        paths.artifact_ref(
            &paths.run_registry_path,
            "run_registry_jsonl",
            "release_evidence",
        )?,
    ];

    let result = GepaRunResult {
        best_candidate,
        manifest_path: paths.manifest_path.display().to_string(),
        event_feed_path: paths.event_feed_path.display().to_string(),
        normalized_event_feed_path: paths.normalized_event_feed_path.display().to_string(),
        cache_profile_path: paths.cache_profile_path.display().to_string(),
        candidate_registry_path: paths.candidate_registry_path.display().to_string(),
        frontier_path: paths.frontier_path.display().to_string(),
        score_chart_path: paths.score_chart_path.display().to_string(),
        run_registry_path: paths.run_registry_path.display().to_string(),
        workspace_db_path: paths.workspace_db_path.display().to_string(),
        artifact_refs,
        cost_usd: total_cost,
        usage: usage_value,
        state_history,
    };
    let result_value = serde_json::to_value(&result)?;
    workspace.record_artifact_refs(&config.run.run_id, &result.artifact_refs)?;
    workspace.record_cache_profile(&config.run.run_id, &cache_profile_record, &cache_access_log)?;
    workspace.record_usage_ledger(&config.run.run_id, &usage_ledger)?;
    workspace.record_stopper_states(&config.run.run_id, &stopper_states)?;
    workspace.record_manifest(
        &config.run.run_id,
        &paths.manifest_path,
        &candidates[best_idx].candidate_id,
        total_cost,
        &result.usage,
        &result_value,
    )?;
    workspace.record_run_finished(
        &config.run.run_id,
        &candidates[best_idx].candidate_id,
        total_cost,
        &result.usage,
    )?;
    paths.write_json(&paths.manifest_path, &result_value)?;
    persist_gepa_cursor(
        &mut workspace,
        &config,
        &mut checkpoint_sequence,
        GepaCursorState {
            phase: GepaCursorPhase::Completed,
            generation: config.gepa.max_generations,
            proposal_index: 0,
            pending_job_id: None,
            pending_effect_id: None,
            pending_reservation_ids: Vec::new(),
            active_evaluation: None,
            candidates: &candidates,
            best_idx: Some(best_idx),
            train_rows: &train_rows,
            minibatch_rows: &minibatch_pool_rows,
            reflection_rows: &reflection_rows,
            heldout_rows: &heldout_rows,
            program: &program,
            objective_set: &objective_set,
            rollout_task_id: &rollout_task_id,
            total_usage: &total_usage,
            total_cost,
            rollout_count,
            stopper_sequence,
            state_machine: &state_machine,
            terminal_summary: Some(result_value.clone()),
            error_summary: None,
            metadata: Map::new(),
        },
        "completed",
        "GEPA run completed",
    )?;
    Ok(result)
}

fn load_rows(
    client: &ContainerClient,
    cache: &mut RequestCache,
    cache_namespace: &str,
    split: &str,
    seeds: &[i64],
    filters: Value,
) -> Result<DatasetRowsResponse> {
    let request_model = DatasetRowsRequest::new(split, seeds, filters);
    let request = serde_json::to_value(&request_model)?;
    let response = cached_call(
        cache,
        &format!("{cache_namespace}:container.dataset_rows"),
        &request,
        || {
            let response = client.dataset_rows_typed(&request_model)?;
            Ok(serde_json::to_value(response)?)
        },
    )?;
    let response: DatasetRowsResponse = serde_json::from_value(response)?;
    response.validate_for_request(&request_model)?;
    Ok(response)
}

fn seed_candidate_payload(
    config: &SynthOptimizerConfig,
    program: &PromptProgram,
) -> Result<BTreeMap<String, String>> {
    if !config.seed_candidate.is_empty() {
        return Ok(config.seed_candidate.clone());
    }
    if !program.seed_candidate.fields.is_empty() {
        return Ok(program.seed_candidate.fields.clone());
    }
    Err(OptimizerError::Config(
        "seed candidate must be provided by [seed_candidate] or /program.seed_candidate"
            .to_string(),
    ))
}

fn declared_objective_set(
    config: &SynthOptimizerConfig,
    program: &PromptProgram,
    train_rows: &[Value],
    heldout_rows: &[Value],
) -> ObjectiveSetRecord {
    let mut seen = BTreeSet::new();
    let mut objectives = Vec::new();
    for objective in &config.gepa.objective_keys {
        let name = objective.trim();
        if !name.is_empty() && seen.insert(name.to_string()) {
            objectives.push((name.to_string(), "gepa.objective_keys".to_string()));
        }
    }

    if objectives.is_empty() {
        for target in &program.target_modules {
            let name = target.objective.trim();
            if name.is_empty() {
                continue;
            }
            if seen.insert(name.to_string()) {
                objectives.push((name.to_string(), "program.target_modules".to_string()));
            }
        }
    }

    if objectives.is_empty() {
        for row in train_rows.iter().chain(heldout_rows.iter()) {
            let Some(name) = row
                .get("objective")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| is_objective_identifier(value))
            else {
                continue;
            };
            if seen.insert(name.to_string()) {
                objectives.push((name.to_string(), "dataset_rows.objective".to_string()));
            }
        }
    }

    if objectives.is_empty() {
        objectives.push((
            "outcome_reward".to_string(),
            "rollout_response.outcome_reward".to_string(),
        ));
    }

    let configured_selection = config
        .gepa
        .selection_objective
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(selection_objective) = configured_selection {
        if seen.insert(selection_objective.to_string()) {
            objectives.insert(
                0,
                (
                    selection_objective.to_string(),
                    "gepa.selection_objective".to_string(),
                ),
            );
        }
    }
    let selection_objective = configured_selection
        .map(str::to_string)
        .or_else(|| objectives.first().map(|(name, _)| name.clone()))
        .unwrap_or_else(|| "outcome_reward".to_string());
    let specs = objectives
        .iter()
        .map(|(objective, source)| {
            let mut spec = ObjectiveSpec::from_objective_score(&ObjectiveScore {
                objective: objective.clone(),
                value: 0.0,
                source: source.clone(),
                rationale: None,
                metadata: Map::new(),
            });
            spec.direction = normalize_gepa_objective_direction(
                config
                    .gepa
                    .objective_directions
                    .get(objective)
                    .map(String::as_str)
                    .unwrap_or("maximize"),
            );
            spec
        })
        .collect::<Vec<_>>();
    let mut metadata = Map::new();
    metadata.insert("program_id".to_string(), json!(program.program_id.clone()));
    metadata.insert("source".to_string(), json!("gepa.run_start"));
    metadata.insert("train_rows".to_string(), json!(train_rows.len()));
    metadata.insert("heldout_rows".to_string(), json!(heldout_rows.len()));
    metadata.insert(
        "frontier_type_source".to_string(),
        json!("gepa.frontier_type"),
    );
    metadata.insert(
        "objective_keys".to_string(),
        json!(config.gepa.objective_keys),
    );
    metadata.insert(
        "objective_directions".to_string(),
        json!(config.gepa.objective_directions),
    );
    ObjectiveSetRecord::from_specs(
        &selection_objective,
        &normalize_gepa_frontier_type(&config.gepa.frontier_type),
        specs,
        metadata,
    )
}

fn is_objective_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

fn align_sensor_frame_objectives(
    frame: &mut SensorFrame,
    objective_set: &ObjectiveSetRecord,
    reward: f64,
) {
    let selection_objective = objective_set.selection_objective.trim();
    if selection_objective.is_empty()
        || frame
            .objective_scores
            .iter()
            .any(|score| score.objective == selection_objective)
    {
        return;
    }
    let original_objectives = frame
        .objective_scores
        .iter()
        .map(|score| score.objective.clone())
        .collect::<Vec<_>>();
    let mut metadata = Map::new();
    metadata.insert(
        "objective_set_id".to_string(),
        json!(objective_set.objective_set_id.clone()),
    );
    metadata.insert("mapped_from_outcome_reward".to_string(), json!(true));
    metadata.insert(
        "original_objectives".to_string(),
        json!(original_objectives),
    );
    frame.objective_scores.push(ObjectiveScore {
        objective: selection_objective.to_string(),
        value: reward,
        source: "objective_set.selection_reward".to_string(),
        rationale: Some(
            "container outcome reward mapped to the declared selection objective".to_string(),
        ),
        metadata,
    });
}

fn normalize_candidate_payload(
    program: &PromptProgram,
    config: &SynthOptimizerConfig,
    parent_payload: &BTreeMap<String, String>,
    proposed_payload: BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    let mutable_fields = program.mutable_field_ids();
    let allowed_fields = if mutable_fields.is_empty() {
        config.candidate.target_modules.clone()
    } else {
        mutable_fields
    };
    let mut payload = parent_payload.clone();
    for (key, value) in proposed_payload {
        if !allowed_fields.iter().any(|field| field == &key) {
            return Err(OptimizerError::Proposer(format!(
                "proposer returned unknown candidate field {key:?}; allowed fields: {}",
                allowed_fields.join(", ")
            )));
        }
        payload.insert(key, value);
    }
    for module_id in &config.candidate.target_modules {
        let value = payload.get(module_id).map(String::as_str).unwrap_or("");
        if value.trim().is_empty() {
            return Err(OptimizerError::Proposer(format!(
                "candidate field {module_id:?} is required and must be non-empty"
            )));
        }
    }
    Ok(payload)
}

fn minibatch_rows(
    rows: &[Value],
    sampler: &GepaBatchSamplerConfig,
    minibatch_size: usize,
    generation: usize,
    proposal_index: usize,
    proposals_per_generation: usize,
) -> Vec<Value> {
    if rows.is_empty() {
        return Vec::new();
    }
    let size = minibatch_size.min(rows.len()).max(1);
    if size >= rows.len() {
        return rows.to_vec();
    }
    let strategy = normalize_gepa_batch_sampler_name(&sampler.name);
    let mut indices = (0..rows.len()).collect::<Vec<_>>();
    if strategy != "ordered_epoch" {
        deterministic_shuffle_indices(&mut indices, rows, generation, proposal_index, &strategy);
    }
    if strategy == "epoch_shuffled" || strategy == "ordered_epoch" {
        let epoch_width = sampler.epoch_width.unwrap_or(size).max(1);
        let cursor = generation
            .saturating_mul(proposals_per_generation.max(1))
            .saturating_add(proposal_index);
        let start = cursor.saturating_mul(epoch_width) % indices.len();
        return (0..size)
            .map(|offset| rows[indices[(start + offset) % indices.len()]].clone())
            .collect();
    }
    if strategy == "stratified" {
        let field = sampler
            .field
            .as_deref()
            .map(str::trim)
            .filter(|field| !field.is_empty())
            .unwrap_or("metadata.difficulty");
        let selected = stratified_minibatch_indices(rows, &indices, size, field);
        if !selected.is_empty() {
            return selected
                .into_iter()
                .map(|idx| rows[idx].clone())
                .collect::<Vec<_>>();
        }
    }
    indices
        .into_iter()
        .take(size)
        .map(|idx| rows[idx].clone())
        .collect()
}

fn normalize_gepa_batch_sampler_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "epoch_shuffled" => "epoch_shuffled".to_string(),
        "ordered_epoch" | "sequential_epoch" => "ordered_epoch".to_string(),
        "stratified" | "stratified_by_field" => "stratified".to_string(),
        _ => "seeded_shuffle".to_string(),
    }
}

fn deterministic_shuffle_indices(
    indices: &mut [usize],
    rows: &[Value],
    generation: usize,
    proposal_index: usize,
    strategy: &str,
) {
    indices.sort_by(|left, right| {
        deterministic_row_shuffle_key(&rows[*left], *left, generation, proposal_index, strategy)
            .cmp(&deterministic_row_shuffle_key(
                &rows[*right],
                *right,
                generation,
                proposal_index,
                strategy,
            ))
            .then_with(|| left.cmp(right))
    });
}

fn deterministic_row_shuffle_key(
    row: &Value,
    index: usize,
    generation: usize,
    proposal_index: usize,
    strategy: &str,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"gepa:minibatch:");
    hasher.update(strategy.as_bytes());
    hasher.update(b":");
    hasher.update(generation.to_le_bytes());
    hasher.update(b":");
    hasher.update(proposal_index.to_le_bytes());
    hasher.update(b":");
    let row_id = row_example_id(row).unwrap_or_else(|_| format!("row:{index}"));
    hasher.update(row_id.as_bytes());
    hasher.finalize().into()
}

fn stratified_minibatch_indices(
    rows: &[Value],
    shuffled_indices: &[usize],
    limit: usize,
    field: &str,
) -> Vec<usize> {
    let mut buckets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for idx in shuffled_indices {
        let key = row_path_value(&rows[*idx], field)
            .and_then(value_to_bucket_key)
            .unwrap_or_else(|| "default".to_string());
        buckets.entry(key).or_default().push(*idx);
    }
    if buckets.len() <= 1 {
        return Vec::new();
    }
    let mut selected = Vec::new();
    while selected.len() < limit && buckets.values().any(|bucket| !bucket.is_empty()) {
        for bucket in buckets.values_mut() {
            if !bucket.is_empty() {
                selected.push(bucket.remove(0));
                if selected.len() >= limit {
                    break;
                }
            }
        }
    }
    selected
}

fn row_path_value<'a>(row: &'a Value, field: &str) -> Option<&'a Value> {
    let mut current = row;
    for part in field.split('.').filter(|part| !part.is_empty()) {
        current = current.get(part)?;
    }
    Some(current)
}

fn value_to_bucket_key(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn parent_minibatch_reference_source_stages() -> &'static [&'static str] {
    &[
        "seed_full_train",
        "candidate_full_train",
        "parent_minibatch_reference",
    ]
}

fn average_reward_for_candidate_rows_from_stages(
    candidate: &CandidateRecord,
    rows: &[Value],
    split: &str,
    source_stages: &[&str],
) -> Result<Option<f64>> {
    if rows.is_empty() {
        return Ok(Some(0.0));
    }
    let mut total = 0.0;
    for row in rows {
        let example_id = row_example_id(row)?;
        let Some(frame) = candidate.sensor_frames.iter().find(|frame| {
            frame.split == split
                && frame.example_id == example_id
                && source_stages
                    .iter()
                    .any(|stage| *stage == frame.evaluation_stage)
        }) else {
            return Ok(None);
        };
        total += frame.reward;
    }
    Ok(Some(total / rows.len() as f64))
}

fn parent_minibatch_reward_for_rows(
    candidate: &CandidateRecord,
    rows: &[Value],
    split: &str,
) -> Result<Option<f64>> {
    average_reward_for_candidate_rows_from_stages(
        candidate,
        rows,
        split,
        parent_minibatch_reference_source_stages(),
    )
}

fn score_vector_for_candidate(input: CandidateScoreVectorInput<'_>) -> Result<ScoreVectorRecord> {
    let row_example_ids = input
        .rows
        .iter()
        .map(row_example_id)
        .collect::<Result<BTreeSet<_>>>()?;
    let declared_objectives = input
        .objective_set
        .objectives
        .iter()
        .map(|objective| objective.name.clone())
        .collect::<BTreeSet<_>>();
    let mut scores = Vec::new();
    for frame in &input.candidate.sensor_frames {
        if frame.split != input.split {
            continue;
        }
        if !input
            .source_stages
            .iter()
            .any(|stage| *stage == frame.evaluation_stage)
        {
            continue;
        }
        if !row_example_ids.contains(&frame.example_id) {
            continue;
        }
        let records = SensorScoreRecords::from_sensor_frame(frame);
        scores.extend(
            records
                .scores
                .into_iter()
                .filter(|score| declared_objectives.contains(&score.objective)),
        );
    }
    if scores.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "missing score vector material for candidate={} split={} evaluation_stage={} source_stages={:?}; no sensor scores matched {} requested rows",
            input.candidate.candidate_id,
            input.split,
            input.evaluation_stage,
            input.source_stages,
            row_example_ids.len()
        )));
    }
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!("gepa.decision_bridge"));
    metadata.insert(
        "source_stages".to_string(),
        json!(input.source_stages.to_vec()),
    );
    metadata.insert("row_count".to_string(), json!(input.rows.len()));
    let vector = ScoreVectorRecord::from_scores(
        input.objective_set,
        &input.candidate.candidate_id,
        input.split,
        input.evaluation_stage,
        &scores,
        metadata,
    );
    let covered_example_ids = vector.example_ids.iter().cloned().collect::<BTreeSet<_>>();
    let missing_example_ids = row_example_ids
        .difference(&covered_example_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_example_ids.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "score vector for candidate={} split={} evaluation_stage={} is missing requested rows {:?}",
            input.candidate.candidate_id, input.split, input.evaluation_stage, missing_example_ids
        )));
    }
    if !vector.missing_objectives.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "score vector for candidate={} split={} evaluation_stage={} is missing objectives {:?}",
            input.candidate.candidate_id,
            input.split,
            input.evaluation_stage,
            vector.missing_objectives
        )));
    }
    if vector.selection_score.is_none() {
        return Err(OptimizerError::Invariant(format!(
            "score vector for candidate={} split={} evaluation_stage={} has no selection objective {:?}",
            input.candidate.candidate_id,
            input.split,
            input.evaluation_stage,
            input.objective_set.selection_objective
        )));
    }
    Ok(vector)
}

fn compare_score_vectors(input: ScoreVectorPreferenceInput<'_>) -> Result<ScoreVectorPreference> {
    let direction = selection_objective_direction(input.objective_set);
    let challenger = input.challenger;
    let incumbent = input.incumbent;

    let mut comparison_metadata = Map::new();
    comparison_metadata.insert("source".to_string(), json!("gepa.decision_bridge"));
    let comparison = ParetoComparisonRecord::from_vectors(
        input.objective_set,
        &input.objective_set.frontier_type,
        input.split,
        input.evaluation_stage,
        challenger,
        incumbent,
        comparison_metadata,
    );
    let selection_delta = challenger
        .selection_score
        .zip(incumbent.selection_score)
        .map(|(left, right)| (left - right) * direction);
    if let Some(criterion) = input.acceptance_criterion {
        let criterion = criterion.to_string();
        let default_acceptance;
        let objective_acceptance = if let Some(config) = input.objective_acceptance {
            config
        } else {
            default_acceptance = GepaObjectiveAcceptanceConfig::default();
            &default_acceptance
        };
        return acceptance_preference_from_vectors(
            &input,
            comparison,
            selection_delta,
            &criterion,
            objective_acceptance,
        );
    }
    let selection_prefers = selection_delta.map(|delta| {
        if input.accept_equal {
            delta >= -f64::EPSILON
        } else {
            delta > f64::EPSILON
        }
    });
    let preferred = match comparison.result.as_str() {
        "challenger_dominates" => true,
        "incumbent_dominates" => false,
        "tie" => input.accept_equal,
        "mixed" | "incomparable" => selection_prefers.ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "score-vector comparison result={} needs selection scores for split={} evaluation_stage={}",
                comparison.result, input.split, input.evaluation_stage
            ))
        })?,
        other => {
            return Err(OptimizerError::Invariant(format!(
                "unknown score-vector comparison result={other} for split={} evaluation_stage={}",
                input.split, input.evaluation_stage
            )));
        }
    };
    let score = json!({
        "schema_version": "gepa_decision_score.v1",
        "decision_source": "score_vector",
        "selection_objective": input.objective_set.selection_objective,
        "objective_set_id": input.objective_set.objective_set_id,
        "objective_set_hash": input.objective_set.objective_set_hash,
        "frontier_type": input.objective_set.frontier_type,
        "split": input.split,
        "evaluation_stage": input.evaluation_stage,
        "comparison_result": comparison.result,
        "comparison": comparison,
        "challenger_score_vector_id": challenger.score_vector_id,
        "incumbent_score_vector_id": incumbent.score_vector_id,
        "challenger_selection_score": challenger.selection_score,
        "incumbent_selection_score": incumbent.selection_score,
        "selection_delta": selection_delta,
        "direction": if direction >= 0.0 { "maximize" } else { "minimize" },
    });
    let mut metadata = Map::new();
    metadata.insert("decision_source".to_string(), json!("score_vector"));
    metadata.insert(
        "comparison_result".to_string(),
        json!(score
            .get("comparison_result")
            .and_then(Value::as_str)
            .unwrap_or("unknown")),
    );
    Ok(ScoreVectorPreference {
        preferred,
        result: score
            .get("comparison_result")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        reason: format!(
            "score-vector comparison result={} selection_objective={}",
            score
                .get("comparison_result")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            input.objective_set.selection_objective
        ),
        score,
        metadata,
    })
}

fn acceptance_preference_from_vectors(
    input: &ScoreVectorPreferenceInput<'_>,
    comparison: ParetoComparisonRecord,
    selection_delta: Option<f64>,
    criterion: &str,
    config: &GepaObjectiveAcceptanceConfig,
) -> Result<ScoreVectorPreference> {
    let criterion = normalize_gepa_acceptance_criterion(criterion);
    let primary_delta = selection_delta.ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "acceptance criterion {criterion} needs selection scores for split={} evaluation_stage={}",
            input.split, input.evaluation_stage
        ))
    })?;
    let margin = input.margin.max(0.0);
    let (accepted, reason, objective_deltas) = match criterion.as_str() {
        "improvement_or_equal" => {
            let accepted = primary_delta >= -margin - f64::EPSILON;
            (
                accepted,
                if accepted {
                    "primary_improvement_or_equal".to_string()
                } else {
                    "primary_regressed".to_string()
                },
                BTreeMap::new(),
            )
        }
        "primary_improvement" => {
            let accepted = primary_delta > margin + f64::EPSILON;
            (
                accepted,
                if accepted {
                    "primary_improvement".to_string()
                } else {
                    "primary_not_improved".to_string()
                },
                BTreeMap::new(),
            )
        }
        _ => {
            let scalar_accepted = primary_delta > margin + f64::EPSILON;
            let objective_deltas = objective_deltas_for_vectors(
                input.objective_set,
                input.challenger,
                input.incumbent,
            );
            if scalar_accepted {
                (true, "primary_improvement".to_string(), objective_deltas)
            } else if objective_deltas.is_empty() {
                (false, "no_objective_scores".to_string(), objective_deltas)
            } else if criterion == "any_objective_improved" {
                let (best_objective, best_delta) = best_objective_delta(&objective_deltas);
                let min_delta = config.min_objective_delta.unwrap_or(0.05);
                let accepted = best_delta >= min_delta;
                (
                    accepted,
                    if accepted {
                        format!("objective_improvement:{best_objective}")
                    } else {
                        "objective_delta_below_threshold".to_string()
                    },
                    objective_deltas,
                )
            } else {
                let (best_objective, best_delta) = best_objective_delta(&objective_deltas);
                let min_delta = config.min_objective_delta.unwrap_or(0.05);
                let tolerance = config.objective_regression_tolerance.unwrap_or(0.10);
                let protected = protected_objectives(config, &objective_deltas);
                let protected_ok = protected.iter().all(|objective| {
                    objective_deltas.get(objective).copied().unwrap_or(0.0) >= -tolerance
                });
                let accepted = best_delta >= min_delta && protected_ok;
                (
                    accepted,
                    if accepted {
                        format!("objective_improvement:{best_objective}")
                    } else if !protected_ok {
                        "protected_objective_regression".to_string()
                    } else {
                        "objective_delta_below_threshold".to_string()
                    },
                    objective_deltas,
                )
            }
        }
    };
    let candidate_objectives = objective_values_as_f64(&input.challenger.objective_values);
    let parent_objectives = objective_values_as_f64(&input.incumbent.objective_values);
    let score = json!({
        "schema_version": "gepa_decision_score.v1",
        "decision_source": "acceptance_criterion",
        "acceptance_criterion": criterion,
        "acceptance_reason": reason,
        "accepted": accepted,
        "selection_objective": input.objective_set.selection_objective,
        "objective_set_id": input.objective_set.objective_set_id,
        "objective_set_hash": input.objective_set.objective_set_hash,
        "frontier_type": input.objective_set.frontier_type,
        "split": input.split,
        "evaluation_stage": input.evaluation_stage,
        "comparison_result": comparison.result,
        "comparison": comparison,
        "challenger_score_vector_id": input.challenger.score_vector_id,
        "incumbent_score_vector_id": input.incumbent.score_vector_id,
        "challenger_selection_score": input.challenger.selection_score,
        "incumbent_selection_score": input.incumbent.selection_score,
        "selection_delta": selection_delta,
        "primary_delta": primary_delta,
        "margin": margin,
        "objective_deltas": objective_deltas,
        "candidate_objectives": candidate_objectives,
        "parent_objectives": parent_objectives,
        "objective_acceptance": {
            "min_objective_delta": config.min_objective_delta.unwrap_or(0.05),
            "objective_regression_tolerance": config.objective_regression_tolerance.unwrap_or(0.10),
            "protected_objectives": config.protected_objectives.clone(),
        },
    });
    let mut metadata = Map::new();
    metadata.insert("decision_source".to_string(), json!("acceptance_criterion"));
    metadata.insert("acceptance_criterion".to_string(), json!(criterion));
    metadata.insert("acceptance_reason".to_string(), json!(reason));
    metadata.insert(
        "comparison_result".to_string(),
        json!(score
            .get("comparison_result")
            .and_then(Value::as_str)
            .unwrap_or("unknown")),
    );
    Ok(ScoreVectorPreference {
        preferred: accepted,
        result: score
            .get("comparison_result")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        reason,
        score,
        metadata,
    })
}

fn normalize_gepa_acceptance_criterion(criterion: &str) -> String {
    match criterion
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "improvement_or_equal" => "improvement_or_equal".to_string(),
        "primary_or_objective" => "primary_or_objective".to_string(),
        "any_objective_improved" => "any_objective_improved".to_string(),
        "protected_objective_guard" => "protected_objective_guard".to_string(),
        _ => "primary_improvement".to_string(),
    }
}

fn objective_deltas_for_vectors(
    objective_set: &ObjectiveSetRecord,
    challenger: &ScoreVectorRecord,
    incumbent: &ScoreVectorRecord,
) -> BTreeMap<String, f64> {
    objective_set
        .objectives
        .iter()
        .filter_map(|objective| {
            let left = challenger.objective_value(&objective.name)?;
            let right = incumbent.objective_value(&objective.name)?;
            Some((
                objective.name.clone(),
                (left - right) * objective_direction_multiplier(&objective.direction),
            ))
        })
        .collect()
}

fn objective_values_as_f64(values: &Map<String, Value>) -> BTreeMap<String, f64> {
    values
        .iter()
        .filter_map(|(objective, value)| value.as_f64().map(|score| (objective.clone(), score)))
        .collect()
}

fn best_objective_delta(deltas: &BTreeMap<String, f64>) -> (String, f64) {
    deltas
        .iter()
        .max_by(|left, right| {
            left.1
                .partial_cmp(right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.0.cmp(left.0))
        })
        .map(|(objective, delta)| (objective.clone(), *delta))
        .unwrap_or_else(|| ("".to_string(), 0.0))
}

fn protected_objectives(
    config: &GepaObjectiveAcceptanceConfig,
    deltas: &BTreeMap<String, f64>,
) -> Vec<String> {
    let configured = config
        .protected_objectives
        .iter()
        .map(|objective| objective.trim())
        .filter(|objective| !objective.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if configured.is_empty() {
        deltas.keys().cloned().collect()
    } else {
        configured
    }
}

fn selection_objective_direction(objective_set: &ObjectiveSetRecord) -> f64 {
    objective_set
        .objectives
        .iter()
        .find(|objective| objective.name == objective_set.selection_objective)
        .map(|objective| objective.direction.trim().to_ascii_lowercase())
        .map(|direction| match direction.as_str() {
            "min" | "minimize" | "lower" | "lower_is_better" | "down" => -1.0,
            _ => 1.0,
        })
        .unwrap_or(1.0)
}

fn normalize_gepa_objective_direction(direction: &str) -> String {
    match direction.trim().to_ascii_lowercase().as_str() {
        "min" | "minimize" | "lower" | "lower_is_better" | "down" => "minimize".to_string(),
        _ => "maximize".to_string(),
    }
}

fn objective_direction_multiplier(direction: &str) -> f64 {
    match normalize_gepa_objective_direction(direction).as_str() {
        "minimize" => -1.0,
        _ => 1.0,
    }
}

fn row_example_id(row: &Value) -> Result<String> {
    dataset_row_identity(row)
}

fn row_seed(row: &Value) -> i64 {
    row.get("seed").and_then(Value::as_i64).unwrap_or(0)
}

fn effective_gepa_seed_pool_seeds(config: &SynthOptimizerConfig) -> BTreeMap<String, Vec<i64>> {
    let pareto_eval = if config.gepa.seed_pools.pareto_eval.is_empty() {
        config.dataset.train_seeds.clone()
    } else {
        config.gepa.seed_pools.pareto_eval.clone()
    };
    let minibatch = if config.gepa.seed_pools.minibatch.is_empty() {
        pareto_eval.clone()
    } else {
        config.gepa.seed_pools.minibatch.clone()
    };
    let reflection = if config.gepa.seed_pools.reflection.is_empty() {
        pareto_eval.clone()
    } else {
        config.gepa.seed_pools.reflection.clone()
    };
    let validation = if config.gepa.seed_pools.validation.is_empty() {
        config.dataset.heldout_seeds.clone()
    } else {
        config.gepa.seed_pools.validation.clone()
    };
    BTreeMap::from([
        ("pareto_eval".to_string(), pareto_eval),
        ("minibatch".to_string(), minibatch),
        ("reflection".to_string(), reflection),
        ("validation".to_string(), validation),
    ])
}

fn seed_pool_rows_value(
    pareto_eval_rows: &[Value],
    minibatch_rows: &[Value],
    reflection_rows: &[Value],
    validation_rows: &[Value],
) -> Value {
    json!({
        "schema_version": "gepa_seed_pools.v1",
        "pareto_eval": {
            "row_count": pareto_eval_rows.len(),
            "seeds": pareto_eval_rows.iter().map(row_seed).collect::<Vec<_>>(),
            "rows": pareto_eval_rows,
        },
        "minibatch": {
            "row_count": minibatch_rows.len(),
            "seeds": minibatch_rows.iter().map(row_seed).collect::<Vec<_>>(),
            "rows": minibatch_rows,
        },
        "reflection": {
            "row_count": reflection_rows.len(),
            "seeds": reflection_rows.iter().map(row_seed).collect::<Vec<_>>(),
            "rows": reflection_rows,
        },
        "validation": {
            "row_count": validation_rows.len(),
            "seeds": validation_rows.iter().map(row_seed).collect::<Vec<_>>(),
            "rows": validation_rows,
        },
    })
}

fn proposer_frontier_summary(
    candidates: &[CandidateRecord],
    train_rows: &[Value],
    best_idx: Option<usize>,
) -> Result<Value> {
    Ok(json!({
        "schema_version": "gepa_frontier_summary.v1",
        "frontier": frontier_members(candidates),
        "snapshot": frontier_snapshot_value(
            candidates,
            train_rows,
            best_idx,
            None,
            "proposer_request",
            None,
            None,
        )?,
    }))
}

fn proposer_started_details(
    config: &SynthOptimizerConfig,
    candidates: &[CandidateRecord],
    generation: usize,
    parent_candidate_id: &str,
    parent_selection: Value,
    run_dir: &Path,
) -> Value {
    let rollout_row_count = candidates
        .iter()
        .map(|candidate| {
            candidate.minibatch_scores.len()
                + candidate.train_scores.len()
                + candidate.sensor_frames.len()
        })
        .sum::<usize>();
    let loss_count = candidates
        .iter()
        .flat_map(|candidate| candidate.sensor_frames.iter())
        .filter(|frame| frame.reward < 1.0)
        .count();
    let win_count = candidates
        .iter()
        .flat_map(|candidate| candidate.sensor_frames.iter())
        .filter(|frame| frame.reward >= 1.0)
        .count();
    json!({
        "generation": generation,
        "backend": config.proposer.backend,
        "model": config.proposer.model,
        "proposal_count": config.gepa.proposals_per_generation,
        "parent_candidate_id": parent_candidate_id,
        "frontier_size": frontier_members(candidates).len(),
        "candidate_count": candidates.len(),
        "rollout_row_count": rollout_row_count,
        "loss_count": loss_count,
        "win_count": win_count,
        "workspace": run_dir
            .join("proposer_workspaces")
            .join(format!("generation_{generation:03}"))
            .display()
            .to_string(),
        "parent_selection": parent_selection,
    })
}

fn proposer_minibatch_failures(candidates: &[CandidateRecord]) -> Value {
    Value::Array(
        candidates
            .iter()
            .filter(|candidate| {
                candidate.minibatch_reward.is_some()
                    && candidate.train_reward.is_none()
                    && !matches!(
                        candidate.status.as_str(),
                        "accepted" | "full_train_evaluated" | "candidate_full_train"
                    )
            })
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "status": candidate.status,
                    "minibatch_reward": candidate.minibatch_reward,
                    "acceptance": candidate.acceptance_metadata,
                })
            })
            .collect(),
    )
}

fn proposer_rollout_trace_artifact_refs(candidates: &[CandidateRecord]) -> Value {
    let refs = candidates
        .iter()
        .flat_map(|candidate| candidate.sensor_frames.iter())
        .flat_map(|frame| frame.artifact_refs.iter())
        .filter(|artifact| artifact.kind == "rollout_trace_payload")
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .collect::<Vec<_>>();
    Value::Array(refs)
}

fn proposer_merge_evidence_artifacts(paths: &ArtifactPaths) -> Result<Value> {
    let artifacts = [
        (&paths.frontier_path, "frontier"),
        (&paths.candidate_registry_path, "candidate_registry"),
        (&paths.normalized_event_feed_path, "events_normalized_jsonl"),
        (&paths.workspace_db_path, "workspace_sqlite"),
    ];
    let mut refs = Vec::new();
    for (path, kind) in artifacts {
        if path.exists() {
            refs.push(serde_json::to_value(paths.artifact_ref(
                path,
                kind,
                "run_evidence",
            )?)?);
        }
    }
    Ok(Value::Array(refs))
}

fn rollout_task_id(program: &PromptProgram) -> String {
    program
        .metadata
        .get("task_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&program.program_id)
        .to_string()
}

fn frontier_members(candidates: &[CandidateRecord]) -> Vec<FrontierMember> {
    let evaluated = candidates
        .iter()
        .filter(|candidate| candidate.train_reward.is_some() && !candidate.train_scores.is_empty())
        .collect::<Vec<_>>();
    let mut frontier = evaluated
        .iter()
        .filter_map(|candidate| {
            let dominated = evaluated.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && candidate_dominates(other, candidate)
            });
            if dominated {
                return None;
            }
            let train_reward = candidate.train_reward?;
            Some(FrontierMember {
                candidate_id: candidate.candidate_id.clone(),
                parent_id: candidate.parent_id.clone(),
                source: candidate.source.clone(),
                train_reward,
                heldout_reward: candidate.heldout_reward,
            })
        })
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    frontier
}

fn frontier_member_ids(frontier: &[FrontierMember]) -> BTreeSet<String> {
    frontier
        .iter()
        .map(|member| member.candidate_id.clone())
        .collect()
}

fn select_proposer_parent_candidate(
    candidates: &[CandidateRecord],
    train_rows: &[Value],
    objective_set: &ObjectiveSetRecord,
    selector: &GepaCandidateSelectorConfig,
    generation: usize,
    run_id: &str,
    fallback_idx: Option<usize>,
) -> Result<ParentSelectionDecision> {
    if candidates.is_empty() {
        return Err(OptimizerError::Invariant(
            "GEPA has no candidate to select as proposer parent".to_string(),
        ));
    }
    let fallback_idx = fallback_idx
        .filter(|idx| *idx < candidates.len())
        .or_else(|| {
            candidates
                .iter()
                .enumerate()
                .rev()
                .find(|(_, candidate)| candidate.train_reward.is_some())
                .map(|(idx, _)| idx)
        })
        .unwrap_or(0);
    let pareto_front = compute_candidate_pareto_front(candidates, train_rows, objective_set)?;
    let strategy = normalize_gepa_candidate_selector_name(&selector.name);
    if pareto_front.win_counts.is_empty() && strategy != "random" {
        let candidate = &candidates[fallback_idx];
        return Ok(ParentSelectionDecision {
            candidate_index: fallback_idx,
            metadata: json!({
                "strategy": strategy,
                "selector": candidate_selector_metadata(selector),
                "reason": "fallback_no_train_frontier_cells",
                "frontier_type": normalize_gepa_frontier_type(&objective_set.frontier_type),
                "candidate_id": candidate.candidate_id,
                "win_count": 0,
                "weight": 1.0,
            }),
        });
    }
    let search_visible = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| candidate.train_reward.map(|_| idx))
        .collect::<Vec<_>>();
    let mut members = pareto_front
        .win_counts
        .keys()
        .copied()
        .filter(|idx| search_visible.contains(idx))
        .collect::<Vec<_>>();
    members.sort_by(|left, right| {
        candidates[*left]
            .candidate_id
            .cmp(&candidates[*right].candidate_id)
    });
    let all_members = if search_visible.is_empty() {
        vec![fallback_idx]
    } else {
        search_visible
    };
    let (selected_idx, weights, reason) = match strategy.as_str() {
        "uniform_pareto" => {
            let weights = members.iter().map(|idx| (*idx, 1usize)).collect::<Vec<_>>();
            (
                select_weighted_parent(run_id, generation, candidates, &weights, fallback_idx),
                weights,
                "uniform_pareto".to_string(),
            )
        }
        "random" => {
            let weights = all_members
                .iter()
                .map(|idx| (*idx, 1usize))
                .collect::<Vec<_>>();
            (
                select_weighted_parent(run_id, generation, candidates, &weights, fallback_idx),
                weights,
                "uniform_all_candidates".to_string(),
            )
        }
        "current_best" => {
            let selected_idx = select_best_frontier_parent(candidates, &pareto_front.win_counts)
                .unwrap_or(fallback_idx);
            (
                selected_idx,
                vec![(
                    selected_idx,
                    std::cmp::max(
                        1usize,
                        pareto_front
                            .win_counts
                            .get(&selected_idx)
                            .copied()
                            .unwrap_or(0),
                    ),
                )],
                "current_best".to_string(),
            )
        }
        "top_k_pareto" => {
            let k = selector.k.unwrap_or(3);
            let mut top_members = members.clone();
            top_members.sort_by(|left, right| {
                pareto_front
                    .win_counts
                    .get(right)
                    .copied()
                    .unwrap_or(0)
                    .cmp(&pareto_front.win_counts.get(left).copied().unwrap_or(0))
                    .then_with(|| {
                        candidates[*left]
                            .candidate_id
                            .cmp(&candidates[*right].candidate_id)
                    })
            });
            top_members.truncate(k);
            top_members.sort_by(|left, right| {
                candidates[*left]
                    .candidate_id
                    .cmp(&candidates[*right].candidate_id)
            });
            let weights = top_members
                .iter()
                .map(|idx| (*idx, 1usize))
                .collect::<Vec<_>>();
            (
                select_weighted_parent(run_id, generation, candidates, &weights, fallback_idx),
                weights,
                format!("top_{k}_pareto"),
            )
        }
        "epsilon_greedy" => {
            let epsilon = selector.epsilon.unwrap_or(0.1);
            let explore =
                deterministic_selector_fraction(run_id, generation, candidates, &strategy)
                    < epsilon;
            if explore {
                let weights = all_members
                    .iter()
                    .map(|idx| (*idx, 1usize))
                    .collect::<Vec<_>>();
                (
                    select_weighted_parent(run_id, generation, candidates, &weights, fallback_idx),
                    weights,
                    "epsilon_explore".to_string(),
                )
            } else {
                let selected_idx =
                    select_best_frontier_parent(candidates, &pareto_front.win_counts)
                        .unwrap_or(fallback_idx);
                (
                    selected_idx,
                    vec![(
                        selected_idx,
                        std::cmp::max(
                            1usize,
                            pareto_front
                                .win_counts
                                .get(&selected_idx)
                                .copied()
                                .unwrap_or(0),
                        ),
                    )],
                    "epsilon_exploit".to_string(),
                )
            }
        }
        _ => {
            let weights = members
                .iter()
                .map(|idx| {
                    (
                        *idx,
                        std::cmp::max(1usize, *pareto_front.win_counts.get(idx).unwrap_or(&0)),
                    )
                })
                .collect::<Vec<_>>();
            (
                select_weighted_parent(run_id, generation, candidates, &weights, fallback_idx),
                weights,
                "pareto_weighted_frontier".to_string(),
            )
        }
    };
    let total_weight = weights
        .iter()
        .fold(0usize, |acc, (_, weight)| acc.saturating_add(*weight));
    let selected_raw_weight = weights
        .iter()
        .find(|(idx, _)| *idx == selected_idx)
        .map(|(_, weight)| *weight)
        .unwrap_or(1);
    let frontier_members = members
        .iter()
        .map(|idx| {
            let selection_weight = weights
                .iter()
                .find(|(candidate_idx, _)| candidate_idx == idx)
                .map(|(_, weight)| *weight)
                .unwrap_or(0);
            json!({
                "candidate_id": candidates[*idx].candidate_id,
                "win_count": pareto_front.win_counts.get(idx).copied().unwrap_or(0),
                "selection_weight": selection_weight,
            })
        })
        .collect::<Vec<_>>();
    Ok(ParentSelectionDecision {
        candidate_index: selected_idx,
        metadata: json!({
            "strategy": strategy,
            "selector": candidate_selector_metadata(selector),
            "reason": reason,
            "frontier_type": pareto_front.frontier_type,
            "candidate_id": candidates[selected_idx].candidate_id,
            "win_count": pareto_front.win_counts.get(&selected_idx).copied().unwrap_or(0),
            "weight": if total_weight == 0 { 1.0 } else { selected_raw_weight as f64 / total_weight as f64 },
            "raw_weight": selected_raw_weight,
            "total_weight": total_weight,
            "frontier_size": members.len(),
            "frontier": frontier_members,
            "cells": pareto_front.cells,
        }),
    })
}

fn normalize_gepa_candidate_selector_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "pareto" | "pareto_weighted" => "pareto_weighted".to_string(),
        "uniform_pareto" => "uniform_pareto".to_string(),
        "random" => "random".to_string(),
        "current_best" => "current_best".to_string(),
        "top_k_pareto" => "top_k_pareto".to_string(),
        "epsilon_greedy" => "epsilon_greedy".to_string(),
        _ => "pareto_weighted".to_string(),
    }
}

fn candidate_selector_metadata(selector: &GepaCandidateSelectorConfig) -> Value {
    json!({
        "name": normalize_gepa_candidate_selector_name(&selector.name),
        "configured_name": selector.name,
        "epsilon": selector.epsilon,
        "k": selector.k,
    })
}

fn select_weighted_parent(
    run_id: &str,
    generation: usize,
    candidates: &[CandidateRecord],
    weights: &[(usize, usize)],
    fallback_idx: usize,
) -> usize {
    let total_weight = weights
        .iter()
        .fold(0usize, |acc, (_, weight)| acc.saturating_add(*weight));
    let bucket = deterministic_weight_bucket(run_id, generation, candidates, weights, total_weight);
    let mut running = 0usize;
    let mut selected_idx = fallback_idx;
    for (idx, weight) in weights {
        running = running.saturating_add(*weight);
        if bucket < running {
            selected_idx = *idx;
            break;
        }
    }
    selected_idx
}

fn select_best_frontier_parent(
    candidates: &[CandidateRecord],
    win_counts: &BTreeMap<usize, usize>,
) -> Option<usize> {
    win_counts.keys().copied().max_by(|left, right| {
        win_counts
            .get(left)
            .copied()
            .unwrap_or(0)
            .cmp(&win_counts.get(right).copied().unwrap_or(0))
            .then_with(|| {
                candidates[*right]
                    .candidate_id
                    .cmp(&candidates[*left].candidate_id)
            })
    })
}

#[derive(Debug)]
struct CandidateParetoFront {
    frontier_type: String,
    win_counts: BTreeMap<usize, usize>,
    cells: Vec<Value>,
}

fn compute_candidate_pareto_front(
    candidates: &[CandidateRecord],
    train_rows: &[Value],
    objective_set: &ObjectiveSetRecord,
) -> Result<CandidateParetoFront> {
    let frontier_type = normalize_gepa_frontier_type(&objective_set.frontier_type);
    let train_example_ids = train_rows
        .iter()
        .map(row_example_id)
        .collect::<Result<BTreeSet<_>>>()?;
    let mut cells = match frontier_type.as_str() {
        "per_objective" => pareto_objective_cells(candidates, &train_example_ids, objective_set),
        "per_example_objective" => {
            pareto_example_objective_cells(candidates, &train_example_ids, objective_set)
        }
        _ => pareto_example_cells(candidates, &train_example_ids, objective_set),
    };
    if cells.is_empty() && frontier_type != "per_example" {
        cells = pareto_example_cells(candidates, &train_example_ids, objective_set);
    }
    let mut win_counts = BTreeMap::new();
    let mut cell_values = Vec::new();
    for cell in cells {
        *win_counts.entry(cell.candidate_index).or_default() += 1;
        cell_values.push(json!({
            "frontier_key": cell.frontier_key,
            "candidate_id": candidates[cell.candidate_index].candidate_id,
            "score": cell.score,
            "example_id": cell.example_id,
            "objective_id": cell.objective_id,
        }));
    }
    Ok(CandidateParetoFront {
        frontier_type,
        win_counts,
        cells: cell_values,
    })
}

#[derive(Clone, Debug)]
struct CandidateParetoCell {
    frontier_key: String,
    candidate_index: usize,
    score: f64,
    example_id: Option<String>,
    objective_id: Option<String>,
}

fn pareto_example_cells(
    candidates: &[CandidateRecord],
    train_example_ids: &BTreeSet<String>,
    objective_set: &ObjectiveSetRecord,
) -> Vec<CandidateParetoCell> {
    let mut winners: BTreeMap<String, CandidateParetoCell> = BTreeMap::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate, train_example_ids) {
            let Some(score) = frame_selection_score(frame, objective_set) else {
                continue;
            };
            upsert_pareto_cell(
                &mut winners,
                frame.example_id.clone(),
                CandidateParetoCell {
                    frontier_key: format!("example:{}", frame.example_id),
                    candidate_index: idx,
                    score,
                    example_id: Some(frame.example_id.clone()),
                    objective_id: None,
                },
                candidates,
            );
        }
    }
    winners.into_values().collect()
}

fn pareto_objective_cells(
    candidates: &[CandidateRecord],
    train_example_ids: &BTreeSet<String>,
    objective_set: &ObjectiveSetRecord,
) -> Vec<CandidateParetoCell> {
    let mut objective_scores: BTreeMap<(usize, String), (f64, usize)> = BTreeMap::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate, train_example_ids) {
            for objective in &objective_set.objectives {
                let Some(score) = frame_objective_score(frame, &objective.name) else {
                    continue;
                };
                let entry = objective_scores
                    .entry((idx, objective.name.clone()))
                    .or_insert((0.0, 0));
                entry.0 += score;
                entry.1 += 1;
            }
        }
    }
    let mut winners: BTreeMap<String, CandidateParetoCell> = BTreeMap::new();
    for ((idx, objective), (sum, count)) in objective_scores {
        if count == 0 {
            continue;
        }
        upsert_pareto_cell(
            &mut winners,
            objective.clone(),
            CandidateParetoCell {
                frontier_key: format!("objective:{objective}"),
                candidate_index: idx,
                score: (sum / count as f64)
                    * objective_set
                        .objectives
                        .iter()
                        .find(|spec| spec.name == objective)
                        .map(|spec| objective_direction_multiplier(&spec.direction))
                        .unwrap_or(1.0),
                example_id: None,
                objective_id: Some(objective),
            },
            candidates,
        );
    }
    winners.into_values().collect()
}

fn pareto_example_objective_cells(
    candidates: &[CandidateRecord],
    train_example_ids: &BTreeSet<String>,
    objective_set: &ObjectiveSetRecord,
) -> Vec<CandidateParetoCell> {
    let mut winners: BTreeMap<String, CandidateParetoCell> = BTreeMap::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate, train_example_ids) {
            for objective in &objective_set.objectives {
                let Some(score) = frame_objective_score(frame, &objective.name) else {
                    continue;
                };
                let key = format!("{}|{}", frame.example_id, objective.name);
                upsert_pareto_cell(
                    &mut winners,
                    key,
                    CandidateParetoCell {
                        frontier_key: format!(
                            "example_objective:{}|{}",
                            frame.example_id, objective.name
                        ),
                        candidate_index: idx,
                        score: score * objective_direction_multiplier(&objective.direction),
                        example_id: Some(frame.example_id.clone()),
                        objective_id: Some(objective.name.clone()),
                    },
                    candidates,
                );
            }
        }
    }
    winners.into_values().collect()
}

fn train_sensor_frames<'a>(
    candidate: &'a CandidateRecord,
    train_example_ids: &'a BTreeSet<String>,
) -> impl Iterator<Item = &'a SensorFrame> + 'a {
    candidate.train_scores.iter().filter_map(|score| {
        if !train_example_ids.is_empty() && !train_example_ids.contains(&score.example_id) {
            return None;
        }
        candidate.sensor_frames.iter().find(|frame| {
            frame.example_id == score.example_id
                && matches!(
                    frame.evaluation_stage.as_str(),
                    "seed_full_train" | "candidate_full_train"
                )
        })
    })
}

fn upsert_pareto_cell(
    winners: &mut BTreeMap<String, CandidateParetoCell>,
    key: String,
    challenger: CandidateParetoCell,
    candidates: &[CandidateRecord],
) {
    let should_replace = winners
        .get(&key)
        .map(|incumbent| {
            challenger.score > incumbent.score + f64::EPSILON
                || ((challenger.score - incumbent.score).abs() <= f64::EPSILON
                    && candidates[challenger.candidate_index].candidate_id
                        < candidates[incumbent.candidate_index].candidate_id)
        })
        .unwrap_or(true);
    if should_replace {
        winners.insert(key, challenger);
    }
}

fn frame_selection_score(frame: &SensorFrame, objective_set: &ObjectiveSetRecord) -> Option<f64> {
    let raw =
        frame_objective_score(frame, &objective_set.selection_objective).or(Some(frame.reward))?;
    Some(raw * selection_objective_direction(objective_set))
}

fn frame_objective_score(frame: &SensorFrame, objective: &str) -> Option<f64> {
    frame
        .objective_scores
        .iter()
        .find(|score| score.objective == objective)
        .map(|score| score.value)
}

fn normalize_gepa_frontier_type(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "per_objective" => "per_objective".to_string(),
        "per_example_objective" => "per_example_objective".to_string(),
        _ => "per_example".to_string(),
    }
}

fn deterministic_weight_bucket(
    run_id: &str,
    generation: usize,
    candidates: &[CandidateRecord],
    weights: &[(usize, usize)],
    total_weight: usize,
) -> usize {
    if total_weight == 0 {
        return 0;
    }
    let mut hasher = Sha256::new();
    hasher.update(run_id.as_bytes());
    hasher.update(b":parent:");
    hasher.update(generation.to_le_bytes());
    for (idx, weight) in weights {
        hasher.update(candidates[*idx].candidate_id.as_bytes());
        hasher.update(b"=");
        hasher.update(weight.to_le_bytes());
        hasher.update(b";");
    }
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_le_bytes(bytes) as usize) % total_weight
}

fn deterministic_selector_fraction(
    run_id: &str,
    generation: usize,
    candidates: &[CandidateRecord],
    strategy: &str,
) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(run_id.as_bytes());
    hasher.update(b":selector:");
    hasher.update(strategy.as_bytes());
    hasher.update(b":");
    hasher.update(generation.to_le_bytes());
    for candidate in candidates {
        hasher.update(candidate.candidate_id.as_bytes());
        hasher.update(b";");
    }
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes) as f64 / u64::MAX as f64
}

fn current_proposal_parent_idx(state: &GepaRunState) -> Result<usize> {
    if let Some(parent_id) = state.cursor.pipeline_state.parent_candidate_id.as_ref() {
        if let Some(parent_idx) = state
            .candidates
            .iter()
            .position(|candidate| &candidate.candidate_id == parent_id)
        {
            return Ok(parent_idx);
        }
    }
    state
        .best_idx
        .filter(|idx| *idx < state.candidates.len())
        .or(if state.candidates.is_empty() {
            None
        } else {
            Some(0)
        })
        .ok_or_else(|| {
            OptimizerError::Invariant("GEPA has no candidate to use as proposal parent".to_string())
        })
}

fn proposal_parent_idx(
    state: &GepaRunState,
    proposal: &ProposedCandidate,
    fallback_idx: usize,
) -> usize {
    proposal
        .parent_candidate_ids
        .iter()
        .find_map(|candidate_id| {
            state
                .candidates
                .iter()
                .position(|candidate| &candidate.candidate_id == candidate_id)
        })
        .unwrap_or(fallback_idx)
}

fn frontier_snapshot_value(
    candidates: &[CandidateRecord],
    train_rows: &[Value],
    best_idx: Option<usize>,
    generation: Option<usize>,
    reason: &str,
    changed_candidate_id: Option<&str>,
    previous_frontier_member_ids: Option<BTreeSet<String>>,
) -> Result<Value> {
    let frontier = frontier_members(candidates);
    let current_frontier_member_ids = frontier_member_ids(&frontier);
    let previous_frontier_size = previous_frontier_member_ids.as_ref().map(BTreeSet::len);
    let (frontier_added_candidate_ids, frontier_removed_candidate_ids): (Vec<String>, Vec<String>) =
        match previous_frontier_member_ids.as_ref() {
            Some(previous) => (
                current_frontier_member_ids
                    .difference(previous)
                    .cloned()
                    .collect(),
                previous
                    .difference(&current_frontier_member_ids)
                    .cloned()
                    .collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
    let train_seeds = train_rows.iter().map(row_seed).collect::<BTreeSet<_>>();
    let train_example_ids = train_rows
        .iter()
        .map(row_example_id)
        .collect::<Result<BTreeSet<_>>>()?;
    let best_candidate = best_idx.and_then(|idx| candidates.get(idx));
    let best_scores = best_candidate
        .map(|candidate| scores_by_example(&candidate.train_scores))
        .unwrap_or_default();
    let best_solved_examples = best_candidate
        .map(|candidate| solved_examples(&candidate.train_scores))
        .unwrap_or_default();
    let best_seed_count = best_candidate
        .map(|candidate| {
            candidate
                .train_scores
                .iter()
                .filter(|score| score_is_solved(score.reward))
                .map(|score| score.seed)
                .collect::<BTreeSet<_>>()
                .len()
        })
        .unwrap_or(0);

    let mut covered_frontier_seeds = BTreeSet::new();
    let mut covered_frontier_examples = BTreeSet::new();
    let mut member_rows = Vec::new();
    for member in &frontier {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.candidate_id == member.candidate_id)
        else {
            continue;
        };
        let evaluated_seed_set = candidate
            .train_scores
            .iter()
            .map(|score| score.seed)
            .collect::<BTreeSet<_>>();
        let solved_seed_set = candidate_solved_seeds(&candidate.train_scores);
        let example_scores = scores_by_example(&candidate.train_scores);
        let solved_example_scores = solved_examples(&candidate.train_scores);
        let covered_seeds = train_seeds
            .iter()
            .copied()
            .filter(|seed| solved_seed_set.contains(seed))
            .collect::<Vec<_>>();
        let missing_seeds = train_seeds
            .iter()
            .copied()
            .filter(|seed| !solved_seed_set.contains(seed))
            .collect::<Vec<_>>();
        let covered_examples = train_example_ids
            .iter()
            .filter(|example_id| solved_example_scores.contains_key(*example_id))
            .cloned()
            .collect::<Vec<_>>();
        let missing_examples = train_example_ids
            .iter()
            .filter(|example_id| !solved_example_scores.contains_key(*example_id))
            .cloned()
            .collect::<Vec<_>>();
        let evaluated_examples = train_example_ids
            .iter()
            .filter(|example_id| example_scores.contains_key(*example_id))
            .cloned()
            .collect::<Vec<_>>();
        covered_frontier_seeds.extend(covered_seeds.iter().copied());
        covered_frontier_examples.extend(covered_examples.iter().cloned());

        let mut wins_vs_best = 0usize;
        let mut losses_vs_best = 0usize;
        let mut ties_vs_best = 0usize;
        for example_id in &train_example_ids {
            let Some(candidate_reward) = example_scores.get(example_id) else {
                continue;
            };
            let Some(best_reward) = best_scores.get(example_id) else {
                continue;
            };
            if *candidate_reward > *best_reward + f64::EPSILON {
                wins_vs_best += 1;
            } else if *candidate_reward + f64::EPSILON < *best_reward {
                losses_vs_best += 1;
            } else {
                ties_vs_best += 1;
            }
        }

        member_rows.push(json!({
            "candidate_id": candidate.candidate_id.clone(),
            "parent_id": candidate.parent_id.clone(),
            "source": candidate.source.clone(),
            "status": candidate.status.clone(),
            "train_reward": candidate.train_reward,
            "heldout_reward": candidate.heldout_reward,
            "covered_seed_count": covered_seeds.len(),
            "missing_seed_count": missing_seeds.len(),
            "covered_seeds": covered_seeds,
            "missing_seeds": missing_seeds,
            "covered_example_count": covered_examples.len(),
            "missing_example_count": missing_examples.len(),
            "covered_examples": covered_examples,
            "missing_examples": missing_examples,
            "evaluated_seed_count": evaluated_seed_set.len(),
            "evaluated_example_count": evaluated_examples.len(),
            "coverage_semantics": "solved_reward_positive",
            "wins_vs_best": wins_vs_best,
            "losses_vs_best": losses_vs_best,
            "ties_vs_best": ties_vs_best,
            "is_best": best_candidate
                .map(|best| best.candidate_id == candidate.candidate_id)
                .unwrap_or(false),
            "is_changed": changed_candidate_id
                .map(|changed| changed == candidate.candidate_id)
                .unwrap_or(false),
        }));
    }

    Ok(json!({
        "generation": generation,
        "reason": reason,
        "changed_candidate_id": changed_candidate_id,
        "best_candidate_id": best_candidate.map(|candidate| candidate.candidate_id.clone()),
        "best_train_reward": best_candidate.and_then(|candidate| candidate.train_reward),
        "candidate_count": candidates.len(),
        "frontier_size": frontier.len(),
        "previous_frontier_size": previous_frontier_size,
        "frontier_size_delta": previous_frontier_size.map(|previous| frontier.len() as i64 - previous as i64),
        "frontier_added_count": frontier_added_candidate_ids.len(),
        "frontier_removed_count": frontier_removed_candidate_ids.len(),
        "frontier_added_candidate_ids": frontier_added_candidate_ids,
        "frontier_removed_candidate_ids": frontier_removed_candidate_ids,
        "train_row_count": train_rows.len(),
        "train_seed_count": train_seeds.len(),
        "train_seeds": train_seeds.iter().copied().collect::<Vec<_>>(),
        "covered_train_seed_count": covered_frontier_seeds.len(),
        "covered_train_seed_percent": coverage_percent(covered_frontier_seeds.len(), train_seeds.len()),
        "covered_train_seeds": covered_frontier_seeds.iter().copied().collect::<Vec<_>>(),
        "covered_train_example_count": covered_frontier_examples.len(),
        "covered_train_example_percent": coverage_percent(covered_frontier_examples.len(), train_example_ids.len()),
        "best_candidate_seed_coverage_percent": coverage_percent(best_seed_count, train_seeds.len()),
        "best_candidate_example_count": best_solved_examples.len(),
        "best_candidate_example_coverage_percent": coverage_percent(best_solved_examples.len(), train_example_ids.len()),
        "coverage_semantics": "solved_reward_positive",
        "frontier": frontier,
        "members": member_rows,
        "coverage": {
            "train_row_count": train_rows.len(),
            "train_seed_count": train_seeds.len(),
            "train_example_count": train_example_ids.len(),
            "covered_train_seed_count": covered_frontier_seeds.len(),
            "covered_train_seed_percent": coverage_percent(covered_frontier_seeds.len(), train_seeds.len()),
            "best_candidate_seed_count": best_seed_count,
            "best_candidate_seed_coverage_percent": coverage_percent(best_seed_count, train_seeds.len()),
            "best_candidate_example_count": best_solved_examples.len(),
            "best_candidate_example_coverage_percent": coverage_percent(best_solved_examples.len(), train_example_ids.len()),
            "covered_train_example_count": covered_frontier_examples.len(),
            "covered_train_example_percent": coverage_percent(covered_frontier_examples.len(), train_example_ids.len()),
            "coverage_semantics": "solved_reward_positive",
        },
    }))
}

fn score_is_solved(reward: f64) -> bool {
    reward > 0.0
}

fn candidate_solved_seeds(scores: &[RolloutScore]) -> BTreeSet<i64> {
    scores
        .iter()
        .filter(|score| score_is_solved(score.reward))
        .map(|score| score.seed)
        .collect()
}

fn solved_examples(scores: &[RolloutScore]) -> BTreeMap<String, f64> {
    scores
        .iter()
        .filter(|score| score_is_solved(score.reward))
        .map(|score| (score.example_id.clone(), score.reward))
        .collect()
}

fn coverage_percent(covered: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (covered as f64 / total as f64) * 100.0
    }
}

fn select_best_train_candidate(
    candidates: &[CandidateRecord],
    objective_set: &ObjectiveSetRecord,
    train_split: &str,
    train_rows: &[Value],
) -> Result<Option<usize>> {
    let mut best_idx = None;
    for (idx, candidate) in candidates.iter().enumerate() {
        if candidate.train_reward.is_none() {
            continue;
        };
        let Some(current_idx) = best_idx else {
            best_idx = Some(idx);
            continue;
        };
        let current = candidates.get(current_idx).ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "best train candidate index {current_idx} is outside candidate registry"
            ))
        })?;
        let challenger_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set,
            candidate,
            rows: train_rows,
            split: train_split,
            source_stages: &["seed_full_train", "candidate_full_train"],
            evaluation_stage: "train_parent_selection",
        })?;
        let incumbent_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set,
            candidate: current,
            rows: train_rows,
            split: train_split,
            source_stages: &["seed_full_train", "candidate_full_train"],
            evaluation_stage: "train_parent_selection",
        })?;
        let preference = compare_score_vectors(ScoreVectorPreferenceInput {
            objective_set,
            split: train_split,
            evaluation_stage: "train_parent_selection",
            challenger: &challenger_vector,
            incumbent: &incumbent_vector,
            accept_equal: false,
            acceptance_criterion: None,
            objective_acceptance: None,
            margin: 0.0,
        })?;
        let deterministic_tie_latest = preference.result == "tie" && idx > current_idx;
        if preference.preferred || deterministic_tie_latest {
            best_idx = Some(idx);
        }
    }
    Ok(best_idx)
}

fn select_best_heldout_candidate(input: HeldoutSelectionInput<'_>) -> Result<Option<usize>> {
    let HeldoutSelectionInput {
        candidates,
        evaluated_indices,
        objective_set,
        heldout_split,
        heldout_rows,
        train_split,
        train_rows,
        incumbent_idx,
    } = input;
    let mut best_idx = incumbent_idx.filter(|idx| {
        evaluated_indices.contains(idx)
            && candidates
                .get(*idx)
                .and_then(|candidate| candidate.heldout_reward)
                .is_some()
    });
    for idx in evaluated_indices.iter().copied() {
        let candidate = candidates.get(idx).ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "heldout candidate index {idx} is outside candidate registry"
            ))
        })?;
        if candidate.heldout_reward.is_none() {
            continue;
        }
        if best_idx == Some(idx) {
            continue;
        }
        let Some(current_idx) = best_idx else {
            best_idx = Some(idx);
            continue;
        };
        let current = candidates.get(current_idx).ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "best heldout candidate index {current_idx} is outside candidate registry"
            ))
        })?;
        let challenger_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set,
            candidate,
            rows: heldout_rows,
            split: heldout_split,
            source_stages: &["heldout"],
            evaluation_stage: "heldout_final_selection",
        })?;
        let incumbent_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set,
            candidate: current,
            rows: heldout_rows,
            split: heldout_split,
            source_stages: &["heldout"],
            evaluation_stage: "heldout_final_selection",
        })?;
        let preference = compare_score_vectors(ScoreVectorPreferenceInput {
            objective_set,
            split: heldout_split,
            evaluation_stage: "heldout_final_selection",
            challenger: &challenger_vector,
            incumbent: &incumbent_vector,
            accept_equal: false,
            acceptance_criterion: None,
            objective_acceptance: None,
            margin: 0.0,
        })?;
        let train_tiebreak_preferred = if preference.result == "tie" {
            let challenger_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set,
                candidate,
                rows: train_rows,
                split: train_split,
                source_stages: &["seed_full_train", "candidate_full_train"],
                evaluation_stage: "heldout_train_tiebreak",
            })?;
            let incumbent_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set,
                candidate: current,
                rows: train_rows,
                split: train_split,
                source_stages: &["seed_full_train", "candidate_full_train"],
                evaluation_stage: "heldout_train_tiebreak",
            })?;
            compare_score_vectors(ScoreVectorPreferenceInput {
                objective_set,
                split: train_split,
                evaluation_stage: "heldout_train_tiebreak",
                challenger: &challenger_train_vector,
                incumbent: &incumbent_train_vector,
                accept_equal: false,
                acceptance_criterion: None,
                objective_acceptance: None,
                margin: 0.0,
            })?
            .preferred
        } else {
            false
        };
        if preference.preferred || train_tiebreak_preferred {
            best_idx = Some(idx);
        }
    }
    Ok(best_idx)
}

fn score_chart_value(
    candidates: &[CandidateRecord],
    seed_idx: usize,
    best_idx: usize,
    chart_path: &Path,
) -> Value {
    let seed_candidate_id = candidates
        .get(seed_idx)
        .map(|candidate| candidate.candidate_id.clone())
        .unwrap_or_default();
    let best_candidate_id = candidates
        .get(best_idx)
        .map(|candidate| candidate.candidate_id.clone())
        .unwrap_or_default();
    let seed_heldout = candidates
        .get(seed_idx)
        .and_then(|candidate| candidate.heldout_reward);
    let mut rows = Vec::new();
    let mut candidate_prompt_diffs = Vec::new();
    let mut train_values = Vec::new();
    let mut heldout_values = Vec::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        let Some(train_reward) = candidate.train_reward else {
            continue;
        };
        let heldout_reward = candidate.heldout_reward;
        let lift_vs_seed = heldout_reward
            .zip(seed_heldout)
            .map(|(heldout, seed)| heldout - seed);
        train_values.push(train_reward);
        if let Some(heldout_reward) = heldout_reward {
            heldout_values.push(heldout_reward);
        }
        rows.push(json!({
            "index": idx,
            "candidate_id": candidate.candidate_id.clone(),
            "source": candidate.source.clone(),
            "status": candidate.status.clone(),
            "train_reward": train_reward,
            "heldout_reward": heldout_reward,
            "lift_vs_seed": lift_vs_seed,
            "is_seed": idx == seed_idx,
            "is_best": idx == best_idx,
        }));
        if idx != seed_idx {
            let diff = prompt_payload_diff(candidates.get(seed_idx), Some(candidate));
            if !diff.is_empty() {
                candidate_prompt_diffs.push(json!({
                    "index": idx,
                    "candidate_id": candidate.candidate_id.clone(),
                    "source": candidate.source.clone(),
                    "status": candidate.status.clone(),
                    "train_reward": train_reward,
                    "heldout_reward": heldout_reward,
                    "diff": diff,
                }));
            }
        }
    }
    json!({
        "chart_path": chart_path.display().to_string(),
        "seed_candidate_id": seed_candidate_id,
        "best_candidate_id": best_candidate_id,
        "baseline_to_best_diff": prompt_payload_diff(
            candidates.get(seed_idx),
            candidates.get(best_idx),
        ),
        "candidate_prompt_diffs": candidate_prompt_diffs,
        "train_values": train_values,
        "heldout_values": heldout_values,
        "candidates": rows,
    })
}

fn prompt_payload_diff(
    baseline: Option<&CandidateRecord>,
    best: Option<&CandidateRecord>,
) -> Vec<Value> {
    let Some(baseline) = baseline else {
        return Vec::new();
    };
    let Some(best) = best else {
        return Vec::new();
    };
    if baseline.candidate_id == best.candidate_id {
        return Vec::new();
    }
    let keys = baseline
        .payload
        .keys()
        .chain(best.payload.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    keys.into_iter()
        .filter_map(|module| {
            let before = baseline.payload.get(&module).cloned().unwrap_or_default();
            let after = best.payload.get(&module).cloned().unwrap_or_default();
            (before != after).then(|| {
                json!({
                    "module": module,
                    "before": before,
                    "after": after,
                })
            })
        })
        .collect()
}

fn render_score_chart_svg(run_id: &str, chart: &Value) -> String {
    let rows = chart
        .get("candidates")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut scores = rows
        .iter()
        .flat_map(|row| {
            [
                chart_row_f64(row, "train_reward"),
                chart_row_f64(row, "heldout_reward"),
            ]
        })
        .flatten()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if scores.is_empty() {
        scores.push(0.0);
        scores.push(1.0);
    }
    let max_score = scores.iter().copied().fold(1.0_f64, f64::max).max(1.0);
    let min_score = scores.iter().copied().fold(0.0_f64, f64::min).min(0.0);
    let score_span = (max_score - min_score).max(0.001);

    let width = 920.0;
    let height = 520.0;
    let left = 76.0;
    let right = 42.0;
    let top = 76.0;
    let bottom = 118.0;
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let n = rows.len().max(1);
    let x_at = |idx: usize| -> f64 {
        if n <= 1 {
            left + plot_width / 2.0
        } else {
            left + idx as f64 / (n - 1) as f64 * plot_width
        }
    };
    let y_at = |score: f64| -> f64 { top + (max_score - score) / score_span * plot_height };

    let mut train_points = String::new();
    let mut heldout_points = String::new();
    for (idx, row) in rows.iter().enumerate() {
        let x = x_at(idx);
        if let Some(score) = chart_row_f64(row, "train_reward") {
            let _ = write!(train_points, "{x:.1},{:.1} ", y_at(score));
        }
        if let Some(score) = chart_row_f64(row, "heldout_reward") {
            let _ = write!(heldout_points, "{x:.1},{:.1} ", y_at(score));
        }
    }

    let best_candidate = chart
        .get("best_candidate_id")
        .and_then(Value::as_str)
        .unwrap_or("-");
    let seed_candidate = chart
        .get("seed_candidate_id")
        .and_then(Value::as_str)
        .unwrap_or("-");
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}" role="img" aria-label="GEPA train and heldout score chart">"#
    );
    let _ = writeln!(
        svg,
        r##"<rect width="100%" height="100%" fill="#fbfaf7"/>"##
    );
    let _ = writeln!(
        svg,
        r##"<text x="{left:.0}" y="34" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="700" fill="#1d2327">GEPA train/heldout score chart</text>"##
    );
    let _ = writeln!(
        svg,
        r##"<text x="{left:.0}" y="56" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" fill="#5e666d">run={}  seed={}  best={}</text>"##,
        xml_escape(run_id),
        xml_escape(seed_candidate),
        xml_escape(best_candidate)
    );
    for tick in 0..=4 {
        let value = min_score + (score_span * tick as f64 / 4.0);
        let y = y_at(value);
        let _ = writeln!(
            svg,
            r##"<line x1="{left:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#e1ded8" stroke-width="1"/>"##,
            width - right
        );
        let _ = writeln!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" text-anchor="end" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" fill="#626a70">{value:.3}</text>"##,
            left - 12.0,
            y + 4.0
        );
    }
    let _ = writeln!(
        svg,
        r##"<line x1="{left:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#9aa0a6" stroke-width="1.2"/>"##,
        top + plot_height,
        width - right,
        top + plot_height
    );
    let _ = writeln!(
        svg,
        r##"<line x1="{left:.1}" y1="{top:.1}" x2="{left:.1}" y2="{:.1}" stroke="#9aa0a6" stroke-width="1.2"/>"##,
        top + plot_height
    );
    if !train_points.trim().is_empty() {
        let _ = writeln!(
            svg,
            r##"<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{}"/>"##,
            train_points.trim()
        );
    }
    if !heldout_points.trim().is_empty() {
        let _ = writeln!(
            svg,
            r##"<polyline fill="none" stroke="#d97706" stroke-width="2.5" points="{}"/>"##,
            heldout_points.trim()
        );
    }
    for (idx, row) in rows.iter().enumerate() {
        let x = x_at(idx);
        let is_seed = chart_row_bool(row, "is_seed");
        let is_best = chart_row_bool(row, "is_best");
        if let Some(score) = chart_row_f64(row, "train_reward") {
            let radius = if is_best { 5.4 } else { 4.0 };
            let _ = writeln!(
                svg,
                r##"<circle cx="{x:.1}" cy="{:.1}" r="{radius:.1}" fill="#2563eb" stroke="#ffffff" stroke-width="1.5"/>"##,
                y_at(score)
            );
        }
        if let Some(score) = chart_row_f64(row, "heldout_reward") {
            let radius = if is_best {
                6.2
            } else if is_seed {
                5.0
            } else {
                4.4
            };
            let stroke = if is_best { "#111827" } else { "#ffffff" };
            let _ = writeln!(
                svg,
                r##"<circle cx="{x:.1}" cy="{:.1}" r="{radius:.1}" fill="#d97706" stroke="{stroke}" stroke-width="1.7"/>"##,
                y_at(score)
            );
        }
        let label = chart_row_string(row, "index").unwrap_or_else(|| idx.to_string());
        let _ = writeln!(
            svg,
            r##"<text x="{x:.1}" y="{:.1}" text-anchor="middle" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" fill="#626a70">{}</text>"##,
            top + plot_height + 22.0,
            xml_escape(&label)
        );
    }
    let legend_x = left;
    let legend_y = height - 58.0;
    let _ = writeln!(
        svg,
        r##"<line x1="{legend_x:.1}" y1="{legend_y:.1}" x2="{:.1}" y2="{legend_y:.1}" stroke="#2563eb" stroke-width="3"/>"##,
        legend_x + 24.0
    );
    let _ = writeln!(
        svg,
        r##"<text x="{:.1}" y="{:.1}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#1f2937">train</text>"##,
        legend_x + 32.0,
        legend_y + 4.0
    );
    let _ = writeln!(
        svg,
        r##"<line x1="{:.1}" y1="{legend_y:.1}" x2="{:.1}" y2="{legend_y:.1}" stroke="#d97706" stroke-width="3"/>"##,
        legend_x + 94.0,
        legend_x + 118.0
    );
    let _ = writeln!(
        svg,
        r##"<text x="{:.1}" y="{:.1}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="#1f2937">heldout</text>"##,
        legend_x + 126.0,
        legend_y + 4.0
    );
    let _ = writeln!(
        svg,
        r##"<text x="{:.1}" y="{:.1}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" fill="#5e666d">Candidate order follows evaluation order. Larger heldout score selects the final best candidate.</text>"##,
        left,
        height - 28.0
    );
    svg.push_str("</svg>\n");
    svg
}

fn chart_row_f64(row: &Value, key: &str) -> Option<f64> {
    row.get(key).and_then(Value::as_f64)
}

fn chart_row_bool(row: &Value, key: &str) -> bool {
    row.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn chart_row_string(row: &Value, key: &str) -> Option<String> {
    row.get(key).and_then(|value| {
        value
            .as_str()
            .map(ToString::to_string)
            .or_else(|| value.as_u64().map(|number| number.to_string()))
            .or_else(|| value.as_i64().map(|number| number.to_string()))
    })
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn candidate_dominates(left: &CandidateRecord, right: &CandidateRecord) -> bool {
    let left_scores = scores_by_example(&left.train_scores);
    let right_scores = scores_by_example(&right.train_scores);
    if right_scores.is_empty() {
        return false;
    }
    let mut strictly_better = left_scores.len() > right_scores.len();
    for (example_id, right_reward) in &right_scores {
        let Some(left_reward) = left_scores.get(example_id) else {
            return false;
        };
        if *left_reward + f64::EPSILON < *right_reward {
            return false;
        }
        if *left_reward > *right_reward + f64::EPSILON {
            strictly_better = true;
        }
    }
    strictly_better
}

fn scores_by_example(scores: &[RolloutScore]) -> BTreeMap<String, f64> {
    scores
        .iter()
        .map(|score| (score.example_id.clone(), score.reward))
        .collect()
}

fn cost_budget_reached(config: &SynthOptimizerConfig, cost_usd: f64) -> bool {
    config.gepa.max_cost_usd > 0.0 && cost_usd >= config.gepa.max_cost_usd
}

fn train_rollout_budget_reached(config: &SynthOptimizerConfig, rollout_count: usize) -> bool {
    rollout_count >= config.gepa.train_rollout_limit()
}

fn budget_status(
    config: &SynthOptimizerConfig,
    rollout_count: usize,
    cost_usd: f64,
) -> &'static str {
    if train_rollout_budget_reached(config, rollout_count) {
        if config.gepa.split_rollout_budgets_enabled() {
            "train_rollout_budget_reached"
        } else {
            "rollout_budget_reached"
        }
    } else if cost_budget_reached(config, cost_usd) {
        "cost_budget_reached"
    } else {
        "within_budget"
    }
}

fn remaining_rollout_capacity(workspace: &WorkspaceStore, run_id: &str) -> Result<usize> {
    let ledger = workspace.budget_ledger_snapshot(run_id)?;
    Ok(ledger
        .remaining_rollouts()
        .map(u64_to_usize_saturating)
        .unwrap_or(usize::MAX))
}

fn remaining_train_rollout_capacity(
    workspace: &WorkspaceStore,
    config: &SynthOptimizerConfig,
    rollout_count: usize,
) -> Result<usize> {
    Ok(
        remaining_rollout_capacity(workspace, &config.run.run_id)?.min(
            config
                .gepa
                .train_rollout_limit()
                .saturating_sub(rollout_count),
        ),
    )
}

fn heldout_rollout_count(candidates: &[CandidateRecord]) -> usize {
    candidates
        .iter()
        .flat_map(|candidate| candidate.sensor_frames.iter())
        .filter(|frame| frame.evaluation_stage == "heldout")
        .count()
}

fn remaining_heldout_rollout_capacity(
    workspace: &WorkspaceStore,
    config: &SynthOptimizerConfig,
    candidates: &[CandidateRecord],
) -> Result<usize> {
    let global_remaining = remaining_rollout_capacity(workspace, &config.run.run_id)?;
    if config.gepa.split_rollout_budgets_enabled() {
        Ok(global_remaining.min(
            config
                .gepa
                .heldout_rollout_limit()
                .saturating_sub(heldout_rollout_count(candidates)),
        ))
    } else {
        Ok(global_remaining)
    }
}

fn next_rollout_budget_breach(
    workspace: &WorkspaceStore,
    config: &SynthOptimizerConfig,
) -> Result<Option<BudgetLimitBreach>> {
    let configured_limits = ConfiguredGepaRunLimits::from_config(config);
    let ledger = workspace.budget_ledger_snapshot(&config.run.run_id)?;
    Ok(ledger.breach_for_request(
        configured_limits
            .rollout_budget_estimate()
            .requested_budget(),
    ))
}

fn rollout_budget_exceeded_error(
    run_id: &str,
    limit: &str,
    requested: usize,
    available: usize,
) -> OptimizerError {
    OptimizerError::BudgetExceeded {
        run_id: run_id.to_string(),
        limit: limit.to_string(),
        requested: requested.to_string(),
        available: available.to_string(),
    }
}

fn rollout_budget_limit_name(config: &SynthOptimizerConfig) -> &'static str {
    if config.gepa.split_rollout_budgets_enabled() {
        "max_train_rollouts"
    } else {
        "max_total_rollouts"
    }
}

fn u64_to_usize_saturating(value: u64) -> usize {
    value.min(usize::MAX as u64) as usize
}

fn push_stopper_snapshot(
    records: &mut Vec<StopperStateRecord>,
    sequence_number: &mut u64,
    config: &SynthOptimizerConfig,
    snapshot: StopperSnapshot<'_>,
) {
    *sequence_number += 1;
    records.push(StopperStateRecord::from_input(StopperStateInput {
        sequence_number: *sequence_number,
        status: snapshot.status,
        reason: snapshot.reason,
        generation: snapshot.generation.map(|generation| generation as u64),
        candidate_id: snapshot.candidate_id,
        evaluation_stage: snapshot.evaluation_stage,
        rollout_count: snapshot.rollout_count as u64,
        max_total_rollouts: config.gepa.effective_max_total_rollouts() as u64,
        cost_usd: snapshot.cost_usd,
        max_cost_usd: config.gepa.max_cost_usd,
        metadata: snapshot.metadata,
    }));
}

fn record_checkpoint_snapshot(
    workspace: &mut WorkspaceStore,
    run_id: &str,
    sequence_number: &mut u64,
    state_machine: &OptimizerStateMachine,
    checkpoint: CheckpointSnapshot<'_>,
) -> Result<()> {
    *sequence_number += 1;
    let record = CheckpointRecord::from_input(CheckpointInput {
        sequence_number: *sequence_number,
        checkpoint_kind: checkpoint.checkpoint_kind,
        status: checkpoint.status,
        run_state: state_machine.state().as_str(),
        reason: checkpoint.reason,
        generation: checkpoint.generation.map(|generation| generation as u64),
        candidate_id: checkpoint.candidate_id,
        evaluation_stage: checkpoint.evaluation_stage,
        best_candidate_id: checkpoint.best_candidate_id,
        candidate_count: checkpoint.candidate_count as u64,
        frontier_count: checkpoint.frontier_count as u64,
        rollout_count: checkpoint.rollout_count as u64,
        cost_usd: checkpoint.cost_usd,
        usage: checkpoint.usage,
        snapshot: checkpoint.snapshot,
        metadata: checkpoint.metadata,
    });
    workspace.record_checkpoint(run_id, &record)
}

fn persist_gepa_cursor(
    workspace: &mut WorkspaceStore,
    config: &SynthOptimizerConfig,
    sequence_number: &mut u64,
    state: GepaCursorState<'_>,
    status: &str,
    reason: &str,
) -> Result<()> {
    *sequence_number += 1;
    let best_candidate_id = state
        .best_idx
        .and_then(|idx| state.candidates.get(idx))
        .map(|candidate| candidate.candidate_id.clone());
    let cursor = GepaCursor {
        schema_version: planner::GEPA_CURSOR_SCHEMA_VERSION.to_string(),
        run_id: config.run.run_id.clone(),
        phase: state.phase,
        generation: state.generation,
        proposal_index: state.proposal_index,
        proposal_queue: Value::Array(Vec::new()),
        heldout_candidate_index: 0,
        pending_job_id: state.pending_job_id,
        pending_effect_id: state.pending_effect_id,
        pending_reservation_ids: state.pending_reservation_ids,
        active_evaluation: state.active_evaluation,
        candidates: serde_json::to_value(state.candidates)?,
        best_candidate_id: best_candidate_id.clone(),
        rollout_task_id: Some(state.rollout_task_id.to_string()),
        rollout_count: state.rollout_count,
        cost_usd: state.total_cost,
        usage: serde_json::to_value(state.total_usage)?,
        usage_ledger: Value::Array(Vec::new()),
        stopper_states: Value::Array(Vec::new()),
        stopper_sequence: state.stopper_sequence,
        checkpoint_sequence: *sequence_number,
        train_rows: serde_json::to_value(state.train_rows)?,
        minibatch_rows: serde_json::to_value(state.minibatch_rows)?,
        reflection_rows: serde_json::to_value(state.reflection_rows)?,
        heldout_rows: serde_json::to_value(state.heldout_rows)?,
        program: serde_json::to_value(state.program)?,
        objective_set: serde_json::to_value(state.objective_set)?,
        state_history: serde_json::to_value(&state.state_machine.history)?,
        pipeline_state: planner::GepaAsyncPipelineCursorState::default(),
        terminal_summary: state.terminal_summary,
        error_summary: state.error_summary,
        metadata: Value::Object(state.metadata),
    };
    let cursor_value = serde_json::to_value(&cursor)?;
    let checkpoint = CheckpointRecord::from_input(CheckpointInput {
        sequence_number: *sequence_number,
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status,
        run_state: cursor.phase.as_str(),
        reason: Some(reason),
        generation: Some(cursor.generation as u64),
        candidate_id: best_candidate_id.as_deref(),
        evaluation_stage: Some(cursor.phase.as_str()),
        best_candidate_id: best_candidate_id.as_deref(),
        candidate_count: cursor.candidates.as_array().map(Vec::len).unwrap_or(0) as u64,
        frontier_count: frontier_members(state.candidates).len() as u64,
        rollout_count: cursor.rollout_count as u64,
        cost_usd: cursor.cost_usd,
        usage: cursor.usage.clone(),
        snapshot: cursor_value,
        metadata: Map::new(),
    });
    workspace.record_checkpoint(&config.run.run_id, &checkpoint)
}

fn checkpoint_snapshot_value(state: CheckpointSnapshotState<'_>) -> Value {
    json!({
        "run_id": state.config.run.run_id,
        "state": state.state_machine.state().as_str(),
        "state_history_count": state.state_machine.history.len(),
        "best_idx": state.best_idx,
        "best_candidate_id": state.best_idx.and_then(|idx| {
            state.candidates.get(idx).map(|candidate| candidate.candidate_id.clone())
        }),
        "candidate_count": state.candidates.len(),
        "candidates": state.candidates,
        "frontier": state.frontier,
        "rollout_count": state.rollout_count,
        "usage": state.total_usage,
        "cost_usd": state.total_cost,
        "max_total_rollouts": state.config.gepa.effective_max_total_rollouts(),
        "max_train_rollouts": state.config.gepa.train_rollout_limit(),
        "max_heldout_rollouts": state.config.gepa.heldout_rollout_limit(),
        "max_cost_usd": state.config.gepa.max_cost_usd,
    })
}

fn append_rollout_usage(records: &mut Vec<UsageLedgerRecord>, eval: &CandidateEvaluation) {
    records.extend(
        eval.sensor_frames
            .iter()
            .map(UsageLedgerRecord::from_sensor_frame),
    );
}

fn proposer_usage_record(
    config: &SynthOptimizerConfig,
    parent: &CandidateRecord,
    generation: usize,
    outcome: &ProposerOutcome,
) -> Result<UsageLedgerRecord> {
    let mut metadata = Map::new();
    metadata.insert("generation".to_string(), json!(generation));
    metadata.insert("proposal_count".to_string(), json!(outcome.proposals.len()));
    metadata.insert(
        "backend".to_string(),
        Value::String(outcome.backend.clone()),
    );
    metadata.insert(
        "warning_count".to_string(),
        json!(outcome.evidence_warnings.len()),
    );
    metadata.insert(
        "warnings".to_string(),
        json!(outcome.evidence_warnings.clone()),
    );
    if let Some(workspace) = &outcome.workspace {
        metadata.insert("workspace".to_string(), Value::String(workspace.clone()));
    }
    Ok(UsageLedgerRecord::from_input(UsageLedgerInput {
        boundary: "proposer.codex",
        source_type: "proposer_generation",
        source_id: &format!("generation_{generation:03}"),
        candidate_id: Some(&parent.candidate_id),
        evaluation_stage: Some("proposal"),
        model: config.proposer.model.as_deref(),
        provider: Some(&outcome.backend),
        call_count: outcome.usage.proposer_calls.max(1),
        usage: serde_json::to_value(&outcome.usage)?,
        cost_usd: outcome.cost_usd,
        metadata,
    }))
}

fn persist_candidate_snapshot(
    workspace: &mut WorkspaceStore,
    run_id: &str,
    candidate: &CandidateRecord,
) -> Result<()> {
    workspace.persist_candidate_registry(run_id, &[serde_json::to_value(candidate)?])
}

fn record_initial_platform_snapshots(
    workspace: &WorkspaceStore,
    config: &SynthOptimizerConfig,
    cache_mode: CacheMode,
    cache_namespace: &str,
    paths: &ArtifactPaths,
) -> Result<()> {
    let configured_limits = ConfiguredGepaRunLimits::from_config(config);
    let config_value = serde_json::to_value(config)?;
    let mut config_metadata = Map::new();
    config_metadata.insert("source".to_string(), json!("gepa_toml_resolved"));
    workspace.record_resolved_run_config(&ResolvedRunConfigRecord::from_input(
        ResolvedRunConfigInput {
            run_id: &config.run.run_id,
            algorithm_id: GEPA_ALGORITHM_ID,
            cache_mode: cache_mode.as_str(),
            cache_namespace,
            output_dir: &config.run.output_dir.display().to_string(),
            config: &config_value,
            metadata: config_metadata,
        },
    ))?;

    let mut limits_metadata = Map::new();
    limits_metadata.insert("algorithm_id".to_string(), json!(GEPA_ALGORITHM_ID));
    limits_metadata.insert(
        "run_dir".to_string(),
        json!(paths.run_dir.display().to_string()),
    );
    limits_metadata.insert(
        "budget_estimates".to_string(),
        json!(configured_limits.budget_estimates()),
    );
    workspace.record_run_limits(
        &configured_limits.to_run_limits_record(&config.run.run_id, limits_metadata),
    )?;
    Ok(())
}

struct DatasetSnapshotCall<'a> {
    run_id: &'a str,
    dataset_id: &'a str,
    split: &'a str,
    seeds: &'a [i64],
    filters: &'a Value,
    response: &'a DatasetRowsResponse,
    dataset_metadata: &'a Value,
}

fn record_dataset_snapshot(
    workspace: &WorkspaceStore,
    call: DatasetSnapshotCall<'_>,
) -> Result<()> {
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!("container.dataset_rows"));
    workspace.record_dataset_snapshot(&DatasetSnapshotRecord::from_input(DatasetSnapshotInput {
        run_id: call.run_id,
        dataset_id: call.dataset_id,
        split: call.split,
        seeds: call.seeds,
        filters: call.filters,
        rows: &call.response.rows,
        dataset_metadata: call.dataset_metadata.clone(),
        rows_metadata: Value::Object(call.response.metadata.clone()),
        metadata,
    }))
}

struct RuntimeEffectPlanInput<'a> {
    run_id: &'a str,
    effect_kind: &'a str,
    lane: &'a str,
    subject_type: &'a str,
    subject_id: &'a str,
    idempotency_key: &'a str,
    job_kind: OptimizerJobKind,
    candidate_id: Option<&'a str>,
    cache_key: Option<String>,
    budget_estimate: RuntimeEffectBudgetEstimate,
    payload: Value,
    dispatch_payload: runtime::RuntimeEffectDispatchPayload,
    metadata: Map<String, Value>,
}

struct RuntimeEffectCompletionInput<'a> {
    planned: &'a RuntimeEffectRecord,
    reservation: &'a BudgetReservationRecord,
    status: &'a str,
    cost_usd: f64,
    usage: &'a UsageTotals,
    rollout_count: u64,
    failure: Option<&'a FailurePayload>,
    metadata: Map<String, Value>,
}

fn record_runtime_effect_planned(
    workspace: &WorkspaceStore,
    input: RuntimeEffectPlanInput<'_>,
) -> Result<runtime::QueuedRuntimeEffect> {
    let limits = workspace.required_run_limits(input.run_id)?;
    input
        .budget_estimate
        .validate_for_limits(input.run_id, input.effect_kind, &limits)?;
    let mut effect = RuntimeEffectRecord::from_input(RuntimeEffectInput {
        run_id: input.run_id,
        effect_kind: input.effect_kind,
        lane: input.lane,
        status: "planned",
        subject_type: input.subject_type,
        subject_id: input.subject_id,
        idempotency_key: input.idempotency_key,
        cache_key: input.cache_key,
        job_id: None,
        budget_reservation_id: None,
        attempt: 1,
        failure_class: None,
        payload: input.payload,
        metadata: input.metadata.clone(),
    });
    let job_id = format!("effect:{}", effect.runtime_effect_id);
    if let Some(existing_job) = workspace.maybe_optimizer_job(input.run_id, &job_id)? {
        let effect_id = existing_job
            .payload
            .get("runtime_effect_id")
            .and_then(Value::as_str)
            .unwrap_or(&effect.runtime_effect_id)
            .to_string();
        let existing_effect = workspace.runtime_effect(input.run_id, &effect_id)?;
        let reservation_id = existing_job
            .payload
            .get("budget_reservation_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "existing GEPA runtime job {} has no budget_reservation_id",
                    existing_job.job_id
                ))
            })?;
        let reservation = workspace.budget_reservation(input.run_id, reservation_id)?;
        return Ok(runtime::QueuedRuntimeEffect {
            effect: existing_effect,
            reservation,
            job: existing_job,
            dispatch: input.dispatch_payload,
        });
    }
    let requested_budget = input.budget_estimate.requested_budget();
    workspace.record_runtime_effect(&effect)?;
    let ledger = workspace.budget_ledger_snapshot(input.run_id)?;
    if let Some(breach) = ledger.breach_for_request(requested_budget) {
        let mut metadata = input.metadata.clone();
        metadata.insert("limit".to_string(), json!(breach.limit.clone()));
        metadata.insert("requested".to_string(), json!(breach.requested.clone()));
        metadata.insert("available".to_string(), json!(breach.available.clone()));
        workspace.record_runtime_effect_admission(&RuntimeEffectAdmissionRecord::from_input(
            RuntimeEffectAdmissionInput {
                run_id: input.run_id,
                runtime_effect_id: &effect.runtime_effect_id,
                effect_kind: input.effect_kind,
                lane: input.lane,
                subject_type: input.subject_type,
                subject_id: input.subject_id,
                idempotency_key: input.idempotency_key,
                status: "rejected",
                rejection_reason: Some("budget_limit_exceeded".to_string()),
                max_cost_usd: input.budget_estimate.max_cost_usd,
                max_prompt_tokens: input.budget_estimate.max_prompt_tokens,
                max_completion_tokens: input.budget_estimate.max_completion_tokens,
                max_total_tokens: input.budget_estimate.max_total_tokens,
                max_rollouts: input.budget_estimate.max_rollouts,
                max_wall_seconds: input.budget_estimate.max_wall_seconds,
                ledger,
                metadata: metadata.clone(),
            },
        ))?;
        effect = RuntimeEffectRecord::from_input(RuntimeEffectInput {
            run_id: input.run_id,
            effect_kind: input.effect_kind,
            lane: input.lane,
            status: "rejected",
            subject_type: input.subject_type,
            subject_id: input.subject_id,
            idempotency_key: input.idempotency_key,
            cache_key: effect.cache_key.clone(),
            job_id: None,
            budget_reservation_id: None,
            attempt: effect.attempt,
            failure_class: Some("budget_exceeded".to_string()),
            payload: effect.payload.clone(),
            metadata,
        });
        workspace.record_runtime_effect(&effect)?;
        return Err(budget_exceeded_error(input.run_id, &breach));
    }
    workspace.record_runtime_effect_admission(&RuntimeEffectAdmissionRecord::from_input(
        RuntimeEffectAdmissionInput {
            run_id: input.run_id,
            runtime_effect_id: &effect.runtime_effect_id,
            effect_kind: input.effect_kind,
            lane: input.lane,
            subject_type: input.subject_type,
            subject_id: input.subject_id,
            idempotency_key: input.idempotency_key,
            status: "admitted",
            rejection_reason: None,
            max_cost_usd: input.budget_estimate.max_cost_usd,
            max_prompt_tokens: input.budget_estimate.max_prompt_tokens,
            max_completion_tokens: input.budget_estimate.max_completion_tokens,
            max_total_tokens: input.budget_estimate.max_total_tokens,
            max_rollouts: input.budget_estimate.max_rollouts,
            max_wall_seconds: input.budget_estimate.max_wall_seconds,
            ledger,
            metadata: input.metadata.clone(),
        },
    ))?;
    let reservation = BudgetReservationRecord::from_input(BudgetReservationInput {
        run_id: input.run_id,
        runtime_effect_id: &effect.runtime_effect_id,
        status: "reserved",
        max_cost_usd: input.budget_estimate.max_cost_usd,
        max_prompt_tokens: input.budget_estimate.max_prompt_tokens,
        max_completion_tokens: input.budget_estimate.max_completion_tokens,
        max_total_tokens: input.budget_estimate.max_total_tokens,
        max_rollouts: input.budget_estimate.max_rollouts,
        max_wall_seconds: input.budget_estimate.max_wall_seconds,
        metadata: input.metadata,
    });
    workspace.record_budget_reservation(&reservation)?;
    record_runtime_effect_job(
        workspace,
        RuntimeEffectJobInput {
            job_id: &job_id,
            run_id: input.run_id,
            kind: input.job_kind.clone(),
            status: OptimizerJobStatus::Pending,
            candidate_id: input.candidate_id,
            effect: &effect,
            reservation: Some(&reservation),
            dispatch_payload: Some(&input.dispatch_payload),
            queue_state: "queued",
            failure: None,
        },
    )?;
    effect.budget_reservation_id = Some(reservation.budget_reservation_id.clone());
    effect.job_id = Some(job_id.clone());
    workspace.record_runtime_effect(&effect)?;
    let job = workspace.optimizer_job(input.run_id, &job_id)?;
    Ok(runtime::QueuedRuntimeEffect {
        effect,
        reservation,
        job,
        dispatch: input.dispatch_payload,
    })
}

struct RuntimeEffectJobInput<'a> {
    job_id: &'a str,
    run_id: &'a str,
    kind: OptimizerJobKind,
    status: OptimizerJobStatus,
    candidate_id: Option<&'a str>,
    effect: &'a RuntimeEffectRecord,
    reservation: Option<&'a BudgetReservationRecord>,
    dispatch_payload: Option<&'a runtime::RuntimeEffectDispatchPayload>,
    queue_state: &'a str,
    failure: Option<&'a FailurePayload>,
}

fn record_runtime_effect_job(
    workspace: &WorkspaceStore,
    input: RuntimeEffectJobInput<'_>,
) -> Result<()> {
    let mut job = OptimizerJob::new(input.job_id, input.run_id, input.kind);
    job.status = input.status;
    job.candidate_id = input.candidate_id.map(str::to_string);
    if matches!(job.kind, OptimizerJobKind::Rollout) {
        job.retry_policy = RetryPolicy {
            max_attempts: 3,
            backoff_seconds: 2,
            retryable_failure_types: vec![
                "synth_optimizer_http_error".to_string(),
                "synth_optimizer_container_error".to_string(),
                "synth_optimizer_failed".to_string(),
            ],
        };
    }
    job.attempt = if matches!(job.status, OptimizerJobStatus::Pending) {
        input.effect.attempt.saturating_sub(1)
    } else {
        input.effect.attempt
    };
    if !matches!(job.status, OptimizerJobStatus::Pending) {
        if let Some(existing) = workspace.maybe_optimizer_job(input.run_id, input.job_id)? {
            job.lease_id = existing.lease_id;
            job.worker_id = existing.worker_id;
            job.leased_at = existing.leased_at;
            job.lease_expires_at = existing.lease_expires_at;
            job.heartbeat_at = existing.heartbeat_at;
            job.next_retry_at = existing.next_retry_at;
            job.retry_policy = existing.retry_policy;
            job.payload = existing.payload;
        }
    }
    if let Some(dispatch_payload) = input.dispatch_payload {
        job.payload = serde_json::to_value(dispatch_payload)?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                OptimizerError::Invariant(
                    "GEPA runtime dispatch payload is not an object".to_string(),
                )
            })?;
    }
    job.payload.insert(
        "runtime_effect_id".to_string(),
        json!(input.effect.runtime_effect_id),
    );
    job.payload
        .insert("effect_kind".to_string(), json!(input.effect.effect_kind));
    job.payload
        .insert("lane".to_string(), json!(input.effect.lane));
    job.payload
        .insert("subject_type".to_string(), json!(input.effect.subject_type));
    job.payload
        .insert("subject_id".to_string(), json!(input.effect.subject_id));
    job.payload.insert(
        "idempotency_key".to_string(),
        json!(input.effect.idempotency_key),
    );
    if let Some(cache_key) = input.effect.cache_key.as_ref() {
        job.payload
            .insert("cache_key".to_string(), json!(cache_key));
    }
    if let Some(reservation) = input.reservation {
        job.payload.insert(
            "budget_reservation_id".to_string(),
            json!(reservation.budget_reservation_id),
        );
    }
    job.payload
        .insert("effect_payload".to_string(), input.effect.payload.clone());
    job.payload
        .insert("queue_state".to_string(), json!(input.queue_state));
    job.failure = input.failure.cloned();
    workspace.record_optimizer_job(&job)
}

fn budget_exceeded_error(run_id: &str, breach: &BudgetLimitBreach) -> OptimizerError {
    OptimizerError::BudgetExceeded {
        run_id: run_id.to_string(),
        limit: breach.limit.clone(),
        requested: breach.requested.clone(),
        available: breach.available.clone(),
    }
}

fn record_runtime_effect_completed(
    workspace: &WorkspaceStore,
    input: RuntimeEffectCompletionInput<'_>,
) -> Result<()> {
    let mut payload = input.planned.payload.clone();
    if let Some(object) = payload.as_object_mut() {
        object.insert("completion_status".to_string(), json!(input.status));
        if let Some(failure) = input.failure {
            object.insert("failure".to_string(), serde_json::to_value(failure)?);
        }
    }
    let mut metadata = input.metadata.clone();
    if let Some(failure) = input.failure {
        metadata.insert("failure".to_string(), serde_json::to_value(failure)?);
    }
    let completed = RuntimeEffectRecord::from_input(RuntimeEffectInput {
        run_id: &input.planned.run_id,
        effect_kind: &input.planned.effect_kind,
        lane: &input.planned.lane,
        status: input.status,
        subject_type: &input.planned.subject_type,
        subject_id: &input.planned.subject_id,
        idempotency_key: &input.planned.idempotency_key,
        cache_key: input.planned.cache_key.clone(),
        job_id: input.planned.job_id.clone(),
        budget_reservation_id: Some(input.reservation.budget_reservation_id.clone()),
        attempt: input.planned.attempt,
        failure_class: input
            .failure
            .map(|failure| failure.failure_class().to_string()),
        payload,
        metadata: metadata.clone(),
    });
    workspace.record_runtime_effect(&completed)?;
    let mut reservation_update = input.reservation.clone();
    reservation_update.status = if input.status == "completed" {
        "committed".to_string()
    } else {
        input.status.to_string()
    };
    workspace.record_budget_reservation(&reservation_update)?;
    let committed_wall_seconds = input.reservation.max_wall_seconds.unwrap_or(0);
    let commit = BudgetCommitRecord::from_input(BudgetCommitInput {
        run_id: &input.planned.run_id,
        runtime_effect_id: &input.planned.runtime_effect_id,
        budget_reservation_id: &input.reservation.budget_reservation_id,
        cost_usd: input.cost_usd,
        prompt_tokens: input.usage.prompt_tokens,
        completion_tokens: input.usage.completion_tokens,
        total_tokens: input.usage.total_tokens,
        rollout_count: input.rollout_count,
        wall_seconds: committed_wall_seconds,
        metadata: metadata.clone(),
    });
    workspace.record_budget_commit(&commit)?;
    workspace.record_budget_release(&budget_release_for_completion(
        input.reservation,
        &commit,
        input.status,
        metadata,
    ))?;
    if let Some(job_id) = input.planned.job_id.as_deref() {
        record_runtime_effect_job(
            workspace,
            RuntimeEffectJobInput {
                job_id,
                run_id: &input.planned.run_id,
                kind: runtime_effect_job_kind(input.planned),
                status: optimizer_job_status_from_effect_status(input.status),
                candidate_id: runtime_effect_candidate_id(input.planned).as_deref(),
                effect: input.planned,
                reservation: Some(input.reservation),
                dispatch_payload: None,
                queue_state: input.status,
                failure: input.failure,
            },
        )?;
    }
    let ledger = workspace.budget_ledger_snapshot(&input.planned.run_id)?;
    if let Some(breach) = ledger.exceeded_limit() {
        return Err(budget_exceeded_error(&input.planned.run_id, &breach));
    }
    Ok(())
}

fn budget_release_for_completion(
    reservation: &BudgetReservationRecord,
    commit: &BudgetCommitRecord,
    status: &str,
    metadata: Map<String, Value>,
) -> BudgetReleaseRecord {
    let reserved = reservation.reserved_budget();
    let committed = commit.committed_budget();
    BudgetReleaseRecord::from_input(BudgetReleaseInput {
        run_id: &reservation.run_id,
        runtime_effect_id: &reservation.runtime_effect_id,
        budget_reservation_id: &reservation.budget_reservation_id,
        release_reason: if status == "completed" {
            "committed_unused_budget"
        } else {
            status
        },
        released_cost_usd: (reserved.cost_usd - committed.cost_usd).max(0.0),
        released_prompt_tokens: reserved
            .prompt_tokens
            .saturating_sub(committed.prompt_tokens),
        released_completion_tokens: reserved
            .completion_tokens
            .saturating_sub(committed.completion_tokens),
        released_total_tokens: reserved.total_tokens.saturating_sub(committed.total_tokens),
        released_rollouts: reserved.rollouts.saturating_sub(committed.rollouts),
        released_wall_seconds: reserved.wall_seconds.saturating_sub(committed.wall_seconds),
        metadata,
    })
}

fn record_runtime_effect_failed(
    workspace: &WorkspaceStore,
    planned: &RuntimeEffectRecord,
    reservation: &BudgetReservationRecord,
    error: &OptimizerError,
    mut metadata: Map<String, Value>,
) -> Result<()> {
    let failure = FailurePayload::from_optimizer_error(error);
    metadata.insert("error_code".to_string(), json!(error.error_code()));
    record_runtime_effect_completed(
        workspace,
        RuntimeEffectCompletionInput {
            planned,
            reservation,
            status: "failed",
            cost_usd: 0.0,
            usage: &UsageTotals::default(),
            rollout_count: 0,
            failure: Some(&failure),
            metadata,
        },
    )
}

fn fail_runtime_effect_and_return<T>(
    workspace: &WorkspaceStore,
    planned: &RuntimeEffectRecord,
    reservation: &BudgetReservationRecord,
    error: OptimizerError,
    phase: &str,
) -> Result<T> {
    let mut metadata = Map::new();
    metadata.insert("failure_phase".to_string(), json!(phase));
    record_runtime_effect_failed(workspace, planned, reservation, &error, metadata)?;
    Err(error)
}

fn runtime_effect_job_kind(effect: &RuntimeEffectRecord) -> OptimizerJobKind {
    match effect.effect_kind.as_str() {
        "candidate_proposal" => OptimizerJobKind::Proposer,
        "container_rollout" => OptimizerJobKind::Rollout,
        _ => OptimizerJobKind::Checkpoint,
    }
}

fn optimizer_job_status_from_effect_status(status: &str) -> OptimizerJobStatus {
    match status {
        "completed" => OptimizerJobStatus::Completed,
        "cancelled" | "canceled" => OptimizerJobStatus::Cancelled,
        "expired" => OptimizerJobStatus::Expired,
        "running" => OptimizerJobStatus::Running,
        "planned" | "reserved" => OptimizerJobStatus::Pending,
        _ => OptimizerJobStatus::Failed,
    }
}

fn runtime_effect_candidate_id(effect: &RuntimeEffectRecord) -> Option<String> {
    effect
        .payload
        .get("candidate_id")
        .or_else(|| effect.payload.get("parent_candidate_id"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn check_cancelled(cancellation: Option<&GepaCancellationSource>) -> Result<()> {
    let Some(cancellation) = cancellation else {
        return Ok(());
    };
    let store = WorkspaceStore::open_existing(&cancellation.service_db_path)?;
    let status = store.run_request_status(&cancellation.request_id)?;
    if status.as_deref() == Some("cancelled") {
        return Err(OptimizerError::Cancelled {
            request_id: cancellation.request_id.clone(),
        });
    }
    if let Some(lease_id) = cancellation.lease_id.as_deref() {
        if store
            .heartbeat_run_request(
                &cancellation.request_id,
                lease_id,
                cancellation.lease_seconds,
            )?
            .is_none()
        {
            let status = store.run_request_status(&cancellation.request_id)?;
            if status.as_deref() == Some("cancelled") {
                return Err(OptimizerError::Cancelled {
                    request_id: cancellation.request_id.clone(),
                });
            }
            return Err(OptimizerError::Invariant(format!(
                "run request {} lost service lease {} during execution",
                cancellation.request_id, lease_id
            )));
        }
    }
    Ok(())
}

fn evaluate_candidate(call: EvaluationCall<'_>) -> Result<CandidateEvaluation> {
    let configured_limits = ConfiguredGepaRunLimits::from_config(call.config);
    let mut reward_sum = 0.0;
    let mut usage = UsageTotals::default();
    let mut cost_usd = 0.0;
    let mut scores = Vec::new();
    let mut sensor_frames = Vec::new();
    for row in call.rows {
        check_cancelled(call.cancellation)?;
        let seed = row.get("seed").and_then(Value::as_i64).unwrap_or(0);
        let overlay = CandidateOverlay {
            candidate: PromptCandidatePayload::from_map(call.candidate.payload.clone()),
            metadata: Map::new(),
        };
        let request = json!({
            "submission_mode": rollout_submission_mode_for_request(call.config),
            "task_id": call.task_id,
            "seed": seed,
            "candidate_id": call.candidate.candidate_id,
            "candidate": overlay.candidate.to_value(),
            "candidate_overlay": overlay,
            "policy": call.config.policy,
            "dataset_row": row,
        });
        let mut cache_metadata = Map::new();
        cache_metadata.insert(
            "candidate_id".to_string(),
            json!(call.candidate.candidate_id),
        );
        cache_metadata.insert("evaluation_stage".to_string(), json!(call.stage));
        let example_id = row_example_id(row)?;
        cache_metadata.insert("example_id".to_string(), json!(example_id));
        cache_metadata.insert("task_id".to_string(), json!(call.task_id));
        let rollout_namespace = format!("{}:container.rollout", call.cache_namespace);
        let planned_cache_key = RequestCache::cache_key_with_profile(
            &rollout_namespace,
            &request,
            ROLLOUT_CACHE_PROFILE,
        );
        let planned_effect_key = RequestCache::cache_key_with_profile(
            &rollout_namespace,
            &json!({"stage": call.stage, "request": request}),
            ROLLOUT_CACHE_PROFILE,
        );
        let mut effect_metadata = cache_metadata.clone();
        effect_metadata.insert("algorithm_id".to_string(), json!(GEPA_ALGORITHM_ID));
        let dispatch_payload =
            runtime::RuntimeEffectDispatchPayload::rollout(runtime::RuntimeRolloutDispatchInput {
                cache_namespace: rollout_namespace.clone(),
                cache_profile: ROLLOUT_CACHE_PROFILE.to_string(),
                cache_metadata: cache_metadata.clone(),
                request: request.clone(),
                candidate_id: call.candidate.candidate_id.clone(),
                stage: call.stage.to_string(),
                example_id: example_id.clone(),
                task_id: call.task_id.to_string(),
            });
        let queued_effect = record_runtime_effect_planned(
            call.workspace,
            RuntimeEffectPlanInput {
                run_id: &call.config.run.run_id,
                effect_kind: "container_rollout",
                lane: "rollout",
                subject_type: "candidate_example",
                subject_id: &format!("{}:{example_id}", call.candidate.candidate_id),
                idempotency_key: &planned_effect_key,
                job_kind: OptimizerJobKind::Rollout,
                candidate_id: Some(&call.candidate.candidate_id),
                cache_key: Some(planned_cache_key.clone()),
                budget_estimate: configured_limits.rollout_budget_estimate(),
                payload: json!({
                    "candidate_id": call.candidate.candidate_id,
                    "example_id": example_id,
                    "stage": call.stage,
                    "task_id": call.task_id,
                }),
                dispatch_payload,
                metadata: effect_metadata,
            },
        )?;
        let rollout_call = {
            match runtime::execute_one_pending_optimizer_job_from_run_workspace(
                call.workspace,
                call.cache,
                call.config,
                call.client,
                &queued_effect.effect.run_id,
                &queued_effect.job.job_id,
                runtime::RuntimeEffectExecutorConfig::inline_default(),
            )? {
                runtime::RuntimeEffectOutcome::Rollout(outcome) => outcome,
                runtime::RuntimeEffectOutcome::Proposer(_) => {
                    return Err(OptimizerError::Invariant(format!(
                        "rollout runtime effect returned proposer outcome job_id={}",
                        queued_effect.job.job_id
                    )));
                }
                runtime::RuntimeEffectOutcome::RolloutBatch(_) => {
                    return Err(OptimizerError::Invariant(format!(
                        "single rollout runtime effect returned batch outcome job_id={}",
                        queued_effect.job.job_id
                    )));
                }
            }
        };
        let response = rollout_call.response.clone();
        let typed_response = rollout_call.typed_response.clone();
        let reward = rollout_call.reward;
        reward_sum += reward;
        let mut sensor_frame = SensorFrame::from_rollout_response(
            &call.candidate.candidate_id,
            row,
            call.stage,
            &response,
        )?;
        align_sensor_frame_objectives(&mut sensor_frame, call.objective_set, reward);
        attach_rollout_trace_artifact(call.paths, &call.config.run.run_id, &mut sensor_frame)?;
        let objective_scores = serde_json::to_value(&sensor_frame.objective_scores)?;
        let materialization =
            rollout_materialization_identity(call.program, call.candidate, call.objective_set);
        let candidate_payload_value = serde_json::to_value(&call.candidate.payload)?;
        let platform_cache_key = Some(rollout_call.cache_key.clone());
        let mut materialization_metadata = Map::new();
        materialization_metadata.insert("cache_hit".to_string(), json!(rollout_call.cache_hit));
        materialization_metadata.insert(
            "rollout_status".to_string(),
            json!(sensor_frame.status.clone()),
        );
        materialization_metadata.insert(
            "rollout_id".to_string(),
            sensor_frame
                .rollout_id
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        call.workspace.record_materialization(
            &call.config.run.run_id,
            &MaterializationRecord::from_input(MaterializationRecordInput {
                candidate_id: &call.candidate.candidate_id,
                candidate_payload: &candidate_payload_value,
                example: row,
                request: &request,
                example_id: &example_id,
                seed,
                split: &sensor_frame.split,
                evaluation_stage: call.stage,
                task_id: call.task_id,
                materialization: materialization.clone(),
                status: "materialized",
                platform_cache_key: platform_cache_key.clone(),
                metadata: materialization_metadata,
            }),
        )?;
        call.workspace.record_evaluation_cache(
            &call.config.run.run_id,
            &EvaluationCacheRecord::from_input(EvaluationCacheRecordInput {
                candidate_payload: &candidate_payload_value,
                example: row,
                request: &request,
                example_id: &example_id,
                materialization,
                source_rollout_id: typed_response
                    .rollout_id
                    .clone()
                    .or_else(|| sensor_frame.rollout_id.clone()),
                reward,
                objective_scores,
                actionable_side_info: sensor_frame
                    .actionable_side_info
                    .clone()
                    .unwrap_or_else(|| json!({})),
                usage: sensor_frame.usage.clone(),
                trace_ref: sensor_frame
                    .trace_digest
                    .as_ref()
                    .map(|digest| format!("trace_sha256:{}", digest.sha256)),
                status: &sensor_frame.status,
                cache_hit: rollout_call.cache_hit,
                platform_cache_key,
                rollout_payload: &response,
                metadata: Map::new(),
            }),
        )?;
        scores.push(RolloutScore {
            example_id,
            seed,
            reward,
        });
        sensor_frames.push(sensor_frame);
        usage.merge(&rollout_call.usage);
        cost_usd += rollout_call.cost_usd;
    }
    let rollout_count = call.rows.len();
    Ok(CandidateEvaluation {
        average_reward: if rollout_count == 0 {
            0.0
        } else {
            reward_sum / rollout_count as f64
        },
        rollout_count,
        usage,
        cost_usd,
        scores,
        sensor_frames,
    })
}

fn propose_candidates(call: ProposerCall<'_>) -> Result<ProposerOutcome> {
    let configured_limits = ConfiguredGepaRunLimits::from_config(call.config);
    let workspace_dir = call
        .paths
        .run_dir
        .join("proposer_workspaces")
        .join(format!("generation_{:03}", call.generation));
    let request = json!({
        "backend": call.config.proposer.backend,
        "execution_mode": call.config.proposer.execution_mode,
        "model": call.config.proposer.model,
        "generation": call.generation,
        "parent": call.parent,
        "candidates": call.candidates,
        "program": call.program,
        "seed_pool_rows": call.seed_pool_rows,
        "workspace_root": call.paths.run_dir,
        "run_artifact_dir": call.paths.run_dir,
        "proposal_artifact_dir": workspace_dir,
        "lever_manifest": LeverManifest::from_prompt_program(call.program),
        "frontier_summary": proposer_frontier_summary(
            call.candidates,
            &call.seed_pool_rows
                .get("pareto_eval")
                .and_then(|value| value.get("rows"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            None,
        )?,
        "minibatch_failures": proposer_minibatch_failures(call.candidates),
        "rollout_trace_artifact_refs": proposer_rollout_trace_artifact_refs(call.candidates),
        "merge_evidence_artifacts": proposer_merge_evidence_artifacts(call.paths)?,
        "target_modules": call.config.candidate.target_modules,
        "proposal_count": call.config.gepa.proposals_per_generation,
    });
    let mut cache_metadata = Map::new();
    cache_metadata.insert("backend".to_string(), json!(&call.config.proposer.backend));
    cache_metadata.insert("generation".to_string(), json!(call.generation));
    cache_metadata.insert(
        "parent_candidate_id".to_string(),
        json!(&call.parent.candidate_id),
    );
    cache_metadata.insert(
        "proposal_count".to_string(),
        json!(call.config.gepa.proposals_per_generation),
    );
    let proposer_namespace = format!("{}:proposer.codex", call.cache_namespace);
    let planned_cache_key =
        RequestCache::cache_key_with_profile(&proposer_namespace, &request, PROPOSER_CACHE_PROFILE);
    let mut effect_metadata = cache_metadata.clone();
    effect_metadata.insert("algorithm_id".to_string(), json!(GEPA_ALGORITHM_ID));
    let dispatch_payload = runtime::RuntimeEffectDispatchPayload::proposer(
        proposer_namespace.clone(),
        PROPOSER_CACHE_PROFILE,
        cache_metadata.clone(),
        request.clone(),
        call.generation,
        call.parent.candidate_id.clone(),
        workspace_dir.display().to_string(),
    );
    let queued_effect = record_runtime_effect_planned(
        call.workspace,
        RuntimeEffectPlanInput {
            run_id: &call.config.run.run_id,
            effect_kind: "candidate_proposal",
            lane: "proposer",
            subject_type: "generation",
            subject_id: &format!("generation_{:03}", call.generation),
            idempotency_key: &planned_cache_key,
            job_kind: OptimizerJobKind::Proposer,
            candidate_id: Some(&call.parent.candidate_id),
            cache_key: Some(planned_cache_key.clone()),
            budget_estimate: configured_limits.proposer_budget_estimate(),
            payload: json!({
                "generation": call.generation,
                "parent_candidate_id": call.parent.candidate_id,
                "backend": call.config.proposer.backend,
            }),
            dispatch_payload,
            metadata: effect_metadata,
        },
    )?;
    let proposer_runtime_outcome = {
        match runtime::execute_one_pending_optimizer_job_from_run_workspace(
            call.workspace,
            call.cache,
            call.config,
            call.client,
            &queued_effect.effect.run_id,
            &queued_effect.job.job_id,
            runtime::RuntimeEffectExecutorConfig::inline_default(),
        )? {
            runtime::RuntimeEffectOutcome::Proposer(outcome) => outcome,
            runtime::RuntimeEffectOutcome::Rollout(_) => {
                return Err(OptimizerError::Invariant(format!(
                    "proposer runtime effect returned rollout outcome job_id={}",
                    queued_effect.job.job_id
                )));
            }
            runtime::RuntimeEffectOutcome::RolloutBatch(_) => {
                return Err(OptimizerError::Invariant(format!(
                    "proposer runtime effect returned rollout batch outcome job_id={}",
                    queued_effect.job.job_id
                )));
            }
        }
    };
    Ok(ProposerOutcome {
        proposals: proposer_runtime_outcome.proposals,
        usage: proposer_runtime_outcome.usage,
        cost_usd: proposer_runtime_outcome.cost_usd,
        backend: proposer_runtime_outcome.backend,
        workspace: proposer_runtime_outcome.workspace,
        evidence_warnings: proposer_runtime_outcome.evidence_warnings,
    })
}

fn run_proposer(
    config: &SynthOptimizerConfig,
    program: &PromptProgram,
    parent: &CandidateRecord,
    candidates: &[CandidateRecord],
    generation: usize,
    seed_pool_rows: Value,
    workspace_dir: std::path::PathBuf,
) -> Result<Value> {
    match config.proposer.backend.as_str() {
        "codex_app_server" => {
            codex_app_server::run_codex_app_server_proposer(codex_app_server::CodexProposerInput {
                config,
                program,
                parent,
                candidates,
                generation,
                seed_pool_rows,
                workspace_dir,
            })
            .map_err(|error| {
                eprintln!("[gepa-proposer] codex_app_server proposer failed: {error}");
                error
            })
        }
        "local_process_json" => Err(OptimizerError::Config(
            "unsupported proposer.backend \"local_process_json\"; GEPA proposer work must use codex_app_server workspace-backed proposing".to_string(),
        )),
        backend => Err(OptimizerError::Config(format!(
            "unsupported proposer.backend {backend:?}; expected codex_app_server"
        ))),
    }
}

fn cached_call(
    cache: &mut RequestCache,
    namespace: &str,
    request: &Value,
    live: impl FnOnce() -> Result<Value>,
) -> Result<Value> {
    Ok(cached_call_with_access(cache, namespace, request, live)?.value)
}

fn cached_call_with_access(
    cache: &mut RequestCache,
    namespace: &str,
    request: &Value,
    live: impl FnOnce() -> Result<Value>,
) -> Result<CachedCallOutcome> {
    let key = RequestCache::cache_key(namespace, request);
    if let Some(value) = cache.get_or_miss(namespace, &key)? {
        return Ok(CachedCallOutcome {
            value,
            cache_key: key,
            cache_hit: true,
        });
    }
    let response = live()?;
    cache.put(namespace, &key, request, &response)?;
    Ok(CachedCallOutcome {
        value: response,
        cache_key: key,
        cache_hit: false,
    })
}

fn cached_profiled_call_with_access(
    cache: &mut RequestCache,
    namespace: &str,
    request: &Value,
    profile: &str,
    metadata: Map<String, Value>,
    live: impl FnOnce() -> Result<Value>,
) -> Result<CachedCallOutcome> {
    if let Some(entry) = cache.find_equivalent(namespace, request, profile)? {
        return Ok(CachedCallOutcome {
            value: entry.response,
            cache_key: entry.cache_key,
            cache_hit: true,
        });
    }
    let key = RequestCache::cache_key_with_profile(namespace, request, profile);
    let response = live()?;
    cache.put_with_metadata(namespace, &key, request, &response, profile, metadata)?;
    Ok(CachedCallOutcome {
        value: response,
        cache_key: key,
        cache_hit: false,
    })
}

fn transition_run(
    workspace: &WorkspaceStore,
    events: &mut EventWriter,
    state_machine: &mut OptimizerStateMachine,
    to: OptimizerRunState,
    trigger: OptimizerTransitionTrigger,
    message: &str,
    details: Value,
) -> Result<()> {
    let details = details.as_object().cloned().unwrap_or_default();
    let transition = state_machine.transition(to, trigger, message, details)?;
    events.emit(
        "optimizer.state.transitioned",
        message,
        serde_json::to_value(&transition)?,
    )?;
    workspace.record_state_transition(state_machine.history.len(), &transition)
}

struct FailedGepaRunInput<'a> {
    workspace: &'a mut WorkspaceStore,
    events: &'a mut EventWriter,
    state_machine: &'a mut OptimizerStateMachine,
    paths: &'a ArtifactPaths,
    registry: &'a RunRegistry,
    cache: &'a mut RequestCache,
    config: &'a SynthOptimizerConfig,
    cache_mode: CacheMode,
    cache_namespace: &'a str,
    best_candidate_id: Option<&'a str>,
    total_cost: f64,
    total_usage: &'a UsageTotals,
    usage_ledger: &'a [UsageLedgerRecord],
    stopper_states: &'a [StopperStateRecord],
    message: &'a str,
    details: Value,
}

fn fail_gepa_run_and_return<T>(input: FailedGepaRunInput<'_>, error: OptimizerError) -> Result<T> {
    let failure = FailurePayload::from_optimizer_error(&error);
    let mut details = input.details.as_object().cloned().unwrap_or_default();
    details.insert("error_code".to_string(), json!(error.error_code()));
    details.insert("failure".to_string(), serde_json::to_value(&failure)?);
    if let Some(best_candidate_id) = input.best_candidate_id {
        details.insert("best_candidate_id".to_string(), json!(best_candidate_id));
    }

    let (terminal_state, trigger, terminal_event_type, terminal_message) = match &error {
        OptimizerError::Cancelled { .. } => (
            OptimizerRunState::Cancelled,
            OptimizerTransitionTrigger::CancelRequested,
            "gepa.run.cancelled",
            "GEPA run cancelled",
        ),
        _ => (
            OptimizerRunState::Failed,
            OptimizerTransitionTrigger::FailureRaised,
            "gepa.run.failed",
            "GEPA run failed",
        ),
    };
    let usage_value = serde_json::to_value(input.total_usage)?;
    input
        .workspace
        .record_usage_ledger(&input.config.run.run_id, input.usage_ledger)?;
    input
        .workspace
        .record_stopper_states(&input.config.run.run_id, input.stopper_states)?;
    let cache_profile_record = CacheProfileRecord::from_profile(input.cache.profile()?);
    let cache_access_log = input.cache.access_log().to_vec();
    let cache_profile = serde_json::to_value(&cache_profile_record.profile)?;
    input
        .paths
        .write_json(&input.paths.cache_profile_path, &cache_profile)?;
    input.workspace.record_cache_profile(
        &input.config.run.run_id,
        &cache_profile_record,
        &cache_access_log,
    )?;
    let manifest_best_candidate_id = input.best_candidate_id.unwrap_or("unavailable");
    let mut failure_manifest = json!({
        "schema_version": "gepa_failure_manifest.v1",
        "run_id": input.config.run.run_id,
        "status": terminal_state.as_str(),
        "best_candidate_id": manifest_best_candidate_id,
        "cost_usd": input.total_cost,
        "usage": usage_value,
        "failure": serde_json::to_value(&failure)?,
        "state_history": serde_json::to_value(&input.state_machine.history)?,
        "event_feed_path": input.paths.event_feed_path.display().to_string(),
        "normalized_event_feed_path": input.paths.normalized_event_feed_path.display().to_string(),
        "cache_profile_path": input.paths.cache_profile_path.display().to_string(),
        "workspace_db_path": input.paths.workspace_db_path.display().to_string(),
    });
    input
        .paths
        .write_json(&input.paths.manifest_path, &failure_manifest)?;
    input.workspace.record_manifest(
        &input.config.run.run_id,
        &input.paths.manifest_path,
        manifest_best_candidate_id,
        input.total_cost,
        &usage_value,
        &failure_manifest,
    )?;
    if !input.state_machine.state().is_terminal() {
        transition_run(
            input.workspace,
            input.events,
            input.state_machine,
            terminal_state,
            trigger,
            terminal_message,
            Value::Object(details.clone()),
        )?;
    }
    if let Some(object) = failure_manifest.as_object_mut() {
        object.insert(
            "state_history".to_string(),
            serde_json::to_value(&input.state_machine.history)?,
        );
    }
    input
        .paths
        .write_json(&input.paths.manifest_path, &failure_manifest)?;
    input.workspace.record_manifest(
        &input.config.run.run_id,
        &input.paths.manifest_path,
        manifest_best_candidate_id,
        input.total_cost,
        &usage_value,
        &failure_manifest,
    )?;
    let phase = if matches!(terminal_state, OptimizerRunState::Cancelled) {
        GepaCursorPhase::Cancelled
    } else {
        GepaCursorPhase::Failed
    };
    let mut cursor = GepaCursor::terminal(
        input.config.run.run_id.clone(),
        phase,
        json!({
            "status": terminal_state.as_str(),
            "best_candidate_id": input.best_candidate_id,
            "cost_usd": input.total_cost,
            "usage": usage_value,
            "failure": serde_json::to_value(&failure)?,
        }),
    );
    cursor.best_candidate_id = input.best_candidate_id.map(str::to_string);
    cursor.cost_usd = input.total_cost;
    cursor.usage = usage_value.clone();
    cursor.state_history = serde_json::to_value(&input.state_machine.history)?;
    cursor.error_summary = Some(failure_manifest.clone());
    let sequence_number = input
        .workspace
        .checkpoint_history(&input.config.run.run_id, None)?
        .last()
        .map(|record| record.sequence_number + 1)
        .unwrap_or(1);
    cursor.checkpoint_sequence = sequence_number;
    let cursor_value = serde_json::to_value(&cursor)?;
    let checkpoint = CheckpointRecord::from_input(CheckpointInput {
        sequence_number,
        checkpoint_kind: GEPA_CURSOR_CHECKPOINT_KIND,
        status: terminal_state.as_str(),
        run_state: terminal_state.as_str(),
        reason: Some(input.message),
        generation: Some(cursor.generation as u64),
        candidate_id: input.best_candidate_id,
        evaluation_stage: Some(cursor.phase.as_str()),
        best_candidate_id: input.best_candidate_id,
        candidate_count: 0,
        frontier_count: 0,
        rollout_count: cursor.rollout_count as u64,
        cost_usd: cursor.cost_usd,
        usage: cursor.usage.clone(),
        snapshot: cursor_value,
        metadata: Map::new(),
    });
    input
        .workspace
        .record_checkpoint(&input.config.run.run_id, &checkpoint)?;
    input.events.emit(
        terminal_event_type,
        input.message,
        json!({
            "run_id": input.config.run.run_id,
            "state": input.state_machine.state().as_str(),
            "cost_usd": input.total_cost,
            "usage": usage_value,
            "error_code": error.error_code(),
            "failure": serde_json::to_value(&failure)?,
        }),
    )?;
    input.events.flush()?;
    input
        .workspace
        .record_event_stream(&input.config.run.run_id, input.events.records())?;
    normalize_event_feed(
        &input.paths.event_feed_path,
        &input.paths.normalized_event_feed_path,
        &input.paths.run_dir,
    )?;
    if matches!(terminal_state, OptimizerRunState::Cancelled) {
        input.workspace.record_run_cancelled_result(
            &input.config.run.run_id,
            input.best_candidate_id,
            input.total_cost,
            &usage_value,
        )?;
        input.registry.append(&RunRegistryEntry::cancelled(
            input.paths,
            input.config,
            input.cache_mode,
            input.cache_namespace,
            input.total_cost,
            usage_value,
        ))?;
    } else {
        input.workspace.record_run_failed(
            &input.config.run.run_id,
            input.best_candidate_id,
            input.total_cost,
            &usage_value,
        )?;
        input.registry.append(&RunRegistryEntry::failed(
            input.paths,
            input.config,
            input.cache_mode,
            input.cache_namespace,
            input.total_cost,
            usage_value,
        ))?;
    }
    Err(error)
}

fn candidate_id(payload: &BTreeMap<String, String>) -> String {
    let value = serde_json::to_value(payload).unwrap_or(Value::Null);
    let mut digest = Sha256::new();
    digest.update(synth_optimizer_platform::cache::stable_json(&value).as_bytes());
    let hex = format!("{:x}", digest.finalize());
    format!("gepa_{}", &hex[..12])
}
