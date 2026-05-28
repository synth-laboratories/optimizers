use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub const GEPA_CURSOR_SCHEMA_VERSION: &str = "gepa_cursor.v1";
pub const GEPA_CURSOR_CHECKPOINT_KIND: &str = "gepa_cursor";

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GepaCursorPhase {
    #[default]
    Initializing,
    SeedFullTrain,
    GenerationStart,
    ProposerWaiting,
    CandidateMinibatch,
    CandidateFullTrain,
    Heldout,
    Finalizing,
    Completed,
    Failed,
    Cancelled,
}

impl GepaCursorPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Initializing => "initializing",
            Self::SeedFullTrain => "seed_full_train",
            Self::GenerationStart => "generation_start",
            Self::ProposerWaiting => "proposer_waiting",
            Self::CandidateMinibatch => "candidate_minibatch",
            Self::CandidateFullTrain => "candidate_full_train",
            Self::Heldout => "heldout",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaCursor {
    #[serde(default = "default_gepa_cursor_schema_version")]
    pub schema_version: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub phase: GepaCursorPhase,
    #[serde(default)]
    pub generation: usize,
    #[serde(default)]
    pub proposal_index: usize,
    #[serde(default)]
    pub proposal_queue: Value,
    #[serde(default)]
    pub heldout_candidate_index: usize,
    #[serde(default)]
    pub pending_job_id: Option<String>,
    #[serde(default)]
    pub pending_effect_id: Option<String>,
    #[serde(default)]
    pub pending_reservation_ids: Vec<String>,
    #[serde(default)]
    pub active_evaluation: Option<Value>,
    #[serde(default)]
    pub candidates: Value,
    #[serde(default)]
    pub best_candidate_id: Option<String>,
    #[serde(default)]
    pub rollout_task_id: Option<String>,
    #[serde(default)]
    pub rollout_count: usize,
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub usage_ledger: Value,
    #[serde(default)]
    pub stopper_states: Value,
    #[serde(default)]
    pub stopper_sequence: u64,
    #[serde(default)]
    pub checkpoint_sequence: u64,
    #[serde(default)]
    pub train_rows: Value,
    #[serde(default)]
    pub minibatch_rows: Value,
    #[serde(default)]
    pub reflection_rows: Value,
    #[serde(default)]
    pub heldout_rows: Value,
    #[serde(default)]
    pub program: Value,
    #[serde(default)]
    pub objective_set: Value,
    #[serde(default)]
    pub state_history: Value,
    #[serde(default)]
    pub pipeline_state: GepaAsyncPipelineCursorState,
    #[serde(default)]
    pub terminal_summary: Option<Value>,
    #[serde(default)]
    pub error_summary: Option<Value>,
    #[serde(default)]
    pub metadata: Value,
}

impl GepaCursor {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            schema_version: GEPA_CURSOR_SCHEMA_VERSION.to_string(),
            run_id: run_id.into(),
            phase: GepaCursorPhase::Initializing,
            generation: 0,
            proposal_index: 0,
            proposal_queue: Value::Array(Vec::new()),
            heldout_candidate_index: 0,
            pending_job_id: None,
            pending_effect_id: None,
            pending_reservation_ids: Vec::new(),
            active_evaluation: None,
            candidates: Value::Array(Vec::new()),
            best_candidate_id: None,
            rollout_task_id: None,
            rollout_count: 0,
            cost_usd: 0.0,
            usage: Value::Null,
            usage_ledger: Value::Array(Vec::new()),
            stopper_states: Value::Array(Vec::new()),
            stopper_sequence: 0,
            checkpoint_sequence: 0,
            train_rows: Value::Array(Vec::new()),
            minibatch_rows: Value::Array(Vec::new()),
            reflection_rows: Value::Array(Vec::new()),
            heldout_rows: Value::Array(Vec::new()),
            program: Value::Null,
            objective_set: Value::Null,
            state_history: Value::Array(Vec::new()),
            pipeline_state: GepaAsyncPipelineCursorState::default(),
            terminal_summary: None,
            error_summary: None,
            metadata: Value::Null,
        }
    }

    pub fn terminal(run_id: impl Into<String>, phase: GepaCursorPhase, metadata: Value) -> Self {
        let mut cursor = Self::new(run_id);
        cursor.phase = phase;
        cursor.metadata = metadata;
        cursor
    }
}

fn default_gepa_cursor_schema_version() -> String {
    GEPA_CURSOR_SCHEMA_VERSION.to_string()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GepaAsyncPipelineCursorState {
    #[serde(default)]
    pub pool_version: u64,
    #[serde(default)]
    pub parent_pool_version: Option<u64>,
    #[serde(default)]
    pub parent_candidate_id: Option<String>,
    #[serde(default)]
    pub in_flight_candidate_count: usize,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub propose_queue: Vec<GepaAsyncLaneWorkItem>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub rollout_queue: Vec<GepaAsyncLaneWorkItem>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub evaluate_queue: Vec<GepaAsyncLaneWorkItem>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub lane_leases: BTreeMap<String, GepaAsyncLaneLease>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub pending_job_ids: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub pending_effect_ids: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub candidate_partials: BTreeMap<String, GepaAsyncCandidatePartial>,
    #[serde(default)]
    pub adaptive_rollout_concurrency: GepaAdaptiveRolloutConcurrencyState,
    #[serde(default)]
    pub terminal_readiness: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GepaAdaptiveRolloutConcurrencyState {
    #[serde(default)]
    pub initialized: bool,
    #[serde(default)]
    pub current_limit: usize,
    #[serde(default)]
    pub successes_since_adjustment: usize,
    #[serde(default)]
    pub completed_rollouts: usize,
    #[serde(default)]
    pub overload_count: usize,
    #[serde(default)]
    pub last_adjustment: Option<GepaAdaptiveRolloutConcurrencyAdjustment>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub adjustments: Vec<GepaAdaptiveRolloutConcurrencyAdjustment>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GepaAdaptiveRolloutConcurrencyAdjustment {
    #[serde(default)]
    pub direction: String,
    #[serde(default)]
    pub old_limit: usize,
    #[serde(default)]
    pub new_limit: usize,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub completed_rollouts: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GepaAsyncLaneWorkItem {
    #[serde(default)]
    pub item_id: String,
    #[serde(default)]
    pub lane: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub generation: usize,
    #[serde(default)]
    pub proposal_index: usize,
    #[serde(default)]
    pub parent_candidate_id: Option<String>,
    #[serde(default)]
    pub parent_pool_version: u64,
    #[serde(default)]
    pub current_pool_version: Option<u64>,
    #[serde(default)]
    pub stale_gap: Option<u64>,
    #[serde(default)]
    pub candidate_ids: Vec<String>,
    #[serde(default)]
    pub partial_id: Option<String>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub effect_id: Option<String>,
    #[serde(default)]
    pub reservation_ids: Vec<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GepaAsyncLaneLease {
    #[serde(default)]
    pub lease_id: String,
    #[serde(default)]
    pub lane: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub generation: usize,
    #[serde(default)]
    pub parent_pool_version: u64,
    #[serde(default)]
    pub partial_id: Option<String>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub effect_id: Option<String>,
    #[serde(default)]
    pub reservation_ids: Vec<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GepaAsyncCandidatePartial {
    #[serde(default)]
    pub partial_id: String,
    #[serde(default)]
    pub lane: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub generation: usize,
    #[serde(default)]
    pub parent_pool_version: u64,
    #[serde(default)]
    pub parent_candidate_id: Option<String>,
    #[serde(default)]
    pub candidate_ids: Vec<String>,
    #[serde(default)]
    pub active_evaluation: Option<Value>,
    #[serde(default)]
    pub proposal_queue: Value,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub rollout_chunks: BTreeMap<String, GepaRolloutChunkPartial>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GepaRolloutChunkPartial {
    #[serde(default)]
    pub chunk_id: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub generation: usize,
    #[serde(default)]
    pub candidate_ids: Vec<String>,
    #[serde(default)]
    pub row_ids: Vec<String>,
    #[serde(default)]
    pub first_row_index: usize,
    #[serde(default)]
    pub row_count: usize,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub effect_id: Option<String>,
    #[serde(default)]
    pub reservation_ids: Vec<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub folded: bool,
    #[serde(default)]
    pub attempt: u32,
    #[serde(default)]
    pub metadata: Value,
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + DeserializeOwned,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum GepaTickAction {
    Noop,
    ClaimRunRequest { request_id: String, run_id: String },
    StartRunRequest { request_id: String, run_id: String },
    PlanRuntimeJob { run_id: String, job_id: String },
    ExecuteRuntimeJob { run_id: String, job_id: String },
    ConsumeRuntimeOutcome { run_id: String, job_id: String },
    SetupRun { run_id: String },
    CheckpointRun { run_id: String, phase: String },
    TerminalizeRun { run_id: String, status: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaPlannerStep {
    pub cursor: GepaCursor,
    pub action: GepaTickAction,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GepaTickOutcome {
    pub request: Option<synth_optimizer_platform::WorkspaceRunRequestStatus>,
    pub result: Option<synth_optimizer_platform::GepaRunResult>,
    pub action: GepaTickAction,
    pub terminal: bool,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct GepaPlanner;

impl GepaPlanner {
    pub fn next_step(cursor: GepaCursor) -> GepaPlannerStep {
        let action = if cursor.phase.is_terminal() {
            GepaTickAction::TerminalizeRun {
                run_id: cursor.run_id.clone(),
                status: format!("{:?}", cursor.phase).to_ascii_lowercase(),
            }
        } else if let Some(job_id) = cursor.pending_job_id.clone() {
            GepaTickAction::ExecuteRuntimeJob {
                run_id: cursor.run_id.clone(),
                job_id,
            }
        } else {
            GepaTickAction::Noop
        };
        GepaPlannerStep {
            cursor,
            action,
            message: "planner step selected".to_string(),
        }
    }
}
