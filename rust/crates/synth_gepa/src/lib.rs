use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use synth_optimizer_platform::limits::{
    BudgetCommitInput, BudgetCommitRecord, BudgetLimitBreach, BudgetReleaseInput,
    BudgetReleaseRecord, BudgetReservationInput, BudgetReservationRecord, RunLimitPolicy,
    RuntimeEffectAdmissionInput, RuntimeEffectAdmissionRecord, RuntimeEffectBudgetEstimate,
};
use synth_optimizer_platform::{
    budget_limit_engine_input, normalize_event_feed, stable_json_hash, task_identity,
    write_run_storage_report, ArtifactPaths, ArtifactRef, CacheMode, CacheProfileRecord,
    CandidateOverlay, CheckpointInput, CheckpointRecord, CheckpointSummaryRecord,
    ConfiguredGepaRunLimits, ContainerClient, ContainerContractSnapshotInput,
    ContainerContractSnapshotRecord, DiskBudget, EvaluationCacheRecord, EvaluationCacheRecordInput,
    EventStreamRecord, EventWriter, EvidenceFrame, FailurePayload, ForecastConfidence,
    GepaBatchSamplerConfig, GepaCandidateSelectorConfig, GepaObjectiveAcceptanceConfig,
    GepaPipelineMode, GepaRunResult, LeverBundle, LeverKind, LeverManifest, LimitDefinition,
    LimitEngine, LimitEngineInput, LimitForecast, LimitKind, LimitObservation, LimitSnapshot,
    LimitStatus, ManagedContainerProcess, MaterializationRecord, MaterializationRecordInput,
    ObjectiveScore, ObjectiveSetRecord, ObjectiveSpec, OptimizerError, OptimizerJob,
    OptimizerJobKind, OptimizerJobStatus, OptimizerRunState, OptimizerStateMachine,
    OptimizerTransition, OptimizerTransitionTrigger, ParetoComparisonRecord, PlanLinkInput,
    PlanLinkRecord, PromptCandidatePayload, PromptProgram, PromptProgramSnapshotInput,
    PromptProgramSnapshotRecord, RequestCache, ResolvedRunConfigInput, ResolvedRunConfigRecord,
    Result, RetryPolicy, RolloutMaterializationIdentity, RunArtifactStore, RunPhaseTimingInput,
    RunRegistry, RunRegistryEntry, RunStorageInspectionInput, RuntimeEffectInput,
    RuntimeEffectRecord, ScoreRecord, ScoreVectorRecord, SensorFrame, StateMachineEntity,
    StopperStateInput, StopperStateRecord, SynthOptimizerConfig, TasksetResponse,
    TasksetSnapshotInput, TasksetSnapshotRecord, TasksetTasksRequest, TasksetTasksResponse,
    TransitionInput, TransitionLog, TransitionSink, UsageLedgerInput, UsageLedgerRecord,
    WorkspaceStore, LIMIT_ENGINE_SCHEMA_VERSION,
};

mod codex_app_server;
mod machines;
pub mod pipeline;
pub mod planner;
pub mod runtime;
pub mod service;

use machines::{
    CandidateEntity, CandidateState, CandidateTrigger, ProposerRoundEntity, ProposerRoundState,
    ProposerRoundTrigger, RolloutEntity, RolloutState, RolloutTrigger,
};

pub fn default_proposer_best_practices() -> &'static str {
    include_str!("prompting_best_practices.md")
}

use pipeline::{
    GepaAsyncPipelinePlan, GepaPipelineRuntimePlan, GepaStaleItemDecision,
    GepaStaleItemDisposition, GepaSyncSerialPlan,
};
use planner::{
    GepaAdaptiveRolloutConcurrencyAdjustment, GepaAdaptiveStageWorkersAdjustment,
    GepaAsyncCandidatePartial, GepaAsyncLaneLease, GepaAsyncLaneWorkItem, GepaCursor,
    GepaCursorPhase, GepaRolloutChunkPartial, GepaRolloutCircuitBreaker, GepaRolloutFailureSample,
    GepaRolloutResilienceState, GepaSpeculativeReleaseRecord, GepaStalenessReviewRecord,
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

fn checkpoint_candidate_records(candidates: &[CandidateRecord]) -> Vec<Value> {
    candidates
        .iter()
        .map(|candidate| {
            let mut value = candidate_value_with_seed_summaries(candidate);
            if let Some(object) = value.as_object_mut() {
                object.insert("sensor_frames".to_string(), Value::Array(Vec::new()));
            }
            value
        })
        .collect()
}

fn candidate_value_with_seed_summaries(candidate: &CandidateRecord) -> Value {
    let mut value = serde_json::to_value(candidate).unwrap_or(Value::Null);
    enrich_candidate_value_with_seed_summaries(&mut value, candidate);
    value
}

fn enrich_candidate_value_with_seed_summaries(value: &mut Value, candidate: &CandidateRecord) {
    let seed_rewards = candidate_seed_rewards(candidate);
    let seed_counts = seed_counts_from_rewards(&seed_rewards);
    if let Some(object) = value.as_object_mut() {
        object.insert("seed_counts".to_string(), seed_counts);
        object.insert("seed_rewards".to_string(), seed_rewards);
    }
}

fn artifact_candidate_records(candidates: &[CandidateRecord]) -> Vec<CandidateRecord> {
    candidates.iter().map(artifact_candidate_record).collect()
}

fn artifact_candidate_record(candidate: &CandidateRecord) -> CandidateRecord {
    let mut candidate = candidate.clone();
    candidate.sensor_frames = candidate
        .sensor_frames
        .iter()
        .map(artifact_sensor_frame)
        .collect();
    candidate
}

fn artifact_sensor_frame(frame: &SensorFrame) -> SensorFrame {
    let mut frame = frame.clone();
    frame.metadata = artifact_sensor_frame_metadata(&frame);
    frame
}

fn artifact_sensor_frame_metadata(frame: &SensorFrame) -> Map<String, Value> {
    let mut metadata = Map::new();
    metadata.insert(
        "schema".to_string(),
        json!("synth_gepa.sensor_frame_artifact_summary.v1"),
    );
    metadata.insert("storage_compacted".to_string(), json!(true));
    metadata.insert(
        "original_metadata_keys".to_string(),
        Value::Array(
            frame
                .metadata
                .keys()
                .map(|key| Value::String(key.clone()))
                .collect(),
        ),
    );
    if let Some(summary) = frame.metadata.get("summary") {
        metadata.insert(
            "summary".to_string(),
            project_json_fields(
                summary,
                &[
                    "outcome_reward",
                    "reward",
                    "achievement_score",
                    "achievement_deltas",
                    "final_achievements",
                    "final_inventory",
                    "objective_scores",
                    "objective_counts",
                    "termination",
                    "turn_count",
                    "unique_actions",
                    "invalid_action_count",
                    "malformed_tool_call_count",
                    "missing_tool_call_count",
                ],
            ),
        );
    }
    if let Some(reward_details) = frame.metadata.get("reward_details") {
        metadata.insert(
            "reward_details".to_string(),
            project_json_fields(
                reward_details,
                &[
                    "achievement_count",
                    "achievement_score",
                    "achievement_universe_count",
                    "achievements",
                    "backend",
                    "env_reward",
                    "objective_achievements",
                    "objective_counts",
                    "objective_scores",
                ],
            ),
        );
    }
    if let Some(rollout_trace) = frame.metadata.get("rollout_trace") {
        metadata.insert(
            "rollout_trace".to_string(),
            compact_rollout_trace_summary(rollout_trace),
        );
    }
    let trace_refs = frame
        .artifact_refs
        .iter()
        .filter(|artifact| artifact.kind == "rollout_trace_payload")
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    if !trace_refs.is_empty() {
        metadata.insert(
            "rollout_trace_artifact_refs".to_string(),
            Value::Array(trace_refs),
        );
    }
    metadata
}

fn compact_rollout_trace_summary(rollout_trace: &Value) -> Value {
    let mut summary = Map::new();
    summary.insert(
        "schema".to_string(),
        json!("synth_gepa.rollout_trace_summary_ref.v1"),
    );
    summary.insert("storage_compacted".to_string(), json!(true));
    for key in [
        "schema_version",
        "rollout_id",
        "trace_correlation_id",
        "task_id",
    ] {
        if let Some(value) = rollout_trace.get(key) {
            summary.insert(key.to_string(), value.clone());
        }
    }
    if let Some(value) = rollout_trace.get("summary") {
        summary.insert(
            "summary".to_string(),
            project_json_fields(
                value,
                &[
                    "outcome_reward",
                    "reward",
                    "achievement_score",
                    "achievement_deltas",
                    "objective_scores",
                    "objective_counts",
                    "termination",
                    "turn_count",
                ],
            ),
        );
    }
    if let Some(value) = rollout_trace.get("outcome") {
        summary.insert(
            "outcome".to_string(),
            project_json_fields(
                value,
                &["status", "success_status", "status_detail", "reward"],
            ),
        );
    }
    Value::Object(summary)
}

fn project_json_fields(value: &Value, keys: &[&str]) -> Value {
    let mut projected = Map::new();
    if let Some(object) = value.as_object() {
        for key in keys {
            if let Some(value) = object.get(*key) {
                projected.insert((*key).to_string(), value.clone());
            }
        }
    }
    Value::Object(projected)
}

fn candidate_seed_rewards(candidate: &CandidateRecord) -> Value {
    let mut grouped = BTreeMap::<String, Vec<Value>>::new();
    let mut seen = BTreeSet::<String>::new();
    for frame in &candidate.sensor_frames {
        let stage = seed_reward_stage(&frame.evaluation_stage);
        let dedupe_key = format!("{stage}\u{0}{}\u{0}{}", frame.task_id, frame.example_id);
        if seen.insert(dedupe_key) {
            grouped
                .entry(stage)
                .or_default()
                .push(seed_reward_row_from_frame(frame));
        }
    }
    if !grouped.contains_key("minibatch") {
        let rows = seed_reward_rows_from_scores("minibatch", &candidate.minibatch_scores);
        if !rows.is_empty() {
            grouped.insert("minibatch".to_string(), rows);
        }
    }
    if !grouped.contains_key("train") {
        let rows = seed_reward_rows_from_scores("train", &candidate.train_scores);
        if !rows.is_empty() {
            grouped.insert("train".to_string(), rows);
        }
    }
    grouped
        .values_mut()
        .for_each(|rows| rows.sort_by(seed_reward_row_cmp));
    Value::Object(
        grouped
            .into_iter()
            .map(|(stage, rows)| (stage, Value::Array(rows)))
            .collect(),
    )
}

fn seed_reward_stage(evaluation_stage: &str) -> String {
    match evaluation_stage {
        "candidate_minibatch" => "minibatch".to_string(),
        "candidate_full_train" | "seed_full_train" => "train".to_string(),
        "parent_minibatch_reference" => "parent_minibatch_reference".to_string(),
        "heldout" => "heldout".to_string(),
        other => other.to_string(),
    }
}

fn seed_reward_row_from_frame(frame: &SensorFrame) -> Value {
    let mut row = Map::new();
    row.insert("seed_id".to_string(), json!(&frame.task_id));
    row.insert("task_id".to_string(), json!(&frame.task_id));
    row.insert("example_id".to_string(), json!(&frame.example_id));
    row.insert("split".to_string(), json!(&frame.split));
    row.insert(
        "evaluation_stage".to_string(),
        json!(&frame.evaluation_stage),
    );
    row.insert("reward".to_string(), json!(frame.reward));
    row.insert("status".to_string(), json!(&frame.status));
    if let Some(success_status) = &frame.success_status {
        row.insert("success_status".to_string(), json!(success_status));
    }
    let metadata = compact_seed_reward_metadata(frame);
    if metadata
        .as_object()
        .is_some_and(|object| !object.is_empty())
    {
        row.insert("metadata".to_string(), metadata);
    }
    Value::Object(row)
}

fn compact_seed_reward_metadata(frame: &SensorFrame) -> Value {
    let mut metadata = Map::new();
    if let Some(actionable_side_info) = &frame.actionable_side_info {
        metadata.insert(
            "actionable_side_info".to_string(),
            actionable_side_info.clone(),
        );
    }
    if !frame.objective_scores.is_empty() {
        metadata.insert(
            "objective_scores".to_string(),
            json!(&frame.objective_scores),
        );
    }
    if !frame.usage.is_null() {
        metadata.insert("usage".to_string(), frame.usage.clone());
    }
    if let Some(trace_digest) = &frame.trace_digest {
        metadata.insert("trace_digest".to_string(), json!(trace_digest));
    }
    if !frame.artifact_refs.is_empty() {
        metadata.insert("artifact_refs".to_string(), json!(&frame.artifact_refs));
    }
    let artifact_metadata = artifact_sensor_frame_metadata(frame);
    for key in [
        "summary",
        "reward_details",
        "rollout_trace",
        "rollout_trace_artifact_refs",
    ] {
        if let Some(value) = artifact_metadata.get(key) {
            metadata.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(metadata)
}

fn seed_reward_rows_from_scores(stage: &str, scores: &[RolloutScore]) -> Vec<Value> {
    scores
        .iter()
        .map(|score| {
            json!({
                "seed_id": &score.task_id,
                "task_id": &score.task_id,
                "example_id": &score.example_id,
                "evaluation_stage": stage,
                "reward": score.reward,
            })
        })
        .collect()
}

fn seed_counts_from_rewards(seed_rewards: &Value) -> Value {
    let mut counts = Map::new();
    if let Some(object) = seed_rewards.as_object() {
        for (stage, rows) in object {
            let count = rows.as_array().map(Vec::len).unwrap_or(0);
            counts.insert(stage.clone(), json!(count));
        }
    }
    Value::Object(counts)
}

fn seed_reward_row_cmp(left: &Value, right: &Value) -> std::cmp::Ordering {
    value_string_field(left, "task_id")
        .cmp(&value_string_field(right, "task_id"))
        .then_with(|| {
            value_string_field(left, "example_id").cmp(&value_string_field(right, "example_id"))
        })
        .then_with(|| {
            value_string_field(left, "evaluation_stage")
                .cmp(&value_string_field(right, "evaluation_stage"))
        })
}

fn value_string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn hydrate_candidate_sensor_frames_from_workspace(
    workspace: &WorkspaceStore,
    run_id: &str,
    candidates: &mut [CandidateRecord],
) -> Result<()> {
    if candidates
        .iter()
        .all(|candidate| !candidate.sensor_frames.is_empty())
    {
        return Ok(());
    }
    let mut frames_by_candidate = BTreeMap::<String, Vec<SensorFrame>>::new();
    for frame in workspace.view().sensor_frames(run_id)? {
        frames_by_candidate
            .entry(frame.candidate_id.clone())
            .or_default()
            .push(frame);
    }
    if frames_by_candidate.is_empty() {
        return Ok(());
    }
    for candidate in candidates {
        if candidate.sensor_frames.is_empty() {
            if let Some(frames) = frames_by_candidate.remove(&candidate.candidate_id) {
                candidate.sensor_frames = frames;
            }
        }
    }
    Ok(())
}

fn candidate_checkpoint_summaries(candidates: &[CandidateRecord]) -> Vec<Value> {
    candidates
        .iter()
        .map(|candidate| {
            let seed_rewards = candidate_seed_rewards(candidate);
            let seed_counts = seed_counts_from_rewards(&seed_rewards);
            json!({
                "candidate_id": &candidate.candidate_id,
                "parent_id": &candidate.parent_id,
                "source": &candidate.source,
                "status": &candidate.status,
                "minibatch_reward": candidate.minibatch_reward,
                "train_reward": candidate.train_reward,
                "heldout_reward": candidate.heldout_reward,
                "minibatch_score_count": candidate.minibatch_scores.len(),
                "train_score_count": candidate.train_scores.len(),
                "sensor_frame_count": candidate.sensor_frames.len(),
                "seed_counts": seed_counts,
                "seed_rewards": seed_rewards,
                "acceptance_score": &candidate.acceptance_score,
            })
        })
        .collect()
}

fn frontier_checkpoint_summaries(frontier: &[FrontierMember]) -> Vec<Value> {
    frontier
        .iter()
        .map(|member| {
            json!({
                "candidate_id": &member.candidate_id,
                "parent_id": &member.parent_id,
                "source": &member.source,
                "train_reward": member.train_reward,
                "heldout_reward": member.heldout_reward,
            })
        })
        .collect()
}

fn compact_terminal_summary(result: &Value) -> Value {
    let mut summary = Map::new();
    summary.insert(
        "schema".to_string(),
        json!("synth_gepa.terminal_summary_ref.v1"),
    );
    summary.insert("storage_compacted".to_string(), json!(true));
    if let Some(manifest_path) = result.get("manifest_path").cloned() {
        summary.insert("manifest_path".to_string(), manifest_path);
    }
    if let Some(workspace_db_path) = result.get("workspace_db_path").cloned() {
        summary.insert("workspace_db_path".to_string(), workspace_db_path);
    }
    if let Some(best_candidate_id) = result
        .get("best_candidate")
        .and_then(|candidate| candidate.get("candidate_id"))
        .cloned()
    {
        summary.insert("best_candidate_id".to_string(), best_candidate_id);
    }
    if let Some(cost_usd) = result.get("cost_usd").cloned() {
        summary.insert("cost_usd".to_string(), cost_usd);
    }
    if let Some(stopped_by) = result.get("stopped_by").cloned() {
        summary.insert("stopped_by".to_string(), stopped_by);
    }
    Value::Object(summary)
}

fn terminal_result_from_cursor(
    context: &GepaRunContext,
    cursor: &GepaCursor,
) -> Result<Option<GepaRunResult>> {
    if let Some(summary) = cursor.terminal_summary.clone() {
        if summary.get("best_candidate").is_some() && summary.get("manifest_path").is_some() {
            return serde_json::from_value(summary)
                .map(Some)
                .map_err(OptimizerError::from);
        }
        if let Some(result) = context.workspace.run_manifest(&context.config.run.run_id)? {
            return serde_json::from_value(result)
                .map(Some)
                .map_err(OptimizerError::from);
        }
        let manifest_path = summary
            .get("manifest_path")
            .and_then(Value::as_str)
            .filter(|path| !path.trim().is_empty());
        if let Some(manifest_path) = manifest_path {
            let path = Path::new(manifest_path);
            if path.exists() {
                let raw =
                    fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
                return serde_json::from_str(&raw)
                    .map(Some)
                    .map_err(OptimizerError::from);
            }
        }
    }
    context
        .workspace
        .run_manifest(&context.config.run.run_id)?
        .map(serde_json::from_value)
        .transpose()
        .map_err(OptimizerError::from)
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
    pub runtime_substrate: String,
    pub workspace: Option<String>,
    #[serde(default)]
    pub evidence_warnings: Vec<String>,
}

/// Public, workspace-backed proposer result for optimizer experiments that need
/// GEPA's exact proposer substrate without adopting GEPA's search dynamics.
/// The caller remains responsible for candidate admission, rollout accounting,
/// and heldout isolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceProposerOutcome {
    pub proposals: Vec<ProposedCandidate>,
    pub response: Value,
    pub backend: String,
    pub runtime_substrate: String,
    pub workspace: String,
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
    /// Resolve every supported GEPA proposal shape into the prompt-program
    /// candidate-field map consumed by an optimizer container.
    pub fn resolved_payload(&self) -> BTreeMap<String, String> {
        self.payload_map()
    }

    /// Resolve a proposal while preserving the single-target compatibility
    /// behavior used by public GEPA.
    pub fn resolved_payload_for_allowed_fields(
        &self,
        allowed_fields: &[String],
    ) -> BTreeMap<String, String> {
        self.payload_map_for_allowed_fields(allowed_fields)
    }

    pub(crate) fn payload_map(&self) -> BTreeMap<String, String> {
        if !self.payload.is_empty() {
            let payload = Self::payload_from_string_payload(&self.payload);
            if !payload.is_empty() {
                return payload;
            }
        }
        if let Some(bundle) = &self.lever_bundle {
            let payload = bundle.to_prompt_payload();
            if !payload.is_empty() {
                return payload;
            }
        }
        for payload_key in ["proposed_payload", "candidate"] {
            if let Some(payload) = self
                .extra
                .get(payload_key)
                .map(Self::payload_from_proposed_payload_value)
                .filter(|payload| !payload.is_empty())
            {
                return payload;
            }
        }
        let flattened_payload =
            Self::payload_from_proposed_payload_value(&Value::Object(self.extra.clone()));
        if !flattened_payload.is_empty() {
            return flattened_payload;
        }
        self.extra
            .iter()
            .filter(|(key, _)| !Self::is_structural_payload_key(key))
            .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
            .collect()
    }

    pub(crate) fn payload_map_for_allowed_fields(
        &self,
        allowed_fields: &[String],
    ) -> BTreeMap<String, String> {
        let payload = self.payload_map();
        if payload
            .keys()
            .any(|key| allowed_fields.iter().any(|field| field == key))
            || allowed_fields.len() != 1
        {
            return payload;
        }

        let Some(target_field) = allowed_fields
            .first()
            .map(String::as_str)
            .map(str::trim)
            .filter(|target_field| !target_field.is_empty())
        else {
            return payload;
        };

        for payload_key in ["proposed_payload", "candidate"] {
            if let Some(single_target_payload) = self.extra.get(payload_key).and_then(|value| {
                Self::single_target_payload_from_unqualified_value(value, target_field)
            }) {
                return single_target_payload;
            }
        }

        Self::single_target_payload_from_unqualified_value(
            &Value::Object(self.extra.clone()),
            target_field,
        )
        .unwrap_or(payload)
    }

    fn is_structural_payload_key(key: &str) -> bool {
        matches!(
            key,
            "metadata"
                | "candidate"
                | "proposed_payload"
                | "module_id"
                | "modules"
                | "target_modules"
                | "mutable"
                | "role"
                | "template_variables"
                | "candidate_field"
                | "program_id"
                | "seed_candidate"
        )
    }

    fn is_single_target_payload_key(key: &str) -> bool {
        matches!(
            key,
            "content"
                | "value"
                | "prompt"
                | "instructions"
                | "text"
                | "role"
                | "modules"
                | "target_modules"
        )
    }

    fn is_single_target_module_key(key: &str) -> bool {
        matches!(
            key,
            "content" | "value" | "prompt" | "instructions" | "text" | "role"
        )
    }

    fn payload_from_string_payload(payload: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        if let Some(module_id) = payload
            .get("module_id")
            .map(String::as_str)
            .map(str::trim)
            .filter(|module_id| !module_id.is_empty())
        {
            for value_key in ["content", "value", "prompt", "instructions", "text"] {
                if let Some(text) = payload
                    .get(value_key)
                    .map(String::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                {
                    return BTreeMap::from([(module_id.to_string(), text.to_string())]);
                }
            }
        }
        payload
            .iter()
            .filter(|(key, _)| !Self::is_structural_payload_key(key))
            .filter(|(_, value)| !value.trim().is_empty())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn single_target_payload_from_unqualified_value(
        value: &Value,
        target_field: &str,
    ) -> Option<BTreeMap<String, String>> {
        let object = value.as_object()?;
        if object.contains_key(target_field) {
            return None;
        }
        if object
            .keys()
            .any(|key| !Self::is_single_target_payload_key(key))
        {
            return None;
        }

        let mut chunks = Vec::new();
        if let Some(text) = ["content", "value", "prompt", "instructions", "text"]
            .iter()
            .find_map(|value_key| {
                object
                    .get(*value_key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
            })
        {
            chunks.push(text.to_string());
        }

        for modules_key in ["modules", "target_modules"] {
            let Some(modules) = object.get(modules_key).and_then(Value::as_array) else {
                continue;
            };
            for module in modules {
                let module_object = module.as_object()?;
                if module_object
                    .get("candidate_field")
                    .or_else(|| module_object.get("module_id"))
                    .is_some()
                {
                    return None;
                }
                if module_object
                    .keys()
                    .any(|key| !Self::is_single_target_module_key(key))
                {
                    return None;
                }
                if let Some(content) = ["content", "value", "prompt", "instructions", "text"]
                    .iter()
                    .find_map(|value_key| {
                        module_object
                            .get(*value_key)
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|text| !text.is_empty())
                    })
                {
                    chunks.push(content.to_string());
                }
            }
        }

        let content = chunks
            .into_iter()
            .filter(|chunk| !chunk.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if content.trim().is_empty() {
            return None;
        }
        Some(BTreeMap::from([(target_field.to_string(), content)]))
    }

    fn payload_from_proposed_payload_value(value: &Value) -> BTreeMap<String, String> {
        let Some(object) = value.as_object() else {
            return BTreeMap::new();
        };
        if let Some(module_id) = object
            .get("module_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|module_id| !module_id.is_empty())
        {
            for value_key in ["content", "value", "prompt", "instructions", "text"] {
                if let Some(text) = object
                    .get(value_key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                {
                    return BTreeMap::from([(module_id.to_string(), text.to_string())]);
                }
            }
        }

        if let Some(modules) = object
            .get("modules")
            .or_else(|| object.get("target_modules"))
            .and_then(Value::as_array)
        {
            let mut payload = BTreeMap::new();
            for module in modules {
                let Some(module_object) = module.as_object() else {
                    continue;
                };
                let Some(candidate_field) = module_object
                    .get("candidate_field")
                    .or_else(|| module_object.get("module_id"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|candidate_field| !candidate_field.is_empty())
                else {
                    continue;
                };
                let Some(content) = ["content", "value", "prompt", "instructions", "text"]
                    .iter()
                    .find_map(|value_key| {
                        module_object
                            .get(*value_key)
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|text| !text.is_empty())
                    })
                else {
                    continue;
                };
                payload.insert(candidate_field.to_string(), content.to_string());
            }
            if !payload.is_empty() {
                return payload;
            }
        }

        object
            .iter()
            .filter(|(key, _)| !Self::is_structural_payload_key(key))
            .filter_map(|(key, value)| {
                value
                    .as_str()
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(|text| (key.clone(), text.to_string()))
            })
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

    pub(crate) fn payload_shape_summary(&self) -> Value {
        let extra_keys = self.extra.keys().cloned().collect::<Vec<_>>();
        let payload_keys = self.payload.keys().cloned().collect::<Vec<_>>();
        let proposed_payload_keys = self
            .extra
            .get("proposed_payload")
            .and_then(Value::as_object)
            .map(|object| object.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let candidate_keys = self
            .extra
            .get("candidate")
            .and_then(Value::as_object)
            .map(|object| object.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        json!({
            "payload_keys": payload_keys,
            "extra_keys": extra_keys,
            "proposed_payload_keys": proposed_payload_keys,
            "candidate_keys": candidate_keys,
            "has_lever_bundle": self.lever_bundle.is_some(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutScore {
    pub example_id: String,
    pub task_id: String,
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
    task_pool_rows: Value,
    paths: &'a ArtifactPaths,
}

struct EvaluationCall<'a> {
    client: &'a ContainerClient,
    workspace: &'a WorkspaceStore,
    paths: &'a ArtifactPaths,
    cache: &'a mut RequestCache,
    events: &'a mut EventWriter,
    rollout_resilience: &'a mut GepaRolloutResilienceState,
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
    transitions: TransitionSink,
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
    last_limit_estimate_check: Option<Instant>,
    last_limit_estimate_checkpoint_sequence: u64,
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
const LIMIT_ESTIMATE_UPDATE_MIN_INTERVAL: Duration = Duration::from_secs(60);
const LIMIT_ESTIMATE_UPDATE_CHECKPOINT_STEP: u64 = 10;

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

#[derive(Clone, Debug)]
struct ReflectiveStalenessReview {
    review_id: String,
    verdict: ReflectiveStalenessVerdict,
    reason: String,
    patched_payload: Option<BTreeMap<String, String>>,
    workspace: Option<String>,
    raw: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReflectiveStalenessVerdict {
    Accept,
    Discard,
    Patch,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ProviderSignal {
    #[serde(default)]
    status_code: Option<u16>,
    #[serde(default)]
    provider_error_code: Option<String>,
    #[serde(default)]
    overload: bool,
    #[serde(default)]
    retryable: bool,
}

#[derive(Clone, Debug)]
struct RolloutExecutionRecord {
    outcome: runtime::RuntimeRolloutOutcome,
    degraded: bool,
    failure: Option<FailurePayload>,
    provider_signal: ProviderSignal,
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
    pub owning_service_url: Option<String>,
    pub artifact_store: Option<Arc<dyn RunArtifactStore>>,
}

#[derive(Clone, Debug)]
pub struct GepaCancellationSource {
    pub service_db_path: PathBuf,
    pub request_id: String,
    pub lease_id: Option<String>,
    pub lease_seconds: u64,
    pub in_process: Option<Arc<AtomicBool>>,
}

pub(crate) fn gepa_home_dir() -> PathBuf {
    if let Some(value) = std::env::var_os("GEPA_HOME") {
        return PathBuf::from(value);
    }
    if let Some(value) = std::env::var_os("HOME") {
        return PathBuf::from(value).join(".gepa");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".gepa")
}

pub(crate) fn rfc3339_now() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

pub fn project_gepa_limit_snapshot(
    run_store: &WorkspaceStore,
    config: &SynthOptimizerConfig,
) -> Result<LimitSnapshot> {
    let run_id = &config.run.run_id;
    let generated_at = rfc3339_now();
    let limits = run_store.required_run_limits(run_id)?;
    let ledger = run_store.budget_ledger_snapshot(run_id)?;
    let view = run_store.view();
    let reservations = view.budget_reservation_records(run_id)?;
    let commits = view.budget_commit_records(run_id)?;
    let admissions = view.runtime_effect_admission_records(run_id)?;
    let timings = view.run_phase_timing_records(run_id)?;
    let mut input = budget_limit_engine_input(
        run_id,
        &limits,
        &ledger,
        &reservations,
        &commits,
        &admissions,
        &timings,
        Some(generated_at.clone()),
    );
    let checkpoints = run_store.checkpoint_summary_history(run_id, None)?;
    let cursor = load_gepa_cursor_from_workspace(run_store, run_id)?;
    append_gepa_search_limits(
        &mut input,
        config,
        &generated_at,
        &checkpoints,
        cursor.as_ref(),
    )?;
    let mut snapshot = LimitEngine::snapshot(input);
    apply_gepa_phase_forecasts(
        &mut snapshot,
        config,
        cursor.as_ref(),
        &checkpoints,
        &generated_at,
    );
    Ok(snapshot)
}

fn append_gepa_search_limits(
    input: &mut LimitEngineInput,
    config: &SynthOptimizerConfig,
    generated_at: &str,
    checkpoints: &[CheckpointSummaryRecord],
    cursor: Option<&GepaCursor>,
) -> Result<()> {
    let run_id = &config.run.run_id;
    let generation_limit = config.gepa.max_generations.max(1) as f64;
    let candidate_limit = config
        .gepa
        .max_generations
        .saturating_mul(config.gepa.proposals_per_generation.max(1))
        .saturating_add(1)
        .max(1) as f64;
    let generation_limit_id = gepa_limit_id(run_id, LimitKind::Generations);
    let candidate_limit_id = gepa_limit_id(run_id, LimitKind::Candidates);
    input.definitions.push(gepa_search_limit_definition(
        run_id,
        LimitKind::Generations,
        generation_limit,
        true,
        json!({
            "configured_by": "gepa.max_generations",
        }),
    ));
    input.definitions.push(gepa_search_limit_definition(
        run_id,
        LimitKind::Candidates,
        candidate_limit,
        false,
        json!({
            "configured_by": "gepa.max_generations * gepa.proposals_per_generation + seed",
            "max_generations": config.gepa.max_generations,
            "proposals_per_generation": config.gepa.proposals_per_generation,
            "seed_candidates": 1,
        }),
    ));
    for checkpoint in checkpoints {
        if let Some(generation) = checkpoint.generation {
            input.observations.push(LimitObservation {
                run_id: run_id.clone(),
                limit_id: generation_limit_id.clone(),
                timestamp: checkpoint.created_at.clone(),
                spent: generation as f64,
                reserved: 0.0,
                source_kind: "checkpoint".to_string(),
                source_id: checkpoint.checkpoint_id.clone(),
            });
        }
        input.observations.push(LimitObservation {
            run_id: run_id.clone(),
            limit_id: candidate_limit_id.clone(),
            timestamp: checkpoint.created_at.clone(),
            spent: checkpoint.candidate_count as f64,
            reserved: 0.0,
            source_kind: "checkpoint".to_string(),
            source_id: checkpoint.checkpoint_id.clone(),
        });
    }
    if let Some(cursor) = cursor {
        input.observations.push(LimitObservation {
            run_id: run_id.clone(),
            limit_id: generation_limit_id,
            timestamp: generated_at.to_string(),
            spent: cursor.generation as f64,
            reserved: 0.0,
            source_kind: "gepa_cursor".to_string(),
            source_id: format!("cursor:{run_id}:current:generation"),
        });
        input.observations.push(LimitObservation {
            run_id: run_id.clone(),
            limit_id: candidate_limit_id,
            timestamp: generated_at.to_string(),
            spent: cursor.candidates.as_array().map(Vec::len).unwrap_or(0) as f64,
            reserved: 0.0,
            source_kind: "gepa_cursor".to_string(),
            source_id: format!("cursor:{run_id}:current:candidates"),
        });
    }
    Ok(())
}

fn apply_gepa_phase_forecasts(
    snapshot: &mut LimitSnapshot,
    config: &SynthOptimizerConfig,
    cursor: Option<&GepaCursor>,
    checkpoints: &[CheckpointSummaryRecord],
    generated_at: &str,
) {
    let Some(cursor) = cursor else {
        return;
    };
    let run_id = &config.run.run_id;
    let generation_limit_id = gepa_limit_id(run_id, LimitKind::Generations);
    let Some(status) = snapshot
        .limits
        .iter_mut()
        .find(|status| status.definition.limit_id == generation_limit_id)
    else {
        return;
    };
    if status.remaining > 0.0 {
        if let Some(forecast) = gepa_generation_phase_forecast(
            run_id,
            &generation_limit_id,
            config.gepa.max_generations,
            cursor.generation,
            checkpoints,
            generated_at,
        ) {
            status.forecast = forecast;
        }
    }
    recompute_nearest_limit(snapshot);
}

fn gepa_generation_phase_forecast(
    run_id: &str,
    limit_id: &str,
    max_generations: usize,
    current_generation: usize,
    checkpoints: &[CheckpointSummaryRecord],
    generated_at: &str,
) -> Option<LimitForecast> {
    let remaining_generations = max_generations.saturating_sub(current_generation);
    if remaining_generations == 0 {
        return Some(gepa_phase_limit_forecast(
            run_id,
            limit_id,
            "exhausted",
            ForecastConfidence::High,
            0,
            0.0,
            generated_at,
        ));
    }
    let completed_durations = completed_generation_durations(checkpoints);
    let current_elapsed =
        current_generation_elapsed_seconds(checkpoints, current_generation, generated_at)
            .unwrap_or(0.0);
    let (model, confidence, generation_seconds) = if completed_durations.len() >= 3 {
        (
            "phase_generation_ar1",
            ForecastConfidence::High,
            phase_ar1_duration(&completed_durations)?,
        )
    } else if !completed_durations.is_empty() {
        (
            "phase_generation_mean",
            ForecastConfidence::Medium,
            completed_durations.iter().sum::<f64>() / completed_durations.len() as f64,
        )
    } else if current_elapsed >= 1.0 {
        (
            "phase_elapsed_fallback",
            ForecastConfidence::Low,
            current_elapsed,
        )
    } else {
        return None;
    };
    if generation_seconds <= 0.0 || !generation_seconds.is_finite() {
        return None;
    }
    let seconds_to_limit =
        (generation_seconds * remaining_generations as f64 - current_elapsed).max(0.0);
    Some(gepa_phase_limit_forecast(
        run_id,
        limit_id,
        model,
        confidence,
        completed_durations.len() as u64,
        seconds_to_limit,
        generated_at,
    ))
}

fn completed_generation_durations(checkpoints: &[CheckpointSummaryRecord]) -> Vec<f64> {
    let mut starts = BTreeMap::<u64, f64>::new();
    for checkpoint in checkpoints {
        if checkpoint.checkpoint_kind != GEPA_CURSOR_CHECKPOINT_KIND
            || checkpoint.run_state != GepaCursorPhase::GenerationStart.as_str()
        {
            continue;
        }
        let Some(generation) = checkpoint.generation else {
            continue;
        };
        let Some(ts) = parse_rfc3339_seconds(&checkpoint.created_at) else {
            continue;
        };
        starts.entry(generation).or_insert(ts);
    }
    checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.checkpoint_kind == "generation_boundary")
        .filter_map(|checkpoint| {
            let generation = checkpoint.generation?;
            let start = starts.get(&generation).copied()?;
            let end = parse_rfc3339_seconds(&checkpoint.created_at)?;
            let seconds = end - start;
            (seconds.is_finite() && seconds > 0.0).then_some(seconds)
        })
        .collect()
}

fn current_generation_elapsed_seconds(
    checkpoints: &[CheckpointSummaryRecord],
    generation: usize,
    generated_at: &str,
) -> Option<f64> {
    let generation = generation as u64;
    let start = checkpoints
        .iter()
        .filter(|checkpoint| {
            checkpoint.checkpoint_kind == GEPA_CURSOR_CHECKPOINT_KIND
                && checkpoint.run_state == GepaCursorPhase::GenerationStart.as_str()
                && checkpoint.generation == Some(generation)
        })
        .filter_map(|checkpoint| parse_rfc3339_seconds(&checkpoint.created_at))
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))?;
    let now = parse_rfc3339_seconds(generated_at)?;
    Some((now - start).max(0.0))
}

fn phase_ar1_duration(durations: &[f64]) -> Option<f64> {
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for pair in durations.windows(2) {
        numerator += (pair[0] - mean) * (pair[1] - mean);
        denominator += (pair[0] - mean).powi(2);
    }
    let phi = if denominator > 0.0 {
        (numerator / denominator).clamp(-0.8, 0.95)
    } else {
        0.0
    };
    let last = *durations.last()?;
    Some((mean + phi * (last - mean)).max(0.0))
}

fn gepa_phase_limit_forecast(
    run_id: &str,
    limit_id: &str,
    model: &str,
    confidence: ForecastConfidence,
    sample_count: u64,
    seconds_to_limit: f64,
    generated_at: &str,
) -> LimitForecast {
    let seconds_to_limit = seconds_to_limit.max(0.0);
    let seconds_to_limit_low = seconds_to_limit * gepa_phase_interval_low_multiplier(confidence);
    let seconds_to_limit_high = seconds_to_limit * gepa_phase_interval_high_multiplier(confidence);
    LimitForecast {
        schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
        forecast_id: format!(
            "limit_forecast_{}",
            &stable_json_hash(&json!({
                "limit_id": limit_id,
                "updated_at": generated_at,
                "model": model,
            }))[..16]
        ),
        run_id: run_id.to_string(),
        limit_id: limit_id.to_string(),
        model: model.to_string(),
        predicted_crossing_at: add_rfc3339_seconds(generated_at, seconds_to_limit),
        seconds_to_limit: Some(seconds_to_limit),
        seconds_to_limit_low: Some(seconds_to_limit_low),
        seconds_to_limit_high: Some(seconds_to_limit_high),
        predicted_crossing_at_low: add_rfc3339_seconds(generated_at, seconds_to_limit_low),
        predicted_crossing_at_high: add_rfc3339_seconds(generated_at, seconds_to_limit_high),
        rate_per_second: None,
        confidence,
        sample_count,
        updated_at: generated_at.to_string(),
    }
}

fn recompute_nearest_limit(snapshot: &mut LimitSnapshot) {
    snapshot.nearest_limit = snapshot
        .limits
        .iter()
        .filter_map(|status| {
            status
                .forecast
                .seconds_to_limit
                .map(|seconds| (seconds, &status.forecast))
        })
        .min_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, forecast)| forecast.clone());
}

fn parse_rfc3339_seconds(value: &str) -> Option<f64> {
    time::OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339)
        .ok()
        .map(|ts| ts.unix_timestamp() as f64 + f64::from(ts.nanosecond()) / 1_000_000_000.0)
}

fn add_rfc3339_seconds(base: &str, seconds: f64) -> Option<String> {
    if !seconds.is_finite() {
        return None;
    }
    let base =
        time::OffsetDateTime::parse(base, &time::format_description::well_known::Rfc3339).ok()?;
    let seconds = seconds.ceil().min(i64::MAX as f64).max(0.0) as i64;
    (base + time::Duration::seconds(seconds))
        .format(&time::format_description::well_known::Rfc3339)
        .ok()
}

fn gepa_phase_interval_low_multiplier(confidence: ForecastConfidence) -> f64 {
    match confidence {
        ForecastConfidence::High => 0.85,
        ForecastConfidence::Medium => 0.75,
        ForecastConfidence::Low => 0.5,
        ForecastConfidence::Unknown => 0.0,
    }
}

fn gepa_phase_interval_high_multiplier(confidence: ForecastConfidence) -> f64 {
    match confidence {
        ForecastConfidence::High => 1.2,
        ForecastConfidence::Medium => 1.5,
        ForecastConfidence::Low => 2.25,
        ForecastConfidence::Unknown => 0.0,
    }
}

fn load_gepa_cursor_from_workspace(
    run_store: &WorkspaceStore,
    run_id: &str,
) -> Result<Option<GepaCursor>> {
    let Some(checkpoint) = run_store.latest_checkpoint(run_id, GEPA_CURSOR_CHECKPOINT_KIND)? else {
        return Ok(None);
    };
    serde_json::from_value(checkpoint.snapshot)
        .map(Some)
        .map_err(OptimizerError::from)
}

fn gepa_search_limit_definition(
    run_id: &str,
    kind: LimitKind,
    max_value: f64,
    hard: bool,
    metadata: Value,
) -> LimitDefinition {
    let mut metadata_map = Map::new();
    metadata_map.insert("algorithm".to_string(), json!("gepa"));
    metadata_map.insert("source".to_string(), metadata);
    LimitDefinition {
        schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
        limit_id: gepa_limit_id(run_id, kind.clone()),
        run_id: run_id.to_string(),
        kind,
        scope: "run".to_string(),
        max_value,
        hard,
        stop_policy: "gepa_search_plan".to_string(),
        source: "gepa_config".to_string(),
        metadata: metadata_map,
    }
}

fn gepa_limit_id(run_id: &str, kind: LimitKind) -> String {
    format!("{run_id}:{}", kind.as_str())
}

pub(crate) fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(path)
}

pub(crate) fn append_global_gepa_run_index(
    paths: &ArtifactPaths,
    config: &SynthOptimizerConfig,
    owning_service_url: Option<&str>,
) -> Result<()> {
    let home = gepa_home_dir();
    fs::create_dir_all(&home).map_err(|source| OptimizerError::io(&home, source))?;
    let index_path = home.join("index.jsonl");
    if index_path.exists() {
        let text = fs::read_to_string(&index_path)
            .map_err(|source| OptimizerError::io(&index_path, source))?;
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if value.get("run_id").and_then(Value::as_str) == Some(config.run.run_id.as_str()) {
                return Ok(());
            }
        }
    }
    let entry = json!({
        "schema": "synth.gepa_run_index.v1",
        "run_id": config.run.run_id.clone(),
        "run_dir": absolute_path(&paths.run_dir).display().to_string(),
        "event_feed_path": absolute_path(&paths.event_feed_path).display().to_string(),
        "run_registry_path": absolute_path(&paths.run_registry_path).display().to_string(),
        "pid": std::process::id(),
        "started_at": rfc3339_now(),
        "owning_service_url": owning_service_url,
    });
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&index_path)
        .map_err(|source| OptimizerError::io(&index_path, source))?;
    writeln!(file, "{}", serde_json::to_string(&entry)?)
        .map_err(|source| OptimizerError::io(&index_path, source))
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
        let prompt_tokens = usage_prompt_tokens(usage);
        let completion_tokens = usage_completion_tokens(usage);
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.total_tokens += usage_total_tokens(usage, prompt_tokens, completion_tokens);
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
        let prompt_tokens = usage_prompt_tokens(usage);
        let completion_tokens = usage_completion_tokens(usage);
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.total_tokens += runtime_total_tokens(fields, usage, prompt_tokens, completion_tokens);
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
        let prompt_tokens = usage_prompt_tokens(fields);
        let completion_tokens = usage_completion_tokens(fields);
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.total_tokens += usage_total_tokens(fields, prompt_tokens, completion_tokens);
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

fn runtime_total_tokens(
    fields: &Value,
    usage: &Value,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> u64 {
    usage_total_tokens(usage, prompt_tokens, completion_tokens)
        .max(field_u64(fields, "total_tokens"))
}

fn usage_total_tokens(usage: &Value, prompt_tokens: u64, completion_tokens: u64) -> u64 {
    usage_u64(usage, &["total_tokens", "totalTokens"])
        .max(prompt_tokens.saturating_add(completion_tokens))
}

fn usage_prompt_tokens(usage: &Value) -> u64 {
    usage_u64(
        usage,
        &[
            "prompt_tokens",
            "input_tokens",
            "inputTokens",
            "promptTokens",
        ],
    )
}

fn usage_completion_tokens(usage: &Value) -> u64 {
    usage_u64(
        usage,
        &[
            "completion_tokens",
            "output_tokens",
            "outputTokens",
            "completionTokens",
            "reasoning_output_tokens",
            "reasoningOutputTokens",
        ],
    )
}

fn usage_u64(usage: &Value, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Value::as_u64))
        .unwrap_or(0)
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
    let paths = if let Some(artifact_store) = options.artifact_store.clone() {
        ArtifactPaths::with_artifact_store(
            &config.run.output_dir,
            &config.run.run_id,
            artifact_store,
        )
    } else {
        ArtifactPaths::new(&config.run.output_dir, &config.run.run_id)
    };
    paths.create()?;
    let transition_log = TransitionLog::open(&paths.run_dir)?;
    let transitions = transition_log.sink();
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
    append_global_gepa_run_index(&paths, &config, options.owning_service_url.as_deref())?;
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
        Some(&transitions),
        OptimizerRunState::Initializing,
        OptimizerTransitionTrigger::RunStarted,
        "GEPA run initializing",
        json!({
            "run_id": config.run.run_id,
            "proposer_model": &config.proposer.model,
            "policy_model": &config.policy.model,
            "policy_provider": &config.policy.provider,
            "train_split": &config.taskset.train_split,
            "heldout_split": &config.taskset.heldout_split,
            "train_ids": &config.taskset.train_ids,
            "heldout_ids": &config.taskset.heldout_ids,
        }),
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
        transitions,
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
        last_limit_estimate_check: None,
        last_limit_estimate_checkpoint_sequence: 0,
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
    let client = ContainerClient::with_headers_and_bearer_env(
        container_url.clone(),
        context.config.container.headers.clone(),
        context.config.container.auth_bearer_env.as_deref(),
    )?;
    let metadata = client.verify_gepa_contract()?;
    let gepa_contract = metadata.resolved_gepa_contract()?;
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

    let taskset_value = cached_call(
        &mut context.cache,
        &format!("{}:container.taskset", context.cache_namespace),
        &json!({"url": container_url, "route": "/taskset"}),
        || {
            let response = client.taskset_typed()?;
            Ok(serde_json::to_value(response)?)
        },
    )?;
    let taskset_response: TasksetResponse = serde_json::from_value(taskset_value.clone())?;
    let taskset_id = taskset_response
        .taskset_id
        .clone()
        .unwrap_or_else(|| "container_taskset".to_string());
    let task_pool_ids = effective_gepa_task_pool_ids(&context.config);
    let pareto_ids = task_pool_ids
        .get("pareto")
        .cloned()
        .unwrap_or_else(|| context.config.taskset.train_ids.clone());
    let minibatch_ids = task_pool_ids
        .get("minibatch")
        .cloned()
        .unwrap_or_else(|| pareto_ids.clone());
    let reflection_ids = task_pool_ids
        .get("reflection")
        .cloned()
        .unwrap_or_else(|| minibatch_ids.clone());
    let heldout_ids = task_pool_ids
        .get("heldout")
        .cloned()
        .unwrap_or_else(|| context.config.taskset.heldout_ids.clone());
    let train_response = load_rows(
        &client,
        &mut context.cache,
        &context.cache_namespace,
        &context.config.taskset.train_split,
        &pareto_ids,
        Value::Object(context.config.taskset.filters.clone()),
    )?;
    let heldout_response = load_rows(
        &client,
        &mut context.cache,
        &context.cache_namespace,
        &context.config.taskset.heldout_split,
        &heldout_ids,
        Value::Object(context.config.taskset.filters.clone()),
    )?;
    let train_rows = train_response.tasks.clone();
    let heldout_rows = heldout_response.tasks.clone();
    let minibatch_rows = if minibatch_ids == pareto_ids {
        train_rows.clone()
    } else {
        load_rows(
            &client,
            &mut context.cache,
            &context.cache_namespace,
            &context.config.taskset.train_split,
            &minibatch_ids,
            Value::Object(context.config.taskset.filters.clone()),
        )?
        .tasks
    };
    let reflection_rows = if reflection_ids == pareto_ids {
        train_rows.clone()
    } else if reflection_ids == minibatch_ids {
        minibatch_rows.clone()
    } else {
        load_rows(
            &client,
            &mut context.cache,
            &context.cache_namespace,
            &context.config.taskset.train_split,
            &reflection_ids,
            Value::Object(context.config.taskset.filters.clone()),
        )?
        .tasks
    };
    record_taskset_snapshot(
        &context.workspace,
        TasksetSnapshotCall {
            run_id: &context.config.run.run_id,
            taskset_id: &taskset_id,
            split: &context.config.taskset.train_split,
            task_ids: &pareto_ids,
            filters: &Value::Object(context.config.taskset.filters.clone()),
            response: &train_response,
            taskset_metadata: &taskset_value,
        },
    )?;
    record_taskset_snapshot(
        &context.workspace,
        TasksetSnapshotCall {
            run_id: &context.config.run.run_id,
            taskset_id: &taskset_id,
            split: &context.config.taskset.heldout_split,
            task_ids: &heldout_ids,
            filters: &Value::Object(context.config.taskset.filters.clone()),
            response: &heldout_response,
            taskset_metadata: &taskset_value,
        },
    )?;
    context.events.emit(
        "taskset.tasks.loaded",
        "Taskset tasks loaded",
        json!({
            "pareto_rows": train_rows.len(),
            "minibatch_rows": minibatch_rows.len(),
            "reflection_rows": reflection_rows.len(),
            "heldout_rows": heldout_rows.len(),
            "task_pools": {
                "pareto": pareto_ids,
                "minibatch": minibatch_ids,
                "reflection": reflection_ids,
                "heldout": heldout_ids,
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
            Some(&context.transitions),
            OptimizerRunState::Ready,
            OptimizerTransitionTrigger::ContainerReady,
            "Container, program, and taskset ready",
            json!({
                "pareto_rows": train_rows.len(),
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
    let mut candidates: Vec<CandidateRecord> = cursor
        .candidates
        .as_array()
        .filter(|rows| !rows.is_empty())
        .map(|_| serde_json::from_value(cursor.candidates.clone()))
        .transpose()?
        .unwrap_or_default();
    hydrate_candidate_sensor_frames_from_workspace(
        &context.workspace,
        &context.config.run.run_id,
        &mut candidates,
    )?;
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
    state.cursor.candidates =
        serde_json::to_value(checkpoint_candidate_records(&state.candidates))?;
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
        .record_checkpoint_compacting_previous(&context.config.run.run_id, &checkpoint)?;
    emit_limit_estimate_update_if_major(context, state.checkpoint_sequence)?;
    Ok(())
}

fn emit_limit_estimate_update_if_major(
    context: &mut GepaRunContext,
    checkpoint_sequence: u64,
) -> Result<()> {
    if !limit_estimate_projection_due(context, checkpoint_sequence) {
        return Ok(());
    }
    let checked_at = Instant::now();
    let snapshot = project_gepa_limit_snapshot(&context.workspace, &context.config)?;
    context.last_limit_estimate_check = Some(checked_at);
    context.last_limit_estimate_checkpoint_sequence = checkpoint_sequence;
    let Some(payload) = limit_estimate_update_payload(&snapshot) else {
        return Ok(());
    };
    if !limit_estimate_update_is_major(context.events.records(), &payload) {
        return Ok(());
    }
    context.events.emit(
        "optimizer.limit.estimate_updated",
        "Limit ETA estimate updated",
        payload,
    )?;
    context
        .workspace
        .record_event_stream(&context.config.run.run_id, context.events.records())
}

fn limit_estimate_projection_due(context: &GepaRunContext, checkpoint_sequence: u64) -> bool {
    let has_prior_estimate = context
        .events
        .records()
        .iter()
        .rev()
        .any(|event| event.event_type == "optimizer.limit.estimate_updated");
    if !has_prior_estimate {
        return true;
    }
    let interval_due = context
        .last_limit_estimate_check
        .map(|last| last.elapsed() >= LIMIT_ESTIMATE_UPDATE_MIN_INTERVAL)
        .unwrap_or(true);
    let checkpoint_due = checkpoint_sequence
        .saturating_sub(context.last_limit_estimate_checkpoint_sequence)
        >= LIMIT_ESTIMATE_UPDATE_CHECKPOINT_STEP;
    interval_due || checkpoint_due
}

fn limit_estimate_update_payload(snapshot: &LimitSnapshot) -> Option<Value> {
    let nearest = nearest_limit_status(snapshot)?;
    Some(json!({
        "run_id": snapshot.run_id,
        "generated_at": snapshot.generated_at,
        "nearest": limit_status_summary(nearest),
        "limits": snapshot
            .limits
            .iter()
            .map(limit_status_summary)
            .collect::<Vec<_>>(),
    }))
}

fn nearest_limit_status(snapshot: &LimitSnapshot) -> Option<&LimitStatus> {
    let nearest = snapshot.nearest_limit.as_ref()?;
    snapshot
        .limits
        .iter()
        .find(|status| status.definition.limit_id == nearest.limit_id)
}

fn limit_status_summary(status: &LimitStatus) -> Value {
    json!({
        "limit_id": status.definition.limit_id,
        "kind": status.definition.kind.as_str(),
        "source": status.definition.source,
        "hard": status.definition.hard,
        "spent": status.spent,
        "reserved": status.reserved,
        "remaining": status.remaining,
        "utilization": status.utilization,
        "max_value": status.definition.max_value,
        "forecast": {
            "model": status.forecast.model,
            "predicted_crossing_at": status.forecast.predicted_crossing_at,
            "seconds_to_limit": status.forecast.seconds_to_limit,
            "seconds_to_limit_low": status.forecast.seconds_to_limit_low,
            "seconds_to_limit_high": status.forecast.seconds_to_limit_high,
            "predicted_crossing_at_low": status.forecast.predicted_crossing_at_low,
            "predicted_crossing_at_high": status.forecast.predicted_crossing_at_high,
            "rate_per_second": status.forecast.rate_per_second,
            "confidence": status.forecast.confidence,
            "sample_count": status.forecast.sample_count,
            "updated_at": status.forecast.updated_at,
        },
    })
}

fn limit_estimate_update_is_major(events: &[EventStreamRecord], payload: &Value) -> bool {
    let Some(current) = payload.get("nearest") else {
        return false;
    };
    let Some(previous) = events
        .iter()
        .rev()
        .find(|event| event.event_type == "optimizer.limit.estimate_updated")
        .and_then(|event| event.fields.get("nearest"))
    else {
        return true;
    };
    if json_str(current, "limit_id") != json_str(previous, "limit_id") {
        return true;
    }
    if json_str(current, "kind") != json_str(previous, "kind") {
        return true;
    }
    let current_eta = current
        .get("forecast")
        .and_then(|forecast| json_f64(forecast, "seconds_to_limit"));
    let previous_eta = previous
        .get("forecast")
        .and_then(|forecast| json_f64(forecast, "seconds_to_limit"));
    match (previous_eta, current_eta) {
        (None, Some(_)) | (Some(_), None) => return true,
        (Some(previous), Some(current)) => {
            if current == 0.0 && previous > 0.0 {
                return true;
            }
            let delta = (current - previous).abs();
            let relative = delta / previous.max(1.0);
            if delta >= 60.0 && relative >= 0.10 {
                return true;
            }
        }
        (None, None) => {}
    }
    let utilization_delta = (json_f64(current, "utilization").unwrap_or(0.0)
        - json_f64(previous, "utilization").unwrap_or(0.0))
    .abs();
    if utilization_delta >= 0.05 {
        return true;
    }
    current
        .get("forecast")
        .and_then(|forecast| json_str(forecast, "confidence"))
        != previous
            .get("forecast")
            .and_then(|forecast| json_str(forecast, "confidence"))
}

fn json_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn json_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn metadata_with_pipeline_state(
    context: &GepaRunContext,
    state: &mut GepaRunState,
    mut metadata: Map<String, Value>,
) -> Result<Map<String, Value>> {
    match GepaPipelineRuntimePlan::from_config(&context.config)? {
        GepaPipelineRuntimePlan::AsyncPipelined(plan)
        | GepaPipelineRuntimePlan::FlashEvolve(plan) => {
            refresh_async_pipeline_cursor_state(context, state, &plan);
            metadata.insert("pipeline".to_string(), plan_metadata(&plan));
            metadata.insert(
                "pipeline_state".to_string(),
                pipeline_state_metadata_summary(&state.cursor.pipeline_state),
            );
        }
        GepaPipelineRuntimePlan::SyncSerial(_) => {}
    }
    Ok(metadata)
}

fn pipeline_state_metadata_summary(state: &planner::GepaAsyncPipelineCursorState) -> Value {
    json!({
        "pool_version": state.pool_version,
        "parent_pool_version": state.parent_pool_version,
        "parent_candidate_id": state.parent_candidate_id,
        "in_flight_candidate_count": state.in_flight_candidate_count,
        "propose_queue_count": state.propose_queue.len(),
        "rollout_queue_count": state.rollout_queue.len(),
        "evaluate_queue_count": state.evaluate_queue.len(),
        "lane_lease_count": state.lane_leases.len(),
        "pending_job_ids": &state.pending_job_ids,
        "pending_effect_ids": &state.pending_effect_ids,
        "candidate_partial_count": state.candidate_partials.len(),
        "speculative_release_count": state.speculative_releases.len(),
        "staleness_review_count": state.staleness_reviews.len(),
        "rollout_concurrency": {
            "initialized": state.adaptive_rollout_concurrency.initialized,
            "current_limit": state.adaptive_rollout_concurrency.current_limit,
            "completed_rollouts": state.adaptive_rollout_concurrency.completed_rollouts,
            "overload_count": state.adaptive_rollout_concurrency.overload_count,
        },
        "rollout_resilience": {
            "scored_rollouts": state.rollout_resilience.scored_rollouts,
            "degraded_rollouts": state.rollout_resilience.degraded_rollouts,
            "last_failure_rate": state.rollout_resilience.last_failure_rate,
        },
    })
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
            advance_gepa_async_pipeline_once(context, state, mode, options, &plan)
        }
        GepaPipelineRuntimePlan::FlashEvolve(plan) => {
            advance_gepa_async_pipeline_once(context, state, mode, options, &plan)
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
        let result = terminal_result_from_cursor(context, &state.cursor)?;
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

fn advance_gepa_async_pipeline_once(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    mode: GepaAdvanceMode,
    options: &GepaExecutionOptions,
    plan: &GepaAsyncPipelinePlan,
) -> Result<GepaAdvanceOutcome> {
    let pipeline_label = plan.label();
    refresh_async_pipeline_cursor_state(context, state, plan);
    if matches!(state.cursor.phase, GepaCursorPhase::Completed) {
        refresh_terminal_run_projection(context, state)?;
        let result = terminal_result_from_cursor(context, &state.cursor)?;
        return Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::TerminalizeRun {
                run_id: context.config.run.run_id.clone(),
                status: "completed".to_string(),
            },
            terminal: true,
            result,
            message: format!("{pipeline_label}: GEPA run already completed"),
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
                "{pipeline_label}: GEPA run already {}",
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
    ensure_adaptive_stage_workers_state(state, plan);

    // Old Phase-1 async cursors used the serial pending-job slot. Finish that
    // in-place before switching the cursor over to lane leases.
    if let Some(job_id) = state.cursor.pending_job_id.clone() {
        let mut outcome =
            advance_pending_runtime_job(context, state, &resources, mode, &job_id, Some(plan))?;
        outcome.message = format!("{pipeline_label} legacy lane: {}", outcome.message);
        return Ok(outcome);
    }

    if matches!(state.cursor.phase, GepaCursorPhase::Initializing) {
        let mut outcome = advance_gepa_sync_serial_once(context, state, mode, options)?;
        outcome.message = format!("{pipeline_label} seed: {}", outcome.message);
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
            if before_rollout_leases >= adaptive_rollout_lane_limit(state, plan) {
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
                "{pipeline_label}: planned {planned_count} rollout jobs to fill lane capacity"
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
        outcome.message = format!("{pipeline_label} terminal: {}", outcome.message);
        return Ok(outcome);
    }
    if async_pipeline_idle(state) && async_pipeline_stopper_satisfied(context, state) {
        let mut outcome = advance_gepa_sync_serial_once(context, state, mode, options)?;
        outcome.message = format!("{pipeline_label} terminal: {}", outcome.message);
        return Ok(outcome);
    }

    adjust_adaptive_stage_workers(context, state, plan)?;
    refresh_async_pipeline_cursor_state(context, state, plan);
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::Noop,
        terminal: false,
        result: None,
        message: format!("{pipeline_label}: waiting for lane capacity or completions"),
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

fn plan_metadata(plan: &GepaAsyncPipelinePlan) -> Value {
    match plan.mode {
        GepaPipelineMode::AsyncPipelined => {
            GepaPipelineRuntimePlan::AsyncPipelined(plan.clone()).metadata()
        }
        GepaPipelineMode::FlashEvolve => {
            GepaPipelineRuntimePlan::FlashEvolve(plan.clone()).metadata()
        }
        GepaPipelineMode::SyncSerial => GepaPipelineRuntimePlan::SyncSerial(GepaSyncSerialPlan {
            rollout_transport: plan.rollout_transport.clone(),
        })
        .metadata(),
    }
}

fn refresh_async_pipeline_cursor_state(
    context: &GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinePlan,
) {
    let pool_version = state
        .candidates
        .iter()
        .filter(|candidate| candidate_train_selectable(candidate))
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
        "adaptive_stage_workers": adaptive_stage_workers_snapshot(state, plan),
        "speculative_releases": state.cursor.pipeline_state.speculative_releases.len(),
        "staleness_reviews": state.cursor.pipeline_state.staleness_reviews.len(),
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
    plan: &GepaAsyncPipelinePlan,
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
                    state.cursor.pipeline_state.parent_pool_version =
                        Some(lease.parent_pool_version);
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
                        maybe_enqueue_speculative_evaluate_item(
                            context,
                            state,
                            resources,
                            plan,
                            &partial_id,
                            lease.parent_pool_version,
                            &active,
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
                outcome.message = format!("{} {}: {}", plan.label(), lease.lane, outcome.message);
                return Ok(Some(outcome));
            }
            OptimizerJobStatus::Failed
            | OptimizerJobStatus::Cancelled
            | OptimizerJobStatus::Expired => {
                if let Some(outcome) =
                    schedule_failed_rollout_retry_if_allowed(context, state, resources, &job)?
                {
                    if let Some(updated) =
                        state.cursor.pipeline_state.lane_leases.get_mut(&lease_key)
                    {
                        updated.status = "retry_scheduled".to_string();
                    }
                    return Ok(Some(outcome));
                }
                if lease.lane == "rollout" {
                    restore_async_partial_as_active(state, lease.partial_id.as_deref())?;
                    state.cursor.pending_job_id = Some(job_id.clone());
                    state.cursor.pending_effect_id = lease.effect_id.clone();
                    state.cursor.pending_reservation_ids = lease.reservation_ids.clone();
                    if consume_failed_rollout_job_as_degraded(context, state, resources, &job)? {
                        state.cursor.pipeline_state.lane_leases.remove(&lease_key);
                        state.cursor.pending_job_id = None;
                        state.cursor.pending_effect_id = None;
                        state.cursor.pending_reservation_ids.clear();
                        if let Some(active) = state.active_evaluation.clone() {
                            let partial_id = lease.partial_id.clone().unwrap_or_else(|| {
                                async_partial_id(&active.stage, active.generation)
                            });
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
                                    "degraded": true,
                                }),
                            )?;
                            maybe_enqueue_speculative_evaluate_item(
                                context,
                                state,
                                resources,
                                plan,
                                &partial_id,
                                lease.parent_pool_version,
                                &active,
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
                        refresh_async_pipeline_cursor_state(context, state, plan);
                        persist_gepa_run_state(
                            context,
                            state,
                            resources,
                            state.cursor.phase.clone(),
                            "completed",
                            "degraded async rollout runtime outcome",
                            Map::new(),
                        )?;
                        return Ok(Some(GepaAdvanceOutcome {
                            action: planner::GepaTickAction::ConsumeRuntimeOutcome {
                                run_id: context.config.run.run_id.clone(),
                                job_id,
                            },
                            terminal: false,
                            result: None,
                            message: format!(
                                "{} rollout: degraded failed runtime job",
                                plan.label()
                            ),
                        }));
                    }
                }
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
        if speculative_tail_obsolete(state, &item) {
            absorb_obsolete_speculative_tail(context, state, &item)?;
            refresh_async_pipeline_cursor_state(context, state, plan);
            let mut outcome = move_to_proposer_waiting(
                context,
                state,
                resources,
                "discarded obsolete speculative tail work",
            )?;
            outcome.message = format!(
                "{} evaluate: discarded obsolete speculative tail {}",
                plan.label(),
                item.item_id
            );
            return Ok(Some(outcome));
        }
        let current_pool_version = state.cursor.pipeline_state.pool_version;
        let stale_decision =
            plan.stale_item_disposition(item.parent_pool_version, current_pool_version);
        match stale_decision.disposition {
            GepaStaleItemDisposition::AcceptAsIs => {}
            GepaStaleItemDisposition::Discard => {
                discard_stale_evaluate_item(context, state, &item, &stale_decision)?;
                refresh_async_pipeline_cursor_state(context, state, plan);
                let mut outcome = move_to_proposer_waiting(
                    context,
                    state,
                    resources,
                    "discarded stale async evaluate work",
                )?;
                outcome.message = format!(
                    "{} evaluate: discarded stale work item {} ({})",
                    plan.label(),
                    item.item_id,
                    stale_decision.reason
                );
                return Ok(Some(outcome));
            }
            GepaStaleItemDisposition::ReflectivePatch => {
                let review = run_reflective_staleness_review(
                    context,
                    state,
                    resources,
                    &item,
                    &stale_decision,
                )?;
                record_reflective_staleness_review(state, &item, &stale_decision, &review);
                match review.verdict {
                    ReflectiveStalenessVerdict::Accept => {
                        attach_staleness_acceptance_metadata(
                            context,
                            state,
                            &item,
                            &stale_decision,
                            Some(&review),
                        )?;
                    }
                    ReflectiveStalenessVerdict::Discard => {
                        discard_stale_evaluate_item(context, state, &item, &stale_decision)?;
                        refresh_async_pipeline_cursor_state(context, state, plan);
                        let mut outcome = move_to_proposer_waiting(
                            context,
                            state,
                            resources,
                            "discarded reflectively reviewed stale async evaluate work",
                        )?;
                        outcome.message = format!(
                            "{} evaluate: reviewer discarded stale work item {} ({})",
                            plan.label(),
                            item.item_id,
                            review.reason
                        );
                        return Ok(Some(outcome));
                    }
                    ReflectiveStalenessVerdict::Patch => {
                        let outcome = requeue_reflective_patch_candidate(
                            context,
                            state,
                            resources,
                            plan,
                            &item,
                            &stale_decision,
                            &review,
                        )?;
                        return Ok(Some(outcome));
                    }
                }
            }
        }
        if !matches!(
            stale_decision.disposition,
            GepaStaleItemDisposition::ReflectivePatch
        ) {
            attach_staleness_acceptance_metadata(context, state, &item, &stale_decision, None)?;
        }
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
        outcome.message = format!("{} evaluate: {}", plan.label(), outcome.message);
        return Ok(Some(outcome));
    }
    Ok(None)
}

fn attach_staleness_acceptance_metadata(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    item: &GepaAsyncLaneWorkItem,
    decision: &GepaStaleItemDecision,
    review: Option<&ReflectiveStalenessReview>,
) -> Result<()> {
    if decision.stale_gap == 0 && review.is_none() {
        return Ok(());
    }
    for candidate_id in &item.candidate_ids {
        let Some(candidate) = state
            .candidates
            .iter_mut()
            .find(|candidate| &candidate.candidate_id == candidate_id)
        else {
            continue;
        };
        candidate.acceptance_metadata.insert(
            "staleness".to_string(),
            json!({
                "disposition": "accept",
                "reason": review.map(|review| review.reason.as_str()).unwrap_or(decision.reason.as_str()),
                "stale_gap": decision.stale_gap,
                "parent_pool_version": item.parent_pool_version,
                "current_pool_version": decision.current_pool_version,
                "stage": item.stage.as_str(),
                "item_id": item.item_id.as_str(),
                "review_id": review.map(|review| review.review_id.as_str()),
                "reviewer_workspace": review.and_then(|review| review.workspace.as_deref()),
            }),
        );
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            candidate,
        )?;
    }
    if let Some(review) = review {
        context.events.emit(
            "pipeline.stale_item.reviewed",
            "Stale pipeline item accepted by reviewer",
            json!({
                "review_id": review.review_id,
                "item_id": item.item_id,
                "verdict": "accept",
                "reason": review.reason,
                "candidate_ids": item.candidate_ids,
                "stale_gap": decision.stale_gap,
                "workspace": review.workspace,
            }),
        )?;
    }
    Ok(())
}

fn run_reflective_staleness_review(
    context: &mut GepaRunContext,
    state: &GepaRunState,
    resources: &GepaStepResources,
    item: &GepaAsyncLaneWorkItem,
    decision: &GepaStaleItemDecision,
) -> Result<ReflectiveStalenessReview> {
    let candidate_ids = if item.candidate_ids.is_empty() {
        state
            .active_evaluation
            .as_ref()
            .map(candidate_ids_for_active)
            .unwrap_or_default()
    } else {
        item.candidate_ids.clone()
    };
    let stale_candidates = candidate_ids
        .iter()
        .filter_map(|candidate_id| {
            state
                .candidates
                .iter()
                .find(|candidate| &candidate.candidate_id == candidate_id)
                .cloned()
        })
        .collect::<Vec<_>>();
    let current_best = state
        .best_idx
        .and_then(|idx| state.candidates.get(idx))
        .cloned();
    let review_workspace = context
        .paths
        .run_dir
        .join("staleness_reviews")
        .join(stable_gepa_id(
            "gepa_review",
            &[&item.item_id, &decision.current_pool_version.to_string()],
        ));
    let response = codex_app_server::run_codex_staleness_reviewer(
        codex_app_server::CodexStalenessReviewerInput {
            config: &context.config,
            program: &resources.program,
            item: serde_json::to_value(item)?,
            stale_candidates,
            current_best,
            pool_summary: reflective_pool_summary(state, decision),
            workspace_dir: review_workspace,
        },
    )?;
    let verdict_value = response.get("verdict").cloned().ok_or_else(|| {
        OptimizerError::Proposer(
            "reflective staleness reviewer response missing verdict".to_string(),
        )
    })?;
    reflective_review_from_value(response, verdict_value)
}

fn reflective_review_from_value(
    response: Value,
    verdict_value: Value,
) -> Result<ReflectiveStalenessReview> {
    let verdict_text = verdict_value
        .get("verdict")
        .or_else(|| verdict_value.get("decision"))
        .and_then(Value::as_str)
        .unwrap_or("accept")
        .trim()
        .to_ascii_lowercase();
    let verdict = match verdict_text.as_str() {
        "accept" | "accept_as_is" => ReflectiveStalenessVerdict::Accept,
        "discard" | "drop" => ReflectiveStalenessVerdict::Discard,
        "patch" | "repair" => ReflectiveStalenessVerdict::Patch,
        other => {
            return Err(OptimizerError::Proposer(format!(
                "reflective staleness reviewer returned unsupported verdict {other:?}"
            )));
        }
    };
    let reason = verdict_value
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("reflective reviewer did not provide a reason")
        .to_string();
    let patched_payload_value = verdict_value
        .get("patched_payload")
        .or_else(|| verdict_value.get("payload"));
    let patched_payload = match patched_payload_value {
        Some(Value::Object(_)) => Some(
            serde_json::from_value(
                patched_payload_value
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new())),
            )
            .map_err(|source| {
                OptimizerError::Proposer(format!(
                    "reflective staleness reviewer returned invalid patched_payload: {source}"
                ))
            })?,
        ),
        Some(Value::Null) | None => None,
        Some(_) => {
            return Err(OptimizerError::Proposer(
                "reflective staleness reviewer patched_payload must be an object or null"
                    .to_string(),
            ));
        }
    };
    if matches!(verdict, ReflectiveStalenessVerdict::Patch) && patched_payload.is_none() {
        return Err(OptimizerError::Proposer(
            "reflective staleness reviewer verdict patch requires patched_payload".to_string(),
        ));
    }
    let review_id = verdict_value
        .get("review_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| stable_gepa_id("gepa_review", &[&reason, &verdict_text]));
    let workspace = response
        .get("workspace")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(ReflectiveStalenessReview {
        review_id,
        verdict,
        reason,
        patched_payload,
        workspace,
        raw: response,
    })
}

fn reflective_pool_summary(state: &GepaRunState, decision: &GepaStaleItemDecision) -> Value {
    let frontier = frontier_members(&state.candidates);
    let recent_pool = state
        .candidates
        .iter()
        .filter(|candidate| candidate_train_selectable(candidate))
        .rev()
        .take(decision.stale_gap.max(1) as usize)
        .cloned()
        .collect::<Vec<_>>();
    json!({
        "current_pool_version": decision.current_pool_version,
        "stale_gap": decision.stale_gap,
        "best_candidate_id": state.cursor.best_candidate_id,
        "frontier": frontier,
        "recent_pool_items": recent_pool,
    })
}

fn record_reflective_staleness_review(
    state: &mut GepaRunState,
    item: &GepaAsyncLaneWorkItem,
    decision: &GepaStaleItemDecision,
    review: &ReflectiveStalenessReview,
) {
    state
        .cursor
        .pipeline_state
        .staleness_reviews
        .push(GepaStalenessReviewRecord {
            review_id: review.review_id.clone(),
            item_id: item.item_id.clone(),
            stage: item.stage.clone(),
            generation: item.generation,
            candidate_ids: item.candidate_ids.clone(),
            verdict: match review.verdict {
                ReflectiveStalenessVerdict::Accept => "accept",
                ReflectiveStalenessVerdict::Discard => "discard",
                ReflectiveStalenessVerdict::Patch => "patch",
            }
            .to_string(),
            reason: review.reason.clone(),
            stale_gap: decision.stale_gap,
            parent_pool_version: item.parent_pool_version,
            current_pool_version: decision.current_pool_version,
            reviewer_workspace: review.workspace.clone(),
        });
}

fn requeue_reflective_patch_candidate(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinePlan,
    item: &GepaAsyncLaneWorkItem,
    decision: &GepaStaleItemDecision,
    review: &ReflectiveStalenessReview,
) -> Result<GepaAdvanceOutcome> {
    let patched_payload = review.patched_payload.clone().ok_or_else(|| {
        OptimizerError::Proposer("reflective patch verdict missing patched payload".to_string())
    })?;
    let parent_id = state
        .best_idx
        .and_then(|idx| state.candidates.get(idx))
        .map(|candidate| candidate.candidate_id.clone())
        .or_else(|| item.parent_candidate_id.clone())
        .ok_or_else(|| {
            OptimizerError::Invariant(
                "reflective patch has no current best or stale parent candidate".to_string(),
            )
        })?;
    for candidate_id in &item.candidate_ids {
        if let Some(candidate) = state
            .candidates
            .iter_mut()
            .find(|candidate| &candidate.candidate_id == candidate_id)
        {
            candidate.status = "patched_stale".to_string();
            candidate.acceptance_metadata.insert(
                "staleness".to_string(),
                json!({
                    "disposition": "patch",
                    "review_id": review.review_id,
                    "reason": review.reason,
                    "stale_gap": decision.stale_gap,
                    "current_pool_version": decision.current_pool_version,
                }),
            );
            persist_candidate_snapshot(
                &mut context.workspace,
                &context.config.run.run_id,
                candidate,
            )?;
        }
    }
    let patched_candidate_id = candidate_id(&patched_payload);
    let patched_idx = if let Some(existing_idx) = state
        .candidates
        .iter()
        .position(|candidate| candidate.candidate_id == patched_candidate_id)
    {
        existing_idx
    } else {
        let lever_bundle = LeverBundle::from_prompt_payload(
            patched_candidate_id.clone(),
            Some(parent_id.clone()),
            &patched_payload,
        );
        let candidate = CandidateRecord {
            lever_bundle,
            candidate_id: patched_candidate_id.clone(),
            payload: patched_payload,
            parent_id: Some(parent_id.clone()),
            source: "reflective_staleness_patch".to_string(),
            status: "registered".to_string(),
            minibatch_reward: None,
            train_reward: None,
            heldout_reward: None,
            minibatch_scores: Vec::new(),
            train_scores: Vec::new(),
            sensor_frames: Vec::new(),
            acceptance_score: Value::Null,
            acceptance_metadata: json_map(vec![
                ("generation", json!(state.cursor.generation)),
                ("proposal", review.raw.clone()),
                ("staleness_review_id", json!(review.review_id)),
                ("staleness_patch_parent", json!(parent_id)),
            ]),
        };
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            &candidate,
        )?;
        record_candidate_registered(
            context,
            &candidate,
            Some(state.cursor.generation),
            json!({
                "source": &candidate.source,
                "parent_id": &candidate.parent_id,
                "generation": state.cursor.generation,
                "staleness_review_id": &review.review_id,
            }),
        )?;
        state.candidates.push(candidate);
        state.candidates.len() - 1
    };
    let minibatch_rows = minibatch_rows(
        &resources.minibatch_rows,
        &context.config.gepa.batch_sampler,
        context.config.gepa.minibatch_size,
        state.cursor.generation,
        item.proposal_index,
        context.config.gepa.proposals_per_generation,
    );
    let mut active = new_rollout_evaluation(
        "candidate_minibatch",
        patched_idx,
        &minibatch_rows,
        state.cursor.generation,
        item.proposal_index,
        None,
    )?;
    active.candidate_id = Some(patched_candidate_id.clone());
    active.parent_id = Some(parent_id.clone());
    let partial_id = format!(
        "async:reflective_patch:candidate_minibatch:generation_{:03}:{}",
        state.cursor.generation, review.review_id
    );
    state.active_evaluation = Some(active.clone());
    upsert_async_partial_from_active(state, &partial_id, "rollout", decision.current_pool_version)?;
    let rollout_item = async_work_item_from_active(
        &active,
        "rollout",
        decision.current_pool_version,
        Some(partial_id.clone()),
    )?;
    state.cursor.pipeline_state.rollout_queue.push(rollout_item);
    if let Some(stale_partial_id) = item.partial_id.as_ref() {
        state
            .cursor
            .pipeline_state
            .candidate_partials
            .remove(stale_partial_id);
    }
    state.active_evaluation = None;
    context.events.emit(
        "pipeline.stale_item.patched",
        "Stale pipeline item patched by reviewer",
        json!({
            "review_id": review.review_id,
            "item_id": item.item_id,
            "stale_candidate_ids": item.candidate_ids,
            "patched_candidate_id": patched_candidate_id,
            "parent_candidate_id": parent_id,
            "stale_gap": decision.stale_gap,
            "workspace": review.workspace,
        }),
    )?;
    refresh_async_pipeline_cursor_state(context, state, plan);
    persist_gepa_run_state(
        context,
        state,
        resources,
        state.cursor.phase.clone(),
        "planned",
        "queued reflective staleness patch candidate",
        Map::new(),
    )?;
    Ok(GepaAdvanceOutcome {
        action: planner::GepaTickAction::CheckpointRun {
            run_id: context.config.run.run_id.clone(),
            phase: state.cursor.phase.as_str().to_string(),
        },
        terminal: false,
        result: None,
        message: format!(
            "{} evaluate: reviewer patched stale work item {} as {}",
            plan.label(),
            item.item_id,
            patched_candidate_id
        ),
    })
}

fn speculative_tail_obsolete(state: &GepaRunState, item: &GepaAsyncLaneWorkItem) -> bool {
    let Some(partial_id) = item.partial_id.as_deref() else {
        return false;
    };
    let Some(partial) = state
        .cursor
        .pipeline_state
        .candidate_partials
        .get(partial_id)
    else {
        return false;
    };
    if !partial
        .metadata
        .get("speculative_tail")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return false;
    }
    let candidate_ids = if item.candidate_ids.is_empty() {
        &partial.candidate_ids
    } else {
        &item.candidate_ids
    };
    !candidate_ids.is_empty()
        && candidate_ids.iter().all(|candidate_id| {
            state
                .candidates
                .iter()
                .find(|candidate| &candidate.candidate_id == candidate_id)
                .is_some_and(|candidate| {
                    matches!(
                        candidate.status.as_str(),
                        "accepted"
                            | "rejected_full_train"
                            | "rejected_minibatch"
                            | "discarded_stale"
                            | "patched_stale"
                            | "deferred_budget"
                    )
                })
        })
}

fn absorb_obsolete_speculative_tail(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    item: &GepaAsyncLaneWorkItem,
) -> Result<()> {
    let active = state.active_evaluation.take();
    if let Some(active) = active {
        if active.is_group() {
            for candidate in &active.candidate_evaluations {
                let eval = evaluation_from_active_candidate(candidate);
                state.total_usage.merge(&eval.usage);
                state.total_cost += eval.cost_usd;
                state.rollout_count += eval.rollout_count;
                append_rollout_usage(&mut state.usage_ledger, &eval);
            }
        } else {
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
        }
    }
    if let Some(partial_id) = item.partial_id.as_ref() {
        state
            .cursor
            .pipeline_state
            .candidate_partials
            .remove(partial_id);
    }
    context.events.emit(
        "pipeline.speculative_tail.discarded",
        "Obsolete speculative tail discarded",
        json!({
            "item_id": item.item_id,
            "partial_id": item.partial_id,
            "stage": item.stage,
            "generation": item.generation,
            "candidate_ids": item.candidate_ids,
        }),
    )?;
    Ok(())
}

fn discard_stale_evaluate_item(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    item: &GepaAsyncLaneWorkItem,
    decision: &GepaStaleItemDecision,
) -> Result<()> {
    let active = state.active_evaluation.take();
    let candidate_ids = active
        .as_ref()
        .map(candidate_ids_for_active)
        .filter(|ids| !ids.is_empty())
        .unwrap_or_else(|| item.candidate_ids.clone());
    for candidate_id in &candidate_ids {
        let Some(candidate) = state
            .candidates
            .iter_mut()
            .find(|candidate| &candidate.candidate_id == candidate_id)
        else {
            continue;
        };
        if matches!(
            candidate.status.as_str(),
            "accepted" | "full_train_evaluated" | "rejected_full_train"
        ) {
            continue;
        }
        candidate.status = "discarded_stale".to_string();
        candidate.acceptance_metadata.insert(
            "staleness".to_string(),
            json!({
                "disposition": "discard",
                "reason": decision.reason.as_str(),
                "stale_gap": decision.stale_gap,
                "parent_pool_version": item.parent_pool_version,
                "current_pool_version": decision.current_pool_version,
                "stage": item.stage.as_str(),
                "item_id": item.item_id.as_str(),
            }),
        );
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            candidate,
        )?;
    }
    if let Some(partial_id) = item.partial_id.as_ref() {
        state
            .cursor
            .pipeline_state
            .candidate_partials
            .remove(partial_id);
    }
    context.events.emit(
        "pipeline.stale_item.discarded",
        "Stale pipeline item discarded",
        json!({
            "item_id": item.item_id.as_str(),
            "partial_id": item.partial_id.as_deref(),
            "stage": item.stage.as_str(),
            "generation": item.generation,
            "candidate_ids": candidate_ids,
            "parent_pool_version": item.parent_pool_version,
            "current_pool_version": decision.current_pool_version,
            "stale_gap": decision.stale_gap,
            "reason": decision.reason.as_str(),
            "policy": context.config.gepa.pipeline.staleness_policy.as_str(),
            "mode": context.config.gepa.pipeline.mode.as_str(),
        }),
    )?;
    Ok(())
}

fn schedule_async_lane_transition(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    mode: GepaAdvanceMode,
    plan: &GepaAsyncPipelinePlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if let Some(outcome) = execute_async_leased_runtime_job(context, state, resources, mode, plan)?
    {
        return Ok(Some(outcome));
    }

    if !plan.uses_generation_barrier() {
        if async_stage_work_pending(state, &["seed_full_train"]) {
            if let Some(outcome) = schedule_async_rollout_job(context, state, resources, plan)? {
                return Ok(Some(outcome));
            }
            return Ok(None);
        }
        if let Some(outcome) =
            schedule_async_candidate_minibatches(context, state, resources, plan)?
        {
            return Ok(Some(outcome));
        }
        if async_stage_work_pending(state, &["candidate_full_train"]) {
            if let Some(outcome) = schedule_async_proposer_job(context, state, resources, plan)? {
                return Ok(Some(outcome));
            }
        }
        if let Some(outcome) = schedule_async_rollout_job(context, state, resources, plan)? {
            return Ok(Some(outcome));
        }
        if let Some(outcome) = schedule_async_proposer_job(context, state, resources, plan)? {
            return Ok(Some(outcome));
        }
        return Ok(None);
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
    plan: &GepaAsyncPipelinePlan,
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
        outcome.message = format!("{} seed: {}", plan.label(), outcome.message);
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
        message: format!("{} seed: queued seed full-train rollout work", plan.label()),
    })
}

fn execute_async_leased_runtime_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    mode: GepaAdvanceMode,
    plan: &GepaAsyncPipelinePlan,
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
        outcome.message = format!("{} {}: {}", plan.label(), lease.lane, outcome.message);
        return Ok(Some(outcome));
    }
    Ok(None)
}

fn schedule_async_rollout_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinePlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if async_lane_lease_count(state, "rollout") >= adaptive_rollout_lane_limit(state, plan) {
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
        outcome.message = format!("{} rollout: {}", plan.label(), outcome.message);
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
            "configured_rollout_workers": adaptive_rollout_lane_limit(state, plan),
            "static_rollout_workers": plan.rollout_workers,
            "adaptive_rollout_concurrency": adaptive_rollout_snapshot(state, plan),
            "adaptive_stage_workers": adaptive_stage_workers_snapshot(state, plan),
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
    outcome.message = format!("{} rollout: {}", plan.label(), outcome.message);
    Ok(Some(outcome))
}

fn schedule_async_candidate_minibatches(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinePlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if state.proposal_queue.is_empty()
        || state.cursor.pipeline_state.in_flight_candidate_count >= plan.max_in_flight_candidates
    {
        return Ok(None);
    }
    if plan.uses_generation_barrier()
        && async_stage_work_pending(
            state,
            &["parent_minibatch_reference", "candidate_minibatch"],
        )
    {
        return Ok(None);
    }
    state.cursor.phase = GepaCursorPhase::ProposerWaiting;
    let before_generation = state.cursor.generation;
    let mut outcome = advance_proposer_waiting(context, state, resources)?;
    if let Some(active) = state.active_evaluation.clone() {
        let partial_id = async_partial_id(&active.stage, active.generation);
        let parent_pool_version = state
            .cursor
            .pipeline_state
            .parent_pool_version
            .unwrap_or(state.cursor.pipeline_state.pool_version);
        upsert_async_partial_from_active(state, &partial_id, "rollout", parent_pool_version)?;
        let item =
            async_work_item_from_active(&active, "rollout", parent_pool_version, Some(partial_id))?;
        state.cursor.pipeline_state.rollout_queue.push(item);
        state.active_evaluation = None;
        if !plan.uses_generation_barrier()
            && state.cursor.proposal_index >= state.proposal_queue.len()
        {
            state.proposal_queue.clear();
            state.cursor.proposal_index = 0;
            state.cursor.generation = before_generation.saturating_add(1);
            state.cursor.pipeline_state.parent_candidate_id = None;
            state.cursor.pipeline_state.parent_pool_version = None;
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
    outcome.message = format!("{} candidate queue: {}", plan.label(), outcome.message);
    Ok(Some(outcome))
}

fn schedule_async_proposer_job(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinePlan,
) -> Result<Option<GepaAdvanceOutcome>> {
    if async_lane_lease_count(state, "propose")
        >= adaptive_stage_worker_limit(state, plan, "propose")
        || !state.cursor.pipeline_state.propose_queue.is_empty()
        || !state.proposal_queue.is_empty()
        || state.cursor.generation >= context.config.gepa.max_generations
        || state.cursor.pipeline_state.in_flight_candidate_count >= plan.max_in_flight_candidates
        || train_rollout_budget_reached(&context.config, state.rollout_count)
        || cost_budget_reached(&context.config, state.total_cost)
    {
        return Ok(None);
    }
    if plan.uses_generation_barrier()
        && async_stage_work_pending(
            state,
            &[
                "parent_minibatch_reference",
                "candidate_minibatch",
                "candidate_full_train",
            ],
        )
    {
        return Ok(None);
    }
    if let Some(train_best_idx) = select_best_train_candidate(
        &state.candidates,
        &resources.objective_set,
        &context.config.taskset.train_split,
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
            Some(&context.transitions),
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
    record_proposer_round_started(
        context,
        &queued.job.job_id,
        state.cursor.generation,
        &parent_id,
        json!({
            "job_id": &queued.job.job_id,
            "runtime_effect_id": &queued.effect.runtime_effect_id,
            "parent_candidate_id": &parent_id,
            "generation": state.cursor.generation,
            "model": &context.config.proposer.model,
            "provider": &context.config.proposer.provider,
        }),
    )?;
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
        message: format!("{} propose: planned proposer job", plan.label()),
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
    let existing_metadata = state
        .cursor
        .pipeline_state
        .candidate_partials
        .get(partial_id)
        .map(|partial| partial.metadata.clone())
        .unwrap_or_else(|| json!({}));
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
            metadata: merge_json_object(
                existing_metadata,
                json!({
                    "proposal_index": active.proposal_index,
                    "is_group": active.is_group(),
                }),
            ),
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

fn maybe_enqueue_speculative_evaluate_item(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    plan: &GepaAsyncPipelinePlan,
    source_partial_id: &str,
    parent_pool_version: u64,
    active: &GepaActiveEvaluation,
) -> Result<bool> {
    if !matches!(plan.mode, GepaPipelineMode::FlashEvolve)
        || !plan.speculative_completion.enabled
        || active.stage == "heldout"
        || active_rollout_evaluation_complete(active)
    {
        return Ok(false);
    }
    if !matches!(
        active.stage.as_str(),
        "candidate_minibatch" | "candidate_full_train"
    ) {
        return Ok(false);
    }
    if async_partial_has_speculative_release(state, source_partial_id) {
        return Ok(false);
    }
    let completed_rows = active_rollout_completed_rows(active);
    let total_rows = active_rollout_total_rows(active);
    if total_rows == 0 {
        return Ok(false);
    }
    let completion_fraction = completed_rows as f64 / total_rows as f64;
    if completion_fraction < plan.speculative_completion.alpha {
        return Ok(false);
    }
    let Some(release_active) = speculative_release_active(active) else {
        return Ok(false);
    };
    if !speculative_partial_exceeds_threshold(context, state, resources, &release_active)? {
        return Ok(false);
    }
    let release_id = stable_gepa_id(
        "gepa_spec",
        &[
            source_partial_id,
            &completed_rows.to_string(),
            &total_rows.to_string(),
        ],
    );
    let evaluate_partial_id = format!("{source_partial_id}:speculative:{release_id}");
    let mut active_clone = release_active;
    active_clone.planned_job_id = None;
    active_clone.effect_id = None;
    active_clone.reservation_id = None;
    let candidate_ids = candidate_ids_for_active(&active_clone);
    state.cursor.pipeline_state.candidate_partials.insert(
        evaluate_partial_id.clone(),
        GepaAsyncCandidatePartial {
            partial_id: evaluate_partial_id.clone(),
            lane: "evaluate".to_string(),
            stage: active_clone.stage.clone(),
            generation: active_clone.generation,
            parent_pool_version,
            parent_candidate_id: active_clone
                .candidate_id
                .clone()
                .or(active_clone.parent_id.clone()),
            candidate_ids: candidate_ids.clone(),
            active_evaluation: Some(serde_json::to_value(&active_clone)?),
            proposal_queue: Value::Null,
            rollout_chunks: BTreeMap::new(),
            metadata: json!({
                "speculative_release": true,
                "release_id": release_id,
                "source_partial_id": source_partial_id,
                "completed_rows": completed_rows,
                "total_rows": total_rows,
                "alpha": plan.speculative_completion.alpha,
            }),
        },
    );
    state
        .cursor
        .pipeline_state
        .evaluate_queue
        .push(GepaAsyncLaneWorkItem {
            item_id: evaluate_partial_id.clone(),
            lane: "evaluate".to_string(),
            stage: active_clone.stage.clone(),
            generation: active_clone.generation,
            proposal_index: active_clone.proposal_index,
            parent_candidate_id: active_clone
                .parent_id
                .clone()
                .or(active_clone.candidate_id.clone()),
            parent_pool_version,
            current_pool_version: Some(parent_pool_version),
            stale_gap: Some(0),
            candidate_ids: candidate_ids.clone(),
            partial_id: Some(evaluate_partial_id.clone()),
            job_id: None,
            effect_id: None,
            reservation_ids: Vec::new(),
            status: "speculative_release".to_string(),
            metadata: json!({
                "speculative_release": true,
                "release_id": release_id,
                "source_partial_id": source_partial_id,
                "completed_rows": completed_rows,
                "total_rows": total_rows,
                "alpha": plan.speculative_completion.alpha,
            }),
        });
    if let Some(source_partial) = state
        .cursor
        .pipeline_state
        .candidate_partials
        .get_mut(source_partial_id)
    {
        source_partial.metadata = merge_json_object(
            source_partial.metadata.clone(),
            json!({
                "speculative_tail": true,
                "speculative_release_id": release_id,
                "speculative_evaluate_partial_id": evaluate_partial_id,
            }),
        );
    }
    state
        .cursor
        .pipeline_state
        .speculative_releases
        .push(GepaSpeculativeReleaseRecord {
            release_id: release_id.clone(),
            source_partial_id: source_partial_id.to_string(),
            evaluate_partial_id: evaluate_partial_id.clone(),
            stage: active_clone.stage.clone(),
            generation: active_clone.generation,
            candidate_ids: candidate_ids.clone(),
            completed_rows,
            total_rows,
            alpha: plan.speculative_completion.alpha,
        });
    context.events.emit(
        "pipeline.speculative_release.enqueued",
        "Speculative pipeline release enqueued",
        json!({
            "release_id": release_id,
            "source_partial_id": source_partial_id,
            "evaluate_partial_id": evaluate_partial_id,
            "stage": active_clone.stage,
            "generation": active_clone.generation,
            "candidate_ids": candidate_ids,
            "completed_rows": completed_rows,
            "total_rows": total_rows,
            "alpha": plan.speculative_completion.alpha,
        }),
    )?;
    Ok(true)
}

fn speculative_release_active(active: &GepaActiveEvaluation) -> Option<GepaActiveEvaluation> {
    if active.is_group() {
        let complete_candidates = active
            .candidate_evaluations
            .iter()
            .filter(|candidate| active_candidate_rollout_complete(candidate))
            .cloned()
            .collect::<Vec<_>>();
        if complete_candidates.is_empty() {
            return None;
        }
        let mut clone = active.clone();
        clone.candidate_evaluations = complete_candidates;
        clone.row_ids = clone
            .candidate_evaluations
            .iter()
            .flat_map(|candidate| {
                candidate
                    .row_ids
                    .iter()
                    .map(|row_id| format!("{}:{row_id}", candidate.candidate_id))
            })
            .collect();
        clone.next_row_index = clone.row_ids.len();
        return Some(clone);
    }
    if active_rollout_evaluation_complete(active) {
        Some(active.clone())
    } else {
        None
    }
}

fn active_candidate_rollout_complete(candidate: &GepaActiveCandidateEvaluation) -> bool {
    !candidate.row_ids.is_empty()
        && active_candidate_completed_rows(candidate) >= candidate.row_ids.len()
}

fn active_candidate_completed_rows(candidate: &GepaActiveCandidateEvaluation) -> usize {
    completed_score_example_count(&candidate.scores).min(candidate.row_ids.len())
}

fn completed_score_example_count(scores: &[RolloutScore]) -> usize {
    scored_example_ids(scores).len()
}

fn scored_example_ids(scores: &[RolloutScore]) -> BTreeSet<&str> {
    scores
        .iter()
        .map(|score| score.example_id.as_str())
        .collect::<BTreeSet<_>>()
}

fn rollout_scores_contain_example(scores: &[RolloutScore], example_id: &str) -> bool {
    scores.iter().any(|score| score.example_id == example_id)
}

fn next_unscored_row_index(row_ids: &[String], scores: &[RolloutScore]) -> usize {
    let scored = scored_example_ids(scores);
    row_ids
        .iter()
        .position(|row_id| !scored.contains(row_id.as_str()))
        .unwrap_or(row_ids.len())
}

fn unscored_rollout_rows(rows: &[Value], scores: &[RolloutScore]) -> Result<Vec<Value>> {
    let scored = scored_example_ids(scores);
    rows.iter()
        .filter_map(|row| match row_example_id(row) {
            Ok(example_id) if scored.contains(example_id.as_str()) => None,
            Ok(_) => Some(Ok(row.clone())),
            Err(error) => Some(Err(error)),
        })
        .collect()
}

fn rollout_row_index_by_example_id(rows: &[Value], stage: &str, example_id: &str) -> Result<usize> {
    for (index, row) in rows.iter().enumerate() {
        if row_example_id(row)? == example_id {
            return Ok(index);
        }
    }
    Err(OptimizerError::Invariant(format!(
        "rollout outcome example_id {example_id} is not part of active {stage} rows"
    )))
}

fn speculative_partial_exceeds_threshold(
    context: &GepaRunContext,
    state: &GepaRunState,
    resources: &GepaStepResources,
    active: &GepaActiveEvaluation,
) -> Result<bool> {
    match active.stage.as_str() {
        "candidate_full_train" => {
            let current_pool_score = state
                .best_idx
                .and_then(|idx| state.candidates.get(idx))
                .and_then(|candidate| candidate.train_reward)
                .unwrap_or(f64::NEG_INFINITY);
            Ok(active_best_partial_reward(active) >= current_pool_score)
        }
        "candidate_minibatch" if active.is_group() => {
            for candidate in &active.candidate_evaluations {
                if candidate_partial_minibatch_exceeds_parent(context, state, resources, candidate)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        "candidate_minibatch" => {
            let candidate = GepaActiveCandidateEvaluation {
                candidate_id: active.candidate_id.clone().unwrap_or_default(),
                candidate_index: active.candidate_index.unwrap_or_default(),
                generation: active.generation,
                proposal_index: active.proposal_index,
                row_ids: active.row_ids.clone(),
                next_row_index: active.next_row_index,
                heldout_candidate_index: active.heldout_candidate_index,
                parent_id: active.parent_id.clone(),
                scores: active.scores.clone(),
                sensor_frames: active.sensor_frames.clone(),
                reward_sum: active.reward_sum,
                usage: active.usage.clone(),
                cost_usd: active.cost_usd,
                rollout_count: active.rollout_count,
                parent_minibatch_reward: active.parent_minibatch_reward,
                decision: active.decision.clone(),
            };
            candidate_partial_minibatch_exceeds_parent(context, state, resources, &candidate)
        }
        _ => Ok(false),
    }
}

fn active_best_partial_reward(active: &GepaActiveEvaluation) -> f64 {
    if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .filter(|candidate| candidate.rollout_count > 0)
            .map(GepaActiveCandidateEvaluation::average_reward)
            .reduce(f64::max)
            .unwrap_or(f64::NEG_INFINITY)
    } else if active.rollout_count > 0 {
        active.average_reward()
    } else {
        f64::NEG_INFINITY
    }
}

fn candidate_partial_minibatch_exceeds_parent(
    context: &GepaRunContext,
    state: &GepaRunState,
    resources: &GepaStepResources,
    active: &GepaActiveCandidateEvaluation,
) -> Result<bool> {
    if active.rollout_count == 0 {
        return Ok(false);
    }
    let candidate = state
        .candidates
        .get(active.candidate_index)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "candidate minibatch index {} is outside candidate registry",
                active.candidate_index
            ))
        })?;
    let parent_id = active
        .parent_id
        .clone()
        .or_else(|| candidate.parent_id.clone())
        .ok_or_else(|| {
            OptimizerError::Invariant(
                "candidate minibatch speculative threshold missing parent".to_string(),
            )
        })?;
    let parent = state
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_id == parent_id)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "candidate minibatch speculative threshold parent {parent_id} is missing"
            ))
        })?;
    let minibatch_rows = minibatch_rows(
        &resources.minibatch_rows,
        &context.config.gepa.batch_sampler,
        context.config.gepa.minibatch_size,
        active.generation,
        active.proposal_index,
        context.config.gepa.proposals_per_generation,
    );
    let Some(parent_reward) = parent_minibatch_reward_for_rows(
        parent,
        &minibatch_rows,
        &context.config.taskset.train_split,
    )?
    else {
        return Ok(false);
    };
    Ok(active.average_reward() > parent_reward)
}

fn async_partial_has_speculative_release(state: &GepaRunState, partial_id: &str) -> bool {
    state
        .cursor
        .pipeline_state
        .speculative_releases
        .iter()
        .any(|release| release.source_partial_id == partial_id)
        || state
            .cursor
            .pipeline_state
            .candidate_partials
            .get(partial_id)
            .and_then(|partial| {
                partial
                    .metadata
                    .get("speculative_release_id")
                    .and_then(Value::as_str)
            })
            .is_some()
}

fn merge_json_object(mut base: Value, patch: Value) -> Value {
    if !base.is_object() {
        base = json!({});
    }
    if let (Some(base), Some(patch)) = (base.as_object_mut(), patch.as_object()) {
        for (key, value) in patch {
            base.insert(key.clone(), value.clone());
        }
    }
    base
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

fn async_job_has_lane_lease(state: &GepaRunState, job_id: &str) -> bool {
    state
        .cursor
        .pipeline_state
        .lane_leases
        .values()
        .any(|lease| lease.job_id.as_deref() == Some(job_id))
}

fn ensure_adaptive_rollout_concurrency_state(
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinePlan,
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

fn adaptive_rollout_worker_limit(state: &GepaRunState, plan: &GepaAsyncPipelinePlan) -> usize {
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

fn adaptive_rollout_snapshot(state: &GepaRunState, plan: &GepaAsyncPipelinePlan) -> Value {
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

fn ensure_adaptive_stage_workers_state(state: &mut GepaRunState, plan: &GepaAsyncPipelinePlan) {
    let adaptive = &plan.adaptive_stage_workers;
    let stage_state = &mut state.cursor.pipeline_state.adaptive_stage_workers;
    if !adaptive.enabled {
        stage_state.propose_limit = plan.propose_workers.max(1);
        stage_state.rollout_limit = plan.rollout_workers.max(1);
        stage_state.evaluate_limit = plan.evaluate_workers.max(1);
        return;
    }
    if !stage_state.initialized {
        stage_state.initialized = true;
        stage_state.propose_limit = plan.propose_workers.clamp(adaptive.min, adaptive.max);
        stage_state.rollout_limit = plan.rollout_workers.clamp(adaptive.min, adaptive.max);
        stage_state.evaluate_limit = plan.evaluate_workers.clamp(adaptive.min, adaptive.max);
    }
}

fn adaptive_stage_worker_limit(
    state: &GepaRunState,
    plan: &GepaAsyncPipelinePlan,
    lane: &str,
) -> usize {
    if !plan.adaptive_stage_workers.enabled {
        return match lane {
            "propose" => plan.propose_workers.max(1),
            "evaluate" => plan.evaluate_workers.max(1),
            _ => plan.rollout_workers.max(1),
        };
    }
    let stage_state = &state.cursor.pipeline_state.adaptive_stage_workers;
    let raw = match lane {
        "propose" => stage_state.propose_limit,
        "evaluate" => stage_state.evaluate_limit,
        _ => stage_state.rollout_limit,
    };
    raw.clamp(
        plan.adaptive_stage_workers.min,
        plan.adaptive_stage_workers.max,
    )
    .max(1)
}

fn adaptive_rollout_lane_limit(state: &GepaRunState, plan: &GepaAsyncPipelinePlan) -> usize {
    adaptive_rollout_worker_limit(state, plan)
        .min(adaptive_stage_worker_limit(state, plan, "rollout"))
        .max(1)
}

fn adaptive_stage_workers_snapshot(state: &GepaRunState, plan: &GepaAsyncPipelinePlan) -> Value {
    let stage_state = &state.cursor.pipeline_state.adaptive_stage_workers;
    json!({
        "enabled": plan.adaptive_stage_workers.enabled,
        "propose_limit": adaptive_stage_worker_limit(state, plan, "propose"),
        "rollout_limit": adaptive_stage_worker_limit(state, plan, "rollout"),
        "evaluate_limit": adaptive_stage_worker_limit(state, plan, "evaluate"),
        "min": plan.adaptive_stage_workers.min,
        "max": plan.adaptive_stage_workers.max,
        "backlog_threshold": plan.adaptive_stage_workers.backlog_threshold,
        "stale_gap_threshold": plan.adaptive_stage_workers.stale_gap_threshold,
        "last_adjustment": stage_state.last_adjustment,
    })
}

fn adjust_adaptive_stage_workers(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinePlan,
) -> Result<()> {
    if !plan.adaptive_stage_workers.enabled {
        return Ok(());
    }
    ensure_adaptive_stage_workers_state(state, plan);
    let backlog_threshold = plan.adaptive_stage_workers.backlog_threshold;
    let max_stale_gap = state
        .cursor
        .pipeline_state
        .evaluate_queue
        .iter()
        .map(|item| {
            state
                .cursor
                .pipeline_state
                .pool_version
                .saturating_sub(item.parent_pool_version)
        })
        .max()
        .unwrap_or(0);
    if state.cursor.pipeline_state.rollout_queue.len() >= backlog_threshold {
        return adjust_adaptive_stage_worker_limit(
            context,
            state,
            plan,
            "rollout",
            "up",
            "rollout_queue_backlog",
        );
    }
    if state.cursor.pipeline_state.evaluate_queue.len() >= backlog_threshold {
        return adjust_adaptive_stage_worker_limit(
            context,
            state,
            plan,
            "evaluate",
            "up",
            "evaluate_queue_backlog",
        );
    }
    if max_stale_gap > plan.adaptive_stage_workers.stale_gap_threshold {
        return adjust_adaptive_stage_worker_limit(
            context,
            state,
            plan,
            "propose",
            "down",
            "stale_gap_pressure",
        );
    }
    if state.cursor.pipeline_state.rollout_queue.is_empty()
        && async_lane_lease_count(state, "rollout") == 0
    {
        return adjust_adaptive_stage_worker_limit(
            context,
            state,
            plan,
            "rollout",
            "down",
            "rollout_lane_idle",
        );
    }
    Ok(())
}

fn adjust_adaptive_stage_worker_limit(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinePlan,
    lane: &str,
    direction: &str,
    reason: &str,
) -> Result<()> {
    let adaptive = &plan.adaptive_stage_workers;
    let stage_state = &mut state.cursor.pipeline_state.adaptive_stage_workers;
    let limit = match lane {
        "propose" => &mut stage_state.propose_limit,
        "evaluate" => &mut stage_state.evaluate_limit,
        _ => &mut stage_state.rollout_limit,
    };
    let old_limit = (*limit).clamp(adaptive.min, adaptive.max).max(1);
    let new_limit = match direction {
        "up" => old_limit
            .saturating_add(1)
            .clamp(adaptive.min, adaptive.max),
        "down" => old_limit
            .saturating_sub(1)
            .clamp(adaptive.min, adaptive.max),
        _ => old_limit,
    };
    if new_limit == old_limit {
        return Ok(());
    }
    *limit = new_limit;
    let adjustment = GepaAdaptiveStageWorkersAdjustment {
        lane: lane.to_string(),
        direction: direction.to_string(),
        old_limit,
        new_limit,
        reason: reason.to_string(),
    };
    stage_state.last_adjustment = Some(adjustment.clone());
    stage_state.adjustments.push(adjustment.clone());
    if stage_state.adjustments.len() > 32 {
        stage_state.adjustments.remove(0);
    }
    context.events.emit(
        "pipeline.stage_workers.adjusted",
        "Adaptive stage worker limit adjusted",
        json!({
            "lane": adjustment.lane,
            "direction": adjustment.direction,
            "old_limit": adjustment.old_limit,
            "new_limit": adjustment.new_limit,
            "reason": adjustment.reason,
        }),
    )?;
    Ok(())
}

fn runtime_rollout_success_count(outcome: &runtime::RuntimeEffectOutcome) -> usize {
    match outcome {
        runtime::RuntimeEffectOutcome::Rollout(_) => 1,
        runtime::RuntimeEffectOutcome::RolloutBatch(outcomes) => outcomes.len(),
        runtime::RuntimeEffectOutcome::Proposer(_) => 0,
    }
}

fn provider_signal_from_error(
    config: &SynthOptimizerConfig,
    error: &OptimizerError,
) -> ProviderSignal {
    let status_code = match error {
        OptimizerError::ContainerHttpStatus { status_code, .. } => Some(*status_code),
        OptimizerError::Http(error) => error.status().map(|status| status.as_u16()),
        _ => http_status_from_error_text(&error.to_string()),
    };
    let overload = status_code.is_some_and(|status| {
        config
            .gepa
            .pipeline
            .adaptive_rollout_concurrency
            .overload_status_codes
            .contains(&status)
    }) || fallback_error_text_is_overload(error);
    ProviderSignal {
        status_code,
        provider_error_code: None,
        overload,
        retryable: rollout_error_is_retryable(error),
    }
}

fn provider_signal_from_failure(
    config: &SynthOptimizerConfig,
    failure: Option<&FailurePayload>,
) -> ProviderSignal {
    let status_code = failure
        .and_then(|failure| failure.details.get("status"))
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .or_else(|| failure.and_then(|failure| http_status_from_error_text(&failure.message)));
    let overload = status_code.is_some_and(|status| {
        config
            .gepa
            .pipeline
            .adaptive_rollout_concurrency
            .overload_status_codes
            .contains(&status)
    });
    ProviderSignal {
        status_code,
        provider_error_code: failure.map(|failure| failure.reason_code.clone()),
        overload,
        retryable: failure.is_some_and(|failure| failure.retryable),
    }
}

fn fallback_error_text_is_overload(error: &OptimizerError) -> bool {
    let lowered = error.to_string().to_ascii_lowercase();
    [
        "429",
        "rate limit",
        "rate_limit",
        "overload",
        "overloaded",
        "too many requests",
        "temporarily unavailable",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn http_status_from_error_text(text: &str) -> Option<u16> {
    let lowered = text.to_ascii_lowercase();
    for marker in ["httpexception", "http status", "status_code", "status"] {
        let Some(index) = lowered.find(marker) else {
            continue;
        };
        let tail = &lowered[index + marker.len()..];
        for token in tail.split(|ch: char| !ch.is_ascii_digit()) {
            if token.len() != 3 {
                continue;
            }
            let Ok(status) = token.parse::<u16>() else {
                continue;
            };
            if (400..=599).contains(&status) {
                return Some(status);
            }
        }
    }
    None
}

fn rollout_error_is_retryable(error: &OptimizerError) -> bool {
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

fn rollout_error_is_degradable(error: &OptimizerError) -> bool {
    rollout_error_is_retryable(error)
}

fn emit_rollout_failure_rate(
    config: &SynthOptimizerConfig,
    events: &mut Option<&mut EventWriter>,
    sample: &GepaRolloutFailureSample,
    resilience: &GepaRolloutResilienceState,
) -> Result<()> {
    if let Some(events) = events.as_deref_mut() {
        events.emit(
            "rollout.failure_rate.updated",
            "Rollout failure rate updated",
            json!({
                "stage": sample.stage,
                "example_id": sample.example_id,
                "infra_failed": sample.infra_failed,
                "failure_class": sample.failure_class,
                "provider_status_code": sample.provider_status_code,
                "rolling_failure_rate": resilience.last_failure_rate,
                "sample_count": resilience.rolling_samples.len(),
                "tolerance": config.gepa.rollout_failure_rate_tolerance,
                "degraded_rollouts": resilience.degraded_rollouts,
                "scored_rollouts": resilience.scored_rollouts,
            }),
        )?;
    }
    Ok(())
}

struct RolloutResilienceObservation<'a> {
    stage: &'a str,
    example_id: &'a str,
    degraded: bool,
    failure: Option<&'a FailurePayload>,
    provider_signal: &'a ProviderSignal,
}

fn record_rollout_resilience_sample(
    config: &SynthOptimizerConfig,
    mut events: Option<&mut EventWriter>,
    resilience: &mut GepaRolloutResilienceState,
    observation: RolloutResilienceObservation<'_>,
) -> Result<()> {
    let sample = GepaRolloutFailureSample {
        infra_failed: observation.degraded,
        stage: observation.stage.to_string(),
        example_id: observation.example_id.to_string(),
        failure_class: observation
            .failure
            .map(|failure| failure.failure_class().to_string()),
        provider_status_code: observation.provider_signal.status_code,
    };
    resilience.rolling_samples.push(sample.clone());
    if resilience.rolling_samples.len() > 32 {
        resilience.rolling_samples.remove(0);
    }
    if observation.degraded {
        resilience.degraded_rollouts = resilience.degraded_rollouts.saturating_add(1);
    } else {
        resilience.scored_rollouts = resilience.scored_rollouts.saturating_add(1);
    }
    let failed = resilience
        .rolling_samples
        .iter()
        .filter(|sample| sample.infra_failed)
        .count();
    resilience.last_failure_rate = if resilience.rolling_samples.is_empty() {
        0.0
    } else {
        failed as f64 / resilience.rolling_samples.len() as f64
    };
    emit_rollout_failure_rate(config, &mut events, &sample, resilience)?;
    if observation.degraded {
        if let Some(events) = &mut events {
            events.emit(
                "rollout.degraded",
                "Rollout degraded",
                json!({
                    "stage": observation.stage,
                    "example_id": observation.example_id,
                    "reward": 0.0,
                    "failure_class": observation.failure.map(FailurePayload::failure_class),
                    "provider_status_code": observation.provider_signal.status_code,
                    "rolling_failure_rate": resilience.last_failure_rate,
                    "tolerance": config.gepa.rollout_failure_rate_tolerance,
                }),
            )?;
        }
    }
    let tolerance = config.gepa.rollout_failure_rate_tolerance;
    if resilience.rolling_samples.len() >= 8 && resilience.last_failure_rate > tolerance {
        let breaker = GepaRolloutCircuitBreaker {
            rolling_rate: resilience.last_failure_rate,
            tolerance,
            sample_count: resilience.rolling_samples.len(),
            reason: "rolling_failure_rate_exceeded".to_string(),
        };
        resilience.last_circuit_breaker = Some(breaker.clone());
        if let Some(events) = &mut events {
            events.emit(
                "rollout.circuit_breaker.tripped",
                "Rollout circuit breaker tripped",
                json!({
                    "rolling_failure_rate": breaker.rolling_rate,
                    "tolerance": breaker.tolerance,
                    "sample_count": breaker.sample_count,
                    "reason": breaker.reason,
                }),
            )?;
        }
        return Err(OptimizerError::Failed(format!(
            "rollout infra failure rate {:.2} exceeds tolerance {:.2} (window={} rollouts)",
            breaker.rolling_rate, breaker.tolerance, breaker.sample_count
        )));
    }
    Ok(())
}

fn check_rollout_section_breaker(
    config: &SynthOptimizerConfig,
    events: Option<&mut EventWriter>,
    resilience: &mut GepaRolloutResilienceState,
    stage: &str,
    scored_count: usize,
    degraded_count: usize,
) -> Result<()> {
    if scored_count == 0 && degraded_count >= 2 {
        let breaker = GepaRolloutCircuitBreaker {
            rolling_rate: resilience.last_failure_rate,
            tolerance: config.gepa.rollout_failure_rate_tolerance,
            sample_count: resilience.rolling_samples.len(),
            reason: "section_zero_scored_rollouts".to_string(),
        };
        resilience.last_circuit_breaker = Some(breaker.clone());
        if let Some(events) = events {
            events.emit(
                "rollout.circuit_breaker.tripped",
                "Rollout circuit breaker tripped",
                json!({
                    "stage": stage,
                    "rolling_failure_rate": breaker.rolling_rate,
                    "tolerance": breaker.tolerance,
                    "sample_count": breaker.sample_count,
                    "reason": breaker.reason,
                    "degraded_count": degraded_count,
                }),
            )?;
        }
        return Err(OptimizerError::Failed(format!(
            "rollout section {stage} produced zero scored rollouts ({degraded_count} degraded)"
        )));
    }
    Ok(())
}

fn record_adaptive_rollout_success(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinePlan,
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
    let new_limit = ((old_limit as f64) * 1.10).ceil() as usize;
    let new_limit = new_limit
        .max(old_limit.saturating_add(1))
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
            "multiplicative_healthy_after_{}_rollout_successes",
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
            "adjustment_mode": "multiplicative",
            "completed_rollouts": adjustment.completed_rollouts,
        }),
    )?;
    Ok(())
}

fn record_adaptive_rollout_overload(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    plan: &GepaAsyncPipelinePlan,
    provider_signal: &ProviderSignal,
) -> Result<()> {
    if !plan.adaptive_rollout_concurrency.enabled || !provider_signal.overload {
        return Ok(());
    }
    ensure_adaptive_rollout_concurrency_state(state, plan);
    let adaptive = &plan.adaptive_rollout_concurrency;
    let adaptive_state = &mut state.cursor.pipeline_state.adaptive_rollout_concurrency;
    adaptive_state.overload_count = adaptive_state.overload_count.saturating_add(1);
    adaptive_state.successes_since_adjustment = 0;
    let old_limit = adaptive_state.current_limit;
    let new_limit = ((old_limit as f64) * 0.50).floor() as usize;
    let new_limit = new_limit.clamp(adaptive.min, adaptive.max);
    if new_limit == old_limit {
        return Ok(());
    }
    adaptive_state.current_limit = new_limit;
    let adjustment = GepaAdaptiveRolloutConcurrencyAdjustment {
        direction: "down".to_string(),
        old_limit,
        new_limit,
        reason: "provider_overload".to_string(),
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
            "adjustment_mode": "multiplicative",
            "provider_status_code": provider_signal.status_code,
            "provider_error_code": provider_signal.provider_error_code,
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

fn candidate_train_selectable(candidate: &CandidateRecord) -> bool {
    candidate.train_reward.is_some()
        && !candidate.train_scores.is_empty()
        && (candidate.source == "seed"
            || matches!(
                candidate.status.as_str(),
                "accepted" | "full_train_evaluated"
            ))
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

fn gepa_pipeline_uses_async_lanes(config: &SynthOptimizerConfig) -> bool {
    matches!(
        config.gepa.pipeline.mode,
        GepaPipelineMode::AsyncPipelined | GepaPipelineMode::FlashEvolve
    )
}

fn rollout_rows_for_chunk<'a>(config: &SynthOptimizerConfig, rows: &'a [Value]) -> &'a [Value] {
    if gepa_pipeline_uses_async_lanes(config) {
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
    if !gepa_pipeline_uses_async_lanes(config) {
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
            let remaining = unscored_rollout_rows(&stage_rows, &candidate_eval.scores)?;
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
    let remaining = unscored_rollout_rows(&rows, &active.scores)?;
    Ok(rollout_rows_for_chunk(config, &remaining).to_vec())
}

fn row_ids_for_chunk(rows: &[Value]) -> Result<Vec<String>> {
    rows.iter().map(row_example_id).collect()
}

fn first_unscored_row_index_for_active(active: &GepaActiveEvaluation) -> usize {
    if active.is_group() {
        return active
            .candidate_evaluations
            .iter()
            .map(|candidate| next_unscored_row_index(&candidate.row_ids, &candidate.scores))
            .min()
            .unwrap_or(active.next_row_index);
    }
    next_unscored_row_index(&active.row_ids, &active.scores)
}

fn async_rollout_chunk_id(active: &GepaActiveEvaluation, rows: &[Value]) -> Result<String> {
    let first_row_index = first_unscored_row_index_for_active(active);
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
    let first_row_index = first_unscored_row_index_for_active(active);
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
    async_plan: Option<&GepaAsyncPipelinePlan>,
) -> Result<GepaAdvanceOutcome> {
    let job = context
        .workspace
        .optimizer_job(&context.config.run.run_id, job_id)?;
    match job.status {
        OptimizerJobStatus::Pending | OptimizerJobStatus::RetryScheduled => {
            if matches!(job.status, OptimizerJobStatus::RetryScheduled)
                && !context
                    .workspace
                    .optimizer_job_claimable(&context.config.run.run_id, job_id)?
            {
                return Ok(GepaAdvanceOutcome {
                    action: planner::GepaTickAction::Noop,
                    terminal: false,
                    result: None,
                    message: format!("GEPA runtime job retry not ready: {job_id}"),
                });
            }
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
                            let provider_signal =
                                provider_signal_from_error(&context.config, &error);
                            record_adaptive_rollout_overload(
                                context,
                                state,
                                plan,
                                &provider_signal,
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
                            if let Some(outcome) = schedule_failed_rollout_retry_if_allowed(
                                context,
                                state,
                                resources,
                                &updated_job,
                            )? {
                                return Ok(outcome);
                            }
                            if async_plan.is_some()
                                && async_job_has_lane_lease(state, &updated_job.job_id)
                                && matches!(updated_job.kind, OptimizerJobKind::Rollout)
                            {
                                persist_gepa_run_state(
                                    context,
                                    state,
                                    resources,
                                    state.cursor.phase.clone(),
                                    "runtime_job_failed_pending_degrade",
                                    "rollout runtime job failed and is pending degradation",
                                    Map::new(),
                                )?;
                                return Ok(GepaAdvanceOutcome {
                                    action: planner::GepaTickAction::ExecuteRuntimeJob {
                                        run_id: context.config.run.run_id.clone(),
                                        job_id: updated_job.job_id,
                                    },
                                    terminal: false,
                                    result: None,
                                    message:
                                        "rollout runtime job failed and is pending degradation"
                                            .to_string(),
                                });
                            }
                            if matches!(updated_job.kind, OptimizerJobKind::Rollout)
                                && consume_failed_rollout_job_as_degraded(
                                    context,
                                    state,
                                    resources,
                                    &updated_job,
                                )?
                            {
                                state.cursor.pending_job_id = None;
                                state.cursor.pending_effect_id = None;
                                state.cursor.pending_reservation_ids.clear();
                                persist_gepa_run_state(
                                    context,
                                    state,
                                    resources,
                                    state.cursor.phase.clone(),
                                    "completed",
                                    "degraded failed rollout runtime job",
                                    Map::new(),
                                )?;
                                return Ok(GepaAdvanceOutcome {
                                    action: planner::GepaTickAction::ConsumeRuntimeOutcome {
                                        run_id: context.config.run.run_id.clone(),
                                        job_id: updated_job.job_id,
                                    },
                                    terminal: false,
                                    result: None,
                                    message: "degraded failed rollout runtime job".to_string(),
                                });
                            }
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
        | OptimizerJobStatus::Expired => {
            if let Some(outcome) =
                schedule_failed_rollout_retry_if_allowed(context, state, resources, &job)?
            {
                return Ok(outcome);
            }
            if matches!(job.kind, OptimizerJobKind::Rollout)
                && consume_failed_rollout_job_as_degraded(context, state, resources, &job)?
            {
                state.cursor.pending_job_id = None;
                state.cursor.pending_effect_id = None;
                state.cursor.pending_reservation_ids.clear();
                persist_gepa_run_state(
                    context,
                    state,
                    resources,
                    state.cursor.phase.clone(),
                    "completed",
                    "degraded failed rollout runtime job",
                    Map::new(),
                )?;
                Ok(GepaAdvanceOutcome {
                    action: planner::GepaTickAction::ConsumeRuntimeOutcome {
                        run_id: context.config.run.run_id.clone(),
                        job_id: job.job_id,
                    },
                    terminal: false,
                    result: None,
                    message: "degraded failed rollout runtime job".to_string(),
                })
            } else {
                consume_failed_runtime_job(context, state, resources, job)
            }
        }
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
    record_runtime_job_transitions(context, state, job_id, job, outcome, wall_seconds)?;

    match outcome {
        runtime::RuntimeEffectOutcome::Proposer(outcome) => {
            fields.insert("runtime_kind".to_string(), json!("proposer"));
            fields.insert("model".to_string(), json!(context.config.proposer.model));
            fields.insert(
                "provider".to_string(),
                json!(&context.config.proposer.provider),
            );
            fields.insert("proposal_count".to_string(), json!(outcome.proposals.len()));
            fields.insert("backend".to_string(), json!(&outcome.backend));
            fields.insert("cache_hit".to_string(), json!(outcome.cache_hit));
            fields.insert("cost_usd".to_string(), json!(outcome.cost_usd));
            fields.insert("usage".to_string(), serde_json::to_value(&outcome.usage)?);
            if let Some(cost_source) = outcome
                .response
                .pointer("/usage/cost_source")
                .and_then(Value::as_str)
            {
                fields.insert("cost_source".to_string(), json!(cost_source));
            }
            fields.insert(
                "total_tokens".to_string(),
                json!(outcome.usage.total_tokens),
            );
            insert_token_throughput_fields(&mut fields, outcome.usage.total_tokens, wall_seconds);
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
            insert_token_throughput_fields(&mut fields, outcome.usage.total_tokens, wall_seconds);
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
            insert_token_throughput_fields(&mut fields, usage.total_tokens, wall_seconds);
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

fn insert_token_throughput_fields(
    fields: &mut Map<String, Value>,
    total_tokens: u64,
    wall_seconds: f64,
) {
    if total_tokens == 0 || !wall_seconds.is_finite() || wall_seconds <= 0.0 {
        return;
    }
    let tokens_per_second = total_tokens as f64 / wall_seconds;
    fields.insert("tokens_per_second".to_string(), json!(tokens_per_second));
    fields.insert(
        "tokens_per_minute".to_string(),
        json!(tokens_per_second * 60.0),
    );
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
        record_candidate_registered(
            context,
            &state.candidates[0],
            None,
            json!({"source": "seed", "stage": "run_start"}),
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
            .all(active_candidate_rollout_complete)
    } else {
        !active.row_ids.is_empty()
            && completed_score_example_count(&active.scores) >= active.row_ids.len()
    }
}

fn active_rollout_completed_rows(active: &GepaActiveEvaluation) -> usize {
    if active.is_group() {
        active
            .candidate_evaluations
            .iter()
            .map(active_candidate_completed_rows)
            .sum()
    } else {
        completed_score_example_count(&active.scores).min(active.row_ids.len())
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

fn record_candidate_transition(
    context: &GepaRunContext,
    candidate_id: &str,
    parent_id: Option<&str>,
    generation: usize,
    to: CandidateState,
    trigger: CandidateTrigger,
    metadata: Value,
) -> Result<()> {
    context.transitions.transition_entity::<CandidateEntity>(
        candidate_id,
        to,
        trigger,
        Some(usize_to_i64(generation)),
        parent_id,
        metadata,
    )?;
    Ok(())
}

fn record_candidate_registered(
    context: &GepaRunContext,
    candidate: &CandidateRecord,
    generation: Option<usize>,
    metadata: Value,
) -> Result<()> {
    context.transitions.transition_entity::<CandidateEntity>(
        &candidate.candidate_id,
        CandidateState::Registered,
        CandidateTrigger::Registered,
        generation.map(usize_to_i64),
        candidate.parent_id.as_deref(),
        metadata,
    )?;
    Ok(())
}

fn record_candidate_evaluation_started_from_details(
    context: &GepaRunContext,
    details: &Value,
) -> Result<()> {
    let Some(stage) = details.get("stage").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(candidate_id) = details.get("candidate_id").and_then(Value::as_str) else {
        return Ok(());
    };
    let generation = details
        .get("generation")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    let transition = match stage {
        "seed_full_train" => Some((
            CandidateState::FullTrainEvaluating,
            CandidateTrigger::EvaluationStarted,
        )),
        "candidate_minibatch" => Some((
            CandidateState::MinibatchEvaluating,
            CandidateTrigger::EvaluationStarted,
        )),
        "candidate_full_train" => Some((
            CandidateState::FullTrainEvaluating,
            CandidateTrigger::EvaluationStarted,
        )),
        "heldout" => Some((
            CandidateState::HeldoutEvaluating,
            CandidateTrigger::HeldoutStarted,
        )),
        _ => None,
    };
    let Some((to, trigger)) = transition else {
        return Ok(());
    };
    if context
        .transitions
        .latest_state(CandidateEntity::ENTITY_TYPE, candidate_id)?
        .as_deref()
        == Some(CandidateEntity::state_name(to))
    {
        return Ok(());
    }
    record_candidate_transition(
        context,
        candidate_id,
        None,
        generation,
        to,
        trigger,
        details.clone(),
    )
}

fn record_proposer_round_started(
    context: &GepaRunContext,
    job_id: &str,
    generation: usize,
    parent_candidate_id: &str,
    metadata: Value,
) -> Result<()> {
    context
        .transitions
        .transition_entity::<ProposerRoundEntity>(
            job_id,
            ProposerRoundState::Requested,
            ProposerRoundTrigger::Requested,
            Some(usize_to_i64(generation)),
            Some(parent_candidate_id),
            metadata.clone(),
        )?;
    context
        .transitions
        .transition_entity::<ProposerRoundEntity>(
            job_id,
            ProposerRoundState::Dispatched,
            ProposerRoundTrigger::Dispatched,
            Some(usize_to_i64(generation)),
            Some(parent_candidate_id),
            metadata.clone(),
        )?;
    context
        .transitions
        .transition_entity::<ProposerRoundEntity>(
            job_id,
            ProposerRoundState::Generating,
            ProposerRoundTrigger::GenerationStarted,
            Some(usize_to_i64(generation)),
            Some(parent_candidate_id),
            metadata,
        )?;
    Ok(())
}

fn record_proposer_round_completed(
    context: &GepaRunContext,
    job_id: &str,
    generation: Option<usize>,
    parent_candidate_id: Option<&str>,
    proposal_count: usize,
    metadata: Value,
) -> Result<()> {
    let generation = generation.map(usize_to_i64);
    if context
        .transitions
        .latest_state(ProposerRoundEntity::ENTITY_TYPE, job_id)?
        .is_none()
    {
        context
            .transitions
            .transition_entity::<ProposerRoundEntity>(
                job_id,
                ProposerRoundState::Requested,
                ProposerRoundTrigger::Requested,
                generation,
                parent_candidate_id,
                metadata.clone(),
            )?;
        context
            .transitions
            .transition_entity::<ProposerRoundEntity>(
                job_id,
                ProposerRoundState::Dispatched,
                ProposerRoundTrigger::Dispatched,
                generation,
                parent_candidate_id,
                metadata.clone(),
            )?;
        context
            .transitions
            .transition_entity::<ProposerRoundEntity>(
                job_id,
                ProposerRoundState::Generating,
                ProposerRoundTrigger::GenerationStarted,
                generation,
                parent_candidate_id,
                metadata.clone(),
            )?;
    }
    context
        .transitions
        .transition_entity::<ProposerRoundEntity>(
            job_id,
            ProposerRoundState::Returned,
            ProposerRoundTrigger::GenerationReturned,
            generation,
            parent_candidate_id,
            metadata.clone(),
        )?;
    let parse_state = if proposal_count > 0 {
        ProposerRoundState::ParsedOk
    } else {
        ProposerRoundState::ParseFailed
    };
    let parse_trigger = if proposal_count > 0 {
        ProposerRoundTrigger::Parsed
    } else {
        ProposerRoundTrigger::ParseFailed
    };
    context
        .transitions
        .transition_entity::<ProposerRoundEntity>(
            job_id,
            parse_state,
            parse_trigger,
            generation,
            parent_candidate_id,
            metadata.clone(),
        )?;
    context
        .transitions
        .transition_entity::<ProposerRoundEntity>(
            job_id,
            ProposerRoundState::Closed,
            ProposerRoundTrigger::Closed,
            generation,
            parent_candidate_id,
            metadata,
        )?;
    Ok(())
}

fn record_runtime_job_transitions(
    context: &GepaRunContext,
    state: &GepaRunState,
    job_id: &str,
    job: &OptimizerJob,
    outcome: &runtime::RuntimeEffectOutcome,
    wall_seconds: f64,
) -> Result<()> {
    let end_ms = current_unix_ms();
    match outcome {
        runtime::RuntimeEffectOutcome::Proposer(outcome) => {
            let generation = state
                .active_evaluation
                .as_ref()
                .map(|active| active.generation);
            let parent_candidate_id = state
                .active_evaluation
                .as_ref()
                .and_then(|active| active.candidate_id.as_deref());
            record_proposer_round_completed(
                context,
                job_id,
                generation,
                parent_candidate_id,
                outcome.proposals.len(),
                json!({
                    "job_id": job_id,
                    "runtime_effect_id": job.payload.get("runtime_effect_id").and_then(Value::as_str),
                    "runtime_kind": "proposer",
                    "model": &context.config.proposer.model,
                    "provider": &context.config.proposer.provider,
                    "backend": &outcome.backend,
                    "proposal_count": outcome.proposals.len(),
                    "wall_seconds": wall_seconds,
                    "cost_usd": outcome.cost_usd,
                    "usage": &outcome.usage,
                }),
            )?;
        }
        runtime::RuntimeEffectOutcome::Rollout(outcome) => {
            record_rollout_transition_span(
                context,
                state,
                job_id,
                job,
                0,
                &outcome.candidate_id,
                &outcome.stage,
                &outcome.example_id,
                outcome.cache_hit,
                outcome.dispatch_wall_seconds.unwrap_or(wall_seconds),
                wall_seconds,
                outcome.cost_usd,
                serde_json::to_value(&outcome.usage)?,
                end_ms,
            )?;
        }
        runtime::RuntimeEffectOutcome::RolloutBatch(outcomes) => {
            for (index, outcome) in outcomes.iter().enumerate() {
                record_rollout_transition_span(
                    context,
                    state,
                    job_id,
                    job,
                    index,
                    &outcome.candidate_id,
                    &outcome.stage,
                    &outcome.example_id,
                    outcome.cache_hit,
                    outcome.dispatch_wall_seconds.unwrap_or(wall_seconds),
                    wall_seconds,
                    outcome.cost_usd,
                    serde_json::to_value(&outcome.usage)?,
                    end_ms,
                )?;
            }
        }
    }
    Ok(())
}

fn record_rollout_transition_span(
    context: &GepaRunContext,
    state: &GepaRunState,
    job_id: &str,
    job: &OptimizerJob,
    index: usize,
    candidate_id: &str,
    stage: &str,
    example_id: &str,
    cache_hit: bool,
    dispatch_wall_seconds: f64,
    job_wall_seconds: f64,
    cost_usd: f64,
    usage: Value,
    end_ms: i64,
) -> Result<()> {
    let duration_ms = seconds_to_millis(dispatch_wall_seconds.max(0.0));
    let start_ms = end_ms.saturating_sub(duration_ms.max(1));
    let generation = state
        .active_evaluation
        .as_ref()
        .map(|active| usize_to_i64(active.generation));
    let runtime_effect_id = job.payload.get("runtime_effect_id").and_then(Value::as_str);
    let rollout_id = format!("{job_id}:rollout:{index:04}");
    let metadata = json!({
        "job_id": job_id,
        "runtime_effect_id": runtime_effect_id,
        "candidate_id": candidate_id,
        "stage": stage,
        "example_id": example_id,
        "cache_hit": cache_hit,
        "dispatch_wall_seconds": dispatch_wall_seconds,
        "job_wall_seconds": job_wall_seconds,
        "cost_usd": cost_usd,
        "usage": usage,
        "model": &context.config.policy.model,
    });
    context.transitions.transition_entity_at::<RolloutEntity>(
        Some(start_ms),
        &rollout_id,
        RolloutState::Queued,
        RolloutTrigger::Scheduled,
        generation,
        Some(candidate_id),
        metadata.clone(),
    )?;
    if cache_hit {
        context.transitions.transition_entity_at::<RolloutEntity>(
            Some(end_ms),
            &rollout_id,
            RolloutState::Cached,
            RolloutTrigger::CacheHit,
            generation,
            Some(candidate_id),
            metadata,
        )?;
        return Ok(());
    }
    context.transitions.transition_entity_at::<RolloutEntity>(
        Some(start_ms),
        &rollout_id,
        RolloutState::Running,
        RolloutTrigger::Started,
        generation,
        Some(candidate_id),
        metadata.clone(),
    )?;
    context.transitions.transition_entity_at::<RolloutEntity>(
        Some(end_ms),
        &rollout_id,
        RolloutState::Completed,
        RolloutTrigger::Succeeded,
        generation,
        Some(candidate_id),
        metadata,
    )?;
    Ok(())
}

fn current_unix_ms() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

fn seconds_to_millis(seconds: f64) -> i64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    i64::try_from((seconds * 1000.0).round() as i128).unwrap_or(i64::MAX)
}

fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn transition_to_rollout_running(
    context: &mut GepaRunContext,
    message: &str,
    details: Value,
) -> Result<()> {
    record_candidate_evaluation_started_from_details(context, &details)?;
    if matches!(
        context.state_machine.state(),
        OptimizerRunState::Ready | OptimizerRunState::Evaluating
    ) {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            Some(&context.transitions),
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
            Some(&context.transitions),
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
    if active_rollout_completed_rows(active) == 0 {
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
    let remaining_rows = unscored_rollout_rows(&rows, &active.scores)?;
    if remaining_rows.is_empty() {
        return finalize_active_rollout_evaluation(context, state, resources);
    }
    let chunk_rows = rollout_rows_for_chunk(&context.config, &remaining_rows);
    let rollout_capacity = remaining_rollout_capacity_for_stage(
        &context.workspace,
        &context.config,
        &state.candidates,
        state.rollout_count,
        &active.stage,
    )?;
    if rollout_capacity < chunk_rows.len() {
        return defer_active_rollout_evaluation_for_budget(
            context,
            state,
            resources,
            chunk_rows.len(),
            rollout_capacity,
        );
    }
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
        let remaining_rows = unscored_rollout_rows(&rows, &candidate_eval.scores)?;
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
        if active_rollout_evaluation_complete(active) {
            return finalize_active_rollout_evaluation(context, state, resources);
        }
        return defer_active_rollout_evaluation_for_budget(context, state, resources, 1, 0);
    }
    let requested_rollouts = groups.iter().map(|group| group.rows.len()).sum::<usize>();
    let rollout_capacity = remaining_rollout_capacity_for_stage(
        &context.workspace,
        &context.config,
        &state.candidates,
        state.rollout_count,
        &active.stage,
    )?;
    if rollout_capacity < requested_rollouts {
        return defer_active_rollout_evaluation_for_budget(
            context,
            state,
            resources,
            requested_rollouts,
            rollout_capacity,
        );
    }
    if active
        .candidate_evaluations
        .iter()
        .all(|candidate| active_candidate_completed_rows(candidate) == 0)
    {
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
            match active.stage.as_str() {
                "candidate_minibatch" => record_candidate_transition(
                    context,
                    &candidate.candidate_id,
                    candidate.parent_id.as_deref(),
                    candidate_eval.generation,
                    CandidateState::MinibatchEvaluating,
                    CandidateTrigger::EvaluationStarted,
                    json!({
                        "stage": "candidate_minibatch",
                        "candidate_id": &candidate.candidate_id,
                        "row_count": candidate_eval.row_ids.len(),
                    }),
                )?,
                "candidate_full_train" => record_candidate_transition(
                    context,
                    &candidate.candidate_id,
                    candidate.parent_id.as_deref(),
                    candidate_eval.generation,
                    CandidateState::FullTrainEvaluating,
                    CandidateTrigger::EvaluationStarted,
                    json!({
                        "stage": "candidate_full_train",
                        "candidate_id": &candidate.candidate_id,
                        "row_count": candidate_eval.row_ids.len(),
                    }),
                )?,
                "heldout" => record_candidate_transition(
                    context,
                    &candidate.candidate_id,
                    candidate.parent_id.as_deref(),
                    candidate_eval.generation,
                    CandidateState::HeldoutEvaluating,
                    CandidateTrigger::HeldoutStarted,
                    json!({
                        "stage": "heldout",
                        "candidate_id": &candidate.candidate_id,
                        "row_count": candidate_eval.row_ids.len(),
                    }),
                )?,
                _ => {}
            }
        }
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
            let prompt_assertions =
                prompt_assertions_for_candidate(&overlay.candidate, &context.config);
            let request = json!({
                "submission_mode": rollout_submission_mode_for_request(&context.config),
                "task_id": resources.rollout_task_id,
                "candidate": overlay.candidate.to_value(),
                "candidate_overlay": overlay,
                "prompt_assertions": prompt_assertions,
                "policy": rollout_policy_for_request(&context.config),
                "task": row,
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

fn rollout_policy_for_request(config: &SynthOptimizerConfig) -> Value {
    if config.policy.enabled {
        json!(&config.policy)
    } else {
        Value::Null
    }
}

fn prompt_assertions_for_candidate(
    candidate: &PromptCandidatePayload,
    config: &SynthOptimizerConfig,
) -> Value {
    let mut expected_candidate_prompts = Map::new();
    for field in &config.candidate.target_modules {
        let Some(prompt) = candidate.fields.get(field) else {
            continue;
        };
        expected_candidate_prompts.insert(
            field.clone(),
            json!({
                "sha256": sha256_text(prompt),
                "bytes": prompt.as_bytes().len(),
                "source": format!("candidate.{field}"),
                "must_reach": "policy_llm_system_message",
            }),
        );
    }
    json!({
        "schema_version": "prompt_assertions.v1",
        "required": !expected_candidate_prompts.is_empty(),
        "proxy_mode": &config.policy.proxy_mode,
        "expected_candidate_prompts": expected_candidate_prompts,
    })
}

fn sha256_text(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
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
                context,
                state,
                resources,
                None,
                response,
                reward,
                usage,
                cost_usd,
                cache_key,
                cache_hit,
                stage.clone(),
                example_id.clone(),
            )?;
            record_rollout_resilience_sample(
                &context.config,
                Some(&mut context.events),
                &mut state.cursor.pipeline_state.rollout_resilience,
                RolloutResilienceObservation {
                    stage: &stage,
                    example_id: &example_id,
                    degraded: false,
                    failure: None,
                    provider_signal: &ProviderSignal::default(),
                },
            )?;
        }
        StoredRuntimeOutcome::RolloutBatch { outcomes } => {
            for outcome in outcomes {
                let stage = outcome.stage.clone();
                let example_id = outcome.example_id.clone();
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
                record_rollout_resilience_sample(
                    &context.config,
                    Some(&mut context.events),
                    &mut state.cursor.pipeline_state.rollout_resilience,
                    RolloutResilienceObservation {
                        stage: &stage,
                        example_id: &example_id,
                        degraded: false,
                        failure: None,
                        provider_signal: &ProviderSignal::default(),
                    },
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

fn consume_failed_rollout_job_as_degraded(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    job: &OptimizerJob,
) -> Result<bool> {
    if !matches!(job.kind, OptimizerJobKind::Rollout)
        || matches!(job.status, OptimizerJobStatus::Cancelled)
    {
        return Ok(false);
    }
    let failure = job.failure.clone().unwrap_or_else(|| {
        FailurePayload::from_optimizer_error(&OptimizerError::Failed(format!(
            "GEPA runtime job {} {}",
            job.job_id,
            job.status.as_str()
        )))
    });
    if !failure.retryable {
        return Ok(false);
    }
    let provider_signal = provider_signal_from_failure(&context.config, Some(&failure));
    let dispatch = runtime::RuntimeEffectDispatchPayload::from_job(job)?;
    match dispatch.dispatch {
        runtime::RuntimeEffectDispatchKind::Rollout {
            candidate_id,
            stage,
            example_id,
            ..
        } => {
            let cache_key = job
                .payload
                .get("cache_key")
                .and_then(Value::as_str)
                .unwrap_or(&job.job_id)
                .to_string();
            let outcome = degraded_runtime_rollout_outcome_for_cache_key(
                &candidate_id,
                &stage,
                &example_id,
                &cache_key,
                &failure,
                &provider_signal,
            )?;
            consume_rollout_outcome(
                context,
                state,
                resources,
                None,
                outcome.response,
                outcome.reward,
                outcome.usage,
                outcome.cost_usd,
                outcome.cache_key,
                outcome.cache_hit,
                outcome.stage.clone(),
                outcome.example_id.clone(),
            )?;
            record_rollout_resilience_sample(
                &context.config,
                Some(&mut context.events),
                &mut state.cursor.pipeline_state.rollout_resilience,
                RolloutResilienceObservation {
                    stage: &stage,
                    example_id: &example_id,
                    degraded: true,
                    failure: Some(&failure),
                    provider_signal: &provider_signal,
                },
            )?;
        }
        runtime::RuntimeEffectDispatchKind::RolloutBatch {
            cache_profile,
            rollouts,
            ..
        } => {
            for (idx, rollout) in rollouts.into_iter().enumerate() {
                let cache_key = format!("{}:{}:{}", job.job_id, cache_profile, idx);
                let outcome = degraded_runtime_rollout_outcome_for_cache_key(
                    &rollout.candidate_id,
                    &rollout.stage,
                    &rollout.example_id,
                    &cache_key,
                    &failure,
                    &provider_signal,
                )?;
                consume_rollout_outcome(
                    context,
                    state,
                    resources,
                    Some(outcome.candidate_id.clone()),
                    outcome.response,
                    outcome.reward,
                    outcome.usage,
                    outcome.cost_usd,
                    outcome.cache_key,
                    outcome.cache_hit,
                    outcome.stage.clone(),
                    outcome.example_id.clone(),
                )?;
                record_rollout_resilience_sample(
                    &context.config,
                    Some(&mut context.events),
                    &mut state.cursor.pipeline_state.rollout_resilience,
                    RolloutResilienceObservation {
                        stage: &rollout.stage,
                        example_id: &rollout.example_id,
                        degraded: true,
                        failure: Some(&failure),
                        provider_signal: &provider_signal,
                    },
                )?;
            }
        }
        runtime::RuntimeEffectDispatchKind::Proposer { .. } => return Ok(false),
    }
    Ok(true)
}

fn schedule_failed_rollout_retry_if_allowed(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    job: &OptimizerJob,
) -> Result<Option<GepaAdvanceOutcome>> {
    if !matches!(job.kind, OptimizerJobKind::Rollout)
        || matches!(job.status, OptimizerJobStatus::Cancelled)
    {
        return Ok(None);
    }
    let failure = job.failure.clone().unwrap_or_else(|| {
        FailurePayload::from_optimizer_error(&OptimizerError::Failed(format!(
            "GEPA runtime job {} {}",
            job.job_id,
            job.status.as_str()
        )))
    });
    if !failure.retryable || job.attempt >= job.retry_policy.max_attempts {
        return Ok(None);
    }
    let backoff_seconds = rollout_retry_backoff_seconds(job);
    let Some(updated) = context.workspace.schedule_terminal_optimizer_job_retry(
        &job.run_id,
        &job.job_id,
        backoff_seconds,
        &failure,
    )?
    else {
        return Ok(None);
    };
    persist_gepa_run_state(
        context,
        state,
        resources,
        state.cursor.phase.clone(),
        "retry_scheduled",
        "scheduled GEPA failed rollout job retry",
        Map::new(),
    )?;
    Ok(Some(GepaAdvanceOutcome {
        action: planner::GepaTickAction::Noop,
        terminal: false,
        result: None,
        message: format!(
            "scheduled GEPA failed rollout job retry: {} attempt={}/{}",
            updated.job_id,
            updated.attempt.saturating_add(1),
            updated.retry_policy.max_attempts
        ),
    }))
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
        runtime_substrate: response
            .get("runtime_substrate")
            .and_then(Value::as_str)
            .unwrap_or(context.config.proposer.runtime_substrate.as_str())
            .to_string(),
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
        "provider".to_string(),
        Value::String(context.config.proposer.provider.clone()),
    );
    metadata.insert(
        "runtime_substrate".to_string(),
        Value::String(outcome.runtime_substrate.clone()),
    );
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
            "model": context.config.proposer.model,
            "provider": context.config.proposer.provider,
            "backend": backend,
            "cost_usd": cost_usd,
            "runtime_substrate": outcome.runtime_substrate.clone(),
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
            Some(&context.transitions),
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
    let row_index = rollout_row_index_by_example_id(&rows, &stage, &example_id)?;
    if rollout_scores_contain_example(&active.scores, &example_id) {
        context.events.emit(
            "rollout.outcome.duplicate_ignored",
            "Duplicate rollout outcome ignored",
            json!({
                "candidate_id": candidate.candidate_id,
                "stage": stage,
                "generation": active.generation,
                "proposal_index": active.proposal_index,
                "example_id": example_id,
            }),
        )?;
        return Ok(());
    }
    let row = rows.get(row_index).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "rollout outcome row index {row_index} is outside {} rows",
            rows.len()
        ))
    })?;
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
    let task_id = row_task_id(row);
    active.scores.push(RolloutScore {
        example_id,
        task_id,
        reward,
    });
    active.sensor_frames.push(sensor_frame);
    active.next_row_index = next_unscored_row_index(&active.row_ids, &active.scores);
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
    let row_index = rollout_row_index_by_example_id(&rows, &stage, &example_id)?;
    if rollout_scores_contain_example(&candidate_eval.scores, &example_id) {
        context.events.emit(
            "rollout.outcome.duplicate_ignored",
            "Duplicate grouped rollout outcome ignored",
            json!({
                "candidate_id": candidate.candidate_id,
                "stage": stage,
                "generation": candidate_eval.generation,
                "proposal_index": candidate_eval.proposal_index,
                "example_id": example_id,
            }),
        )?;
        return Ok(());
    }
    let row = rows.get(row_index).ok_or_else(|| {
        OptimizerError::Invariant(format!(
            "rollout outcome row index {row_index} is outside {} rows",
            rows.len()
        ))
    })?;
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
    let task_id = row_task_id(row);
    candidate_eval.scores.push(RolloutScore {
        example_id,
        task_id,
        reward,
    });
    candidate_eval.sensor_frames.push(sensor_frame);
    candidate_eval.next_row_index =
        next_unscored_row_index(&candidate_eval.row_ids, &candidate_eval.scores);
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
    let task_id = row_task_id(row);
    let overlay = CandidateOverlay {
        candidate: PromptCandidatePayload::from_map(candidate.payload.clone()),
        metadata: Map::new(),
    };
    let prompt_assertions = prompt_assertions_for_candidate(&overlay.candidate, &context.config);
    let request = json!({
        "submission_mode": rollout_submission_mode_for_request(&context.config),
        "task_id": resources.rollout_task_id,
        "candidate": overlay.candidate.to_value(),
        "candidate_overlay": overlay,
        "prompt_assertions": prompt_assertions,
        "policy": rollout_policy_for_request(&context.config),
        "task": row,
        "metadata": {
                "candidate_id": candidate.candidate_id,
                "task_id": task_id,
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
            task_id: &task_id,
            split: &sensor_frame.split,
            evaluation_stage: stage,
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
            Some(&context.transitions),
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
    normalize_event_feed(
        &context.paths.event_feed_path,
        &context.paths.normalized_event_feed_path,
        &context.paths.run_dir,
    )?;
    let storage_summary = record_terminal_storage_snapshot(
        &context.paths,
        &context.config.run.run_id,
        &mut context.events,
    )?;
    context.events.flush()?;
    context
        .workspace
        .record_event_stream(&context.config.run.run_id, context.events.records())?;
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
            Some(storage_summary.clone()),
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
            Some(storage_summary.clone()),
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
    state.cursor.candidates =
        serde_json::to_value(checkpoint_candidate_records(&state.candidates))?;
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
        .record_checkpoint_compacting_previous(&context.config.run.run_id, &checkpoint)
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
            Some(&context.transitions),
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
            Some(&context.transitions),
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

fn defer_active_rollout_evaluation_for_budget(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    resources: &GepaStepResources,
    requested_rollouts: usize,
    available_rollouts: usize,
) -> Result<GepaAdvanceOutcome> {
    let active = state.active_evaluation.clone().ok_or_else(|| {
        OptimizerError::Invariant("cannot defer without active evaluation".to_string())
    })?;
    if context.state_machine.state() == OptimizerRunState::RolloutRunning {
        transition_run(
            &context.workspace,
            &mut context.events,
            &mut context.state_machine,
            Some(&context.transitions),
            OptimizerRunState::Evaluating,
            OptimizerTransitionTrigger::RolloutsFinished,
            "Rollouts stopped by budget",
            json!({
                "stage": active.stage,
                "requested_rollouts": requested_rollouts,
                "available_rollouts": available_rollouts,
            }),
        )?;
    }
    if active.is_group() {
        let mut complete_candidates = Vec::new();
        for candidate_active in active.candidate_evaluations.iter().cloned() {
            if active_candidate_rollout_complete(&candidate_active) {
                complete_candidates.push(candidate_active);
            } else {
                commit_budget_deferred_active_candidate(
                    context,
                    state,
                    &active.stage,
                    &candidate_active,
                    requested_rollouts,
                    available_rollouts,
                )?;
            }
        }
        if !complete_candidates.is_empty() {
            let mut complete_active = active.clone();
            complete_active.candidate_evaluations = complete_candidates;
            complete_active.row_ids = complete_active
                .candidate_evaluations
                .iter()
                .flat_map(|candidate| {
                    candidate
                        .row_ids
                        .iter()
                        .map(|row_id| format!("{}:{row_id}", candidate.candidate_id))
                })
                .collect();
            complete_active.next_row_index = complete_active.row_ids.len();
            state.active_evaluation = Some(complete_active.clone());
            return finalize_active_rollout_group(context, state, resources, complete_active);
        }
    } else if active_rollout_evaluation_complete(&active) {
        return finalize_active_rollout_evaluation(context, state, resources);
    } else {
        commit_budget_deferred_active_rollout(
            context,
            state,
            &active,
            requested_rollouts,
            available_rollouts,
        )?;
    }
    state.cursor.proposal_index = state.proposal_queue.len();
    state.active_evaluation = None;
    move_to_proposer_waiting(
        context,
        state,
        resources,
        "active rollout evaluation deferred by budget",
    )
}

fn commit_budget_deferred_active_rollout(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    active: &GepaActiveEvaluation,
    requested_rollouts: usize,
    available_rollouts: usize,
) -> Result<()> {
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
    if let Some(candidate_idx) = active.candidate_index {
        let candidate = state.candidates.get_mut(candidate_idx).ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "budget-deferred candidate index {candidate_idx} is outside candidate registry"
            ))
        })?;
        if matches!(
            active.stage.as_str(),
            "candidate_minibatch" | "candidate_full_train"
        ) {
            candidate.status = "deferred_budget".to_string();
        }
        candidate.sensor_frames.extend(eval.sensor_frames.clone());
        persist_candidate_snapshot(
            &mut context.workspace,
            &context.config.run.run_id,
            candidate,
        )?;
        context.events.emit(
            "candidate.deferred",
            "Candidate deferred by rollout budget",
            json!({
                "candidate_id": candidate.candidate_id,
                "generation": active.generation,
                "stage": active.stage,
                "completed_rollouts": eval.rollout_count,
                "required_rollouts": active.row_ids.len(),
                "requested_rollouts": requested_rollouts,
                "available_rollouts": available_rollouts,
            }),
        )?;
    }
    Ok(())
}

fn commit_budget_deferred_active_candidate(
    context: &mut GepaRunContext,
    state: &mut GepaRunState,
    stage: &str,
    candidate_active: &GepaActiveCandidateEvaluation,
    requested_rollouts: usize,
    available_rollouts: usize,
) -> Result<()> {
    let eval = evaluation_from_active_candidate(candidate_active);
    state.total_usage.merge(&eval.usage);
    state.total_cost += eval.cost_usd;
    state.rollout_count += eval.rollout_count;
    append_rollout_usage(&mut state.usage_ledger, &eval);
    let candidate = state
        .candidates
        .get_mut(candidate_active.candidate_index)
        .ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "budget-deferred candidate index {} is outside candidate registry",
                candidate_active.candidate_index
            ))
        })?;
    if matches!(stage, "candidate_minibatch" | "candidate_full_train") {
        candidate.status = "deferred_budget".to_string();
    }
    candidate.sensor_frames.extend(eval.sensor_frames.clone());
    persist_candidate_snapshot(
        &mut context.workspace,
        &context.config.run.run_id,
        candidate,
    )?;
    context.events.emit(
        "candidate.deferred",
        "Candidate deferred by rollout budget",
        json!({
            "candidate_id": candidate.candidate_id,
            "generation": candidate_active.generation,
            "stage": stage,
            "completed_rollouts": eval.rollout_count,
            "required_rollouts": candidate_active.row_ids.len(),
            "requested_rollouts": requested_rollouts,
            "available_rollouts": available_rollouts,
        }),
    )?;
    Ok(())
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
    record_candidate_transition(
        context,
        &candidate_id,
        None,
        active.generation,
        CandidateState::FullTrainEvaluated,
        CandidateTrigger::EvaluationFinished,
        json!({
            "stage": "seed_full_train",
            "candidate_id": &candidate_id,
            "train_reward": eval.average_reward,
        }),
    )?;
    record_candidate_transition(
        context,
        &candidate_id,
        None,
        active.generation,
        CandidateState::Accepted,
        CandidateTrigger::FullTrainAccepted,
        json!({
            "stage": "seed_full_train",
            "candidate_id": &candidate_id,
            "train_reward": eval.average_reward,
            "seed": true,
        }),
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
            Some(&context.transitions),
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
        &context.config.taskset.train_split,
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
        split: &context.config.taskset.train_split,
        source_stages: &["candidate_minibatch"],
        evaluation_stage: "candidate_minibatch",
    })?;
    let parent_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
        objective_set: &resources.objective_set,
        candidate: &state.candidates[parent_idx],
        rows: &minibatch_rows,
        split: &context.config.taskset.train_split,
        source_stages: parent_minibatch_reference_source_stages(),
        evaluation_stage: "parent_minibatch_reference",
    })?;
    ensure_paired_minibatch_score_vectors(&candidate_minibatch_vector, &parent_minibatch_vector)?;
    let reflection_alignment =
        minibatch_reflection_alignment(&minibatch_rows, &resources.reflection_rows)?;
    let minibatch_preference = compare_score_vectors(ScoreVectorPreferenceInput {
        objective_set: &resources.objective_set,
        split: &context.config.taskset.train_split,
        evaluation_stage: "candidate_minibatch",
        challenger: &candidate_minibatch_vector,
        incumbent: &parent_minibatch_vector,
        accept_equal: false,
        acceptance_criterion: Some("primary_improvement"),
        objective_acceptance: None,
        margin: 0.0,
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
            "minibatch_delta": eval.average_reward - parent_minibatch_reward,
            "accepted_minibatch": decision.accepted_minibatch,
            "parent_minibatch_source_stage": "parent_minibatch_reference",
            "candidate_minibatch_task_ids": candidate_minibatch_vector.task_ids.clone(),
            "parent_minibatch_task_ids": parent_minibatch_vector.task_ids.clone(),
            "reflection_alignment": reflection_alignment,
        }),
    )?;
    record_candidate_transition(
        context,
        &state.candidates[candidate_idx].candidate_id,
        Some(&parent_id),
        active.generation,
        CandidateState::MinibatchEvaluated,
        CandidateTrigger::EvaluationFinished,
        json!({
            "stage": "candidate_minibatch",
            "candidate_id": &state.candidates[candidate_idx].candidate_id,
            "parent_id": &parent_id,
            "minibatch_reward": eval.average_reward,
            "parent_minibatch_reward": parent_minibatch_reward,
            "minibatch_delta": eval.average_reward - parent_minibatch_reward,
            "accepted_minibatch": decision.accepted_minibatch,
            "candidate_minibatch_task_ids": candidate_minibatch_vector.task_ids.clone(),
            "parent_minibatch_task_ids": parent_minibatch_vector.task_ids.clone(),
        }),
    )?;
    if !decision.accepted_minibatch {
        decision.reason = minibatch_preference.reason;
        state.candidates[candidate_idx].status = "rejected_minibatch".to_string();
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            Some(&parent_id),
            active.generation,
            CandidateState::RejectedMinibatch,
            CandidateTrigger::MinibatchRejected,
            serde_json::to_value(&decision)?,
        )?;
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
    record_candidate_transition(
        context,
        &state.candidates[candidate_idx].candidate_id,
        Some(&parent_id),
        active.generation,
        CandidateState::AcceptedMinibatch,
        CandidateTrigger::MinibatchAccepted,
        serde_json::to_value(&decision)?,
    )?;
    let full_train_capacity =
        remaining_train_rollout_capacity(&context.workspace, &context.config, state.rollout_count)?;
    if full_train_capacity < resources.train_rows.len() {
        state.candidates[candidate_idx].status = "deferred_budget".to_string();
        decision.reason = "insufficient rollout budget for full-train evaluation".to_string();
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            Some(&parent_id),
            active.generation,
            CandidateState::DeferredBudget,
            CandidateTrigger::DeferredBudget,
            serde_json::to_value(&decision)?,
        )?;
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
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            Some(&parent_id),
            active.generation,
            CandidateState::DeferredBudget,
            CandidateTrigger::DeferredBudget,
            serde_json::to_value(&decision)?,
        )?;
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
            &context.config.taskset.train_split,
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
            split: &context.config.taskset.train_split,
            source_stages: &["candidate_minibatch"],
            evaluation_stage: "candidate_minibatch",
        })?;
        let parent_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set: &resources.objective_set,
            candidate: &state.candidates[parent_idx],
            rows: &minibatch_rows,
            split: &context.config.taskset.train_split,
            source_stages: parent_minibatch_reference_source_stages(),
            evaluation_stage: "parent_minibatch_reference",
        })?;
        ensure_paired_minibatch_score_vectors(
            &candidate_minibatch_vector,
            &parent_minibatch_vector,
        )?;
        let reflection_alignment =
            minibatch_reflection_alignment(&minibatch_rows, &resources.reflection_rows)?;
        let minibatch_preference = compare_score_vectors(ScoreVectorPreferenceInput {
            objective_set: &resources.objective_set,
            split: &context.config.taskset.train_split,
            evaluation_stage: "candidate_minibatch",
            challenger: &candidate_minibatch_vector,
            incumbent: &parent_minibatch_vector,
            accept_equal: false,
            acceptance_criterion: Some("primary_improvement"),
            objective_acceptance: None,
            margin: 0.0,
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
                "minibatch_delta": eval.average_reward - parent_minibatch_reward,
                "accepted_minibatch": decision.accepted_minibatch,
                "parent_minibatch_source_stage": "parent_minibatch_reference",
                "candidate_minibatch_task_ids": candidate_minibatch_vector.task_ids.clone(),
                "parent_minibatch_task_ids": parent_minibatch_vector.task_ids.clone(),
                "reflection_alignment": reflection_alignment,
            }),
        )?;
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            Some(&parent_id),
            candidate_active.generation,
            CandidateState::MinibatchEvaluated,
            CandidateTrigger::EvaluationFinished,
            json!({
                "stage": "candidate_minibatch",
                "candidate_id": &state.candidates[candidate_idx].candidate_id,
                "parent_id": &parent_id,
                "minibatch_reward": eval.average_reward,
                "parent_minibatch_reward": parent_minibatch_reward,
                "minibatch_delta": eval.average_reward - parent_minibatch_reward,
                "accepted_minibatch": decision.accepted_minibatch,
                "candidate_minibatch_task_ids": candidate_minibatch_vector.task_ids.clone(),
                "parent_minibatch_task_ids": parent_minibatch_vector.task_ids.clone(),
            }),
        )?;
        if !decision.accepted_minibatch {
            decision.reason = minibatch_preference.reason;
            state.candidates[candidate_idx].status = "rejected_minibatch".to_string();
            record_candidate_transition(
                context,
                &state.candidates[candidate_idx].candidate_id,
                Some(&parent_id),
                candidate_active.generation,
                CandidateState::RejectedMinibatch,
                CandidateTrigger::MinibatchRejected,
                serde_json::to_value(&decision)?,
            )?;
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
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            Some(&parent_id),
            candidate_active.generation,
            CandidateState::AcceptedMinibatch,
            CandidateTrigger::MinibatchAccepted,
            serde_json::to_value(&decision)?,
        )?;
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
            record_candidate_transition(
                context,
                &state.candidates[candidate_idx].candidate_id,
                Some(&parent_id),
                candidate_active.generation,
                CandidateState::DeferredBudget,
                CandidateTrigger::DeferredBudget,
                serde_json::to_value(&decision)?,
            )?;
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
            Some(&context.transitions),
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
        split: &context.config.taskset.train_split,
        source_stages: &["candidate_full_train"],
        evaluation_stage: "candidate_full_train",
    })?;
    let best_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
        objective_set: &resources.objective_set,
        candidate: &state.candidates[best_idx],
        rows: &resources.train_rows,
        split: &context.config.taskset.train_split,
        source_stages: &["seed_full_train", "candidate_full_train"],
        evaluation_stage: "best_full_train_reference",
    })?;
    let train_preference = compare_score_vectors(ScoreVectorPreferenceInput {
        objective_set: &resources.objective_set,
        split: &context.config.taskset.train_split,
        evaluation_stage: "candidate_full_train",
        challenger: &candidate_train_vector,
        incumbent: &best_train_vector,
        accept_equal: false,
        acceptance_criterion: Some("primary_improvement"),
        objective_acceptance: None,
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
    let parent_id_for_transition = state.candidates[candidate_idx].parent_id.clone();
    record_candidate_transition(
        context,
        &state.candidates[candidate_idx].candidate_id,
        parent_id_for_transition.as_deref(),
        active.generation,
        CandidateState::FullTrainEvaluated,
        CandidateTrigger::EvaluationFinished,
        json!({
            "stage": "candidate_full_train",
            "candidate_id": &state.candidates[candidate_idx].candidate_id,
            "parent_id": &state.candidates[candidate_idx].parent_id,
            "train_reward": eval.average_reward,
            "best_train_reward": state.candidates[best_idx].train_reward,
            "accepted_full_train": accepted,
        }),
    )?;
    record_candidate_transition(
        context,
        &state.candidates[candidate_idx].candidate_id,
        parent_id_for_transition.as_deref(),
        active.generation,
        if accepted {
            CandidateState::Accepted
        } else {
            CandidateState::RejectedFullTrain
        },
        if accepted {
            CandidateTrigger::FullTrainAccepted
        } else {
            CandidateTrigger::FullTrainRejected
        },
        serde_json::to_value(&decision)?,
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
            split: &context.config.taskset.train_split,
            source_stages: &["candidate_full_train"],
            evaluation_stage: "candidate_full_train",
        })?;
        let best_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
            objective_set: &resources.objective_set,
            candidate: &state.candidates[best_idx],
            rows: &resources.train_rows,
            split: &context.config.taskset.train_split,
            source_stages: &["seed_full_train", "candidate_full_train"],
            evaluation_stage: "best_full_train_reference",
        })?;
        let train_preference = compare_score_vectors(ScoreVectorPreferenceInput {
            objective_set: &resources.objective_set,
            split: &context.config.taskset.train_split,
            evaluation_stage: "candidate_full_train",
            challenger: &candidate_train_vector,
            incumbent: &best_train_vector,
            accept_equal: false,
            acceptance_criterion: Some("primary_improvement"),
            objective_acceptance: None,
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
        let parent_id_for_transition = state.candidates[candidate_idx].parent_id.clone();
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            parent_id_for_transition.as_deref(),
            candidate_active.generation,
            CandidateState::FullTrainEvaluated,
            CandidateTrigger::EvaluationFinished,
            json!({
                "stage": "candidate_full_train",
                "candidate_id": &state.candidates[candidate_idx].candidate_id,
                "parent_id": &state.candidates[candidate_idx].parent_id,
                "train_reward": eval.average_reward,
                "best_train_reward": state.candidates[best_idx].train_reward,
                "accepted_full_train": accepted,
            }),
        )?;
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            parent_id_for_transition.as_deref(),
            candidate_active.generation,
            if accepted {
                CandidateState::Accepted
            } else {
                CandidateState::RejectedFullTrain
            },
            if accepted {
                CandidateTrigger::FullTrainAccepted
            } else {
                CandidateTrigger::FullTrainRejected
            },
            serde_json::to_value(&decision)?,
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
        &context.config.taskset.train_split,
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
            Some(&context.transitions),
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
    record_proposer_round_started(
        context,
        &queued.job.job_id,
        state.cursor.generation,
        &parent_id,
        json!({
            "job_id": &queued.job.job_id,
            "runtime_effect_id": &queued.effect.runtime_effect_id,
            "parent_candidate_id": &parent_id,
            "generation": state.cursor.generation,
            "model": &context.config.proposer.model,
            "provider": &context.config.proposer.provider,
        }),
    )?;
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
        "runtime_substrate": context.config.proposer.runtime_substrate.as_str(),
        "model": context.config.proposer.model,
        "generation": state.cursor.generation,
        "parent": parent,
        "candidates": state.candidates,
        "program": resources.program,
        "task_pool_rows": task_pool_rows_value(
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
    cache_metadata.insert(
        "runtime_substrate".to_string(),
        json!(context.config.proposer.runtime_substrate.as_str()),
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
                "runtime_substrate": context.config.proposer.runtime_substrate.as_str(),
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
        let allowed_fields = candidate_allowed_fields(&resources.program, &context.config);
        let proposed_payload = proposed_payload_for_candidate_admission(
            &proposal,
            &allowed_fields,
            state.cursor.generation,
            proposal_index,
        )?;
        let payload = normalize_candidate_payload(
            &resources.program,
            &context.config,
            &proposal_parent.payload,
            proposed_payload,
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
            record_candidate_registered(
                context,
                &candidate,
                Some(state.cursor.generation),
                json!({
                    "source": &candidate.source,
                    "parent_id": &candidate.parent_id,
                    "generation": state.cursor.generation,
                    "proposal_index": proposal_index,
                }),
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
            &context.config.taskset.train_split,
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
        Ok(GepaPipelineRuntimePlan::AsyncPipelined(plan))
        | Ok(GepaPipelineRuntimePlan::FlashEvolve(plan)) => plan.max_in_flight_candidates.max(1),
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
            Some(&context.transitions),
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
            .max(minimum_terminal_heldout_candidate_count(
                &all_heldout_indices,
                state.best_idx,
            ))
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
            terminalize_gepa_run_state(
                context,
                state,
                GepaCursorPhase::Failed,
                "failed",
                "Heldout required but skipped due to limits",
                json!({
                    "schema_version": "gepa_terminal_heldout_required.v1",
                    "error_code": "gepa_terminal_heldout_not_evaluated",
                    "message": "GEPA cannot complete with a train-only best candidate when heldout rows are configured",
                    "best_candidate_id": state.candidates[best_idx].candidate_id,
                    "required_rollouts": required_rollouts,
                    "available_rollouts": available_rollouts,
                }),
            )?;
            return Ok(GepaAdvanceOutcome {
                action: planner::GepaTickAction::TerminalizeRun {
                    run_id: context.config.run.run_id.clone(),
                    status: "failed".to_string(),
                },
                terminal: true,
                result: None,
                message: "heldout required but skipped due to limits".to_string(),
            });
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
            heldout_split: &context.config.taskset.heldout_split,
            heldout_rows: &resources.heldout_rows,
            train_split: &context.config.taskset.train_split,
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
    let best_train_reward = state
        .best_idx
        .and_then(|idx| state.candidates.get(idx))
        .and_then(|candidate| candidate.train_reward);
    let mut indices = state
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| {
            heldout_candidate_eligible(candidate, state.best_idx, best_train_reward, idx)
                .then_some(idx)
        })
        .collect::<Vec<_>>();
    if indices.is_empty() {
        if let Some(best_idx) = state.best_idx {
            indices.push(best_idx);
        }
    }
    indices
}

fn heldout_candidate_eligible(
    candidate: &CandidateRecord,
    best_idx: Option<usize>,
    best_train_reward: Option<f64>,
    candidate_idx: usize,
) -> bool {
    let Some(train_reward) = candidate.train_reward else {
        return false;
    };
    if candidate.train_scores.is_empty() {
        return false;
    }
    if best_idx == Some(candidate_idx) || candidate.source == "seed" {
        return true;
    }
    if !matches!(candidate.status.as_str(), "rejected_full_train") {
        return false;
    }
    let Some(best_train_reward) = best_train_reward else {
        return false;
    };
    train_reward + f64::EPSILON >= best_train_reward
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
    if max_candidates < minimum_terminal_heldout_candidate_count(&indices, best_idx) {
        return Vec::new();
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

fn minimum_terminal_heldout_candidate_count(indices: &[usize], best_idx: Option<usize>) -> usize {
    if indices.is_empty() {
        return 0;
    }
    if best_idx.is_some_and(|idx| idx != 0 && indices.contains(&idx)) && indices.contains(&0) {
        return 2;
    }
    1
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
    let parent_id_for_transition = state.candidates[candidate_idx].parent_id.clone();
    record_candidate_transition(
        context,
        &state.candidates[candidate_idx].candidate_id,
        parent_id_for_transition.as_deref(),
        active.generation,
        CandidateState::HeldoutScored,
        CandidateTrigger::HeldoutFinished,
        json!({
            "stage": "heldout",
            "candidate_id": &state.candidates[candidate_idx].candidate_id,
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
        let parent_id_for_transition = state.candidates[candidate_idx].parent_id.clone();
        record_candidate_transition(
            context,
            &state.candidates[candidate_idx].candidate_id,
            parent_id_for_transition.as_deref(),
            candidate_active.generation,
            CandidateState::HeldoutScored,
            CandidateTrigger::HeldoutFinished,
            json!({
                "stage": "heldout",
                "candidate_id": &state.candidates[candidate_idx].candidate_id,
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
            Some(&context.transitions),
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
    let Some(heldout_best_reward) = state.candidates[best_idx].heldout_reward else {
        terminalize_gepa_run_state(
            context,
            state,
            GepaCursorPhase::Failed,
            "failed",
            "Best candidate missing terminal heldout evidence",
            json!({
                "schema_version": "gepa_terminal_heldout_required.v1",
                "error_code": "gepa_best_candidate_missing_heldout",
                "message": "GEPA cannot complete with a best candidate that has no heldout_reward",
                "best_candidate_id": state.candidates[best_idx].candidate_id,
                "train_reward": state.candidates[best_idx].train_reward,
                "heldout_evaluated_candidate_count": state
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.heldout_reward.is_some())
                    .count(),
            }),
        )?;
        return Ok(GepaAdvanceOutcome {
            action: planner::GepaTickAction::TerminalizeRun {
                run_id: context.config.run.run_id.clone(),
                status: "failed".to_string(),
            },
            terminal: true,
            result: None,
            message: "best candidate missing terminal heldout evidence".to_string(),
        });
    };
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
            Some(&context.transitions),
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
    let artifact_candidates = artifact_candidate_records(&state.candidates);
    let best_candidate = serde_json::to_value(&artifact_candidates[best_idx])?;
    let candidate_registry = serde_json::to_value(&artifact_candidates)?;
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
    normalize_event_feed(
        &context.paths.event_feed_path,
        &context.paths.normalized_event_feed_path,
        &context.paths.run_dir,
    )?;
    let storage_summary = record_terminal_storage_snapshot(
        &context.paths,
        &context.config.run.run_id,
        &mut context.events,
    )?;
    context.events.flush()?;
    context
        .workspace
        .record_event_stream(&context.config.run.run_id, context.events.records())?;
    context.registry.append(&RunRegistryEntry::finished(
        &context.paths,
        &context.config,
        context.cache_mode,
        &context.cache_namespace,
        state.candidates[best_idx].candidate_id.clone(),
        state.total_cost,
        usage_value.clone(),
        Some(storage_summary.clone()),
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
            &context.paths.storage_report_path,
            "storage_report",
            "local_ops",
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
        storage_report_path: context.paths.storage_report_path.display().to_string(),
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
    state.cursor.terminal_summary = Some(compact_terminal_summary(&result_value));
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

fn record_terminal_storage_snapshot(
    paths: &ArtifactPaths,
    run_id: &str,
    events: &mut EventWriter,
) -> Result<Value> {
    let report = write_run_storage_report(RunStorageInspectionInput {
        run_dir: paths.run_dir.clone(),
        run_id: Some(run_id.to_string()),
        terminal: Some(true),
    })?;
    let summary = storage_registry_summary(&report);
    events.emit(
        "storage.snapshot.recorded",
        "GEPA terminal storage snapshot recorded",
        json!({
            "run_id": run_id,
            "storage": summary,
            "storage_report_path": paths.storage_report_path.display().to_string(),
        }),
    )?;
    Ok(summary)
}

fn storage_registry_summary(report: &Value) -> Value {
    json!({
        "schema": "synth.optimizer.storage_registry_summary.v1",
        "bytes": report.get("bytes").cloned().unwrap_or(json!(0)),
        "reclaimable_bytes": report.get("reclaimable_bytes").cloned().unwrap_or(json!(0)),
        "terminal": report.get("terminal").cloned().unwrap_or(json!(false)),
        "terminal_status": report.get("terminal_status").cloned().unwrap_or(Value::Null),
        "storage_report_path": report
            .get("storage_report_path")
            .cloned()
            .unwrap_or(Value::Null),
        "recommendation": report.get("recommendation").cloned().unwrap_or(Value::Null),
    })
}

pub fn execute_gepa_with_options(
    mut config: SynthOptimizerConfig,
    options: GepaExecutionOptions,
) -> Result<GepaRunResult> {
    config.resolve_runtime_targets()?;
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
        if let Some(result) = terminal_result_from_cursor(&context, &restored_cursor)? {
            return Ok(result);
        }
    }
    let container_inputs = ensure_container_inputs(&mut context)?;
    let GepaRunContext {
        paths,
        mut workspace,
        registry,
        mut events,
        mut state_machine,
        transitions,
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
    let mut rollout_resilience = GepaRolloutResilienceState::default();
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
                    transitions: &transitions,
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
                    transitions: &transitions,
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
            Some(&transitions),
            OptimizerRunState::RolloutQueueing,
            OptimizerTransitionTrigger::RolloutsQueued,
            "Seed candidate rollouts queued",
            json!({"candidate_id": candidates[0].candidate_id, "stage": "seed_full_train"}),
        )?;
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            Some(&transitions),
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
            events: &mut events,
            rollout_resilience: &mut rollout_resilience,
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
                        transitions: &transitions,
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
            Some(&transitions),
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
            Some(&transitions),
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
            &config.taskset.train_split,
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
            Some(&transitions),
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
            task_pool_rows: task_pool_rows_value(
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
                        transitions: &transitions,
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
            "provider".to_string(),
            Value::String(config.proposer.provider.clone()),
        );
        metadata.insert(
            "runtime_substrate".to_string(),
            Value::String(proposer_outcome.runtime_substrate.clone()),
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
                "model": config.proposer.model,
                "provider": config.proposer.provider,
                "backend": proposer_outcome.backend,
                "cost_usd": proposer_outcome.cost_usd,
                "runtime_substrate": proposer_outcome.runtime_substrate,
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
                Some(&transitions),
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
            Some(&transitions),
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
                &config.taskset.train_split,
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
                    Some(&transitions),
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
                    events: &mut events,
                    rollout_resilience: &mut rollout_resilience,
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
                    Some(&transitions),
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
                &config.taskset.train_split,
            )?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "parent candidate {} is missing minibatch reference scores for generation {}",
                    proposal_parent.candidate_id, generation
                ))
            })?;
            let allowed_fields = candidate_allowed_fields(&program, &config);
            let proposed_payload = proposed_payload_for_candidate_admission(
                &proposal,
                &allowed_fields,
                generation,
                proposal_index,
            )?;
            let payload = normalize_candidate_payload(
                &program,
                &config,
                &proposal_parent.payload,
                proposed_payload,
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
                Some(&transitions),
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
                events: &mut events,
                rollout_resilience: &mut rollout_resilience,
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
                            transitions: &transitions,
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
                Some(&transitions),
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
                    split: &config.taskset.train_split,
                    source_stages: &["candidate_minibatch"],
                    evaluation_stage: "candidate_minibatch",
                })?;
            let parent_minibatch_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set: &objective_set,
                candidate: &proposal_parent,
                rows: &minibatch_rows,
                split: &config.taskset.train_split,
                source_stages: parent_minibatch_reference_source_stages(),
                evaluation_stage: "parent_minibatch_reference",
            })?;
            ensure_paired_minibatch_score_vectors(
                &candidate_minibatch_vector,
                &parent_minibatch_vector,
            )?;
            let reflection_alignment =
                minibatch_reflection_alignment(&minibatch_rows, &reflection_rows)?;
            let minibatch_preference = compare_score_vectors(ScoreVectorPreferenceInput {
                objective_set: &objective_set,
                split: &config.taskset.train_split,
                evaluation_stage: "candidate_minibatch",
                challenger: &candidate_minibatch_vector,
                incumbent: &parent_minibatch_vector,
                accept_equal: false,
                acceptance_criterion: Some("primary_improvement"),
                objective_acceptance: None,
                margin: 0.0,
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
                    "minibatch_delta": eval.average_reward - parent_minibatch_reward,
                    "accepted_minibatch": minibatch_preference.preferred,
                    "parent_minibatch_source_stage": "parent_minibatch_reference",
                    "candidate_minibatch_task_ids": candidate_minibatch_vector.task_ids.clone(),
                    "parent_minibatch_task_ids": parent_minibatch_vector.task_ids.clone(),
                    "reflection_alignment": reflection_alignment,
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
                    Some(&transitions),
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
                    Some(&transitions),
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
                    Some(&transitions),
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
                Some(&transitions),
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
                Some(&transitions),
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
                events: &mut events,
                rollout_resilience: &mut rollout_resilience,
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
                            transitions: &transitions,
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
                Some(&transitions),
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
                split: &config.taskset.train_split,
                source_stages: &["candidate_full_train"],
                evaluation_stage: "candidate_full_train",
            })?;
            let best_train_vector = score_vector_for_candidate(CandidateScoreVectorInput {
                objective_set: &objective_set,
                candidate: &candidates[best_idx],
                rows: &train_rows,
                split: &config.taskset.train_split,
                source_stages: &["seed_full_train", "candidate_full_train"],
                evaluation_stage: "best_full_train_reference",
            })?;
            let train_preference = compare_score_vectors(ScoreVectorPreferenceInput {
                objective_set: &objective_set,
                split: &config.taskset.train_split,
                evaluation_stage: "candidate_full_train",
                challenger: &candidate_train_vector,
                incumbent: &best_train_vector,
                accept_equal: false,
                acceptance_criterion: Some("primary_improvement"),
                objective_acceptance: None,
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
                Some(&transitions),
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
                Some(&transitions),
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
    let all_heldout_indices = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| {
            heldout_candidate_eligible(
                candidate,
                Some(best_idx),
                candidates
                    .get(best_idx)
                    .and_then(|candidate| candidate.train_reward),
                idx,
            )
            .then_some(idx)
        })
        .collect::<Vec<_>>();
    let mut heldout_indices = all_heldout_indices.clone();
    if heldout_indices.is_empty() {
        heldout_indices.push(best_idx);
    }
    let mut heldout_rollout_delta = 0usize;
    let mut heldout_cost_delta = 0.0;
    let heldout_available_rollouts =
        remaining_heldout_rollout_capacity(&workspace, &config, &candidates)?;
    let heldout_budget_breach = next_rollout_budget_breach(&workspace, &config)?;
    if heldout_budget_breach.is_none() {
        heldout_indices = budgeted_heldout_candidate_indices(
            &candidates,
            heldout_indices,
            Some(best_idx),
            heldout_available_rollouts,
            heldout_rows.len(),
        );
    }
    let heldout_required_rollouts = heldout_indices
        .len()
        .max(minimum_terminal_heldout_candidate_count(
            &all_heldout_indices,
            Some(best_idx),
        ))
        .saturating_mul(heldout_rows.len());
    let heldout_skipped = heldout_indices.is_empty()
        || heldout_available_rollouts < heldout_required_rollouts
        || heldout_budget_breach.is_some();
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
        let error = OptimizerError::Invariant(
            "GEPA cannot complete with a train-only best candidate when heldout rows are configured"
                .to_string(),
        );
        return fail_gepa_run_and_return(
            FailedGepaRunInput {
                workspace: &mut workspace,
                events: &mut events,
                state_machine: &mut state_machine,
                transitions: &transitions,
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
                message: "Heldout required but skipped due to limits",
                details: json!({
                    "schema_version": "gepa_terminal_heldout_required.v1",
                    "error_code": "gepa_terminal_heldout_not_evaluated",
                    "stage": "heldout",
                    "best_candidate_id": candidates[best_idx].candidate_id,
                    "required_rollouts": heldout_required_rollouts,
                    "available_rollouts": heldout_available_rollouts,
                }),
            },
            error,
        );
    } else {
        let total_heldout_candidates = all_heldout_indices.len();
        if heldout_indices.len() < total_heldout_candidates {
            events.emit(
                "heldout.partial",
                "Heldout limited to budgeted candidate subset",
                json!({
                    "candidate_count": heldout_indices.len(),
                    "total_candidate_count": total_heldout_candidates,
                    "available_rollouts": heldout_available_rollouts,
                    "required_rollouts": heldout_required_rollouts,
                }),
            )?;
        }
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            Some(&transitions),
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
            Some(&transitions),
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
                events: &mut events,
                rollout_resilience: &mut rollout_resilience,
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
                            transitions: &transitions,
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
            heldout_split: &config.taskset.heldout_split,
            heldout_rows: &heldout_rows,
            train_split: &config.taskset.train_split,
            train_rows: &train_rows,
            incumbent_idx: Some(best_idx),
        })?
        .unwrap_or(best_idx);
        transition_run(
            &workspace,
            &mut events,
            &mut state_machine,
            Some(&transitions),
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
        Some(&transitions),
        OptimizerRunState::Completed,
        OptimizerTransitionTrigger::RunCompleted,
        "GEPA run completed",
        json!({
            "best_candidate_id": candidates[best_idx].candidate_id,
            "heldout_reward": heldout_best_reward,
            "heldout_skipped": heldout_skipped,
        }),
    )?;

    let artifact_candidates = artifact_candidate_records(&candidates);
    let best_candidate = serde_json::to_value(&artifact_candidates[best_idx])?;
    let candidate_registry = serde_json::to_value(&artifact_candidates)?;
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
    normalize_event_feed(
        &paths.event_feed_path,
        &paths.normalized_event_feed_path,
        &paths.run_dir,
    )?;
    let storage_summary =
        record_terminal_storage_snapshot(&paths, &config.run.run_id, &mut events)?;
    events.flush()?;
    workspace.record_event_stream(&config.run.run_id, events.records())?;
    registry.append(&RunRegistryEntry::finished(
        &paths,
        &config,
        cache_mode,
        &cache_namespace,
        candidates[best_idx].candidate_id.clone(),
        total_cost,
        usage_value.clone(),
        Some(storage_summary.clone()),
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
        paths.artifact_ref(&paths.storage_report_path, "storage_report", "local_ops")?,
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
        storage_report_path: paths.storage_report_path.display().to_string(),
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
            terminal_summary: Some(compact_terminal_summary(&result_value)),
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
    task_ids: &[String],
    filters: Value,
) -> Result<TasksetTasksResponse> {
    let request_model = TasksetTasksRequest::new(split, task_ids, filters);
    let request = serde_json::to_value(&request_model)?;
    let response = cached_call(
        cache,
        &format!("{cache_namespace}:container.taskset_tasks"),
        &request,
        || {
            let response = client.taskset_tasks_typed(&request_model)?;
            Ok(serde_json::to_value(response)?)
        },
    )?;
    let response: TasksetTasksResponse = serde_json::from_value(response)?;
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
                objectives.push((name.to_string(), "tasks.objective".to_string()));
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

fn proposed_payload_for_candidate_admission(
    proposal: &ProposedCandidate,
    allowed_fields: &[String],
    generation: usize,
    proposal_index: usize,
) -> Result<BTreeMap<String, String>> {
    let proposed_payload = proposal.payload_map_for_allowed_fields(allowed_fields);
    if proposed_payload.is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "proposer proposal generation={generation} index={proposal_index} returned no mutable payload; shape={}",
            proposal.payload_shape_summary()
        )));
    }
    Ok(proposed_payload)
}

fn candidate_allowed_fields(program: &PromptProgram, config: &SynthOptimizerConfig) -> Vec<String> {
    let mutable_fields = program.mutable_field_ids();
    if mutable_fields.is_empty() {
        config.candidate.target_modules.clone()
    } else {
        mutable_fields
    }
}

fn normalize_candidate_payload(
    program: &PromptProgram,
    config: &SynthOptimizerConfig,
    parent_payload: &BTreeMap<String, String>,
    proposed_payload: BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    let allowed_fields = candidate_allowed_fields(program, config);
    let proposed_payload =
        normalize_single_module_proposal_shape(&allowed_fields, proposed_payload);
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

fn normalize_single_module_proposal_shape(
    allowed_fields: &[String],
    proposed_payload: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    if proposed_payload
        .keys()
        .any(|key| allowed_fields.iter().any(|field| field == key))
    {
        return proposed_payload;
    }

    let Some(module_id) = proposed_payload.get("module_id").map(String::as_str) else {
        return proposed_payload;
    };
    if !allowed_fields.iter().any(|field| field == module_id) {
        return proposed_payload;
    }

    for value_key in ["content", "value", "prompt", "instructions", "text"] {
        if let Some(value) = proposed_payload.get(value_key) {
            if !value.trim().is_empty() {
                return BTreeMap::from([(module_id.to_string(), value.clone())]);
            }
        }
    }
    proposed_payload
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
    &["parent_minibatch_reference"]
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

fn ensure_paired_minibatch_score_vectors(
    candidate: &ScoreVectorRecord,
    parent: &ScoreVectorRecord,
) -> Result<()> {
    ensure_matching_minibatch_ids(
        "task_id",
        &candidate.task_ids,
        &parent.task_ids,
        candidate,
        parent,
    )?;
    ensure_matching_minibatch_ids(
        "example_id",
        &candidate.example_ids,
        &parent.example_ids,
        candidate,
        parent,
    )
}

fn ensure_matching_minibatch_ids(
    label: &str,
    candidate_ids: &[String],
    parent_ids: &[String],
    candidate: &ScoreVectorRecord,
    parent: &ScoreVectorRecord,
) -> Result<()> {
    let candidate_set = candidate_ids.iter().cloned().collect::<BTreeSet<_>>();
    let parent_set = parent_ids.iter().cloned().collect::<BTreeSet<_>>();
    if candidate_set == parent_set {
        return Ok(());
    }
    let missing_from_parent = candidate_set
        .difference(&parent_set)
        .cloned()
        .collect::<Vec<_>>();
    let missing_from_candidate = parent_set
        .difference(&candidate_set)
        .cloned()
        .collect::<Vec<_>>();
    Err(OptimizerError::Invariant(format!(
        "paired minibatch score vectors must have identical {label} sets: candidate={} parent={} candidate_stage={} parent_stage={} missing_from_parent={:?} missing_from_candidate={:?}",
        candidate.candidate_id,
        parent.candidate_id,
        candidate.evaluation_stage,
        parent.evaluation_stage,
        missing_from_parent,
        missing_from_candidate
    )))
}

fn minibatch_reflection_alignment(
    minibatch_rows: &[Value],
    reflection_rows: &[Value],
) -> Result<Value> {
    let minibatch_task_ids = minibatch_rows
        .iter()
        .map(task_identity)
        .collect::<Result<BTreeSet<_>>>()?;
    let reflection_task_ids = reflection_rows
        .iter()
        .map(task_identity)
        .collect::<Result<BTreeSet<_>>>()?;
    let missing_from_reflection = minibatch_task_ids
        .difference(&reflection_task_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !missing_from_reflection.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "minibatch rows must be a subset of reflection rows; missing_from_reflection={:?}",
            missing_from_reflection
        )));
    }
    Ok(json!({
        "minibatch_task_ids": minibatch_task_ids.into_iter().collect::<Vec<_>>(),
        "reflection_task_ids": reflection_task_ids.into_iter().collect::<Vec<_>>(),
        "minibatch_subset_of_reflection": true,
    }))
}

fn score_vector_frame_matches_split(
    frame: &SensorFrame,
    requested_split: &str,
    source_stages: &[&str],
) -> bool {
    if frame.split == requested_split {
        return true;
    }
    // Heldout evaluation frames are tagged with the synthetic split "heldout"
    // while the heldout rows retain their dataset split, such as "test".
    frame.split == "heldout"
        && frame.evaluation_stage == "heldout"
        && source_stages.iter().any(|stage| *stage == "heldout")
}

fn score_vector_for_candidate(input: CandidateScoreVectorInput<'_>) -> Result<ScoreVectorRecord> {
    let requested_example_ids = input
        .rows
        .iter()
        .map(row_example_id)
        .collect::<Result<Vec<_>>>()?;
    let mut row_example_ids = BTreeSet::new();
    for example_id in &requested_example_ids {
        if !row_example_ids.insert(example_id.clone()) {
            return Err(OptimizerError::Invariant(format!(
                "score vector for candidate={} split={} evaluation_stage={} requested duplicate row {}",
                input.candidate.candidate_id, input.split, input.evaluation_stage, example_id
            )));
        }
    }
    let declared_objectives = input
        .objective_set
        .objectives
        .iter()
        .map(|objective| objective.name.clone())
        .collect::<BTreeSet<_>>();
    let mut scores = Vec::new();
    let mut duplicate_frame_count = 0usize;
    for example_id in &requested_example_ids {
        let mut row_score_sets = BTreeMap::<String, Vec<ScoreRecord>>::new();
        let mut matching_frame_count = 0usize;
        for frame in &input.candidate.sensor_frames {
            if !score_vector_frame_matches_split(frame, input.split, input.source_stages) {
                continue;
            }
            if !input
                .source_stages
                .iter()
                .any(|stage| *stage == frame.evaluation_stage)
            {
                continue;
            }
            if frame.example_id != *example_id {
                continue;
            }
            matching_frame_count += 1;
            let row_scores =
                score_records_for_vector_frame(input.objective_set, &declared_objectives, frame);
            if row_scores.is_empty() {
                continue;
            }
            row_score_sets
                .entry(score_record_material_key(&row_scores))
                .or_insert(row_scores);
        }
        if matching_frame_count > 1 {
            duplicate_frame_count += matching_frame_count - 1;
        }
        if row_score_sets.is_empty() {
            return Err(OptimizerError::Invariant(format!(
                "missing score vector material for candidate={} split={} evaluation_stage={} source_stages={:?}; no declared objective scores matched requested row {}",
                input.candidate.candidate_id,
                input.split,
                input.evaluation_stage,
                input.source_stages,
                example_id
            )));
        }
        if row_score_sets.len() > 1 {
            return Err(OptimizerError::Invariant(format!(
                "conflicting score vector material for candidate={} split={} evaluation_stage={} source_stages={:?}; requested row {} matched {} distinct score payloads",
                input.candidate.candidate_id,
                input.split,
                input.evaluation_stage,
                input.source_stages,
                example_id,
                row_score_sets.len()
            )));
        }
        if let Some((_, row_scores)) = row_score_sets.into_iter().next() {
            scores.extend(row_scores);
        }
    }
    if scores.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "missing score vector material for candidate={} split={} evaluation_stage={} source_stages={:?}; no declared objective scores matched {} requested rows",
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
    metadata.insert(
        "requested_example_ids".to_string(),
        json!(requested_example_ids),
    );
    metadata.insert(
        "deduped_duplicate_frame_count".to_string(),
        json!(duplicate_frame_count),
    );
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

fn score_records_for_vector_frame(
    objective_set: &ObjectiveSetRecord,
    declared_objectives: &BTreeSet<String>,
    frame: &SensorFrame,
) -> Vec<ScoreRecord> {
    frame
        .objective_scores
        .iter()
        .filter(|score| declared_objectives.contains(&score.objective))
        .filter_map(|score| {
            objective_set
                .objectives
                .iter()
                .find(|objective| objective.name == score.objective)
                .map(|objective| ScoreRecord::from_sensor_frame(frame, objective, score))
        })
        .collect::<Vec<_>>()
}

fn score_record_material_key(scores: &[ScoreRecord]) -> String {
    let mut parts = scores
        .iter()
        .map(|score| {
            format!(
                "{}\u{0}{}\u{0}{:.17}",
                score.objective, score.source, score.value
            )
        })
        .collect::<Vec<_>>();
    parts.sort();
    parts.join("\u{1}")
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
    task_identity(row)
}

fn row_task_id(row: &Value) -> String {
    task_identity(row).unwrap_or_else(|_| "unknown".to_string())
}

fn effective_gepa_task_pool_ids(config: &SynthOptimizerConfig) -> BTreeMap<String, Vec<String>> {
    BTreeMap::from([
        ("pareto".to_string(), config.gepa.task_pools.pareto.clone()),
        (
            "minibatch".to_string(),
            config.gepa.task_pools.minibatch.clone(),
        ),
        (
            "reflection".to_string(),
            config.gepa.task_pools.reflection.clone(),
        ),
        (
            "heldout".to_string(),
            config.gepa.task_pools.heldout.clone(),
        ),
    ])
}

fn task_pool_rows_value(
    pareto_rows: &[Value],
    minibatch_rows: &[Value],
    reflection_rows: &[Value],
    heldout_rows: &[Value],
) -> Value {
    json!({
        "schema_version": "gepa_task_pools.v1",
        "pareto": {
            "row_count": pareto_rows.len(),
            "task_ids": pareto_rows.iter().map(row_task_id).collect::<Vec<_>>(),
            "rows": pareto_rows,
        },
        "minibatch": {
            "row_count": minibatch_rows.len(),
            "task_ids": minibatch_rows.iter().map(row_task_id).collect::<Vec<_>>(),
            "rows": minibatch_rows,
        },
        "reflection": {
            "row_count": reflection_rows.len(),
            "task_ids": reflection_rows.iter().map(row_task_id).collect::<Vec<_>>(),
            "rows": reflection_rows,
        },
        "heldout": {
            "row_count": heldout_rows.len(),
            "task_ids": heldout_rows.iter().map(row_task_id).collect::<Vec<_>>(),
            "rows": heldout_rows,
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
        .filter(|candidate| candidate_train_selectable(candidate))
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
        .filter(|idx| candidates.get(*idx).is_some_and(candidate_train_selectable))
        .or_else(|| {
            candidates
                .iter()
                .enumerate()
                .rev()
                .find(|(_, candidate)| candidate_train_selectable(candidate))
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
        .filter_map(|(idx, candidate)| candidate_train_selectable(candidate).then_some(idx))
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
        if !candidate_train_selectable(candidate) {
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
        if !candidate_train_selectable(candidate) {
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
        if !candidate_train_selectable(candidate) {
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
    let train_task_ids = train_rows.iter().map(row_task_id).collect::<BTreeSet<_>>();
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
    let best_task_count = best_candidate
        .map(|candidate| {
            candidate
                .train_scores
                .iter()
                .filter(|score| score_is_solved(score.reward))
                .map(|score| score.task_id.clone())
                .collect::<BTreeSet<_>>()
                .len()
        })
        .unwrap_or(0);

    let mut covered_frontier_task_ids = BTreeSet::new();
    let mut covered_frontier_examples = BTreeSet::new();
    let mut member_rows = Vec::new();
    for member in &frontier {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.candidate_id == member.candidate_id)
        else {
            continue;
        };
        let evaluated_task_id_set = candidate
            .train_scores
            .iter()
            .map(|score| score.task_id.clone())
            .collect::<BTreeSet<_>>();
        let solved_task_id_set = candidate_solved_task_ids(&candidate.train_scores);
        let example_scores = scores_by_example(&candidate.train_scores);
        let solved_example_scores = solved_examples(&candidate.train_scores);
        let covered_task_ids = train_task_ids
            .iter()
            .filter(|task_id| solved_task_id_set.contains(*task_id))
            .cloned()
            .collect::<Vec<_>>();
        let missing_task_ids = train_task_ids
            .iter()
            .filter(|task_id| !solved_task_id_set.contains(*task_id))
            .cloned()
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
        covered_frontier_task_ids.extend(covered_task_ids.iter().cloned());
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
            "covered_task_id_count": covered_task_ids.len(),
            "missing_task_id_count": missing_task_ids.len(),
            "covered_task_ids": covered_task_ids,
            "missing_task_ids": missing_task_ids,
            "covered_example_count": covered_examples.len(),
            "missing_example_count": missing_examples.len(),
            "covered_examples": covered_examples,
            "missing_examples": missing_examples,
            "evaluated_task_id_count": evaluated_task_id_set.len(),
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
        "train_task_id_count": train_task_ids.len(),
        "train_task_ids": train_task_ids.iter().cloned().collect::<Vec<_>>(),
        "covered_train_task_id_count": covered_frontier_task_ids.len(),
        "covered_train_task_id_percent": coverage_percent(covered_frontier_task_ids.len(), train_task_ids.len()),
        "covered_train_task_ids": covered_frontier_task_ids.iter().cloned().collect::<Vec<_>>(),
        "covered_train_example_count": covered_frontier_examples.len(),
        "covered_train_example_percent": coverage_percent(covered_frontier_examples.len(), train_example_ids.len()),
        "best_candidate_task_id_coverage_percent": coverage_percent(best_task_count, train_task_ids.len()),
        "best_candidate_example_count": best_solved_examples.len(),
        "best_candidate_example_coverage_percent": coverage_percent(best_solved_examples.len(), train_example_ids.len()),
        "coverage_semantics": "solved_reward_positive",
        "frontier": frontier,
        "members": member_rows,
        "coverage": {
            "train_row_count": train_rows.len(),
            "train_task_id_count": train_task_ids.len(),
            "train_example_count": train_example_ids.len(),
            "covered_train_task_id_count": covered_frontier_task_ids.len(),
            "covered_train_task_id_percent": coverage_percent(covered_frontier_task_ids.len(), train_task_ids.len()),
            "best_candidate_task_id_count": best_task_count,
            "best_candidate_task_id_coverage_percent": coverage_percent(best_task_count, train_task_ids.len()),
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

fn candidate_solved_task_ids(scores: &[RolloutScore]) -> BTreeSet<String> {
    scores
        .iter()
        .filter(|score| score_is_solved(score.reward))
        .map(|score| score.task_id.clone())
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
        if !candidate_train_selectable(candidate) {
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

fn remaining_rollout_capacity_for_stage(
    workspace: &WorkspaceStore,
    config: &SynthOptimizerConfig,
    candidates: &[CandidateRecord],
    rollout_count: usize,
    stage: &str,
) -> Result<usize> {
    if stage == "heldout" {
        remaining_heldout_rollout_capacity(workspace, config, candidates)
    } else {
        remaining_train_rollout_capacity(workspace, config, rollout_count)
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
        candidates: serde_json::to_value(checkpoint_candidate_records(state.candidates))?,
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
    workspace.record_checkpoint_compacting_previous(&config.run.run_id, &checkpoint)
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
        "candidates": candidate_checkpoint_summaries(state.candidates),
        "frontier": frontier_checkpoint_summaries(&state.frontier),
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
        "provider".to_string(),
        Value::String(config.proposer.provider.clone()),
    );
    metadata.insert(
        "runtime_substrate".to_string(),
        Value::String(outcome.runtime_substrate.clone()),
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
        provider: Some(&config.proposer.provider),
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
    let mut value = serde_json::to_value(candidate)?;
    enrich_candidate_value_with_seed_summaries(&mut value, candidate);
    workspace.persist_candidate_registry(run_id, &[value])
}

pub(crate) fn record_initial_platform_snapshots(
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

struct TasksetSnapshotCall<'a> {
    run_id: &'a str,
    taskset_id: &'a str,
    split: &'a str,
    task_ids: &'a [String],
    filters: &'a Value,
    response: &'a TasksetTasksResponse,
    taskset_metadata: &'a Value,
}

fn record_taskset_snapshot(
    workspace: &WorkspaceStore,
    call: TasksetSnapshotCall<'_>,
) -> Result<()> {
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!("container.taskset_tasks"));
    workspace.record_taskset_snapshot(&TasksetSnapshotRecord::from_input(TasksetSnapshotInput {
        run_id: call.run_id,
        taskset_id: call.taskset_id,
        split: call.split,
        task_ids: call.task_ids,
        filters: call.filters,
        tasks: &call.response.tasks,
        taskset_metadata: call.taskset_metadata.clone(),
        tasks_metadata: Value::Object(call.response.metadata.clone()),
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

fn compact_completed_optimizer_job_payload(payload: &Map<String, Value>) -> Map<String, Value> {
    let mut compacted = Map::new();
    compacted.insert(
        "schema_version".to_string(),
        json!("synth_gepa.runtime_job_summary.v1"),
    );
    compacted.insert("storage_compacted".to_string(), json!(true));
    for key in [
        "schema_version",
        "dispatch_kind",
        "runtime_effect_id",
        "effect_kind",
        "lane",
        "subject_type",
        "subject_id",
        "idempotency_key",
        "cache_key",
        "budget_reservation_id",
        "queue_state",
        "generation",
        "parent_candidate_id",
        "candidate_id",
        "stage",
        "example_id",
        "task_id",
        "proposer_workspace_dir",
        "effect_payload",
        "runtime_outcome",
    ] {
        if let Some(value) = payload.get(key) {
            compacted.insert(
                if key == "schema_version" {
                    "original_schema_version".to_string()
                } else {
                    key.to_string()
                },
                value.clone(),
            );
        }
    }
    if let Some(request) = payload.get("request") {
        compacted.insert(
            "request_summary".to_string(),
            compact_runtime_request_summary(request),
        );
    }
    if let Some(rollouts) = payload.get("rollouts").and_then(Value::as_array) {
        compacted.insert("rollout_count".to_string(), json!(rollouts.len()));
        compacted.insert(
            "rollouts".to_string(),
            Value::Array(
                rollouts
                    .iter()
                    .map(compact_runtime_rollout_dispatch_summary)
                    .collect(),
            ),
        );
    }
    compacted
}

fn compact_runtime_request_summary(request: &Value) -> Value {
    let mut summary = Map::new();
    summary.insert(
        "schema".to_string(),
        json!("synth_gepa.runtime_request_summary.v1"),
    );
    summary.insert("storage_compacted".to_string(), json!(true));
    for key in [
        "backend",
        "execution_mode",
        "runtime_substrate",
        "model",
        "generation",
        "proposal_count",
        "target_modules",
        "parent_candidate_id",
        "workspace_root",
        "run_artifact_dir",
        "proposal_artifact_dir",
    ] {
        if let Some(value) = request.get(key) {
            summary.insert(key.to_string(), value.clone());
        }
    }
    if let Some(parent) = request.get("parent") {
        summary.insert(
            "parent".to_string(),
            compact_candidate_value_summary(parent),
        );
    }
    if let Some(candidates) = request.get("candidates").and_then(Value::as_array) {
        summary.insert("candidate_count".to_string(), json!(candidates.len()));
        summary.insert(
            "candidates".to_string(),
            Value::Array(
                candidates
                    .iter()
                    .map(compact_candidate_value_summary)
                    .collect(),
            ),
        );
    }
    if let Some(task_pool_rows) = request.get("task_pool_rows") {
        summary.insert(
            "task_pool_rows".to_string(),
            compact_task_pool_rows(task_pool_rows),
        );
    }
    if let Some(value) = request.get("frontier_summary") {
        summary.insert("frontier_summary".to_string(), value.clone());
    }
    if let Some(value) = request.get("minibatch_failures") {
        summary.insert("minibatch_failures".to_string(), value.clone());
    }
    if let Some(value) = request.get("rollout_trace_artifact_refs") {
        summary.insert(
            "rollout_trace_artifact_ref_count".to_string(),
            json!(value.as_array().map(Vec::len).unwrap_or(0)),
        );
    }
    if let Some(value) = request.get("merge_evidence_artifacts") {
        summary.insert("merge_evidence_artifacts".to_string(), value.clone());
    }
    Value::Object(summary)
}

fn compact_runtime_rollout_dispatch_summary(rollout: &Value) -> Value {
    let mut summary = Map::new();
    for key in ["candidate_id", "stage", "example_id", "task_id"] {
        if let Some(value) = rollout.get(key) {
            summary.insert(key.to_string(), value.clone());
        }
    }
    if let Some(request) = rollout.get("request") {
        summary.insert(
            "request_summary".to_string(),
            compact_rollout_request_summary(request),
        );
    }
    Value::Object(summary)
}

fn compact_rollout_request_summary(request: &Value) -> Value {
    let mut summary = Map::new();
    for key in ["task_id", "candidate_id", "policy"] {
        if let Some(value) = request.get(key) {
            summary.insert(key.to_string(), value.clone());
        }
    }
    if let Some(candidate) = request.get("candidate") {
        summary.insert("candidate".to_string(), candidate.clone());
    }
    if let Some(task) = request.get("task") {
        summary.insert(
            "task".to_string(),
            project_json_fields(
                task,
                &["task_id", "task_instance_id", "split", "seed", "objective"],
            ),
        );
    }
    Value::Object(summary)
}

fn compact_candidate_value_summary(candidate: &Value) -> Value {
    let mut summary = Map::new();
    for key in [
        "candidate_id",
        "parent_id",
        "source",
        "status",
        "minibatch_reward",
        "train_reward",
        "heldout_reward",
        "acceptance_score",
    ] {
        if let Some(value) = candidate.get(key) {
            summary.insert(key.to_string(), value.clone());
        }
    }
    if let Some(payload) = candidate.get("payload") {
        summary.insert("payload".to_string(), payload.clone());
    }
    if let Some(scores) = candidate.get("minibatch_scores").and_then(Value::as_array) {
        summary.insert("minibatch_score_count".to_string(), json!(scores.len()));
    }
    if let Some(scores) = candidate.get("train_scores").and_then(Value::as_array) {
        summary.insert("train_score_count".to_string(), json!(scores.len()));
    }
    if let Some(frames) = candidate.get("sensor_frames").and_then(Value::as_array) {
        summary.insert("sensor_frame_count".to_string(), json!(frames.len()));
        summary.insert(
            "sensor_frames".to_string(),
            Value::Array(
                frames
                    .iter()
                    .map(compact_sensor_frame_value_summary)
                    .collect(),
            ),
        );
    }
    Value::Object(summary)
}

fn compact_sensor_frame_value_summary(frame: &Value) -> Value {
    let mut summary = Map::new();
    for key in [
        "sensor_frame_id",
        "candidate_id",
        "rollout_id",
        "example_id",
        "task_id",
        "split",
        "evaluation_stage",
        "reward",
        "status",
        "success_status",
        "objective_scores",
        "usage",
        "trace_digest",
        "failure",
        "artifact_refs",
    ] {
        if let Some(value) = frame.get(key) {
            summary.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(summary)
}

fn compact_task_pool_rows(task_pool_rows: &Value) -> Value {
    let mut summary = Map::new();
    if let Some(object) = task_pool_rows.as_object() {
        if let Some(schema_version) = object.get("schema_version") {
            summary.insert("schema_version".to_string(), schema_version.clone());
        }
        for (name, pool) in object {
            if name == "schema_version" {
                continue;
            }
            let row_count = pool
                .get("row_count")
                .and_then(Value::as_u64)
                .or_else(|| {
                    pool.get("rows")
                        .and_then(Value::as_array)
                        .map(|rows| rows.len() as u64)
                })
                .unwrap_or(0);
            let mut pool_summary = Map::new();
            pool_summary.insert("row_count".to_string(), json!(row_count));
            if let Some(task_ids) = pool.get("task_ids") {
                pool_summary.insert("task_ids".to_string(), task_ids.clone());
            }
            summary.insert(name.clone(), Value::Object(pool_summary));
        }
    }
    Value::Object(summary)
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
            job.attempt = existing.attempt;
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
    if job.status == OptimizerJobStatus::Completed {
        job.payload = compact_completed_optimizer_job_payload(&job.payload);
    }
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
    record_run_phase_timing_from_effect(
        workspace,
        input.planned,
        &completed,
        input.cost_usd,
        input.usage,
        input.rollout_count,
        metadata.clone(),
    )?;
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

fn record_run_phase_timing_from_effect(
    workspace: &WorkspaceStore,
    planned: &RuntimeEffectRecord,
    completed: &RuntimeEffectRecord,
    cost_usd: f64,
    usage: &UsageTotals,
    rollout_count: u64,
    metadata: Map<String, Value>,
) -> Result<()> {
    if !matches!(planned.lane.as_str(), "proposer" | "rollout") {
        return Ok(());
    }
    let stage = phase_timing_stage(planned);
    let generation = phase_timing_generation(planned);
    let candidate_id = phase_timing_candidate_id(planned);
    let item_count = phase_timing_item_count(planned, &metadata, rollout_count);
    let wall_seconds = phase_timing_wall_seconds(planned, completed)
        .or_else(|| metadata.get("wall_seconds").and_then(Value::as_f64))
        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
    let timing = synth_optimizer_platform::RunPhaseTimingRecord::from_input(RunPhaseTimingInput {
        run_id: &planned.run_id,
        lane: &planned.lane,
        kind: &planned.effect_kind,
        stage,
        generation,
        candidate_id,
        subject_type: &planned.subject_type,
        subject_id: &planned.subject_id,
        status: &completed.status,
        started_at: &planned.planned_at,
        finished_at: completed.terminal_at.clone(),
        wall_seconds,
        item_count,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        cost_usd: Some(cost_usd),
        source_effect_id: &planned.runtime_effect_id,
        metadata,
    });
    workspace.record_run_phase_timing(&timing)
}

fn phase_timing_stage(effect: &RuntimeEffectRecord) -> Option<String> {
    effect
        .payload
        .get("stages")
        .and_then(Value::as_array)
        .and_then(|stages| {
            if stages.len() == 1 {
                stages.first().and_then(Value::as_str).map(str::to_string)
            } else {
                None
            }
        })
        .or_else(|| {
            effect
                .metadata
                .get("stages")
                .and_then(Value::as_object)
                .and_then(|stages| {
                    if stages.len() == 1 {
                        stages.keys().next().cloned()
                    } else {
                        None
                    }
                })
        })
}

fn phase_timing_generation(effect: &RuntimeEffectRecord) -> Option<u64> {
    effect
        .payload
        .get("generation")
        .and_then(Value::as_u64)
        .or_else(|| {
            effect
                .subject_id
                .strip_prefix("generation_")
                .and_then(|value| value.parse::<u64>().ok())
        })
}

fn phase_timing_candidate_id(effect: &RuntimeEffectRecord) -> Option<String> {
    effect
        .payload
        .get("candidate_ids")
        .and_then(Value::as_array)
        .and_then(|candidate_ids| {
            if candidate_ids.len() == 1 {
                candidate_ids
                    .first()
                    .and_then(Value::as_str)
                    .map(str::to_string)
            } else {
                None
            }
        })
        .or_else(|| {
            effect
                .payload
                .get("parent_candidate_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn phase_timing_item_count(
    effect: &RuntimeEffectRecord,
    metadata: &Map<String, Value>,
    rollout_count: u64,
) -> Option<u64> {
    if effect.lane == "rollout" {
        return effect
            .payload
            .get("rollout_count")
            .and_then(Value::as_u64)
            .or_else(|| metadata.get("rollout_count").and_then(Value::as_u64))
            .or_else(|| (rollout_count > 0).then_some(rollout_count));
    }
    if effect.lane == "proposer" {
        return metadata
            .get("proposal_count")
            .and_then(Value::as_u64)
            .or_else(|| effect.payload.get("proposal_count").and_then(Value::as_u64));
    }
    None
}

fn phase_timing_wall_seconds(
    planned: &RuntimeEffectRecord,
    completed: &RuntimeEffectRecord,
) -> Option<f64> {
    let finished_at = completed.terminal_at.as_deref()?;
    let start = time::OffsetDateTime::parse(
        &planned.planned_at,
        &time::format_description::well_known::Rfc3339,
    )
    .ok()?;
    let end =
        time::OffsetDateTime::parse(finished_at, &time::format_description::well_known::Rfc3339)
            .ok()?;
    let seconds = (end - start).as_seconds_f64();
    (seconds.is_finite() && seconds >= 0.0).then_some(seconds)
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
    if cancellation
        .in_process
        .as_ref()
        .is_some_and(|token| token.load(Ordering::SeqCst))
    {
        return Err(OptimizerError::Cancelled {
            request_id: cancellation.request_id.clone(),
        });
    }
    if cancellation.service_db_path.as_os_str().is_empty() {
        return Ok(());
    }
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
    let mut section_scored = 0usize;
    let mut section_degraded = 0usize;
    for row in call.rows {
        check_cancelled(call.cancellation)?;
        let task_id = row_task_id(row);
        let overlay = CandidateOverlay {
            candidate: PromptCandidatePayload::from_map(call.candidate.payload.clone()),
            metadata: Map::new(),
        };
        let prompt_assertions = prompt_assertions_for_candidate(&overlay.candidate, call.config);
        let request = json!({
            "submission_mode": rollout_submission_mode_for_request(call.config),
            "task_id": call.task_id,
            "task_id": task_id,
            "candidate_id": call.candidate.candidate_id,
            "candidate": overlay.candidate.to_value(),
            "candidate_overlay": overlay,
            "prompt_assertions": prompt_assertions,
            "policy": rollout_policy_for_request(call.config),
            "task": row,
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
        let rollout_execution = execute_sync_rollout_job(
            call.workspace,
            call.cache,
            call.config,
            call.client,
            &queued_effect,
            RolloutExecutionIdentity {
                candidate_id: &call.candidate.candidate_id,
                stage: call.stage,
                example_id: &example_id,
            },
        )?;
        if rollout_execution.degraded {
            section_degraded = section_degraded.saturating_add(1);
        } else {
            section_scored = section_scored.saturating_add(1);
        }
        record_rollout_resilience_sample(
            call.config,
            Some(&mut *call.events),
            call.rollout_resilience,
            RolloutResilienceObservation {
                stage: call.stage,
                example_id: &example_id,
                degraded: rollout_execution.degraded,
                failure: rollout_execution.failure.as_ref(),
                provider_signal: &rollout_execution.provider_signal,
            },
        )?;
        let rollout_call = rollout_execution.outcome;
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
                task_id: &task_id,
                split: &sensor_frame.split,
                evaluation_stage: call.stage,
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
            task_id,
            reward,
        });
        sensor_frames.push(sensor_frame);
        usage.merge(&rollout_call.usage);
        cost_usd += rollout_call.cost_usd;
    }
    let rollout_count = call.rows.len();
    check_rollout_section_breaker(
        call.config,
        Some(&mut *call.events),
        call.rollout_resilience,
        call.stage,
        section_scored,
        section_degraded,
    )?;
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

struct RolloutExecutionIdentity<'a> {
    candidate_id: &'a str,
    stage: &'a str,
    example_id: &'a str,
}

fn execute_sync_rollout_job(
    workspace: &WorkspaceStore,
    cache: &mut RequestCache,
    config: &SynthOptimizerConfig,
    client: &ContainerClient,
    queued_effect: &runtime::QueuedRuntimeEffect,
    identity: RolloutExecutionIdentity<'_>,
) -> Result<RolloutExecutionRecord> {
    loop {
        match runtime::execute_one_pending_optimizer_job_from_run_workspace(
            workspace,
            cache,
            config,
            client,
            &queued_effect.effect.run_id,
            &queued_effect.job.job_id,
            runtime::RuntimeEffectExecutorConfig::inline_default(),
        ) {
            Ok(runtime::RuntimeEffectOutcome::Rollout(outcome)) => {
                return Ok(RolloutExecutionRecord {
                    outcome: *outcome,
                    degraded: false,
                    failure: None,
                    provider_signal: ProviderSignal::default(),
                });
            }
            Ok(runtime::RuntimeEffectOutcome::Proposer(_)) => {
                return Err(OptimizerError::Invariant(format!(
                    "rollout runtime effect returned proposer outcome job_id={}",
                    queued_effect.job.job_id
                )));
            }
            Ok(runtime::RuntimeEffectOutcome::RolloutBatch(_)) => {
                return Err(OptimizerError::Invariant(format!(
                    "single rollout runtime effect returned batch outcome job_id={}",
                    queued_effect.job.job_id
                )));
            }
            Err(error) => {
                let provider_signal = provider_signal_from_error(config, &error);
                let updated_job = workspace
                    .optimizer_job(&queued_effect.effect.run_id, &queued_effect.job.job_id)?;
                if matches!(updated_job.status, OptimizerJobStatus::RetryScheduled) {
                    sleep_for_retry_scheduled_job(&updated_job);
                    continue;
                }
                if updated_job.status.is_terminal() {
                    let failure = updated_job
                        .failure
                        .clone()
                        .unwrap_or_else(|| FailurePayload::from_optimizer_error(&error));
                    if failure.retryable
                        && updated_job.attempt < updated_job.retry_policy.max_attempts
                    {
                        if let Some(retry_job) = workspace.schedule_terminal_optimizer_job_retry(
                            &updated_job.run_id,
                            &updated_job.job_id,
                            rollout_retry_backoff_seconds(&updated_job),
                            &failure,
                        )? {
                            sleep_for_retry_scheduled_job(&retry_job);
                            continue;
                        }
                    }
                    if !failure.retryable && !rollout_error_is_degradable(&error) {
                        return Err(error);
                    }
                    let provider_signal = if provider_signal.status_code.is_some()
                        || provider_signal.overload
                        || provider_signal.retryable
                    {
                        provider_signal
                    } else {
                        provider_signal_from_failure(config, Some(&failure))
                    };
                    let outcome = degraded_runtime_rollout_outcome(
                        queued_effect,
                        identity.candidate_id,
                        identity.stage,
                        identity.example_id,
                        &failure,
                        &provider_signal,
                    )?;
                    return Ok(RolloutExecutionRecord {
                        outcome,
                        degraded: true,
                        failure: Some(failure),
                        provider_signal,
                    });
                }
                return Err(error);
            }
        }
    }
}

fn sleep_for_retry_scheduled_job(job: &OptimizerJob) {
    thread::sleep(Duration::from_secs(rollout_retry_backoff_seconds(job)));
}

fn rollout_retry_backoff_seconds(job: &OptimizerJob) -> u64 {
    let exponent = job.attempt.saturating_sub(1).min(8);
    job.retry_policy
        .backoff_seconds
        .saturating_mul(1_u64 << exponent)
        .max(1)
}

fn degraded_runtime_rollout_outcome(
    queued_effect: &runtime::QueuedRuntimeEffect,
    candidate_id: &str,
    stage: &str,
    example_id: &str,
    failure: &FailurePayload,
    provider_signal: &ProviderSignal,
) -> Result<runtime::RuntimeRolloutOutcome> {
    let cache_key = queued_effect
        .effect
        .cache_key
        .clone()
        .or_else(|| {
            queued_effect
                .job
                .payload
                .get("cache_key")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| queued_effect.effect.idempotency_key.clone());
    degraded_runtime_rollout_outcome_for_cache_key(
        candidate_id,
        stage,
        example_id,
        &cache_key,
        failure,
        provider_signal,
    )
}

fn degraded_runtime_rollout_outcome_for_cache_key(
    candidate_id: &str,
    stage: &str,
    example_id: &str,
    cache_key: &str,
    failure: &FailurePayload,
    provider_signal: &ProviderSignal,
) -> Result<runtime::RuntimeRolloutOutcome> {
    let response =
        degraded_rollout_response(candidate_id, stage, example_id, failure, provider_signal)?;
    let typed_response = synth_optimizer_platform::RolloutResponse::from_value(response.clone())?;
    typed_response.validate_for_gepa()?;
    let usage = UsageTotals {
        rollout_calls: 1,
        ..UsageTotals::default()
    };
    Ok(runtime::RuntimeRolloutOutcome {
        candidate_id: candidate_id.to_string(),
        response,
        typed_response,
        reward: 0.0,
        usage,
        cost_usd: 0.0,
        cache_key: cache_key.to_string(),
        cache_hit: false,
        stage: stage.to_string(),
        example_id: example_id.to_string(),
        dispatch_wall_seconds: None,
        dispatch_chunk_index: None,
        dispatch_chunk_size: None,
    })
}

fn degraded_rollout_response(
    candidate_id: &str,
    stage: &str,
    example_id: &str,
    failure: &FailurePayload,
    provider_signal: &ProviderSignal,
) -> Result<Value> {
    let mut digest = Sha256::new();
    digest.update(candidate_id.as_bytes());
    digest.update(stage.as_bytes());
    digest.update(example_id.as_bytes());
    digest.update(failure.message.as_bytes());
    let rollout_id = format!("degraded:{:x}", digest.finalize());
    Ok(json!({
        "rollout_id": rollout_id,
        "status": "failed",
        "success_status": "infra_degraded",
        "summary": {
            "outcome_reward": 0.0,
            "degraded": true,
            "failure_class": failure.failure_class(),
        },
        "reward_info": {
            "outcome_reward": 0.0,
            "score": 0.0,
            "details": {
                "objective": "outcome_reward",
                "degraded": true,
            },
        },
        "usage": {},
        "metadata": {
            "degraded": true,
            "failure": serde_json::to_value(failure)?,
            "provider_signal": serde_json::to_value(provider_signal)?,
        },
        "actionable_side_info": {
            "degraded": true,
            "failure_class": failure.failure_class(),
            "message": &failure.message,
        },
    }))
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
        "runtime_substrate": call.config.proposer.runtime_substrate.as_str(),
        "model": call.config.proposer.model,
        "generation": call.generation,
        "parent": call.parent,
        "candidates": call.candidates,
        "program": call.program,
        "task_pool_rows": call.task_pool_rows,
        "workspace_root": call.paths.run_dir,
        "run_artifact_dir": call.paths.run_dir,
        "proposal_artifact_dir": workspace_dir,
        "lever_manifest": LeverManifest::from_prompt_program(call.program),
        "frontier_summary": proposer_frontier_summary(
            call.candidates,
            &call.task_pool_rows
                .get("pareto")
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
    cache_metadata.insert(
        "runtime_substrate".to_string(),
        json!(call.config.proposer.runtime_substrate.as_str()),
    );
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
                "runtime_substrate": call.config.proposer.runtime_substrate.as_str(),
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
        runtime_substrate: proposer_runtime_outcome.runtime_substrate,
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
    task_pool_rows: Value,
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
                task_pool_rows,
                workspace_dir,
            })
            .map_err(|error| {
                eprintln!("[gepa-proposer] codex_app_server proposer failed: {error}");
                error
            })
        }
        // Direct OpenAI-compatible /chat/completions proposer. "deepseek_chat" is the
        // back-compat name; "chat_completions" is the provider-agnostic one (deepseek | nvidia).
        "deepseek_chat" | "chat_completions" => {
            codex_app_server::run_deepseek_chat_proposer(codex_app_server::CodexProposerInput {
                config,
                program,
                parent,
                candidates,
                generation,
                task_pool_rows,
                workspace_dir,
            })
            .map_err(|error| {
                eprintln!(
                    "[gepa-proposer] chat_completions proposer ({}) failed: {error}",
                    config.proposer.provider
                );
                error
            })
        }
        "local_process_json" => Err(OptimizerError::Config(
            "unsupported proposer.backend \"local_process_json\"; GEPA proposer work must use codex_app_server workspace-backed proposing".to_string(),
        )),
        backend => Err(OptimizerError::Config(format!(
            "unsupported proposer.backend {backend:?}; expected codex_app_server, chat_completions, or deepseek_chat"
        ))),
    }
}

/// Invoke the same workspace proposer used by public GEPA and return decoded
/// proposals. This deliberately does not evaluate or select candidates: it is
/// the narrow extension boundary used by experimental optimizer dynamics while
/// keeping proposer model, prompt, auth, and evidence hygiene identical.
pub fn propose_workspace_candidates(
    config: &SynthOptimizerConfig,
    program: &PromptProgram,
    parent: &CandidateRecord,
    candidates: &[CandidateRecord],
    generation: usize,
    task_pool_rows: Value,
    workspace_dir: PathBuf,
) -> Result<WorkspaceProposerOutcome> {
    let workspace = workspace_dir.display().to_string();
    let mut response = run_proposer(
        config,
        program,
        parent,
        candidates,
        generation,
        task_pool_rows,
        workspace_dir,
    )?;
    if let Some(object) = response.as_object_mut() {
        object.insert("workspace".to_string(), json!(&workspace));
    }

    let default_evidence = response
        .get("manifest")
        .and_then(|manifest| manifest.get("evidence"))
        .cloned()
        .unwrap_or(Value::Null);
    let proposal_values = response
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut proposals = Vec::with_capacity(proposal_values.len());
    for (proposal_index, value) in proposal_values.into_iter().enumerate() {
        let mut proposal = serde_json::from_value::<ProposedCandidate>(value).map_err(|source| {
            OptimizerError::Proposer(format!(
                "workspace proposer proposal index={proposal_index} is invalid: {source}"
            ))
        })?;
        if proposal.evidence.is_null() {
            proposal.evidence = default_evidence.clone();
        }
        if proposal.payload_map().is_empty() {
            return Err(OptimizerError::Proposer(format!(
                "workspace proposer proposal index={proposal_index} returned no mutable payload; shape={}",
                proposal.payload_shape_summary()
            )));
        }
        proposals.push(proposal);
    }
    if proposals.is_empty() {
        return Err(OptimizerError::Proposer(
            "workspace proposer returned no proposals".to_string(),
        ));
    }

    let backend = response
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or(config.proposer.backend.as_str())
        .to_string();
    let runtime_substrate = response
        .get("runtime_substrate")
        .and_then(Value::as_str)
        .unwrap_or(config.proposer.runtime_substrate.as_str())
        .to_string();
    let evidence_warnings = response
        .get("evidence_warnings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();

    Ok(WorkspaceProposerOutcome {
        proposals,
        response,
        backend,
        runtime_substrate,
        workspace,
        evidence_warnings,
    })
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
    transitions: Option<&TransitionSink>,
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
    if let Some(transitions) = transitions {
        let mut metadata = transition.details.clone();
        metadata.insert("message".to_string(), json!(message));
        metadata.insert("at".to_string(), json!(&transition.at));
        transitions.record(TransitionInput {
            ts_unix_ms: None,
            entity_type: "run",
            entity_id: &transition.run_id,
            from_state: Some(transition.from.as_str()),
            to_state: transition.to.as_str(),
            trigger: transition.trigger.as_str(),
            generation: metadata
                .get("generation")
                .and_then(Value::as_i64)
                .or_else(|| {
                    metadata
                        .get("generation")
                        .and_then(Value::as_u64)
                        .map(|n| n as i64)
                }),
            parent_id: None,
            metadata: Value::Object(metadata),
        })?;
    }
    workspace.record_state_transition(state_machine.history.len(), &transition)
}

struct FailedGepaRunInput<'a> {
    workspace: &'a mut WorkspaceStore,
    events: &'a mut EventWriter,
    state_machine: &'a mut OptimizerStateMachine,
    transitions: &'a TransitionSink,
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
            Some(input.transitions),
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
        .record_checkpoint_compacting_previous(&input.config.run.run_id, &checkpoint)?;
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
    normalize_event_feed(
        &input.paths.event_feed_path,
        &input.paths.normalized_event_feed_path,
        &input.paths.run_dir,
    )?;
    let storage_summary =
        record_terminal_storage_snapshot(input.paths, &input.config.run.run_id, input.events)?;
    input.events.flush()?;
    input
        .workspace
        .record_event_stream(&input.config.run.run_id, input.events.records())?;
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
            Some(storage_summary.clone()),
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
            Some(storage_summary.clone()),
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
