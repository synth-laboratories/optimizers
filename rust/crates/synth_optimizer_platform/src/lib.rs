pub mod agent_runtime;
pub mod artifact_store;
pub mod artifacts;
pub mod cache;
pub mod candidates;
pub mod checkpoints;
pub mod config;
pub mod configured_limits;
pub mod container_contract;
pub mod data_models;
pub mod disk_budget;
pub mod error;
mod event_visualization;
pub mod events;
pub mod evidence;
pub mod failures;
pub mod fsm;
pub mod http;
pub mod invariants;
pub mod jobs;
pub mod levers;
pub mod limit_engine;
pub mod limits;
pub mod observability;
pub mod operations;
pub mod process;
pub mod projections;
pub mod prompt_program;
pub mod registry;
pub mod resources;
pub mod rollouts;
pub mod runtime_records;
pub mod scores;
pub mod sensors;
pub mod state_machine;
pub mod stopper;
pub mod storage_maintenance;
pub mod usage;
pub mod workspace;

pub use agent_runtime::{
    ensure_turn_completed, extract_thread_id, prepare_proposer_codex_launch,
    record_manifest_validation, run_command_exec, run_turn, sandbox_policy_for_mode,
    text_turn_input, usage_from_message, usage_from_messages, AgentCommandExecOutcome,
    AgentRuntimeSubstrate, AgentTurnOutcome, CodexAppServerClient, CodexAppServerLaunch,
    CodexAppServerProcessLaunch, CodexCommandExecRequest, CodexTurnRequest, ExecutionSubstrate,
    NanoAgentTurnIdentity, NanoCodexEvent, NanoCodexExecution, NanoCodexLatency,
    NanoCodexRecordedOutcome, NanoCodexSessionPool, NanoCodexTurnReceipt, NanoCodexTurnRequest,
    ProposerCodexLaunch, ResolvedRoleAgentConfig, RoleAgentConfig, RoleAgentTurnRequestInput,
    SupervisorReceipt,
};
pub use artifact_store::{LocalDevStore, RunArtifactStore, StoredRunArtifact};
pub use artifacts::{ArtifactPaths, ArtifactRef, GepaRunResult};
pub use cache::{
    normalize_for_cache_profile, stable_json_hash, CacheAccessRecord, CacheEntry, CacheMode,
    CacheProfile, CacheProfileRecord, RequestCache,
};
pub use candidates::{
    AcceptanceDecisionInput, AcceptanceDecisionRecord, CandidateDeltaInput, CandidateDeltaRecord,
    CandidatePayloadInput, CandidatePayloadRecord, FrontierCellInput, FrontierCellRecord,
    PlanLinkInput, PlanLinkRecord,
};
pub use checkpoints::{CheckpointInput, CheckpointRecord, CheckpointSummaryRecord};
pub use config::{
    proposer_auth_mode_normalized, proposer_uses_chatgpt_auth, resolve_chatgpt_codex_home_source,
    resolve_proposer_auth_launch_mode, validate_chatgpt_proposer_config,
    validate_chatgpt_proposer_model, CacheConfig, CandidateConfig, ContainerConfig,
    ContainerPoolTargetConfig, GepaAdaptiveRolloutConcurrencyConfig,
    GepaAdaptiveStageWorkersConfig, GepaBatchSamplerConfig, GepaCandidateSelectorConfig,
    GepaConfig, GepaObjectiveAcceptanceConfig, GepaPipelineConfig, GepaPipelineMode,
    GepaPipelineWorkers, GepaSpeculativeCompletionConfig, GepaStalenessPolicy, GepaTaskPoolsConfig,
    NanoCodexConfig, PolicyConfig, ProposerAuthLaunchMode, ProposerConfig, ProposerDaytonaConfig,
    ProposerDockerConfig, ProposerPromptConfig, RunConfig, SynthOptimizerConfig, TasksetConfig,
    CHATGPT_PROPOSER_MODELS,
};
pub use configured_limits::{
    ConfiguredGepaRunLimits, GepaRuntimeEffectBudgetEstimates, GEPA_LIMIT_STOP_POLICY,
};
pub use container_contract::{
    task_identity, CanonicalChoice, CanonicalMessage, CanonicalRequest, CanonicalResponse,
    CanonicalUsage, ContainerMetadata, ContainerMetadataResponse, GepaOptimizerContract,
    HealthResponse, OptimizerContracts, RewardInfo, RolloutActorSpec, RolloutRequest,
    RolloutResponse, RolloutTraceSpanV4, RolloutTraceV4, TasksetResponse, TasksetTasksRequest,
    TasksetTasksResponse, TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_NAME,
};
pub use data_models::{
    evaluation_cache_key_fields, materialization_record_json, objective_set_hash,
    EvaluationCacheIdentity, EvaluationCacheRecord, EvaluationCacheRecordInput,
    MaterializationRecord, MaterializationRecordInput, RolloutMaterializationIdentity,
    EVALUATION_CACHE_KEY_FIELDS_SCHEMA_VERSION, EVALUATION_CACHE_PROFILE,
    EVALUATION_CACHE_SCHEMA_VERSION, MATERIALIZATION_SCHEMA_VERSION,
};
pub use disk_budget::{directory_size_bytes, DiskBudget, DiskBudgetConfig, DiskBudgetState};
pub use error::{OptimizerError, Result};
pub use events::{
    compare_normalized_event_feeds, normalize_event_feed, optimizer_event_feed_path_for,
    replay_event_feed, EventStreamRecord, EventWriter,
};
pub use evidence::{
    EvidenceFrame, SensorDerivedRecords, SubagentInvocation, SubagentResult, TraceAnnotation,
    VerifierJob,
};
pub use failures::{FailurePayload, OptimizerFailureType};
pub use fsm::{
    EntityMachine, StateMachineEntity, TransitionInput, TransitionLog, TransitionRow,
    TransitionSink,
};
pub use http::ContainerClient;
pub use invariants::{
    CountMismatchInput, InvariantReport, InvariantViolation, InvariantViolationInput,
};
pub use jobs::{OptimizerJob, OptimizerJobKind, OptimizerJobStatus, RetryPolicy};
pub use levers::{LeverBundle, LeverKind, LeverManifest, LeverSpec};
pub use limit_engine::{
    budget_limit_engine_input, budget_limit_snapshot, ForecastConfidence, LimitDefinition,
    LimitEngine, LimitEngineInput, LimitForecast, LimitKind, LimitObservation, LimitProgressEvent,
    LimitSnapshot, LimitStatus, LIMIT_ENGINE_SCHEMA_VERSION,
};
pub use limits::{
    BudgetCommitInput, BudgetCommitRecord, BudgetLedgerSnapshot, BudgetLedgerTotals,
    BudgetLimitBreach, BudgetReleaseInput, BudgetReleaseRecord, BudgetReservationInput,
    BudgetReservationRecord, RunLimitPolicy, RunLimitsInput, RunLimitsRecord,
    RuntimeEffectAdmissionInput, RuntimeEffectAdmissionRecord, RuntimeEffectBudgetEstimate,
    BUDGET_COMMIT_SCHEMA_VERSION, BUDGET_RELEASE_SCHEMA_VERSION, BUDGET_RESERVATION_SCHEMA_VERSION,
    RUNTIME_EFFECT_ADMISSION_SCHEMA_VERSION, RUN_LIMITS_SCHEMA_VERSION,
};
pub use observability::{
    container_child_eval_ref, gepa_proposer_policy_ref, optimizer_event_log_id, policy_ref,
    proposer_delta_chunks_from_protocol, proposer_delta_chunks_from_response,
    proposer_delta_fields, OptimizerAlgorithm as ObservationOptimizerAlgorithm, OptimizerEvent,
    OptimizerItem, OptimizerItemType, OptimizerLogLevel, OptimizerStateSlice,
    OptimizerStateSliceKind, A3_TASK, BANKING77_EVAL_HARNESS, CHILD_ROLLOUT_ATTACHED_EVENT_TYPE,
    DEFAULT_PROPOSER_DELTA_CHANNEL, GEPA_PROPOSER_HARNESS, LUNA_MED_POLICY_CONFIG,
    OPTIMIZER_EVENT_SCHEMA_VERSION, OPTIMIZER_STATE_SLICE_SCHEMA_VERSION,
    PROPOSER_DELTA_EVENT_TYPE, SOL_MED_POLICY_CONFIG,
};
pub use operations::OperationRecord;
pub use process::ManagedContainerProcess;
pub use projections::ProjectionFreshnessRecord;
pub use prompt_program::{
    CandidateOverlay, PromptCandidatePayload, PromptModule, PromptProgram, TargetModule,
};
pub use registry::{RunRegistry, RunRegistryEntry};
pub use resources::{ResourceLeaseRecord, ResourceLeaseRecordInput};
pub use rollouts::{RolloutEventRecord, RolloutRecord, SensorRolloutRecords};
pub use runtime_records::{
    runtime_record_json, ContainerContractSnapshotInput, ContainerContractSnapshotRecord,
    PromptProgramSnapshotInput, PromptProgramSnapshotRecord, RenderedOptimizerStateInput,
    RenderedOptimizerStateRecord, ResolvedRunConfigInput, ResolvedRunConfigRecord,
    RunPhaseTimingInput, RunPhaseTimingRecord, RuntimeEffectInput, RuntimeEffectRecord,
    TasksetSnapshotInput, TasksetSnapshotRecord, CONTAINER_CONTRACT_SNAPSHOT_SCHEMA_VERSION,
    PROMPT_PROGRAM_SNAPSHOT_SCHEMA_VERSION, RENDERED_OPTIMIZER_STATE_SCHEMA_VERSION,
    RESOLVED_RUN_CONFIG_SCHEMA_VERSION, RUNTIME_EFFECT_SCHEMA_VERSION,
    RUN_PHASE_TIMING_SCHEMA_VERSION, TASKSET_SNAPSHOT_SCHEMA_VERSION,
};
pub use scores::{
    ObjectiveSetRecord, ObjectiveSpec, ParetoComparisonRecord, ScoreRecord, ScoreVectorRecord,
    SensorScoreRecords,
};
pub use sensors::{ObjectiveScore, SensorFrame, TraceDigest};
pub use state_machine::{
    OptimizerRunState, OptimizerStateMachine, OptimizerTransition, OptimizerTransitionTrigger,
};
pub use stopper::{StopperStateInput, StopperStateRecord};
pub use storage_maintenance::{
    compact_run_storage, delete_run_storage, inspect_run_storage, inspect_run_storage_summary,
    inspect_workspace_storage_health, write_run_storage_report, RunStorageInspectionInput,
    RunStorageMaintenanceInput, StorageHealthThresholds, StorageMaintenanceProfile,
    WorkspaceStorageHealthInput,
};
pub use usage::{fold_reported_cost, UsageLedgerInput, UsageLedgerRecord};
pub use workspace::{
    workspace_status, OptimizationRunStartedInput, WorkspaceEntityCounts,
    WorkspaceRunRequestStatus, WorkspaceRunStatus, WorkspaceStateTransitionStatus, WorkspaceStatus,
    WorkspaceStore, WorkspaceView,
};

pub const GEPA_OPTIMIZER_CONTRACT_VERSION: &str = "synth_optimizers.gepa.v2";
