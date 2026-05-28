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
pub mod http;
pub mod invariants;
pub mod jobs;
pub mod levers;
pub mod limits;
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
pub mod usage;
pub mod workspace;

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
pub use checkpoints::{CheckpointInput, CheckpointRecord};
pub use config::{
    CacheConfig, CandidateConfig, ContainerConfig, DatasetConfig,
    GepaAdaptiveRolloutConcurrencyConfig, GepaBatchSamplerConfig, GepaCandidateSelectorConfig,
    GepaConfig, GepaObjectiveAcceptanceConfig, GepaPipelineConfig, GepaPipelineMode,
    GepaPipelineWorkers, GepaSeedPoolsConfig, GepaStalenessPolicy, PolicyConfig, ProposerConfig,
    RunConfig, SynthOptimizerConfig,
};
pub use configured_limits::{
    ConfiguredGepaRunLimits, GepaRuntimeEffectBudgetEstimates, GEPA_LIMIT_STOP_POLICY,
};
pub use container_contract::{
    dataset_row_identity, CanonicalChoice, CanonicalMessage, CanonicalRequest, CanonicalResponse,
    CanonicalUsage, ContainerMetadata, ContainerMetadataResponse, DatasetResponse,
    DatasetRowsRequest, DatasetRowsResponse, GepaOptimizerContract, HealthResponse,
    OptimizerContracts, RewardInfo, RolloutActorSpec, RolloutRequest, RolloutResponse,
    RolloutTraceSpanV4, RolloutTraceV4, TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_NAME,
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
    compare_normalized_event_feeds, normalize_event_feed, replay_event_feed, EventStreamRecord,
    EventWriter,
};
pub use evidence::{
    EvidenceFrame, SensorDerivedRecords, SubagentInvocation, SubagentResult, TraceAnnotation,
    VerifierJob,
};
pub use failures::{FailurePayload, OptimizerFailureType};
pub use http::ContainerClient;
pub use invariants::{
    CountMismatchInput, InvariantReport, InvariantViolation, InvariantViolationInput,
};
pub use jobs::{OptimizerJob, OptimizerJobKind, OptimizerJobStatus, RetryPolicy};
pub use levers::{LeverBundle, LeverKind, LeverManifest, LeverSpec};
pub use limits::{
    BudgetCommitInput, BudgetCommitRecord, BudgetLedgerSnapshot, BudgetLedgerTotals,
    BudgetLimitBreach, BudgetReleaseInput, BudgetReleaseRecord, BudgetReservationInput,
    BudgetReservationRecord, RunLimitPolicy, RunLimitsInput, RunLimitsRecord,
    RuntimeEffectAdmissionInput, RuntimeEffectAdmissionRecord, RuntimeEffectBudgetEstimate,
    BUDGET_COMMIT_SCHEMA_VERSION, BUDGET_RELEASE_SCHEMA_VERSION, BUDGET_RESERVATION_SCHEMA_VERSION,
    RUNTIME_EFFECT_ADMISSION_SCHEMA_VERSION, RUN_LIMITS_SCHEMA_VERSION,
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
    DatasetSnapshotInput, DatasetSnapshotRecord, PromptProgramSnapshotInput,
    PromptProgramSnapshotRecord, RenderedOptimizerStateInput, RenderedOptimizerStateRecord,
    ResolvedRunConfigInput, ResolvedRunConfigRecord, RuntimeEffectInput, RuntimeEffectRecord,
    CONTAINER_CONTRACT_SNAPSHOT_SCHEMA_VERSION, DATASET_SNAPSHOT_SCHEMA_VERSION,
    PROMPT_PROGRAM_SNAPSHOT_SCHEMA_VERSION, RENDERED_OPTIMIZER_STATE_SCHEMA_VERSION,
    RESOLVED_RUN_CONFIG_SCHEMA_VERSION, RUNTIME_EFFECT_SCHEMA_VERSION,
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
pub use usage::{UsageLedgerInput, UsageLedgerRecord};
pub use workspace::{
    workspace_status, WorkspaceEntityCounts, WorkspaceRunRequestStatus, WorkspaceRunStatus,
    WorkspaceStateTransitionStatus, WorkspaceStatus, WorkspaceStore, WorkspaceView,
};

pub const GEPA_OPTIMIZER_CONTRACT_VERSION: &str = "synth_optimizers.gepa.v1";
